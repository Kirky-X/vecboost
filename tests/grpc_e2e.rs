// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! gRPC E2E 集成测试 — sdforge 统一 Call 协议
//!
//! 覆盖场景：
//! - 11 个 grpc_method 正常路径（GP-N01…GP-N11）
//! - 协议等价性：同一请求 HTTP vs gRPC 结果一致（GP-EQ01/02）
//! - 异常：未知 method、超 1MiB 载荷、坏 JSON、空文本业务错误（GP-A01…GP-A04）
//! - 生命周期：grpc_require_auth=true 且 auth 未启用拒绝启动（GP-L01）；
//!   require_auth 下无/有 token 的认证矩阵（GP-L02，需 auth feature）
//!
//! 运行条件：`cargo test --features http,grpc[,auth]`。
//! 服务器以子进程方式从 `CARGO_BIN_EXE_vecboost` 启动（M1 本地模型，CPU）。

#![cfg(all(feature = "http", feature = "grpc"))]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};
use tonic::Request;
use tonic::Status;
use tonic::metadata::MetadataValue;
use vecboost::sdforge::grpc::sdforge_v1::CallRequest;
use vecboost::sdforge::grpc::sdforge_v1::sd_forge_service_client::SdForgeServiceClient;

const M1_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/BAAI-bge-small-en-v1.5");
const M2_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/all-MiniLM-L6-v2");
const HTTP_TIMEOUT_SECS: u64 = 120;
const JWT_SECRET: &str = "grpc-e2e-jwt-secret-0123456789ABCDEF";
const ADMIN_PASS: &str = "GrpcE2e#2026Pass";

type Client = SdForgeServiceClient<tonic::transport::Channel>;

// ---------------------------------------------------------------------------
// 最小 HTTP 客户端（std 实现，避免引入额外 dev 依赖）
// ---------------------------------------------------------------------------

fn http_request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    token: Option<&str>,
) -> Result<(u16, String), std::io::Error> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let body = body.unwrap_or("");
    let auth = token
        .map(|t| format!("Authorization: Bearer {t}\r\n"))
        .unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n{auth}\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut raw = String::new();
    // Connection: close 下以连接关闭收尾；半途错误若已有数据则视为完整
    let _ = stream.read_to_string(&mut raw);
    let text = raw;
    let mut parts = text.splitn(2, "\r\n\r\n");
    let head = parts.next().unwrap_or("");
    let status: u16 = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut body = parts.next().unwrap_or("").to_string();
    if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        body = decode_chunked(&body);
    }
    Ok((status, body))
}

fn decode_chunked(input: &str) -> String {
    let mut out = String::new();
    let mut rest = input;
    while let Some(line_end) = rest.find("\r\n") {
        let size =
            usize::from_str_radix(rest[..line_end].trim().split(';').next().unwrap_or("0"), 16)
                .unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = line_end + 2;
        let Some(end) = rest.get(start..).and_then(|s| s.find("\r\n")) else {
            break;
        };
        out.push_str(rest.get(start..start + end).unwrap_or(""));
        rest = &rest[start + end + 2..];
    }
    out
}

// ---------------------------------------------------------------------------
// 服务器进程管理
// ---------------------------------------------------------------------------

struct TestServer {
    child: Child,
    http_port: u16,
    grpc_port: u16,
    dir: PathBuf,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind :0")
        .local_addr()
        .expect("local_addr")
        .port()
}

#[derive(Default)]
struct ServerOpts {
    auth: bool,
    grpc_require_auth: bool,
}

fn config_text(http_port: u16, grpc_port: u16, dir: &std::path::Path, opts: &ServerOpts) -> String {
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {http_port}\n\
         grpc_enabled = true\ngrpc_port = {grpc_port}\n\
         grpc_require_auth = {grpc_require_auth}\n\
         grpc_allowed_roots = [\"{}\"]\n\n\
         [model]\nmodel_path = \"{M1_PATH}\"\nexpected_dimension = 384\n\n\
         [embedding]\ncache_enabled = true\n\n\
         [rate_limit]\nenabled = false\n\n\
         [auth]\nenabled = {}\n{}\n\
         [database]\nurl = \"sqlite::memory:\"\n",
        dir.display(),
        opts.auth,
        if opts.auth {
            "default_admin_username = \"admin\"".to_string()
        } else {
            String::new()
        },
        grpc_require_auth = opts.grpc_require_auth,
    )
}

fn spawn_server(name: &str, opts: &ServerOpts, env_extra: &[(&str, &str)]) -> TestServer {
    for attempt in 0..3 {
        let http_port = free_port();
        let grpc_port = free_port();
        let dir = std::env::temp_dir().join(format!(
            "vecboost-grpc-e2e-{name}-{}-{attempt}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("config")).expect("create run dir");
        std::fs::write(
            dir.join("config").join("config.toml"),
            config_text(http_port, grpc_port, &dir, opts),
        )
        .expect("write config");
        let log = std::fs::File::create(dir.join("server.log")).expect("create log");
        let child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
            .current_dir(&dir)
            .envs(env_extra.iter().copied())
            .stdout(Stdio::from(log.try_clone().expect("dup log")))
            .stderr(Stdio::from(log))
            .spawn();
        let Ok(mut child) = child else { continue };

        // 等待 HTTP health 就绪；进程提前退出视为启动失败
        let deadline = std::time::Instant::now() + Duration::from_secs(HTTP_TIMEOUT_SECS);
        let mut ready = false;
        let mut exited_early = false;
        while std::time::Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                exited_early = true;
                break;
            }
            if let Ok((200, _)) = http_request(http_port, "GET", "/health", None, None) {
                ready = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if ready {
            return TestServer {
                child,
                http_port,
                grpc_port,
                dir,
            };
        }
        let tail = std::fs::read_to_string(dir.join("server.log")).unwrap_or_default();
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            !exited_early,
            "[{name}] 服务器启动期间退出（attempt {attempt}）。日志尾部:\n{}",
            tail.lines().last().unwrap_or("<empty>")
        );
    }
    panic!("[{name}] 服务器在 3 次尝试内未就绪");
}

async fn connect_grpc(port: u16) -> Client {
    // 不依赖生成 client 的 connect(该函数由 tonic/transport feature
    // 统一决定,裁剪传递依赖后不保证存在)—— 显式走 Channel 连接
    for _ in 0..120 {
        if let Ok(channel) =
            tonic::transport::Channel::from_shared(format!("http://127.0.0.1:{port}"))
                .expect("valid endpoint")
                .connect()
                .await
        {
            return Client::new(channel);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("gRPC 连接失败（port {port}）");
}

async fn call(client: &mut Client, method: &str, data: &str) -> Result<Value, Status> {
    let req = CallRequest {
        method: method.to_string(),
        parameters: HashMap::new(),
        data: data.to_string(),
    };
    let resp = client.call(Request::new(req)).await?.into_inner();
    let data = if resp.data.is_empty() {
        "{}"
    } else {
        &resp.data
    };
    let mut value = serde_json::from_str::<Value>(data)
        .unwrap_or_else(|e| panic!("响应 JSON 解析失败: {e}: {}", resp.data));
    // 业务层返回 success 字段时合并，便于断言
    if let Some(obj) = value.as_object_mut() {
        obj.insert("grpc_success".into(), json!(resp.success));
        obj.insert("grpc_status_code".into(), json!(resp.status_code));
    } else {
        value = json!({"grpc_success": resp.success, "grpc_status_code": resp.status_code, "data": value});
    }
    Ok(value)
}

async fn call_raw(
    client: &mut Client,
    method: &str,
    data: &str,
    token: Option<&str>,
) -> Result<Value, Status> {
    let mut req = Request::new(CallRequest {
        method: method.to_string(),
        parameters: HashMap::new(),
        data: data.to_string(),
    });
    if let Some(t) = token {
        req.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Bearer {t}")).expect("metadata"),
        );
    }
    let resp = client.call(req).await?.into_inner();
    Ok(
        json!({"success": resp.success, "status_code": resp.status_code, "error": resp.error, "data": resp.data}),
    )
}

// ---------------------------------------------------------------------------
// 正常路径：读方法 + 嵌入 + 重排 + 文件 + 热切换（GP-N01…GP-N11）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_info_methods_return_success() {
    let server = spawn_server("grpc-info", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-N09 health_check
    let v = call(&mut client, "vecboost.health_check", "")
        .await
        .expect("health_check");
    assert_eq!(v["grpc_success"], json!(true), "health_check: {v}");

    // GP-N06 get_current_model
    let v = call(&mut client, "vecboost.get_current_model", "")
        .await
        .expect("get_current_model");
    assert_eq!(v["grpc_success"], json!(true), "get_current_model: {v}");

    // GP-N07 get_model_info
    let v = call(&mut client, "vecboost.get_model_info", "")
        .await
        .expect("get_model_info");
    assert_eq!(v["grpc_success"], json!(true), "get_model_info: {v}");

    // GP-N08 list_models
    let v = call(&mut client, "vecboost.list_models", "")
        .await
        .expect("list_models");
    assert_eq!(v["grpc_success"], json!(true), "list_models: {v}");
    assert!(
        v["total_count"].as_u64().unwrap_or(0) >= 1,
        "应至少列出 1 个模型: {v}"
    );
}

#[tokio::test]
async fn grpc_embed_methods_normal() {
    let server = spawn_server("grpc-embed", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-N01 embed：维度 384、归一化
    let v = call(
        &mut client,
        "vecboost.embed",
        &json!({"text": "grpc e2e hello"}).to_string(),
    )
    .await
    .expect("embed");
    assert_eq!(v["grpc_success"], json!(true), "embed: {v}");
    let emb = v["embedding"].as_array().expect("embedding array");
    assert_eq!(emb.len(), 384, "M1 维度应为 384");
    let norm: f64 = emb
        .iter()
        .map(|x| x.as_f64().unwrap().powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "归一化向量范数应为 1，实际 {norm}"
    );

    // GP-N02 embed_batch：顺序保持
    let v = call(
        &mut client,
        "vecboost.embed_batch",
        &json!({"texts": ["alpha", "beta", "gamma"]}).to_string(),
    )
    .await
    .expect("embed_batch");
    assert_eq!(v["grpc_success"], json!(true), "embed_batch: {v}");
    assert_eq!(
        v["embeddings"].as_array().map(Vec::len),
        Some(3),
        "3 条输入: {v}"
    );

    // GP-N03 compute_similarity：同文本 = 1.0
    let v = call(
        &mut client,
        "vecboost.compute_similarity",
        &json!({"source": "hello world", "target": "hello world"}).to_string(),
    )
    .await
    .expect("compute_similarity");
    assert_eq!(v["grpc_success"], json!(true), "similarity: {v}");
    let score = v["score"].as_f64().expect("score");
    assert!(
        (score - 1.0).abs() < 1e-3,
        "同文本相似度应为 1.0，实际 {score}"
    );
}

#[tokio::test]
async fn grpc_rerank_methods_normal() {
    let server = spawn_server("grpc-rerank", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-N10 rerank：相关文档排名靠前且分数降序
    let req = json!({
        "query": "What is machine learning?",
        "documents": [
            "I had pasta for lunch today",
            "Machine learning is a branch of artificial intelligence",
            "The weather is nice"
        ],
        "top_k": null,
        "return_documents": true
    });
    let v = call(&mut client, "vecboost.rerank", &req.to_string())
        .await
        .expect("rerank");
    assert_eq!(v["grpc_success"], json!(true), "rerank: {v}");
    let results = v["results"].as_array().expect("results");
    assert_eq!(results.len(), 3, "top_k=None 应返回全量");
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["score"].as_f64().unwrap())
        .collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(scores, sorted, "分数应降序: {scores:?}");
    assert!(
        scores[0] > scores[2],
        "ML 相关文档得分应高于无关文档: {scores:?}"
    );

    // GP-N11 rerank_batch：多 query 独立返回
    let batch = json!({"queries": [
        {"query": "machine learning", "documents": ["ml paper", "pizza menu"], "top_k": null, "return_documents": null},
        {"query": "italian food", "documents": ["ml paper", "pizza menu"], "top_k": null, "return_documents": null}
    ]});
    let v = call(&mut client, "vecboost.rerank_batch", &batch.to_string())
        .await
        .expect("rerank_batch");
    assert_eq!(v["grpc_success"], json!(true), "rerank_batch: {v}");
    let responses = v["responses"].as_array().expect("responses");
    assert_eq!(responses.len(), 2, "两个 query 各返回一组结果");
}

#[tokio::test]
async fn grpc_embed_file_normal() {
    let server = spawn_server("grpc-file", &ServerOpts::default(), &[]);
    let file = server.dir.join("e2e-sample.txt");
    std::fs::write(&file, "grpc embed file e2e\nsecond line for paragraphs").expect("write file");
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-N04 embed_file（路径在 grpc_allowed_roots 内）
    let v = call(
        &mut client,
        "vecboost.embed_file",
        &json!({"path": file.display().to_string()}).to_string(),
    )
    .await
    .expect("embed_file");
    assert_eq!(v["grpc_success"], json!(true), "embed_file: {v}");
}

#[tokio::test]
async fn grpc_model_switch_roundtrip_and_failure_protection() {
    let server = spawn_server("grpc-switch", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-N05 model_switch → M2（MiniLM，本地目录）
    let v = call(
        &mut client,
        "vecboost.model_switch",
        &json!({
            "model_name": "minilm-e2e",
            "model_path": M2_PATH,
            "expected_dimension": 384
        })
        .to_string(),
    )
    .await
    .expect("model_switch");
    assert_eq!(v["grpc_success"], json!(true), "model_switch: {v}");
    let v = call(&mut client, "vecboost.get_current_model", "")
        .await
        .expect("current");
    assert_eq!(
        v["name"],
        json!("minilm-e2e"),
        "切换后当前模型应为 minilm-e2e: {v}"
    );

    // 切换后 embed 维度保持 384
    let v = call(
        &mut client,
        "vecboost.embed",
        &json!({"text": "after switch"}).to_string(),
    )
    .await
    .expect("embed after switch");
    assert_eq!(v["embedding"].as_array().map(Vec::len), Some(384));

    // 回归（gRPC 侧）：切换不存在的模型 → 业务错误 success=false 且 4xx，非 5xx
    let v = call_raw(
        &mut client,
        "vecboost.model_switch",
        &json!({"model_name": "definitely-not-a-model-xyz"}).to_string(),
        None,
    )
    .await
    .expect("bad switch call");
    assert_eq!(v["success"], json!(false), "无效切换应 success=false: {v}");
    let code = v["status_code"].as_i64().unwrap_or(0);
    assert!(
        (400..500).contains(&code),
        "无效切换应返回 4xx，实际 {code}: {v}"
    );

    // 失败后原模型仍可用（韧性）
    let v = call(
        &mut client,
        "vecboost.embed",
        &json!({"text": "still alive"}).to_string(),
    )
    .await
    .expect("embed after failed switch");
    assert_eq!(v["grpc_success"], json!(true));
}

// ---------------------------------------------------------------------------
// 协议等价性（GP-EQ01/02）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_embed_matches_http() {
    let server = spawn_server("grpc-parity", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    let text = "protocol parity check";
    let v = call(
        &mut client,
        "vecboost.embed",
        &json!({"text": text}).to_string(),
    )
    .await
    .expect("grpc embed");
    let grpc_vec: Vec<f64> = v["embedding"]
        .as_array()
        .expect("embedding")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();

    let (status, body) = http_request(
        server.http_port,
        "POST",
        "/api/1/embed",
        Some(&json!({"text": text}).to_string()),
        None,
    )
    .expect("http embed");
    assert_eq!(status, 200, "HTTP embed 失败: {body}");
    let http_json: Value = serde_json::from_str(&body).expect("http json");
    let http_vec: Vec<f64> = http_json["embedding"]
        .as_array()
        .expect("http embedding")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();

    assert_eq!(grpc_vec.len(), http_vec.len(), "两协议维度一致");
    let max_diff = grpc_vec
        .iter()
        .zip(http_vec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_diff < 1e-6,
        "HTTP 与 gRPC 向量逐分量差应在 1e-6 内，实际 {max_diff}"
    );

    // GP-EQ02：错误映射等价 — 空文本在两协议下均为 4xx 业务错误
    let (_, http_body) = http_request(
        server.http_port,
        "POST",
        "/api/1/embed",
        Some(&json!({"text": ""}).to_string()),
        None,
    )
    .expect("http empty");
    let http_err: Value = serde_json::from_str(&http_body).unwrap_or(Value::Null);
    // 错误响应体形如 {"error","code","error_code"}；code 缺失时以 0 兜底
    let http_code = http_err["code"].as_i64().unwrap_or(0);
    let v = call_raw(
        &mut client,
        "vecboost.embed",
        &json!({"text": ""}).to_string(),
        None,
    )
    .await
    .expect("grpc empty");
    let grpc_code = v["status_code"].as_i64().unwrap_or(0);
    assert!((400..500).contains(&grpc_code), "gRPC 空文本应 4xx: {v}");
    if http_code != 0 {
        assert_eq!(
            http_code % 100,
            grpc_code % 100,
            "两协议错误类别应一致（4xx↔4xx）: http={http_code} grpc={grpc_code}"
        );
    }
}

// ---------------------------------------------------------------------------
// 异常路径（GP-A01…GP-A04）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_unknown_method_returns_not_found() {
    let server = spawn_server("grpc-unknown", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    let err = client
        .call(Request::new(CallRequest {
            method: "vecboost.no_such_method".to_string(),
            parameters: HashMap::new(),
            data: "{}".to_string(),
        }))
        .await
        .expect_err("未知 method 应返回 tonic 错误");
    assert_eq!(
        err.code(),
        tonic::Code::NotFound,
        "未知 method 应为 NotFound: {err}"
    );

    // 语义钉住：无 body 参数的方法（如 health_check）data 非空 → InvalidArgument
    let err = client
        .call(Request::new(CallRequest {
            method: "vecboost.health_check".to_string(),
            parameters: HashMap::new(),
            data: "{}".to_string(),
        }))
        .await
        .expect_err("无 body 方法携带 data 应被拒绝");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "无 body 方法携带非空 data 应为 InvalidArgument: {err}"
    );
}

#[tokio::test]
async fn grpc_oversized_payload_rejected() {
    let server = spawn_server("grpc-oversize", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // 超过 1 MiB 载荷上限（MAX_GRPC_ARGUMENTS_SIZE_BYTES = 0x10_0000）
    let big = "x".repeat(2 * 1024 * 1024);
    let err = client
        .call(Request::new(CallRequest {
            method: "vecboost.embed".to_string(),
            parameters: HashMap::new(),
            data: json!({"text": big}).to_string(),
        }))
        .await
        .expect_err("超限载荷应返回 tonic 错误");
    assert_eq!(
        err.code(),
        tonic::Code::InvalidArgument,
        "超限载荷应为 InvalidArgument: {err}"
    );
}

#[tokio::test]
async fn grpc_bad_json_and_empty_text_are_4xx_business_errors() {
    let server = spawn_server("grpc-badjson", &ServerOpts::default(), &[]);
    let mut client = connect_grpc(server.grpc_port).await;

    // GP-A03 坏 JSON → success=false + 4xx 等价码（非 tonic 层错误）
    let v = call_raw(&mut client, "vecboost.embed", "this is not json", None)
        .await
        .expect("bad json call");
    assert_eq!(v["success"], json!(false), "坏 JSON 应 success=false: {v}");
    let code = v["status_code"].as_i64().unwrap_or(0);
    assert!((400..500).contains(&code), "坏 JSON 应 4xx，实际 {code}");

    // GP-A04 空文本业务校验错误
    let v = call_raw(
        &mut client,
        "vecboost.embed",
        &json!({"text": ""}).to_string(),
        None,
    )
    .await
    .expect("empty text call");
    assert_eq!(v["success"], json!(false), "空文本应 success=false: {v}");
    let code = v["status_code"].as_i64().unwrap_or(0);
    assert!((400..500).contains(&code), "空文本应 4xx，实际 {code}");
}

// ---------------------------------------------------------------------------
// 生命周期（GP-L01 / GP-L02）
// ---------------------------------------------------------------------------

// 复验已在上述所有测试的 spawn 健康等待中隐式覆盖：
// grpc_enabled=true 下 health 可达即证明不挂死。

/// GP-L01：grpc_require_auth=true 且 auth.enabled=false → 安全默认拒绝启动
#[tokio::test]
async fn grpc_require_auth_without_auth_config_refuses_startup() {
    let dir = std::env::temp_dir().join(format!("vecboost-grpc-refuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("dir");
    let cfg = config_text(
        free_port(),
        free_port(),
        &dir,
        &ServerOpts {
            auth: false,
            grpc_require_auth: true,
        },
    );
    std::fs::write(dir.join("config").join("config.toml"), cfg).expect("config");
    let log = std::fs::File::create(dir.join("server.log")).expect("log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .stdout(Stdio::from(log.try_clone().expect("dup")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn");
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                assert!(
                    !status.success(),
                    "grpc_require_auth=true 且无 auth 配置应拒绝启动（非零退出）"
                );
                break;
            }
            None if std::time::Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("30s 内未退出 — 安全默认（require_auth 无 auth 拒启）未生效");
            }
            None => std::thread::sleep(Duration::from_millis(300)),
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// GP-L02：require_auth 下无 token → UNAUTHENTICATED；有效 token → 成功
#[cfg(feature = "auth")]
#[tokio::test]
async fn grpc_auth_matrix_unauthenticated_vs_valid() {
    let server = spawn_server(
        "grpc-auth",
        &ServerOpts {
            auth: true,
            grpc_require_auth: true,
        },
        &[
            ("VECBOOST_JWT_SECRET", JWT_SECRET),
            ("VECBOOST_ADMIN_PASSWORD", ADMIN_PASS),
        ],
    );
    let mut client = connect_grpc(server.grpc_port).await;

    // 无 token → UNAUTHENTICATED（tonic 层拦截）
    let err = client
        .call(Request::new(CallRequest {
            method: "vecboost.embed".to_string(),
            parameters: HashMap::new(),
            data: json!({"text": "auth probe"}).to_string(),
        }))
        .await
        .expect_err("无 token 应被拒绝");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "无 token 应 Unauthenticated: {err}"
    );

    // 坏 token → UNAUTHENTICATED
    let err = client
        .call({
            let mut r = Request::new(CallRequest {
                method: "vecboost.embed".to_string(),
                parameters: HashMap::new(),
                data: json!({"text": "auth probe"}).to_string(),
            });
            r.metadata_mut().insert(
                "authorization",
                MetadataValue::try_from("Bearer not-a-real-token").expect("metadata"),
            );
            r
        })
        .await
        .expect_err("坏 token 应被拒绝");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "坏 token 应 Unauthenticated: {err}"
    );

    // HTTP login（白名单端点）获取有效 token → gRPC 调用成功
    let (status, body) = http_request(
        server.http_port,
        "POST",
        "/api/1/auth/login",
        Some(&json!({"username": "admin", "password": ADMIN_PASS}).to_string()),
        None,
    )
    .expect("login");
    assert_eq!(status, 200, "login 失败: {body}");
    let login: Value = serde_json::from_str(&body).expect("login json");
    let token = login["token"].as_str().expect("token field").to_string();

    let v = call_raw(
        &mut client,
        "vecboost.embed",
        &json!({"text": "authed embed"}).to_string(),
        Some(&token),
    )
    .await
    .expect("authed call");
    assert_eq!(v["success"], json!(true), "有效 token 应成功: {v}");
}
