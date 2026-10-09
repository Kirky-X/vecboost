// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Detect 端点 HTTP 集成测试 — /api/1/detect（R-language-detection-*）
//!
//! 沿用 grpc_e2e 的真实服务器子进程模式（`CARGO_BIN_EXE_vecboost` + 本地
//! M1 模型路径，CPU，离线可跑）；本地模型缺席时打印 SKIP 后返回（同
//! grpc_e2e / model_snapshot_regression 先例）。检测服务本身无引擎依赖，
//! 但服务器启动需加载任一模型，SKIP 守卫仍锚定模型在位。
//!
//! 覆盖：wire 形态钉（五键齐全 + confidence 数值域 [0,1]）、边界输入
//! （空文本 400 / 纯数字 200+`language:null` / 超长 400 / 控制字符 400）、
//! Accept-Language zh/en 文案、限流 429 形态、401 未登录形态
//! （需 auth feature，cfg 门控同 grpc_e2e 认证矩阵先例）。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::common::{TestServer, free_port, model_ready};

use serde_json::{Value, json};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/BAAI-bge-small-en-v1.5");
const HTTP_TIMEOUT_SECS: u64 = 120;

// 401 认证矩阵用例专用（auth feature 下编译，否则 -D warnings dead_code）
#[cfg(feature = "auth")]
const JWT_SECRET: &str = "detect-e2e-jwt-secret-0123456789ABCDEF";
#[cfg(feature = "auth")]
const ADMIN_PASS: &str = "DetectE2e#2026Pass";

fn skip_without_model(test_name: &str) -> bool {
    if model_ready(MODEL_PATH) {
        return false;
    }
    println!("SKIP [{test_name}]: 本地模型缺失（models/ 未 checkout），跳过 detect HTTP 集成测试");
    true
}

// ---------------------------------------------------------------------------
// 服务器进程管理（复用 crate::common::{TestServer, free_port, model_ready}）
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ServerOpts {
    auth: bool,
    rate_limit_rpm: Option<u64>,
}

fn config_text(http_port: u16, opts: &ServerOpts) -> String {
    let rate_limit_section = match opts.rate_limit_rpm {
        Some(rpm) => format!(
            "[rate_limit]\nenabled = true\nip_requests_per_minute = {rpm}\nheaders_enabled = true\n\n"
        ),
        None => "[rate_limit]\nenabled = false\n\n".to_string(),
    };
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {http_port}\n\
         grpc_enabled = false\n\n\
         [model]\nmodel_path = \"{MODEL_PATH}\"\nexpected_dimension = 384\n\n\
         [embedding]\ncache_enabled = true\n\n\
         {rate_limit_section}\
         [auth]\nenabled = {}\n{}\n\
         [database]\nurl = \"sqlite::memory:\"\n",
        opts.auth,
        if opts.auth {
            "default_admin_username = \"admin\"".to_string()
        } else {
            String::new()
        },
    )
}

fn spawn_detect_server(name: &str, opts: &ServerOpts, env_extra: &[(&str, &str)]) -> TestServer {
    let http_port = free_port();
    let dir =
        std::env::temp_dir().join(format!("vecboost-detect-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        config_text(http_port, opts),
    )
    .expect("write config");
    let log = std::fs::File::create(dir.join("server.log")).expect("create log");
    let child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .envs(env_extra.iter().copied())
        .stdout(Stdio::from(log.try_clone().expect("dup log")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn vecboost binary");
    let mut child = child;

    // 等待 HTTP health 就绪；进程提前退出视为启动失败
    let deadline = std::time::Instant::now() + Duration::from_secs(HTTP_TIMEOUT_SECS);
    let mut ready = false;
    while std::time::Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        if let Ok((200, _)) = http_request(http_port, "GET", "/health", None, None, None) {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if !ready {
        let tail = std::fs::read_to_string(dir.join("server.log")).unwrap_or_default();
        let _ = child.kill();
        let _ = child.wait();
        let all_lines: Vec<&str> = tail.lines().collect();
        let last = all_lines.len().saturating_sub(10);
        panic!(
            "[{name}] detect 测试服务器未就绪。日志尾部:\n{}",
            all_lines[last..].join("\n")
        );
    }
    TestServer {
        child,
        http_port,
        dir,
    }
}

// ---------------------------------------------------------------------------
// HTTP 辅助（grpc_e2e 同款裸 TCP，扩展 Accept-Language 头）
// ---------------------------------------------------------------------------

/// 返回 (status, 响应头键值对（键小写）, body)
type HttpResponse = (u16, Vec<(String, String)>, String);

fn http_request_full(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    token: Option<&str>,
    accept_language: Option<&str>,
) -> Result<HttpResponse, std::io::Error> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let body = body.unwrap_or("");
    let auth = token
        .map(|t| format!("Authorization: Bearer {t}\r\n"))
        .unwrap_or_default();
    let lang = accept_language
        .map(|l| format!("Accept-Language: {l}\r\n"))
        .unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n{auth}{lang}\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut raw = String::new();
    let _ = stream.read_to_string(&mut raw);
    let mut parts = raw.splitn(2, "\r\n\r\n");
    let head = parts.next().unwrap_or("");
    let status: u16 = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad status line"))?;
    let headers = head
        .lines()
        .skip(1)
        .filter_map(|l| {
            l.split_once(':')
                .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        })
        .collect();
    let body = parts.next().unwrap_or("").to_string();
    Ok((status, headers, body))
}

fn http_request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    token: Option<&str>,
    accept_language: Option<&str>,
) -> Result<(u16, String), std::io::Error> {
    let (status, _, body) = http_request_full(port, method, path, body, token, accept_language)?;
    Ok((status, body))
}

fn detect_body(text: &str) -> String {
    json!({"text": text}).to_string()
}

// ---------------------------------------------------------------------------
// wire 形态钉 + 边界输入
// ---------------------------------------------------------------------------

#[tokio::test]
async fn detect_wire_shape_and_undetectable_semantics() {
    if skip_without_model("detect_wire_shape_and_undetectable_semantics") {
        return;
    }
    let server = spawn_detect_server("detect-wire", &ServerOpts::default(), &[]);

    // 正常英文句：200 + 五键齐全 + 映射 + confidence 数值域
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body(
            "The quick brown fox jumps over the lazy dog near the riverbank every morning.",
        )),
        None,
        None,
    )
    .expect("detect request");
    assert_eq!(status, 200, "英文句应 200: {body}");
    let resp: Value = serde_json::from_str(&body).expect("response json");
    for key in [
        "language",
        "script",
        "confidence",
        "is_reliable",
        "processing_time_ms",
    ] {
        assert!(resp.get(key).is_some(), "missing key {key}, body={body}");
    }
    assert_eq!(resp["language"], json!("en"), "639-1 映射: {body}");
    assert_eq!(resp["script"], json!("Latin"));
    let confidence = resp["confidence"].as_f64().expect("confidence f64");
    assert!(
        (0.0..=1.0).contains(&confidence),
        "confidence 数值域 [0,1]: {confidence}"
    );
    assert_eq!(resp["is_reliable"], json!(true));
    assert!(resp["processing_time_ms"].is_u64());

    // 纯数字：200 语义的 language:null + is_reliable:false（显性表达，非 Err）
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("1234567890")),
        None,
        None,
    )
    .expect("digits request");
    assert_eq!(status, 200, "纯数字应 200（不可判定显性表达）: {body}");
    let resp: Value = serde_json::from_str(&body).expect("digits json");
    assert_eq!(
        resp["language"],
        json!(null),
        "language 必须显性 null: {body}"
    );
    assert_eq!(resp["is_reliable"], json!(false));
    assert_eq!(resp["confidence"], json!(0.0));
}

#[tokio::test]
async fn detect_empty_text_400_with_accept_language_zh_en() {
    if skip_without_model("detect_empty_text_400_with_accept_language_zh_en") {
        return;
    }
    let server = spawn_detect_server("detect-i18n", &ServerOpts::default(), &[]);

    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("")),
        None,
        Some("zh"),
    )
    .expect("zh request");
    assert_eq!(status, 400, "空文本应 400: {body}");
    assert!(
        body.contains("文本不能为空"),
        "Accept-Language=zh 必须回中文文案: {body}"
    );

    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("")),
        None,
        Some("en"),
    )
    .expect("en request");
    assert_eq!(status, 400, "空文本应 400: {body}");
    assert!(
        body.contains("must not be empty"),
        "Accept-Language=en 必须回英文文案: {body}"
    );
}

#[tokio::test]
async fn detect_rejects_oversize_and_control_chars() {
    if skip_without_model("detect_rejects_oversize_and_control_chars") {
        return;
    }
    let server = spawn_detect_server("detect-boundary", &ServerOpts::default(), &[]);

    // 超长：10_001 汉字（按 chars().count() 计，非字节）→ 400
    let oversized = "水".repeat(10_001);
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body(&oversized)),
        None,
        None,
    )
    .expect("oversize request");
    assert_eq!(
        status,
        400,
        "10_001 汉字必须 400: {}",
        &body[..body.len().min(200)]
    );
    assert!(
        body.contains("10001"),
        "超长文案必须携带实际长度插值: {}",
        &body[..body.len().min(300)]
    );

    // 控制字符 \u0001 → 400（serde_json 裸控制字节与转义形态双防线之上，
    // 此处验证转义可解析形态同样被域层防线拒绝）
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(r#"{"text":"line1\u0001line2"}"#),
        None,
        None,
    )
    .expect("control char request");
    assert_eq!(status, 400, "\\u0001 控制字符必须 400: {body}");
}

// ---------------------------------------------------------------------------
// 限流 429 形态（auth feature：全局限流中间件 auth_rate_limit_middleware
// 为 auth 门控，非 auth 构建无限流链路，本测试随门裁剪——grpc_e2e
// decision_auth_rate_limit_hit_returns_429_with_headers 同款先例）
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
#[tokio::test]
async fn detect_rate_limit_returns_429_with_headers() {
    if skip_without_model("detect_rate_limit_returns_429_with_headers") {
        return;
    }
    let server = spawn_detect_server(
        "detect-rl429",
        &ServerOpts {
            rate_limit_rpm: Some(2),
            ..Default::default()
        },
        &[],
    );
    let mut hit = false;
    for i in 0..8 {
        let (status, headers, _body) = http_request_full(
            server.http_port,
            "POST",
            "/api/1/detect",
            Some(&detect_body("hello world")),
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("request {i} failed: {e}"));
        if status == 429 {
            hit = true;
            let has = |name: &str| headers.iter().any(|(k, _)| k == name);
            assert!(
                has("ratelimit-limit") && has("ratelimit-remaining") && has("ratelimit-reset"),
                "429 必须携带 IETF RateLimit-* 头: {headers:?}"
            );
            assert!(has("retry-after"), "429 必须携带 Retry-After: {headers:?}");
            break;
        }
    }
    assert!(
        hit,
        "超过 ip_requests_per_minute=2 后 /api/1/detect 必须 429"
    );
}

// ---------------------------------------------------------------------------
// 401 未登录形态（auth feature；http-only 下无鉴权面，cfg 门控同 grpc_e2e）
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
#[tokio::test]
async fn detect_unauthenticated_returns_401() {
    if skip_without_model("detect_unauthenticated_returns_401") {
        return;
    }
    let server = spawn_detect_server(
        "detect-401",
        &ServerOpts {
            auth: true,
            ..Default::default()
        },
        &[
            ("VECBOOST_JWT_SECRET", JWT_SECRET),
            ("VECBOOST_ADMIN_PASSWORD", ADMIN_PASS),
        ],
    );

    // 无 token → 401
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("hello world")),
        None,
        None,
    )
    .expect("no-token request");
    assert_eq!(status, 401, "未登录必须 401: {body}");

    // 坏 token → 401
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("hello world")),
        Some("not-a-real-token"),
        None,
    )
    .expect("bad-token request");
    assert_eq!(status, 401, "坏 token 必须 401: {body}");

    // 有效登录 → 200（对照：证明 401 来自鉴权而非端点缺陷）
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/auth/login",
        Some(&json!({"username": "admin", "password": ADMIN_PASS}).to_string()),
        None,
        None,
    )
    .expect("login request");
    assert_eq!(status, 200, "login 失败: {body}");
    let login: Value = serde_json::from_str(&body).expect("login json");
    let token = login["token"].as_str().expect("token field").to_string();
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/detect",
        Some(&detect_body("hello world")),
        Some(&token),
        None,
    )
    .expect("authed request");
    assert_eq!(status, 200, "有效 token 应 200: {body}");
}
