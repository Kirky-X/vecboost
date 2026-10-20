// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! HTTP 业务面集成测试：一个真实 onnx 服务进程覆盖 `/api/v1/*` 与 `/v1/embeddings`。
//!
//! 与 detect/translate 的 candle 专属用例不同，本目标用 ONNX bundle
//! （`models/Xenova-bge-small-en-v1.5`）启动默认 onnx 引擎，因此在
//! `default`/`http` 构建下即可运行；模型缺席时打印 SKIP 返回。
//!
//! 覆盖重点（第二期覆盖率门禁）：健康检查三档、指标端点、模型元数据、
//! embed/batch/similarity/search 的成功与参数错误分支、`/embed/file`
//! 的允许根与越界/缺失分支、model switch/unload、OpenAI 兼容端点与坏 JSON。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);
const TRANSLATE_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");
const READY_TIMEOUT: Duration = Duration::from_secs(120);

fn model_ready() -> bool {
    Path::new(MODEL_PATH).join("model.onnx").exists()
        && Path::new(MODEL_PATH).join("tokenizer.json").exists()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind :0")
        .local_addr()
        .expect("local_addr")
        .port()
}

fn try_request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut raw = String::new();
    let _ = stream.read_to_string(&mut raw);
    let head = raw.split("\r\n\r\n").next().unwrap_or("");
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let resp_body = raw
        .split_once("\r\n\r\n")
        .map(|x| x.1)
        .unwrap_or("")
        .to_string();
    Ok((status, resp_body))
}

fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    try_request(port, method, path, body).expect("server reachable")
}

fn config_text(port: u16, read_root: &Path) -> String {
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\n\
         grpc_enabled = false\ngrpc_allowed_roots = [\"{root}\"]\n\n\
         [model]\nmodel_path = \"{model}\"\nmodel_repo = \"Xenova/bge-small-en-v1.5\"\n\
         engine_type = \"onnx\"
expected_dimension = 384
batch_size = 8

\
         [embedding]\ncache_enabled = true\n\n\
         [rate_limit]\nenabled = false\n\n\
         [database]\nurl = \"sqlite::memory:\"\n\n\
         [auth]\nenabled = false\n\n\
         [[translation.pairs]]\nsource_language = \"en\"\ntarget_language = \"zh\"\n\
         model_path = \"{translate_model}\"\n",
        root = read_root.display(),
        model = MODEL_PATH,
        translate_model = TRANSLATE_MODEL_PATH,
    )
}

#[test]
fn http_surface_success_and_error_paths() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失（models/Xenova-bge-small-en-v1.5）");
        return;
    }

    let http_port = free_port();
    let dir = std::env::temp_dir().join(format!("vecboost-http-surface-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    let document = dir.join("document.txt");
    std::fs::write(
        &document,
        "VecBoost 是 Rust 编写的推理服务。\n\n第二段用于文件聚合嵌入。\n\n第三段。\n",
    )
    .expect("write document");
    std::fs::write(
        dir.join("config").join("config.toml"),
        config_text(http_port, &dir),
    )
    .expect("write config");

    let log = std::fs::File::create(dir.join("server.log")).expect("create log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .arg("--warmup")
        .arg("1")
        .env(
            "ORT_DYLIB_PATH",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/3rdparty/onnxruntime/libonnxruntime.so"
            ),
        )
        .stdout(Stdio::from(log.try_clone().expect("dup log")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn vecboost");

    let deadline = Instant::now() + READY_TIMEOUT;
    let mut ready = false;
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        if let Ok((200, _)) = try_request(http_port, "GET", "/health", None) {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if !ready {
        let tail = std::fs::read_to_string(dir.join("server.log")).unwrap_or_default();
        let _ = child.kill();
        let _ = child.wait();
        panic!(
            "服务器未就绪，日志尾部:\n{}",
            tail.lines()
                .rev()
                .take(20)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    let body = |path: &str, payload: &str| request(http_port, "POST", path, Some(payload));

    // ── 健康检查三档 ──
    assert_eq!(request(http_port, "GET", "/health", None).0, 200);
    assert_eq!(request(http_port, "GET", "/health?depth=full", None).0, 200);
    assert_eq!(
        request(http_port, "GET", "/health?depth=bogus", None).0,
        200,
        "未知 depth 按浅探处理"
    );

    // ── 指标端点 ──
    assert_eq!(request(http_port, "GET", "/metrics", None).0, 200);

    // ── 模型元数据 ──
    assert_eq!(request(http_port, "GET", "/api/v1/models", None).0, 200);
    assert_eq!(
        request(http_port, "GET", "/api/v1/model/current", None).0,
        200
    );
    assert_eq!(request(http_port, "GET", "/api/v1/model/info", None).0, 200);

    // ── embed ──
    let (s, b) = body("/api/v1/embed", r#"{"text":"hello vecboost"}"#);
    assert_eq!(s, 200, "embed 应成功: {b}");
    assert!(b.contains("embedding"));
    assert!(
        (400..500).contains(&body("/api/v1/embed", r#"{"text":""}"#).0),
        "空文本应拒绝"
    );
    assert!(
        (400..500).contains(&body("/api/v1/embed", r#"{"text":123}"#).0),
        "字段类型错误应拒绝"
    );

    // ── embed/batch ──
    assert_eq!(body("/api/v1/embed/batch", r#"{"texts":["a","b"]}"#).0, 200);
    assert!(
        (400..500).contains(&body("/api/v1/embed/batch", r#"{"texts":[]}"#).0),
        "空批次应拒绝"
    );

    // ── similarity ──
    assert_eq!(
        body(
            "/api/v1/similarity",
            r#"{"source":"a","target":"b","metric":"cosine"}"#
        )
        .0,
        200
    );
    assert!(
        (400..500).contains(
            &body(
                "/api/v1/similarity",
                r#"{"source":"a","target":"b","metric":"bogus"}"#
            )
            .0
        ),
        "未知度量应拒绝"
    );

    // ── search ──
    assert_eq!(
        body(
            "/api/v1/search",
            r#"{"query":"vecboost","texts":["vecboost","other"],"top_k":1}"#
        )
        .0,
        200
    );
    assert!(
        (400..500).contains(&body("/api/v1/search", r#"{"query":"x","texts":[]}"#).0),
        "空候选集应拒绝"
    );

    // ── embed/file：允许根内成功、缺失与越界拒绝 ──
    assert_eq!(
        body(
            "/api/v1/embed/file",
            &format!(r#"{{"path":"{}"}}"#, document.display())
        )
        .0,
        200
    );
    assert!(
        (400..500).contains(&body("/api/v1/embed/file", r#"{"path":"/etc/passwd"}"#).0),
        "根外路径应拒绝"
    );
    assert!(
        (400..500).contains(&body("/api/v1/embed/file", r#"{"path":"missing.txt"}"#).0),
        "缺失文件应拒绝"
    );

    // ── model switch / unload ──
    assert!(
        (400..500).contains(
            &body(
                "/api/v1/model/switch",
                r#"{"model_name":"nope","model_path":"/definitely/missing"}"#
            )
            .0
        ),
        "不存在的切换路径应拒绝"
    );
    let (s, _) = body("/api/v1/model/unload", r#"{"model_name":"never-loaded"}"#);
    assert!(
        (200..300).contains(&s) || s == 404,
        "卸载未加载模型：就绪态 2xx，无 ModelManager 的主路径 404；got {s}"
    );

    // ── OpenAI 兼容端点 ──
    assert_eq!(
        body(
            "/v1/embeddings",
            r#"{"input":"hello","model":"Xenova/bge-small-en-v1.5"}"#
        )
        .0,
        200
    );
    assert_eq!(
        body(
            "/v1/embeddings",
            r#"{"input":["a","b"],"model":"Xenova/bge-small-en-v1.5","encoding_format":"base64"}"#
        )
        .0,
        200
    );
    assert!(
        (400..500).contains(
            &body(
                "/v1/embeddings",
                r#"{"input":"hello","model":"unknown-model"}"#
            )
            .0
        ),
        "未知模型应拒绝"
    );
    assert!(
        (400..500).contains(&body("/v1/embeddings", r#"{"input":123}"#).0),
        "非法 input 应拒绝"
    );

    // ── 翻译：成功 / 未配置语言对 / 空文本 ──
    assert_eq!(
        body(
            "/api/v1/translate",
            r#"{"text":"Hello world","source_language":"en","target_language":"zh"}"#
        )
        .0,
        200
    );
    assert!(
        (400..500).contains(
            &body(
                "/api/v1/translate",
                r#"{"text":"Hello","source_language":"en","target_language":"fr"}"#
            )
            .0
        ),
        "未配置语言对必须拒绝"
    );
    assert!(
        (400..500).contains(
            &body(
                "/api/v1/translate",
                r#"{"text":"","source_language":"en","target_language":"zh"}"#
            )
            .0
        ),
        "空文本必须拒绝"
    );

    // ── 坏 JSON ──
    assert!(
        (400..500).contains(&body("/api/v1/embed", r#"{"text":"#).0),
        "坏 JSON 应 4xx"
    );

    // 优雅终止（SIGTERM）：子进程需正常退出以便覆盖率/日志正常 flush；
    // SIGKILL 会丢失子进程的 profile 计数器与 shutdown 路径。
    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status();
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}

/// 启动期 fail-fast：配置缺失/非法、绑定安全闸门、加密强制闸门。
/// 全部在模型加载前返回，不依赖任何模型资产。
#[test]
fn startup_fail_fast_surface() {
    let dir = std::env::temp_dir().join(format!("vecboost-startup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");

    fn run(dir: &Path, config: &str, envs: &[(&str, &str)]) -> (Option<i32>, String) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vecboost"));
        cmd.current_dir(dir).arg("--config").arg(config);
        for (k, v) in envs {
            cmd.env(k, v);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if child.try_wait().ok().flatten().is_none() {
            let _ = child.kill();
            let _ = child.wait();
            return (None, "timeout".to_string());
        }
        let out = child.wait_with_output().expect("wait_with_output");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), text)
    }

    // 1) 配置文件不存在 → exit 2，并点名路径
    let (code, text) = run(&dir, "missing-config.toml", &[]);
    assert_eq!(code, Some(2), "缺配置应 exit 2；stderr={text}");
    assert!(text.contains("missing-config.toml"));

    // 2) 非法 TOML → 非零退出（当前配置加载器在解析失败时 panic，exit 101；
    //    已登记为待收敛缺陷：应改为 Err → exit 1，不在本变更扩大范围）
    let bad = dir.join("bad.toml");
    std::fs::write(&bad, "[server\nport = ").expect("write bad toml");
    let (code, _) = run(&dir, bad.to_str().expect("utf8"), &[]);
    assert!(matches!(code, Some(c) if c != 0), "非法配置必须非零退出");

    // 3) 非回环绑定 + 认证关闭 → 拒绝启动（无模型加载）
    let insecure = dir.join("insecure.toml");
    std::fs::write(
        &insecure,
        "[server]\nhost = \"0.0.0.0\"\nport = 18999\n\n[auth]\nenabled = false\n",
    )
    .expect("write insecure config");
    let (code, text) = run(&dir, insecure.to_str().expect("utf8"), &[]);
    assert!(
        matches!(code, Some(c) if c != 0),
        "非回环+无认证必须拒绝启动；code={code:?} out={text}"
    );

    // 4) 强制加密但无 key → 拒绝启动
    let plaintext = dir.join("plaintext.toml");
    std::fs::write(
        &plaintext,
        "[server]\nhost = \"127.0.0.1\"\nport = 18998\n\n\
         [auth]\nenabled = true\njwt_secret = \"plaintext-secret\"\n\
         default_admin_username = \"admin\"\ndefault_admin_password = \"Plaintext#2026\"\n",
    )
    .expect("write plaintext config");
    let (code, text) = run(
        &dir,
        plaintext.to_str().expect("utf8"),
        &[("VECBOOST_REQUIRE_ENCRYPTION", "1")],
    );
    assert!(
        matches!(code, Some(c) if c != 0),
        "强制加密且无 key 必须拒绝启动；code={code:?} out={text}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
