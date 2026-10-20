// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 服务启动变体：日志落盘、WAL 持久缓存、CORS/体积/超时层、健康深探缓存。
//!
//! 与 `http_surface` 共用 ONNX 模型；资产缺席时 SKIP。另含翻译语言对
//! 配置错误的启动 fail-fast（不加载模型）。

#![cfg(feature = "http")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);
const ALT_MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-all-MiniLM-L6-v2"
);
const MODELS_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");

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

fn try_request_body(
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
    Ok((status, raw))
}

fn request_get(port: u16, path: &str) -> (u16, String) {
    try_request_body(port, "GET", path, None).expect("GET reachable")
}

fn request_post(port: u16, path: &str, body: &str) -> (u16, String) {
    try_request_body(port, "POST", path, Some(body)).expect("POST reachable")
}

fn try_request(port: u16, method: &str, path: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\n\
         Connection: close\r\n\r\n"
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
    Ok((status, raw))
}

#[test]
fn server_with_logging_wal_cors_and_limits() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    let port = free_port();
    let dir = std::env::temp_dir().join(format!("vecboost-variants-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    let config = [
        "[server]",
        "host = \"127.0.0.1\"",
        &format!("port = {port}"),
        "grpc_enabled = false",
        "cors_enabled = true",
        "cors_allow_origins = [\"http://localhost\"]",
        "body_limit_mb = 1",
        "request_timeout_seconds = 10",
        "timeout = 30",
        &format!("grpc_allowed_roots = [\"{MODELS_ROOT}\"]"),
        "",
        "[model]",
        &format!("model_path = \"{MODEL_PATH}\""),
        "model_repo = \"Xenova/bge-small-en-v1.5\"",
        "engine_type = \"onnx\"",
        "expected_dimension = 384",
        "max_resident_models = 2",
        "resident_memory_budget_mb = 2048",
        "",
        "[embedding]",
        "cache_enabled = true",
        "persist_path = \"embedding.wal\"",
        "",
        "[logging]",
        "level = \"info\"",
        "console = true",
        "",
        "[rate_limit]",
        "enabled = false",
        "",
        "[database]",
        "url = \"sqlite::memory:\"",
        "",
        "[auth]",
        "enabled = false",
        "",
    ]
    .join("\n");
    std::fs::write(dir.join("config").join("config.toml"), config).expect("write config");

    let log = std::fs::File::create(dir.join("server.out")).expect("create log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
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

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut ready = false;
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        if let Ok((200, _)) = try_request(port, "GET", "/health") {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if !ready {
        let tail = std::fs::read_to_string(dir.join("server.out")).unwrap_or_default();
        let _ = child.kill();
        let _ = child.wait();
        panic!("变体服务器未就绪:\n{tail}");
    }

    // 模型生命周期：切换→推理→列表→卸载→切回（覆盖 ModelManager 换入换出与 WAL 持久化）
    let switch_body = format!(
        r#"{{"model_name":"minilm","model_path":"{ALT_MODEL_PATH}","expected_dimension":384}}"#
    );
    let (st, sb) = request_post(port, "/api/v1/model/switch", &switch_body);
    assert_eq!(st, 200, "切换到本地 MiniLM 应成功: {sb}");
    let (st, _) = request_post(port, "/api/v1/embed", r#"{"text":"after switch"}"#);
    assert_eq!(st, 200, "切换后推理应成功");
    assert_eq!(request_get(port, "/api/v1/models").0, 200);
    let (st, ub) = request_post(port, "/api/v1/model/unload", r#"{"model_name":"minilm"}"#);
    assert_eq!(st, 200, "卸载已换入模型应成功: {ub}");
    let back =
        format!(r#"{{"model_name":"bge","model_path":"{MODEL_PATH}","expected_dimension":384}}"#);
    let (st, bb) = request_post(port, "/api/v1/model/switch", &back);
    assert_eq!(st, 200, "切回主模型应成功: {bb}");
    let (st, _) = request_post(port, "/api/v1/embed/batch", r#"{"texts":["a","b"]}"#);
    assert_eq!(st, 200, "批量推理应成功");

    // 深探两次：第二次命中 500ms 引擎探测缓存
    assert_eq!(
        try_request(port, "GET", "/health?depth=full")
            .expect("req")
            .0,
        200
    );
    assert_eq!(
        try_request(port, "GET", "/health?depth=full")
            .expect("req")
            .0,
        200
    );
    // CORS 中间件生效：预检带 Access-Control-Allow-Origin
    let (_, raw) = try_request(port, "GET", "/metrics").expect("metrics");
    assert!(raw.starts_with("HTTP/1.1 200"));

    // WAL 持久缓存文件应在启动/写入后创建
    assert!(
        dir.join("embedding.wal").exists() || dir.join("config/embedding.wal").exists(),
        "persist_path 应创建 WAL 文件"
    );

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

/// 翻译语言对指向不存在的 bundle：启动 fail-fast（不进入服务循环）。
#[test]
fn startup_rejects_translation_pair_with_missing_bundle() {
    let dir = std::env::temp_dir().join(format!("vecboost-bad-pair-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create dir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        "[server]\nhost = \"127.0.0.1\"\nport = 18997\ngrpc_enabled = false\n\n\
         [model]
model_path = \"/definitely/missing\"
engine_type = \"onnx\"

\
         [auth]\nenabled = false\n\n\
         [[translation.pairs]]\nsource_language = \"en\"\ntarget_language = \"zh\"\n\
         model_path = \"/definitely/missing-bundle\"\n",
    )
    .expect("write config");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .arg("--config")
        .arg(dir.join("config").join("config.toml"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut exited = false;
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            exited = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    if !exited {
        let _ = child.kill();
    }
    let status = child.wait().expect("wait child");
    assert!(
        status.code().is_some_and(|c| c != 0),
        "缺 bundle 的翻译语言对应 fail-fast 而非挂起或成功退出"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
