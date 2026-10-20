// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Redis 共享状态服务面：限流 / 精确缓存 / 语义模糊索引三处 `backend="redis"`。
//!
//! 需要本机或 CI service 的 Redis（`VECBOOST_TEST_REDIS_URL`，默认
//! `redis://127.0.0.1:6379`）；不可达时 SKIP。

#![cfg(all(feature = "http", feature = "distributed-redis"))]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);

fn model_ready() -> bool {
    Path::new(MODEL_PATH).join("model.onnx").exists()
        && Path::new(MODEL_PATH).join("tokenizer.json").exists()
}

fn redis_url() -> String {
    std::env::var("VECBOOST_TEST_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

fn redis_reachable(url: &str) -> bool {
    let host_port = url
        .trim_start_matches("redis://")
        .split('/')
        .next()
        .unwrap_or("127.0.0.1:6379");
    TcpStream::connect(host_port).is_ok()
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
    Ok((
        status,
        raw.split_once("\r\n\r\n")
            .map(|x| x.1)
            .unwrap_or("")
            .to_string(),
    ))
}

fn request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    try_request(port, method, path, body).expect("server reachable")
}

fn config_text(port: u16, redis: &str) -> String {
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\ngrpc_enabled = false\n\n\
         [model]\nmodel_path = \"{MODEL_PATH}\"\nmodel_repo = \"Xenova/bge-small-en-v1.5\"\n\
         engine_type = \"onnx\"\nexpected_dimension = 384\n\n\
         [embedding]
cache_enabled = true
cache_backend = \"redis\"
redis_url = \"{redis}\"

\
         [rate_limit]
enabled = true
backend = \"redis\"
redis_url = \"{redis}\"
\
         ip_requests_per_minute = 600\n\n\
         [semantic_cache]
enabled = true
backend = \"redis\"
redis_url = \"{redis}\"

\
         [database]\nurl = \"sqlite::memory:\"\n\n\
         [auth]\nenabled = false\n"
    )
}

#[test]
fn redis_shared_backends_serve_requests() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    let redis = redis_url();
    if !redis_reachable(&redis) {
        println!("SKIP: Redis 不可达（{redis}）");
        return;
    }

    let port = free_port();
    let dir = std::env::temp_dir().join(format!("vecboost-redis-surface-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        config_text(port, &redis),
    )
    .expect("write config");

    let log = std::fs::File::create(dir.join("server.log")).expect("create log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .env(
            "ORT_DYLIB_PATH",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/3rdparty/onnxruntime/libonnxruntime.so"
            ),
        )
        .env("OXCACHE_ALLOW_INSECURE_REDIS", "I_UNDERSTAND_THE_RISKS")
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
        if let Ok((200, _)) = try_request(port, "GET", "/health", None) {
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
            "Redis 后端服务器未就绪:\n{}",
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

    // 健康与共享后端就绪
    assert_eq!(request(port, "GET", "/health", None).0, 200);
    assert_eq!(request(port, "GET", "/health?depth=full", None).0, 200);
    assert_eq!(request(port, "GET", "/metrics", None).0, 200);

    // 精确缓存 Redis：两次相同请求（第二次命中共享缓存）
    for _ in 0..2 {
        assert_eq!(
            request(
                port,
                "POST",
                "/api/v1/embed",
                Some(r#"{"text":"redis shared"}"#)
            )
            .0,
            200
        );
    }
    // 语义模糊层 Redis：近义请求（命中共享 trigram 索引）
    assert_eq!(
        request(
            port,
            "POST",
            "/api/v1/embed",
            Some(r#"{"text":"redis shared state"}"#)
        )
        .0,
        200
    );
    // 限流 Redis 后端仍正常放行
    assert_eq!(
        request(
            port,
            "POST",
            "/api/v1/embed/batch",
            Some(r#"{"texts":["a","b"]}"#)
        )
        .0,
        200
    );

    // 优雅终止，flush 覆盖率
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
