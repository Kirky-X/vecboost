// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 启动期配置护栏与服务侧限流面。
//!
//! 覆盖三类"配置矛盾必须拒绝启动"的分支（语义缓存/精确缓存/限流的 redis 后端
//! 缺 URL、翻译 token 窗口矛盾），以及 `/metrics` 的限流闸门与公开路径豁免。
//! 需要本地 ONNX 模型与 ORT dylib，缺席时 SKIP。

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
const ORT_DYLIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnxruntime/libonnxruntime.so"
);
const TRANSLATE_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");

fn assets_ready() -> bool {
    Path::new(MODEL_PATH).join("model.onnx").is_file() && Path::new(ORT_DYLIB).is_file()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind :0")
        .local_addr()
        .expect("local_addr")
        .port()
}

fn head(toml: &str, port: u16) -> String {
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\ngrpc_enabled = false\n\n\
         [model]\nmodel_path = \"{MODEL_PATH}\"\nmodel_repo = \"Xenova/bge-small-en-v1.5\"\n\
         engine_type = \"onnx\"\nexpected_dimension = 384\n\n\
         [database]\nurl = \"sqlite::memory:\"\n\n{toml}"
    )
}

/// 运行到进程退出（超时视作"未 fail-fast"直接失败）。
fn run_until_exit(name: &str, toml: &str, envs: &[(&str, &str)]) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("vecboost-guard-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("mkdir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        head(toml, free_port()),
    )
    .expect("write config");

    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .env("ORT_DYLIB_PATH", ORT_DYLIB)
        .env("OXCACHE_ALLOW_INSECURE_REDIS", "I_UNDERSTAND_THE_RISKS")
        .envs(envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let deadline = Instant::now() + Duration::from_secs(90);
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
    let out = child.wait_with_output().expect("wait output");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        exited,
        "[{name}] 应在 90s 内因配置矛盾退出，而不是挂起等待请求"
    );
    (out.status.code(), text)
}

#[test]
fn semantic_cache_redis_backend_without_url_fails_startup() {
    if !assets_ready() {
        println!("SKIP: 模型或 ORT dylib 缺失");
        return;
    }
    let (code, text) = run_until_exit(
        "semantic-no-url",
        "[semantic_cache]\nenabled = true\nbackend = \"redis\"\n\n[auth]\nenabled = false\n",
        &[],
    );
    assert!(matches!(code, Some(c) if c != 0), "输出:\n{text}");
    assert!(
        text.contains("redis_url"),
        "应点名缺失的 redis_url，输出:\n{text}"
    );
}

#[test]
fn embedding_cache_redis_backend_without_url_fails_startup() {
    if !assets_ready() {
        println!("SKIP: 模型或 ORT dylib 缺失");
        return;
    }
    let (code, text) = run_until_exit(
        "cache-no-url",
        "[embedding]\ncache_enabled = true\ncache_backend = \"redis\"\n\n[auth]\nenabled = false\n",
        &[],
    );
    assert!(matches!(code, Some(c) if c != 0), "输出:\n{text}");
    assert!(
        text.contains("redis_url"),
        "应点名缺失的 redis_url，输出:\n{text}"
    );
}

#[test]
fn rate_limit_redis_backend_without_url_fails_startup() {
    if !assets_ready() {
        println!("SKIP: 模型或 ORT dylib 缺失");
        return;
    }
    let (code, text) = run_until_exit(
        "ratelimit-no-url",
        "[rate_limit]\nenabled = true\nbackend = \"redis\"\n\n[auth]\nenabled = false\n",
        &[],
    );
    assert!(matches!(code, Some(c) if c != 0), "输出:\n{text}");
    assert!(
        text.contains("redis_url"),
        "应点名缺失的 redis_url，输出:\n{text}"
    );
}

#[test]
fn translation_pair_with_inverted_token_window_fails_startup() {
    if !assets_ready() {
        println!("SKIP: 模型或 ORT dylib 缺失");
        return;
    }
    let (code, text) = run_until_exit(
        "translate-window",
        &format!(
            "[auth]\nenabled = false\n\n\
             [[translation.pairs]]\nsource_language = \"en\"\ntarget_language = \"zh\"\n\
             model_path = \"{TRANSLATE_MODEL}\"\nmax_new_tokens = 8\nmin_new_tokens = 64\n"
        ),
        &[],
    );
    assert!(matches!(code, Some(c) if c != 0), "输出:\n{text}");
    assert!(
        text.contains("min_new_tokens") || text.contains("translation.pairs"),
        "应点名翻译语言对的窗口矛盾，输出:\n{text}"
    );
}

/// `/metrics` 限流闸门：IP 维度超阈值后必须 429，且 `/health` 仍公开放行。
#[test]
fn metrics_endpoint_is_rate_limited_while_health_stays_public() {
    if !assets_ready() {
        println!("SKIP: 模型或 ORT dylib 缺失");
        return;
    }
    let port = free_port();
    let dir = std::env::temp_dir().join(format!("vecboost-metrics-rl-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("mkdir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        head(
            "[rate_limit]\nenabled = true\nip_requests_per_minute = 3\n\n\
             [auth]\nenabled = false\n",
            port,
        ),
    )
    .expect("write config");

    let log = std::fs::File::create(dir.join("server.log")).expect("create log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .env("ORT_DYLIB_PATH", ORT_DYLIB)
        .stdout(Stdio::from(log.try_clone().expect("dup")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn");

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut ready = false;
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        if let Ok((200, _)) = get(port, "/health") {
            ready = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    if !ready {
        let tail = std::fs::read_to_string(dir.join("server.log")).unwrap_or_default();
        let _ = child.kill();
        let _ = child.wait();
        panic!("限流测试服务器未就绪:\n{tail}");
    }

    let mut seen_429 = false;
    for _ in 0..12 {
        if let Ok((status, _)) = get(port, "/metrics")
            && status == 429
        {
            seen_429 = true;
            break;
        }
    }
    assert!(
        seen_429,
        "ip_requests_per_minute=3 时 /metrics 必须出现 429"
    );
    assert_eq!(
        get(port, "/health").expect("health").0,
        200,
        "健康检查须始终公开放行"
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

fn get(port: u16, path: &str) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes())?;
    let mut raw = String::new();
    let _ = stream.read_to_string(&mut raw);
    let status = raw
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    Ok((status, raw))
}
