// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 认证面集成测试（`auth` feature）：登录/鉴权/刷新/登出闭环。
//!
//! 与 `http_surface.rs` 使用同一 ONNX 模型与裸 TCP 请求方式，但启用
//! `[auth]`（JWT + 内存会话）以覆盖 `src/auth/middleware.rs` 与
//! `src/api/auth.rs` 的 401/403/400 分支。模型缺席时 SKIP。

#![cfg(feature = "auth")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);
const ADMIN_PASSWORD: &str = "AuthSurface#2026Pass";
const JWT_SECRET: &str = "auth-surface-jwt-secret-0123456789ABCDEF";

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
    token: Option<&str>,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
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
        raw.splitn(2, "\r\n\r\n").nth(1).unwrap_or("").to_string(),
    ))
}

fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    token: Option<&str>,
) -> (u16, String) {
    try_request(port, method, path, body, token).expect("server reachable")
}

fn config_text(port: u16) -> String {
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\n\
         grpc_enabled = false\n\n\
         [model]\nmodel_path = \"{MODEL_PATH}\"\nmodel_repo = \"Xenova/bge-small-en-v1.5\"\n\
         engine_type = \"onnx\"\nexpected_dimension = 384\n\n\
         [embedding]\ncache_enabled = true\n\n\
         [rate_limit]\nenabled = false\n\n\
         [database]\nurl = \"sqlite::memory:\"\n\n\
         [auth]\nenabled = true\njwt_secret = \"{JWT_SECRET}\"\n\
         default_admin_username = \"admin\"\ndefault_admin_password = \"{ADMIN_PASSWORD}\"\n"
    )
}

#[test]
fn auth_login_authorize_refresh_logout() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失（models/Xenova-bge-small-en-v1.5）");
        return;
    }

    let port = free_port();
    let dir = std::env::temp_dir().join(format!("vecboost-auth-surface-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    std::fs::write(dir.join("config").join("config.toml"), config_text(port))
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
        if let Ok((200, _)) = try_request(port, "GET", "/health", None, None) {
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
            "认证测试服务器未就绪，日志尾部:\n{}",
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

    // 健康检查与登录端点公开（无需 token）
    assert_eq!(request(port, "GET", "/health", None, None).0, 200);

    // 登录：成功 / 密码错误 / 用户名校验
    let (status, body) = request(
        port,
        "POST",
        "/api/v1/auth/login",
        Some(&format!(
            r#"{{"username":"admin","password":"{ADMIN_PASSWORD}"}}"#
        )),
        None,
    );
    assert_eq!(status, 200, "登录应成功: {body}");
    let token = body
        .split("\"token\":\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("登录响应必须含 token")
        .to_string();

    assert!(
        (400..500).contains(
            &request(
                port,
                "POST",
                "/api/v1/auth/login",
                Some(r#"{"username":"admin","password":"wrong-password"}"#),
                None
            )
            .0
        ),
        "错误密码必须拒绝"
    );
    assert!(
        (400..500).contains(
            &request(
                port,
                "POST",
                "/api/v1/auth/login",
                Some(r#"{"username":"bad user!","password":"x"}"#),
                None
            )
            .0
        ),
        "非法用户名格式必须拒绝"
    );

    // 受保护端点：无 token 401，有 token 200
    assert_eq!(
        request(port, "GET", "/api/v1/models", None, None).0,
        401,
        "受保护端点无 token 必须 401"
    );
    assert_eq!(
        request(port, "GET", "/api/v1/models", None, Some(&token)).0,
        200,
        "合法 token 必须放行"
    );
    assert_eq!(
        request(port, "GET", "/api/v1/auth/me", None, Some(&token)).0,
        200
    );

    // 刷新：非法 refresh_token 必须拒绝
    assert!(
        (400..500).contains(
            &request(
                port,
                "POST",
                "/api/v1/auth/refresh",
                Some(r#"{"refresh_token":"not-a-token"}"#),
                None
            )
            .0
        ),
        "非法 refresh_token 必须拒绝"
    );

    // 登出后 token 立即失效
    assert!(
        (200..300).contains(&request(port, "POST", "/api/v1/auth/logout", None, Some(&token)).0),
        "登出应成功"
    );
    assert_eq!(
        request(port, "GET", "/api/v1/models", None, Some(&token)).0,
        401,
        "登出后的 token 必须失效"
    );

    // 优雅终止（SIGTERM）：让子进程走过 shutdown 钩子并 flush 覆盖率计数器。
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
