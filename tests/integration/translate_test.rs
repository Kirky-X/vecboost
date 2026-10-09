// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Translate 端点 HTTP 集成测试 — /api/1/translate
//!
//! 沿用 detect_test 的真实服务器子进程模式。翻译路由表在启动期由
//! [translation].pairs 急切构建：真机用例消费本地 opus-mt bundle（缺席
//! SKIP，同 detect_test 先例）；零配置/超限/未配置对用例仅依赖服务器
//! 启动。配对用例在「测试二进制未启用 onnx feature」构建面（判据命令
//! `--features http`）显性 SKIP；完整覆盖用 `--features http,onnx`。
//!
//! 覆盖：wire 五键形态钉（含 detected_source_language 回填）、
//! 未配置 pair 400（含 pairs 清单文案）、零配置 400
//! translate-not-configured、超 token 400、auto 不可靠检测 400、
//! 401 未登录（auth feature 门控）、429 限流形态（auth feature 门控）。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::common::{TestServer, free_port, model_ready};

use serde_json::{Value, json};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/BAAI-bge-small-en-v1.5");
const OPUS_EN_ZH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");
const HTTP_TIMEOUT_SECS: u64 = 120;

#[cfg(feature = "auth")]
const JWT_SECRET: &str = "translate-e2e-jwt-secret-0123456789ABCDEF";
#[cfg(feature = "auth")]
const ADMIN_PASS: &str = "TranslateE2e#2026Pass";

/// opus bundle 探测：真机翻译用例的 SKIP 守卫（onnx 权重文件在位即可，
/// encoder/decoder 的 onnx 子目录文件由 T013 下载流程保证成套）
fn opus_ready() -> bool {
    Path::new(OPUS_EN_ZH)
        .join("onnx/encoder_model.onnx")
        .exists()
        && Path::new(OPUS_EN_ZH)
            .join("onnx/decoder_model_merged.onnx")
            .exists()
        && Path::new(OPUS_EN_ZH).join("tokenizer.json").exists()
}

// TestServer, free_port, model_ready 复用 crate::common

#[derive(Default)]
struct ServerOpts {
    auth: bool,
    rate_limit_rpm: Option<u64>,
    /// [translation] 段整体替换（None = 零配置段不写入）
    translation_section: Option<String>,
}

fn config_text(http_port: u16, opts: &ServerOpts) -> String {
    let rate_limit_section = match opts.rate_limit_rpm {
        Some(rpm) => format!(
            "[rate_limit]\nenabled = true\nip_requests_per_minute = {rpm}\nheaders_enabled = true\n\n"
        ),
        None => "[rate_limit]\nenabled = false\n\n".to_string(),
    };
    let translation_section = opts.translation_section.clone().unwrap_or_default();
    format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {http_port}\n\
         grpc_enabled = false\n\n\
         [model]\nmodel_path = \"{MODEL_PATH}\"\nexpected_dimension = 384\n\n\
         [embedding]\ncache_enabled = true\n\n\
         {rate_limit_section}\
         [auth]\nenabled = {}\n{}\n\
         [database]\nurl = \"sqlite::memory:\"\n\
         {translation_section}",
        opts.auth,
        if opts.auth {
            "default_admin_username = \"admin\"".to_string()
        } else {
            String::new()
        },
    )
}

/// 启动失败时返回 Err（日志尾部）。配对用例据此区分「二进制未启用
/// onnx feature」（SKIP 语义，判据命令 --features http 构建面）与真异常。
fn spawn_translate_server(
    name: &str,
    opts: &ServerOpts,
    env_extra: &[(&str, &str)],
) -> Result<TestServer, String> {
    let http_port = free_port();
    let dir = std::env::temp_dir().join(format!(
        "vecboost-translate-e2e-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("config")).expect("create run dir");
    std::fs::write(
        dir.join("config").join("config.toml"),
        config_text(http_port, opts),
    )
    .expect("write config");
    let log = std::fs::File::create(dir.join("server.log")).expect("create log");
    // ort/load-dynamic dylib：服务器 cwd 为临时目录，相对默认落位不可解析，
    // 注入绝对路径（环境变量优先口径，调用方 env_extra 显式值可覆盖）
    let ort_dylib = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnxruntime/libonnxruntime.so"
        )
        .to_string()
    });
    let child = Command::new(env!("CARGO_BIN_EXE_vecboost"))
        .current_dir(&dir)
        .env("ORT_DYLIB_PATH", &ort_dylib)
        .envs(env_extra.iter().copied())
        .stdout(Stdio::from(log.try_clone().expect("dup log")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn vecboost binary");
    let mut child = child;

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
        return Err(format!(
            "[{name}] translate 测试服务器未就绪。日志尾部:\n{}",
            all_lines[last..].join("\n")
        ));
    }
    Ok(TestServer {
        child,
        http_port,
        dir,
    })
}

/// 配对用例的启动处理：二进制未启用 onnx feature 时返回 None（调用方
/// 打印 SKIP 后 return，判据命令 `--features http` 构建面），其余启动失败
/// 按异常 panic。
fn require_pair_server(server: Result<TestServer, String>, test: &str) -> Option<TestServer> {
    match server {
        Ok(s) => Some(s),
        Err(log) if log.contains("未启用 `onnx` feature") => {
            println!(
                "SKIP [{test}]: 测试二进制未启用 onnx feature（--features http 构建面）；配对用例需 --features http,onnx 构建"
            );
            None
        }
        Err(log) => panic!("{log}"),
    }
}

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
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
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

fn translate_body(src: Option<&str>, tgt: &str, text: &str) -> String {
    match src {
        Some(s) => json!({"text": text, "source_language": s, "target_language": tgt}).to_string(),
        None => json!({"text": text, "target_language": tgt}).to_string(),
    }
}

fn en_zh_translation_section(extra: &str) -> Option<String> {
    Some(format!(
        "[translation]\n\n[[translation.pairs]]\nsource_language = \"en\"\ntarget_language = \"zh\"\nmodel_path = \"{OPUS_EN_ZH}\"\n{extra}\n"
    ))
}

// ---------------------------------------------------------------------------
// 真机翻译：wire 五键形态 + auto 检测回填（SKIP 守卫锚 bundle）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn translate_wire_shape_and_real_translation() {
    if !model_ready(MODEL_PATH) || !opus_ready() {
        println!("SKIP [translate_wire_shape_and_real_translation]: 本地模型/bundle 缺失");
        return;
    }
    let server = match require_pair_server(
        spawn_translate_server(
            "translate-wire",
            &ServerOpts {
                translation_section: en_zh_translation_section(""),
                ..Default::default()
            },
            &[],
        ),
        "translate_wire_shape_and_real_translation",
    ) {
        Some(s) => s,
        None => return,
    };
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", "Hello, my dog is cute")),
        None,
        None,
    )
    .expect("translate request");
    assert_eq!(status, 200, "真机翻译应 200: {body}");
    let resp: Value = serde_json::from_str(&body).expect("response json");
    for key in [
        "translation",
        "source_language",
        "target_language",
        "detected_source_language",
        "processing_time_ms",
    ] {
        assert!(resp.get(key).is_some(), "missing key {key}, body={body}");
    }
    assert!(!resp["translation"].as_str().unwrap_or("").trim().is_empty());
    assert!(
        !resp["translation"].as_str().unwrap().contains("</s>"),
        "特殊 token 不得泄漏: {body}"
    );
    assert_eq!(resp["source_language"], json!("en"), "请求回显");
    assert_eq!(resp["target_language"], json!("zh"));
    // explicit 请求不回填检测
    assert_eq!(resp["detected_source_language"], json!(null));

    // auto：source_language 缺省 → 检测回填 en（样例用长句：whatlang 对
    // 短句可靠性阈值高，短句 unreliable 走 400 translate-detect-failed）
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(
            None,
            "zh",
            "The quick brown fox jumps over the lazy dog near the riverbank every morning.",
        )),
        None,
        None,
    )
    .expect("auto request");
    assert_eq!(status, 200, "auto 翻译应 200: {body}");
    let resp: Value = serde_json::from_str(&body).expect("auto json");
    assert_eq!(
        resp["detected_source_language"],
        json!("en"),
        "检测回填: {body}"
    );
    assert_eq!(resp["source_language"], json!(null), "auto 语义 null 显性");
}

// ---------------------------------------------------------------------------
// 未配置 pair / 零配置 / 超限 / 不可靠检测
// ---------------------------------------------------------------------------

#[tokio::test]
async fn translate_unconfigured_pair_400_with_pairs_list() {
    if !model_ready(MODEL_PATH) {
        println!("SKIP [translate_unconfigured_pair_400_with_pairs_list]: 本地模型缺失");
        return;
    }
    let server = match require_pair_server(
        spawn_translate_server(
            "translate-pair400",
            &ServerOpts {
                translation_section: en_zh_translation_section(""),
                ..Default::default()
            },
            &[],
        ),
        "translate_unconfigured_pair_400_with_pairs_list",
    ) {
        Some(s) => s,
        None => return,
    };
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "fr", "bonjour")),
        None,
        None,
    )
    .expect("pair request");
    assert_eq!(status, 400, "未配置对必须 400: {body}");
    assert!(body.contains("en-fr"), "消息必须含请求对: {body}");
    assert!(body.contains("en-zh"), "消息必须含已配置对清单: {body}");
}

#[tokio::test]
async fn translate_zero_config_400_not_configured() {
    if !model_ready(MODEL_PATH) {
        println!("SKIP [translate_zero_config_400_not_configured]: 本地模型缺失");
        return;
    }
    // 零 [translation] 段：服务无条件预构建为空表（端点注册、显性拒绝）
    let server = spawn_translate_server("translate-zero", &ServerOpts::default(), &[])
        .unwrap_or_else(|log| panic!("{log}"));
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", "hello")),
        None,
        Some("en"),
    )
    .expect("zero request");
    assert_eq!(status, 400, "零配置必须 400（端点注册、显性拒绝）: {body}");
    // T024 落 FTL 后消息为本地化文案（Accept-Language=en 锚定英文），断言
    // 文案锚点而非裸键名
    assert!(
        body.contains("no language pairs configured"),
        "零配置文案必须为 translate-not-configured 渲染: {body}"
    );
}

#[tokio::test]
async fn translate_over_budget_400() {
    if !model_ready(MODEL_PATH) || !opus_ready() {
        println!("SKIP [translate_over_budget_400]: 本地模型/bundle 缺失");
        return;
    }
    let server = match require_pair_server(
        spawn_translate_server(
            "translate-budget",
            &ServerOpts {
                translation_section: en_zh_translation_section("max_source_tokens = 8"),
                ..Default::default()
            },
            &[],
        ),
        "translate_over_budget_400",
    ) {
        Some(s) => s,
        None => return,
    };
    let long_text = "word ".repeat(20);
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", long_text.trim())),
        None,
        Some("en"),
    )
    .expect("budget request");
    assert_eq!(status, 400, "超 token 必须 400: {body}");
    // 文案锚点 + 动态值（实际 21 / 上限 8，locale 无关）
    assert!(
        body.contains("exceeds the pair limit") && body.contains("21") && body.contains("8"),
        "消息必须为超限文案渲染（含实际/上限）: {body}"
    );
}

#[tokio::test]
async fn translate_unreliable_auto_detection_400() {
    if !model_ready(MODEL_PATH) || !opus_ready() {
        println!("SKIP [translate_unreliable_auto_detection_400]: 本地模型/bundle 缺失");
        return;
    }
    let server = match require_pair_server(
        spawn_translate_server(
            "translate-detect",
            &ServerOpts {
                translation_section: en_zh_translation_section(""),
                ..Default::default()
            },
            &[],
        ),
        "translate_unreliable_auto_detection_400",
    ) {
        Some(s) => s,
        None => return,
    };
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(None, "zh", "1234567890")),
        None,
        Some("en"),
    )
    .expect("detect request");
    assert_eq!(status, 400, "不可靠 auto 检测必须 400: {body}");
    assert!(
        body.contains("detection is unreliable"),
        "消息必须为检测失败文案渲染: {body}"
    );
}

// ---------------------------------------------------------------------------
// 401 / 429（auth feature 门控，同 detect_test 先例）
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
#[tokio::test]
async fn translate_unauthenticated_returns_401() {
    if !model_ready(MODEL_PATH) {
        println!("SKIP [translate_unauthenticated_returns_401]: 本地模型缺失");
        return;
    }
    let server = spawn_translate_server(
        "translate-401",
        &ServerOpts {
            auth: true,
            ..Default::default()
        },
        &[
            ("VECBOOST_JWT_SECRET", JWT_SECRET),
            ("VECBOOST_ADMIN_PASSWORD", ADMIN_PASS),
        ],
    )
    .unwrap_or_else(|log| panic!("{log}"));
    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", "hello")),
        None,
        None,
    )
    .expect("no-token request");
    assert_eq!(status, 401, "未登录必须 401: {body}");

    let (status, _, body) = http_request_full(
        server.http_port,
        "POST",
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", "hello")),
        Some("not-a-real-token"),
        None,
    )
    .expect("bad-token request");
    assert_eq!(status, 401, "坏 token 必须 401: {body}");

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
        "/api/1/translate",
        Some(&translate_body(Some("en"), "zh", "hello")),
        Some(&token),
        None,
    )
    .expect("authed request");
    assert_eq!(
        status, 400,
        "有效 token 下零配置应 400（非 401 鉴权面）: {body}"
    );
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn translate_rate_limit_returns_429_with_headers() {
    if !model_ready(MODEL_PATH) {
        println!("SKIP [translate_rate_limit_returns_429_with_headers]: 本地模型缺失");
        return;
    }
    let server = spawn_translate_server(
        "translate-rl429",
        &ServerOpts {
            rate_limit_rpm: Some(2),
            ..Default::default()
        },
        &[],
    )
    .unwrap_or_else(|log| panic!("{log}"));
    let mut hit = false;
    for i in 0..8 {
        let (status, headers, _body) = http_request_full(
            server.http_port,
            "POST",
            "/api/1/translate",
            Some(&translate_body(Some("en"), "zh", "hello")),
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
    assert!(hit, "超过 ip_requests_per_minute=2 后必须 429");
}
