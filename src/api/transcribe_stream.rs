// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 流式转录 WebSocket 端点（opt-in feature `transcribe-stream`）。
//!
//! 与现有 `POST /api/v1/transcribe` 完全分离：独立 model/Session、独立
//! session 状态表、独立背压，现有 POST 路径零改动。
//!
//! 协议（sdforge WebSocketMessage JSON）：
//!   client → {"type":"request","id":"..","method":"audio","params":{
//!       "session_id":"s1","audio":"<base64 pcm i16>","end":false}}
//!   server → {"type":"response","id":"..","result":{
//!       "type":"partial"|"final","text":"..","session_id":"s1"}}

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use sdforge::forge;
use sdforge::websocket::{BoxFuture, WebSocketHandler, WebSocketMessage};
use serde_json::json;

use crate::config::app::StreamTranscribeConfig;
use crate::engine::transcribe::correction::{ContextBiasing, Rescorer, correct_candidates};
use crate::engine::transcribe::streaming_asr::{StreamState, StreamingAsr, advance_decode};

/// handler 全局状态：一套模型（Mutex 串行，与 Qwen3/Whisper 同口径）+ 多流状态表。
pub struct StreamingWsHandler {
    asr: Mutex<StreamingAsr>,
    states: Mutex<HashMap<String, (StreamState, std::time::Instant)>>,
    biasing: ContextBiasing,
    rescorer: Rescorer,
    max_sessions: usize,
}

impl StreamingWsHandler {
    /// 从配置装配；模型加载失败返回错误（fail-fast）。
    pub fn try_new(cfg: &StreamTranscribeConfig) -> Result<Self, String> {
        let words_path = cfg.biasing_words_path.as_deref();
        let asr = StreamingAsr::load(
            std::path::Path::new(&cfg.models_dir),
            0,
            words_path,
            cfg.biasing_weight,
        )
        .map_err(|e| format!("streaming model load failed: {e}"))?;
        let words: Vec<String> = match cfg.biasing_words_path.as_deref() {
            Some(p) => match std::fs::read_to_string(p) {
                Ok(text) => text
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect(),
                Err(e) => {
                    log::warn!(
                        "{}",
                        crate::i18n::tr_with_args(
                            "log-stream-biasing-list-unavailable-a4f1",
                            crate::i18n::tr_args(&[("path", p), ("e", &e.to_string())]),
                        )
                    );
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        let biasing = ContextBiasing::from_words(words.clone(), cfg.biasing_weight);
        let rescorer = Rescorer::new(&words, cfg.lm_weight);
        Ok(Self {
            asr: Mutex::new(asr),
            states: Mutex::new(HashMap::new()),
            biasing,
            rescorer,
            max_sessions: cfg.max_sessions,
        })
    }
}

/// 启动期由 build_http_router 注入的流配置（forge 工厂零参，无法直接传参）。
static STREAM_CONFIG: std::sync::OnceLock<StreamTranscribeConfig> = std::sync::OnceLock::new();

/// 进程级共享 handler（启动期构造，请求路径零 panic）。
static HANDLER: std::sync::OnceLock<Arc<StreamingWsHandler>> = std::sync::OnceLock::new();

/// 启动期注入配置并预构造 handler：模型/词表加载失败时返回 Err，
/// 由 main.rs 拒绝启动（fail-fast），避免请求路径 panic（安全审查 H1）。
pub fn init_stream_handler(cfg: StreamTranscribeConfig) -> Result<(), String> {
    STREAM_CONFIG
        .set(cfg.clone())
        .map_err(|_| "stream config already initialized".to_string())?;
    let handler = Arc::new(StreamingWsHandler::try_new(&cfg)?);
    HANDLER
        .set(handler)
        .map_err(|_| "stream handler already initialized".to_string())
}

/// 已初始化的 handler（启动期 init_stream_handler 保证存在）。
fn shared_handler() -> Arc<StreamingWsHandler> {
    HANDLER
        .get()
        .expect("stream handler must be initialized before WS endpoint is mounted")
        .clone()
}

impl WebSocketHandler for StreamingWsHandler {
    fn handle(&self, message: WebSocketMessage) -> BoxFuture<'static, WebSocketMessage> {
        let handler = shared_handler();
        Box::pin(async move {
            const SESSION_TTL: std::time::Duration = std::time::Duration::from_secs(30);
            // 稳定错误码（详情写日志，不回传客户端——防内部细节泄漏，安全审查 F6）
            let err = |id: &str, code: &str| WebSocketMessage::Error {
                id: id.to_string(),
                error: code.to_string(),
            };
            let WebSocketMessage::Request { id, method, params } = message else {
                return err("", "invalid_message");
            };
            if method != "audio" {
                return err(&id, "unknown_method");
            }

            // TTL 清理：sdforge 无断连回调，靠时间戳惰性回收 abandoned 会话（安全 F4）
            {
                let mut states = handler.states.lock().unwrap();
                states.retain(|_, (_, last)| last.elapsed() < SESSION_TTL);
            }

            let end = params.get("end").and_then(|v| v.as_bool()).unwrap_or(false);
            let client_sid = params.get("session_id").and_then(|v| v.as_str());

            // 结束帧：final + 纠错 + 移除
            if end {
                let text = {
                    let mut states = handler.states.lock().unwrap();
                    let sid = client_sid.unwrap_or("");
                    match states.remove(sid) {
                        Some((mut state, _)) => {
                            let asr = handler.asr.lock().unwrap();
                            asr.finalize(&mut state); // flush 尾音（架构 F2）
                            let cands = asr.finish_candidates(&state);
                            drop(asr);
                            correct_candidates(&cands, &handler.biasing, &handler.rescorer)
                        }
                        None => String::new(),
                    }
                };
                let sid = client_sid.unwrap_or("").to_string();
                return WebSocketMessage::Response {
                    id,
                    result: json!({"type": "final", "text": text, "session_id": sid}),
                };
            }

            // 音频帧：base64 → i16 PCM → f32
            let audio_b64 = params.get("audio").and_then(|v| v.as_str()).unwrap_or("");
            let bytes =
                match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, audio_b64)
                {
                    Ok(b) => b,
                    Err(e) => {
                        log::debug!("ws audio base64 decode failed: {e}");
                        return err(&id, "invalid_audio");
                    }
                };
            // 单帧上限 = 1 个推理 chunk（5120 样本 / 10240 字节 i16）。
            // 原 768KB 阈值在 1MB 消息上限下不可触发（base64 3/4 解码率 + JSON
            // 开销使最大解码值恰好低于阈值），无法限制单条消息内的推理次数
            // （安全复审 F5）。单帧 5120 样本 → 至多 1 次三图推理。
            if bytes.len() > 5120 * 2 {
                return err(&id, "audio_too_large");
            }
            let pcm: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();

            // session_id：首帧由服务端生成随机不可猜 id；后续必须携带该 id
            // （防跨会话污染，安全审查 F3）
            let (sid, is_new) = {
                let mut states = handler.states.lock().unwrap();
                match client_sid {
                    Some(s) if states.contains_key(s) => {
                        let entry = states.get_mut(s).unwrap();
                        entry.1 = std::time::Instant::now();
                        (s.to_string(), false)
                    }
                    Some(s) => {
                        // 客户端自报未知 sid：拒绝（防伪造占用/污染）
                        if !s.is_empty() {
                            return err(&id, "unknown_session");
                        }
                        let new = format!("s-{}", uuid::Uuid::new_v4());
                        if states.len() >= handler.max_sessions {
                            return err(&id, "too_many_sessions");
                        }
                        // 官方绑定版：state 需由 recognizer 创建（OnlineStream）。
                        // 锁顺序 states → asr，与推理/end 路径一致，无死锁环。
                        let asr = handler.asr.lock().unwrap();
                        states.insert(new.clone(), (asr.new_state(), std::time::Instant::now()));
                        drop(asr);
                        (new, true)
                    }
                    None => {
                        let new = format!("s-{}", uuid::Uuid::new_v4());
                        if states.len() >= handler.max_sessions {
                            return err(&id, "too_many_sessions");
                        }
                        let asr = handler.asr.lock().unwrap();
                        states.insert(new.clone(), (asr.new_state(), std::time::Instant::now()));
                        drop(asr);
                        (new, true)
                    }
                }
            };

            // 推理：用 remove/insert 取出 state，避免跨推理持有 states 锁（性能审查 M2）
            let mut st = {
                let mut states = handler.states.lock().unwrap();
                states.remove(&sid).map(|(s, _)| s)
            };
            let Some(mut st) = st else {
                return err(&id, "session_lost");
            };

            let asr_result = {
                let mut asr = handler.asr.lock().unwrap();
                let mut result = Ok(());
                for chunk in pcm.chunks(320) {
                    match asr.encode_frame(&mut st, chunk) {
                        // PCM 未凑满一个 chunk 时返回空 logits 是正常路径（每 20ms
                        // 一帧、每 320ms 才推理一次）——不可当错误（架构复审 H1-2）
                        Ok(logits) if logits.is_empty() => {}
                        Ok(logits) => {
                            if let Err(e) = advance_decode(&mut st, &logits) {
                                result = Err(e);
                                break;
                            }
                        }
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
                result
            };
            if let Err(e) = asr_result {
                log::error!("ws streaming inference failed: {e}");
                // 归还 state（错误时也要归还，避免会话丢失）
                let mut states = handler.states.lock().unwrap();
                states.insert(sid.clone(), (st, std::time::Instant::now()));
                return err(&id, "infer_failed");
            }
            let text = {
                let asr = handler.asr.lock().unwrap();
                asr.partial_text(&st)
            };
            // 归还 state + 更新时间戳
            {
                let mut states = handler.states.lock().unwrap();
                states.insert(sid.clone(), (st, std::time::Instant::now()));
            }

            WebSocketMessage::Response {
                id,
                result: json!({
                    "type": "partial", "text": text, "session_id": sid,
                    "new_session": is_new,
                }),
            }
        })
    }
}

/// forge 工厂：注册 WS 路由 `/api/v1/transcribe/stream`。
#[forge(
    name = "transcribe_stream",
    version = "v1",
    ws_path = "/api/v1/transcribe/stream"
)]
pub fn create_transcribe_stream_handler() -> Arc<dyn WebSocketHandler> {
    shared_handler()
}
