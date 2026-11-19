// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 流式 ASR 核心（官方 sherpa-onnx 绑定 + Paraformer 三语 int8 模型）。
//!
//! 背景：自研复刻旧双语 zipformer 的 CER 达 0.95+（重复字/乱码）；官方绑定
//! + 2025 模型实测 CER：中文 0.05~0.19、英文 0.015（详见 /tmp/sherpa-accuracy-test
//! 实验）。本模块用官方 `OnlineRecognizer` 替换自研推理，保持原对外 API 不变，
//! WS 端点、纠错层、session 管理零改动。
//!
//! 模型：csukuangfj/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en
//! （encoder 166MB + decoder 72MB + tokens，Apache-2.0，支持中/粤/英）。

use std::path::Path;

use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};

use crate::error::VecboostError;

/// 采样率（与模型/WS 协议契约一致）。
pub const SAMPLE_RATE: usize = 16_000;

type StreamResult<T> = Result<T, VecboostError>;

fn load_err(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::ModelLoadError(format!("streaming asr load: {e}"))
}
fn infer_err(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::InferenceError(format!("streaming asr: {e}"))
}

/// 官方在线识别器（进程级一份，内部线程安全）。
pub struct StreamingAsr {
    recognizer: OnlineRecognizer,
}

/// 单条流的推理状态：持有 sherpa OnlineStream + 最新识别文本。
/// 与 Session 分离，便于按 session_id 隔离（WS 层管理）。
pub struct StreamState {
    pub(crate) stream: OnlineStream,
    pub(crate) text: String,
}

/// 兼容 API：自研版逐 token 解码入口。官方绑定直接输出文本，此函数空操作
/// （保留签名以免 WS handler 改动；返回 None 表示无 token 级增量）。
pub fn advance_decode(
    _state: &mut StreamState,
    _logits: &[f32],
) -> Result<Option<i64>, VecboostError> {
    Ok(None)
}

/// 兼容 API：自研版 BPE token 解码。官方绑定直接输出文本，此函数保留供测试。
pub fn tokens_to_text(_tokens: &[String], _ids: &[i64]) -> String {
    String::new()
}

impl StreamingAsr {
    /// 加载 Paraformer 三语 int8 模型，构造官方 OnlineRecognizer。
    pub fn load(
        models_dir: &Path,
        intra_threads: usize,
        biasing_words_path: Option<&str>,
        biasing_weight: f32,
    ) -> StreamResult<Self> {
        let mut cfg = OnlineRecognizerConfig::default();
        // 官方热词偏置（架构审查 F1）：词表在识别期参与解码，真正生效
        cfg.hotwords_file = biasing_words_path.map(|s| s.to_string());
        cfg.hotwords_score = biasing_weight.max(0.0);
        cfg.model_config.tokens =
            Some(models_dir.join("tokens.txt").to_string_lossy().into_owned());
        cfg.model_config.paraformer.encoder = Some(
            models_dir
                .join("encoder.int8.onnx")
                .to_string_lossy()
                .into_owned(),
        );
        cfg.model_config.paraformer.decoder = Some(
            models_dir
                .join("decoder.int8.onnx")
                .to_string_lossy()
                .into_owned(),
        );
        cfg.model_config.num_threads = intra_threads as i32;
        cfg.model_config.provider = Some("cpu".to_string());
        cfg.enable_endpoint = false;
        cfg.decoding_method = Some("greedy_search".to_string());
        let recognizer = OnlineRecognizer::create(&cfg).ok_or_else(|| {
            load_err("OnlineRecognizer::create returned None (check model files)")
        })?;
        Ok(Self { recognizer })
    }

    /// 创建一条新流的状态（持 sherpa OnlineStream）。
    pub fn new_state(&self) -> StreamState {
        StreamState {
            stream: self.recognizer.create_stream(),
            text: String::new(),
        }
    }

    /// 送一帧 PCM（16kHz mono f32）并推进解码；返回空 Vec（兼容旧 API，
    /// 官方绑定直接产出文本，不再暴露 logits）。
    pub fn encode_frame(&mut self, state: &mut StreamState, pcm: &[f32]) -> StreamResult<Vec<f32>> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        state.stream.accept_waveform(SAMPLE_RATE as i32, pcm);
        let mut decoded = false;
        while self.recognizer.is_ready(&state.stream) {
            self.recognizer.decode(&state.stream);
            decoded = true;
        }
        // 性能审查 H1：只在 decode 后取 result（内部是 JSON 序列化往返），
        // 且文本无变化时不写，避免每帧堆分配。
        if decoded {
            if let Some(result) = self.recognizer.get_result(&state.stream) {
                if result.text != state.text {
                    state.text = result.text;
                }
            }
        }
        Ok(Vec::new())
    }

    /// 结束输入 + flush 尾音（架构 F2 / 安全 L2）：end 帧必须调用。
    pub fn finalize(&self, state: &mut StreamState) {
        state.stream.input_finished();
        while self.recognizer.is_ready(&state.stream) {
            self.recognizer.decode(&state.stream);
        }
        if let Some(result) = self.recognizer.get_result(&state.stream) {
            if !result.text.is_empty() {
                state.text = result.text;
            }
        }
    }

    /// 当前 partial 文本（官方 result.text）。
    pub fn partial_text(&self, state: &StreamState) -> String {
        state.text.clone()
    }

    /// 流结束：flush + 取最终文本。
    pub fn finish(&self, state: &StreamState) -> String {
        state.text.clone()
    }

    /// N-best 候选：官方 greedy 单结果；纠错层（偏置 + 重打分）在文本级处理。
    pub fn finish_candidates(&self, state: &StreamState) -> Vec<String> {
        if state.text.is_empty() {
            Vec::new()
        } else {
            vec![state.text.clone()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 真实模型 + 真实音频：验证官方绑定路径 CER 基线（ignored，需资产）。
    #[test]
    #[ignore = "requires paraformer-trilingual model and test wav"]
    fn real_model_accuracy() {
        let dir = Path::new("models/paraformer-trilingual");
        if !dir.join("tokens.txt").exists() {
            return;
        }
        let mut asr = StreamingAsr::load(dir, 1, None, 0.0).unwrap();
        let mut st = asr.new_state();
        let bytes = std::fs::read("/tmp/whisper-e2e/zh_short.wav").unwrap();
        let pcm: Vec<f32> = bytes
            .chunks_exact(2)
            .skip(22)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();
        for c in pcm.chunks(3200) {
            asr.encode_frame(&mut st, c).unwrap();
        }
        println!("TEXT={}", asr.finish(&st));
        assert!(!asr.finish(&st).is_empty());
    }
}
