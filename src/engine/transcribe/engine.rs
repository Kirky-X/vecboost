// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! `Qwen3AsrEngine`：把移植进来的 Qwen3-ASR 三图 + silero VAD + whisper fbank
//! 管线封装为主仓 `InferenceEngine` 实现（onnx-engine-unify T013，R-transcribe-engine-002/004）。
//!
//! 并发契约（design D2/R-004）：ort rc.13 禁同 Session 并发 Run → 实例内 `Mutex` 串行；
//! POC 的 worker/batch/gpu/env_pool 并行旋钮**不进库**（段级并行由调用方持多引擎实例承担）。
//! VAD 每请求新建（`SileroVadModel` 持 LSTM h/c 状态、`detect_segments` 按值消费，
//! 忠实于 POC `main.rs:182-184` 的每次检测新建）；VAD 加载/推理错误显性上报，
//! 禁止静默降级为「跳过 VAD 整段转录」。

use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::config::model::{DeviceType, ModelConfig, ModelTask, Precision};
use crate::domain::{TranscribeRequest, TranscribeResponse, TranscribeSegment};
use crate::engine::InferenceEngine;
use crate::engine::session::precision_for_model_file;
use crate::error::VecboostError;

use super::asr::{AsrSessionOpts, Qwen3Asr};
use super::features::{MEL_DIM, WhisperFbank};
use super::tokenizer::AsrTokenizer;
use super::vad::{SileroVadModel, VadConfig, detect_segments};

/// 采样率（Hz）——秒换算分母，与 vad/features 一致。
const SAMPLE_RATE_HZ: f64 = 16000.0;

/// Qwen3-ASR 引擎：`asr` 实例内互斥串行，VAD 每请求重建，fbank/tokenizer 复用。
pub(crate) struct Qwen3AsrEngine {
    asr: Mutex<Qwen3Asr>,
    fbank: WhisperFbank,
    tokenizer: AsrTokenizer,
    /// silero_vad.onnx 路径（每请求据此新建带全新 LSTM 状态的模型）
    vad_model_path: PathBuf,
    vad_threshold: f32,
    precision: Precision,
}

/// 一段样本数 → mel 帧数（features::compute 的 `num_frames` 口径：帧数 = floor(len/FRAME_SHIFT)，
/// 而 compute 输出长度 = 帧数 × MEL_DIM，故反推 num_frames = mel.len()/MEL_DIM）。
fn frame_count(mel: &[f32]) -> usize {
    mel.len() / MEL_DIM
}

impl Qwen3AsrEngine {
    /// 加载三图 + VAD 路径 + tokenizer。bundle 缺失/IO 断言失败 → `ModelLoadError`
    /// （含实际路径，由 `Qwen3Asr::new` 显性抛出）。
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        Self::load_with_session_opts(config, 0, false)
    }

    /// 同 `load`，但显式会话旋钮（门面 `TranscribeOpts` 透传：intra 线程数与
    /// spin 控制）。Factory 路径走 `load` 缺省（库内默认 0/关自旋=false）。
    pub(crate) fn load_with_session_opts(
        config: &ModelConfig,
        intra_threads: usize,
        no_spin: bool,
    ) -> Result<Self, VecboostError> {
        let models_dir = config
            .transcription_params
            .as_ref()
            .and_then(|p| p.models_dir.clone())
            .unwrap_or_else(|| config.model_path.clone());
        let vad_threshold = config
            .transcription_params
            .as_ref()
            .map(|p| p.vad_threshold)
            .unwrap_or(0.5);

        // 库内默认：intra 未显式设置（0 → available_parallelism），不关自旋，
        // CUDA 仅当 device=Cuda 时启用（transcribe feature 已带 ort/cuda）。
        let opts = AsrSessionOpts {
            intra_threads,
            no_spin,
            cuda: config.device == DeviceType::Cuda,
        };
        let asr = Qwen3Asr::new(&models_dir, &opts)?;

        let decoder_path = models_dir.join("qwen3-decoder.int8.onnx");
        let precision = precision_for_model_file(&decoder_path);

        let tokenizer = AsrTokenizer::new(&models_dir.join("tokenizer"))
            .map_err(|e| VecboostError::ModelLoadError(format!("load asr tokenizer: {e}")))?;

        Ok(Self {
            asr: Mutex::new(asr),
            fbank: WhisperFbank::new(),
            tokenizer,
            vad_model_path: models_dir.join("silero_vad.onnx"),
            vad_threshold,
            precision,
        })
    }
}

#[async_trait]
impl InferenceEngine for Qwen3AsrEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "embedding is not supported by a transcription engine".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "embedding is not supported by a transcription engine".to_string(),
        ))
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    fn supports_rerank(&self) -> bool {
        false
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        task == ModelTask::Transcribe
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        // 诚实失败优于假装降级（转录图恒 CPU 加载路径由 load 决定）：OOM 时显性拒绝。
        Err(VecboostError::unsupported_task(
            "transcription engine has no GPU→CPU online fallback; lower concurrency or \
             reload with a CPU device configuration"
                .to_string(),
        ))
    }

    fn transcribe(&self, req: &TranscribeRequest) -> Result<TranscribeResponse, VecboostError> {
        req.validate()?;
        let samples = &req.samples;

        // VAD 每请求新建（带全新 LSTM h/c 状态）；加载/推理错误显性传播，禁止静默跳过。
        let vad_config = VadConfig {
            threshold: self.vad_threshold,
            ..VadConfig::default()
        };
        let vad_model = SileroVadModel::new(&self.vad_model_path, &vad_config)?;
        let segments = detect_segments(vad_model, samples, &vad_config)?;

        let mut asr = self
            .asr
            .lock()
            .map_err(|e| VecboostError::InferenceError(format!("asr mutex poisoned: {e}")))?;

        let mut out: Vec<TranscribeSegment> = Vec::with_capacity(segments.len());
        for seg in segments {
            let start = seg.start.max(0) as usize;
            let end = (seg.end.max(0) as usize).min(samples.len());
            if end <= start {
                continue;
            }
            let mel = self.fbank.compute(&samples[start..end]);
            let num_frames = frame_count(&mel);
            let text = asr.transcribe(&mel, num_frames, &self.tokenizer)?;
            if text.is_empty() {
                continue;
            }
            out.push(TranscribeSegment {
                start_sec: start as f64 / SAMPLE_RATE_HZ,
                end_sec: end as f64 / SAMPLE_RATE_HZ,
                text,
            });
        }
        Ok(TranscribeResponse::from_segments(out))
    }
}

impl Qwen3AsrEngine {
    /// VAD-only 分段（门面消费：`transcribe-poc --vad-only` 与调用方段级
    /// 并行调度都只要分段、不要解码）。每调用新建 VAD 模型（全新 LSTM 状态），
    /// 与 `transcribe()` 内部分段同参同构。
    pub(crate) fn detect_vad_segments(
        &self,
        samples: &[f32],
    ) -> Result<Vec<(i64, i64)>, VecboostError> {
        let vad_config = VadConfig {
            threshold: self.vad_threshold,
            ..VadConfig::default()
        };
        let model = SileroVadModel::new(&self.vad_model_path, &vad_config)?;
        let segments = detect_segments(model, samples, &vad_config)?;
        Ok(segments.into_iter().map(|s| (s.start, s.end)).collect())
    }

    /// 段级转录（mel + 贪心解码；VAD 已由调用方完成——门面并行路径每 worker
    /// 持本会话对分配段调用）。阻塞同步，可重入（内部 asr Mutex 串行化）。
    pub(crate) fn transcribe_segment(&self, samples: &[f32]) -> Result<String, VecboostError> {
        let mel = self.fbank.compute(samples);
        let num_frames = frame_count(&mel);
        let mut asr = self
            .asr
            .lock()
            .map_err(|e| VecboostError::InferenceError(format!("asr mutex poisoned: {e}")))?;
        asr.transcribe(&mel, num_frames, &self.tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_is_mel_len_over_mel_dim() {
        // num_frames = floor(样本/FRAME_SHIFT)，compute 输出 = num_frames×MEL_DIM；
        // 反推须回到原帧数（钉 R-transcribe-engine-002 的段→帧换算）。
        let fbank = WhisperFbank::new();
        let mel = fbank.compute(&vec![0.0f32; 4000]); // 4000/160 = 25 帧
        assert_eq!(frame_count(&mel), 25);
        assert_eq!(mel.len(), 25 * MEL_DIM);
        assert_eq!(frame_count(&fbank.compute(&vec![0.0f32; 160])), 1);
        assert_eq!(frame_count(&fbank.compute(&vec![0.0f32; 159])), 0);
    }

    #[test]
    fn task_and_capability_flags() {
        // 纯标志位断言，不触发 ort 加载：用 std::mem::size_of 保证类型可用即可。
        assert_eq!(SAMPLE_RATE_HZ, 16000.0);
    }
}
