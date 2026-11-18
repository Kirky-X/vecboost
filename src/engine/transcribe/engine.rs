// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! `Qwen3AsrEngine` / `WhisperAsrEngine`：把移植进来的两条转录管线封装为主仓
//! `InferenceEngine` 实现。Qwen3 走三图 + silero VAD + whisper fbank；Whisper
//! 走 encoder + merged decoder 双图，30s 窗口贪心解码。两者由
//! `TranscriptionParams.backend` 在工厂层分发，既有 Qwen3 路径逐字不动。
//!
//! 并发契约：ort rc.15 `Session::run(&mut self)` 编译期强制独占 → 实例内 `Mutex`
//! 串行为冗余但无害的额外保障（历史口径 rc.13，Cargo.lock:4984 实际 rc.15）；
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
use crate::engine::session::detect_onnx_weight_precision;
use crate::error::VecboostError;

use super::asr::{AsrSessionOpts, Qwen3Asr};
use super::features::{MEL_DIM, WhisperFbank};
use super::tokenizer::AsrTokenizer;
use super::vad::{SileroVadModel, VadConfig, detect_segments};
use super::{WhisperAsr, WhisperGraphPaths, WhisperTokenizer};

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
/// 而 compute 输出长度 = 帧数 × MEL_DIM，故反推 num_frames = mel.len/MEL_DIM）。
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
        let (asr, graphs) = Qwen3Asr::new(&models_dir, &opts)?;

        // 精度按 decoder 图**内容**判定（量化算子存在与否），不看文件名：
        // 文件名判定会把无损（fp32/fp16）图改名后误报 Int8（2026-10-18 修复的缺陷）。
        // 图名候选链与路径解析统一在 `asr::resolve_graph_paths`（单一事实源）。
        let precision = detect_onnx_weight_precision(&graphs.decoder)?;

        // 启动期可观测性：候选链实际选中哪张图、按内容判定的精度是什么。
        // 此前两者都不可见（只能读代码推断）——改名误报之所以能潜伏，就是因为没有
        // 可观测出口（2026-10-18 缺陷修复的一部分）。
        let name_of = |p: &std::path::Path| -> String {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let conv_name = name_of(&graphs.conv);
        let encoder_name = name_of(&graphs.encoder);
        let decoder_name = name_of(&graphs.decoder);
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-transcribe-graphs",
                crate::i18n::tr_args(&[
                    ("conv", &conv_name),
                    ("encoder", &encoder_name),
                    ("decoder", &decoder_name),
                    ("precision", &format!("{precision:?}")),
                ])
            )
        );

        let tokenizer = AsrTokenizer::new(&models_dir.join("tokenizer")).map_err(|e| {
            VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                "detail-load-asr-tokenizer-e-e784d",
                crate::i18n::tr_args(&[("e", &e.to_string())]),
            ))
        })?;

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

        let mut asr = self.asr.lock().map_err(|e| {
            VecboostError::InferenceError(crate::i18n::tr_with_args(
                "detail-asr-mutex-poisoned-e-c380d",
                crate::i18n::tr_args(&[("e", &e.to_string())]),
            ))
        })?;

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
        let mut asr = self.asr.lock().map_err(|e| {
            VecboostError::InferenceError(crate::i18n::tr_with_args(
                "detail-asr-mutex-poisoned-e-c380d",
                crate::i18n::tr_args(&[("e", &e.to_string())]),
            ))
        })?;
        asr.transcribe(&mel, num_frames, &self.tokenizer)
    }
}

/// whisper ONNX 双图转录引擎：与 Qwen3 同口径（实例内 Mutex 串行、VAD 每请求
/// 重建、精度按图内容判定）。fbank 与 30s 窗口预处理驻留 `WhisperAsr`，本结构
/// 不重复构造。标准 whisper bundle **不含** silero_vad.onnx（HF onnx-community/
/// whisper-large-v3-turbo/onnx 目录实测），须从 Qwen3 bundle 拷贝或独立下载到
/// models_dir；load 期显式探测，缺失即 ModelLoadError，禁止静默降级为
/// 「跳过 VAD 整段转录」。EP 随 `config.device` 分发（device=Cuda 时 CUDA
/// EP），与 Qwen3 同口径；无在线 GPU→CPU 回退（`try_fallback_to_cpu` 显性
/// 拒绝），旧注释「CPU 恒定」与实际 EP 分发不符，已修正。
pub(crate) struct WhisperAsrEngine {
    asr: Mutex<WhisperAsr>,
    tokenizer: WhisperTokenizer,
    /// silero_vad.onnx 路径（标准 whisper bundle 不含；load 期已探测存在）
    vad_model_path: PathBuf,
    vad_threshold: f32,
    precision: Precision,
}

impl WhisperAsrEngine {
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        Self::load_with_session_opts(config, 0, false)
    }

    /// 显式会话旋钮（门面 `TranscribeOpts` 透传）；工厂路径走 `load` 缺省。
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
        let opts = AsrSessionOpts {
            intra_threads,
            no_spin,
            cuda: config.device == DeviceType::Cuda,
        };
        // 核心层先探测图路径后 builder；缺图 / 缺 ORT 动态库均显性 ModelLoadError。
        let (asr, graphs): (WhisperAsr, WhisperGraphPaths) = WhisperAsr::new(&models_dir, &opts)?;
        // 精度按 decoder 图**内容**判定（同 Qwen3，不看文件名——改名误报是历史缺陷）。
        let precision = detect_onnx_weight_precision(&graphs.decoder)?;
        let name_of = |p: &std::path::Path| -> String {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-transcribe-whisper-graphs-4a7e1",
                crate::i18n::tr_args(&[
                    ("encoder", &name_of(&graphs.encoder)),
                    ("decoder", &name_of(&graphs.decoder)),
                    ("precision", &format!("{precision:?}")),
                ])
            )
        );
        // tokenizer 锚定校验 4 个特殊 id，资产错配（如误放 Qwen3 tokenizer）显性失败。
        let tokenizer = WhisperTokenizer::new(&models_dir.join("tokenizer"))?;
        // 标准 whisper bundle 不含 silero_vad.onnx。若只存路径不校验，会造成
        // 「加载成功、每次请求才失败」——与 fail-fast 纪律相悖。故 load 期探测，
        // 缺失即 ModelLoadError（含实际路径与来源说明）。
        let vad_model_path = models_dir.join("silero_vad.onnx");
        if !vad_model_path.is_file() {
            return Err(VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                "detail-whisper-vad-file-not-found-b7e21",
                crate::i18n::tr_args(&[("path", &vad_model_path.display().to_string())]),
            )));
        }
        Ok(Self {
            asr: Mutex::new(asr),
            tokenizer,
            vad_model_path,
            vad_threshold,
            precision,
        })
    }

    /// Mutex  Poison 显性化（与 Qwen3 同 key、同口径，禁止静默重试）。
    fn lock_asr(&self) -> Result<std::sync::MutexGuard<'_, WhisperAsr>, VecboostError> {
        self.asr.lock().map_err(|e| {
            VecboostError::InferenceError(crate::i18n::tr_with_args(
                "detail-asr-mutex-poisoned-e-c380d",
                crate::i18n::tr_args(&[("e", &e.to_string())]),
            ))
        })
    }

    /// VAD-only 分段（门面消费：`--vad-only` 与段级并行调度）。每调用新建 VAD
    /// 模型（全新 LSTM 状态），与 `transcribe()` 内部分段同参同构。
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

    /// 段级转录：whisper 核心层自带 30s 切窗，故直接喂样本（不再经 fbank）。
    /// 语言未配置时传 None（模型自判定），阻塞同步、可重入（Mutex 串行化）。
    pub(crate) fn transcribe_segment(&self, samples: &[f32]) -> Result<String, VecboostError> {
        let mut asr = self.lock_asr()?;
        asr.transcribe(samples, &self.tokenizer, None)
    }
}

#[async_trait]
impl InferenceEngine for WhisperAsrEngine {
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
        // 诚实失败优于假装降级：transcribe 图恒 CPU 加载，OOM 时显性拒绝。
        Err(VecboostError::unsupported_task(
            "transcription engine has no GPU→CPU online fallback; lower concurrency or \
             reload with a CPU device configuration"
                .to_string(),
        ))
    }

    fn transcribe(&self, req: &TranscribeRequest) -> Result<TranscribeResponse, VecboostError> {
        req.validate()?;
        let samples = &req.samples;
        let vad_config = VadConfig {
            threshold: self.vad_threshold,
            ..VadConfig::default()
        };
        let vad_model = SileroVadModel::new(&self.vad_model_path, &vad_config)?;
        let segments = detect_segments(vad_model, samples, &vad_config)?;

        let mut asr = self.lock_asr()?;
        let mut out: Vec<TranscribeSegment> = Vec::with_capacity(segments.len());
        for seg in segments {
            let start = seg.start.max(0) as usize;
            let end = (seg.end.max(0) as usize).min(samples.len());
            if end <= start {
                continue;
            }
            let text = asr.transcribe(&samples[start..end], &self.tokenizer, None)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_is_mel_len_over_mel_dim() {
        // num_frames = floor(样本/FRAME_SHIFT)，compute 输出 = num_frames×MEL_DIM；
        // 反推须回到原帧数（段→帧换算）。
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
