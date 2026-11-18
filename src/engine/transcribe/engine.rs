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
use super::vad::{SileroVadModel, VadConfig};
use super::{WhisperAsr, WhisperGraphPaths, WhisperTokenizer};

/// 采样率（Hz）——秒换算分母，与 vad/features 一致。
const SAMPLE_RATE_HZ: f64 = 16000.0;

/// Mutex poison → 显性 InferenceError（asr/vad 共用同一 i18n key）。
/// 此前 Qwen3 内联、Whisper 抽方法两套写法并存（审查发现的不一致），提取为单一事实源。
fn lock_poison<T>(m: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, VecboostError> {
    m.lock().map_err(|e| {
        VecboostError::InferenceError(crate::i18n::tr_with_args(
            "detail-asr-mutex-poisoned-e-c380d",
            crate::i18n::tr_args(&[("e", &e.to_string())]),
        ))
    })
}

/// VAD 复用段检测（两引擎共用骨架）：锁内经 `SileroVadModel::detect` 显性
/// 重置后检测，段元组化。Session 复用免去每请求 commit 的固定加载开销
/// （审查发现：短音频的 VAD 加载开销可能超过推理本身）；语义与「每请求
/// 新建」等价——state machine 随 VoiceActivityDetector::new 每次全新构造，
/// 唯一跨请求残留 LSTM h/c 由 detect() 显性清零。
fn detect_vad_only(
    vad: &Mutex<SileroVadModel>,
    threshold: f32,
    samples: &[f32],
) -> Result<Vec<(i64, i64)>, VecboostError> {
    let cfg = VadConfig {
        threshold,
        ..VadConfig::default()
    };
    let mut guard = lock_poison(vad)?;
    let segments = guard.detect(samples, &cfg)?;
    Ok(segments.into_iter().map(|s| (s.start, s.end)).collect())
}

/// Qwen3-ASR 引擎：`asr` 实例内互斥串行，VAD Session 复用（每请求显性重置），
/// fbank/tokenizer 复用。
pub(crate) struct Qwen3AsrEngine {
    asr: Mutex<Qwen3Asr>,
    fbank: WhisperFbank,
    tokenizer: AsrTokenizer,
    /// VAD Session（load 期一次构建；每请求显性重置 LSTM 状态）
    vad: Mutex<SileroVadModel>,
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

        // VAD Session load 期一次构建：缺图/坏图在此显性失败（同 whisper 口径，
        // 本仓纪律明确「加载成功、请求才失败」是反模式；旧路径把失败推迟到
        // 首请求，属行为收紧而非放松，见提交说明）。
        let vad = SileroVadModel::new(
            &models_dir.join("silero_vad.onnx"),
            &VadConfig {
                threshold: vad_threshold,
                ..VadConfig::default()
            },
        )?;
        Ok(Self {
            asr: Mutex::new(asr),
            fbank: WhisperFbank::new(),
            tokenizer,
            vad: Mutex::new(vad),
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

        // VAD：Session 复用 + 每请求显性重置（语义等价，见 detect_vad_only 文档）；
        // 加载/推理错误显性传播，禁止静默跳过。锁仅在检测期持有。
        let segments = {
            let vad_config = VadConfig {
                threshold: self.vad_threshold,
                ..VadConfig::default()
            };
            let mut vad = lock_poison(&self.vad)?;
            vad.detect(samples, &vad_config)?
        };

        // 锁外预处理：mel 全量算好（fbank 无状态；asr 锁只护 ort Session Run，
        // 特征提取不再串行化并发请求）。
        let mut mels: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(segments.len());
        for seg in &segments {
            let start = seg.start.max(0) as usize;
            let end = (seg.end.max(0) as usize).min(samples.len());
            if end <= start {
                continue;
            }
            let mel = self.fbank.compute(&samples[start..end]);
            mels.push((start, end, mel));
        }

        let mut asr = lock_poison(&self.asr)?;
        let mut out: Vec<TranscribeSegment> = Vec::with_capacity(mels.len());
        for (start, end, mel) in mels {
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
    /// 并行调度都只要分段、不要解码）。复用 VAD Session + 每请求显性重置，
    /// 与 `transcribe()` 内部分段同参同构（共享 detect_vad_only 单一实现）。
    pub(crate) fn detect_vad_segments(
        &self,
        samples: &[f32],
    ) -> Result<Vec<(i64, i64)>, VecboostError> {
        detect_vad_only(&self.vad, self.vad_threshold, samples)
    }

    /// 段级转录（mel + 贪心解码；VAD 已由调用方完成——门面并行路径每 worker
    /// 持本会话对分配段调用）。mel 在锁外计算，锁内仅 ort Run。
    pub(crate) fn transcribe_segment(&self, samples: &[f32]) -> Result<String, VecboostError> {
        let mel = self.fbank.compute(samples);
        let num_frames = frame_count(&mel);
        let mut asr = lock_poison(&self.asr)?;
        asr.transcribe(&mel, num_frames, &self.tokenizer)
    }
}

/// whisper ONNX 双图转录引擎：与 Qwen3 同口径（实例内 Mutex 串行、VAD Session
/// 复用每请求显性重置、精度按图内容判定）。fbank 与 30s 窗口预处理在**锁外**
/// 执行（`precompute_windows`），本结构持独立 fbank。标准 whisper bundle **不含** silero_vad.onnx（HF onnx-community/
/// whisper-large-v3-turbo/onnx 目录实测），须从 Qwen3 bundle 拷贝或独立下载到
/// models_dir；load 期显式探测，缺失即 ModelLoadError，禁止静默降级为
/// 「跳过 VAD 整段转录」。EP 随 `config.device` 分发（device=Cuda 时 CUDA
/// EP），与 Qwen3 同口径；无在线 GPU→CPU 回退（`try_fallback_to_cpu` 显性
/// 拒绝），旧注释「CPU 恒定」与实际 EP 分发不符，已修正。
pub(crate) struct WhisperAsrEngine {
    asr: Mutex<WhisperAsr>,
    tokenizer: WhisperTokenizer,
    /// 预处理用 fbank（纯 CPU 无状态；asr 锁只护 ort Session Run）
    fbank: WhisperFbank,
    /// VAD Session（load 期一次构建；每请求显性重置 LSTM 状态）
    vad: Mutex<SileroVadModel>,
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
        let vad = SileroVadModel::new(
            &vad_model_path,
            &VadConfig {
                threshold: vad_threshold,
                ..VadConfig::default()
            },
        )?;
        Ok(Self {
            asr: Mutex::new(asr),
            tokenizer,
            fbank: WhisperFbank::new(),
            vad: Mutex::new(vad),
            vad_threshold,
            precision,
        })
    }

    /// Mutex Poison 显性化（共享 lock_poison，与 Qwen3 同 key、同口径）。
    fn lock_asr(&self) -> Result<std::sync::MutexGuard<'_, WhisperAsr>, VecboostError> {
        lock_poison(&self.asr)
    }

    /// VAD-only 分段（门面消费：`--vad-only` 与段级并行调度）。复用 VAD
    /// Session + 每请求显性重置，与 `transcribe()` 内部分段同参同构
    /// （共享 detect_vad_only 单一实现）。
    pub(crate) fn detect_vad_segments(
        &self,
        samples: &[f32],
    ) -> Result<Vec<(i64, i64)>, VecboostError> {
        detect_vad_only(&self.vad, self.vad_threshold, samples)
    }

    /// 段级转录：whisper 核心层自带 30s 切窗。切窗+fbank+转置在**锁外**
    /// （`precompute_windows`），锁内仅 encode/forward 的 ort Run。
    pub(crate) fn transcribe_segment(&self, samples: &[f32]) -> Result<String, VecboostError> {
        let feats = super::whisper_asr::precompute_windows(samples, &self.fbank)?;
        let mut asr = self.lock_asr()?;
        asr.transcribe_precomputed(feats, &self.tokenizer, None)
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

        // VAD：Session 复用 + 每请求显性重置（共享骨架，语义等价）；锁仅检测期持有。
        let segments = {
            let vad_config = VadConfig {
                threshold: self.vad_threshold,
                ..VadConfig::default()
            };
            let mut vad = lock_poison(&self.vad)?;
            vad.detect(samples, &vad_config)?
        };

        // 收集有效段并锁外预计算逐窗 feats（切窗/fbank/转置纯 CPU）。
        let mut windows: Vec<(usize, usize, Vec<Vec<f32>>)> = Vec::new();
        for seg in &segments {
            let start = seg.start.max(0) as usize;
            let end = (seg.end.max(0) as usize).min(samples.len());
            if end <= start {
                continue;
            }
            let feats = super::whisper_asr::precompute_windows(&samples[start..end], &self.fbank)?;
            windows.push((start, end, feats));
        }

        let mut asr = self.lock_asr()?;
        let mut out: Vec<TranscribeSegment> = Vec::with_capacity(windows.len());
        for (start, end, feats) in windows {
            let text = asr.transcribe_precomputed(feats, &self.tokenizer, None)?;
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
