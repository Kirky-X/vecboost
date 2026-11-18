// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 转录门面实现（接口契约见 `mod.rs`）。

use std::path::Path;

use crate::config::model::{
    DeviceType, ModelConfig, ModelTask, TranscribeBackend, TranscriptionParams,
};
use crate::engine::transcribe::engine::{Qwen3AsrEngine, WhisperAsrEngine};

use super::{
    AnyTranscribeEngine, TRANSCRIBE_SAMPLE_RATE_HZ, TranscribeOpts, TranscribeSegment,
    TranscribeSession,
};

/// 门面后端探测：models_dir 下命中 whisper 双图候选链即按 whisper 装配，否则维持
/// 既有 Qwen3 三图路径。`open` 签名不破（main.rs 与 POC 薄壳显式构造 `TranscribeOpts`，
/// 不可加字段），故配置面的 backend 经 models_dir 缺省目录间接生效；需要显式指定
/// 后端时用 `open_with_backend`。探测只读文件系统，不改变加载行为。
fn detect_backend(models_dir: &Path) -> Result<TranscribeBackend, crate::error::VecboostError> {
    let whisper = crate::engine::transcribe::whisper_asr::resolve_graph_paths(models_dir).is_ok();
    let qwen3 = crate::engine::transcribe::asr::resolve_graph_paths(models_dir).is_ok();
    match (whisper, qwen3) {
        // 两套图共存时不再静默选 whisper：目录无法表达配置意图，显性失败
        // （审查发现的混合 bundle 歧义）；open_with_backend 显式路径不受影响。
        (true, true) => Err(crate::error::VecboostError::ConfigError(
            crate::i18n::tr("transcribe-backend-ambiguous").to_string(),
        )),
        (true, false) => Ok(TranscribeBackend::Whisper),
        // qwen3 独占或二者皆缺（皆缺由后续 Qwen3Asr::load 显性报缺图，
        // 与旧路径失败面一致）。
        (false, _) => Ok(TranscribeBackend::Qwen3Asr),
    }
}

/// 打开转录会话（签名兼容 POC 薄壳）：后端由 models_dir 内容探测决定；
/// bundle 缺失/IO 断言失败 → `ModelLoadError`（含实际路径）。
pub fn open(
    models_dir: &Path,
    opts: &TranscribeOpts,
) -> Result<TranscribeSession, crate::error::VecboostError> {
    open_with_backend(models_dir, opts, detect_backend(models_dir)?)
}

/// 显式指定后端打开会话（供需要精确选择 Qwen3 / whisper 的调用方）。
pub fn open_with_backend(
    models_dir: &Path,
    opts: &TranscribeOpts,
    backend: TranscribeBackend,
) -> Result<TranscribeSession, crate::error::VecboostError> {
    let config = ModelConfig {
        name: "transcribe-facade".to_string(),
        engine_type: crate::config::model::EngineType::Onnx,
        model_path: models_dir.to_path_buf(),
        tokenizer_path: None,
        device: if opts.cuda {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        },
        max_batch_size: 1,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        quantized: false,
        task: ModelTask::Transcribe,
        decision_params: None,
        translation_params: None,
        transcription_params: Some(TranscriptionParams {
            models_dir: Some(models_dir.to_path_buf()),
            vad_threshold: opts.vad_threshold,
            backend,
        }),
    };
    // 会话旋钮显式透传（加载期读取；0 → available_parallelism）
    let inner = match backend {
        TranscribeBackend::Qwen3Asr => AnyTranscribeEngine::Qwen3(
            Qwen3AsrEngine::load_with_session_opts(&config, opts.intra_threads, opts.no_spin)?,
        ),
        TranscribeBackend::Whisper => AnyTranscribeEngine::Whisper(
            WhisperAsrEngine::load_with_session_opts(&config, opts.intra_threads, opts.no_spin)?,
        ),
    };
    Ok(TranscribeSession { inner })
}

/// 读取 16kHz 单声道 WAV → f32 样本（[-1,1]）。 PoC 不做重采样：输入即契约。
pub fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, crate::error::VecboostError> {
    crate::engine::transcribe::wav::read_wav_16k_mono(path)
}

impl TranscribeSession {
    /// VAD-only 分段（不加载解码器 fire 路径之外的会话；每调用新建 VAD 模型）。
    pub fn detect_segments(
        &self,
        samples: &[f32],
    ) -> Result<Vec<TranscribeSegment>, crate::error::VecboostError> {
        let spans = match &self.inner {
            AnyTranscribeEngine::Qwen3(engine) => engine.detect_vad_segments(samples)?,
            AnyTranscribeEngine::Whisper(engine) => engine.detect_vad_segments(samples)?,
        };
        Ok(spans
            .into_iter()
            .map(|(start, end)| TranscribeSegment {
                start_sample: start,
                end_sample: end,
                start_sec: start as f64 / TRANSCRIBE_SAMPLE_RATE_HZ,
                end_sec: end as f64 / TRANSCRIBE_SAMPLE_RATE_HZ,
            })
            .collect())
    }

    /// 段级转录：samples 为单个语音段的样本切片（调用方已按分段切好）。
    pub fn transcribe_segment(
        &self,
        samples: &[f32],
    ) -> Result<String, crate::error::VecboostError> {
        match &self.inner {
            AnyTranscribeEngine::Qwen3(engine) => engine.transcribe_segment(samples),
            AnyTranscribeEngine::Whisper(engine) => engine.transcribe_segment(samples),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_missing_bundle_is_explicit_error() {
        let dir = std::env::temp_dir().join("vecboost-transcribe-missing-bundle");
        let _ = std::fs::remove_dir_all(&dir);
        let err = open(&dir, &TranscribeOpts::default())
            .expect_err("缺失 bundle 必须显性 ModelLoadError");
        assert!(
            matches!(err, crate::error::VecboostError::ModelLoadError(_)),
            "err={err:?}"
        );
    }

    #[test]
    fn read_wav_missing_file_is_explicit_error() {
        let p = std::env::temp_dir().join("vecboost-transcribe-no-such.wav");
        let _ = std::fs::remove_file(&p);
        let err = read_wav_16k_mono(&p).expect_err("缺失 wav 必须显性报错");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn opts_default_matches_poc_baseline() {
        let o = TranscribeOpts::default();
        assert_eq!(o.vad_threshold, 0.5);
        assert_eq!(o.intra_threads, 0);
        assert!(!o.no_spin);
        assert!(!o.cuda);
    }
}
