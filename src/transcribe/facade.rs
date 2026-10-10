// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 转录门面实现（接口契约见 `mod.rs`）。

use std::path::Path;

use crate::config::model::{DeviceType, ModelConfig, ModelTask, TranscriptionParams};
use crate::engine::transcribe::engine::Qwen3AsrEngine;

use super::{TRANSCRIBE_SAMPLE_RATE_HZ, TranscribeOpts, TranscribeSegment, TranscribeSession};

/// 打开转录会话：models_dir 指向 bundle 根（三图 + silero_vad.onnx +
/// tokenizer/）。bundle 缺失/IO 断言失败 → `ModelLoadError`（含实际路径）。
pub fn open(
    models_dir: &Path,
    opts: &TranscribeOpts,
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
        }),
    };
    // 会话旋钮显式透传（加载期读取；0 → available_parallelism）
    let engine = Qwen3AsrEngine::load_with_session_opts(&config, opts.intra_threads, opts.no_spin)?;
    Ok(TranscribeSession { inner: engine })
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
        let spans = self.inner.detect_vad_segments(samples)?;
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
        self.inner.transcribe_segment(samples)
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
