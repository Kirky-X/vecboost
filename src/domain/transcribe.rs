// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 转录领域类型（R-transcribe-engine-001）。
//!
//! 输入为 16kHz 单声道 f32 `[-1,1]` 样本（`read_wav_16k_mono` 的产物）；
//! `validate` 以域层 `ValidationError` 显性拒绝非法输入（非空、全部有限、
//! 长度 ≤ 2h×16000），不做静默截断。响应 `text` 为 `segments` 文本按序拼接。

use serde::{Deserialize, Serialize};

use crate::error::VecboostError;

/// 采样率（Hz）与最长可接受音频时长（小时）——长度上限 = 2h×16000。
pub const TRANSCRIBE_SAMPLE_RATE: usize = 16000;
pub const TRANSCRIBE_MAX_SECONDS: usize = 2 * 60 * 60;
pub const TRANSCRIBE_MAX_SAMPLES: usize = TRANSCRIBE_SAMPLE_RATE * TRANSCRIBE_MAX_SECONDS;

/// 转录请求：16kHz 单声道 f32 样本（取值 `[-1,1]`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct TranscribeRequest {
    pub samples: Vec<f32>,
}

impl std::str::FromStr for TranscribeRequest {
    /// # 红线（cli/grpc forge 落地时）
    /// `Err = serde_json::Error` 携带内部结构名与字节偏移，handler 必须
    /// 映射为通用 `VecboostError::validation_error` 文案，禁止原样透传客户端
    /// （translate 同款红线，src/domain/translate.rs:29-37）。
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl TranscribeRequest {
    /// 输入校验：非空、全部有限（拒绝 NaN/±Inf）、长度 ≤ `TRANSCRIBE_MAX_SAMPLES`。
    /// 违规显性返回（域层 `ValidationError` 语义），禁止静默截断或忽略。
    pub fn validate(&self) -> Result<(), VecboostError> {
        if self.samples.is_empty() {
            return Err(VecboostError::validation_error(
                "transcribe samples must not be empty".to_string(),
            ));
        }
        if self.samples.len() > TRANSCRIBE_MAX_SAMPLES {
            return Err(VecboostError::validation_error(format!(
                "transcribe samples too long: {} > {TRANSCRIBE_MAX_SAMPLES}",
                self.samples.len()
            )));
        }
        if self.samples.iter().any(|v| !v.is_finite()) {
            return Err(VecboostError::validation_error(
                "transcribe samples contain non-finite value (NaN/Inf)".to_string(),
            ));
        }
        Ok(())
    }
}

/// 单个语音段：起止秒 + 段文本。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct TranscribeSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

/// 转录响应：`text` 为 `segments` 文本按序拼接；全静音输入 segments 为空、text 为空。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct TranscribeResponse {
    pub text: String,
    pub segments: Vec<TranscribeSegment>,
}

impl TranscribeResponse {
    /// 由段列表构造响应，`text` 强制等于各段文本按序拼接（域层不变式）。
    pub fn from_segments(segments: Vec<TranscribeSegment>) -> Self {
        let text = segments.iter().map(|s| s.text.as_str()).collect::<String>();
        Self { text, segments }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_empty() {
        let req = TranscribeRequest { samples: vec![] };
        assert!(req.validate().is_err());
    }

    #[test]
    fn validate_rejects_nan_and_inf() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let req = TranscribeRequest {
                samples: vec![0.1, bad, 0.2],
            };
            let err = req.validate().unwrap_err();
            assert!(
                err.error_detail().contains("non-finite"),
                "应拒绝非有限值 {bad}，got {err:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_overlong() {
        let req = TranscribeRequest {
            samples: vec![0.0; TRANSCRIBE_MAX_SAMPLES + 1],
        };
        let err = req.validate().unwrap_err();
        assert!(err.error_detail().contains("too long"), "got {err:?}");
    }

    #[test]
    fn validate_accepts_finite_in_range() {
        let req = TranscribeRequest {
            samples: vec![-1.0, -0.5, 0.0, 0.5, 1.0],
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn text_equals_concatenated_segments() {
        let resp = TranscribeResponse::from_segments(vec![
            TranscribeSegment {
                start_sec: 0.0,
                end_sec: 1.0,
                text: "hello".to_string(),
            },
            TranscribeSegment {
                start_sec: 1.0,
                end_sec: 2.0,
                text: " world".to_string(),
            },
        ]);
        assert_eq!(resp.text, "hello world");
        assert_eq!(resp.segments.len(), 2);
    }
}
