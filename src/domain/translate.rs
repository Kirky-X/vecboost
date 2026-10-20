// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Translate 领域类型域：翻译请求/响应契约。
//!
//! 只定形状、serde 表示与输入校验；不含翻译逻辑。语言码为发布码契约
//! （ISO 639-1 优先、无等价者 639-3 码，全小写），与 /detect、配置面
//! 同一谓词同一口径，无大小写归一。

use crate::error::VecboostError;
use crate::utils::constants::MAX_TEXT_LENGTH;
use crate::utils::validator::input::{echo, has_disallowed_control_chars, is_valid_language_code};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
#[cfg(feature = "schema")]
use utoipa::ToSchema;

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct TranslateRequest {
    pub text: String,
    /// 源语言；None = auto 检测（检测由服务层执行，
    /// `is_reliable=false` 显性拒绝，检测结果回填响应 `detected_source_language`）
    #[serde(default)]
    pub source_language: Option<String>,
    pub target_language: String,
}

impl FromStr for TranslateRequest {
    /// # 红线（cli/grpc forge 落地时）
    /// `Err = serde_json::Error` 携带内部结构名与字节偏移，handler 必须
    /// 映射为通用 `VecboostError::validation_error` 文案，禁止原样透传客户端
    /// （decision 同款红线，src/domain/decision.rs:92-100）。
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl TranslateRequest {
    /// 输入校验：非空/长度上限（按 `chars.count` 计）/控制字符防线
    /// （`\t` `\n` `\r` 豁免）/语言码格式（大小写敏感）/source ≠ target。
    /// i18n 文案语义（translate-empty-text / translate-too-long /
    /// translate-unsupported 等）在服务层表达，域层以 ValidationError
    /// 显性返回，禁止静默截断或忽略。
    pub fn validate(&self) -> Result<(), VecboostError> {
        if self.text.is_empty() {
            return Err(VecboostError::validation_error(
                "translate text must not be empty".to_string(),
            ));
        }
        let text_chars = self.text.chars().count();
        if text_chars > MAX_TEXT_LENGTH {
            return Err(VecboostError::validation_error(format!(
                "translate text too long: {text_chars} chars > {MAX_TEXT_LENGTH}"
            )));
        }
        if has_disallowed_control_chars(&self.text) {
            // 不回显原文：控制字符直入 wire/日志即日志注入面（decision 同款防线）
            return Err(VecboostError::validation_error(
                "translate text contains control characters".to_string(),
            ));
        }
        if !is_valid_language_code(&self.target_language) {
            return Err(VecboostError::validation_error(format!(
                "invalid target_language: {} (lowercase ISO 639-1 codes only)",
                echo(&self.target_language)
            )));
        }
        if let Some(source) = &self.source_language {
            if !is_valid_language_code(source) {
                return Err(VecboostError::validation_error(format!(
                    "invalid source_language: {} (lowercase ISO 639-1 codes only)",
                    echo(source)
                )));
            }
            if source == &self.target_language {
                return Err(VecboostError::validation_error(format!(
                    "source_language and target_language must differ (both {})",
                    echo(source)
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct TranslateResponse {
    pub translation: String,
    /// 请求回显（auto 语义时为 null，显性出现不省略键）
    pub source_language: Option<String>,
    pub target_language: String,
    /// auto 检测回填（显式指定源语言时为 null）
    pub detected_source_language: Option<String>,
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> TranslateRequest {
        TranslateRequest {
            text: "The quick brown fox jumps over the lazy dog.".to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        }
    }

    #[test]
    fn test_validate_accepts_valid_request() {
        valid_request().validate().unwrap();
    }

    #[test]
    fn test_validate_accepts_auto_source() {
        // source_language = None 即 auto 检测语义
        let req = TranslateRequest {
            source_language: None,
            ..valid_request()
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_empty_text() {
        let req = TranslateRequest {
            text: String::new(),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(matches!(err, VecboostError::ValidationError(_)));
        assert!(
            err.error_detail().contains("empty"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_text_over_char_limit() {
        // CJK 口径：10_001 个汉字按字节 30_003，按字符计 10_001 超限
        let req = TranslateRequest {
            text: "水".repeat(MAX_TEXT_LENGTH + 1),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("too long"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_char_count_boundary_passes() {
        let req = TranslateRequest {
            text: "水".repeat(MAX_TEXT_LENGTH),
            ..valid_request()
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_control_chars() {
        let req = TranslateRequest {
            text: "line1\u{1}line2".to_string(),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("control characters"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_allows_tab_newline_carriage_return() {
        let req = TranslateRequest {
            text: "line1\nline2\ttabbed\rend".to_string(),
            ..valid_request()
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_same_source_and_target() {
        let req = TranslateRequest {
            source_language: Some("en".to_string()),
            target_language: "en".to_string(),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("differ") || err.error_detail().contains("same"),
            "source == target 必须显性拒绝，err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_uppercase_language_code() {
        let req = TranslateRequest {
            target_language: "ZH".to_string(),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("ZH"),
            "大写目标码必须拒绝，err={}",
            err.error_detail()
        );

        let req = TranslateRequest {
            source_language: Some("EN".to_string()),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("EN"),
            "大写源码必须拒绝，err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_malformed_language_code() {
        for code in ["xx1", "toolong", "e"] {
            let req = TranslateRequest {
                target_language: code.to_string(),
                ..valid_request()
            };
            assert!(
                matches!(
                    req.validate().unwrap_err(),
                    VecboostError::ValidationError(_)
                ),
                "非法目标码 {code} 必须拒绝"
            );
            let req = TranslateRequest {
                source_language: Some(code.to_string()),
                ..valid_request()
            };
            assert!(
                matches!(
                    req.validate().unwrap_err(),
                    VecboostError::ValidationError(_)
                ),
                "非法源码 {code} 必须拒绝"
            );
        }
    }

    #[test]
    fn test_validate_invalid_code_error_echoes_single_line() {
        let req = TranslateRequest {
            source_language: Some("a\nFAKE LOG LINE".to_string()),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(!detail.contains('\n'), "回显必须单行化，detail={detail:?}");
    }

    #[test]
    fn test_translate_request_deserialize() {
        // 缺 source_language 字段 → None（auto 语义，零破坏）
        let auto: TranslateRequest =
            serde_json::from_str(r#"{"text":"hello","target_language":"zh"}"#).unwrap();
        assert_eq!(auto.source_language, None);

        // 显式 null 与缺省等价
        let null_src: TranslateRequest = serde_json::from_str(
            r#"{"text":"hello","source_language":null,"target_language":"zh"}"#,
        )
        .unwrap();
        assert_eq!(null_src.source_language, None);

        let full: TranslateRequest = serde_json::from_str(
            r#"{"text":"hello","source_language":"en","target_language":"zh"}"#,
        )
        .unwrap();
        assert_eq!(full.source_language.as_deref(), Some("en"));
    }

    #[test]
    fn test_translate_request_from_str() {
        let req: TranslateRequest = r#"{"text":"hello","target_language":"zh"}"#.parse().unwrap();
        assert_eq!(req.target_language, "zh");
        assert!(r#"{"text":123}"#.parse::<TranslateRequest>().is_err());
    }

    #[test]
    fn test_translate_response_wire_shape_all_five_keys() {
        let resp = TranslateResponse {
            translation: "你好".to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
            detected_source_language: None,
            processing_time_ms: 3,
        };
        let v = serde_json::to_value(&resp).unwrap();
        for key in [
            "translation",
            "source_language",
            "target_language",
            "detected_source_language",
            "processing_time_ms",
        ] {
            assert!(v.get(key).is_some(), "missing key {key}, json={v}");
        }
        assert_eq!(v["translation"], "你好");
        assert_eq!(v["target_language"], "zh");
    }

    #[test]
    fn test_translate_response_null_keys_explicit() {
        // auto 语义：detected_source_language 有值、source_language null 显性出现
        let resp = TranslateResponse {
            translation: "你好".to_string(),
            source_language: None,
            target_language: "zh".to_string(),
            detected_source_language: Some("en".to_string()),
            processing_time_ms: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(
            json.contains("\"source_language\":null"),
            "null 必须显性出现不省略键，json={json}"
        );
        assert!(
            json.contains("\"detected_source_language\":\"en\""),
            "json={json}"
        );
    }
}
