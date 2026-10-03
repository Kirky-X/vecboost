// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Detect 领域类型域：语言检测请求/响应契约。
//!
//! 只定形状、serde 表示与输入校验；不含检测逻辑。不可判定以
//! `language: null` + `is_reliable: false` 显性表达（`Option::None`
//! 序列化为 null、不省略键），禁止静默成功或虚构语言。

use crate::error::VecboostError;
use crate::text::is_published_language_code;
use crate::utils::constants::MAX_TEXT_LENGTH;
use crate::utils::validator::input::{echo, has_disallowed_control_chars, is_valid_language_code};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
#[cfg(feature = "schema")]
use utoipa::ToSchema;

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DetectRequest {
    pub text: String,
    /// 语言码收窄白名单（发布码契约：ISO 639-1 优先、无 639-1 等价者
    /// 639-3 码，全小写）；缺省或空表 = 不收窄
    #[serde(default)]
    pub allow: Option<Vec<String>>,
}

impl FromStr for DetectRequest {
    /// # 红线（cli/grpc forge 落地时）
    /// `Err = serde_json::Error` 携带内部结构名与字节偏移，handler 必须
    /// 映射为通用 `VecboostError::validation_error` 文案，禁止原样透传客户端
    /// （decision 同款红线，src/domain/decision.rs:92-100）。
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl DetectRequest {
    /// 输入校验：非空/长度上限（按 `chars().count()` 计，CJK 不因字节数
    /// 提前触发）/控制字符防线（`\t` `\n` `\r` 豁免）/allow 逐项语言码
    /// 格式。i18n 文案语义（detect-empty-text / detect-too-long /
    /// detect-invalid-allowlist）在服务层表达，域层以 ValidationError
    /// 显性返回，禁止静默截断或忽略。
    pub fn validate(&self) -> Result<(), VecboostError> {
        if self.text.is_empty() {
            return Err(VecboostError::validation_error(
                "detect text must not be empty".to_string(),
            ));
        }
        let text_chars = self.text.chars().count();
        if text_chars > MAX_TEXT_LENGTH {
            return Err(VecboostError::validation_error(format!(
                "detect text too long: {text_chars} chars > {MAX_TEXT_LENGTH}"
            )));
        }
        if has_disallowed_control_chars(&self.text) {
            // 不回显原文：控制字符直入 wire/日志即日志注入面（decision 同款防线）
            return Err(VecboostError::validation_error(
                "detect text contains control characters".to_string(),
            ));
        }
        if let Some(allow) = &self.allow {
            for code in allow {
                // 双重校验：格式（^[a-z]{2,3}$）+ 发布集合成员（"sw"/"xx" 等
                // 格式合法码不在 whatlang 发布集，穿透会收窄失效——评审 BUG-1）
                if !is_valid_language_code(code) || !is_published_language_code(code) {
                    return Err(VecboostError::validation_error(format!(
                        "invalid language code in allow: {}",
                        echo(code)
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DetectResponse {
    /// 不可判定为 `None`（序列化为 null，显性出现不省略键）
    pub language: Option<String>,
    pub script: Option<String>,
    pub confidence: f64,
    pub is_reliable: bool,
    pub processing_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> DetectRequest {
        DetectRequest {
            text: "The quick brown fox jumps over the lazy dog.".to_string(),
            allow: None,
        }
    }

    #[test]
    fn test_validate_accepts_valid_request() {
        valid_request().validate().unwrap();
    }

    #[test]
    fn test_validate_accepts_allowlist() {
        let req = DetectRequest {
            allow: Some(vec!["zh".to_string(), "en".to_string()]),
            ..valid_request()
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_accepts_empty_allowlist() {
        // 空表 = 不收窄（与 None 等价），非错误
        let req = DetectRequest {
            allow: Some(vec![]),
            ..valid_request()
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_empty_text() {
        let req = DetectRequest {
            text: String::new(),
            allow: None,
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
        // CJK 场景：10_001 个汉字按字节 30_003，但按字符计仅 10_001 超限
        let req = DetectRequest {
            text: "水".repeat(MAX_TEXT_LENGTH + 1),
            allow: None,
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
        // 10_000 个汉字恰在上界内放行（口径按字符而非字节）
        let req = DetectRequest {
            text: "水".repeat(MAX_TEXT_LENGTH),
            allow: None,
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_control_chars() {
        let req = DetectRequest {
            text: "line1\u{1}line2".to_string(),
            allow: None,
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
        let req = DetectRequest {
            text: "line1\nline2\ttabbed\rend".to_string(),
            allow: None,
        };
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_uppercase_language_code() {
        // 大写显性拒绝，无大小写归一（与配置面同一口径）
        let req = DetectRequest {
            allow: Some(vec!["EN".to_string()]),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("EN"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_malformed_language_code() {
        let req = DetectRequest {
            allow: Some(vec!["xx1".to_string()]),
            ..valid_request()
        };
        assert!(matches!(
            req.validate().unwrap_err(),
            VecboostError::ValidationError(_)
        ));

        // 过长（4 位）同样拒绝
        let req = DetectRequest {
            allow: Some(vec!["toolong".to_string()]),
            ..valid_request()
        };
        assert!(req.validate().is_err(), "4 位码必须拒绝");
    }

    #[test]
    fn test_validate_rejects_valid_format_unpublished_code() {
        // 评审 BUG-1 回归钉："xx" 格式合法但未发布，必须 ValidationError
        // 而非穿透到检测核心 panic
        let req = DetectRequest {
            allow: Some(vec!["xx".to_string(), "sw".to_string()]),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        assert!(matches!(err, VecboostError::ValidationError(_)));
        assert!(
            err.error_detail().contains("xx"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_invalid_code_error_echoes_single_line() {
        // 非法码经 echo 字面量化回显：换行注入不得进入错误 detail
        let req = DetectRequest {
            allow: Some(vec!["a\nFAKE LOG LINE".to_string()]),
            ..valid_request()
        };
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(!detail.contains('\n'), "回显必须单行化，detail={detail:?}");
    }

    #[test]
    fn test_detect_request_allow_defaults_to_none() {
        let req: DetectRequest = serde_json::from_str(r#"{"text":"hello"}"#).unwrap();
        assert_eq!(req.allow, None, "缺 allow 字段必须回落 None（零破坏）");
    }

    #[test]
    fn test_detect_request_deserialize_with_allow() {
        let req: DetectRequest =
            serde_json::from_str(r#"{"text":"hello","allow":["zh","en"]}"#).unwrap();
        assert_eq!(req.allow, Some(vec!["zh".to_string(), "en".to_string()]));
    }

    #[test]
    fn test_detect_request_from_str() {
        // cli/grpc forge 代码按 str::parse 解析请求（FromStr 契约）
        let req: DetectRequest = r#"{"text":"hello","allow":["zh"]}"#.parse().unwrap();
        assert_eq!(req.text, "hello");
        assert_eq!(req.allow, Some(vec!["zh".to_string()]));
        assert!(r#"{"text":123}"#.parse::<DetectRequest>().is_err());
    }

    #[test]
    fn test_detect_response_wire_shape_all_five_keys() {
        let resp = DetectResponse {
            language: Some("en".to_string()),
            script: Some("Latin".to_string()),
            confidence: 0.98,
            is_reliable: true,
            processing_time_ms: 3,
        };
        let v = serde_json::to_value(&resp).unwrap();
        // 五键齐全，None 不省略键的对称面：有值时键位显性存在
        for key in [
            "language",
            "script",
            "confidence",
            "is_reliable",
            "processing_time_ms",
        ] {
            assert!(v.get(key).is_some(), "missing key {key}, json={v}");
        }
        assert_eq!(v["language"], "en");
        assert_eq!(v["script"], "Latin");
    }

    #[test]
    fn test_detect_response_undetectable_language_serializes_null() {
        // 不可判定语义：language 键显性出现且为 null（禁止 skip 省略）
        let resp = DetectResponse {
            language: None,
            script: None,
            confidence: 0.0,
            is_reliable: false,
            processing_time_ms: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(
            json.contains("\"language\":null"),
            "null 必须显性出现不省略键，json={json}"
        );
        assert!(json.contains("\"script\":null"), "json={json}");
        assert!(json.contains("\"is_reliable\":false"), "json={json}");
    }
}
