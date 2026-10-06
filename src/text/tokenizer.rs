// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use crate::error::VecboostError;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
}

impl Encoding {
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utf8ValidationResult {
    pub is_valid: bool,
    pub invalid_byte_position: Option<usize>,
    pub invalid_byte_value: Option<u8>,
    pub error_message: Option<String>,
}

impl Utf8ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            invalid_byte_position: None,
            invalid_byte_value: None,
            error_message: None,
        }
    }

    pub fn invalid(position: usize, byte: u8, reason: &str) -> Self {
        Self {
            is_valid: false,
            invalid_byte_position: Some(position),
            invalid_byte_value: Some(byte),
            error_message: Some(reason.to_string()),
        }
    }
}

/// Validate UTF-8 encoding of a byte slice.
pub fn validate_utf8_bytes(bytes: &[u8]) -> Utf8ValidationResult {
    let mut position = 0;

    while position < bytes.len() {
        let byte = bytes[position];
        let (expected_continuation, char_len) = match byte {
            0x00..=0x7F => (0, 1),
            0xC2..=0xDF => (1, 2),
            0xE0..=0xEF => (2, 3),
            0xF0..=0xF4 => (3, 4),
            _ => {
                return Utf8ValidationResult::invalid(
                    position,
                    byte,
                    "Invalid UTF-8 lead byte (not a valid start of multi-byte sequence)",
                );
            }
        };

        if position + char_len > bytes.len() {
            return Utf8ValidationResult::invalid(
                position,
                byte,
                &format!(
                    "Incomplete UTF-8 sequence: expected {} continuation bytes, but data ends",
                    expected_continuation
                ),
            );
        }

        for j in 1..char_len {
            let cont_byte = bytes[position + j];
            if !matches!(cont_byte, 0x80..=0xBF) {
                return Utf8ValidationResult::invalid(
                    position + j,
                    cont_byte,
                    "Invalid UTF-8 continuation byte (expected 0x80-0xBF range)",
                );
            }
        }

        position += char_len;
    }

    Utf8ValidationResult::valid()
}

/// Validate UTF-8 encoding of a string.
pub fn validate_utf8(text: &str) -> Utf8ValidationResult {
    validate_utf8_bytes(text.as_bytes())
}

/// 同步 tokenizer 封装——embed 热路径统一入口（HF tokenizers + max_length
/// 截断 + UTF-8 校验）。同步 API 面向 `spawn_blocking` 推理上下文；
/// 历史上的 moka 异步缓存路径（async encode/hit 统计）在生产编码路径
/// 改走 `spawn_blocking` 后无消费者，已随死代码清理移除。
#[derive(Debug, Clone)]
pub struct SyncTokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
}

impl SyncTokenizer {
    pub fn new(tokenizer: HfTokenizer, max_length: usize) -> Self {
        Self {
            tokenizer,
            max_length,
        }
    }

    /// 从 tokenizer.json 加载（默认 max_length=512），失败 →
    /// VecboostError(含路径)，禁止静默回退。
    pub fn from_file(path: &str) -> Result<Self, VecboostError> {
        Self::from_file_with_max_length(path, 512)
    }

    pub fn from_file_with_max_length(path: &str, max_length: usize) -> Result<Self, VecboostError> {
        let tokenizer = HfTokenizer::from_file(path).map_err(|e| {
            VecboostError::tokenization_error(format!(
                "Failed to load tokenizer from file '{}': {}. \
                Please ensure the file exists and is a valid tokenizer.json.",
                path, e
            ))
        })?;
        Self::validate_and_build(tokenizer, max_length)
    }

    fn validate_and_build(
        tokenizer: HfTokenizer,
        max_length: usize,
    ) -> Result<Self, VecboostError> {
        if max_length == 0 {
            return Err(VecboostError::invalid_input(format!(
                "max_length must be greater than 0, got {}",
                max_length
            )));
        }
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// 批量编码时的序列截断上限（candle_engine 单条与批量编码共用）
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// 同步 token 计数（用于 API usage.prompt_tokens 与服务层 token 预算预判）
    pub fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        if text.is_empty() {
            return Ok(0);
        }
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            VecboostError::tokenization_error(format!("count_tokens failed: {}", e))
        })?;
        Ok(encoding.get_ids().len())
    }

    /// 同步编码：UTF-8 校验 → HF 编码 → 按 max_length 截断。
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, VecboostError> {
        if text.is_empty() {
            return Err(VecboostError::invalid_input(
                "Cannot encode empty text".to_string(),
            ));
        }

        let utf8_result = validate_utf8(text);
        if !utf8_result.is_valid {
            return Err(VecboostError::invalid_input(format!(
                "UTF-8 encoding validation failed at byte {} (value 0x{:02x}): {}. \
                The input contains invalid or incomplete UTF-8 sequences.",
                utf8_result.invalid_byte_position.unwrap_or(0),
                utf8_result.invalid_byte_value.unwrap_or(0),
                utf8_result
                    .error_message
                    .unwrap_or_else(|| "Unknown UTF-8 error".to_string())
            )));
        }

        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}. \
                The text may contain unsupported characters or be too long.",
                    text.len(),
                    e
                ))
            })?;

        Ok(Self::truncate_encoding(encoding, self.max_length))
    }

    /// 将 HF Encoding 转为我们的 Encoding，按 max_length 截断
    fn truncate_encoding(encoding: tokenizers::Encoding, max_length: usize) -> Encoding {
        let ids = encoding.get_ids().to_vec();
        let truncated_ids = if ids.len() > max_length {
            ids[..max_length].to_vec()
        } else {
            ids
        };

        let attention_mask = encoding.get_attention_mask().to_vec();
        let truncated_mask = if attention_mask.len() > max_length {
            attention_mask[..max_length].to_vec()
        } else {
            attention_mask
        };

        let type_ids = encoding.get_type_ids().to_vec();
        let truncated_type_ids = if type_ids.len() > max_length {
            type_ids[..max_length].to_vec()
        } else {
            type_ids
        };

        Encoding {
            ids: truncated_ids,
            attention_mask: truncated_mask,
            type_ids: truncated_type_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Encoding 结构体测试(平台无关)
    // ========================================================================

    #[test]
    fn test_encoding_getters_return_correct_slices() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            type_ids: vec![0, 0, 0],
        };
        assert_eq!(encoding.get_ids(), &[1, 2, 3]);
        assert_eq!(encoding.get_attention_mask(), &[1, 1, 1]);
        assert_eq!(encoding.get_type_ids(), &[0, 0, 0]);
    }

    #[test]
    fn test_encoding_getters_on_empty() {
        let encoding = Encoding {
            ids: vec![],
            attention_mask: vec![],
            type_ids: vec![],
        };
        assert!(encoding.get_ids().is_empty());
        assert!(encoding.get_attention_mask().is_empty());
        assert!(encoding.get_type_ids().is_empty());
    }

    #[test]
    fn test_encoding_eq_clone_debug() {
        let encoding = Encoding {
            ids: vec![1],
            attention_mask: vec![1],
            type_ids: vec![0],
        };
        let cloned = encoding.clone();
        assert_eq!(encoding, cloned);
        let debug_str = format!("{:?}", encoding);
        assert!(debug_str.contains("Encoding"));
    }

    // ========================================================================
    // UTF-8 验证测试(平台无关)
    // ========================================================================

    #[test]
    fn test_utf8_validation_result_valid_constructor() {
        let r = Utf8ValidationResult::valid();
        assert!(r.is_valid);
        assert_eq!(r.invalid_byte_position, None);
        assert_eq!(r.invalid_byte_value, None);
        assert_eq!(r.error_message, None);
    }

    #[test]
    fn test_utf8_validation_result_invalid_constructor() {
        let r = Utf8ValidationResult::invalid(7, 0xFF, "bad lead byte");
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(7));
        assert_eq!(r.invalid_byte_value, Some(0xFF));
        assert_eq!(r.error_message.as_deref(), Some("bad lead byte"));
    }

    #[test]
    fn test_validate_utf8_empty_string_is_valid() {
        let r = validate_utf8("");
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_utf8_ascii() {
        let r = validate_utf8("Hello, World! 123 \t\n");
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_utf8_two_byte_sequence() {
        let r = validate_utf8("éñüΩ");
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_utf8_three_byte_sequence() {
        let r = validate_utf8("中文日本語한국어");
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_utf8_four_byte_sequence_emoji() {
        let r = validate_utf8("🚀🎉🦀😀");
        assert!(r.is_valid);
    }

    #[test]
    fn test_validate_utf8_invalid_lead_byte_0x80() {
        let bytes = [0x80u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert_eq!(r.invalid_byte_value, Some(0x80));
        assert!(r.error_message.as_deref().unwrap().contains("lead byte"));
    }

    #[test]
    fn test_validate_utf8_invalid_lead_byte_0xc0() {
        let bytes = [0xC0u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert_eq!(r.invalid_byte_value, Some(0xC0));
    }

    #[test]
    fn test_validate_utf8_invalid_lead_byte_0xff() {
        let bytes = [0xFFu8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_value, Some(0xFF));
    }

    #[test]
    fn test_validate_utf8_incomplete_two_byte_sequence() {
        let bytes = [0xC2u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert_eq!(r.invalid_byte_value, Some(0xC2));
        assert!(
            r.error_message
                .as_deref()
                .unwrap()
                .contains("Incomplete UTF-8 sequence")
        );
    }

    #[test]
    fn test_validate_utf8_incomplete_three_byte_sequence() {
        let bytes = [0xE0u8, 0x80u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert!(
            r.error_message
                .as_deref()
                .unwrap()
                .contains("expected 2 continuation")
        );
    }

    #[test]
    fn test_validate_utf8_incomplete_four_byte_sequence() {
        let bytes = [0xF0u8, 0x80u8, 0x80u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert!(
            r.error_message
                .as_deref()
                .unwrap()
                .contains("expected 3 continuation")
        );
    }

    #[test]
    fn test_validate_utf8_invalid_continuation_byte() {
        let bytes = [0xC2u8, 0xFFu8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(1));
        assert_eq!(r.invalid_byte_value, Some(0xFF));
        assert!(
            r.error_message
                .as_deref()
                .unwrap()
                .contains("continuation byte")
        );
    }

    #[test]
    fn test_validate_utf8_valid_then_invalid_continuation_byte() {
        let bytes = [b'a', 0xE0u8, 0x80u8, 0x00u8];
        let r = validate_utf8_bytes(&bytes);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(3));
        assert_eq!(r.invalid_byte_value, Some(0x00));
    }

    // ========================================================================
    // 本地 tokenizer.json 加载测试(使用 models/ 下真实模型)
    // ========================================================================

    /// 辅助:获取第一个可用本地模型的 tokenizer.json 路径
    fn find_local_tokenizer_path() -> Option<String> {
        let models = [
            "models/all-MiniLM-L6-v2",
            "models/BAAI-bge-small-en-v1.5",
            "models/BAAI-bge-small-zh-v1.5",
            "models/multilingual-e5-small",
        ];
        for m in &models {
            let p = format!("{}/tokenizer.json", m);
            if std::path::Path::new(&p).exists() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn test_tokenizer_from_file_success() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let tokenizer = SyncTokenizer::from_file(&path).unwrap();
        let encoding = tokenizer.encode("hello world", true).unwrap();
        assert!(!encoding.ids.is_empty());
    }

    #[test]
    fn test_tokenizer_from_file_nonexistent_returns_error() {
        let result = SyncTokenizer::from_file("/nonexistent/path/tokenizer.json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let detail = err.error_detail().to_string();
        assert!(
            detail.contains("/nonexistent/path/tokenizer.json"),
            "got: {}",
            detail
        );
    }

    #[test]
    fn test_tokenizer_from_file_zero_max_length_returns_error() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let result = SyncTokenizer::from_file_with_max_length(&path, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().error_detail().contains("max_length"));
    }

    #[test]
    fn test_tokenizer_encode_with_local_model() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let tokenizer = SyncTokenizer::from_file(&path).unwrap();
        let encoding = tokenizer.encode("hello world", true).unwrap();
        assert!(!encoding.ids.is_empty());
        assert_eq!(encoding.ids.len(), encoding.attention_mask.len());
        assert_eq!(encoding.ids.len(), encoding.type_ids.len());
    }

    #[test]
    fn test_tokenizer_encode_empty_returns_error() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let tokenizer = SyncTokenizer::from_file(&path).unwrap();
        let result = tokenizer.encode("", true);
        assert!(result.is_err());
        assert!(result.unwrap_err().error_detail().contains("empty text"));
    }
}
