// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
use oxcache::backend::MokaMemoryBackend;
use oxcache::cache::Cache;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::Tokenizer as HfTokenizer;
use xxhash_rust::xxh3::Xxh3;

pub const DEFAULT_CACHE_SIZE: usize = 1024;
pub const MAX_CACHE_SIZE: usize = 8192;

/// 统一 tokenizer——全平台走 HuggingFace tokenizers crate。
/// 加载失败 → VecboostError(含尝试路径),禁止静默回退。
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CacheHitStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl Default for CacheHitStats {
    fn default() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }
}

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

#[allow(dead_code)]
impl Tokenizer {
    pub fn from_pretrained(model_id: &str) -> Result<Self, VecboostError> {
        Self::from_pretrained_with_max_length(model_id, 512)
    }

    pub fn from_pretrained_with_max_length(
        model_id: &str,
        max_length: usize,
    ) -> Result<Self, VecboostError> {
        // Try local directory first (model_id as path with tokenizer.json)
        let tokenizer_path = std::path::Path::new(model_id).join("tokenizer.json");
        if tokenizer_path.exists() {
            return Self::from_file_with_max_length(
                tokenizer_path.to_string_lossy().as_ref(),
                max_length,
            );
        }
        // Try HF hub
        HfTokenizer::from_pretrained(model_id, None)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to load tokenizer from model '{}': {}. \
                Please check that the model ID is correct and the model is available. \
                For local models, ensure the tokenizer.json file exists.",
                    model_id, e
                ))
            })
            .and_then(|tokenizer| Self::validate_and_build(tokenizer, max_length))
    }

    pub fn from_file(path: &str) -> Result<Self, VecboostError> {
        Self::from_file_with_max_length(path, 512)
    }

    /// 从 tokenizer.json 加载,失败 → VecboostError(含路径),禁止静默回退。
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

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, VecboostError> {
        if ids.is_empty() {
            return Err(VecboostError::invalid_input(
                "Cannot decode empty token ids".to_string(),
            ));
        }

        if ids
            .iter()
            .any(|&id| id >= self.tokenizer.get_vocab_size(true) as u32)
        {
            return Err(VecboostError::tokenization_error(format!(
                "Invalid token id found: one or more ids exceed vocabulary size ({}). \
                This may indicate corrupted or incompatible token ids.",
                self.tokenizer.get_vocab_size(true)
            )));
        }

        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to decode {} token ids: {}. \
                The ids may be invalid or incompatible with this tokenizer.",
                    ids.len(),
                    e
                ))
            })
    }

    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn get_max_length(&self) -> usize {
        self.max_length
    }

    /// 供 onnx_engine 使用的 padding 配置方法
    pub fn get_padding_mut(&mut self) -> Option<&mut tokenizers::PaddingParams> {
        self.tokenizer.get_padding_mut()
    }

    /// 供 onnx_engine 使用的 padding 配置方法
    pub fn with_padding(&mut self, padding: Option<tokenizers::PaddingParams>) {
        self.tokenizer.with_padding(padding);
    }

    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, VecboostError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        for (i, &text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(VecboostError::invalid_input(format!(
                    "Cannot encode empty text at batch index {}",
                    i
                )));
            }

            let utf8_result = validate_utf8(text);
            if !utf8_result.is_valid {
                return Err(VecboostError::invalid_input(format!(
                    "UTF-8 encoding validation failed at byte {} (value 0x{:02x}) in text at batch index {}: {}. \
                    The input contains invalid or incomplete UTF-8 sequences.",
                    utf8_result.invalid_byte_position.unwrap_or(0),
                    utf8_result.invalid_byte_value.unwrap_or(0),
                    i,
                    utf8_result
                        .error_message
                        .unwrap_or_else(|| "Unknown UTF-8 error".to_string())
                )));
            }
        }

        let texts_str: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();

        let batch = self
            .tokenizer
            .encode_batch(texts_str, add_special_tokens)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to encode batch of {} texts: {}. \
                Some texts may be invalid or too long.",
                    texts.len(),
                    e
                ))
            })?;

        Ok(batch
            .into_iter()
            .map(|encoding| Self::truncate_encoding(encoding, self.max_length))
            .collect())
    }

    /// 将 HF Encoding 转为我们的 Encoding,按 max_length 截断
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

/// 统一 CachedTokenizer——全平台使用 HF tokenizers,stats 改 AtomicU64。
#[derive(Debug)]
#[allow(dead_code)]
pub struct CachedTokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
    cache: Cache<String, Encoding>,
    hits: AtomicU64,
    misses: AtomicU64,
}

#[allow(dead_code)]
impl CachedTokenizer {
    pub fn new(tokenizer: HfTokenizer, max_length: usize, cache_size: usize) -> Self {
        let capacity = cache_size.clamp(1, MAX_CACHE_SIZE) as u64;
        let moka = MokaMemoryBackend::builder().capacity(capacity).build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        Self {
            tokenizer,
            max_length,
            cache,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn with_default_cache(tokenizer: HfTokenizer, max_length: usize) -> Self {
        Self::new(tokenizer, max_length, DEFAULT_CACHE_SIZE)
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// 同步 token 计数(用于 API usage.prompt_tokens)
    pub fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        if text.is_empty() {
            return Ok(0);
        }
        let encoding = self.tokenizer.encode(text, true).map_err(|e| {
            VecboostError::tokenization_error(format!("count_tokens failed: {}", e))
        })?;
        Ok(encoding.get_ids().len())
    }

    pub fn stats(&self) -> CacheHitStats {
        CacheHitStats {
            hits: AtomicU64::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.misses.load(Ordering::Relaxed)),
        }
    }

    fn hash_key(&self, text: &str, add_special_tokens: bool) -> String {
        let mut hasher = Xxh3::new();
        text.hash(&mut hasher);
        add_special_tokens.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{:016x}_{}", hash, add_special_tokens)
    }

    pub async fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, VecboostError> {
        let key = self.hash_key(text, add_special_tokens);

        if let Some(cached) = self.cache.get(&key).await.ok().flatten() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached);
        }

        let encoding = self.encode_uncached(text, add_special_tokens).await?;

        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = self.cache.set(&key, &encoding).await {
            log::warn!("Failed to cache tokenization result: {}", e);
        }

        Ok(encoding)
    }

    async fn encode_uncached(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, VecboostError> {
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

        Ok(Tokenizer::truncate_encoding(encoding, self.max_length))
    }

    /// T034: 同步 encode——绕过异步缓存，直接调用底层 tokenizer。
    ///
    /// 供 spawn_blocking 上下文使用，避免 async-in-sync 问题。
    pub fn encode_sync(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, VecboostError> {
        if text.is_empty() {
            return Err(VecboostError::invalid_input(
                "Cannot encode empty text".to_string(),
            ));
        }

        let utf8_result = validate_utf8(text);
        if !utf8_result.is_valid {
            return Err(VecboostError::tokenization_error(format!(
                "Invalid UTF-8: {:?}",
                utf8_result.error_message
            )));
        }

        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}",
                    text.len(),
                    e
                ))
            })?;

        Ok(Tokenizer::truncate_encoding(encoding, self.max_length))
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
    // 缓存常量测试
    // ========================================================================

    #[test]
    fn test_cache_constants() {
        assert_eq!(DEFAULT_CACHE_SIZE, 1024);
        assert_eq!(MAX_CACHE_SIZE, 8192);
        const _: () = {
            assert!(DEFAULT_CACHE_SIZE < MAX_CACHE_SIZE);
        };
    }

    #[test]
    fn test_cache_hit_stats_default() {
        let stats = CacheHitStats::default();
        assert_eq!(stats.hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 0);
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
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        assert_eq!(tokenizer.get_max_length(), 512);
        assert!(tokenizer.get_vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_from_file_nonexistent_returns_error() {
        let result = Tokenizer::from_file("/nonexistent/path/tokenizer.json");
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
        let result = Tokenizer::from_file_with_max_length(&path, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().error_detail().contains("max_length"));
    }

    #[test]
    fn test_tokenizer_encode_with_local_model() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let tokenizer = Tokenizer::from_file(&path).unwrap();
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
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let result = tokenizer.encode("", true);
        assert!(result.is_err());
        assert!(result.unwrap_err().error_detail().contains("empty text"));
    }

    #[test]
    fn test_tokenizer_encode_batch_empty() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let result = tokenizer.encode_batch(&[], true).unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_cached_tokenizer_encode_miss_then_hit() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let hf_tok = HfTokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(hf_tok, 512);

        let _ = cached.encode("hello", true).await.unwrap();
        assert_eq!(cached.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cached.misses.load(Ordering::Relaxed), 1);

        let _ = cached.encode("hello", true).await.unwrap();
        assert_eq!(cached.hits.load(Ordering::Relaxed), 1);
        assert_eq!(cached.misses.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_different_keys() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let hf_tok = HfTokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(hf_tok, 512);

        let _ = cached.encode("hello", false).await.unwrap();
        let _ = cached.encode("hello", true).await.unwrap();
        assert_eq!(cached.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cached.misses.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_empty_text_no_cache() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let hf_tok = HfTokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(hf_tok, 512);

        let result = cached.encode("", true).await;
        assert!(result.is_err());
        assert_eq!(cached.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cached.misses.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hash_key_differs_by_special_tokens() {
        let Some(path) = find_local_tokenizer_path() else {
            eprintln!("SKIP: no local model found");
            return;
        };
        let hf_tok = HfTokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(hf_tok, 512);
        let key_false = cached.hash_key("hello", false);
        let key_true = cached.hash_key("hello", true);
        assert_ne!(key_false, key_true);
    }
}
