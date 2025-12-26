// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use lru::LruCache;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use tokenizers::Tokenizer as HfTokenizer;
use tokio::sync::Mutex;
use xxhash_rust::xxh3::Xxh3;

pub const DEFAULT_CACHE_SIZE: usize = 1024;
pub const MAX_CACHE_SIZE: usize = 8192;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
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

pub fn validate_utf8(text: &str) -> Utf8ValidationResult {
    let bytes = text.as_bytes();
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

pub fn validate_utf8_bytes(bytes: &[u8]) -> Utf8ValidationResult {
    let mut position = 0;

    while position < bytes.len() {
        let byte = bytes[position];
        let (expected_continuation, char_len) = match byte {
            0x00..=0x7F => (0, 1),
            0xC2..=0xDF => (1, 2),
            0xE0..=0xEF => (2, 3),
            0xF0..=0xF4 => (3, 4),
            0x80..=0xBF => {
                return Utf8ValidationResult::invalid(
                    position,
                    byte,
                    "Unexpected UTF-8 continuation byte (found outside valid sequence)",
                );
            }
            0xF5..=0xFF => {
                return Utf8ValidationResult::invalid(
                    position,
                    byte,
                    "Invalid UTF-8 lead byte (reserved for future use)",
                );
            }
            _ => {
                return Utf8ValidationResult::invalid(position, byte, "Invalid UTF-8 byte value");
            }
        };

        if position + char_len > bytes.len() {
            return Utf8ValidationResult::invalid(
                position,
                byte,
                &format!(
                    "Incomplete UTF-8 sequence at end of data: expected {} continuation bytes, got {}",
                    expected_continuation,
                    bytes.len() - position - 1
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

impl Tokenizer {
    pub fn from_pretrained(model_id: &str) -> Result<Self, AppError> {
        Self::from_pretrained_with_max_length(model_id, 512)
    }

    pub fn from_pretrained_with_max_length(
        model_id: &str,
        max_length: usize,
    ) -> Result<Self, AppError> {
        HfTokenizer::from_pretrained(model_id, None)
            .map_err(|e| {
                AppError::tokenization_error(format!(
                    "Failed to load tokenizer from model '{}': {}. \
                Please check that the model ID is correct and the model is available. \
                For local models, ensure the tokenizer.json file exists.",
                    model_id, e
                ))
            })
            .and_then(|tokenizer| {
                if max_length == 0 {
                    Err(AppError::invalid_input(format!(
                        "max_length must be greater than 0, got {}",
                        max_length
                    )))
                } else {
                    Ok(Self {
                        tokenizer,
                        max_length,
                    })
                }
            })
    }

    pub fn from_file(path: &str) -> Result<Self, AppError> {
        Self::from_file_with_max_length(path, 512)
    }

    pub fn from_file_with_max_length(path: &str, max_length: usize) -> Result<Self, AppError> {
        HfTokenizer::from_file(path)
            .map_err(|e| {
                AppError::tokenization_error(format!(
                    "Failed to load tokenizer from file '{}': {}. \
                Please ensure the file exists and is a valid tokenizer file.",
                    path, e
                ))
            })
            .and_then(|tokenizer| {
                if max_length == 0 {
                    Err(AppError::invalid_input(format!(
                        "max_length must be greater than 0, got {}",
                        max_length
                    )))
                } else {
                    Ok(Self {
                        tokenizer,
                        max_length,
                    })
                }
            })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, AppError> {
        if text.is_empty() {
            return Err(AppError::invalid_input(
                "Cannot encode empty text".to_string(),
            ));
        }

        let utf8_result = validate_utf8(text);
        if !utf8_result.is_valid {
            return Err(AppError::invalid_input(format!(
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
                AppError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}. \
                The text may contain unsupported characters or be too long.",
                    text.len(),
                    e
                ))
            })?;

        let ids = encoding.get_ids().to_vec();
        let truncated_ids: Vec<u32> = if ids.len() > self.max_length {
            ids[..self.max_length].to_vec()
        } else {
            ids
        };

        let attention_mask = encoding.get_attention_mask().to_vec();
        let truncated_mask: Vec<u32> = if attention_mask.len() > self.max_length {
            attention_mask[..self.max_length].to_vec()
        } else {
            attention_mask
        };

        let type_ids = encoding.get_type_ids().to_vec();
        let truncated_type_ids: Vec<u32> = if type_ids.len() > self.max_length {
            type_ids[..self.max_length].to_vec()
        } else {
            type_ids
        };

        Ok(Encoding {
            ids: truncated_ids,
            attention_mask: truncated_mask,
            type_ids: truncated_type_ids,
            tokens: encoding.get_tokens().to_vec(),
        })
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, AppError> {
        if ids.is_empty() {
            return Err(AppError::invalid_input(
                "Cannot decode empty token ids".to_string(),
            ));
        }

        if ids
            .iter()
            .any(|&id| id >= self.tokenizer.get_vocab_size(true) as u32)
        {
            return Err(AppError::tokenization_error(format!(
                "Invalid token id found: one or more ids exceed vocabulary size ({}). \
                This may indicate corrupted or incompatible token ids.",
                self.tokenizer.get_vocab_size(true)
            )));
        }

        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| {
                AppError::tokenization_error(format!(
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

    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        for (i, &text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(AppError::invalid_input(format!(
                    "Cannot encode empty text at batch index {}",
                    i
                )));
            }

            let utf8_result = validate_utf8(text);
            if !utf8_result.is_valid {
                return Err(AppError::invalid_input(format!(
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
                AppError::tokenization_error(format!(
                    "Failed to encode batch of {} texts: {}. \
                Some texts may be invalid or too long.",
                    texts.len(),
                    e
                ))
            })?;

        let mut encodings = Vec::with_capacity(batch.len());

        for encoding in batch.into_iter() {
            let ids = encoding.get_ids().to_vec();
            let truncated_ids: Vec<u32> = if ids.len() > self.max_length {
                ids[..self.max_length].to_vec()
            } else {
                ids
            };

            let attention_mask = encoding.get_attention_mask().to_vec();
            let truncated_mask: Vec<u32> = if attention_mask.len() > self.max_length {
                attention_mask[..self.max_length].to_vec()
            } else {
                attention_mask
            };

            let type_ids = encoding.get_type_ids().to_vec();
            let truncated_type_ids: Vec<u32> = if type_ids.len() > self.max_length {
                type_ids[..self.max_length].to_vec()
            } else {
                type_ids
            };

            encodings.push(Encoding {
                ids: truncated_ids,
                attention_mask: truncated_mask,
                type_ids: truncated_type_ids,
                tokens: encoding.get_tokens().to_vec(),
            });
        }

        Ok(encodings)
    }
}

#[derive(Debug)]
pub struct CachedTokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
    cache: Mutex<LruCache<String, Encoding>>,
    stats: Mutex<CacheHitStats>,
}

#[derive(Debug, Default)]
pub struct CacheHitStats {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub capacity: usize,
}

impl CachedTokenizer {
    pub fn new(tokenizer: HfTokenizer, max_length: usize, cache_size: usize) -> Self {
        let capacity = std::cmp::min(
            NonZeroUsize::new(std::cmp::max(cache_size, 1)).unwrap(),
            NonZeroUsize::new(MAX_CACHE_SIZE).unwrap(),
        );
        Self {
            tokenizer,
            max_length,
            cache: Mutex::new(LruCache::new(capacity)),
            stats: Mutex::new(CacheHitStats::default()),
        }
    }

    pub fn with_default_cache(tokenizer: HfTokenizer, max_length: usize) -> Self {
        Self::new(tokenizer, max_length, DEFAULT_CACHE_SIZE)
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    fn hash_key(&self, text: &str, add_special_tokens: bool) -> String {
        let mut hasher = Xxh3::new();
        text.hash(&mut hasher);
        add_special_tokens.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{:016x}_{}", hash, add_special_tokens)
    }

    pub async fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, AppError> {
        let key = self.hash_key(text, add_special_tokens);

        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get(&key) {
                let mut stats = self.stats.lock().await;
                stats.hits += 1;
                return Ok(cached.clone());
            }
        }

        let encoding = self.encode_uncached(text, add_special_tokens).await?;

        let mut cache = self.cache.lock().await;
        let mut stats = self.stats.lock().await;
        stats.misses += 1;
        cache.push(key, encoding.clone());

        Ok(encoding)
    }

    async fn encode_uncached(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, AppError> {
        if text.is_empty() {
            return Err(AppError::invalid_input(
                "Cannot encode empty text".to_string(),
            ));
        }

        let utf8_result = validate_utf8(text);
        if !utf8_result.is_valid {
            return Err(AppError::invalid_input(format!(
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
                AppError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}. \
                The text may contain unsupported characters or be too long.",
                    text.len(),
                    e
                ))
            })?;

        let ids = encoding.get_ids().to_vec();
        let truncated_ids: Vec<u32> = if ids.len() > self.max_length {
            ids[..self.max_length].to_vec()
        } else {
            ids
        };

        let attention_mask = encoding.get_attention_mask().to_vec();
        let truncated_mask: Vec<u32> = if attention_mask.len() > self.max_length {
            attention_mask[..self.max_length].to_vec()
        } else {
            attention_mask
        };

        let type_ids = encoding.get_type_ids().to_vec();
        let truncated_type_ids: Vec<u32> = if type_ids.len() > self.max_length {
            type_ids[..self.max_length].to_vec()
        } else {
            type_ids
        };

        Ok(Encoding {
            ids: truncated_ids,
            attention_mask: truncated_mask,
            type_ids: truncated_type_ids,
            tokens: encoding.get_tokens().to_vec(),
        })
    }

    pub async fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, AppError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let encoding = self.encode(text, add_special_tokens).await?;
            results.push(encoding);
        }
        Ok(results)
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.lock().await;
        cache.clear();
    }

    pub async fn get_cache_stats(&self) -> CacheStats {
        let cache = self.cache.lock().await;
        let stats = self.stats.lock().await;
        CacheStats {
            hits: stats.hits,
            misses: stats.misses,
            size: cache.len(),
            capacity: cache.cap().get(),
        }
    }

    pub async fn cache_len(&self) -> usize {
        let cache = self.cache.lock().await;
        cache.len()
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, AppError> {
        if ids.is_empty() {
            return Err(AppError::invalid_input(
                "Cannot decode empty token ids".to_string(),
            ));
        }

        if ids
            .iter()
            .any(|&id| id >= self.tokenizer.get_vocab_size(true) as u32)
        {
            return Err(AppError::tokenization_error(format!(
                "Invalid token id found: one or more ids exceed vocabulary size ({}). \
                This may indicate corrupted or incompatible token ids.",
                self.tokenizer.get_vocab_size(true)
            )));
        }

        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| {
                AppError::tokenization_error(format!(
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
}

#[derive(Debug, Clone)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens
    }

    pub fn get_word_ids(&self) -> Option<&[u32]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_local_tokenizer_path() -> String {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dev".to_string());
        format!(
            "{}/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json",
            home
        )
    }

    #[test]
    fn test_tokenizer_creation() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_tokenizer_encode_english() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file_with_max_length(&path, 512).unwrap();

        let encoding = tokenizer.encode("Hello world", true).unwrap();
        assert!(!encoding.is_empty());
        assert_eq!(encoding.ids[0], 101);
    }

    #[test]
    fn test_tokenizer_encode_chinese() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file_with_max_length(&path, 512).unwrap();

        let encoding = tokenizer.encode("你好世界", false).unwrap();
        assert!(!encoding.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_mixed() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file_with_max_length(&path, 512).unwrap();

        let encoding = tokenizer.encode("Hello 你好 world 世界", false).unwrap();
        assert!(!encoding.is_empty());
        assert!(!encoding.is_empty());
    }

    #[test]
    fn test_tokenizer_decode() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file_with_max_length(&path, 512).unwrap();

        let encoding = tokenizer.encode("Hello world", false).unwrap();
        let decoded = tokenizer.decode(&encoding.ids, true);
        assert!(decoded.is_ok());
    }

    #[test]
    fn test_tokenizer_max_length_limit() {
        let path = get_local_tokenizer_path();
        let short_max_length: usize = 10;
        let tokenizer = Tokenizer::from_file_with_max_length(&path, short_max_length).unwrap();

        let long_text = "This is a very long text that should be truncated ".repeat(100);
        let encoding = tokenizer.encode(&long_text, false).unwrap();
        assert!(encoding.len() <= short_max_length);
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let vocab_size = tokenizer.get_vocab_size();
        assert!(vocab_size > 0);
    }

    #[test]
    fn test_encoding_methods() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file_with_max_length(&path, 512).unwrap();

        let encoding = tokenizer.encode("Hello world", false).unwrap();

        assert_eq!(encoding.len(), encoding.ids.len());
        assert!(!encoding.is_empty());
        assert!(!encoding.get_ids().is_empty());
        assert!(!encoding.get_tokens().is_empty());
    }

    #[tokio::test]
    async fn test_cached_tokenizer_basic() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(tokenizer.tokenizer, 512);

        let encoding1 = cached.encode("Hello world", true).await.unwrap();
        assert!(!encoding1.is_empty());

        let stats = cached.get_cache_stats().await;
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_cache_hit() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(tokenizer.tokenizer, 512);

        let text = "Cache test string";
        let encoding1 = cached.encode(text, true).await.unwrap();
        let encoding2 = cached.encode(text, true).await.unwrap();

        assert_eq!(encoding1.ids, encoding2.ids);

        let stats = cached.get_cache_stats().await;
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_different_params() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(tokenizer.tokenizer, 512);

        let text = "Test text";
        let _encoding1 = cached.encode(text, true).await.unwrap();
        let _encoding2 = cached.encode(text, false).await.unwrap();

        let stats = cached.get_cache_stats().await;
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_clear_cache() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(tokenizer.tokenizer, 512);

        let _ = cached.encode("Test", true).await.unwrap();
        let stats_before = cached.get_cache_stats().await;
        assert!(stats_before.size > 0);

        cached.clear_cache().await;
        let stats_after = cached.get_cache_stats().await;
        assert_eq!(stats_after.size, 0);
    }

    #[tokio::test]
    async fn test_cached_tokenizer_batch_encoding() {
        let path = get_local_tokenizer_path();
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let cached = CachedTokenizer::with_default_cache(tokenizer.tokenizer, 512);

        let texts = vec!["First text", "Second text", "Third text"];
        let encodings = cached.encode_batch(&texts, true).await.unwrap();

        assert_eq!(encodings.len(), 3);
        assert!(!encodings.is_empty());
    }

    #[test]
    fn test_utf8_validation_valid_ascii() {
        let result = validate_utf8("Hello world");
        assert!(result.is_valid);
    }

    #[test]
    fn test_utf8_validation_valid_chinese() {
        let result = validate_utf8("你好世界");
        assert!(result.is_valid);
    }

    #[test]
    fn test_utf8_validation_valid_mixed() {
        let result = validate_utf8("Hello 你好 world 世界 🌍");
        assert!(result.is_valid);
    }

    #[test]
    fn test_utf8_validation_invalid_continuation() {
        let invalid = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xC0, 0x80]; // Invalid continuation byte
        let result = validate_utf8_bytes(&invalid);
        assert!(!result.is_valid);
        assert!(result.invalid_byte_position.is_some());
    }

    #[test]
    fn test_utf8_validation_incomplete_sequence() {
        let incomplete = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xE4]; // Incomplete 3-byte sequence
        let result = validate_utf8_bytes(&incomplete);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_utf8_validation_invalid_lead() {
        let invalid = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xFF]; // Invalid lead byte
        let result = validate_utf8_bytes(&invalid);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_utf8_validation_bytes_valid() {
        let bytes = "Hello 你好".as_bytes();
        let result = validate_utf8_bytes(bytes);
        assert!(result.is_valid);
    }

    #[test]
    fn test_utf8_validation_bytes_invalid() {
        let invalid_bytes = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F, 0xFF, 0x00];
        let result = validate_utf8_bytes(&invalid_bytes);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_utf8_validation_result_factory_methods() {
        let valid = Utf8ValidationResult::valid();
        assert!(valid.is_valid);
        assert!(valid.invalid_byte_position.is_none());

        let invalid = Utf8ValidationResult::invalid(5, 0xFF, "test error");
        assert!(!invalid.is_valid);
        assert_eq!(invalid.invalid_byte_position, Some(5));
        assert_eq!(invalid.invalid_byte_value, Some(0xFF));
        assert!(invalid.error_message.is_some());
    }
}
