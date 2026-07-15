// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::error::VecboostError;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
#[cfg(target_os = "macos")]
use tokenizers::Encoding as HfEncoding;
#[cfg(target_os = "macos")]
use tokenizers::Tokenizer as HfTokenizer;

#[cfg(not(target_os = "macos"))]
type HfTokenizer = Tokenizer;

use oxcache::backend::MokaMemoryBackend;
use oxcache::cache::Cache;
use tokio::sync::Mutex;
use xxhash_rust::xxh3::Xxh3;

pub const DEFAULT_CACHE_SIZE: usize = 1024;
pub const MAX_CACHE_SIZE: usize = 8192;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    max_length: usize,
    special_tokens: HashMap<String, u32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct CachedTokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
    cache: Cache<String, Encoding>,
    stats: Mutex<CacheHitStats>,
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
#[derive(Debug)]
pub struct CachedTokenizer {
    tokenizer: HfTokenizer,
    max_length: usize,
    cache: Cache<String, Encoding>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
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

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens
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

#[cfg(target_os = "macos")]
impl Tokenizer {
    pub fn from_pretrained(model_id: &str) -> Result<Self, VecboostError> {
        Self::from_pretrained_with_max_length(model_id, 512)
    }

    pub fn from_pretrained_with_max_length(
        model_id: &str,
        max_length: usize,
    ) -> Result<Self, VecboostError> {
        HfTokenizer::from_pretrained(model_id, None)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to load tokenizer from model '{}': {}. \
                Please check that the model ID is correct and the model is available. \
                For local models, ensure the tokenizer.json file exists.",
                    model_id, e
                ))
            })
            .and_then(|tokenizer| {
                if max_length == 0 {
                    Err(VecboostError::invalid_input(format!(
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

    pub fn from_file(path: &str) -> Result<Self, VecboostError> {
        Self::from_file_with_max_length(path, 512)
    }

    pub fn from_file_with_max_length(path: &str, max_length: usize) -> Result<Self, VecboostError> {
        HfTokenizer::from_file(path)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to load tokenizer from file '{}': {}. \
                Please ensure the file exists and is a valid tokenizer file.",
                    path, e
                ))
            })
            .and_then(|tokenizer| {
                if max_length == 0 {
                    Err(VecboostError::invalid_input(format!(
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

#[cfg(not(target_os = "macos"))]
impl Tokenizer {
    pub fn from_pretrained(_model_id: &str) -> Result<Self, VecboostError> {
        Self::new(512)
    }

    pub fn from_pretrained_with_max_length(
        _model_id: &str,
        max_length: usize,
    ) -> Result<Self, VecboostError> {
        Self::new(max_length)
    }

    pub fn from_file(_path: &str) -> Result<Self, VecboostError> {
        Self::new(512)
    }

    pub fn from_file_with_max_length(
        _path: &str,
        max_length: usize,
    ) -> Result<Self, VecboostError> {
        Self::new(max_length)
    }

    pub fn new(max_length: usize) -> Result<Self, VecboostError> {
        let mut vocab = HashMap::new();
        let mut special_tokens = HashMap::new();

        let common_words = [
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "is",
            "are",
            "was",
            "were",
            "been",
            "being",
            "has",
            "had",
            "having",
            "does",
            "did",
            "doing",
            "can",
            "could",
            "should",
            "would",
            "will",
            "shall",
            "may",
            "might",
            "must",
            "need",
            "ought",
            "used",
            "get",
            "got",
            "getting",
            "make",
            "made",
            "making",
            "go",
            "going",
            "went",
            "gone",
            "come",
            "coming",
            "came",
            "see",
            "seeing",
            "saw",
            "know",
            "knowing",
            "knew",
            "known",
            "think",
            "thinking",
            "thought",
            "take",
            "taking",
            "find",
            "finding",
            "found",
            "give",
            "giving",
            "gave",
            "tell",
            "telling",
            "told",
            "become",
            "becoming",
            "became",
            "leave",
            "leaving",
            "left",
            "put",
            "putting",
            "mean",
            "meaning",
            "meant",
            "keep",
            "keeping",
            "kept",
            "let",
            "letting",
            "begin",
            "beginning",
            "began",
            "seem",
            "seeming",
            "seemed",
            "help",
            "helping",
            "helped",
            "show",
            "showing",
            "showed",
            "hear",
            "hearing",
            "heard",
            "play",
            "playing",
            "played",
            "run",
            "running",
            "ran",
            "move",
            "moving",
            "moved",
            "like",
            "live",
            "believe",
            "hold",
            "holding",
            "held",
            "bring",
            "bringing",
            "brought",
            "happen",
            "happening",
            "happened",
            "write",
            "writing",
            "wrote",
            "provide",
            "providing",
            "provided",
            "sit",
            "sitting",
            "sat",
            "stand",
            "standing",
            "stood",
            "lose",
            "losing",
            "lost",
            "pay",
            "paying",
            "paid",
            "meet",
            "meeting",
            "met",
            "include",
            "including",
            "included",
            "continue",
            "continuing",
            "continued",
            "set",
            "setting",
            "learn",
            "learning",
            "learned",
            "change",
            "changing",
            "changed",
            "lead",
            "leading",
            "led",
            "understand",
            "understanding",
            "understood",
            "watch",
            "watching",
            "watched",
            "follow",
            "following",
            "followed",
            "stop",
            "stopping",
            "stopped",
            "create",
            "creating",
            "created",
            "speak",
            "speaking",
            "spoke",
            "read",
            "allow",
            "adding",
            "added",
            "add",
            "open",
            "opening",
            "opened",
            "walk",
            "walking",
            "walked",
            "win",
            "winning",
            "won",
            "offer",
            "offering",
            "offered",
            "remember",
            "remembering",
            "remembered",
            "love",
            "consider",
            "appear",
            "appearing",
            "appeared",
            "buy",
            "buying",
            "bought",
            "wait",
            "waiting",
            "waited",
            "serve",
            "serving",
            "served",
            "die",
            "dying",
            "died",
            "send",
            "sending",
            "sent",
            "expect",
            "expecting",
            "expected",
            "build",
            "building",
            "built",
            "stay",
            "staying",
            "stayed",
            "fall",
            "falling",
            "fell",
            "fallen",
            "cut",
            "cutting",
            "reach",
            "reaching",
            "reached",
            "kill",
            "killing",
            "killed",
            "remain",
            "remaining",
            "remained",
            "really",
            "now",
            "even",
            "new",
            "newer",
            "newest",
            "good",
            "better",
            "best",
            "great",
            "greater",
            "greatest",
            "high",
            "higher",
            "highest",
            "little",
            "smaller",
            "smallest",
            "long",
            "longer",
            "longest",
            "next",
            "first",
            "last",
            "young",
            "younger",
            "youngest",
            "important",
            "few",
            "fewer",
            "fewest",
            "big",
            "bigger",
            "biggest",
            "different",
            "small",
            "large",
            "large",
            "larger",
            "largest",
            "right",
            "wrong",
            "true",
            "false",
            "real",
            "best",
            "better",
            "well",
            "also",
            "just",
            "still",
            "back",
            "much",
            "many",
            "way",
            "well",
            "even",
            "then",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "able",
            "over",
            "after",
            "first",
            "two",
            "most",
            "made",
            "made",
            "made",
            "thing",
            "things",
            "every",
            "where",
            "when",
            "who",
            "what",
            "how",
            "which",
        ];

        for (id, word) in common_words.iter().enumerate() {
            vocab.insert(word.to_string(), id as u32);
        }

        special_tokens.insert("[PAD]".to_string(), vocab.len() as u32);
        special_tokens.insert("[UNK]".to_string(), (vocab.len() + 1) as u32);
        special_tokens.insert("[CLS]".to_string(), (vocab.len() + 2) as u32);
        special_tokens.insert("[SEP]".to_string(), (vocab.len() + 3) as u32);
        special_tokens.insert("[MASK]".to_string(), (vocab.len() + 4) as u32);

        Ok(Self {
            vocab,
            max_length,
            special_tokens,
        })
    }

    fn wordpiece_tokenize(&self, text: &str) -> Vec<String> {
        let re = Regex::new(r"\w+|[^\w\s]+").unwrap();
        re.find_iter(text)
            .map(|m| m.as_str().to_lowercase())
            .collect()
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
                utf8_result.invalid_byte_position.unwrap_or(0),
                utf8_result
                    .error_message
                    .unwrap_or_else(|| "Unknown UTF-8 error".to_string())
            )));
        }

        let mut tokens = Vec::new();
        let mut ids = Vec::new();
        let mut attention_mask = Vec::new();
        let mut type_ids = Vec::new();

        if add_special_tokens && let Some(&id) = self.special_tokens.get("[CLS]") {
            tokens.push("[CLS]".to_string());
            ids.push(id);
            attention_mask.push(1);
            type_ids.push(0);
        }

        let words = self.wordpiece_tokenize(text);
        let unknown_id = self.special_tokens.get("[UNK]").copied().unwrap_or(0);

        for word in words {
            if self.vocab.contains_key(&word) {
                if let Some(&id) = self.vocab.get(&word) {
                    tokens.push(word.clone());
                    ids.push(id);
                    attention_mask.push(1);
                    type_ids.push(0);
                }
            } else {
                tokens.push("[UNK]".to_string());
                ids.push(unknown_id);
                attention_mask.push(1);
                type_ids.push(0);
            }
        }

        if add_special_tokens && let Some(&id) = self.special_tokens.get("[SEP]") {
            tokens.push("[SEP]".to_string());
            ids.push(id);
            attention_mask.push(1);
            type_ids.push(0);
        }

        let truncated_ids: Vec<u32> = if ids.len() > self.max_length {
            ids[..self.max_length].to_vec()
        } else {
            ids
        };

        let truncated_mask: Vec<u32> = if attention_mask.len() > self.max_length {
            attention_mask[..self.max_length].to_vec()
        } else {
            attention_mask
        };

        let truncated_type_ids: Vec<u32> = if type_ids.len() > self.max_length {
            type_ids[..self.max_length].to_vec()
        } else {
            type_ids
        };

        let truncated_tokens: Vec<String> = if tokens.len() > self.max_length {
            tokens[..self.max_length].to_vec()
        } else {
            tokens
        };

        Ok(Encoding {
            ids: truncated_ids,
            attention_mask: truncated_mask,
            type_ids: truncated_type_ids,
            tokens: truncated_tokens,
        })
    }

    pub fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String, VecboostError> {
        if ids.is_empty() {
            return Err(VecboostError::invalid_input(
                "Cannot decode empty token ids".to_string(),
            ));
        }

        let mut result = Vec::new();
        for id in ids {
            let mut found = false;
            for (word, &word_id) in &self.vocab {
                if word_id == *id {
                    result.push(word.clone());
                    found = true;
                    break;
                }
            }
            if !found {
                for (token, &token_id) in &self.special_tokens {
                    if token_id == *id {
                        result.push(token.clone());
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                result.push(format!("[UNK:{}]", id));
            }
        }

        Ok(result.join(" "))
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len() + self.special_tokens.len()
    }

    pub fn get_max_length(&self) -> usize {
        self.max_length
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

        let mut encodings = Vec::with_capacity(texts.len());
        for &text in texts {
            let encoding = self.encode(text, add_special_tokens)?;
            encodings.push(encoding);
        }

        Ok(encodings)
    }
}

#[cfg(target_os = "macos")]
impl CachedTokenizer {
    pub fn new(tokenizer: HfTokenizer, max_length: usize, cache_size: usize) -> Self {
        let capacity = cache_size.clamp(1, MAX_CACHE_SIZE) as u64;
        let moka = MokaMemoryBackend::builder().capacity(capacity).build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        Self {
            tokenizer,
            max_length,
            cache,
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

    pub async fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, VecboostError> {
        let key = self.hash_key(text, add_special_tokens);

        if let Some(cached) = self.cache.get(&key).await.ok().flatten() {
            let mut stats = self.stats.lock().await;
            stats.hits += 1;
            return Ok(cached);
        }

        let encoding = self.encode_uncached(text, add_special_tokens).await?;

        let mut stats = self.stats.lock().await;
        stats.misses += 1;
        let _ = self.cache.set(&key, &encoding).await;

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
}

#[cfg(not(target_os = "macos"))]
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

    pub async fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, VecboostError> {
        let key = self.hash_key(text, add_special_tokens);

        if let Some(cached) = self.cache.get(&key).await.ok().flatten() {
            let mut stats = self.stats.lock().await;
            stats.hits += 1;
            return Ok(cached);
        }

        let encoding = self.encode_uncached(text, add_special_tokens).await?;

        let mut stats = self.stats.lock().await;
        stats.misses += 1;
        let _ = self.cache.set(&key, &encoding).await;

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

        self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| {
                VecboostError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}. \
                The text may contain unsupported characters or be too long.",
                    text.len(),
                    e
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    #![allow(invalid_from_utf8_unchecked)]
    use super::*;

    #[test]
    fn test_encoding_getters_return_correct_slices() {
        let encoding = Encoding {
            ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            type_ids: vec![0, 0, 0],
            tokens: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };
        assert_eq!(encoding.get_ids(), &[1, 2, 3]);
        assert_eq!(encoding.get_attention_mask(), &[1, 1, 1]);
        assert_eq!(encoding.get_type_ids(), &[0, 0, 0]);
        assert_eq!(
            encoding.get_tokens(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_encoding_getters_on_empty() {
        let encoding = Encoding {
            ids: vec![],
            attention_mask: vec![],
            type_ids: vec![],
            tokens: vec![],
        };
        assert!(encoding.get_ids().is_empty());
        assert!(encoding.get_attention_mask().is_empty());
        assert!(encoding.get_type_ids().is_empty());
        assert!(encoding.get_tokens().is_empty());
    }

    #[test]
    fn test_encoding_eq_clone_debug() {
        let encoding = Encoding {
            ids: vec![1],
            attention_mask: vec![1],
            type_ids: vec![0],
            tokens: vec!["x".to_string()],
        };
        let cloned = encoding.clone();
        assert_eq!(encoding, cloned);
        let debug_str = format!("{:?}", encoding);
        assert!(debug_str.contains("Encoding"));
    }

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
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert_eq!(r.invalid_byte_value, Some(0x80));
        assert!(r.error_message.as_deref().unwrap().contains("lead byte"));
    }

    #[test]
    fn test_validate_utf8_invalid_lead_byte_0xc0() {
        let bytes = [0xC0u8];
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(0));
        assert_eq!(r.invalid_byte_value, Some(0xC0));
    }

    #[test]
    fn test_validate_utf8_invalid_lead_byte_0xff() {
        let bytes = [0xFFu8];
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_value, Some(0xFF));
    }

    #[test]
    fn test_validate_utf8_incomplete_two_byte_sequence() {
        let bytes = [0xC2u8];
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
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
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
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
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
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
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
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
        let s = unsafe { std::str::from_utf8_unchecked(&bytes) };
        let r = validate_utf8(s);
        assert!(!r.is_valid);
        assert_eq!(r.invalid_byte_position, Some(3));
        assert_eq!(r.invalid_byte_value, Some(0x00));
    }

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
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_stats_struct() {
        let stats = CacheStats {
            hits: 5,
            misses: 3,
            size: 2,
            capacity: 1024,
        };
        assert_eq!(stats.hits, 5);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.size, 2);
        assert_eq!(stats.capacity, 1024);
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[cfg(not(target_os = "macos"))]
    mod non_macos {
        use super::*;

        fn make_tokenizer() -> Tokenizer {
            Tokenizer::new(512).expect("Failed to create tokenizer")
        }

        #[test]
        fn test_tokenizer_new_success() {
            let tokenizer = Tokenizer::new(512).unwrap();
            assert_eq!(tokenizer.get_max_length(), 512);
            assert!(tokenizer.get_vocab_size() > 0);
        }

        #[test]
        fn test_tokenizer_new_populates_vocab_and_specials() {
            let tokenizer = Tokenizer::new(64).unwrap();
            assert!(tokenizer.vocab.contains_key("the"));
            assert!(tokenizer.vocab.contains_key("be"));
            assert!(tokenizer.special_tokens.contains_key("[CLS]"));
            assert!(tokenizer.special_tokens.contains_key("[SEP]"));
            assert!(tokenizer.special_tokens.contains_key("[PAD]"));
            assert!(tokenizer.special_tokens.contains_key("[UNK]"));
            assert_eq!(
                tokenizer.get_vocab_size(),
                tokenizer.vocab.len() + tokenizer.special_tokens.len()
            );
        }

        #[test]
        fn test_tokenizer_from_pretrained_default_max_length() {
            let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
            assert_eq!(tokenizer.get_max_length(), 512);
        }

        #[test]
        fn test_tokenizer_from_pretrained_with_max_length() {
            let tokenizer = Tokenizer::from_pretrained_with_max_length("any-model", 128).unwrap();
            assert_eq!(tokenizer.get_max_length(), 128);
        }

        #[test]
        fn test_tokenizer_from_file_default_max_length() {
            let tokenizer = Tokenizer::from_file("/nonexistent/path.json").unwrap();
            assert_eq!(tokenizer.get_max_length(), 512);
        }

        #[test]
        fn test_tokenizer_from_file_with_max_length() {
            let tokenizer =
                Tokenizer::from_file_with_max_length("/nonexistent/path.json", 256).unwrap();
            assert_eq!(tokenizer.get_max_length(), 256);
        }

        #[test]
        fn test_tokenizer_get_max_length() {
            let tokenizer = Tokenizer::new(42).unwrap();
            assert_eq!(tokenizer.get_max_length(), 42);
        }

        #[test]
        fn test_tokenizer_get_vocab_size_includes_specials() {
            let tokenizer = make_tokenizer();
            let vocab_only = tokenizer.vocab.len();
            let specials_only = tokenizer.special_tokens.len();
            assert_eq!(tokenizer.get_vocab_size(), vocab_only + specials_only);
        }

        #[test]
        fn test_tokenizer_encode_empty_text_returns_error() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.encode("", true);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("empty text"));
        }

        #[test]
        fn test_tokenizer_encode_known_words_without_special_tokens() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("the be to", false).unwrap();
            assert_eq!(encoding.tokens, vec!["the", "be", "to"]);
            assert_eq!(encoding.ids.len(), 3);
            assert_eq!(encoding.attention_mask, vec![1, 1, 1]);
            assert_eq!(encoding.type_ids, vec![0, 0, 0]);
        }

        #[test]
        fn test_tokenizer_encode_known_words_ids_match_vocab() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("the be to", false).unwrap();
            assert_eq!(encoding.ids[0], *tokenizer.vocab.get("the").unwrap());
            assert_eq!(encoding.ids[1], *tokenizer.vocab.get("be").unwrap());
            assert_eq!(encoding.ids[2], *tokenizer.vocab.get("to").unwrap());
        }

        #[test]
        fn test_tokenizer_encode_with_special_tokens_wraps_cls_sep() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("the be", true).unwrap();
            assert_eq!(encoding.tokens.len(), 4);
            assert_eq!(encoding.tokens[0], "[CLS]");
            assert_eq!(encoding.tokens[1], "the");
            assert_eq!(encoding.tokens[2], "be");
            assert_eq!(encoding.tokens[3], "[SEP]");
            assert_eq!(
                encoding.ids[0],
                *tokenizer.special_tokens.get("[CLS]").unwrap()
            );
            assert_eq!(
                encoding.ids[3],
                *tokenizer.special_tokens.get("[SEP]").unwrap()
            );
            assert_eq!(encoding.attention_mask, vec![1, 1, 1, 1]);
        }

        #[test]
        fn test_tokenizer_encode_unknown_word_becomes_unk() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("xyzqwerty", false).unwrap();
            assert_eq!(encoding.tokens, vec!["[UNK]"]);
            assert_eq!(
                encoding.ids,
                vec![*tokenizer.special_tokens.get("[UNK]").unwrap()]
            );
        }

        #[test]
        fn test_tokenizer_encode_mixed_known_unknown() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("the zzzz be", false).unwrap();
            assert_eq!(encoding.tokens, vec!["the", "[UNK]", "be"]);
            assert_eq!(encoding.ids.len(), 3);
        }

        #[test]
        fn test_tokenizer_encode_punctuation_tokens() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("!!!", false).unwrap();
            assert_eq!(encoding.tokens.len(), 1);
            assert_eq!(encoding.tokens[0], "[UNK]");
        }

        #[test]
        fn test_tokenizer_encode_lowercase_normalization() {
            let tokenizer = make_tokenizer();
            let upper = tokenizer.encode("THE", false).unwrap();
            let lower = tokenizer.encode("the", false).unwrap();
            assert_eq!(upper.ids, lower.ids);
            assert_eq!(upper.tokens, lower.tokens);
        }

        #[test]
        fn test_tokenizer_encode_truncation_applies_to_all_fields() {
            let tokenizer = Tokenizer::new(3).unwrap();
            let encoding = tokenizer.encode("the be to of and", true).unwrap();
            assert_eq!(encoding.ids.len(), 3);
            assert_eq!(encoding.attention_mask.len(), 3);
            assert_eq!(encoding.type_ids.len(), 3);
            assert_eq!(encoding.tokens.len(), 3);
            assert_eq!(encoding.tokens[0], "[CLS]");
        }

        #[test]
        fn test_tokenizer_encode_truncation_without_special_tokens() {
            let tokenizer = Tokenizer::new(2).unwrap();
            let encoding = tokenizer.encode("the be to of and", false).unwrap();
            assert_eq!(encoding.ids.len(), 2);
            assert_eq!(encoding.tokens.len(), 2);
            assert_eq!(encoding.tokens, vec!["the", "be"]);
        }

        #[test]
        fn test_tokenizer_encode_truncation_when_text_fits() {
            let tokenizer = Tokenizer::new(100).unwrap();
            let encoding = tokenizer.encode("the be", true).unwrap();
            assert_eq!(encoding.tokens.len(), 4);
            assert_eq!(encoding.ids.len(), 4);
        }

        #[test]
        fn test_tokenizer_decode_empty_ids_returns_error() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.decode(&[], true);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("empty token ids"));
        }

        #[test]
        fn test_tokenizer_decode_known_id() {
            let tokenizer = make_tokenizer();
            let the_id = *tokenizer.vocab.get("the").unwrap();
            let result = tokenizer.decode(&[the_id], false).unwrap();
            assert_eq!(result, "the");
        }

        #[test]
        fn test_tokenizer_decode_multiple_known_ids() {
            let tokenizer = make_tokenizer();
            let the_id = *tokenizer.vocab.get("the").unwrap();
            let be_id = *tokenizer.vocab.get("be").unwrap();
            let to_id = *tokenizer.vocab.get("to").unwrap();
            let result = tokenizer.decode(&[the_id, be_id, to_id], false).unwrap();
            assert_eq!(result, "the be to");
        }

        #[test]
        fn test_tokenizer_decode_special_token_id() {
            let tokenizer = make_tokenizer();
            let cls_id = *tokenizer.special_tokens.get("[CLS]").unwrap();
            let result = tokenizer.decode(&[cls_id], false).unwrap();
            assert_eq!(result, "[CLS]");
        }

        #[test]
        fn test_tokenizer_decode_unknown_id_returns_unk_placeholder() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.decode(&[u32::MAX], false).unwrap();
            assert_eq!(result, format!("[UNK:{}]", u32::MAX));
        }

        #[test]
        fn test_tokenizer_decode_mixed_known_special_unknown() {
            let tokenizer = make_tokenizer();
            let the_id = *tokenizer.vocab.get("the").unwrap();
            let cls_id = *tokenizer.special_tokens.get("[CLS]").unwrap();
            let result = tokenizer
                .decode(&[cls_id, the_id, 99_999_999], false)
                .unwrap();
            assert_eq!(result, format!("[CLS] the [UNK:{}]", 99_999_999));
        }

        #[test]
        fn test_tokenizer_decode_skip_special_tokens_param_does_not_affect_output() {
            let tokenizer = make_tokenizer();
            let cls_id = *tokenizer.special_tokens.get("[CLS]").unwrap();
            let with_skip = tokenizer.decode(&[cls_id], true).unwrap();
            let without_skip = tokenizer.decode(&[cls_id], false).unwrap();
            assert_eq!(with_skip, without_skip);
            assert_eq!(with_skip, "[CLS]");
        }

        #[test]
        fn test_tokenizer_encode_batch_empty_returns_empty_vec() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.encode_batch(&[], true).unwrap();
            assert!(result.is_empty());
        }

        #[test]
        fn test_tokenizer_encode_batch_with_empty_text_at_index_returns_error() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.encode_batch(&["the", "", "be"], false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("batch index 1"));
        }

        #[test]
        fn test_tokenizer_encode_batch_success() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.encode_batch(&["the be", "to of"], true).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].tokens, vec!["[CLS]", "the", "be", "[SEP]"]);
            assert_eq!(result[1].tokens, vec!["[CLS]", "to", "of", "[SEP]"]);
        }

        #[test]
        fn test_tokenizer_encode_batch_without_special_tokens() {
            let tokenizer = make_tokenizer();
            let result = tokenizer.encode_batch(&["the be", "to of"], false).unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].tokens, vec!["the", "be"]);
            assert_eq!(result[1].tokens, vec!["to", "of"]);
        }

        #[test]
        fn test_tokenizer_encode_batch_propagates_truncation() {
            let tokenizer = Tokenizer::new(2).unwrap();
            let result = tokenizer
                .encode_batch(&["the be to of and"], false)
                .unwrap();
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].tokens.len(), 2);
        }

        #[test]
        fn test_tokenizer_encode_batch_invalid_utf8_returns_error() {
            let tokenizer = make_tokenizer();
            let bytes = [0xC2u8];
            let bad = unsafe { std::str::from_utf8_unchecked(&bytes) };
            let result = tokenizer.encode_batch(&["the", bad], false);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("batch index 1"));
        }

        #[test]
        fn test_tokenizer_clone_is_equal() {
            let tokenizer = make_tokenizer();
            let cloned = tokenizer.clone();
            assert_eq!(tokenizer.get_max_length(), cloned.get_max_length());
            assert_eq!(tokenizer.get_vocab_size(), cloned.get_vocab_size());
        }

        #[test]
        fn test_tokenizer_wordpiece_tokenize_via_encode_mixed_alphanum_and_punct() {
            let tokenizer = make_tokenizer();
            let encoding = tokenizer.encode("the, be.", false).unwrap();
            assert_eq!(encoding.tokens.len(), 4);
            assert_eq!(encoding.tokens[0], "the");
            assert_eq!(encoding.tokens[1], "[UNK]");
            assert_eq!(encoding.tokens[2], "be");
            assert_eq!(encoding.tokens[3], "[UNK]");
        }

        #[tokio::test]
        async fn test_cached_tokenizer_new_max_length_preserved() {
            let tokenizer = Tokenizer::new(256).unwrap();
            let cached = CachedTokenizer::new(tokenizer, 256, 64);
            assert_eq!(cached.max_length(), 256);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_with_default_cache() {
            let tokenizer = Tokenizer::new(128).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 128);
            assert_eq!(cached.max_length(), 128);
            // oxcache::Cache 不暴露 capacity getter,验证初始 len 为 0
            assert_eq!(cached.cache.len().await.unwrap_or(0), 0);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_cache_size_clamped_to_max() {
            let tokenizer = Tokenizer::new(128).unwrap();
            let _cached = CachedTokenizer::new(tokenizer, 128, 100_000);
            // oxcache::Cache 通过 MokaMemoryBackend 配置 capacity,
            // 运行时不暴露 cap getter。此处仅验证构造不 panic。
        }

        #[tokio::test]
        async fn test_cached_tokenizer_cache_size_zero_becomes_one() {
            let tokenizer = Tokenizer::new(128).unwrap();
            let cached = CachedTokenizer::new(tokenizer, 128, 0);
            // oxcache::Cache 不暴露 cap getter,验证初始 len 为 0
            assert_eq!(cached.cache.len().await.unwrap_or(0), 0);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_cache_miss_then_hit_stats() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);

            let first = cached.encode("the be to", true).await.unwrap();
            {
                let stats = cached.stats.lock().await;
                assert_eq!(stats.hits, 0);
                assert_eq!(stats.misses, 1);
            }

            let second = cached.encode("the be to", true).await.unwrap();
            {
                let stats = cached.stats.lock().await;
                assert_eq!(stats.hits, 1);
                assert_eq!(stats.misses, 1);
            }
            assert_eq!(first, second);

            let third = cached.encode("the be to", true).await.unwrap();
            {
                let stats = cached.stats.lock().await;
                assert_eq!(stats.hits, 2);
                assert_eq!(stats.misses, 1);
            }
            assert_eq!(third, first);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_different_special_tokens_not_shared() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);

            let _ = cached.encode("the", false).await.unwrap();
            let _ = cached.encode("the", true).await.unwrap();

            let stats = cached.stats.lock().await;
            assert_eq!(stats.hits, 0);
            assert_eq!(stats.misses, 2);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_different_text_separate_entries() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);

            let _ = cached.encode("the", false).await.unwrap();
            let _ = cached.encode("be", false).await.unwrap();
            let _ = cached.encode("the", false).await.unwrap();

            let stats = cached.stats.lock().await;
            assert_eq!(stats.hits, 1);
            assert_eq!(stats.misses, 2);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_empty_text_returns_error() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let result = cached.encode("", true).await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("empty text"));
            let stats = cached.stats.lock().await;
            assert_eq!(stats.hits, 0);
            assert_eq!(stats.misses, 0);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_invalid_utf8_returns_error() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let bytes = [0xC2u8];
            let bad = unsafe { std::str::from_utf8_unchecked(&bytes) };
            let result = cached.encode(bad, false).await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, crate::error::VecboostError::InvalidInput(_)));
            assert!(err.to_string().contains("UTF-8"));
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_returns_correct_encoding_content() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let encoding = cached.encode("the be", true).await.unwrap();
            assert_eq!(encoding.tokens, vec!["[CLS]", "the", "be", "[SEP]"]);
            assert_eq!(encoding.attention_mask, vec![1, 1, 1, 1]);
            assert_eq!(encoding.type_ids, vec![0, 0, 0, 0]);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_encode_failure_does_not_consume_cache_slot() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);

            let _ = cached.encode("", true).await;
            let cache_size_after_failure = cached.cache.len().await.unwrap_or(0);
            assert_eq!(cache_size_after_failure, 0);
            let stats = cached.stats.lock().await;
            assert_eq!(stats.misses, 0);
            assert_eq!(stats.hits, 0);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_eviction_under_capacity_pressure() {
            // oxcache::Cache 基于 moka W-TinyLFU,异步驱逐,不保证严格 LRU。
            // 验证容量压力下最终会有驱逐(而非全部驻留)。
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::new(tokenizer, 512, 2);

            // 插入超量 key 触发驱逐
            for word in ["a", "b", "c", "d", "e", "f", "g", "h"] {
                let _ = cached.encode(word, false).await.unwrap();
            }

            // 等待 moka 异步驱逐完成
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;

            let len = cached.cache.len().await.unwrap_or(0);
            assert!(
                len <= 8,
                "cache should evict some entries under capacity pressure, got len={}",
                len
            );
            let stats = cached.stats.lock().await;
            assert_eq!(stats.misses, 8);
            assert_eq!(stats.hits, 0);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_hash_key_differs_by_special_tokens() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let key_false = cached.hash_key("the", false);
            let key_true = cached.hash_key("the", true);
            assert_ne!(key_false, key_true);
            assert!(key_false.ends_with("_false"));
            assert!(key_true.ends_with("_true"));
        }

        #[tokio::test]
        async fn test_cached_tokenizer_hash_key_differs_by_text() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let key_a = cached.hash_key("the", false);
            let key_b = cached.hash_key("be", false);
            assert_ne!(key_a, key_b);
        }

        #[tokio::test]
        async fn test_cached_tokenizer_hash_key_same_input_same_hash() {
            let tokenizer = Tokenizer::new(512).unwrap();
            let cached = CachedTokenizer::with_default_cache(tokenizer, 512);
            let k1 = cached.hash_key("the be to", true);
            let k2 = cached.hash_key("the be to", true);
            assert_eq!(k1, k2);
        }
    }
}
