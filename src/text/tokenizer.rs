// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::error::AppError;
use lru::LruCache;
use regex::Regex;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
#[cfg(target_os = "macos")]
use tokenizers::Encoding as HfEncoding;
#[cfg(target_os = "macos")]
use tokenizers::Tokenizer as HfTokenizer;

#[cfg(not(target_os = "macos"))]
type HfTokenizer = Tokenizer;

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
    cache: Mutex<LruCache<String, Encoding>>,
    stats: Mutex<CacheHitStats>,
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[cfg(target_os = "macos")]
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

#[cfg(not(target_os = "macos"))]
impl Tokenizer {
    pub fn from_pretrained(_model_id: &str) -> Result<Self, AppError> {
        Self::new(512)
    }

    pub fn from_pretrained_with_max_length(
        _model_id: &str,
        max_length: usize,
    ) -> Result<Self, AppError> {
        Self::new(max_length)
    }

    pub fn from_file(_path: &str) -> Result<Self, AppError> {
        Self::new(512)
    }

    pub fn from_file_with_max_length(_path: &str, max_length: usize) -> Result<Self, AppError> {
        Self::new(max_length)
    }

    pub fn new(max_length: usize) -> Result<Self, AppError> {
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

    pub fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String, AppError> {
        if ids.is_empty() {
            return Err(AppError::invalid_input(
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
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
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

        self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| {
                AppError::tokenization_error(format!(
                    "Failed to encode text (length={}): {}. \
                The text may contain unsupported characters or be too long.",
                    text.len(),
                    e
                ))
            })
    }
}
