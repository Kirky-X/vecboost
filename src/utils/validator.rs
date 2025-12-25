// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use std::num::NonZeroUsize;

pub use super::constants::{
    MAX_BATCH_SIZE, MAX_CONCURRENT_REQUESTS, MAX_FILE_SIZE_BYTES, MAX_SEARCH_RESULTS,
    MAX_TEXT_LENGTH, MIN_TEXT_LENGTH,
};

#[derive(Debug, Clone, Copy)]
pub struct ValidationConfig {
    pub max_text_length: NonZeroUsize,
    pub min_text_length: usize,
    pub max_batch_size: NonZeroUsize,
    pub max_search_results: NonZeroUsize,
    pub max_concurrent_requests: NonZeroUsize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_text_length: NonZeroUsize::new(MAX_TEXT_LENGTH).unwrap(),
            min_text_length: MIN_TEXT_LENGTH,
            max_batch_size: NonZeroUsize::new(MAX_BATCH_SIZE).unwrap(),
            max_search_results: NonZeroUsize::new(MAX_SEARCH_RESULTS).unwrap(),
            max_concurrent_requests: NonZeroUsize::new(MAX_CONCURRENT_REQUESTS).unwrap(),
        }
    }
}

impl ValidationConfig {
    pub fn new(
        max_text_length: Option<NonZeroUsize>,
        min_text_length: Option<usize>,
        max_batch_size: Option<NonZeroUsize>,
        max_search_results: Option<NonZeroUsize>,
        max_concurrent_requests: Option<NonZeroUsize>,
    ) -> Self {
        Self {
            max_text_length: max_text_length
                .unwrap_or_else(|| NonZeroUsize::new(MAX_TEXT_LENGTH).unwrap()),
            min_text_length: min_text_length.unwrap_or(MIN_TEXT_LENGTH),
            max_batch_size: max_batch_size
                .unwrap_or_else(|| NonZeroUsize::new(MAX_BATCH_SIZE).unwrap()),
            max_search_results: max_search_results
                .unwrap_or_else(|| NonZeroUsize::new(MAX_SEARCH_RESULTS).unwrap()),
            max_concurrent_requests: max_concurrent_requests
                .unwrap_or_else(|| NonZeroUsize::new(MAX_CONCURRENT_REQUESTS).unwrap()),
        }
    }
}

pub trait TextValidator {
    fn validate_text(&self, text: &str) -> Result<(), AppError>;
    fn validate_batch(&self, texts: &[String]) -> Result<(), AppError>;
    fn validate_search(
        &self,
        query: &str,
        texts: &[String],
        top_k: Option<usize>,
    ) -> Result<(), AppError>;
}

pub struct InputValidator {
    config: ValidationConfig,
}

impl InputValidator {
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    pub fn with_default() -> Self {
        Self::new(ValidationConfig::default())
    }

    fn validate_text_content(&self, text: &str) -> Result<(), AppError> {
        if text.is_empty() {
            return Err(AppError::InvalidInput("Text cannot be empty".to_string()));
        }

        let char_count = text.chars().count();
        if char_count < self.config.min_text_length {
            return Err(AppError::InvalidInput(format!(
                "Text too short: {} characters (minimum: {})",
                char_count, self.config.min_text_length
            )));
        }

        if char_count > self.config.max_text_length.get() {
            return Err(AppError::InvalidInput(format!(
                "Text too long: {} characters (maximum: {})",
                char_count,
                self.config.max_text_length.get()
            )));
        }

        if text.trim().is_empty() {
            return Err(AppError::InvalidInput(
                "Text contains only whitespace".to_string(),
            ));
        }

        Ok(())
    }
}

impl TextValidator for InputValidator {
    fn validate_text(&self, text: &str) -> Result<(), AppError> {
        self.validate_text_content(text)
    }

    fn validate_batch(&self, texts: &[String]) -> Result<(), AppError> {
        if texts.is_empty() {
            return Err(AppError::InvalidInput("Batch cannot be empty".to_string()));
        }

        if texts.len() > self.config.max_batch_size.get() {
            return Err(AppError::InvalidInput(format!(
                "Batch size {} exceeds maximum {}",
                texts.len(),
                self.config.max_batch_size.get()
            )));
        }

        for (idx, text) in texts.iter().enumerate() {
            self.validate_text_content(text).map_err(|e| {
                AppError::InvalidInput(format!(
                    "Validation failed for text at index {}: {}",
                    idx, e
                ))
            })?;
        }

        Ok(())
    }

    fn validate_search(
        &self,
        query: &str,
        texts: &[String],
        top_k: Option<usize>,
    ) -> Result<(), AppError> {
        self.validate_text_content(query)?;

        if texts.is_empty() {
            return Err(AppError::InvalidInput(
                "Search texts list cannot be empty".to_string(),
            ));
        }

        if texts.len() > self.config.max_search_results.get() {
            return Err(AppError::InvalidInput(format!(
                "Search results count {} exceeds maximum {}",
                texts.len(),
                self.config.max_search_results.get()
            )));
        }

        if let Some(k) = top_k {
            if k == 0 {
                return Err(AppError::InvalidInput(
                    "top_k must be at least 1".to_string(),
                ));
            }
            if k > self.config.max_search_results.get() {
                return Err(AppError::InvalidInput(format!(
                    "top_k {} exceeds maximum {}",
                    k,
                    self.config.max_search_results.get()
                )));
            }
        }

        for (idx, text) in texts.iter().enumerate() {
            self.validate_text_content(text).map_err(|e| {
                AppError::InvalidInput(format!(
                    "Validation failed for search text at index {}: {}",
                    idx, e
                ))
            })?;
        }

        Ok(())
    }
}

impl InputValidator {
    fn validate_file_size(&self, path: &str) -> Result<(), AppError> {
        use std::fs;

        match fs::metadata(path) {
            Ok(metadata) => {
                let file_size = metadata.len();
                if file_size > MAX_FILE_SIZE_BYTES {
                    let size_mb = file_size as f64 / (1024.0 * 1024.0);
                    let max_mb = MAX_FILE_SIZE_BYTES as f64 / (1024.0 * 1024.0);
                    return Err(AppError::InvalidInput(format!(
                        "File size {:.2} MB exceeds maximum allowed size {:.2} MB",
                        size_mb, max_mb
                    )));
                }
                Ok(())
            }
            Err(e) => Err(AppError::InvalidInput(format!(
                "Cannot access file {}: {}",
                path, e
            ))),
        }
    }
}

pub trait FileValidator {
    fn validate_file(&self, path: &str) -> Result<(), AppError>;
}

impl FileValidator for InputValidator {
    fn validate_file(&self, path: &str) -> Result<(), AppError> {
        self.validate_file_size(path)
    }
}

#[cfg(test)]
mod validator_tests {
    use super::*;

    #[test]
    fn test_empty_text_validation() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_text("").is_err());
    }

    #[test]
    fn test_whitespace_text_validation() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_text("   ").is_err());
    }

    #[test]
    fn test_valid_text_validation() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_text("Hello, world!").is_ok());
    }

    #[test]
    fn test_batch_size_limit() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(1000),
            Some(1),
            NonZeroUsize::new(3),
            NonZeroUsize::new(100),
            NonZeroUsize::new(100),
        );
        let validator = InputValidator::new(config);

        let texts = vec![
            "text1".to_string(),
            "text2".to_string(),
            "text3".to_string(),
            "text4".to_string(),
        ];
        assert!(validator.validate_batch(&texts).is_err());
    }

    #[test]
    fn test_search_validation() {
        let validator = InputValidator::with_default();
        assert!(validator
            .validate_search("query", &["text1".to_string()], Some(5))
            .is_ok());
        assert!(validator
            .validate_search("", &["text1".to_string()], Some(5))
            .is_err());
        assert!(validator
            .validate_search("query", &[], Some(5))
            .is_err());
    }
}
