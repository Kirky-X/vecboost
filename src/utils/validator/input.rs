// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
use crate::utils::constants::{
    MAX_BATCH_SIZE, MAX_CONCURRENT_REQUESTS, MAX_FILE_SIZE_BYTES, MAX_SEARCH_RESULTS,
    MAX_TEXT_LENGTH, MIN_TEXT_LENGTH,
};
use std::io::BufRead;
use std::num::NonZeroUsize;

const ALLOWED_FILE_EXTENSIONS: &[&str] = &[
    "txt", "md", "json", "csv", "xml", "html", "htm", "rst", "py", "rs", "js", "ts",
];

const TEXT_FILE_MAGIC_NUMBERS: &[(&[u8], &[u8], &str)] = &[
    (&[0xEF, 0xBB, 0xBF], &[], "UTF-8 BOM"),
    (&[0xFF, 0xFE], &[], "UTF-16 LE BOM"),
    (&[0xFE, 0xFF], &[], "UTF-16 BE BOM"),
    (
        &[0x3C, 0x21, 0x64, 0x6F, 0x63, 0x74, 0x79, 0x70, 0x65],
        &[],
        "HTML/DOCTYPE",
    ),
    (&[0x3C, 0x68, 0x74, 0x6D, 0x6C], &[], "HTML"),
    (&[0x7B, 0x22], &[], "JSON"),
    (&[0x3C, 0x3F, 0x78, 0x6D, 0x6C], &[], "XML"),
];

const MAX_MAGIC_BYTES: usize = 16;

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
    fn validate_text(&self, text: &str) -> Result<(), VecboostError>;
    fn validate_batch(&self, texts: &[String]) -> Result<(), VecboostError>;
    fn validate_search(
        &self,
        query: &str,
        texts: &[String],
        top_k: Option<usize>,
    ) -> Result<(), VecboostError>;
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

    fn validate_text_content(&self, text: &str) -> Result<(), VecboostError> {
        if text.is_empty() {
            return Err(VecboostError::InvalidInput(
                "Text cannot be empty".to_string(),
            ));
        }

        let char_count = text.chars().count();
        if char_count < self.config.min_text_length {
            return Err(VecboostError::InvalidInput(format!(
                "Text too short: {} characters (minimum: {})",
                char_count, self.config.min_text_length
            )));
        }

        if char_count > self.config.max_text_length.get() {
            return Err(VecboostError::InvalidInput(format!(
                "Text too long: {} characters (maximum: {})",
                char_count,
                self.config.max_text_length.get()
            )));
        }

        if text.trim().is_empty() {
            return Err(VecboostError::InvalidInput(
                "Text contains only whitespace".to_string(),
            ));
        }

        Ok(())
    }
}

impl TextValidator for InputValidator {
    fn validate_text(&self, text: &str) -> Result<(), VecboostError> {
        self.validate_text_content(text)
    }

    fn validate_batch(&self, texts: &[String]) -> Result<(), VecboostError> {
        if texts.is_empty() {
            return Err(VecboostError::InvalidInput(
                "Batch cannot be empty".to_string(),
            ));
        }

        if texts.len() > self.config.max_batch_size.get() {
            return Err(VecboostError::InvalidInput(format!(
                "Batch size {} exceeds maximum {}",
                texts.len(),
                self.config.max_batch_size.get()
            )));
        }

        for (idx, text) in texts.iter().enumerate() {
            self.validate_text_content(text).map_err(|e| {
                VecboostError::InvalidInput(format!(
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
    ) -> Result<(), VecboostError> {
        self.validate_text_content(query)?;

        if texts.is_empty() {
            return Err(VecboostError::InvalidInput(
                "Search texts list cannot be empty".to_string(),
            ));
        }

        if texts.len() > self.config.max_search_results.get() {
            return Err(VecboostError::InvalidInput(format!(
                "Search results count {} exceeds maximum {}",
                texts.len(),
                self.config.max_search_results.get()
            )));
        }

        if let Some(k) = top_k {
            if k == 0 {
                return Err(VecboostError::InvalidInput(
                    "top_k must be at least 1".to_string(),
                ));
            }
            if k > self.config.max_search_results.get() {
                return Err(VecboostError::InvalidInput(format!(
                    "top_k {} exceeds maximum {}",
                    k,
                    self.config.max_search_results.get()
                )));
            }
        }

        for (idx, text) in texts.iter().enumerate() {
            self.validate_text_content(text).map_err(|e| {
                VecboostError::InvalidInput(format!(
                    "Validation failed for search text at index {}: {}",
                    idx, e
                ))
            })?;
        }

        Ok(())
    }
}

impl InputValidator {
    fn validate_file_size(&self, path: &str) -> Result<(), VecboostError> {
        use std::fs;

        match fs::metadata(path) {
            Ok(metadata) => {
                let file_size = metadata.len();
                if file_size > MAX_FILE_SIZE_BYTES {
                    let size_mb = file_size as f64 / (1024.0 * 1024.0);
                    let max_mb = MAX_FILE_SIZE_BYTES as f64 / (1024.0 * 1024.0);
                    return Err(VecboostError::InvalidInput(format!(
                        "File size {:.2} MB exceeds maximum allowed size {:.2} MB",
                        size_mb, max_mb
                    )));
                }
                Ok(())
            }
            Err(e) => Err(VecboostError::InvalidInput(format!(
                "Cannot access file {}: {}",
                path, e
            ))),
        }
    }

    fn validate_file_extension(&self, path: &str) -> Result<(), VecboostError> {
        let ext = path
            .rsplit('.')
            .next()
            .ok_or_else(|| VecboostError::InvalidInput("File has no extension".to_string()))?;

        let ext_lower = ext.to_ascii_lowercase();
        if !ALLOWED_FILE_EXTENSIONS.contains(&ext_lower.as_str()) {
            return Err(VecboostError::InvalidInput(format!(
                "File extension '.{}' is not allowed. Allowed extensions: {:?}",
                ext, ALLOWED_FILE_EXTENSIONS
            )));
        }

        Ok(())
    }

    fn validate_file_content(&self, path: &str) -> Result<(), VecboostError> {
        use std::fs::File;
        use std::io::Read;

        let file = File::open(path)
            .map_err(|e| VecboostError::InvalidInput(format!("Cannot open file: {}", e)))?;

        let mut buffer = [0u8; MAX_MAGIC_BYTES];
        let mut reader = std::io::BufReader::new(file);

        reader
            .read(&mut buffer)
            .map_err(|e| VecboostError::InvalidInput(format!("Cannot read file: {}", e)))?;

        let bytes_read = reader
            .fill_buf()
            .map_err(|e| VecboostError::InvalidInput(format!("Cannot read file buffer: {}", e)))?;

        if bytes_read.is_empty() {
            return Ok(());
        }

        let mut has_text_marker = false;
        for (magic, mask, _name) in TEXT_FILE_MAGIC_NUMBERS {
            let magic_len = magic.len();
            if bytes_read.len() >= magic_len {
                let mut matches = true;
                for (i, &magic_byte) in magic.iter().enumerate() {
                    let mask_byte = mask.get(i).copied().unwrap_or(0xFF);
                    if (bytes_read[i] & mask_byte) != magic_byte {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    has_text_marker = true;
                    break;
                }
            }
        }

        if !has_text_marker {
            for &byte in bytes_read.iter().take(256) {
                if byte < 0x09 || (byte > 0x0A && byte < 0x20 && byte != 0x1E && byte != 0x1F) {
                    return Err(VecboostError::InvalidInput(
                        "File contains non-text binary data".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn validate_file_path(&self, path: &str) -> Result<(), VecboostError> {
        self.validate_file_extension(path)?;
        self.validate_file_size(path)?;
        self.validate_file_content(path)?;
        Ok(())
    }
}

pub trait FileValidator {
    fn validate_file(&self, path: &str) -> Result<(), VecboostError>;
}

impl FileValidator for InputValidator {
    fn validate_file(&self, path: &str) -> Result<(), VecboostError> {
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
        assert!(
            validator
                .validate_search("query", &["text1".to_string()], Some(5))
                .is_ok()
        );
        assert!(
            validator
                .validate_search("", &["text1".to_string()], Some(5))
                .is_err()
        );
        assert!(validator.validate_search("query", &[], Some(5)).is_err());
    }

    #[test]
    fn test_sql_injection_attempts_rejected_by_text_validator() {
        // SQL 注入尝试作为文本输入:验证器不会因 SQL 语法报错,
        // 但应正常接受为合法文本(嵌入服务处理任意文本)。
        // 这里验证这些输入不会导致 panic 或异常行为(它们是合法文本)。
        let validator = InputValidator::with_default();
        // 经典 SQL 注入模式作为文本应被接受(文本验证只检查长度/空)
        assert!(validator.validate_text("' OR 1=1 --").is_ok());
        assert!(validator.validate_text("'; DROP TABLE users; --").is_ok());
        assert!(
            validator
                .validate_text("UNION SELECT * FROM passwords")
                .is_ok()
        );
        assert!(validator.validate_text("admin'--").is_ok());
        assert!(
            validator
                .validate_text("1; EXEC xp_cmdshell('dir')")
                .is_ok()
        );
    }

    #[test]
    fn test_xss_attempts_as_text_are_valid() {
        // XSS 模式作为文本输入是合法的(嵌入服务不解释 HTML/JS)
        let validator = InputValidator::with_default();
        assert!(
            validator
                .validate_text("<script>alert('xss')</script>")
                .is_ok()
        );
        assert!(
            validator
                .validate_text("<img src=x onerror=alert(1)>")
                .is_ok()
        );
        assert!(
            validator
                .validate_text("javascript:document.cookie")
                .is_ok()
        );
        assert!(validator.validate_text("<svg/onload=alert(1)>").is_ok());
    }

    #[test]
    fn test_special_characters_in_text() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_text("Hello, 世界!").is_ok());
        assert!(validator.validate_text("Emoji: 😀🎉").is_ok());
        assert!(validator.validate_text("Tab\tcharacter").is_ok());
        assert!(validator.validate_text("Newline\nin text").is_ok());
        assert!(validator.validate_text("Symbols: @#$%^&*()").is_ok());
        assert!(validator.validate_text("Mixed: abc123!@#中文").is_ok());
    }

    #[test]
    fn test_text_exceeding_max_length_rejected() {
        let validator = InputValidator::with_default();
        // MAX_TEXT_LENGTH = 10000
        let long_text = "a".repeat(MAX_TEXT_LENGTH + 1);
        let result = validator.validate_text(&long_text);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("too long")),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_text_at_max_length_accepted() {
        let validator = InputValidator::with_default();
        let text = "a".repeat(MAX_TEXT_LENGTH);
        assert!(validator.validate_text(&text).is_ok());
    }

    #[test]
    fn test_text_below_min_length_rejected() {
        // MIN_TEXT_LENGTH = 1, 空字符串已被空检查捕获,
        // 但配置更高 min 时应触发 too short
        let config = ValidationConfig::new(
            NonZeroUsize::new(MAX_TEXT_LENGTH),
            Some(5), // min_text_length = 5
            NonZeroUsize::new(MAX_BATCH_SIZE),
            NonZeroUsize::new(MAX_SEARCH_RESULTS),
            NonZeroUsize::new(MAX_CONCURRENT_REQUESTS),
        );
        let validator = InputValidator::new(config);
        assert!(validator.validate_text("ab").is_err());
        let result = validator.validate_text("ab");
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("too short")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_text_exactly_at_min_length_accepted() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(MAX_TEXT_LENGTH),
            Some(3),
            NonZeroUsize::new(MAX_BATCH_SIZE),
            NonZeroUsize::new(MAX_SEARCH_RESULTS),
            NonZeroUsize::new(MAX_CONCURRENT_REQUESTS),
        );
        let validator = InputValidator::new(config);
        assert!(validator.validate_text("abc").is_ok());
    }

    #[test]
    fn test_batch_empty_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_batch(&[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_batch_with_empty_text_rejected_with_index() {
        let validator = InputValidator::with_default();
        let texts = vec!["valid".to_string(), "".to_string()];
        let result = validator.validate_batch(&texts);
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("index 1"));
            }
            _ => panic!("Expected InvalidInput with index"),
        }
    }

    #[test]
    fn test_batch_within_limit_accepted() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(MAX_TEXT_LENGTH),
            Some(1),
            NonZeroUsize::new(3),
            NonZeroUsize::new(MAX_SEARCH_RESULTS),
            NonZeroUsize::new(MAX_CONCURRENT_REQUESTS),
        );
        let validator = InputValidator::new(config);
        let texts = vec!["t1".to_string(), "t2".to_string(), "t3".to_string()];
        assert!(validator.validate_batch(&texts).is_ok());
    }

    #[test]
    fn test_batch_exceeds_max_size_rejected() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(MAX_TEXT_LENGTH),
            Some(1),
            NonZeroUsize::new(2),
            NonZeroUsize::new(MAX_SEARCH_RESULTS),
            NonZeroUsize::new(MAX_CONCURRENT_REQUESTS),
        );
        let validator = InputValidator::new(config);
        let texts = vec!["t1".to_string(), "t2".to_string(), "t3".to_string()];
        let result = validator.validate_batch(&texts);
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("exceeds maximum")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_search_with_top_k_zero_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_search("query", &["text1".to_string()], Some(0));
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("top_k must be at least 1")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_search_with_top_k_exceeding_max_rejected() {
        let validator = InputValidator::with_default();
        let k = MAX_SEARCH_RESULTS + 1;
        let result = validator.validate_search("query", &["text1".to_string()], Some(k));
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("exceeds maximum")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_search_with_empty_query_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_search("", &["text1".to_string()], Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_search_with_whitespace_query_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_search("   ", &["text1".to_string()], Some(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_search_texts_exceed_max_rejected() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(MAX_TEXT_LENGTH),
            Some(1),
            NonZeroUsize::new(MAX_BATCH_SIZE),
            NonZeroUsize::new(2),
            NonZeroUsize::new(MAX_CONCURRENT_REQUESTS),
        );
        let validator = InputValidator::new(config);
        let texts = vec!["t1".to_string(), "t2".to_string(), "t3".to_string()];
        let result = validator.validate_search("query", &texts, Some(1));
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("exceeds maximum")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_search_with_invalid_text_in_list_rejected_with_index() {
        let validator = InputValidator::with_default();
        let texts = vec!["valid".to_string(), "".to_string()];
        let result = validator.validate_search("query", &texts, Some(1));
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("index 1")),
            _ => panic!("Expected InvalidInput with index"),
        }
    }

    #[test]
    fn test_search_with_none_top_k_accepted() {
        let validator = InputValidator::with_default();
        let result = validator.validate_search("query", &["text1".to_string()], None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_config_default_values() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_text_length.get(), MAX_TEXT_LENGTH);
        assert_eq!(config.min_text_length, MIN_TEXT_LENGTH);
        assert_eq!(config.max_batch_size.get(), MAX_BATCH_SIZE);
        assert_eq!(config.max_search_results.get(), MAX_SEARCH_RESULTS);
        assert_eq!(
            config.max_concurrent_requests.get(),
            MAX_CONCURRENT_REQUESTS
        );
    }

    #[test]
    fn test_validation_config_new_with_custom_values() {
        let config = ValidationConfig::new(
            NonZeroUsize::new(5000),
            Some(2),
            NonZeroUsize::new(50),
            NonZeroUsize::new(500),
            NonZeroUsize::new(50),
        );
        assert_eq!(config.max_text_length.get(), 5000);
        assert_eq!(config.min_text_length, 2);
        assert_eq!(config.max_batch_size.get(), 50);
        assert_eq!(config.max_search_results.get(), 500);
        assert_eq!(config.max_concurrent_requests.get(), 50);
    }

    #[test]
    fn test_validation_config_new_with_none_uses_defaults() {
        let config = ValidationConfig::new(None, None, None, None, None);
        assert_eq!(config.max_text_length.get(), MAX_TEXT_LENGTH);
        assert_eq!(config.min_text_length, MIN_TEXT_LENGTH);
        assert_eq!(config.max_batch_size.get(), MAX_BATCH_SIZE);
        assert_eq!(config.max_search_results.get(), MAX_SEARCH_RESULTS);
        assert_eq!(
            config.max_concurrent_requests.get(),
            MAX_CONCURRENT_REQUESTS
        );
    }

    #[test]
    fn test_file_extension_validation_allowed_extensions() {
        let validator = InputValidator::with_default();
        // 这些扩展名在 ALLOWED_FILE_EXTENSIONS 中
        assert!(validator.validate_file_extension("doc.txt").is_ok());
        assert!(validator.validate_file_extension("readme.md").is_ok());
        assert!(validator.validate_file_extension("data.json").is_ok());
        assert!(validator.validate_file_extension("file.csv").is_ok());
        assert!(validator.validate_file_extension("page.html").is_ok());
        assert!(validator.validate_file_extension("page.htm").is_ok());
        assert!(validator.validate_file_extension("data.xml").is_ok());
        assert!(validator.validate_file_extension("code.py").is_ok());
        assert!(validator.validate_file_extension("code.rs").is_ok());
        assert!(validator.validate_file_extension("code.js").is_ok());
        assert!(validator.validate_file_extension("code.ts").is_ok());
        assert!(validator.validate_file_extension("doc.rst").is_ok());
    }

    #[test]
    fn test_file_extension_validation_disallowed_extensions() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_file_extension("malware.exe").is_err());
        assert!(validator.validate_file_extension("archive.zip").is_err());
        assert!(validator.validate_file_extension("image.png").is_err());
        assert!(validator.validate_file_extension("binary.bin").is_err());
    }

    #[test]
    fn test_file_extension_validation_case_insensitive() {
        let validator = InputValidator::with_default();
        assert!(validator.validate_file_extension("FILE.TXT").is_ok());
        assert!(validator.validate_file_extension("File.Md").is_ok());
        assert!(validator.validate_file_extension("DATA.JSON").is_ok());
    }

    #[test]
    fn test_file_extension_validation_no_extension_rejected() {
        let validator = InputValidator::with_default();
        // "noext" 没有 "." 分隔符,rsplit('.') 返回整个字符串 "noext"
        // 不在允许列表中,应被拒绝
        let result = validator.validate_file_extension("noext");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_size_validation_nonexistent_file_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_file_size("/nonexistent/path/file.txt");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("Cannot access file")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_file_size_validation_small_file_accepted() {
        let validator = InputValidator::with_default();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vecboost_validator_test_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "hello").unwrap();
        let result = validator.validate_file_size(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_content_validation_text_file_accepted() {
        let validator = InputValidator::with_default();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vecboost_validator_text_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "This is plain text content").unwrap();
        let result = validator.validate_file_content(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_content_validation_json_file_accepted() {
        let validator = InputValidator::with_default();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vecboost_validator_json_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, "{\"key\": \"value\"}").unwrap();
        let result = validator.validate_file_content(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_content_validation_empty_file_accepted() {
        let validator = InputValidator::with_default();
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vecboost_validator_empty_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, "").unwrap();
        let result = validator.validate_file_content(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        // 空文件应该被接受(bytes_read 为空时直接返回 Ok)
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_content_validation_nonexistent_file_rejected() {
        let validator = InputValidator::with_default();
        let result = validator.validate_file_content("/nonexistent/path/file.txt");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("Cannot open file")),
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_file_path_validation_combines_all_checks() {
        let validator = InputValidator::with_default();
        // 不存在的文件路径,扩展名合法但文件不存在 → 应失败
        let result = validator.validate_file_path("/nonexistent/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_path_validation_rejects_disallowed_extension() {
        let validator = InputValidator::with_default();
        let result = validator.validate_file_path("/some/path/file.exe");
        // 扩展名检查在前,应先失败
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("not allowed")),
            _ => panic!("Expected InvalidInput for disallowed extension"),
        }
    }

    #[test]
    fn test_file_validator_trait_validate_file_checks_size() {
        let validator = InputValidator::with_default();
        // FileValidator trait 只调用 validate_file_size
        let result = validator.validate_file("/nonexistent/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_text_validator_trait_validate_text_with_unicode() {
        let validator = InputValidator::with_default();
        // 多字节字符应按 char count 计数,不是字节
        let text = "中文".to_string(); // 2 个字符,6 字节
        assert!(validator.validate_text(&text).is_ok());
    }

    #[test]
    fn test_whitespace_only_text_rejected_with_correct_message() {
        let validator = InputValidator::with_default();
        let result = validator.validate_text("   \t\n  ");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("whitespace")),
            _ => panic!("Expected InvalidInput for whitespace-only text"),
        }
    }
}
