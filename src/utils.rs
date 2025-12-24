// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

pub const MAX_TEXT_LENGTH: usize = 10_000;
pub const MIN_TEXT_LENGTH: usize = 1;
pub const MAX_BATCH_SIZE: usize = 100;
pub const MAX_SEARCH_RESULTS: usize = 1000;
pub const MAX_CONCURRENT_REQUESTS: usize = 100;
pub const DEFAULT_TOP_K: usize = 5;
pub const MAX_TOP_K: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl std::str::FromStr for SimilarityMetric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(SimilarityMetric::Cosine),
            "euclidean" => Ok(SimilarityMetric::Euclidean),
            "dot" | "dotproduct" | "dot_product" => Ok(SimilarityMetric::DotProduct),
            "manhattan" | "l1" => Ok(SimilarityMetric::Manhattan),
            _ => Err(format!("Unknown similarity metric: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AggregationMode {
    #[default]
    SlidingWindow,
    Paragraph,
    FixedSize,
    Average,
    MaxPooling,
    MinPooling,
}

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
            .validate_search("query", &vec!["text1".to_string()], Some(5))
            .is_ok());
        assert!(validator
            .validate_search("", &vec!["text1".to_string()], Some(5))
            .is_err());
        assert!(validator
            .validate_search("query", &vec![], Some(5))
            .is_err());
    }
}

/// 计算余弦相似度
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> Result<f32, AppError> {
    if v1.len() != v2.len() {
        return Err(AppError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// 计算欧几里得距离
pub fn euclidean_distance(v1: &[f32], v2: &[f32]) -> Result<f32, AppError> {
    if v1.len() != v2.len() {
        return Err(AppError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    let squared_distance: f32 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();

    Ok(squared_distance.sqrt())
}

/// 计算点积
pub fn dot_product(v1: &[f32], v2: &[f32]) -> Result<f32, AppError> {
    if v1.len() != v2.len() {
        return Err(AppError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    Ok(v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum())
}

/// 计算曼哈顿距离
pub fn manhattan_distance(v1: &[f32], v2: &[f32]) -> Result<f32, AppError> {
    if v1.len() != v2.len() {
        return Err(AppError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    Ok(v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum())
}

/// 统一的相似度计算函数
pub fn calculate_similarity(
    v1: &[f32],
    v2: &[f32],
    metric: SimilarityMetric,
) -> Result<f32, AppError> {
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(v1, v2),
        SimilarityMetric::Euclidean => {
            let distance = euclidean_distance(v1, v2)?;
            Ok(1.0 / (1.0 + distance))
        }
        SimilarityMetric::DotProduct => dot_product(v1, v2),
        SimilarityMetric::Manhattan => {
            let distance = manhattan_distance(v1, v2)?;
            Ok(1.0 / (1.0 + distance))
        }
    }
}

/// 向量归一化 (L2 Norm)
pub fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    normalize_l2(v)
}

#[cfg(test)]
mod similarity_tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_basic() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        assert!((cosine_similarity(&v1, &v2).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        assert!((cosine_similarity(&v1, &v2).unwrap() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_error_on_mismatch() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!(cosine_similarity(&v1, &v2).is_err());
    }

    #[test]
    fn test_euclidean_distance_same() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&v1, &v2).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];
        assert!((euclidean_distance(&v1, &v2).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_error_on_mismatch() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(euclidean_distance(&v1, &v2).is_err());
    }

    #[test]
    fn test_dot_product_basic() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&v1, &v2).unwrap(), 32.0);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        assert_eq!(dot_product(&v1, &v2).unwrap(), 0.0);
    }

    #[test]
    fn test_dot_product_error_on_mismatch() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(dot_product(&v1, &v2).is_err());
    }

    #[test]
    fn test_manhattan_distance_same() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!((manhattan_distance(&v1, &v2).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_basic() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];
        assert!((manhattan_distance(&v1, &v2).unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_error_on_mismatch() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(manhattan_distance(&v1, &v2).is_err());
    }

    #[test]
    fn test_calculate_similarity_cosine() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!(
            (calculate_similarity(&v1, &v2, SimilarityMetric::Cosine).unwrap() - 1.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_calculate_similarity_euclidean() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![1.0, 0.0];
        assert!(
            (calculate_similarity(&v1, &v2, SimilarityMetric::Euclidean).unwrap() - 1.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_calculate_similarity_dot_product() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![3.0, 4.0];
        assert_eq!(
            calculate_similarity(&v1, &v2, SimilarityMetric::DotProduct).unwrap(),
            11.0
        );
    }

    #[test]
    fn test_calculate_similarity_manhattan() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![0.0, 0.0];
        assert!(
            (calculate_similarity(&v1, &v2, SimilarityMetric::Manhattan).unwrap() - 1.0).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_similarity_metric_parse() {
        assert_eq!(
            "cosine".parse::<SimilarityMetric>().unwrap(),
            SimilarityMetric::Cosine
        );
        assert_eq!(
            "euclidean".parse::<SimilarityMetric>().unwrap(),
            SimilarityMetric::Euclidean
        );
        assert_eq!(
            "dot_product".parse::<SimilarityMetric>().unwrap(),
            SimilarityMetric::DotProduct
        );
        assert_eq!(
            "manhattan".parse::<SimilarityMetric>().unwrap(),
            SimilarityMetric::Manhattan
        );
        assert_eq!(
            "l1".parse::<SimilarityMetric>().unwrap(),
            SimilarityMetric::Manhattan
        );
    }

    #[test]
    fn test_similarity_metric_default() {
        assert_eq!(SimilarityMetric::default(), SimilarityMetric::Cosine);
    }

    #[test]
    fn test_similarity_metric_parse_error() {
        assert!("invalid".parse::<SimilarityMetric>().is_err());
    }
}
