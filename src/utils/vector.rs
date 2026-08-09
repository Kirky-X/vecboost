// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(feature = "schema")]
use utoipa::ToSchema;

use crate::error::VecboostError;
use crate::utils::vector_simd;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl FromStr for SimilarityMetric {
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
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum AggregationMode {
    #[default]
    SlidingWindow,
    Document,
    Paragraph,
    Paragraphs,
    FixedSize,
    Average,
    MaxPooling,
    MinPooling,
}

pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> Result<f32, VecboostError> {
    if v1.len() != v2.len() {
        return Err(VecboostError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    let dot_product = vector_simd::dot_product_chunked(v1, v2);
    let norm_a = vector_simd::sum_of_squares_chunked(v1).sqrt();
    let norm_b = vector_simd::sum_of_squares_chunked(v2).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot_product / (norm_a * norm_b))
}

pub fn euclidean_distance(v1: &[f32], v2: &[f32]) -> Result<f32, VecboostError> {
    if v1.len() != v2.len() {
        return Err(VecboostError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    let squared_distance = vector_simd::squared_euclidean_chunked(v1, v2);

    Ok(squared_distance.sqrt())
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> Result<f32, VecboostError> {
    if v1.len() != v2.len() {
        return Err(VecboostError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    Ok(vector_simd::dot_product_chunked(v1, v2))
}

pub fn manhattan_distance(v1: &[f32], v2: &[f32]) -> Result<f32, VecboostError> {
    if v1.len() != v2.len() {
        return Err(VecboostError::InvalidInput(format!(
            "Vector dimensions mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    Ok(vector_simd::manhattan_distance_chunked(v1, v2))
}

pub fn calculate_similarity(
    v1: &[f32],
    v2: &[f32],
    metric: SimilarityMetric,
) -> Result<f32, VecboostError> {
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

pub fn calculate_similarity_batch(
    query: &[f32],
    candidates: &[&[f32]],
    metric: SimilarityMetric,
) -> Result<Vec<f32>, VecboostError> {
    candidates
        .iter()
        .map(|candidate| calculate_similarity(query, candidate, metric))
        .collect()
}

pub fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Truncate a vector to the specified dimension.
/// Returns original vector if target >= original or target == 0.
///
/// 注意：Matryoshka 场景下截断会破坏单位向量语义（子向量范数 < 原范数），
/// 调用方必须在截断后调用 [`normalize_l2`] 重新归一化，以保证余弦相似度正确。
pub fn truncate_vector(vector: &[f32], target_dimension: usize) -> Vec<f32> {
    if target_dimension == 0 || target_dimension >= vector.len() {
        vector.to_vec()
    } else {
        vector[..target_dimension].to_vec()
    }
}

/// Validate dimension parameter against maximum allowed dimension.
pub fn validate_dimension(target: Option<usize>, max_dimension: usize) -> Result<(), String> {
    match target {
        Some(0) => Err("dimensions must be greater than 0".to_string()),
        Some(d) if d > max_dimension => Err(format!(
            "dimensions {} exceeds model maximum {}",
            d, max_dimension
        )),
        _ => Ok(()),
    }
}

/// 计算截断向量的信息保留率（能量比）。
///
/// 信息保留率 = ||v[:d]||² / ||v||²
/// 值域 [0.0, 1.0]，越接近 1.0 表示截断后保留的信息越多。
/// 空向量返回 0.0。
pub fn information_retention_rate(embedding: &[f32], target_dim: usize) -> f32 {
    if embedding.is_empty() {
        return 0.0;
    }
    let full_energy: f32 = embedding.iter().map(|x| x * x).sum();
    if full_energy == 0.0 {
        return 0.0;
    }
    let d = target_dim.min(embedding.len());
    let truncated_energy: f32 = embedding[..d].iter().map(|x| x * x).sum();
    truncated_energy / full_energy
}

/// 下游任务类型，用于自适应维度选择。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// 检索任务（推荐 1024 维）
    Retrieval,
    /// 聚类任务（推荐 512 维）
    Clustering,
    /// 分类任务（推荐 256 维）
    Classification,
    /// 语义搜索（推荐 768 维）
    SemanticSearch,
}

impl std::str::FromStr for TaskType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "retrieval" => Ok(TaskType::Retrieval),
            "clustering" => Ok(TaskType::Clustering),
            "classification" => Ok(TaskType::Classification),
            "semantic_search" | "semanticsearch" | "semantic-search" => {
                Ok(TaskType::SemanticSearch)
            }
            _ => Err(format!(
                "Unknown task type: {}. Valid: retrieval, clustering, classification, semantic_search"
            , s)),
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Retrieval => write!(f, "retrieval"),
            TaskType::Clustering => write!(f, "clustering"),
            TaskType::Classification => write!(f, "classification"),
            TaskType::SemanticSearch => write!(f, "semantic_search"),
        }
    }
}

/// 根据任务类型推荐 embedding 维度。
///
/// 推荐值基于 Matryoshka embedding 模型的经验值，
/// 结果不超过 max_dim。
pub fn recommended_dimension(task: TaskType, max_dim: usize) -> usize {
    let recommended = match task {
        TaskType::Retrieval => 1024,
        TaskType::Clustering => 512,
        TaskType::Classification => 256,
        TaskType::SemanticSearch => 768,
    };
    recommended.min(max_dim)
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

    // Batch similarity tests
    #[test]
    fn test_calculate_similarity_batch_basic() {
        let query = vec![1.0, 0.0];
        let c1: Vec<f32> = vec![1.0, 0.0];
        let c2: Vec<f32> = vec![0.0, 1.0];
        let c3: Vec<f32> = vec![-1.0, 0.0];
        let candidates: Vec<&[f32]> = vec![&c1, &c2, &c3];
        let results = calculate_similarity_batch(&query, &candidates, SimilarityMetric::Cosine).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-6);
        assert!((results[1] - 0.0).abs() < 1e-6);
        assert!((results[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_similarity_batch_empty_candidates() {
        let query = vec![1.0, 0.0];
        let candidates: Vec<&[f32]> = vec![];
        let results = calculate_similarity_batch(&query, &candidates, SimilarityMetric::Cosine).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_calculate_similarity_batch_dimension_mismatch() {
        let query = vec![1.0, 0.0];
        let c1: Vec<f32> = vec![1.0, 0.0, 0.0]; // different dimension
        let candidates: Vec<&[f32]> = vec![&c1];
        let result = calculate_similarity_batch(&query, &candidates, SimilarityMetric::Cosine);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_similarity_batch_consistency() {
        let query = vec![1.0, 2.0, 3.0];
        let c1: Vec<f32> = vec![4.0, 5.0, 6.0];
        let c2: Vec<f32> = vec![7.0, 8.0, 9.0];
        let candidates: Vec<&[f32]> = vec![&c1, &c2];
        let batch_results =
            calculate_similarity_batch(&query, &candidates, SimilarityMetric::DotProduct).unwrap();
        let individual_0 = calculate_similarity(&query, &c1, SimilarityMetric::DotProduct).unwrap();
        let individual_1 = calculate_similarity(&query, &c2, SimilarityMetric::DotProduct).unwrap();
        assert!((batch_results[0] - individual_0).abs() < 1e-6);
        assert!((batch_results[1] - individual_1).abs() < 1e-6);
    }

    // Truncation tests
    #[test]
    fn test_truncate_vector_smaller() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let truncated = truncate_vector(&v, 3);
        assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
        assert_eq!(truncated.len(), 3);
    }

    #[test]
    fn test_truncate_vector_same() {
        let v = vec![1.0, 2.0, 3.0];
        let truncated = truncate_vector(&v, 3);
        assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_truncate_vector_larger() {
        let v = vec![1.0, 2.0, 3.0];
        let truncated = truncate_vector(&v, 10);
        assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_truncate_vector_zero() {
        let v = vec![1.0, 2.0, 3.0];
        let truncated = truncate_vector(&v, 0);
        assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_validate_dimension_valid() {
        assert!(validate_dimension(Some(512), 1024).is_ok());
        assert!(validate_dimension(None, 1024).is_ok());
        assert!(validate_dimension(Some(1024), 1024).is_ok());
    }

    #[test]
    fn test_validate_dimension_invalid_too_small() {
        let result = validate_dimension(Some(0), 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("greater than 0"));
    }

    #[test]
    fn test_validate_dimension_invalid_too_large() {
        let result = validate_dimension(Some(2048), 1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds model maximum"));
    }

    // ========================================================================
    // Matryoshka: information retention rate
    // ========================================================================

    #[test]
    fn test_information_retention_rate_full_dim() {
        let v: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
        let rate = information_retention_rate(&v, 1024);
        assert!((rate - 1.0).abs() < 1e-6, "full dim should retain all energy");
    }

    #[test]
    fn test_information_retention_rate_half_dim() {
        // 均匀分布向量：前 512 维应保留约 50% 能量
        let v: Vec<f32> = (0..1024).map(|_| 1.0).collect();
        let rate = information_retention_rate(&v, 512);
        assert!((rate - 0.5).abs() < 1e-6, "uniform vector half dim should be ~0.5, got {}", rate);
    }

    #[test]
    fn test_information_retention_rate_empty() {
        assert_eq!(information_retention_rate(&[], 128), 0.0);
    }

    #[test]
    fn test_information_retention_rate_zero_vec() {
        let v = vec![0.0; 128];
        assert_eq!(information_retention_rate(&v, 64), 0.0);
    }

    #[test]
    fn test_matryoshka_energy_distribution() {
        // 生成 1024 维向量，验证截断到不同维度的能量保留率递增
        let v: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.001).sin() + 1.0).collect();
        let rate_128 = information_retention_rate(&v, 128);
        let rate_256 = information_retention_rate(&v, 256);
        let rate_512 = information_retention_rate(&v, 512);
        let rate_768 = information_retention_rate(&v, 768);
        assert!(rate_128 < rate_256, "128 < 256: {} vs {}", rate_128, rate_256);
        assert!(rate_256 < rate_512, "256 < 512: {} vs {}", rate_256, rate_512);
        assert!(rate_512 < rate_768, "512 < 768: {} vs {}", rate_512, rate_768);
    }

    // ========================================================================
    // Matryoshka: TaskType and recommended_dimension
    // ========================================================================

    #[test]
    fn test_task_type_from_str() {
        assert_eq!("retrieval".parse::<TaskType>().unwrap(), TaskType::Retrieval);
        assert_eq!("clustering".parse::<TaskType>().unwrap(), TaskType::Clustering);
        assert_eq!("classification".parse::<TaskType>().unwrap(), TaskType::Classification);
        assert_eq!("semantic_search".parse::<TaskType>().unwrap(), TaskType::SemanticSearch);
        assert_eq!("semantic-search".parse::<TaskType>().unwrap(), TaskType::SemanticSearch);
        assert!("invalid".parse::<TaskType>().is_err());
    }

    #[test]
    fn test_task_type_display() {
        assert_eq!(format!("{}", TaskType::Retrieval), "retrieval");
        assert_eq!(format!("{}", TaskType::SemanticSearch), "semantic_search");
    }

    #[test]
    fn test_recommended_dimension_within_max() {
        assert_eq!(recommended_dimension(TaskType::Retrieval, 2048), 1024);
        assert_eq!(recommended_dimension(TaskType::Clustering, 2048), 512);
        assert_eq!(recommended_dimension(TaskType::Classification, 2048), 256);
        assert_eq!(recommended_dimension(TaskType::SemanticSearch, 2048), 768);
    }

    #[test]
    fn test_recommended_dimension_capped_by_max() {
        assert_eq!(recommended_dimension(TaskType::Retrieval, 512), 512);
        assert_eq!(recommended_dimension(TaskType::SemanticSearch, 256), 256);
    }
}
