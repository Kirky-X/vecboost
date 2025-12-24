// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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
}
