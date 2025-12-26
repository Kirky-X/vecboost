// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Numerical operations module using ndarray for high-performance vector computations.
//!
//! This module provides efficient implementations of vector operations using the
//! ndarray crate, which offers NumPy-like functionality for Rust.
//!
//! # Features
//!
//! - Vector arithmetic operations (add, subtract, multiply, divide)
//! - Norm and distance calculations
//! - Similarity metrics (cosine, euclidean, dot product, manhattan)
//! - Matrix-vector operations
//! - Batch processing support
//!
//! # Example
//!
//! ```
//! use vecboost::utils::ndarray_ops::*;
//!
//! let v1 = NdArrayVector::from(vec![1.0, 2.0, 3.0]);
//! let v2 = NdArrayVector::from(vec![4.0, 5.0, 6.0]);
//!
//! let similarity = cosine_similarity_ndarray(&v1, &v2).unwrap();
//! assert!(similarity > 0.9);
//! ```

use ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Dim, Ix1, Ix2, LinalgScalar};
use std::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "onnx")]
use crate::error::AppError;

pub type NdArrayVector = Array1<f32>;
pub type NdArrayMatrix = Array2<f32>;

#[derive(Debug, Clone, PartialEq)]
pub struct VectorOperations;

impl VectorOperations {
    #[inline]
    pub fn cosine_similarity(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
        if v1.len() != v2.len() {
            return Err(format!(
                "Vector dimensions mismatch: {} vs {}",
                v1.len(),
                v2.len()
            ));
        }

        let dot_product = v1.dot(v2);
        let norm_a = v1.mapv_into(|x| x.powi(2)).sum().sqrt();
        let norm_b = v2.mapv_into(|x| x.powi(2)).sum().sqrt();

        if norm_a < 1e-12 || norm_b < 1e-12 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    #[inline]
    pub fn euclidean_distance(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
        if v1.len() != v2.len() {
            return Err(format!(
                "Vector dimensions mismatch: {} vs {}",
                v1.len(),
                v2.len()
            ));
        }

        let diff = v1 - v2;
        let squared_distance: f32 = diff.mapv_into(|x| x.powi(2)).sum();

        Ok(squared_distance.sqrt())
    }

    #[inline]
    pub fn dot_product(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
        if v1.len() != v2.len() {
            return Err(format!(
                "Vector dimensions mismatch: {} vs {}",
                v1.len(),
                v2.len()
            ));
        }

        Ok(v1.dot(v2))
    }

    #[inline]
    pub fn manhattan_distance(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
        if v1.len() != v2.len() {
            return Err(format!(
                "Vector dimensions mismatch: {} vs {}",
                v1.len(),
                v2.len()
            ));
        }

        let diff = (v1 - v2).mapv_into(|x| x.abs());
        Ok(diff.sum())
    }

    #[inline]
    pub fn normalize_l2(v: &mut NdArrayVector) {
        let norm: f32 = v.mapv_into(|x| x.powi(2)).sum().sqrt();
        if norm > 1e-12 {
            *v = v.mapv_into(|x| x / norm);
        }
    }

    #[inline]
    pub fn normalize_batch(vectors: &mut NdArrayMatrix) {
        let norms: NdArrayVector = vectors
            .axis_iter(Axis(1))
            .map(|col| col.mapv_into(|x| x.powi(2)).sum().sqrt())
            .collect();

        for (mut col, &norm) in vectors.axis_iter_mut(Axis(1)).zip(norms.iter()) {
            if norm > 1e-12 {
                col.mapv_into_inplace(|x| x / norm);
            }
        }
    }

    #[inline]
    pub fn batch_cosine_similarity(query: &NdArrayVector, candidates: &NdArrayMatrix) -> NdArrayVector {
        let normalized_query = {
            let mut q = query.clone();
            VectorOperations::normalize_l2(&mut q);
            q
        };

        let mut normalized_candidates = candidates.clone();
        VectorOperations::normalize_batch(&mut normalized_candidates);

        normalized_candidates.t().dot(&normalized_query)
    }

    #[inline]
    pub fn batch_euclidean_distances(query: &NdArrayVector, candidates: &NdArrayMatrix) -> NdArrayVector {
        let query_squared_norm = query.mapv_into(|x| x.powi(2)).sum();
        let candidates_squared_norms: NdArrayVector = candidates
            .axis_iter(Axis(1))
            .map(|col| col.mapv_into(|x| x.powi(2)).sum())
            .collect();

        let cross_terms = candidates.t().dot(query);

        let distances: NdArrayVector = candidates_squared_norms
            .iter()
            .zip(cross_terms.iter())
            .map(|(&cand_norm, &cross)| (query_squared_norm + cand_norm - 2.0 * cross).max(0.0))
            .map(|x| x.sqrt())
            .collect();

        distances
    }

    #[inline]
    pub fn mean_pooling(embeddings: &NdArrayMatrix) -> NdArrayVector {
        embeddings.axis_mean(Axis(1)).unwrap().to_owned()
    }

    #[inline]
    pub fn max_pooling(embeddings: &NdArrayMatrix) -> NdArrayVector {
        embeddings.axis_iter(Axis(1))
            .map(|row| row.fold(std::f32::MIN, |acc, &x| acc.max(x)))
            .collect()
    }

    #[inline]
    pub fn min_pooling(embeddings: &NdArrayMatrix) -> NdArrayVector {
        embeddings.axis_iter(Axis(1))
            .map(|row| row.fold(std::f32::MAX, |acc, &x| acc.min(x)))
            .collect()
    }

    #[inline]
    pub fn weighted_mean_pooling(embeddings: &NdArrayMatrix, weights: &NdArrayVector) -> Result<NdArrayVector, String> {
        if embeddings.ncols() != weights.len() {
            return Err(format!(
                "Embeddings columns ({}) != weights length ({})",
                embeddings.ncols(),
                weights.len()
            ));
        }

        let weighted: NdArrayMatrix = embeddings * weights.mapv_into(|x| x.sqrt());
        let weight_sum = weights.sum();
        if weight_sum < 1e-12 {
            return Err("Sum of weights is too small".to_string());
        }

        Ok(weighted.axis_sum(Axis(1)) / weight_sum)
    }
}

#[inline]
pub fn cosine_similarity_ndarray(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
    VectorOperations::cosine_similarity(v1, v2)
}

#[inline]
pub fn euclidean_distance_ndarray(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
    VectorOperations::euclidean_distance(v1, v2)
}

#[inline]
pub fn dot_product_ndarray(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
    VectorOperations::dot_product(v1, v2)
}

#[inline]
pub fn manhattan_distance_ndarray(v1: &NdArrayVector, v2: &NdArrayVector) -> Result<f32, String> {
    VectorOperations::manhattan_distance(v1, v2)
}

#[inline]
pub fn normalize_l2_ndarray(v: &mut NdArrayVector) {
    VectorOperations::normalize_l2(v);
}

#[inline]
pub fn normalize_batch_ndarray(vectors: &mut NdArrayMatrix) {
    VectorOperations::normalize_batch(vectors);
}

#[inline]
pub fn batch_similarity_search(
    query: &NdArrayVector,
    candidates: &NdArrayMatrix,
    top_k: usize,
    metric: crate::utils::SimilarityMetric,
) -> Result<Vec<(usize, f32)>, String> {
    let scores: NdArrayVector = match metric {
        crate::utils::SimilarityMetric::Cosine => {
            VectorOperations::batch_cosine_similarity(query, candidates)
        }
        crate::utils::SimilarityMetric::Euclidean => {
            let distances = VectorOperations::batch_euclidean_distances(query, candidates);
            distances.mapv_into(|d| 1.0 / (1.0 + d))
        }
        crate::utils::SimilarityMetric::DotProduct => {
            candidates.t().dot(query)
        }
        crate::utils::SimilarityMetric::Manhattan => {
            let query_norm = query.mapv_into(|x| x.abs());
            let candidates_norm: NdArrayVector = candidates
                .axis_iter(Axis(1))
                .map(|row| row.mapv_into(|x| x.abs()).sum())
                .collect();
            let cross_terms = candidates.t().dot(query);

            query_norm.sum() + candidates_norm - 2.0 * cross_terms
        }
    };

    let mut indexed_scores: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < indexed_scores.len() {
        indexed_scores.truncate(top_k);
    }

    Ok(indexed_scores)
}

#[cfg(feature = "onnx")]
pub fn convert_ort_to_ndarray(ort_embedding: &ort::Tensor) -> Result<NdArrayVector, AppError> {
    use ort::TensorElementDataType;

    match ort_embedding.data_type() {
        TensorElementDataType::Float32 => {
            let values = ort_embedding
                .as_slice::<f32>()
                .map_err(|e| AppError::InferenceError(format!("Failed to cast ORT tensor: {}", e)))?;
            Ok(NdArrayVector::from(values.to_vec()))
        }
        _ => Err(AppError::InferenceError(format!(
            "Unsupported tensor data type for ndarray conversion: {:?}",
            ort_embedding.data_type()
        ))),
    }
}

#[cfg(feature = "onnx")]
pub fn convert_ndarray_to_ort(vector: &NdArrayVector) -> ort::Tensor {
    ort::Tensor::from_slice(vector.as_slice().to_vec(), &[vector.len()]).unwrap()
}

#[cfg(test)]
mod ndarray_tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_same() {
        let v1 = NdArrayVector::from(vec![1.0, 0.0, 0.0]);
        let v2 = NdArrayVector::from(vec![1.0, 0.0, 0.0]);
        let similarity = VectorOperations::cosine_similarity(&v1, &v2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = NdArrayVector::from(vec![1.0, 0.0]);
        let v2 = NdArrayVector::from(vec![0.0, 1.0]);
        let similarity = VectorOperations::cosine_similarity(&v1, &v2).unwrap();
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = NdArrayVector::from(vec![1.0, 0.0]);
        let v2 = NdArrayVector::from(vec![-1.0, 0.0]);
        let similarity = VectorOperations::cosine_similarity(&v1, &v2).unwrap();
        assert!((similarity - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_error_on_mismatch() {
        let v1 = NdArrayVector::from(vec![1.0, 0.0]);
        let v2 = NdArrayVector::from(vec![1.0, 0.0, 0.0]);
        assert!(VectorOperations::cosine_similarity(&v1, &v2).is_err());
    }

    #[test]
    fn test_euclidean_distance_same() {
        let v1 = NdArrayVector::from(vec![1.0, 2.0, 3.0]);
        let v2 = NdArrayVector::from(vec![1.0, 2.0, 3.0]);
        let distance = VectorOperations::euclidean_distance(&v1, &v2).unwrap();
        assert!((distance - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let v1 = NdArrayVector::from(vec![0.0, 0.0]);
        let v2 = NdArrayVector::from(vec![3.0, 4.0]);
        let distance = VectorOperations::euclidean_distance(&v1, &v2).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_basic() {
        let v1 = NdArrayVector::from(vec![1.0, 2.0, 3.0]);
        let v2 = NdArrayVector::from(vec![4.0, 5.0, 6.0]);
        let product = VectorOperations::dot_product(&v1, &v2).unwrap();
        assert_eq!(product, 32.0);
    }

    #[test]
    fn test_manhattan_distance_basic() {
        let v1 = NdArrayVector::from(vec![0.0, 0.0]);
        let v2 = NdArrayVector::from(vec![3.0, 4.0]);
        let distance = VectorOperations::manhattan_distance(&v1, &v2).unwrap();
        assert!((distance - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_l2() {
        let mut v = NdArrayVector::from(vec![3.0, 4.0]);
        VectorOperations::normalize_l2(&mut v);
        let expected = NdArrayVector::from(vec![0.6, 0.8]);
        assert!(v.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn test_mean_pooling() {
        let embeddings = NdArrayMatrix::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let pooled = VectorOperations::mean_pooling(&embeddings);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pooling() {
        let embeddings = NdArrayMatrix::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let pooled = VectorOperations::max_pooling(&embeddings);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 3.0).abs() < 1e-6);
        assert!((pooled[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_pooling() {
        let embeddings = NdArrayMatrix::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let pooled = VectorOperations::min_pooling(&embeddings);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 1.0).abs() < 1e-6);
        assert!((pooled[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = NdArrayVector::from(vec![1.0, 0.0, 0.0]);
        let candidates = NdArrayMatrix::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        let scores = VectorOperations::batch_cosine_similarity(&query, &candidates);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);
        assert!((scores[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_pooling() {
        let embeddings = NdArrayMatrix::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let weights = NdArrayVector::from(vec![1.0, 1.0]);
        let pooled = VectorOperations::weighted_mean_pooling(&embeddings, &weights).unwrap();
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 5.0).abs() < 1e-6);
    }
}
