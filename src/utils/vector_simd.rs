// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Chunk-based vector operations with 4-wide manual unrolling.
//!
//! The compiler auto-vectorizes these loops into SIMD instructions
//! (SSE2/AVX on x86-64, NEON on ARM64). Benchmarks show this outperforms
//! explicit SIMD crates like `wide` because the compiler can generate
//! aligned vector loads from contiguous slice access patterns.

/// Compute dot product of two f32 slices (4-wide unrolled).
///
/// The slices must have equal length. Caller is responsible for validation.
#[inline]
pub fn dot_product_chunked(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(
        v1.len(),
        v2.len(),
        "dot_product_chunked: slice length mismatch ({} vs {})",
        v1.len(),
        v2.len()
    );
    let len = v1.len();
    let chunks = len / 4;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        sum += v1[base] * v2[base]
            + v1[base + 1] * v2[base + 1]
            + v1[base + 2] * v2[base + 2]
            + v1[base + 3] * v2[base + 3];
    }

    // Handle remaining elements
    let tail_start = chunks * 4;
    for i in tail_start..len {
        sum += v1[i] * v2[i];
    }

    sum
}

/// Compute sum of squares of a f32 slice (4-wide unrolled).
#[inline]
pub fn sum_of_squares_chunked(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 4;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        let a = v[base];
        let b = v[base + 1];
        let c = v[base + 2];
        let d = v[base + 3];
        sum += a * a + b * b + c * c + d * d;
    }

    // Handle remaining elements
    let tail_start = chunks * 4;
    for val in &v[tail_start..] {
        sum += val * val;
    }

    sum
}

/// Compute squared euclidean distance (4-wide unrolled).
#[inline]
pub fn squared_euclidean_chunked(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(
        v1.len(),
        v2.len(),
        "squared_euclidean_chunked: slice length mismatch ({} vs {})",
        v1.len(),
        v2.len()
    );
    let len = v1.len();
    let chunks = len / 4;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = v1[base] - v2[base];
        let d1 = v1[base + 1] - v2[base + 1];
        let d2 = v1[base + 2] - v2[base + 2];
        let d3 = v1[base + 3] - v2[base + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        let d = v1[i] - v2[i];
        sum += d * d;
    }

    sum
}

/// Compute manhattan distance (4-wide unrolled).
#[inline]
pub fn manhattan_distance_chunked(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(
        v1.len(),
        v2.len(),
        "manhattan_distance_chunked: slice length mismatch ({} vs {})",
        v1.len(),
        v2.len()
    );
    let len = v1.len();
    let chunks = len / 4;

    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        sum += (v1[base] - v2[base]).abs()
            + (v1[base + 1] - v2[base + 1]).abs()
            + (v1[base + 2] - v2[base + 2]).abs()
            + (v1[base + 3] - v2[base + 3]).abs();
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        sum += (v1[i] - v2[i]).abs();
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot_product(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    fn scalar_sum_of_squares(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum()
    }

    fn scalar_squared_euclidean(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    fn scalar_manhattan(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum()
    }

    const TOLERANCE: f32 = 1e-6;

    #[test]
    fn test_dot_product_chunked_empty() {
        assert_eq!(dot_product_chunked(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_chunked_single() {
        assert!((dot_product_chunked(&[3.0], &[4.0]) - 12.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_dot_product_chunked_exact_chunk() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let expected = scalar_dot_product(&v1, &v2);
        assert!((dot_product_chunked(&v1, &v2) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_dot_product_chunked_with_remainder() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let expected = scalar_dot_product(&v1, &v2);
        assert!((dot_product_chunked(&v1, &v2) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_dot_product_chunked_large() {
        let dim = 1024;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 + 0.5).collect();
        let expected = scalar_dot_product(&v1, &v2);
        assert!((dot_product_chunked(&v1, &v2) - expected).abs() < TOLERANCE * expected.abs());
    }

    #[test]
    fn test_sum_of_squares_chunked_empty() {
        assert_eq!(sum_of_squares_chunked(&[]), 0.0);
    }

    #[test]
    fn test_sum_of_squares_chunked_single() {
        assert!((sum_of_squares_chunked(&[3.0]) - 9.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_sum_of_squares_chunked_exact_chunk() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let expected = scalar_sum_of_squares(&v);
        assert!((sum_of_squares_chunked(&v) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_sum_of_squares_chunked_with_remainder() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = scalar_sum_of_squares(&v);
        assert!((sum_of_squares_chunked(&v) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_sum_of_squares_chunked_large() {
        let dim = 768;
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let expected = scalar_sum_of_squares(&v);
        assert!((sum_of_squares_chunked(&v) - expected).abs() < TOLERANCE * expected.abs());
    }

    #[test]
    fn test_squared_euclidean_chunked_basic() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];
        let expected = scalar_squared_euclidean(&v1, &v2);
        assert!((squared_euclidean_chunked(&v1, &v2) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_squared_euclidean_chunked_same() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(squared_euclidean_chunked(&v, &v).abs() < TOLERANCE);
    }

    #[test]
    fn test_squared_euclidean_chunked_large() {
        let dim = 384;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 + 0.5).collect();
        let expected = scalar_squared_euclidean(&v1, &v2);
        assert!(
            (squared_euclidean_chunked(&v1, &v2) - expected).abs() < TOLERANCE * expected.abs()
        );
    }

    #[test]
    fn test_manhattan_distance_chunked_basic() {
        let v1 = vec![0.0, 0.0];
        let v2 = vec![3.0, 4.0];
        let expected = scalar_manhattan(&v1, &v2);
        assert!((manhattan_distance_chunked(&v1, &v2) - expected).abs() < TOLERANCE);
    }

    #[test]
    fn test_manhattan_distance_chunked_same() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(manhattan_distance_chunked(&v, &v).abs() < TOLERANCE);
    }

    #[test]
    fn test_manhattan_distance_chunked_large() {
        let dim = 1024;
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 + 0.5).collect();
        let expected = scalar_manhattan(&v1, &v2);
        assert!(
            (manhattan_distance_chunked(&v1, &v2) - expected).abs() < TOLERANCE * expected.abs()
        );
    }
}
