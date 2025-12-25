// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use crate::utils::{normalize_l2, AggregationMode};

pub struct EmbeddingAggregator {
    mode: AggregationMode,
    normalize: bool,
}

impl EmbeddingAggregator {
    pub fn new(mode: AggregationMode) -> Self {
        Self {
            mode,
            normalize: true,
        }
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn aggregate(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>, AppError> {
        if embeddings.is_empty() {
            return Err(AppError::InvalidInput(
                "Cannot aggregate empty embeddings".to_string(),
            ));
        }

        if embeddings.len() == 1 {
            let mut result = embeddings[0].clone();
            if self.normalize {
                normalize_l2(&mut result);
            }
            return Ok(result);
        }

        let dimension = embeddings[0].len();
        let result = match self.mode {
            AggregationMode::Average
            | AggregationMode::SlidingWindow
            | AggregationMode::Paragraph
            | AggregationMode::Paragraphs
            | AggregationMode::Document => self.average_pooling(embeddings, dimension),
            AggregationMode::MaxPooling => self.max_pooling(embeddings, dimension),
            AggregationMode::MinPooling => self.min_pooling(embeddings, dimension),
            AggregationMode::FixedSize => self.average_pooling(embeddings, dimension),
        };

        if self.normalize {
            let mut normalized = result;
            normalize_l2(&mut normalized);
            Ok(normalized)
        } else {
            Ok(result)
        }
    }

    pub fn aggregate_with_weights(
        &self,
        embeddings: &[Vec<f32>],
        weights: &[f32],
    ) -> Result<Vec<f32>, AppError> {
        if embeddings.is_empty() {
            return Err(AppError::InvalidInput(
                "Cannot aggregate empty embeddings".to_string(),
            ));
        }

        if embeddings.len() != weights.len() {
            return Err(AppError::InvalidInput(
                "Embeddings and weights must have the same length".to_string(),
            ));
        }

        let dimension = embeddings[0].len();
        let mut aggregated = vec![0.0f32; dimension];
        let weight_sum: f32 = weights.iter().sum();

        if weight_sum == 0.0 {
            return self.aggregate(embeddings);
        }

        for (embedding, &weight) in embeddings.iter().zip(weights.iter()) {
            let normalized_weight = weight / weight_sum;
            for (i, val) in embedding.iter().enumerate() {
                aggregated[i] += val * normalized_weight;
            }
        }

        if self.normalize {
            normalize_l2(&mut aggregated);
        }

        Ok(aggregated)
    }

    fn average_pooling(&self, embeddings: &[Vec<f32>], dimension: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; dimension];
        let count = embeddings.len() as f32;

        for embedding in embeddings {
            for (i, val) in embedding.iter().enumerate() {
                result[i] += val / count;
            }
        }

        result
    }

    fn max_pooling(&self, embeddings: &[Vec<f32>], dimension: usize) -> Vec<f32> {
        let mut result = vec![f32::MIN; dimension];

        for embedding in embeddings {
            for (i, val) in embedding.iter().enumerate() {
                if val > &result[i] {
                    result[i] = *val;
                }
            }
        }

        result
    }

    fn min_pooling(&self, embeddings: &[Vec<f32>], dimension: usize) -> Vec<f32> {
        let mut result = vec![f32::MAX; dimension];

        for embedding in embeddings {
            for (i, val) in embedding.iter().enumerate() {
                if val < &result[i] {
                    result[i] = *val;
                }
            }
        }

        result
    }

    pub fn get_mode(&self) -> AggregationMode {
        self.mode
    }
}

pub fn calculate_overlap_weights(
    chunk_count: usize,
    overlap_size: usize,
    chunk_size: usize,
) -> Vec<f32> {
    if chunk_count <= 1 {
        return vec![1.0];
    }

    let mut weights = vec![1.0; chunk_count];
    let non_overlap_size = chunk_size - overlap_size;

    for (i, weight) in weights.iter_mut().enumerate().take(chunk_count) {
        let start_pos = i * non_overlap_size;
        let end_pos = start_pos + chunk_size;

        let mut overlap_weight = 1.0;

        if i > 0 {
            let prev_end = (i - 1) * non_overlap_size + chunk_size;
            let overlap_start = prev_end.saturating_sub(overlap_size);
            let overlap_end = start_pos;
            if overlap_end > overlap_start {
                let overlap_tokens = overlap_end - overlap_start;
                let overlap_ratio = overlap_tokens as f32 / chunk_size as f32;
                overlap_weight *= 1.0 - overlap_ratio * 0.5;
            }
        }

        if i < chunk_count - 1 {
            let next_start = (i + 1) * non_overlap_size;
            if end_pos > next_start {
                let overlap_tokens = end_pos - next_start;
                let overlap_ratio = overlap_tokens as f32 / chunk_size as f32;
                overlap_weight *= 1.0 - overlap_ratio * 0.5;
            }
        }

        *weight = overlap_weight.max(0.1);
    }

    let weight_sum: f32 = weights.iter().sum();
    weights.iter().map(|w| w / weight_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_creation() {
        let aggregator = EmbeddingAggregator::new(AggregationMode::Average);
        assert_eq!(aggregator.get_mode(), AggregationMode::Average);
    }

    #[test]
    fn test_aggregator_with_normalize() {
        let aggregator_normalized =
            EmbeddingAggregator::new(AggregationMode::Average).with_normalize(true);
        let aggregator_not_normalized =
            EmbeddingAggregator::new(AggregationMode::Average).with_normalize(false);

        let embeddings = vec![vec![2.0, 0.0, 0.0]];

        let normalized_result = aggregator_normalized.aggregate(&embeddings).unwrap();
        let not_normalized_result = aggregator_not_normalized.aggregate(&embeddings).unwrap();

        println!("Normalized: {:?}", normalized_result);
        println!("Not normalized: {:?}", not_normalized_result);

        assert!((normalized_result[0] - 1.0).abs() < 1e-6);
        assert!((not_normalized_result[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_embedding() {
        let aggregator = EmbeddingAggregator::new(AggregationMode::Average);
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let result = aggregator.aggregate(&embeddings).unwrap();

        assert_eq!(result, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_average_pooling() {
        let aggregator = EmbeddingAggregator::new(AggregationMode::Average).with_normalize(false);
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = aggregator.aggregate(&embeddings).unwrap();

        let expected = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        assert!(result
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn test_max_pooling() {
        let aggregator =
            EmbeddingAggregator::new(AggregationMode::MaxPooling).with_normalize(false);
        let embeddings = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.2, 0.9, 0.4],
            vec![0.3, 0.4, 0.8],
        ];
        let result = aggregator.aggregate(&embeddings).unwrap();

        assert_eq!(result, vec![1.0, 0.9, 0.8]);
    }

    #[test]
    fn test_min_pooling() {
        let aggregator =
            EmbeddingAggregator::new(AggregationMode::MinPooling).with_normalize(false);
        let embeddings = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.2, 0.9, 0.4],
            vec![0.3, 0.4, 0.8],
        ];
        let result = aggregator.aggregate(&embeddings).unwrap();

        assert_eq!(result, vec![0.2, 0.4, 0.3]);
    }

    #[test]
    fn test_weighted_aggregation() {
        let aggregator = EmbeddingAggregator::new(AggregationMode::Average).with_normalize(false);
        let embeddings = vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let weights = vec![0.5, 0.5];
        let result = aggregator
            .aggregate_with_weights(&embeddings, &weights)
            .unwrap();

        println!("Weighted aggregation result: {:?}", result);
        println!("Weight sum: {}", weights.iter().sum::<f32>());

        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "Expected result[0] ≈ 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-6,
            "Expected result[1] ≈ 1.0, got {}",
            result[1]
        );
    }

    #[test]
    fn test_empty_embeddings_error() {
        let aggregator = EmbeddingAggregator::new(AggregationMode::Average);
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = aggregator.aggregate(&embeddings);

        assert!(result.is_err());
    }

    #[test]
    fn test_overlap_weights() {
        let weights = calculate_overlap_weights(3, 102, 512);
        assert_eq!(weights.len(), 3);

        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
