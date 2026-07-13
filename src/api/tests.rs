// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::*;
use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::RwLock;

/// Deterministic mock engine for API layer tests.
///
/// Generates a normalized embedding by hashing the text bytes, so the same
/// input always yields the same vector. This mirrors the TestEngine used in
/// `service/embedding.rs` tests.
struct TestEngine {
    dimension: usize,
}

impl TestEngine {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();

        let mut hash: u64 = 1469598103934665603;
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }

        let mut state = hash;
        for val in embedding.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let float_val = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *val = float_val;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl InferenceEngine for TestEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }

    fn precision(&self) -> &Precision {
        const PRECISION: Precision = Precision::Fp32;
        &PRECISION
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

fn make_service(dimension: usize) -> EmbeddingService {
    let temp_dir = tempdir().unwrap();
    let mock_engine = TestEngine::new(dimension);
    let model_config = ModelConfig {
        name: "test-model".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from(temp_dir.path()),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(dimension),
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
    };
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(mock_engine));
    let _ = temp_dir; // keep temp dir alive for the test
    EmbeddingService::new(engine, Some(model_config))
}

#[tokio::test]
async fn test_embed_returns_vector() {
    let service = make_service(384);
    let req = EmbedRequest {
        text: "Hello world".to_string(),
        normalize: Some(true),
    };
    let result = embed(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.dimension, 384);
    assert_eq!(response.embedding.len(), 384);
    // Verify the vector is L2-normalized
    let norm: f32 = response.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_embed_empty_text_returns_error() {
    let service = make_service(384);
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = embed(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[tokio::test]
async fn test_embed_batch_returns_vectors() {
    let service = make_service(384);
    let texts = vec![
        "Hello world".to_string(),
        "Rust is great".to_string(),
        "Embedding vectors".to_string(),
    ];
    let req = BatchEmbedRequest {
        texts: texts.clone(),
        mode: None,
        normalize: Some(true),
    };
    let result = embed_batch(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.embeddings.len(), 3);
    assert_eq!(response.dimension, 384);
    for (i, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.embedding.len(), 384);
        assert_eq!(emb.text_preview, texts[i]);
    }
}

#[tokio::test]
async fn test_embed_batch_empty_returns_error() {
    let service = make_service(384);
    let req = BatchEmbedRequest {
        texts: vec![],
        mode: None,
        normalize: None,
    };
    let result = embed_batch(&service, req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_compute_similarity_returns_score() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "Hello world".to_string(),
        target: "Hello rust".to_string(),
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    // Cosine similarity of normalized vectors is in [-1, 1]
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[tokio::test]
async fn test_compute_similarity_identical_texts_returns_one() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "identical text".to_string(),
        target: "identical text".to_string(),
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    // Identical normalized vectors have cosine similarity 1.0
    assert!((response.score - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_compute_similarity_empty_source_returns_error() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "valid target".to_string(),
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[tokio::test]
async fn test_compute_similarity_empty_target_returns_error() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "valid source".to_string(),
        target: "".to_string(),
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}
