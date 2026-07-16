// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::*;
use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::service::embedding::EmbeddingService;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::RwLock;

/// Ensure the global SERVICE is initialized for tests that need it.
///
/// Uses `SERVICE.set()` directly (not `init_service`) to avoid panicking if
/// another test already initialized the OnceLock (process-global state).
fn ensure_service_initialized() {
    if SERVICE.get().is_none() {
        let svc = Arc::new(RwLock::new(make_service(384)));
        let _ = SERVICE.set(svc);
    }
}

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

// ---------------------------------------------------------------------------
// init_service / service() tests
// ---------------------------------------------------------------------------

#[test]
fn test_init_service_sets_global() {
    ensure_service_initialized();
    let result = service();
    assert!(result.is_ok(), "service() should return Ok after init");
}

#[test]
fn test_init_service_no_op_on_double_init() {
    ensure_service_initialized();
    let first = service().expect("service should be initialized");
    let svc = Arc::new(RwLock::new(make_service(384)));
    init_service(svc);
    let second = service().expect("service should still be initialized");
    assert!(
        Arc::ptr_eq(&first, &second),
        "init_service should be a no-op on double init (first caller wins)"
    );
}

// ---------------------------------------------------------------------------
// to_api_error / uuid_like_id tests
// ---------------------------------------------------------------------------

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_uuid_like_id_format() {
    let id = uuid_like_id();
    assert!(id.starts_with("err-"));
    let suffix = &id[4..];
    assert!(
        suffix.parse::<u128>().is_ok(),
        "suffix should be numeric, got: {}",
        suffix
    );
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_uuid_like_id_unique() {
    let id1 = uuid_like_id();
    let id2 = uuid_like_id();
    assert_ne!(id1, id2, "consecutive ids should differ");
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_invalid_input() {
    let err = VecboostError::InvalidInput("bad input".to_string());
    let api_err = to_api_error(&err);
    match api_err {
        ApiError::InvalidInput {
            message,
            field,
            value,
        } => {
            assert_eq!(message, "bad input");
            assert!(field.is_none());
            assert!(value.is_none());
        }
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_model_load_error() {
    let err = VecboostError::ModelLoadError("model not found".to_string());
    let api_err = to_api_error(&err);
    match api_err {
        ApiError::Internal {
            message, error_id, ..
        } => {
            assert!(message.contains("Model load error"));
            assert!(message.contains("model not found"));
            assert!(error_id.starts_with("err-"));
        }
        other => panic!("Expected Internal, got {:?}", other),
    }
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_other_variants_become_internal() {
    let variants = vec![
        VecboostError::ConfigError("cfg err".to_string()),
        VecboostError::InferenceError("inf err".to_string()),
        VecboostError::AuthenticationError("auth err".to_string()),
        VecboostError::DatabaseError("db err".to_string()),
        VecboostError::InternalError("internal err".to_string()),
        VecboostError::RateLimitExceeded("rl err".to_string()),
        VecboostError::ValidationError("val err".to_string()),
        VecboostError::IoError("io err".to_string()),
        VecboostError::NotFound("nf err".to_string()),
        VecboostError::SecurityError("sec err".to_string()),
        VecboostError::ModelNotLoaded("not loaded".to_string()),
        VecboostError::ModelFileCorrupted("corrupted".to_string()),
        VecboostError::ModelIntegrityError("integrity".to_string()),
        VecboostError::TokenizationError("tok err".to_string()),
        VecboostError::OutOfMemory("oom".to_string()),
    ];
    for err in variants {
        let api_err = to_api_error(&err);
        match api_err {
            ApiError::Internal {
                message, error_id, ..
            } => {
                assert!(!message.is_empty(), "message should not be empty");
                assert!(error_id.starts_with("err-"));
            }
            other => panic!("Expected Internal for {:?}, got {:?}", err, other),
        }
    }
}

// ---------------------------------------------------------------------------
// forge_embed / forge_embed_batch / forge_compute_similarity tests
// ---------------------------------------------------------------------------

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_success() {
    ensure_service_initialized();
    let req = EmbedRequest {
        text: "hello forge".to_string(),
        normalize: Some(true),
    };
    let result = forge_embed(req).await;
    assert!(
        result.is_ok(),
        "forge_embed should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert_eq!(response.dimension, 384);
    assert_eq!(response.embedding.len(), 384);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_empty_text_returns_error() {
    ensure_service_initialized();
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = forge_embed(req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ApiError::InvalidInput { message, .. } => {
            assert!(
                message.contains("empty") || message.contains("invalid"),
                "expected empty/invalid message, got: {}",
                message
            );
        }
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_batch_success() {
    ensure_service_initialized();
    let req = BatchEmbedRequest {
        texts: vec!["text1".to_string(), "text2".to_string()],
        mode: None,
        normalize: Some(true),
    };
    let result = forge_embed_batch(req).await;
    assert!(result.is_ok(), "forge_embed_batch should succeed");
    let response = result.unwrap();
    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.dimension, 384);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_batch_empty_returns_error() {
    ensure_service_initialized();
    let req = BatchEmbedRequest {
        texts: vec![],
        mode: None,
        normalize: None,
    };
    let result = forge_embed_batch(req).await;
    assert!(result.is_err());
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_compute_similarity_success() {
    ensure_service_initialized();
    let req = SimilarityRequest {
        source: "hello".to_string(),
        target: "world".to_string(),
    };
    let result = forge_compute_similarity(req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_compute_similarity_empty_source_error() {
    ensure_service_initialized();
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "target".to_string(),
    };
    let result = forge_compute_similarity(req).await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// cli_embed / cli_compute_similarity tests
// ---------------------------------------------------------------------------

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_embed_success() {
    ensure_service_initialized();
    let req = EmbedRequest {
        text: "hello cli".to_string(),
        normalize: None,
    };
    let result = cli_embed(req).await;
    assert!(
        result.is_ok(),
        "cli_embed should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(!response.embedding.is_empty());
    assert!(response.dimension > 0);
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_embed_empty_text_returns_error() {
    ensure_service_initialized();
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = cli_embed(req).await;
    assert!(result.is_err());
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_compute_similarity_success() {
    ensure_service_initialized();
    let req = SimilarityRequest {
        source: "source text".to_string(),
        target: "target text".to_string(),
    };
    let result = cli_compute_similarity(req).await;
    assert!(
        result.is_ok(),
        "cli_compute_similarity should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_compute_similarity_empty_source_returns_error() {
    ensure_service_initialized();
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "target".to_string(),
    };
    let result = cli_compute_similarity(req).await;
    assert!(result.is_err());
}
