// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! OpenAI Compatible Embedding Routes
//!
//! Provides API endpoints that conform to the OpenAI Embeddings API specification.

use crate::domain::openai_embedding::{
    EmbeddingObject, EncodingFormat, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::{VecboostState, error::VecboostError};
use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::response::IntoResponse;
use std::net::SocketAddr;

/// Extract the real client IP address
fn extract_real_ip(addr: SocketAddr) -> String {
    addr.ip().to_string()
}

/// OpenAI-compatible embedding handler
///
/// This handler provides an endpoint that conforms to the OpenAI Embeddings API.
/// See: https://platform.openai.com/docs/api-reference/embeddings
pub async fn openai_embed_handler(
    State(state): State<VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<OpenAIEmbedRequest>,
) -> Result<impl IntoResponse, VecboostError> {
    // Validate input
    if req.input.is_empty() {
        return Err(VecboostError::InvalidInput(
            "input cannot be empty".to_string(),
        ));
    }

    // Validate input array size (OpenAI limit is 2048)
    if req.input.len() > 2048 {
        return Err(VecboostError::InvalidInput(
            "input array too large (max 2048 items)".to_string(),
        ));
    }

    // Extract encoding format (reserved for future base64 support)
    let _encoding_format = req
        .encoding_format
        .as_deref()
        .and_then(EncodingFormat::parse)
        .unwrap_or(EncodingFormat::Float);

    // Extract client IP for rate limiting
    let ip = extract_real_ip(addr);

    // Check rate limiting if enabled
    if state.rate_limit_enabled {
        let global_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Global)
            .await;
        let ip_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Ip(ip.clone()))
            .await;

        if global_remaining == 0 || ip_remaining == 0 {
            return Err(VecboostError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    // Process the embedding request using existing service
    let service_guard = state.service.read().await;

    let texts = req.input.to_vec();

    // Use batch processing for both single and multiple inputs
    let batch_req = crate::domain::BatchEmbedRequest {
        texts: texts.clone(),
        mode: None,
        normalize: Some(true),
    };

    let batch_response = service_guard
        .process_batch(batch_req, req.dimensions)
        .await?;

    // Build OpenAI-formatted response
    let embedding_objects: Vec<EmbeddingObject> = batch_response
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(idx, result)| EmbeddingObject {
            object: "embedding".to_string(),
            embedding: result.embedding,
            index: idx,
        })
        .collect();

    // Calculate token usage (approximate based on text length)
    let total_chars: usize = texts.iter().map(|s| s.len()).sum();
    let prompt_tokens = (total_chars / 4) as u32; // Rough approximation: 4 chars per token

    let response = OpenAIEmbedResponse {
        object: "list".to_string(),
        data: embedding_objects,
        model: req.model.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    };

    Ok(Json(response))
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
    use crate::domain::openai_embedding::OpenAIInput;
    use crate::engine::InferenceEngine;
    use crate::error::VecboostError;
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    };
    use crate::rate_limit::LimiteronAdapter;
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use axum::response::IntoResponse;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;

    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for TestEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            static PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_test_state() -> VecboostState {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            engine,
            Some(model_config),
        )));
        std::mem::forget(temp_dir);

        let rate_limiter = Arc::new(LimiteronAdapter::with_default_config());
        let pipeline_queue = Arc::new(PriorityRequestQueue::new(0));
        let response_channel = Arc::new(ResponseChannel::new());

        #[cfg(feature = "auth")]
        {
            VecboostState {
                service,
                jwt_manager: None,
                user_store: None,
                auth_enabled: false,
                csrf_config: None,
                csrf_token_store: None,
                metrics_collector: None,
                prometheus_collector: None,
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
                worker_manager: Arc::new(WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    Arc::new(RwLock::new(EmbeddingService::new(
                        Arc::new(RwLock::new(TestEngine::new(384))),
                        None,
                    ))),
                )),
                kit: None,
            }
        }

        #[cfg(not(feature = "auth"))]
        {
            VecboostState {
                service,
                auth_enabled: false,
                metrics_collector: None,
                prometheus_collector: None,
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
                worker_manager: Arc::new(WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    Arc::new(RwLock::new(EmbeddingService::new(
                        Arc::new(RwLock::new(TestEngine::new(384))),
                        None,
                    ))),
                )),
                kit: None,
            }
        }
    }

    #[test]
    fn test_extract_real_ip_ipv4() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let ip = extract_real_ip(addr);
        assert_eq!(ip, "127.0.0.1");
    }

    #[test]
    fn test_extract_real_ip_ipv6() {
        let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8080);
        let ip = extract_real_ip(addr);
        assert_eq!(ip, "::1");
    }

    #[tokio::test]
    async fn test_openai_embed_handler_empty_input_returns_error() {
        let state = make_test_state();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002);
        let req = OpenAIEmbedRequest {
            input: OpenAIInput::Multiple(vec![]),
            model: "test-model".to_string(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };

        let result = openai_embed_handler(State(state), ConnectInfo(addr), Json(req)).await;
        assert!(result.is_err(), "empty input should return error");
        match result.err().unwrap() {
            VecboostError::InvalidInput(msg) => {
                assert!(
                    msg.contains("empty"),
                    "error should mention empty input, got: {}",
                    msg
                );
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_openai_embed_handler_too_many_inputs_returns_error() {
        let state = make_test_state();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002);
        let too_many: Vec<String> = (0..2049).map(|i| format!("text{}", i)).collect();
        let req = OpenAIEmbedRequest {
            input: OpenAIInput::Multiple(too_many),
            model: "test-model".to_string(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };

        let result = openai_embed_handler(State(state), ConnectInfo(addr), Json(req)).await;
        assert!(result.is_err(), "too many inputs should return error");
        match result.err().unwrap() {
            VecboostError::InvalidInput(msg) => {
                assert!(
                    msg.contains("2048"),
                    "error should mention max 2048 items, got: {}",
                    msg
                );
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_openai_embed_handler_single_input_returns_success() {
        let state = make_test_state();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002);
        let req = OpenAIEmbedRequest {
            input: OpenAIInput::Single("hello world".to_string()),
            model: "test-model".to_string(),
            encoding_format: Some("float".to_string()),
            dimensions: None,
            user: None,
        };

        let result = openai_embed_handler(State(state), ConnectInfo(addr), Json(req)).await;
        assert!(result.is_ok(), "single input should succeed");
        let response = result.unwrap().into_response();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "successful embedding should return 200 OK"
        );
    }

    #[tokio::test]
    async fn test_openai_embed_handler_multiple_inputs_returns_success() {
        let state = make_test_state();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002);
        let req = OpenAIEmbedRequest {
            input: OpenAIInput::Multiple(vec![
                "hello world".to_string(),
                "second text".to_string(),
            ]),
            model: "test-model".to_string(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };

        let result = openai_embed_handler(State(state), ConnectInfo(addr), Json(req)).await;
        assert!(result.is_ok(), "multiple inputs should succeed");
        let response = result.unwrap().into_response();
        assert_eq!(
            response.status(),
            axum::http::StatusCode::OK,
            "successful batch embedding should return 200 OK"
        );
    }

    #[tokio::test]
    async fn test_openai_embed_handler_at_internal_batch_limit_succeeds() {
        let state = make_test_state();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002);
        // The OpenAI handler allows up to 2048 items, but the internal
        // InputValidator enforces MAX_BATCH_SIZE=100. Use exactly 100 to test
        // the internal boundary that actually applies.
        let inputs: Vec<String> = (0..100).map(|i| format!("text{}", i)).collect();
        let req = OpenAIEmbedRequest {
            input: OpenAIInput::Multiple(inputs),
            model: "test-model".to_string(),
            encoding_format: None,
            dimensions: None,
            user: None,
        };

        let result = openai_embed_handler(State(state), ConnectInfo(addr), Json(req)).await;
        assert!(
            result.is_ok(),
            "exactly 100 inputs (MAX_BATCH_SIZE) should be accepted (boundary check)"
        );
    }
}
