// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 流水线请求处理函数

use crate::AppState;
use crate::domain::EmbedRequest;
use crate::error::VecboostError;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// 处理流水线请求
pub async fn handle_pipeline_request(
    state: AppState,
    req: EmbedRequest,
    ip: String,
) -> Result<axum::Json<crate::domain::EmbedResponse>, VecboostError> {
    // 生成请求 ID
    let request_id = Uuid::new_v4().to_string();

    // 创建响应通道
    let response_rx = state.response_channel.register(request_id.clone()).await;

    // 构建队列请求
    let priority = state
        .priority_calculator
        .calculate(crate::pipeline::PriorityInput {
            base_priority: crate::pipeline::Priority::Normal,
            time_until_timeout: Duration::from_secs(30),
            user_tier: None,
            source: crate::pipeline::RequestSource::http(ip.clone()),
            queue_length: state.pipeline_queue.size(),
        });

    let (tx, _) = oneshot::channel();

    let queued_request = crate::pipeline::QueuedRequest {
        request_id: request_id.clone(),
        embed_request: req,
        priority,
        submitted_at: std::time::Instant::now(),
        timeout: Duration::from_secs(30),
        source: crate::pipeline::RequestSource::http(ip),
        response_tx: tx,
    };

    // 提交到流水线队列
    state.pipeline_queue.enqueue(queued_request).await?;

    // 等待响应
    match tokio::time::timeout(Duration::from_secs(30), response_rx).await {
        Ok(Ok(Ok(response))) => Ok(axum::Json(response)),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(_)) => Err(VecboostError::InternalError(
            "Response channel error".to_string(),
        )),
        Err(_) => Err(VecboostError::ValidationError(
            "Request timeout".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, Precision};
    use crate::domain::EmbedRequest;
    use crate::engine::InferenceEngine;
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    };
    use crate::rate_limit::LimiteronAdapter;
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use std::sync::Arc;
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
            &Precision::Fp32
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

    struct ErrorEngine;

    #[async_trait]
    impl InferenceEngine for ErrorEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::InferenceError(
                "mock inference failure".to_string(),
            ))
        }
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Err(VecboostError::InferenceError(
                "mock batch inference failure".to_string(),
            ))
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
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

    fn create_test_state(
        queue_capacity: usize,
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    ) -> AppState {
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let queue = Arc::new(PriorityRequestQueue::new(queue_capacity));
        let response_channel = Arc::new(ResponseChannel::new());
        let priority_calculator = Arc::new(PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            WorkerConfig::default(),
            Arc::clone(&service),
        ));

        AppState {
            service,
            #[cfg(feature = "auth")]
            jwt_manager: None,
            #[cfg(feature = "auth")]
            user_store: None,
            auth_enabled: false,
            #[cfg(feature = "auth")]
            csrf_config: None,
            #[cfg(feature = "auth")]
            csrf_token_store: None,
            metrics_collector: None,
            prometheus_collector: None,
            rate_limiter: Arc::new(LimiteronAdapter::with_default_config()),
            ip_whitelist: vec![],
            rate_limit_enabled: false,
            audit_logger: None,
            pipeline_enabled: true,
            pipeline_queue: queue,
            response_channel,
            priority_calculator,
            worker_manager,
            kit: None,
        }
    }

    /// 验证 handle_pipeline_request 成功路径——入队、worker 处理、返回 EmbedResponse。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_success() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine);
        let queue = Arc::clone(&state.pipeline_queue);
        let response_channel = Arc::clone(&state.response_channel);
        let service = Arc::clone(&state.service);

        let consumer = tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(req) = queue.dequeue().await {
                    let request_id = req.request_id.clone();
                    let service_guard = service.read().await;
                    let result = service_guard.process_text(req.embed_request, None).await;
                    drop(service_guard);
                    response_channel.complete(request_id, result).await;
                    return;
                }
                if tokio::time::Instant::now() >= deadline {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_ok(), "handle_pipeline_request should succeed");
        let response = result.unwrap();
        assert_eq!(response.0.embedding.len(), 8);
        assert_eq!(response.0.dimension, 8);

        consumer.await.unwrap();
    }

    /// 验证 handle_pipeline_request 在队列满时返回 RateLimitExceeded。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_queue_full() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(0, engine);

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::RateLimitExceeded(msg) => {
                assert!(msg.contains("Queue is full"));
            }
            other => panic!("expected RateLimitExceeded, got {:?}", other),
        }
    }

    /// 验证 handle_pipeline_request 在引擎出错时传播 InferenceError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_engine_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(ErrorEngine));
        let state = create_test_state(100, engine);
        let queue = Arc::clone(&state.pipeline_queue);
        let response_channel = Arc::clone(&state.response_channel);
        let service = Arc::clone(&state.service);

        let consumer = tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(req) = queue.dequeue().await {
                    let request_id = req.request_id.clone();
                    let service_guard = service.read().await;
                    let result = service_guard.process_text(req.embed_request, None).await;
                    drop(service_guard);
                    response_channel.complete(request_id, result).await;
                    return;
                }
                if tokio::time::Instant::now() >= deadline {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }

        consumer.await.unwrap();
    }

    /// 验证 handle_pipeline_request 在 response channel sender 被 drop 时返回 InternalError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_response_channel_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine);
        let queue = Arc::clone(&state.pipeline_queue);
        let response_channel = Arc::clone(&state.response_channel);

        let consumer = tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(_req) = queue.dequeue().await {
                    response_channel.clear().await;
                    return;
                }
                if tokio::time::Instant::now() >= deadline {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InternalError(msg) => {
                assert!(msg.contains("Response channel error"));
            }
            other => panic!("expected InternalError, got {:?}", other),
        }

        consumer.await.unwrap();
    }
}
