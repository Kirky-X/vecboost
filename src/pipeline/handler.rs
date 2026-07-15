// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 流水线请求处理函数

use crate::VecboostState;
use crate::domain::EmbedRequest;
use crate::error::VecboostError;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// 处理流水线请求
pub async fn handle_pipeline_request(
    state: VecboostState,
    req: EmbedRequest,
    ip: String,
) -> Result<axum::Json<crate::domain::EmbedResponse>, VecboostError> {
    // 生成请求 ID
    let request_id = Uuid::new_v4().to_string();

    // 创建响应通道
    let response_rx = state
        .kit
        .require::<crate::module_registry::ResponseChannelModule>()
        .expect("ResponseChannelModule not registered")
        .register(request_id.clone())
        .await;

    // 构建队列请求
    let priority = state
        .kit
        .require::<crate::module_registry::PriorityCalculatorModule>()
        .expect("PriorityCalculatorModule not registered")
        .calculate(crate::pipeline::PriorityInput {
            base_priority: crate::pipeline::Priority::Normal,
            time_until_timeout: Duration::from_secs(30),
            user_tier: None,
            source: crate::pipeline::RequestSource::http(ip.clone()),
            queue_length: state
                .kit
                .require::<crate::module_registry::PipelineQueueModule>()
                .expect("PipelineQueueModule not registered")
                .size(),
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
    state
        .kit
        .require::<crate::module_registry::PipelineQueueModule>()
        .expect("PipelineQueueModule not registered")
        .enqueue(queued_request)
        .await?;

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

    async fn create_test_state(
        queue_capacity: usize,
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    ) -> VecboostState {
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
        let rate_limiter = Arc::new(LimiteronAdapter::with_default_config());

        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(service.clone());
        kit.set_config(rate_limiter.clone());
        kit.set_config(queue.clone());
        kit.set_config(response_channel.clone());
        kit.set_config(priority_calculator.clone());
        kit.set_config(worker_manager.clone());
        kit.set_config(Vec::<String>::new());
        kit.set_config(crate::module_registry::AuthEnabled(false));
        kit.set_config(crate::module_registry::RateLimitEnabled(false));
        kit.set_config(crate::module_registry::PipelineEnabled(true));
        kit.set_config(crate::module_registry::CacheConfig {
            enabled: false,
            size: 0,
        });
        kit.set_config(crate::module_registry::DbConfig { enabled: false });
        kit.set_config(None::<Arc<crate::audit::AuditLogger>>);
        kit.set_config(None::<Arc<crate::metrics::InferenceCollector>>);
        kit.set_config(None::<Arc<crate::metrics::PrometheusCollector>>);
        #[cfg(feature = "auth")]
        {
            kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
            kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
        }

        kit.register::<crate::module_registry::EmbeddingModule>()
            .unwrap();
        kit.register::<crate::module_registry::RateLimitModule>()
            .unwrap();
        kit.register::<crate::module_registry::CacheModule>()
            .unwrap();
        kit.register::<crate::module_registry::DbModule>().unwrap();
        kit.register::<crate::module_registry::AuditModule>()
            .unwrap();
        kit.register::<crate::module_registry::MetricsCollectorModule>()
            .unwrap();
        kit.register::<crate::module_registry::PrometheusCollectorModule>()
            .unwrap();
        kit.register::<crate::module_registry::IpWhitelistModule>()
            .unwrap();
        kit.register::<crate::module_registry::AuthEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::RateLimitEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::PipelineEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::PipelineQueueModule>()
            .unwrap();
        kit.register::<crate::module_registry::ResponseChannelModule>()
            .unwrap();
        kit.register::<crate::module_registry::PriorityCalculatorModule>()
            .unwrap();
        kit.register::<crate::module_registry::WorkerManagerModule>()
            .unwrap();
        #[cfg(feature = "auth")]
        {
            kit.register::<crate::module_registry::AuthModule>()
                .unwrap();
            kit.register::<crate::module_registry::UserStoreModule>()
                .unwrap();
            kit.register::<crate::module_registry::CsrfConfigModule>()
                .unwrap();
            kit.register::<crate::module_registry::CsrfTokenStoreModule>()
                .unwrap();
        }

        let kit = kit.build().await.expect("Failed to build AsyncKit");
        VecboostState { kit: Arc::new(kit) }
    }

    /// 验证 handle_pipeline_request 成功路径——入队、worker 处理、返回 EmbedResponse。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_success() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
        let state = create_test_state(0, engine).await;

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
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");

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

    /// 验证 handle_pipeline_request 在超时时返回 ValidationError。
    /// 使用 start_paused 模拟时间流逝,不启动 consumer 让 response 永远 pending。
    #[tokio::test(start_paused = true)]
    async fn test_handle_pipeline_request_timeout() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::ValidationError(msg) => {
                assert!(msg.contains("timeout"), "got: {}", msg);
            }
            other => panic!("expected ValidationError, got {:?}", other),
        }
    }

    /// 验证 handle_pipeline_request 在 normalize=false 时不做归一化。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_normalize_false() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(4)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            normalize: Some(false),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.0.dimension, 4);

        consumer.await.unwrap();
    }

    /// 验证 handle_pipeline_request 支持不同维度的引擎。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_different_dimensions() {
        for dim in &[1usize, 2, 16, 128] {
            let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
                Arc::new(RwLock::new(TestEngine::new(*dim)));
            let state = create_test_state(100, engine).await;
            let queue = state
                .kit
                .require::<crate::module_registry::PipelineQueueModule>()
                .expect("PipelineQueueModule not registered");
            let response_channel = state
                .kit
                .require::<crate::module_registry::ResponseChannelModule>()
                .expect("ResponseChannelModule not registered");
            let service = state
                .kit
                .require::<crate::module_registry::EmbeddingModule>()
                .expect("EmbeddingModule not registered");

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
                text: format!("test dim {}", dim),
                normalize: Some(true),
            };
            let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
            assert!(result.is_ok(), "dim {} should succeed", dim);
            let response = result.unwrap();
            assert_eq!(response.0.embedding.len(), *dim);
            assert_eq!(response.0.dimension, *dim);

            consumer.await.unwrap();
        }
    }

    /// 验证 handle_pipeline_request 空文本返回 ValidationError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_empty_text() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            text: "".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput for empty text, got {:?}", other),
        }

        consumer.await.unwrap();
    }

    /// 验证 handle_pipeline_request 空白文本返回 ValidationError。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_whitespace_text() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            text: "   ".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput for whitespace text, got {:?}", other),
        }

        consumer.await.unwrap();
    }

    /// Verify TestEngine::embed returns the expected dimension.
    #[test]
    fn test_test_engine_embed_returns_correct_dimension() {
        let engine = TestEngine::new(256);
        let result = engine.embed("test").expect("embed should succeed");
        assert_eq!(result.len(), 256);
        assert!(result.iter().all(|&v| v == 0.5));
    }

    /// Verify TestEngine::embed_batch returns one vector per input.
    #[test]
    fn test_test_engine_embed_batch_returns_correct_count() {
        let engine = TestEngine::new(128);
        let texts = vec!["a".to_string(), "b".to_string()];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 128);
    }

    /// Verify TestEngine::embed_batch with empty input returns empty vec.
    #[test]
    fn test_test_engine_embed_batch_empty() {
        let engine = TestEngine::new(64);
        let texts: Vec<String> = vec![];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert!(result.is_empty());
    }

    /// Verify TestEngine::precision returns Fp32.
    #[test]
    fn test_test_engine_precision() {
        let engine = TestEngine::new(8);
        assert_eq!(engine.precision(), &Precision::Fp32);
    }

    /// Verify TestEngine::supports_mixed_precision returns false.
    #[test]
    fn test_test_engine_supports_mixed_precision() {
        let engine = TestEngine::new(8);
        assert!(!engine.supports_mixed_precision());
    }

    /// Verify TestEngine::try_fallback_to_cpu returns Ok.
    #[tokio::test]
    async fn test_test_engine_try_fallback_to_cpu() {
        let mut engine = TestEngine::new(8);
        let config = ModelConfig::default();
        let result = engine.try_fallback_to_cpu(&config).await;
        assert!(result.is_ok());
    }

    /// Verify ErrorEngine::embed returns InferenceError.
    #[test]
    fn test_error_engine_embed_fails() {
        let engine = ErrorEngine;
        let result = engine.embed("test");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    /// Verify ErrorEngine::embed_batch returns InferenceError.
    #[test]
    fn test_error_engine_embed_batch_fails() {
        let engine = ErrorEngine;
        let texts = vec!["a".to_string(), "b".to_string()];
        let result = engine.embed_batch(&texts);
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock batch inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    /// Verify ErrorEngine::precision returns Fp32.
    #[test]
    fn test_error_engine_precision() {
        let engine = ErrorEngine;
        assert_eq!(engine.precision(), &Precision::Fp32);
    }

    /// Verify ErrorEngine::supports_mixed_precision returns false.
    #[test]
    fn test_error_engine_supports_mixed_precision() {
        let engine = ErrorEngine;
        assert!(!engine.supports_mixed_precision());
    }

    /// Verify ErrorEngine::try_fallback_to_cpu returns Ok.
    #[tokio::test]
    async fn test_error_engine_try_fallback_to_cpu() {
        let mut engine = ErrorEngine;
        let config = ModelConfig::default();
        let result = engine.try_fallback_to_cpu(&config).await;
        assert!(result.is_ok());
    }

    /// Verify handle_pipeline_request with a longer text succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_long_text() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            text: "a".repeat(1000),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "192.168.1.1".to_string()).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.0.embedding.len(), 8);

        consumer.await.unwrap();
    }

    /// Verify handle_pipeline_request with normalize=None uses default (true).
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_normalize_none() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(4)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            normalize: None,
        };
        let result = handle_pipeline_request(state, req, "10.0.0.1".to_string()).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.0.dimension, 4);

        consumer.await.unwrap();
    }

    /// Verify handle_pipeline_request with different IP addresses works.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_different_ips() {
        for ip in &["127.0.0.1", "192.168.0.1", "10.0.0.1", "172.16.0.1"] {
            let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
                Arc::new(RwLock::new(TestEngine::new(4)));
            let state = create_test_state(100, engine).await;
            let queue = state
                .kit
                .require::<crate::module_registry::PipelineQueueModule>()
                .expect("PipelineQueueModule not registered");
            let response_channel = state
                .kit
                .require::<crate::module_registry::ResponseChannelModule>()
                .expect("ResponseChannelModule not registered");
            let service = state
                .kit
                .require::<crate::module_registry::EmbeddingModule>()
                .expect("EmbeddingModule not registered");

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
                text: format!("test from {}", ip),
                normalize: Some(true),
            };
            let result = handle_pipeline_request(state, req, ip.to_string()).await;
            assert!(result.is_ok(), "request from {} should succeed", ip);

            consumer.await.unwrap();
        }
    }

    /// Verify handle_pipeline_request generates unique request IDs.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_unique_request_ids() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(4)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

        let consumer = tokio::spawn(async move {
            let mut ids = Vec::new();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(req) = queue.dequeue().await {
                    ids.push(req.request_id.clone());
                    let request_id = req.request_id.clone();
                    let service_guard = service.read().await;
                    let result = service_guard.process_text(req.embed_request, None).await;
                    drop(service_guard);
                    response_channel.complete(request_id, result).await;
                    if ids.len() >= 2 {
                        assert_ne!(ids[0], ids[1], "request IDs should be unique");
                        return;
                    }
                }
                if tokio::time::Instant::now() >= deadline {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        // Send first request
        let req1 = EmbedRequest {
            text: "first".to_string(),
            normalize: Some(true),
        };
        let state1 = state.clone();
        let handle1 = tokio::spawn(async move {
            handle_pipeline_request(state1, req1, "127.0.0.1".to_string()).await
        });

        // Send second request
        let req2 = EmbedRequest {
            text: "second".to_string(),
            normalize: Some(true),
        };
        let handle2 = tokio::spawn(async move {
            handle_pipeline_request(state, req2, "127.0.0.1".to_string()).await
        });

        let r1 = handle1.await.unwrap();
        let r2 = handle2.await.unwrap();
        assert!(r1.is_ok());
        assert!(r2.is_ok());

        consumer.await.unwrap();
    }

    /// Verify handle_pipeline_request with unicode text succeeds.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_handle_pipeline_request_unicode_text() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let state = create_test_state(100, engine).await;
        let queue = state
            .kit
            .require::<crate::module_registry::PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let response_channel = state
            .kit
            .require::<crate::module_registry::ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let service = state
            .kit
            .require::<crate::module_registry::EmbeddingModule>()
            .expect("EmbeddingModule not registered");

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
            text: "你好世界 🌍 Привет мир".to_string(),
            normalize: Some(true),
        };
        let result = handle_pipeline_request(state, req, "127.0.0.1".to_string()).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.0.embedding.len(), 8);

        consumer.await.unwrap();
    }
}
