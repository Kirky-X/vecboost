// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use super::priority::PriorityCalculator;
use super::queue::QueuedRequest;
use super::response_channel::ResponseChannel;
use super::worker::WorkerManager;
use crate::domain::EmbedResponse;
use crate::error::VecboostError;
use crate::service::embedding::EmbeddingService;

/// 流水线调度器
pub struct PipelineScheduler {
    /// 优先级计算器
    priority_calculator: PriorityCalculator,
    /// 响应通道
    response_channel: Arc<ResponseChannel>,
    /// Worker 管理器
    worker_manager: Arc<WorkerManager>,
    /// 嵌入服务
    service: Arc<RwLock<EmbeddingService>>,
}

impl PipelineScheduler {
    pub fn new(
        priority_calculator: PriorityCalculator,
        response_channel: Arc<ResponseChannel>,
        worker_manager: Arc<WorkerManager>,
        service: Arc<RwLock<EmbeddingService>>,
    ) -> Self {
        debug!("Creating PipelineScheduler");

        Self {
            priority_calculator,
            response_channel,
            worker_manager,
            service,
        }
    }

    /// 处理请求
    pub async fn process_request(
        &self,
        request: QueuedRequest,
    ) -> Result<EmbedResponse, VecboostError> {
        debug!("Processing request {}", request.request_id);

        let service = self.service.read().await;
        service.process_text(request.embed_request, None).await
    }

    /// 获取 Worker 管理器
    pub fn worker_manager(&self) -> Arc<WorkerManager> {
        Arc::clone(&self.worker_manager)
    }

    /// 获取响应通道
    pub fn response_channel(&self) -> Arc<ResponseChannel> {
        Arc::clone(&self.response_channel)
    }

    /// 获取优先级计算器
    pub fn priority_calculator(&self) -> &PriorityCalculator {
        &self.priority_calculator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, Precision};
    use crate::domain::EmbedRequest;
    use crate::engine::InferenceEngine;
    use crate::pipeline::config::{PriorityConfig, WorkerConfig};
    use crate::pipeline::priority::{Priority, RequestSource};
    use crate::pipeline::queue::PriorityRequestQueue;
    use async_trait::async_trait;
    use std::time::{Duration, Instant};

    /// 测试用 mock 引擎,返回固定维度的向量
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

    fn create_test_service() -> Arc<RwLock<EmbeddingService>> {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(384)));
        Arc::new(RwLock::new(EmbeddingService::new(engine, None)))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_scheduler_creation() {
        let priority_calculator = PriorityCalculator::new(PriorityConfig::default());
        let response_channel = Arc::new(ResponseChannel::new());
        let service = create_test_service();

        let worker_manager = Arc::new(WorkerManager::new(
            Arc::new(PriorityRequestQueue::new(100)),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        let scheduler = PipelineScheduler::new(
            priority_calculator,
            response_channel,
            worker_manager,
            service,
        );

        assert_eq!(scheduler.worker_manager().current_workers(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_request_success() {
        let priority_calculator = PriorityCalculator::new(PriorityConfig::default());
        let response_channel = Arc::new(ResponseChannel::new());
        let service = create_test_service();

        let worker_manager = Arc::new(WorkerManager::new(
            Arc::new(PriorityRequestQueue::new(100)),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        let scheduler = PipelineScheduler::new(
            priority_calculator,
            response_channel,
            worker_manager,
            service,
        );

        let request = QueuedRequest {
            request_id: "test-001".to_string(),
            embed_request: EmbedRequest {
                text: "hello world".to_string(),
                normalize: None,
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tokio::sync::oneshot::channel().0,
        };

        let result = scheduler.process_request(request).await;
        assert!(result.is_ok(), "process_request should succeed");
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
        assert_eq!(response.embedding.len(), 384);

        // process_text applies L2 normalization unconditionally.
        // For vec![0.5; 384], norm = sqrt(384 * 0.25) = sqrt(96) ≈ 9.798.
        // After normalization, each value = 0.5 / sqrt(96) ≈ 0.051031.
        let expected = 0.5f32 / (384f32 * 0.25f32).sqrt();
        assert!(
            response
                .embedding
                .iter()
                .all(|&v| (v - expected).abs() < 1e-6),
            "all values should equal L2-normalized 0.5, got {:?}",
            &response.embedding[..5.min(response.embedding.len())]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_request_empty_text() {
        let priority_calculator = PriorityCalculator::new(PriorityConfig::default());
        let response_channel = Arc::new(ResponseChannel::new());
        let service = create_test_service();

        let worker_manager = Arc::new(WorkerManager::new(
            Arc::new(PriorityRequestQueue::new(100)),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        let scheduler = PipelineScheduler::new(
            priority_calculator,
            response_channel,
            worker_manager,
            service,
        );

        let request = QueuedRequest {
            request_id: "test-002".to_string(),
            embed_request: EmbedRequest {
                text: "".to_string(),
                normalize: None,
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tokio::sync::oneshot::channel().0,
        };

        let result = scheduler.process_request(request).await;
        assert!(result.is_err(), "empty text should return validation error");
    }
}
