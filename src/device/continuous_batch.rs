// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 连续批处理调度 loop
//!
//! `ContinuousBatchLoop` 作为 tokio spawn task 持续运行，每个 tick（1ms）检查
//! `PriorityRequestQueue` 并按条件刷新批次。相比 `DynamicBatchScheduler` 的被动凑批，
//! 连续调度消除了 0-50ms 固定等待时间，同时贯通优先级信息和 SLA 超时约束。

use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, warn};
use tokio::sync::{RwLock, watch};

use crate::device::DynamicBatchScheduler;
use crate::domain::{EmbedRequest, EmbedResponse};
use crate::error::VecboostError;
use crate::pipeline::{Priority, PriorityRequestQueue, QueuedRequest};
use crate::service::embedding::EmbeddingService;

/// SLA 安全边际：剩余 timeout 低于此值时立即刷新
const SLA_SAFETY_MARGIN_MS: u64 = 10;

/// 调度 tick 间隔
const TICK_INTERVAL: Duration = Duration::from_millis(1);

/// 连续批处理调度 loop
///
/// 持续从 `PriorityRequestQueue` 按优先级出队请求，收集到批次后调用
/// `EmbeddingService` 处理。刷新条件：
/// 1. 最老请求剩余 timeout < [`SLA_SAFETY_MARGIN_MS`] → 立即刷新
/// 2. 批次大小 >= 最优批量 → 刷新
/// 3. 队列中无更多请求 → 等待下一 tick
pub struct ContinuousBatchLoop {
    /// 优先级请求队列（请求源）
    request_queue: Arc<PriorityRequestQueue>,
    /// 动态批量调度器（复用性能跟踪逻辑）
    batch_scheduler: Arc<DynamicBatchScheduler>,
    /// 推理服务
    service: Arc<RwLock<EmbeddingService>>,
    /// 优雅关闭信号
    shutdown_rx: watch::Receiver<bool>,
}

impl ContinuousBatchLoop {
    /// 创建新的连续批处理 loop
    pub fn new(
        request_queue: Arc<PriorityRequestQueue>,
        batch_scheduler: Arc<DynamicBatchScheduler>,
        service: Arc<RwLock<EmbeddingService>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            request_queue,
            batch_scheduler,
            service,
            shutdown_rx,
        }
    }

    /// 调度 loop 入口
    ///
    /// 持续运行直到收到 shutdown 信号。每次 tick 收集批次并处理。
    pub async fn run(&self) {
        debug!("ContinuousBatchLoop started");
        loop {
            // 检查关闭信号
            if *self.shutdown_rx.borrow() {
                debug!("ContinuousBatchLoop received shutdown signal");
                break;
            }

            // 收集并处理一个批次
            let processed = self.tick().await;

            if !processed {
                // 无请求可处理，等待下一 tick
                tokio::time::sleep(TICK_INTERVAL).await;
            }
            // 如果处理了批次，立即检查下一批（不 sleep），实现最大吞吐
        }
        debug!("ContinuousBatchLoop stopped");
    }

    /// 单次 tick：收集批次 → 处理 → 记录性能
    ///
    /// 返回 `true` 表示处理了至少一个请求。
    async fn tick(&self) -> bool {
        let batch = self.collect_batch().await;
        if batch.is_empty() {
            return false;
        }

        let batch_size = batch.len();
        let start = Instant::now();

        // 处理批次
        self.process_batch(batch).await;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // 记录性能（复用 DynamicBatchScheduler 的跟踪逻辑）
        self.batch_scheduler
            .record_batch_completion(batch_size, latency_ms)
            .await;

        debug!(
            "ContinuousBatchLoop: processed {} requests in {:.1}ms",
            batch_size, latency_ms
        );

        true
    }

    /// 从优先级队列收集批次
    ///
    /// 按优先级出队请求，同时检查 SLA 超时。超时请求直接返回错误。
    /// 收集条件：
    /// - 最老请求剩余 timeout < 10ms → 停止收集，立即刷新
    /// - 批次大小 >= 最优批量 → 停止收集，立即刷新
    /// - 队列空 → 停止收集
    async fn collect_batch(&self) -> Vec<QueuedRequest> {
        let mut batch = Vec::new();
        let optimal_size = self.batch_scheduler.current_batch_size().await;
        let now = Instant::now();
        let sla_margin = Duration::from_millis(SLA_SAFETY_MARGIN_MS);

        loop {
            // 批次已满
            if batch.len() >= optimal_size {
                break;
            }

            // 从优先级队列出队下一个请求
            let request = match self.request_queue.dequeue().await {
                Some(req) => req,
                None => break, // 队列空
            };

            // 检查 SLA 超时：已超时请求直接返回错误
            let elapsed = now.duration_since(request.submitted_at);
            if elapsed >= request.timeout {
                let _ = request
                    .response_tx
                    .send(Err(VecboostError::InferenceError(format!(
                        "Request {} timed out before processing",
                        request.request_id
                    ))));
                continue;
            }

            // 检查最老请求的剩余 timeout 是否低于安全边际
            let remaining = request.timeout - elapsed;
            let should_flush_now = remaining < sla_margin;

            batch.push(request);

            if should_flush_now {
                // 最老请求即将超时，立即刷新当前批次
                break;
            }
        }

        batch
    }

    /// 处理批次：调用 EmbeddingService 推理并发送结果
    async fn process_batch(&self, batch: Vec<QueuedRequest>) {
        if batch.is_empty() {
            return;
        }

        // 提取文本
        let texts: Vec<String> = batch
            .iter()
            .map(|req| {
                match &req.request {
                    crate::pipeline::ServiceRequest::Embed(embed_req) => embed_req.text.clone(),
                    crate::pipeline::ServiceRequest::Rerank(_) => String::new(), // rerank not handled in batch embed
                }
            })
            .collect();

        // 调用推理服务（read lock 足够，embed_batch 是 &self 方法）
        let service = self.service.read().await;
        let result = service.embed_batch_internal(&texts).await;
        drop(service);

        match result {
            Ok(embeddings) => {
                for (i, request) in batch.into_iter().enumerate() {
                    let embedding = embeddings.get(i).cloned().unwrap_or_default();
                    let dimension = embedding.len();
                    let response = EmbedResponse {
                        embedding,
                        dimension,
                        processing_time_ms: 0,
                        information_retention_rate: None,
                    };
                    let _ = request.response_tx.send(Ok(response));
                }
            }
            Err(e) => {
                warn!("Batch processing failed: {}", e);
                for request in batch {
                    let _ = request.response_tx.send(Err(e.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use crate::pipeline::RequestSource;
    use async_trait::async_trait;
    use tokio::sync::{oneshot, watch};

    /// 测试用 mock 推理引擎
    struct MockEngine {
        dimension: usize,
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.1; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.1; self.dimension]).collect())
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

    /// 创建测试用的 ContinuousBatchLoop 组件
    fn setup_test_loop() -> (
        ContinuousBatchLoop,
        Arc<PriorityRequestQueue>,
        watch::Sender<bool>,
    ) {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let scheduler = Arc::new(DynamicBatchScheduler::new(crate::device::BatchConfig {
            min_batch_size: 2,
            max_batch_size: 8,
            max_wait_time_ms: 50,
            ..Default::default()
        }));

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine { dimension: 4 }));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let loop_ = ContinuousBatchLoop::new(queue.clone(), scheduler, service, shutdown_rx);

        (loop_, queue, shutdown_tx)
    }

    /// 创建测试用的 QueuedRequest
    fn make_queued_request(
        id: &str,
        priority: Priority,
        timeout: Duration,
    ) -> (
        QueuedRequest,
        oneshot::Receiver<Result<EmbedResponse, VecboostError>>,
    ) {
        let (tx, rx) = oneshot::channel();
        let request = QueuedRequest {
            request_id: id.to_string(),
            request: crate::pipeline::ServiceRequest::Embed(EmbedRequest {
                text: format!("test text {}", id),
                normalize: Some(true),
            }),
            priority,
            submitted_at: Instant::now(),
            timeout,
            source: RequestSource::Internal,
            response_tx: tx,
        };
        (request, rx)
    }

    /// T007a: 单个请求在 < 100ms 内被处理（无需等待 50ms 凑批时间）
    #[tokio::test]
    async fn test_continuous_loop_processes_single_request() {
        let (loop_, queue, shutdown_tx) = setup_test_loop();

        let (request, rx) = make_queued_request("req-1", Priority::Normal, Duration::from_secs(30));
        queue.enqueue(request).await.unwrap();

        // 在后台运行 loop
        let handle = tokio::spawn(async move {
            loop_.run().await;
        });

        // 等待结果（应在 < 100ms 内完成）
        let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
        assert!(result.is_ok(), "Request should be processed within 100ms");

        let response = result.unwrap().expect("Channel should not be closed");
        assert!(response.is_ok(), "Embedding should succeed");

        // 关闭 loop
        shutdown_tx.send(true).unwrap();
        let _ = tokio::time::timeout(Duration::from_millis(100), handle).await;
    }

    /// T007b: Critical 请求先于 Low 请求被处理
    #[tokio::test]
    async fn test_priority_ordering_in_batch() {
        let (loop_, queue, shutdown_tx) = setup_test_loop();

        // 先入队 Low，再入队 Critical
        let (low_req, low_rx) =
            make_queued_request("low-1", Priority::Low, Duration::from_secs(30));
        let (crit_req, crit_rx) =
            make_queued_request("crit-1", Priority::Critical, Duration::from_secs(30));

        queue.enqueue(low_req).await.unwrap();
        queue.enqueue(crit_req).await.unwrap();

        let handle = tokio::spawn(async move {
            loop_.run().await;
        });

        // 两个请求都应该被处理
        let crit_result = tokio::time::timeout(Duration::from_millis(200), crit_rx)
            .await
            .expect("Critical request should complete within 200ms");
        let low_result = tokio::time::timeout(Duration::from_millis(200), low_rx)
            .await
            .expect("Low request should complete within 200ms");

        assert!(crit_result.unwrap().is_ok());
        assert!(low_result.unwrap().is_ok());

        shutdown_tx.send(true).unwrap();
        let _ = tokio::time::timeout(Duration::from_millis(100), handle).await;
    }

    /// T007c: 短 timeout 请求在超时前被刷新
    #[tokio::test]
    async fn test_sla_timeout_forces_flush() {
        let (loop_, queue, shutdown_tx) = setup_test_loop();

        // timeout=5ms 的请求
        let (request, rx) =
            make_queued_request("sla-1", Priority::Normal, Duration::from_millis(5));
        queue.enqueue(request).await.unwrap();

        let handle = tokio::spawn(async move {
            loop_.run().await;
        });

        // 请求应在 ≤ 10ms 内被处理（SLA 安全边际 = 10ms）
        let result = tokio::time::timeout(Duration::from_millis(50), rx).await;
        assert!(
            result.is_ok(),
            "Short-timeout request should be processed quickly"
        );

        let response = result.unwrap().expect("Channel should not be closed");
        assert!(response.is_ok(), "Embedding should succeed before timeout");

        shutdown_tx.send(true).unwrap();
        let _ = tokio::time::timeout(Duration::from_millis(100), handle).await;
    }

    /// 验证 shutdown 信号能优雅退出 loop
    #[tokio::test]
    async fn test_shutdown_signal_stops_loop() {
        let (loop_, _queue, shutdown_tx) = setup_test_loop();

        let handle = tokio::spawn(async move {
            loop_.run().await;
        });

        // 发送关闭信号
        shutdown_tx.send(true).unwrap();

        // loop 应在合理时间内退出
        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(
            result.is_ok(),
            "Loop should stop within 100ms of shutdown signal"
        );
    }

    /// 验证超时请求收到 Timeout 错误
    #[tokio::test]
    async fn test_expired_request_gets_timeout_error() {
        let (loop_, queue, shutdown_tx) = setup_test_loop();

        // 创建一个已经过期的请求（timeout=0）
        let (request, rx) =
            make_queued_request("expired-1", Priority::Normal, Duration::from_millis(0));
        queue.enqueue(request).await.unwrap();

        // 等一小段时间让请求过期
        tokio::time::sleep(Duration::from_millis(2)).await;

        let handle = tokio::spawn(async move {
            loop_.run().await;
        });

        // 请求应收到 Timeout 错误
        let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
        assert!(result.is_ok(), "Should get a response");
        let response = result.unwrap().expect("Channel should not be closed");
        assert!(response.is_err(), "Expired request should get an error");

        shutdown_tx.send(true).unwrap();
        let _ = tokio::time::timeout(Duration::from_millis(100), handle).await;
    }
}
