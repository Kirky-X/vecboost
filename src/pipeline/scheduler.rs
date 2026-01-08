// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::Arc;
use tracing::debug;

use super::priority::PriorityCalculator;
use super::queue::QueuedRequest;
use super::response_channel::ResponseChannel;
use super::worker::WorkerManager;
use crate::domain::EmbedResponse;
use crate::error::AppError;

/// 流水线调度器
pub struct PipelineScheduler {
    /// 优先级计算器
    priority_calculator: PriorityCalculator,
    /// 响应通道
    response_channel: Arc<ResponseChannel>,
    /// Worker 管理器
    worker_manager: Arc<WorkerManager>,
}

impl PipelineScheduler {
    pub fn new(
        priority_calculator: PriorityCalculator,
        response_channel: Arc<ResponseChannel>,
        worker_manager: Arc<WorkerManager>,
    ) -> Self {
        debug!("Creating PipelineScheduler");

        Self {
            priority_calculator,
            response_channel,
            worker_manager,
        }
    }

    /// 处理请求
    pub async fn process_request(&self, request: QueuedRequest) -> Result<EmbedResponse, AppError> {
        debug!("Processing request {}", request.request_id);

        // TODO: 实际的请求处理逻辑
        // 这里应该调用 EmbeddingService

        // 模拟处理
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        Ok(EmbedResponse {
            embedding: vec![0.0; 768],
            dimension: 768,
            processing_time_ms: 10,
        })
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

    #[tokio::test]
    async fn test_scheduler_creation() {
        let priority_calculator =
            PriorityCalculator::new(crate::pipeline::config::PriorityConfig::default());
        let response_channel = Arc::new(ResponseChannel::new());
        let worker_manager = Arc::new(WorkerManager::new(
            Arc::new(crate::pipeline::queue::PriorityRequestQueue::new(100)),
            response_channel.clone(),
            crate::pipeline::worker::WorkerConfig::default(),
        ));

        let scheduler =
            PipelineScheduler::new(priority_calculator, response_channel, worker_manager);

        assert_eq!(scheduler.worker_manager().current_workers(), 0);
    }
}
