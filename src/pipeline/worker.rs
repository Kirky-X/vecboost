// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, trace};

use super::queue::PriorityRequestQueue;
use super::response_channel::ResponseChannel;
use crate::domain::EmbedResponse;
use crate::error::AppError;

/// Worker 任务
#[derive(Debug)]
pub enum WorkerTask {
    ProcessRequest {
        request_id: String,
        text: String,
        normalize: Option<bool>,
    },
    Shutdown,
}

/// Worker 配置
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// 最小 Worker 数量
    pub min_workers: usize,
    /// 最大 Worker 数量
    pub max_workers: usize,
    /// 扩容阈值
    pub scale_up_threshold: usize,
    /// 缩容阈值
    pub scale_down_threshold: usize,
    /// 空闲超时（秒）
    pub idle_timeout_secs: u64,
    /// 检查间隔（秒）
    pub scale_check_interval_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            scale_up_threshold: 100,
            scale_down_threshold: 10,
            idle_timeout_secs: 60,
            scale_check_interval_secs: 5,
        }
    }
}

/// Worker 管理器
pub struct WorkerManager {
    /// 最小 Worker 数量
    min_workers: usize,
    /// 最大 Worker 数量
    max_workers: usize,
    /// 当前 Worker 数量
    current_workers: Arc<AtomicUsize>,
    /// 请求队列
    queue: Arc<PriorityRequestQueue>,
    /// 响应通道
    response_channel: Arc<ResponseChannel>,
    /// 任务发送器
    task_sender: mpsc::Sender<WorkerTask>,
    /// 任务接收器
    task_receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<WorkerTask>>>,
    /// 配置
    config: WorkerConfig,
}

impl WorkerManager {
    pub fn new(
        queue: Arc<PriorityRequestQueue>,
        response_channel: Arc<ResponseChannel>,
        config: WorkerConfig,
    ) -> Self {
        let (task_sender, task_receiver) = mpsc::channel(1000);

        Self {
            min_workers: config.min_workers,
            max_workers: config.max_workers,
            current_workers: Arc::new(AtomicUsize::new(0)),
            queue,
            response_channel,
            task_sender,
            task_receiver: Arc::new(tokio::sync::Mutex::new(task_receiver)),
            config,
        }
    }

    /// 启动 Worker 管理器
    pub async fn start(&mut self) -> Result<(), AppError> {
        info!(
            "Starting WorkerManager with min_workers={}, max_workers={}",
            self.min_workers, self.max_workers
        );

        // 启动初始 Workers
        for _ in 0..self.min_workers {
            self.spawn_worker().await;
        }

        // 启动扩缩容监控任务
        self.start_scaling_monitor().await;

        info!("WorkerManager started successfully");

        Ok(())
    }

    /// 启动 Worker
    async fn spawn_worker(&self) {
        let worker_id = self.current_workers.fetch_add(1, Ordering::SeqCst);
        let queue = Arc::clone(&self.queue);
        let response_channel = Arc::clone(&self.response_channel);

        info!("Worker {} started", worker_id);

        tokio::spawn(async move {
            Self::worker_loop(worker_id, queue, response_channel).await;
        });
    }

    /// Worker 循环
    async fn worker_loop(
        worker_id: usize,
        queue: Arc<PriorityRequestQueue>,
        response_channel: Arc<ResponseChannel>,
    ) {
        debug!("Worker {} loop started", worker_id);

        let mut idle_count: usize = 0;
        const MAX_IDLE_COUNT: usize = 10; // 最大空闲计数

        loop {
            // 从队列获取请求
            let request = match queue.dequeue().await {
                Some(req) => {
                    // 重置空闲计数
                    idle_count = 0;
                    req
                }
                None => {
                    // 队列为空，使用指数退避
                    idle_count = idle_count.saturating_add(1usize);
                    let wait_ms =
                        std::cmp::min(100usize * (1usize << idle_count.min(6)), 5000usize); // 最大 5 秒

                    debug!(
                        "Worker {} queue empty, waiting {}ms (idle_count={})",
                        worker_id, wait_ms, idle_count
                    );

                    tokio::time::sleep(Duration::from_millis(wait_ms as u64)).await;

                    // 如果长时间空闲，可以让 worker 退出或休眠
                    if idle_count > MAX_IDLE_COUNT {
                        trace!(
                            "Worker {} idle for too long, considering shutdown",
                            worker_id
                        );
                        // TODO: 实现 worker 优雅关闭机制
                    }

                    continue;
                }
            };

            debug!(
                "Worker {} processing request {}",
                worker_id, request.request_id
            );

            // 处理请求
            let result = Self::process_request(&request).await;

            // 发送响应
            response_channel
                .complete(request.request_id.clone(), result)
                .await;

            debug!(
                "Worker {} completed request {}",
                worker_id, request.request_id
            );
        }
    }

    /// 处理请求
    async fn process_request(
        _request: &super::queue::QueuedRequest,
    ) -> Result<EmbedResponse, AppError> {
        // TODO: 实际的请求处理逻辑
        // 这里应该调用 EmbeddingService

        // 模拟处理
        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(EmbedResponse {
            embedding: vec![0.0; 768],
            dimension: 768,
            processing_time_ms: 10,
        })
    }

    /// 启动扩缩容监控
    async fn start_scaling_monitor(&self) {
        let queue = Arc::clone(&self.queue);
        let current_workers = Arc::clone(&self.current_workers);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.scale_check_interval_secs));

            loop {
                interval.tick().await;

                let queue_size = queue.size();
                let current = current_workers.load(Ordering::SeqCst);

                // 检查是否需要扩容
                if queue_size > config.scale_up_threshold && current < config.max_workers {
                    let new_workers = std::cmp::min(
                        (queue_size / config.scale_up_threshold).saturating_sub(1),
                        config.max_workers - current,
                    );

                    info!(
                        "Scaling up: adding {} workers (queue size: {})",
                        new_workers, queue_size
                    );

                    for _ in 0..new_workers {
                        current_workers.fetch_add(1, Ordering::SeqCst);
                        // TODO: 实际启动 worker
                    }
                }

                // 检查是否需要缩容
                if queue_size < config.scale_down_threshold && current > config.min_workers {
                    info!(
                        "Scaling down: removing workers (queue size: {})",
                        queue_size
                    );
                    // TODO: 实际移除 worker
                }
            }
        });
    }

    /// 获取当前 Worker 数量
    pub fn current_workers(&self) -> usize {
        self.current_workers.load(Ordering::Relaxed)
    }

    /// 停止所有 Workers
    pub async fn stop(&self) {
        info!("Stopping WorkerManager...");

        // 发送关闭信号
        for _ in 0..self.current_workers.load(Ordering::Relaxed) {
            let _ = self.task_sender.send(WorkerTask::Shutdown).await;
        }

        info!("WorkerManager stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_manager_creation() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();

        let manager = WorkerManager::new(queue, response_channel, config);

        assert_eq!(manager.current_workers(), 0);
    }

    #[tokio::test]
    async fn test_worker_manager_start() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 2,
            max_workers: 4,
            ..Default::default()
        };

        let mut manager = WorkerManager::new(queue, response_channel, config);

        manager.start().await.unwrap();

        assert_eq!(manager.current_workers(), 2);
    }
}
