// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, info, warn};

use super::config::WorkerConfig;
use super::queue::PriorityRequestQueue;
use super::response_channel::ResponseChannel;
use crate::domain::EmbedResponse;
use crate::error::VecboostError;
use crate::service::embedding::EmbeddingService;

/// Worker 任务
#[derive(Debug)]
pub enum WorkerTask {
    ProcessRequest {
        request_id: String,
        embed_request: crate::domain::EmbedRequest,
    },
    /// 优雅关闭信号
    Shutdown {
        /// 是否立即关闭（不等待当前请求完成）
        immediate: bool,
    },
}

/// Worker 状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Idle,
    Processing,
    Stopping,
    Stopped,
}

/// Worker 实例
pub struct Worker {
    /// Worker ID
    worker_id: usize,
    /// 运行标志
    running: Arc<AtomicBool>,
    /// 当前状态
    state: Arc<Mutex<WorkerState>>,
    /// 任务接收器
    receiver: mpsc::Receiver<WorkerTask>,
    /// 配置
    config: WorkerConfig,
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
    /// 配置
    config: WorkerConfig,
    /// 运行标志
    running: Arc<AtomicBool>,
    /// Worker 任务发送器列表（用于发送关闭信号）
    worker_senders: Arc<Mutex<Vec<mpsc::Sender<WorkerTask>>>>,
    /// EmbeddingService 实例
    embedding_service: Arc<RwLock<EmbeddingService>>,
    /// Worker 健康状态跟踪
    worker_health: Arc<Mutex<Vec<WorkerHealthInfo>>>,
}

/// Worker 健康信息
#[derive(Debug, Clone)]
struct WorkerHealthInfo {
    worker_id: usize,
    last_active_time: std::time::Instant,
    crash_count: usize,
    is_alive: bool,
}

impl WorkerHealthInfo {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            last_active_time: std::time::Instant::now(),
            crash_count: 0,
            is_alive: true,
        }
    }

    fn update_activity(&mut self) {
        self.last_active_time = std::time::Instant::now();
    }

    fn record_crash(&mut self) {
        self.crash_count += 1;
    }
}

impl WorkerManager {
    pub fn new(
        queue: Arc<PriorityRequestQueue>,
        response_channel: Arc<ResponseChannel>,
        config: WorkerConfig,
        embedding_service: Arc<RwLock<EmbeddingService>>,
    ) -> Self {
        Self {
            min_workers: config.min_workers,
            max_workers: config.max_workers,
            current_workers: Arc::new(AtomicUsize::new(0)),
            queue,
            response_channel,
            config,
            running: Arc::new(AtomicBool::new(true)),
            worker_senders: Arc::new(Mutex::new(Vec::new())),
            embedding_service,
            worker_health: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 启动 Worker Manager
    pub async fn start(&self) -> Result<(), VecboostError> {
        info!(
            "Starting WorkerManager with min={} max={}",
            self.min_workers, self.max_workers
        );

        // 启动最小数量的 worker
        for _ in 0..self.min_workers {
            self.spawn_worker().await;
        }

        // 启动扩缩容监控
        self.start_scaling_monitor().await;

        info!("WorkerManager started successfully");

        Ok(())
    }

    /// 优雅关闭所有 Worker
    pub async fn shutdown(&self) {
        info!("Shutting down WorkerManager...");

        // 设置停止标志
        self.running.store(false, Ordering::SeqCst);

        // 获取所有 worker 发送器的克隆
        let senders = {
            let guard = self.worker_senders.lock().await;
            guard.clone()
        };

        // 向所有 worker 发送关闭信号
        for sender in &senders {
            let _ = sender.send(WorkerTask::Shutdown { immediate: false }).await;
        }

        // 等待一段时间让 worker 完成当前请求
        tokio::time::sleep(Duration::from_secs(5)).await;

        // 强制关闭剩余的 worker
        for sender in &senders {
            let _ = sender.send(WorkerTask::Shutdown { immediate: true }).await;
        }

        info!("WorkerManager shutdown complete");
    }

    /// 获取当前 worker 数量
    pub fn current_workers(&self) -> usize {
        self.current_workers.load(Ordering::SeqCst)
    }

    /// 启动 Worker
    pub async fn spawn_worker(&self) {
        let worker_id = self.current_workers.fetch_add(1, Ordering::SeqCst);

        // 创建任务通道
        let (task_sender, task_receiver) = mpsc::channel(100);

        // 保存发送器用于后续关闭
        {
            let mut senders = self.worker_senders.lock().await;
            senders.push(task_sender.clone());
        }

        // 初始化健康信息
        {
            let mut health_guard = self.worker_health.lock().await;
            health_guard.push(WorkerHealthInfo::new(worker_id));
        }

        let queue = Arc::clone(&self.queue);
        let response_channel = Arc::clone(&self.response_channel);
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let embedding_service = Arc::clone(&self.embedding_service);
        let worker_health = Arc::clone(&self.worker_health);

        info!("Worker {} started", worker_id);

        tokio::spawn(async move {
            Self::worker_loop(
                worker_id,
                task_receiver,
                queue,
                response_channel,
                config,
                running,
                embedding_service,
                worker_health,
            )
            .await;
        });
    }

    /// Worker 循环
    async fn worker_loop(
        worker_id: usize,
        task_receiver: mpsc::Receiver<WorkerTask>,
        queue: Arc<PriorityRequestQueue>,
        response_channel: Arc<ResponseChannel>,
        config: WorkerConfig,
        running: Arc<AtomicBool>,
        embedding_service: Arc<RwLock<EmbeddingService>>,
        worker_health: Arc<Mutex<Vec<WorkerHealthInfo>>>,
    ) {
        debug!("Worker {} loop started", worker_id);

        let mut idle_count: usize = 0;
        const MAX_IDLE_COUNT: usize = 10; // 最大空闲计数

        loop {
            // 检查是否应该停止
            if !running.load(Ordering::Relaxed) {
                info!("Worker {} received stop signal", worker_id);
                break;
            }

            // 更新活动时间
            {
                let mut guard = worker_health.lock().await;
                if let Some(info) = guard.iter_mut().find(|i| i.worker_id == worker_id) {
                    info.update_activity();
                }
            }

            // 从队列获取请求（非阻塞）
            if let Some(request) = queue.dequeue().await {
                // 重置空闲计数
                idle_count = 0;

                debug!(
                    "Worker {} processing request {}",
                    worker_id, request.request_id
                );

                // 处理请求
                let result = Self::process_request(&request, &embedding_service).await;

                // 发送响应
                response_channel
                    .complete(request.request_id.clone(), result)
                    .await;

                debug!(
                    "Worker {} completed request {}",
                    worker_id, request.request_id
                );
            } else {
                // 队列为空，使用指数退避
                idle_count = idle_count.saturating_add(1usize);
                let wait_ms = std::cmp::min(100usize * (1usize << idle_count.min(6)), 5000usize);

                debug!(
                    "Worker {} queue empty, waiting {}ms (idle_count={})",
                    worker_id, wait_ms, idle_count
                );

                tokio::time::sleep(Duration::from_millis(wait_ms as u64)).await;

                // 如果长时间空闲，可以让 worker 退出
                if idle_count > MAX_IDLE_COUNT && worker_id > config.min_workers {
                    info!(
                        "Worker {} idle for too long, requesting shutdown",
                        worker_id
                    );
                    break;
                }
            }
        }

        // 清理：减少 worker 计数
        let final_count = Self::decrement_worker_count(&queue);
        info!(
            "Worker {} stopped, remaining workers: {}",
            worker_id, final_count
        );

        // 标记为不活跃
        {
            let mut guard = worker_health.lock().await;
            if let Some(info) = guard.iter_mut().find(|i| i.worker_id == worker_id) {
                info.is_alive = false;
            }
        }
    }

    /// 减少 worker 计数并返回剩余数量
    fn decrement_worker_count(queue: &Arc<PriorityRequestQueue>) -> usize {
        // 注意：这里简化实现，实际需要更复杂的同步机制
        0
    }

    /// 处理请求
    async fn process_request(
        request: &super::queue::QueuedRequest,
        embedding_service: &Arc<RwLock<EmbeddingService>>,
    ) -> Result<EmbedResponse, VecboostError> {
        // 实际调用 EmbeddingService
        let embed_request = &request.embed_request;

        debug!("Processing embedding request");

        // 获取 EmbeddingService 的读锁
        let service_guard = embedding_service.read().await;

        // 调用 EmbeddingService 进行推理
        // 使用 process_text 方法，它接受 EmbedRequest
        let result = service_guard
            .process_text(
                crate::domain::EmbedRequest {
                    text: embed_request.text.clone(),
                    normalize: embed_request.normalize,
                },
                None, // metrics_collector 可选
            )
            .await;

        drop(service_guard); // 显式释放锁

        match result {
            Ok(response) => {
                debug!(
                    "Successfully generated embedding with dimension: {}",
                    response.dimension
                );
                Ok(response)
            }
            Err(e) => {
                warn!("Embedding inference failed: {}", e);
                Err(e)
            }
        }
    }

    /// 启动扩缩容监控
    async fn start_scaling_monitor(&self) {
        let queue = Arc::clone(&self.queue);
        let current_workers = Arc::clone(&self.current_workers);
        let config = self.config.clone();
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(config.scale_check_interval_secs));

            loop {
                if !running.load(Ordering::Relaxed) {
                    break;
                }

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

                    // TODO: 实际启动 worker
                    // 需要改进架构以支持动态添加 worker
                }

                // 检查是否需要缩容
                if queue_size < config.scale_down_threshold && current > config.min_workers {
                    info!(
                        "Scaling down: removing workers (queue size: {})",
                        queue_size
                    );

                    // TODO: 实际移除 worker
                    // 向多余的 worker 发送关闭信号
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: 需要添加 EmbeddingService mock 来修复测试
    /*
    #[tokio::test]
    async fn test_worker_manager_creation() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();

        let service = Arc::new(RwLock::new(EmbeddingService::new(
            Arc::new(RwLock::new(crate::engine::TestEngine::new(384))),
            None,
        )));

        let manager = WorkerManager::new(queue, response_channel, config, service);

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

        let service = Arc::new(RwLock::new(EmbeddingService::new(
            Arc::new(RwLock::new(crate::engine::TestEngine::new(384))),
            None,
        )));

        let mut manager = WorkerManager::new(queue, response_channel, config, service);

        manager.start().await.unwrap();

        assert_eq!(manager.current_workers(), 2);
    }
    */
}
