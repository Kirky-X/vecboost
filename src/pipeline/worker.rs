// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(dead_code)]

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

    /// 启动 Worker(对外入口,委托给 spawn_single_worker)
    pub async fn spawn_worker(&self) {
        Self::spawn_single_worker(
            &self.current_workers,
            &self.worker_senders,
            &self.worker_health,
            &self.queue,
            &self.response_channel,
            &self.embedding_service,
            &self.config,
            &self.running,
        )
        .await;
    }

    /// 启动单个 Worker(静态方法,供 spawn_worker 和 start_scaling_monitor 复用)
    ///
    /// 设计理由:`start_scaling_monitor` 内部 `tokio::spawn` 闭包无法持有 `&self`,
    /// 故按 `worker_loop` 既有惯例以静态方法 + Arc 引用形式暴露。
    #[allow(clippy::too_many_arguments)]
    async fn spawn_single_worker(
        current_workers: &Arc<AtomicUsize>,
        worker_senders: &Arc<Mutex<Vec<mpsc::Sender<WorkerTask>>>>,
        worker_health: &Arc<Mutex<Vec<WorkerHealthInfo>>>,
        queue: &Arc<PriorityRequestQueue>,
        response_channel: &Arc<ResponseChannel>,
        embedding_service: &Arc<RwLock<EmbeddingService>>,
        config: &WorkerConfig,
        running: &Arc<AtomicBool>,
    ) {
        let worker_id = current_workers.fetch_add(1, Ordering::SeqCst);

        // 创建任务通道
        let (task_sender, task_receiver) = mpsc::channel(100);

        // 保存发送器用于后续关闭
        {
            let mut senders = worker_senders.lock().await;
            senders.push(task_sender.clone());
        }

        // 初始化健康信息
        {
            let mut health_guard = worker_health.lock().await;
            health_guard.push(WorkerHealthInfo::new(worker_id));
        }

        let queue = Arc::clone(queue);
        let response_channel = Arc::clone(response_channel);
        let config = config.clone();
        let running = Arc::clone(running);
        let embedding_service = Arc::clone(embedding_service);
        let worker_health = Arc::clone(worker_health);
        let current_workers = Arc::clone(current_workers);

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
                current_workers,
            )
            .await;
        });
    }

    /// Worker 循环
    #[allow(clippy::too_many_arguments)]
    async fn worker_loop(
        worker_id: usize,
        mut task_receiver: mpsc::Receiver<WorkerTask>,
        queue: Arc<PriorityRequestQueue>,
        response_channel: Arc<ResponseChannel>,
        config: WorkerConfig,
        running: Arc<AtomicBool>,
        embedding_service: Arc<RwLock<EmbeddingService>>,
        worker_health: Arc<Mutex<Vec<WorkerHealthInfo>>>,
        current_workers: Arc<AtomicUsize>,
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

            // 同时等待队列请求和关闭信号(T009: 消费 task_receiver 的 Shutdown 信号,
            // 使 start_scaling_monitor 缩容发送的 Shutdown 能及时被 worker 接收)
            tokio::select! {
                Some(task) = task_receiver.recv() => {
                    match task {
                        WorkerTask::Shutdown { immediate } => {
                            if immediate {
                                info!(
                                    "Worker {} received immediate shutdown signal",
                                    worker_id
                                );
                            } else {
                                info!(
                                    "Worker {} received graceful shutdown signal",
                                    worker_id
                                );
                            }
                            break;
                        }
                        WorkerTask::ProcessRequest { .. } => {
                            // 当前请求通过 queue 路由,task_receiver 仅承载 Shutdown 信号。
                            // 保留分支以防未来切换到直接派发模式。
                            debug!(
                                "Worker {} received ProcessRequest via task_receiver \
                                 (ignored, queue is primary route)",
                                worker_id
                            );
                        }
                    }
                }
                // 从队列获取请求
                Some(request) = queue.dequeue() => {
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
                }
                // 队列为空且无关闭信号:指数退避
                else => {
                    idle_count = idle_count.saturating_add(1usize);
                    let wait_ms =
                        std::cmp::min(100usize * (1usize << idle_count.min(6)), 5000usize);

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
        }

        // 清理：减少 worker 计数
        let final_count = Self::decrement_worker_count(&current_workers);
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
    fn decrement_worker_count(current_workers: &Arc<AtomicUsize>) -> usize {
        current_workers.fetch_sub(1, Ordering::SeqCst) - 1
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
        let worker_senders = Arc::clone(&self.worker_senders);
        let worker_health = Arc::clone(&self.worker_health);
        let response_channel = Arc::clone(&self.response_channel);
        let embedding_service = Arc::clone(&self.embedding_service);

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

                // 扩容:队列压力超过阈值且未达上限
                if queue_size > config.scale_up_threshold && current < config.max_workers {
                    let new_workers = std::cmp::min(
                        (queue_size / config.scale_up_threshold).saturating_sub(1),
                        config.max_workers - current,
                    );

                    if new_workers > 0 {
                        info!(
                            "Scaling up: adding {} workers (queue size: {})",
                            new_workers, queue_size
                        );
                        for _ in 0..new_workers {
                            Self::spawn_single_worker(
                                &current_workers,
                                &worker_senders,
                                &worker_health,
                                &queue,
                                &response_channel,
                                &embedding_service,
                                &config,
                                &running,
                            )
                            .await;
                        }
                    }
                }

                // 缩容:队列压力低于阈值且超过最小值
                if queue_size < config.scale_down_threshold && current > config.min_workers {
                    // 先清理已退出 worker 的失效 sender(receiver 已 drop),
                    // 避免 take(to_remove) 取到失效 sender 导致缩容数量不足
                    let mut senders = worker_senders.lock().await;
                    let before = senders.len();
                    senders.retain(|s| !s.is_closed());
                    let cleaned = before - senders.len();
                    if cleaned > 0 {
                        debug!(
                            "Cleaned {} stale worker senders (before: {}, after: {})",
                            cleaned,
                            before,
                            senders.len()
                        );
                    }

                    let to_remove = current - config.min_workers;
                    if to_remove == 0 || senders.is_empty() {
                        continue;
                    }
                    info!(
                        "Scaling down: removing {} workers (queue size: {})",
                        to_remove, queue_size
                    );
                    // 从最新加入的 worker 开始发送关闭信号(保留 min_workers 个最早的)
                    for sender in senders.iter().rev().take(to_remove) {
                        let _ = sender.send(WorkerTask::Shutdown { immediate: false }).await;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;

    /// 测试用 Mock 推理引擎——返回固定 8 维零向量,不依赖任何外部模型。
    /// 定义在测试模块内,遵循 embedding.rs::tests 的 TestEngine 既有惯例。
    struct MockEngine;

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 8])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
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

    // TODO: 需要添加 EmbeddingService mock 来修复以下集成测试
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

    /// T004 H1: 验证 decrement_worker_count 真实递减(非硬编码 0)。
    #[test]
    fn test_decrement_worker_count_actually_decrements() {
        let counter = Arc::new(AtomicUsize::new(5));
        let remaining = WorkerManager::decrement_worker_count(&counter);
        assert_eq!(
            remaining, 4,
            "decrement_worker_count must return new count (old-1), not hardcoded 0"
        );
        assert_eq!(
            counter.load(Ordering::SeqCst),
            4,
            "counter must be decremented from 5 to 4"
        );
    }

    /// T004 H1: 边界场景——单 worker 停止后计数归零。
    #[test]
    fn test_decrement_worker_count_from_one_to_zero() {
        let counter = Arc::new(AtomicUsize::new(1));
        let remaining = WorkerManager::decrement_worker_count(&counter);
        assert_eq!(remaining, 0, "single worker stop should bring count to 0");
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// T004 H1: 连续递减多次,确认每次都生效(排除"只减一次"的假实现)。
    #[test]
    fn test_decrement_worker_count_multiple_times() {
        let counter = Arc::new(AtomicUsize::new(3));
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 2);
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 1);
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// T005 H2: 验证 spawn_worker(对外入口)真实递增 current_workers。
    /// 此前 start_scaling_monitor 中的扩容逻辑是 TODO 注释,无法实际增加 worker;
    /// 现已通过 spawn_single_worker 落地。
    #[tokio::test]
    async fn test_spawn_worker_increments_current_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        assert_eq!(manager.current_workers(), 0, "initial count must be 0");
        manager.spawn_worker().await;
        assert_eq!(
            manager.current_workers(),
            1,
            "spawn_worker must increment counter via fetch_add (H2 fix)"
        );
        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 2);

        // 清理:停止 spawned workers,避免任务泄漏
        manager.running.store(false, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// T005 H2: 验证 spawn_single_worker(静态方法,被 start_scaling_monitor 调用)
    /// 真实递增计数器——这是扩容逻辑落地的核心证据。
    #[tokio::test]
    async fn test_spawn_single_worker_increments_counter() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        let running = Arc::clone(&manager.running);
        let current_workers = Arc::clone(&manager.current_workers);
        let worker_senders = Arc::clone(&manager.worker_senders);
        let worker_health = Arc::clone(&manager.worker_health);
        let queue_clone = Arc::clone(&manager.queue);
        let response_channel_clone = Arc::clone(&manager.response_channel);
        let embedding_service_clone = Arc::clone(&manager.embedding_service);
        let config_clone = manager.config.clone();

        assert_eq!(current_workers.load(Ordering::SeqCst), 0);
        WorkerManager::spawn_single_worker(
            &current_workers,
            &worker_senders,
            &worker_health,
            &queue_clone,
            &response_channel_clone,
            &embedding_service_clone,
            &config_clone,
            &running,
        )
        .await;
        assert_eq!(
            current_workers.load(Ordering::SeqCst),
            1,
            "spawn_single_worker (used by scaling monitor) must increment counter"
        );

        // 清理
        running.store(false, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// T009 H2-补: 验证 worker_loop 消费 task_receiver 的 Shutdown 信号,
    /// 在 2s 内退出(而非等 idle timeout ~30s 或永不退出)。
    ///
    /// 此前 worker_loop 的 _task_receiver 参数带下划线(未使用),
    /// start_scaling_monitor 缩容发送的 Shutdown 信号被忽略;
    /// 且 worker_id=0 < min_workers=2,idle timeout 也不触发,
    /// worker 实际上永远不会因 Shutdown 退出。
    #[tokio::test]
    async fn test_worker_loop_consumes_graceful_shutdown_within_2s() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 1, "worker must be spawned");

        // 发送优雅关闭信号(模拟 start_scaling_monitor 缩容)
        {
            let senders = manager.worker_senders.lock().await;
            assert!(
                !senders.is_empty(),
                "worker_senders must contain spawned worker's sender"
            );
            senders[0]
                .send(WorkerTask::Shutdown { immediate: false })
                .await
                .expect("send graceful Shutdown must succeed");
        }

        // 验证 worker 在 2s 内退出(current_workers 归零)
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "worker did not shut down within 2s after graceful Shutdown signal \
                     (current_workers={}) — worker_loop is not consuming task_receiver (T009 regression)",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// T009 H2-补: 验证 immediate=true 时 worker 也退出(立即关闭路径)。
    #[tokio::test]
    async fn test_worker_loop_consumes_immediate_shutdown_within_2s() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 1);

        {
            let senders = manager.worker_senders.lock().await;
            senders[0]
                .send(WorkerTask::Shutdown { immediate: true })
                .await
                .expect("send immediate Shutdown must succeed");
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "worker did not shut down within 2s after immediate Shutdown \
                     (current_workers={}) — worker_loop is not consuming task_receiver",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// T009-补: 验证 worker 退出后 sender 变为 closed,
    /// start_scaling_monitor 的 retain(!s.is_closed()) 能正确清理失效 sender。
    ///
    /// T009 修复后 worker 会因 Shutdown 退出,但 sender 留在 worker_senders 中。
    /// 若不清理,下次缩容 take(to_remove) 可能取到失效 sender,导致缩容数量不足。
    #[tokio::test]
    async fn test_worker_exit_marks_sender_closed_for_cleanup() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 1);

        // 发送 Shutdown 让 worker 退出
        {
            let senders = manager.worker_senders.lock().await;
            senders[0]
                .send(WorkerTask::Shutdown { immediate: false })
                .await
                .expect("send Shutdown must succeed");
        }

        // 等待 worker 退出
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("worker did not exit within 2s");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // 验证 sender 已 closed(worker 退出后 receiver drop,is_closed 返回 true)
        {
            let senders = manager.worker_senders.lock().await;
            assert!(
                senders[0].is_closed(),
                "sender must be closed after worker exit (required for retain cleanup in scaling_monitor)"
            );
        }
    }
}
