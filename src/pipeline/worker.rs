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
    use crate::domain::EmbedRequest;
    use crate::engine::InferenceEngine;
    use crate::pipeline::priority::{Priority, RequestSource};
    use crate::pipeline::queue::QueuedRequest;
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

    /// 测试用错误引擎——始终返回 InferenceError,用于验证错误传播。
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

    /// 验证 WorkerManager::new 初始化所有字段为正确默认值。
    #[tokio::test]
    async fn test_worker_manager_new_initial_state() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(queue, response_channel, config, service);

        assert_eq!(manager.current_workers(), 0);
        assert!(manager.running.load(Ordering::SeqCst));
        assert!(manager.worker_senders.lock().await.is_empty());
        assert!(manager.worker_health.lock().await.is_empty());
    }

    /// 验证 start() 启动 min_workers 个 worker 并记录健康信息。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_manager_start_spawns_min_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 3,
            max_workers: 8,
            ..Default::default()
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.start().await.unwrap();

        assert_eq!(
            manager.current_workers(),
            3,
            "start() must spawn min_workers workers"
        );
        assert_eq!(
            manager.worker_senders.lock().await.len(),
            3,
            "must have 3 senders"
        );
        assert_eq!(
            manager.worker_health.lock().await.len(),
            3,
            "must have 3 health entries"
        );

        // 清理:停止所有 worker
        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 shutdown() 设置 running=false 并让所有 worker 退出。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_manager_shutdown_stops_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 2);

        // shutdown 内部 sleep 5s,用 timeout 包装避免无限等待
        tokio::time::timeout(Duration::from_secs(15), manager.shutdown())
            .await
            .expect("shutdown should complete within 15s");

        assert!(
            !manager.running.load(Ordering::SeqCst),
            "running flag must be false after shutdown"
        );

        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "workers did not exit after shutdown, remaining: {}",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 process_request 成功路径——调用 EmbeddingService 返回 EmbedResponse。
    #[tokio::test]
    async fn test_process_request_success() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-process-1".to_string(),
            embed_request: EmbedRequest {
                text: "hello world".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };

        let result = WorkerManager::process_request(&request, &service).await;
        assert!(result.is_ok(), "process_request should succeed");
        let response = result.unwrap();
        assert_eq!(response.dimension, 8);
        assert_eq!(response.embedding.len(), 8);
    }

    /// 验证 process_request 在引擎返回错误时传播 InferenceError。
    #[tokio::test]
    async fn test_process_request_engine_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(ErrorEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-process-err".to_string(),
            embed_request: EmbedRequest {
                text: "hello".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };

        let result = WorkerManager::process_request(&request, &service).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    /// 验证 worker_loop 从队列消费请求并通过 response_channel 发送响应。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_processes_queued_request() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );

        let rx = response_channel.register("test-loop-1".to_string()).await;

        let (tx, _) = tokio::sync::oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-loop-1".to_string(),
            embed_request: EmbedRequest {
                text: "hello world".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };
        queue.enqueue(request).await.unwrap();

        manager.spawn_worker().await;

        let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(result.is_ok(), "response should arrive within 5s");
        let response_result = result.unwrap().unwrap();
        assert!(response_result.is_ok());
        let response = response_result.unwrap();
        assert_eq!(response.dimension, 8);

        // 清理
        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 worker_loop 处理请求时引擎出错,response_channel 收到错误响应。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_propagates_engine_error() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(ErrorEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );

        let rx = response_channel.register("test-loop-err".to_string()).await;

        let (tx, _) = tokio::sync::oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-loop-err".to_string(),
            embed_request: EmbedRequest {
                text: "hello".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };
        queue.enqueue(request).await.unwrap();

        manager.spawn_worker().await;

        let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(result.is_ok(), "response should arrive within 5s");
        let response_result = result.unwrap().unwrap();
        assert!(response_result.is_err());
        match response_result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("mock inference failure"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }

        // 清理
        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 worker 退出后健康信息标记为 is_alive=false。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_exit_marks_health_dead() {
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
                .send(WorkerTask::Shutdown { immediate: false })
                .await
                .unwrap();
        }

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

        let health = manager.worker_health.lock().await;
        assert_eq!(health.len(), 1);
        assert!(
            !health[0].is_alive,
            "worker must be marked as dead after exit"
        );
    }

    /// 验证 WorkerHealthInfo::new 初始化字段。
    #[test]
    fn test_worker_health_info_new() {
        let health = WorkerHealthInfo::new(42);
        assert_eq!(health.worker_id, 42);
        assert_eq!(health.crash_count, 0);
        assert!(health.is_alive);
    }

    /// 验证 update_activity 更新 last_active_time。
    #[test]
    fn test_worker_health_info_update_activity() {
        let mut health = WorkerHealthInfo::new(0);
        let original = health.last_active_time;
        std::thread::sleep(Duration::from_millis(5));
        health.update_activity();
        assert!(health.last_active_time > original);
    }

    /// 验证 record_crash 递增 crash_count。
    #[test]
    fn test_worker_health_info_record_crash() {
        let mut health = WorkerHealthInfo::new(0);
        assert_eq!(health.crash_count, 0);
        health.record_crash();
        assert_eq!(health.crash_count, 1);
        health.record_crash();
        assert_eq!(health.crash_count, 2);
    }

    /// 验证 WorkerTask::Shutdown 变体 immediate 字段。
    #[test]
    fn test_worker_task_shutdown_variants() {
        let graceful = WorkerTask::Shutdown { immediate: false };
        let immediate = WorkerTask::Shutdown { immediate: true };

        match graceful {
            WorkerTask::Shutdown { immediate: false } => {}
            _ => panic!("graceful shutdown should have immediate=false"),
        }
        match immediate {
            WorkerTask::Shutdown { immediate: true } => {}
            _ => panic!("immediate shutdown should have immediate=true"),
        }
    }

    /// 验证 WorkerState 所有变体的相等性。
    #[test]
    fn test_worker_state_variants() {
        assert_eq!(WorkerState::Idle, WorkerState::Idle);
        assert_eq!(WorkerState::Processing, WorkerState::Processing);
        assert_eq!(WorkerState::Stopping, WorkerState::Stopping);
        assert_eq!(WorkerState::Stopped, WorkerState::Stopped);
        assert_ne!(WorkerState::Idle, WorkerState::Processing);
        assert_ne!(WorkerState::Stopping, WorkerState::Stopped);
    }

    /// 验证 worker_loop 在 running=false 时立即退出。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_exits_immediately_when_not_running() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(queue, response_channel, config, service);
        manager.running.store(false, Ordering::SeqCst);

        manager.spawn_worker().await;
        // Note: current_workers() may already be 0 here because worker_loop
        // checks running=false and exits immediately. The race between spawn
        // and exit makes asserting current_workers()==1 unreliable.

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "worker did not exit within 5s when running=false (current_workers={})",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let health = manager.worker_health.lock().await;
        assert_eq!(health.len(), 1);
        assert!(!health[0].is_alive, "worker must be marked dead after exit");
    }

    /// 验证 worker_loop 收到 ProcessRequest 任务时仅 debug 日志,不真正处理。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_ignores_process_request_via_task_receiver() {
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
                .send(WorkerTask::ProcessRequest {
                    request_id: "ignored-1".to_string(),
                    embed_request: EmbedRequest {
                        text: "hello".to_string(),
                        normalize: Some(true),
                    },
                })
                .await
                .expect("send ProcessRequest must succeed");
        }

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            manager.current_workers(),
            1,
            "worker must still be running after receiving ProcessRequest"
        );

        {
            let senders = manager.worker_senders.lock().await;
            senders[0]
                .send(WorkerTask::Shutdown { immediate: true })
                .await
                .unwrap();
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("worker did not exit after Shutdown");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 spawn_worker 后 worker_senders 与 worker_health 一致增长。
    #[tokio::test]
    async fn test_spawn_worker_appends_sender_and_health() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        manager.spawn_worker().await;
        manager.spawn_worker().await;

        assert_eq!(manager.current_workers(), 3);
        assert_eq!(manager.worker_senders.lock().await.len(), 3);
        assert_eq!(manager.worker_health.lock().await.len(), 3);

        let health = manager.worker_health.lock().await;
        let mut ids: Vec<usize> = health.iter().map(|h| h.worker_id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2]);

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 worker_loop 处理多个排队请求。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_processes_multiple_requests() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );

        let mut rxs = Vec::new();
        for i in 0..5 {
            let req_id = format!("multi-{}", i);
            rxs.push((i, response_channel.register(req_id).await));
        }

        for i in 0..5 {
            let (tx, _) = tokio::sync::oneshot::channel();
            let request = QueuedRequest {
                request_id: format!("multi-{}", i),
                embed_request: EmbedRequest {
                    text: format!("text-{}", i),
                    normalize: Some(true),
                },
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
                response_tx: tx,
            };
            queue.enqueue(request).await.unwrap();
        }

        manager.spawn_worker().await;

        for (i, rx) in rxs {
            let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
            assert!(result.is_ok(), "response {} should arrive within 5s", i);
            let response_result = result.unwrap().unwrap();
            assert!(response_result.is_ok());
            assert_eq!(response_result.unwrap().dimension, 8);
        }

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 worker_loop 在运行过程中持续更新 last_active_time。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_updates_activity_time() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;

        let initial_time = {
            let health = manager.worker_health.lock().await;
            health[0].last_active_time
        };

        // Poll for last_active_time refresh instead of fixed sleep.
        // Worker loop updates activity at the start of each iteration; under coverage
        // instrumentation the loop may be slow to schedule, so poll up to 10s.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut updated_time = initial_time;
        loop {
            {
                let health = manager.worker_health.lock().await;
                updated_time = health[0].last_active_time;
            }
            if updated_time > initial_time {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "last_active_time was not refreshed within 10s \
                     (initial={:?}, current={:?})",
                    initial_time, updated_time
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            updated_time > initial_time,
            "last_active_time must be refreshed during worker loop"
        );

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        // Poll for worker to stop (up to 5s) instead of fixed sleep.
        let stop_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let all_stopped = {
                let health = manager.worker_health.lock().await;
                health.iter().all(|h| !h.is_alive)
            };
            if all_stopped {
                break;
            }
            if tokio::time::Instant::now() >= stop_deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 process_request 处理 normalize=None 的请求也能成功。
    #[tokio::test]
    async fn test_process_request_with_none_normalize() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-none-norm".to_string(),
            embed_request: EmbedRequest {
                text: "hello".to_string(),
                normalize: None,
            },
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };

        let result = WorkerManager::process_request(&request, &service).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimension, 8);
    }

    /// 验证 shutdown() 在 worker_senders 为空时也能安全完成。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_shutdown_with_no_workers_is_safe() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        assert_eq!(manager.current_workers(), 0);

        tokio::time::timeout(Duration::from_secs(15), manager.shutdown())
            .await
            .expect("shutdown with no workers should complete");

        assert!(!manager.running.load(Ordering::SeqCst));
    }

    /// 验证 spawn_single_worker 在 worker_health 列表中追加新条目。
    #[tokio::test]
    async fn test_spawn_single_worker_records_health_with_correct_id() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.running.store(false, Ordering::SeqCst);

        WorkerManager::spawn_single_worker(
            &manager.current_workers,
            &manager.worker_senders,
            &manager.worker_health,
            &manager.queue,
            &manager.response_channel,
            &manager.embedding_service,
            &manager.config,
            &manager.running,
        )
        .await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("worker did not exit within 2s");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let health = manager.worker_health.lock().await;
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].worker_id, 0);
        assert!(!health[0].is_alive);
    }

    /// 验证 start_scaling_monitor 在队列压力超过阈值时扩容 worker。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_scaling_monitor_scales_up_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(500));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 0,
            max_workers: 4,
            scale_up_threshold: 10,
            scale_down_threshold: 5,
            idle_timeout_secs: 60,
            scale_check_interval_secs: 1,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );

        for i in 0..200 {
            let (tx, _) = tokio::sync::oneshot::channel();
            let request = QueuedRequest {
                request_id: format!("scale-up-{}", i),
                embed_request: EmbedRequest {
                    text: format!("text-{}", i),
                    normalize: Some(true),
                },
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
                response_tx: tx,
            };
            queue.enqueue(request).await.unwrap();
        }

        manager.start().await.unwrap();
        assert_eq!(manager.current_workers(), 0, "start() spawns min_workers=0");

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if manager.current_workers() > 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "scaling monitor did not scale up within 10s (current_workers={})",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(
            manager.current_workers() > 0,
            "workers should have been scaled up"
        );

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        let stop_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= stop_deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 start_scaling_monitor 在队列空闲时缩容 worker。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_scaling_monitor_scales_down_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 1,
            max_workers: 4,
            scale_up_threshold: 100,
            scale_down_threshold: 10,
            idle_timeout_secs: 60,
            scale_check_interval_secs: 1,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );

        manager.start().await.unwrap();
        assert_eq!(manager.current_workers(), 1, "start() spawns min_workers=1");

        manager.spawn_worker().await;
        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 3);

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if manager.current_workers() < 3 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "scaling monitor did not scale down within 10s (current_workers={})",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(
            manager.current_workers() < 3,
            "workers should have been scaled down"
        );

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        let stop_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= stop_deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 worker_loop 在队列空且 task channel 关闭时进入 idle backoff else 分支。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_enters_idle_backoff_when_channel_closed() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 0,
            max_workers: 2,
            ..Default::default()
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 1);

        // 关闭 task channel:清空 senders 使 task_receiver.recv() 返回 None,
        // 同时队列为空 → select! 进入 else 分支(idle backoff)
        {
            let mut senders = manager.worker_senders.lock().await;
            senders.clear();
        }

        // 等待 idle backoff 首次执行(首次 sleep = 100 * 2^1 = 200ms)
        tokio::time::sleep(Duration::from_millis(300)).await;

        // worker 仍在运行(idle_count 未超 MAX_IDLE_COUNT=10 或 worker_id 不大于 min_workers)
        assert_eq!(
            manager.current_workers(),
            1,
            "worker should still be running after idle backoff"
        );

        // 设置 running=false,worker 在下次循环顶部退出
        manager.running.store(false, Ordering::SeqCst);

        // 轮询等待退出(idle backoff sleep 可能达 5s,给 10s 余量)
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "worker did not exit within 10s after running=false (current_workers={})",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
