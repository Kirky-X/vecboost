// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

#![allow(
    dead_code,
    reason = "WorkerManager is used via queue in handler; tests cover all methods, production uses shared queue"
)]

use log::{debug, error, info, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};

use super::config::WorkerConfig;
use super::queue::{PriorityRequestQueue, QueuedRequest, ServiceRequest};
use super::response_channel::ResponseChannel;
use crate::domain::{EmbedResponse, ServiceResponse};
use crate::error::VecboostError;
use crate::service::decision::DecisionService;
use crate::service::embedding::EmbeddingService;

/// Worker 任务枚举 — 通过 mpsc channel 发送给 worker loop。
///
/// 包含两种变体：处理请求和优雅关闭。
#[derive(Debug)]
pub enum WorkerTask {
    /// 优雅关闭信号
    Shutdown {
        /// 是否立即关闭（不等待当前请求完成）
        immediate: bool,
    },
}

/// Worker 生命周期状态。
///
/// 状态转换：Idle → Processing → Idle（循环）；
/// 收到 Shutdown 后转为 Stopping → Stopped。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Idle,
    Processing,
    Stopping,
    Stopped,
}

/// Worker 实例 — 从任务队列接收请求并调用 EmbeddingService 处理。
///
/// 每个 Worker 在独立 tokio task 中运行，通过 mpsc channel 接收任务，
/// 通过 ResponseChannel 返回结果。
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

/// 时间窗批组装纯函数。
///
/// 语义：首请求到达后开启 `batch_wait_ms` 窗口，窗口内继续出队，
/// 凑满 `max_batch_size` 或窗口关闭即返回；`batch_wait_ms=0` 时立即
/// 返回仅首请求（严格还原排空式，kill-switch 内建）。
///
/// `try_dequeue` 为非阻塞出队闭包（返回 `None` 表示当前无请求），
/// 函数在窗口内以 1ms 粒度轮询以接住陆续到达的请求，不引入新线程/任务。
pub async fn assemble_batch<F, Fut>(
    first: QueuedRequest,
    mut try_dequeue: F,
    max_batch_size: usize,
    batch_wait_ms: u64,
) -> Vec<QueuedRequest>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Option<QueuedRequest>>,
{
    let mut batch = vec![first];
    let cap = max_batch_size.max(1);
    if batch.len() >= cap {
        return batch;
    }
    if batch_wait_ms == 0 {
        return batch;
    }
    let deadline = tokio::time::Instant::now() + Duration::from_millis(batch_wait_ms);
    loop {
        if batch.len() >= cap {
            break;
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            break;
        }
        match try_dequeue().await {
            Some(req) => {
                batch.push(req);
            }
            None => {
                let remaining = deadline.saturating_duration_since(now);
                let sleep_for = std::cmp::min(remaining, Duration::from_millis(1));
                if sleep_for.is_zero() {
                    break;
                }
                tokio::time::sleep(sleep_for).await;
            }
        }
    }
    batch
}

/// Worker 管理器 — 管理 worker 生命周期和自动伸缩。
///
/// 负责：
/// - 启动/停止 worker（`spawn_worker` / `shutdown`）
/// - 根据队列负载自动扩缩容（`start_scaling_monitor`）
/// - 跟踪 worker 健康状态和崩溃计数
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
    /// 决策服务实例——决策请求出队后由它执行（未注入时决策请求显性报错，
    /// 不静默路由 embed 服务）
    decision_service: Option<Arc<RwLock<DecisionService>>>,
    /// Worker 健康状态跟踪
    worker_health: Arc<Mutex<Vec<WorkerHealthInfo>>>,
    /// 后台任务集合（worker loops + scaling monitor）
    bg_tasks: Arc<Mutex<tokio::task::JoinSet<()>>>,
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
        let max_workers = config.max_workers;
        Self {
            min_workers: config.min_workers,
            max_workers: config.max_workers,
            current_workers: Arc::new(AtomicUsize::new(0)),
            queue,
            response_channel,
            config,
            running: Arc::new(AtomicBool::new(true)),
            worker_senders: Arc::new(Mutex::new(Vec::with_capacity(max_workers))),
            embedding_service,
            decision_service: None,
            worker_health: Arc::new(Mutex::new(Vec::with_capacity(max_workers))),
            bg_tasks: Arc::new(Mutex::new(tokio::task::JoinSet::new())),
        }
    }

    /// 注入决策服务（server 装配；决策请求出队后由它执行）
    pub fn with_decision_service(mut self, decision_service: Arc<RwLock<DecisionService>>) -> Self {
        self.decision_service = Some(decision_service);
        self
    }

    /// 启动 Worker Manager
    pub async fn start(&self) -> Result<(), VecboostError> {
        info!(
            "Starting WorkerManager with min={} max={}",
            self.min_workers, self.max_workers
        );

        for _ in 0..self.min_workers {
            self.spawn_worker().await;
        }

        self.start_scaling_monitor().await;

        info!("WorkerManager started successfully");

        Ok(())
    }

    /// 优雅关闭所有 Worker
    ///
    /// 排空语义（T032）：进入 shutdown 即清空队列并对每个排队请求完成
    /// 503（不再悬挂至客户端超时）；在途批获得 `drain_timeout_secs`（默认 5s）
    /// 的完成窗口，到期后 abort。`Shutdown { immediate: true }` 跳过等待。
    pub async fn shutdown(&self) {
        info!("Shutting down WorkerManager...");

        self.running.store(false, Ordering::SeqCst);

        // 排空队列：排队请求快速失败（503），不再无限悬挂
        let drained = self.queue.dequeue_all_for_shutdown().await;
        for req in drained {
            self.response_channel
                .complete(
                    req.request_id.clone(),
                    Err(VecboostError::InferenceError(
                        "server is shutting down".to_string(),
                    )),
                )
                .await;
        }

        let senders = {
            let guard = self.worker_senders.lock().await;
            guard.clone()
        };

        for sender in &senders {
            let _ = sender.send(WorkerTask::Shutdown { immediate: false }).await;
        }

        // 可配排空窗口（此前硬编码 5s）
        let drain = Duration::from_secs(self.config.drain_timeout_secs);
        tokio::time::sleep(drain).await;

        // 强制关闭剩余的 worker
        for sender in &senders {
            let _ = sender.send(WorkerTask::Shutdown { immediate: true }).await;
        }

        // Abort all background tasks (worker loops + scaling monitor)
        {
            let mut tasks = self.bg_tasks.lock().await;
            tasks.abort_all();
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
            &self.decision_service,
            &self.config,
            &self.running,
            &self.bg_tasks,
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
        decision_service: &Option<Arc<RwLock<DecisionService>>>,
        config: &WorkerConfig,
        running: &Arc<AtomicBool>,
        bg_tasks: &Arc<Mutex<tokio::task::JoinSet<()>>>,
    ) {
        let worker_id = current_workers.fetch_add(1, Ordering::SeqCst);

        let (task_sender, task_receiver) = mpsc::channel(100);

        // 保存发送器用于后续关闭
        {
            let mut senders = worker_senders.lock().await;
            senders.push(task_sender.clone());
        }

        {
            let mut health_guard = worker_health.lock().await;
            health_guard.push(WorkerHealthInfo::new(worker_id));
        }

        let queue = Arc::clone(queue);
        let response_channel = Arc::clone(response_channel);
        let config = config.clone();
        let running = Arc::clone(running);
        let embedding_service = Arc::clone(embedding_service);
        let decision_service = decision_service.clone();
        let worker_health = Arc::clone(worker_health);
        let current_workers = Arc::clone(current_workers);

        info!("Worker {} started", worker_id);

        let handle = tokio::spawn(async move {
            Self::worker_loop(
                worker_id,
                task_receiver,
                queue,
                response_channel,
                config,
                running,
                embedding_service,
                decision_service,
                worker_health,
                current_workers,
            )
            .await;
        });

        // Track the worker task for lifecycle management
        bg_tasks.lock().await.spawn(async move {
            let _ = handle.await;
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
        decision_service: Option<Arc<RwLock<DecisionService>>>,
        worker_health: Arc<Mutex<Vec<WorkerHealthInfo>>>,
        current_workers: Arc<AtomicUsize>,
    ) {
        debug!("Worker {} loop started", worker_id);

        let mut idle_count: usize = 0;
        let mut last_reap = std::time::Instant::now();
        const MAX_IDLE_COUNT: usize = 10; // 最大空闲计数

        loop {
            if !running.load(Ordering::Relaxed) {
                info!("Worker {} received stop signal", worker_id);
                break;
            }

            {
                let mut guard = worker_health.lock().await;
                if let Some(info) = guard.iter_mut().find(|i| i.worker_id == worker_id) {
                    info.update_activity();
                }
            }

            // 同时等待队列请求和关闭信号
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
                    }
                }
                Some(request) = queue.dequeue() => {
                    idle_count = 0;

                    // SLA 收割——排队超过自身 timeout 的请求完成超时响应，不入推理。
                    // 与老化跳级正交：收割保证任何请求的排队滞留有上界。
                    // 时间门控 250ms：收割目标 ≥30s SLA，全表扫描频率无需随批循环
                    // （每 5ms 一次在深队列下持写锁重排，审查 PERF-6）
                    if last_reap.elapsed() >= Duration::from_millis(250) {
                        last_reap = std::time::Instant::now();
                        for expired_req in queue.dequeue_expired().await {
                            queue.cancellations().release(&expired_req.request_id);
                            response_channel
                                .complete(
                                    expired_req.request_id.clone(),
                                    Err(VecboostError::RequestTimeout(format!(
                                        "request waited {:.1}s in queue, exceeding its {}s SLA",
                                        expired_req.submitted_at.elapsed().as_secs_f64(),
                                        expired_req.timeout.as_secs_f64()
                                    ))),
                                )
                                .await;
                        }
                        // 过期响应条目清扫：取消丢弃路径（断连/服务端超时置位
                        // 取消后出队即弃）的通道条目永远不会被 complete——worker
                        // 看不到该请求，deregister 仅覆盖入队失败分支。条目按
                        // 自身 timeout 到期在此移除，闭合无界增长面（三路评审
                        // M×3）；此时 handler 已返回、rx 已 drop，清扫零副作用
                        response_channel.cleanup_expired().await;
                    }

                    // 时间窗动态拼批——首请求后开 batch_wait_ms 窗口继续聚合，
                    // batch_wait_ms=0 时 assemble_batch 立即返回仅首请求（旧排空语义）。
                    let wait_start = tokio::time::Instant::now();
                    let batch = assemble_batch(
                        request,
                        || queue.dequeue(),
                        config.max_batch_size,
                        config.batch_wait_ms,
                    )
                    .await;
                    let waited_secs = wait_start.elapsed().as_secs_f64();
                    // 埋点：批次大小与窗口等待时长，按批构成选标签——决策接入
                    // 共享拼批后混批是常态，恒挂 embed 标签会把决策负载误计入
                    // embed 容量画像（三路评审 M）。HistogramVec operation 为
                    // 自由标签，无需新注册
                    #[cfg(feature = "http")]
                    if let Some(collector) = crate::metrics::prometheus_exporter::global_collector()
                    {
                        let has_embed = batch
                            .iter()
                            .any(|r| matches!(r.request, ServiceRequest::Embed(_)));
                        let has_decision = batch
                            .iter()
                            .any(|r| matches!(r.request, ServiceRequest::Decision(_)));
                        let operation = match (has_embed, has_decision) {
                            (true, false) => "embed",
                            (false, true) => "decision",
                            _ => "mixed",
                        };
                        collector.observe_batch(operation, batch.len(), waited_secs);
                    }
                    debug!(
                        "Worker {} assembled batch of {} (wait {:.3}s, window {}ms)",
                        worker_id,
                        batch.len(),
                        waited_secs,
                        config.batch_wait_ms
                    );

                    // SLA 淘汰——拼批窗口内可能有请求到达自身 timeout，出批时再查一次
                    let now = std::time::Instant::now();
                    let mut valid_batch = Vec::with_capacity(batch.len());
                    for req in batch {
                        // 排队延迟可观测（T033）：入队→出队等待时长；
                        // 决策请求走专属指标（P2-3），embed/rerank 维持既有口径
                        #[cfg(feature = "http")]
                        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector()
                        {
                            let wait_secs = now.duration_since(req.submitted_at).as_secs_f64();
                            match &req.request {
                                ServiceRequest::Decision(_) => {
                                    collector.observe_decision_queue_wait(wait_secs);
                                }
                                _ => collector.observe_queue_wait("embed", wait_secs),
                            }
                        }
                        if now.duration_since(req.submitted_at) >= req.timeout {
                            warn!(
                                "Request {} expired in queue ({:.1}s), rejecting",
                                req.request_id,
                                now.duration_since(req.submitted_at).as_secs_f64()
                            );
                            queue.cancellations().release(&req.request_id);
                            response_channel
                                .complete(
                                    req.request_id.clone(),
                                    Err(VecboostError::RequestTimeout(format!(
                                        "request waited {:.1}s in queue, exceeding its {}s SLA",
                                        now.duration_since(req.submitted_at).as_secs_f64(),
                                        req.timeout.as_secs_f64()
                                    ))),
                                )
                                .await;
                        } else {
                            valid_batch.push(req);
                        }
                    }

                    if valid_batch.is_empty() {
                        continue;
                    }

                    debug!(
                        "Worker {} processing batch of {} requests",
                        worker_id, valid_batch.len()
                    );

                    // panic 隔离（T029）：推理 panic 不再带走 worker——
                    // 经 JoinHandle 捕获，批内未完成请求统一补 InternalError。
                    // 整批 mem::take move 进闭包（决策请求含递归 Value 深拷贝
                    // 成本高且由用户输入定大小，热路径不做整批 clone，三路
                    // 评审 M）；批后清理只按 request_id 释放取消注册表
                    let batch_for_panic: Vec<String> = valid_batch
                        .iter()
                        .map(|r| r.request_id.clone())
                        .collect();
                    let svc = Arc::clone(&embedding_service);
                    let ds = decision_service.clone();
                    let chan = Arc::clone(&response_channel);
                    let batch_owned = std::mem::take(&mut valid_batch);
                    let handle = tokio::spawn(async move {
                        Self::process_batch_requests(&batch_owned, &svc, ds.as_ref(), &chan).await
                    });
                    match handle.await {
                        Ok(()) => {}
                        Err(join_err) => {
                            error!(
                                "Worker {} inference panicked: {}",
                                worker_id,
                                join_err
                            );
                            for rid in &batch_for_panic {
                                response_channel
                                    .complete(
                                        rid.clone(),
                                        Err(VecboostError::InternalError(
                                            "inference task panicked".to_string(),
                                        )),
                                    )
                                    .await;
                            }
                        }
                    }
                    // 释放取消注册表条目（响应已送达或已补错）——
                    // batch_for_panic 与整批的 request_id 集合一致
                    for rid in &batch_for_panic {
                        queue.cancellations().release(rid);
                    }
                }
                // 队列为空时等待入队通知，消除指数退避轮询
                // 使用 timeout 实现空闲超时退出
                result = tokio::time::timeout(
                    Duration::from_secs(config.idle_timeout_secs),
                    queue.notify().notified(),
                ) => {
                    match result {
                        Ok(()) => {
                            debug!("Worker {} notified of new request", worker_id);
                            idle_count = 0;
                        }
                        Err(_) => {
                            // 空闲超时:顺带收割过期请求（兜底全部 worker 挂起时
                            // 队列残留过期项的场景），再增加空闲计数
                            for expired_req in queue.dequeue_expired().await {
                                queue.cancellations().release(&expired_req.request_id);
                                response_channel
                                    .complete(
                                        expired_req.request_id.clone(),
                                        Err(VecboostError::RequestTimeout(format!(
                                            "request waited {:.1}s in queue, exceeding its {}s SLA",
                                            expired_req.submitted_at.elapsed().as_secs_f64(),
                                            expired_req.timeout.as_secs_f64()
                                        ))),
                                    )
                                    .await;
                            }
                            // 空闲路径同样清扫过期响应条目（取消丢弃泄漏类的
                            // 兜底通道，与出队分支收割点同语义）
                            response_channel.cleanup_expired().await;
                            idle_count = idle_count.saturating_add(1);
                            debug!(
                                "Worker {} idle timeout ({}s), idle_count={}",
                                worker_id, config.idle_timeout_secs, idle_count
                            );
                        }
                    }
                }
            }

            // 如果长时间空闲且队列为空，让 worker 退出
            if idle_count > MAX_IDLE_COUNT && queue.size() == 0 && worker_id > config.min_workers {
                info!(
                    "Worker {} idle for too long, requesting shutdown",
                    worker_id
                );
                break;
            }
        }

        let final_count = Self::decrement_worker_count(&current_workers);
        info!(
            "Worker {} stopped, remaining workers: {}",
            worker_id, final_count
        );

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
        decision_service: Option<&Arc<RwLock<DecisionService>>>,
    ) -> Result<ServiceResponse, VecboostError> {
        match &request.request {
            ServiceRequest::Embed(embed_request) => {
                debug!("Processing embedding request");

                let service_guard = embedding_service.read().await;

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
                        Ok(ServiceResponse::Embed(response))
                    }
                    Err(e) => {
                        warn!("Embedding inference failed: {}", e);
                        Err(e)
                    }
                }
            }
            ServiceRequest::Rerank(_) => Err(VecboostError::InternalError(
                "Rerank not supported by embedding worker".to_string(),
            )),
            ServiceRequest::Decision(decision_req) => {
                // 决策执行保留 decide 契约：DecisionService::process_decision
                // 内部以 spawn_blocking 包裹阻塞推理（0c2b164 口径不回退）
                let service = decision_service.ok_or_else(|| {
                    VecboostError::InternalError(
                        "Decision service not configured for worker".to_string(),
                    )
                })?;
                let service_guard = service.read().await;
                let result = service_guard
                    .process_decision(decision_req.clone(), crate::domain::decision::MAX_QUESTIONS)
                    .await;
                drop(service_guard);
                result.map(ServiceResponse::Decision)
            }
        }
    }

    /// 批量处理请求——用 embed_batch 合并推理，按 request_id 切分结果分别 complete。
    ///
    /// 单条文本失败仅该请求收错，不影响其他请求。
    /// 决策请求不并入 embed 拼批（问题级 collate 是 DecisionPipeline 内部能力）：
    /// 先逐个 `tokio::spawn` 并发执行（decide 内部 spawn_blocking，多决策可借
    /// 引擎读写锁并发），embed 推理与批内决策并行推进——决策不 inline await，
    /// 否则队头阻塞同批 embed 推理、拖慢 worker 出队节奏（三路评审 M）。
    async fn process_batch_requests(
        batch: &[super::queue::QueuedRequest],
        embedding_service: &Arc<RwLock<EmbeddingService>>,
        decision_service: Option<&Arc<RwLock<DecisionService>>>,
        response_channel: &Arc<ResponseChannel>,
    ) {
        if batch.is_empty() {
            return;
        }

        // 单条请求走快速路径
        if batch.len() == 1 {
            let result =
                Self::process_request(&batch[0], embedding_service, decision_service).await;
            response_channel
                .complete(batch[0].request_id.clone(), result)
                .await;
            return;
        }

        let mut texts = Vec::with_capacity(batch.len());
        let mut normalize_flags = Vec::with_capacity(batch.len());
        let mut valid_indices = Vec::with_capacity(batch.len());
        // 批内决策的并发执行句柄：(request_id, JoinHandle)
        let mut pending_decisions: Vec<(
            String,
            tokio::task::JoinHandle<Result<ServiceResponse, VecboostError>>,
        )> = Vec::new();

        for (i, req) in batch.iter().enumerate() {
            match &req.request {
                ServiceRequest::Embed(embed_req) => {
                    texts.push(embed_req.text.clone());
                    normalize_flags.push(embed_req.normalize.unwrap_or(false));
                    valid_indices.push(i);
                }
                ServiceRequest::Rerank(_) => {
                    // Rerank 不支持，直接给该请求返回错误
                    response_channel
                        .complete(
                            req.request_id.clone(),
                            Err(VecboostError::InternalError(
                                "Rerank not supported by embedding worker".to_string(),
                            )),
                        )
                        .await;
                }
                ServiceRequest::Decision(decision_req) => {
                    // spawn 并发执行（decide 内部 spawn_blocking 不占 worker 面），
                    // 结果在 embed 推理发起后统一收齐
                    match decision_service {
                        Some(service) => {
                            let service = Arc::clone(service);
                            let decision_req = decision_req.clone();
                            let request_id = req.request_id.clone();
                            let handle = tokio::spawn(async move {
                                let guard = service.read().await;
                                guard
                                    .process_decision(
                                        decision_req,
                                        crate::domain::decision::MAX_QUESTIONS,
                                    )
                                    .await
                                    .map(ServiceResponse::Decision)
                            });
                            pending_decisions.push((request_id, handle));
                        }
                        None => {
                            response_channel
                                .complete(
                                    req.request_id.clone(),
                                    Err(VecboostError::InternalError(
                                        "Decision service not configured for worker".to_string(),
                                    )),
                                )
                                .await;
                        }
                    }
                }
            }
        }

        // texts 为空（纯决策批）只跳过 embed 推理段——决策收齐循环必须
        // 无条件执行：早退会把已 spawn 的 JoinHandle 随局部变量 drop（脱离
        // 监控），panic 无 JoinError 映射、等待方悬挂到 SLA 超时，且 worker
        // 不等决策完成即出队下一批、任务堆积不受配额约束（三路评审 H×3）
        if !texts.is_empty() {
            let service_guard = embedding_service.read().await;
            let batch_started = std::time::Instant::now();
            let batch_result = service_guard.embed_batch_texts(&texts).await;
            drop(service_guard);

            match batch_result {
                Ok(embeddings) => {
                    // 批内各请求共享本次批量推理耗时（拼批语义下的真实处理时长）
                    let batch_millis = batch_started.elapsed().as_millis();
                    // 按 request_id 切分结果（j 为收集序——texts/normalize_flags 按
                    // 收集序 push，idx 是批内原始位置，混批时两者不同；既有实现
                    // 以下标互查在 embed 不位于 0 位时越界，三路评审混批测试实抓）
                    for (j, &idx) in valid_indices.iter().enumerate() {
                        let req = &batch[idx];
                        if j < embeddings.len() {
                            let mut embedding = embeddings[j].clone();
                            if normalize_flags[j] {
                                crate::utils::vector::normalize_l2(&mut embedding).ok();
                            }
                            let dimension = embedding.len();
                            response_channel
                                .complete(
                                    req.request_id.clone(),
                                    Ok(ServiceResponse::Embed(EmbedResponse {
                                        dimension,
                                        embedding,
                                        processing_time_ms: batch_millis,
                                        information_retention_rate: None,
                                    })),
                                )
                                .await;
                        } else {
                            // 引擎返回的向量数少于输入
                            response_channel
                                .complete(
                                    req.request_id.clone(),
                                    Err(VecboostError::InternalError(
                                        "Batch inference returned fewer embeddings than inputs"
                                            .to_string(),
                                    )),
                                )
                                .await;
                        }
                    }
                }
                Err(e) => {
                    // 批量推理失败——所有请求收错
                    warn!("Batch inference failed: {}", e);
                    for req in batch.iter() {
                        if matches!(req.request, ServiceRequest::Embed(_)) {
                            response_channel
                                .complete(req.request_id.clone(), Err(e.clone()))
                                .await;
                        }
                    }
                }
            }
        }

        // 收齐批内决策结果（spawn 时已与 embed 推理并行推进；纯决策批也走到
        // 这里）。决策任务 panic 经 JoinError 显性映射 InferenceError，不吞错
        // 不悬挂等待方；收齐返回前完成，保证决策执行速率与出队节奏耦合（节流）
        for (request_id, handle) in pending_decisions {
            let result = match handle.await {
                Ok(result) => result,
                Err(join_err) => {
                    log::error!("decision inference task failed: {join_err}");
                    Err(VecboostError::InferenceError(
                        "decision inference task failed (engine task panicked); see server logs"
                            .to_string(),
                    ))
                }
            };
            response_channel.complete(request_id, result).await;
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
        let decision_service = self.decision_service.clone();

        let bg_tasks = Arc::clone(&self.bg_tasks);
        let bg_tasks_for_spawn = Arc::clone(&bg_tasks);

        bg_tasks_for_spawn.lock().await.spawn(async move {
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
                                &decision_service,
                                &config,
                                &running,
                                &bg_tasks,
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
    use crate::config::model::{ModelConfig, ModelTask, Precision};
    use crate::domain::{
        DecisionAnswer, DecisionAnswerBody, DecisionQuestion, DecisionRequest, DecisionResponse,
        EmbedRequest, QuestionType, ServiceResponse,
    };
    use crate::engine::InferenceEngine;
    use crate::pipeline::priority::{Priority, RequestSource};
    use crate::pipeline::queue::QueuedRequest;
    use crate::service::decision::DecisionService;
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::Arc as StdArc;
    use tokio::sync::Mutex as TokioMutex;

    fn make_queued(id: &str) -> QueuedRequest {
        QueuedRequest {
            request_id: id.to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: id.to_string(),
                normalize: Some(false),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        }
    }

    #[tokio::test]
    async fn test_assemble_batch_collects_arrivals_within_window() {
        let pending: StdArc<TokioMutex<VecDeque<QueuedRequest>>> =
            StdArc::new(TokioMutex::new(VecDeque::new()));
        let pending_clone = StdArc::clone(&pending);
        // 窗口内陆续到达 2 个后续请求
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            pending_clone.lock().await.push_back(make_queued("req-2"));
            tokio::time::sleep(Duration::from_millis(5)).await;
            pending_clone.lock().await.push_back(make_queued("req-3"));
        });
        let first = make_queued("req-1");
        let batch = assemble_batch(
            first,
            || {
                let pending = StdArc::clone(&pending);
                async move { pending.lock().await.pop_front() }
            },
            8,
            50,
        )
        .await;
        assert_eq!(batch.len(), 3, "窗口内陆续到达 3 请求应单次组装返回 3 条");
        assert_eq!(batch[0].request_id, "req-1");
    }

    #[tokio::test]
    async fn test_assemble_batch_zero_wait_returns_first_only() {
        let first = make_queued("only");
        let batch = assemble_batch(first, || async { Some(make_queued("late")) }, 8, 0).await;
        assert_eq!(batch.len(), 1, "batch_wait_ms=0 应立即返回仅首请求");
        assert_eq!(batch[0].request_id, "only");
    }

    #[tokio::test]
    async fn test_assemble_batch_full_returns_early() {
        let first = make_queued("a");
        let batch = assemble_batch(first, || async { Some(make_queued("extra")) }, 2, 100).await;
        assert_eq!(batch.len(), 2, "凑满 max_batch_size 应提前返回");
    }

    #[tokio::test]
    async fn test_process_batch_same_text_byte_equal_and_ordered() {
        use crate::service::embedding::EmbeddingService;
        // 不变量：同一文本在批内经 scatter 回填的结果，与该文本单独走
        // embed_batch_texts 的结果字节等同，且顺序与输入一一对应。
        // （注：worker 批路径 embed_batch_texts 与单请求 process_text 全管线
        // 在归一化/分块上本就存在既有差异，此处只断言 scatter 自身的正确性。）
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let channel = Arc::new(ResponseChannel::new());
        let mk = |id: &str, text: &str| QueuedRequest {
            request_id: id.to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: text.to_string(),
                normalize: Some(false),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        // 同一函数单文本基线
        let baseline = service
            .read()
            .await
            .embed_batch_texts(&["hello world".to_string()])
            .await
            .expect("baseline embed must succeed");
        // 批内相同文本两条 + 一条不同文本，验证 scatter 保序与字节等同
        let b1 = mk("b1", "hello world");
        let b2 = mk("b2", "hello world");
        let b3 = mk("b3", "other text");
        let batch = vec![b1, b2, b3];
        let rx1 = channel.register("b1".to_string()).await;
        let rx2 = channel.register("b2".to_string()).await;
        let rx3 = channel.register("b3".to_string()).await;
        WorkerManager::process_batch_requests(&batch, &service, None, &channel).await;
        let unwrap_embed = |r: ServiceResponse| match r {
            ServiceResponse::Embed(resp) => resp,
            other => panic!("expected Embed response, got {other:?}"),
        };
        let r1 = unwrap_embed(rx1.await.expect("b1 response").expect("b1 ok"));
        let r2 = unwrap_embed(rx2.await.expect("b2 response").expect("b2 ok"));
        let r3 = unwrap_embed(rx3.await.expect("b3 response").expect("b3 ok"));
        assert_eq!(
            r1.embedding, baseline[0],
            "批内结果须与同函数单文本基线字节等同"
        );
        assert_eq!(r1.embedding, r2.embedding, "相同文本批内 scatter 须一致");
        assert_eq!(r1.dimension, r3.dimension);
    }

    /// 测试用 Mock 推理引擎——返回固定 8 维非零向量（归一化安全），不依赖任何外部模型。
    /// 定义在测试模块内,遵循 embedding.rs::tests 的 TestEngine 既有惯例。
    struct MockEngine;

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            // 返回非零向量以避免 normalize_l2 返回 Err
            Ok(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts
                .iter()
                .map(|_| vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .collect())
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

    /// 测试用 panic 引擎——armed 时 embed/embed_batch 均 panic，
    /// 随后自解除（验证 panic 隔离与 worker 存活）。
    struct PanicOnceEngine {
        armed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl InferenceEngine for PanicOnceEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            if self.armed.swap(false, Ordering::SeqCst) {
                panic!("engine exploded in single inference");
            }
            Ok(vec![1.0, 2.0, 3.0, 4.0])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            if self.armed.swap(false, Ordering::SeqCst) {
                panic!("engine exploded in batch inference");
            }
            Ok(texts.iter().map(|_| vec![1.0, 2.0, 3.0, 4.0]).collect())
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

    /// 测试用慢引擎——embed/embed_batch 睡 300ms（占住 worker 模拟在途推理）。
    struct SlowEngine;

    #[async_trait]
    impl InferenceEngine for SlowEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            std::thread::sleep(Duration::from_millis(300));
            Ok(vec![1.0, 2.0, 3.0, 4.0])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            std::thread::sleep(Duration::from_millis(300));
            Ok(texts.iter().map(|_| vec![1.0, 2.0, 3.0, 4.0]).collect())
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

    #[tokio::test]
    async fn test_worker_manager_creation() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

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

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.start().await.unwrap();

        assert_eq!(manager.current_workers(), 2);
    }

    /// 验证 decrement_worker_count 真实递减(非硬编码 0)。
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

    /// 边界场景——单 worker 停止后计数归零。
    #[test]
    fn test_decrement_worker_count_from_one_to_zero() {
        let counter = Arc::new(AtomicUsize::new(1));
        let remaining = WorkerManager::decrement_worker_count(&counter);
        assert_eq!(remaining, 0, "single worker stop should bring count to 0");
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// 连续递减多次,确认每次都生效(排除“只减一次”的假实现)。
    #[test]
    fn test_decrement_worker_count_multiple_times() {
        let counter = Arc::new(AtomicUsize::new(3));
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 2);
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 1);
        assert_eq!(WorkerManager::decrement_worker_count(&counter), 0);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// 验证 spawn_worker 真实递增 current_workers。
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

    /// 验证 spawn_single_worker(静态方法,被 start_scaling_monitor 调用)
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
        let bg_tasks = Arc::new(Mutex::new(tokio::task::JoinSet::new()));
        WorkerManager::spawn_single_worker(
            &current_workers,
            &worker_senders,
            &worker_health,
            &queue_clone,
            &response_channel_clone,
            &embedding_service_clone,
            &None,
            &config_clone,
            &running,
            &bg_tasks,
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

    /// 验证 worker_loop 消费 task_receiver 的 Shutdown 信号,
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
                     (current_workers={}) — worker_loop is not consuming task_receiver",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 验证 immediate=true 时 worker 也退出(立即关闭路径)。
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

    /// 验证 worker 退出后 sender 变为 closed,
    /// start_scaling_monitor 的 retain(!s.is_closed()) 能正确清理失效 sender。
    ///
    /// worker 因 Shutdown 退出,但 sender 留在 worker_senders 中。
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
        let request = QueuedRequest {
            request_id: "test-process-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "hello world".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };

        let result = WorkerManager::process_request(&request, &service, None).await;
        assert!(result.is_ok(), "process_request should succeed");
        let ServiceResponse::Embed(response) = result.unwrap() else {
            panic!("expected Embed response");
        };
        assert_eq!(response.dimension, 8);
        assert_eq!(response.embedding.len(), 8);
    }

    /// 验证 process_request 在引擎返回错误时传播 InferenceError。
    #[tokio::test]
    async fn test_process_request_engine_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(ErrorEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let request = QueuedRequest {
            request_id: "test-process-err".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "hello".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };

        let result = WorkerManager::process_request(&request, &service, None).await;
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
        let request = QueuedRequest {
            request_id: "test-loop-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "hello world".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        queue.enqueue(request).await.unwrap();

        manager.spawn_worker().await;

        let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(result.is_ok(), "response should arrive within 5s");
        let response_result = result.unwrap().unwrap();
        assert!(response_result.is_ok());
        let ServiceResponse::Embed(response) = response_result.unwrap() else {
            panic!("expected Embed response");
        };
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
        let request = QueuedRequest {
            request_id: "test-loop-err".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "hello".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
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

    /// 固定作答的 mock 决策引擎（同构 service/decision.rs tests 的 mock），
    /// 决策请求经 worker 出队后由 DecisionService 以 decide 契约（内部
    /// spawn_blocking）执行，worker 不把决策并入 embed 拼批。
    struct DecisionCapableEngine;

    #[async_trait]
    impl InferenceEngine for DecisionCapableEngine {
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
        fn supports_task(&self, task: ModelTask) -> bool {
            matches!(task, ModelTask::Embedding | ModelTask::Decision)
        }
        fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
            let answers = req
                .questions
                .iter()
                .map(|q| DecisionAnswer {
                    question: q.name.clone(),
                    answer: DecisionAnswerBody::Noul { p_true: 0.7 },
                })
                .collect();
            Ok(DecisionResponse {
                answers,
                processing_time_ms: 0,
            })
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// 决策请求经 worker_loop 出队到达 worker，执行结果经 response_channel
    /// 回达等待方（ServiceResponse::Decision，答案正确）。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_loop_processes_decision_request() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let decision_engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(DecisionCapableEngine));
        let decision_service = Arc::new(RwLock::new(DecisionService::new(decision_engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        )
        .with_decision_service(decision_service);

        let rx = response_channel
            .register("test-loop-decision".to_string())
            .await;
        let request = QueuedRequest {
            request_id: "test-loop-decision".to_string(),
            request: ServiceRequest::Decision(DecisionRequest {
                state: serde_json::json!({}),
                questions: vec![DecisionQuestion {
                    name: "confident".to_string(),
                    qtype: QuestionType::Noul,
                    instructions: "state your p(true)".to_string(),
                    options: vec![],
                }],
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        queue.enqueue(request).await.unwrap();

        manager.spawn_worker().await;

        let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert!(result.is_ok(), "decision response should arrive within 5s");
        let response_result = result.unwrap().unwrap();
        assert!(response_result.is_ok(), "decision request should succeed");
        match response_result.unwrap() {
            ServiceResponse::Decision(resp) => {
                assert_eq!(resp.answers.len(), 1);
                assert_eq!(resp.answers[0].question, "confident");
                match &resp.answers[0].answer {
                    DecisionAnswerBody::Noul { p_true } => {
                        assert!((p_true - 0.7).abs() < 1e-5);
                    }
                    other => panic!("Expected Noul answer, got {other:?}"),
                }
            }
            other => panic!("Expected ServiceResponse::Decision, got {other:?}"),
        }

        // 清理
        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 混批并发钉（三路评审 M）：批内决策与 embed 同批时决策 spawn 并发执行、
    /// embed 推理不被决策 inline await 阻塞（慢决策 + 快 embed 同批，embed
    /// 必须先于决策完成回达），两类请求最终都收到正确响应。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_batch_mixed_embed_and_decision_run_concurrently() {
        struct SlowDecisionEngine;

        #[async_trait]
        impl InferenceEngine for SlowDecisionEngine {
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
            fn supports_task(&self, task: ModelTask) -> bool {
                matches!(task, ModelTask::Embedding | ModelTask::Decision)
            }
            fn decide(&self, _req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
                // 慢决策：占住 inline await 的旧实现会把同批 embed 拖到其后
                std::thread::sleep(Duration::from_millis(300));
                Ok(DecisionResponse {
                    answers: vec![DecisionAnswer {
                        question: "confident".to_string(),
                        answer: DecisionAnswerBody::Noul { p_true: 0.7 },
                    }],
                    processing_time_ms: 300,
                })
            }
            async fn try_fallback_to_cpu(
                &mut self,
                _config: &ModelConfig,
            ) -> Result<(), VecboostError> {
                Ok(())
            }
        }

        let service = Arc::new(RwLock::new(EmbeddingService::new(
            Arc::new(RwLock::new(MockEngine)) as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let decision_service = Arc::new(RwLock::new(DecisionService::new(
            Arc::new(RwLock::new(SlowDecisionEngine))
                as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let channel = Arc::new(ResponseChannel::new());

        let decision_req = QueuedRequest {
            request_id: "mixed-decision".to_string(),
            request: ServiceRequest::Decision(DecisionRequest {
                state: serde_json::json!({}),
                questions: vec![DecisionQuestion {
                    name: "confident".to_string(),
                    qtype: QuestionType::Noul,
                    instructions: "state your p(true)".to_string(),
                    options: vec![],
                }],
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        // embed 在决策之后入批（数组序），旧 inline 实现会先等 300ms 决策完成
        let embed_req = make_queued("mixed-embed");
        let batch = vec![decision_req, embed_req];

        let decision_rx = channel.register("mixed-decision".to_string()).await;
        let embed_rx = channel.register("mixed-embed".to_string()).await;

        let start = std::time::Instant::now();
        WorkerManager::process_batch_requests(&batch, &service, Some(&decision_service), &channel)
            .await;
        let elapsed = start.elapsed();

        // embed 先回达且在决策的 300ms 推理窗口内完成（未被队头阻塞）
        let embed_result = tokio::time::timeout(Duration::from_millis(250), embed_rx).await;
        assert!(
            embed_result.is_ok(),
            "embed must complete within the slow-decision window (no head-of-line blocking)"
        );
        let ServiceResponse::Embed(embed_resp) = embed_result.unwrap().unwrap().unwrap() else {
            panic!("expected Embed response");
        };
        assert_eq!(embed_resp.dimension, 8);

        // 决策最终也正确回达（收齐路径）
        let decision_result = tokio::time::timeout(Duration::from_secs(5), decision_rx).await;
        assert!(decision_result.is_ok(), "decision response should arrive");
        let ServiceResponse::Decision(decision_resp) = decision_result.unwrap().unwrap().unwrap()
        else {
            panic!("expected Decision response");
        };
        assert_eq!(decision_resp.answers.len(), 1);

        // 整批耗时应与单次慢决策同量级（决策与 embed 并行，而非串行叠加）
        assert!(
            elapsed < Duration::from_millis(900),
            "mixed batch should not serialize decision + embed, took {elapsed:?}"
        );
    }

    /// 纯决策批回归钉（三路评审 H×3）：批内 2 条慢决策、0 条 embed 时，
    /// process_batch_requests 返回前两条响应必须已全部 complete（旧缺陷：
    /// texts.is_empty() 早退跳过收齐循环，JoinHandle 脱离监控——响应靠
    /// detached 任务碰巧回达、panic 无映射、节流失效）。断言
    /// pending_count 归零证明 complete 发生在本函数内而非 detached 任务。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_batch_decision_only_batch_completes_before_return() {
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            Arc::new(RwLock::new(MockEngine)) as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let decision_service = Arc::new(RwLock::new(DecisionService::new(
            Arc::new(RwLock::new(DecisionCapableEngine))
                as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let channel = Arc::new(ResponseChannel::new());

        let mk = |id: &str| QueuedRequest {
            request_id: id.to_string(),
            request: ServiceRequest::Decision(DecisionRequest {
                state: serde_json::json!({}),
                questions: vec![DecisionQuestion {
                    name: "confident".to_string(),
                    qtype: QuestionType::Noul,
                    instructions: "state your p(true)".to_string(),
                    options: vec![],
                }],
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        let batch = vec![mk("dec-only-1"), mk("dec-only-2")];
        let rx1 = channel.register("dec-only-1".to_string()).await;
        let rx2 = channel.register("dec-only-2".to_string()).await;

        WorkerManager::process_batch_requests(&batch, &service, Some(&decision_service), &channel)
            .await;

        // 返回即收齐：不依赖 detached 任务、无 30s SLA 悬挂
        assert_eq!(
            channel.pending_count().await,
            0,
            "decision-only batch must be fully completed before process_batch_requests returns"
        );
        for (rx, rid) in [(rx1, "dec-only-1"), (rx2, "dec-only-2")] {
            let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
            assert!(result.is_ok(), "{rid} response should already be sent");
            let ServiceResponse::Decision(resp) = result.unwrap().unwrap().unwrap() else {
                panic!("{rid} expected Decision response");
            };
            assert_eq!(resp.answers.len(), 1);
        }
    }

    /// 纯决策批 panic 传播钉（三路评审 H）：批内决策任务 panic 时等待方
    /// 必须及时收到 InferenceError（JoinError 显性映射），而非 detached
    /// 悬挂到 SLA 超时收 RequestTimeout。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_process_batch_decision_only_batch_panics_map_to_inference_error() {
        struct PanickingDecisionEngine;
        #[async_trait]
        impl InferenceEngine for PanickingDecisionEngine {
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
            fn supports_task(&self, task: ModelTask) -> bool {
                matches!(task, ModelTask::Embedding | ModelTask::Decision)
            }
            fn decide(&self, _req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
                panic!("decision engine exploded");
            }
            async fn try_fallback_to_cpu(
                &mut self,
                _config: &ModelConfig,
            ) -> Result<(), VecboostError> {
                Ok(())
            }
        }

        let service = Arc::new(RwLock::new(EmbeddingService::new(
            Arc::new(RwLock::new(MockEngine)) as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let decision_service = Arc::new(RwLock::new(DecisionService::new(
            Arc::new(RwLock::new(PanickingDecisionEngine))
                as Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
            None,
        )));
        let channel = Arc::new(ResponseChannel::new());

        let mk = |id: &str| QueuedRequest {
            request_id: id.to_string(),
            request: ServiceRequest::Decision(DecisionRequest {
                state: serde_json::json!({}),
                questions: vec![DecisionQuestion {
                    name: "confident".to_string(),
                    qtype: QuestionType::Noul,
                    instructions: "state your p(true)".to_string(),
                    options: vec![],
                }],
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };
        let batch = vec![mk("dec-panic-1"), mk("dec-panic-2")];
        let rx1 = channel.register("dec-panic-1".to_string()).await;
        let rx2 = channel.register("dec-panic-2".to_string()).await;

        WorkerManager::process_batch_requests(&batch, &service, Some(&decision_service), &channel)
            .await;

        assert_eq!(channel.pending_count().await, 0);
        for (rx, rid) in [(rx1, "dec-panic-1"), (rx2, "dec-panic-2")] {
            let result = tokio::time::timeout(Duration::from_secs(2), rx).await;
            assert!(result.is_ok(), "{rid} must not hang past the collect loop");
            match result.unwrap().unwrap() {
                Err(VecboostError::InferenceError(msg)) => {
                    assert!(
                        msg.contains("decision inference task"),
                        "panic must map to InferenceError, got: {msg}"
                    );
                }
                other => panic!("{rid} expected InferenceError, got {other:?}"),
            }
        }
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
            let health = manager.worker_health.lock().await;
            if health.len() == 1 && !health[0].is_alive && manager.current_workers() == 0 {
                break;
            }
            drop(health);
            if tokio::time::Instant::now() >= deadline {
                let health = manager.worker_health.lock().await;
                panic!(
                    "worker did not exit within 2s (current_workers={}, health_len={}, is_alive={})",
                    manager.current_workers(),
                    health.len(),
                    health.first().map(|h| h.is_alive).unwrap_or(false)
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
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
            let health = manager.worker_health.lock().await;
            if health.len() == 1 && !health[0].is_alive && manager.current_workers() == 0 {
                break;
            }
            drop(health);
            if tokio::time::Instant::now() >= deadline {
                let health = manager.worker_health.lock().await;
                panic!(
                    "worker did not exit within 5s when running=false (current_workers={}, health_len={}, is_alive={})",
                    manager.current_workers(),
                    health.len(),
                    health.first().map(|h| h.is_alive).unwrap_or(false)
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// 回归钉（T029/D03）：引擎推理 panic 被隔离——批内请求收到
    /// InternalError，worker 存活并继续服务后续请求。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_worker_survives_engine_panic() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig::default();
        let panic_armed = Arc::new(AtomicBool::new(true));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(PanicOnceEngine {
                armed: Arc::clone(&panic_armed),
            }));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );
        manager.spawn_worker().await;

        let rx1 = response_channel.register("panic-1".to_string()).await;
        let rx2 = response_channel.register("after-panic".to_string()).await;
        // 错峰入队：两请求分属不同批（panic 只波及第一批）
        queue
            .enqueue(QueuedRequest {
                request_id: "panic-1".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "trigger".to_string(),
                    normalize: Some(false),
                }),
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
            })
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        queue
            .enqueue(QueuedRequest {
                request_id: "after-panic".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "recover".to_string(),
                    normalize: Some(false),
                }),
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
            })
            .await
            .unwrap();

        let r1 = tokio::time::timeout(Duration::from_secs(5), rx1)
            .await
            .expect("panic batch must complete promptly")
            .expect("channel open");
        assert!(
            r1.is_err(),
            "panicked batch must yield explicit error, got {:?}",
            r1
        );

        // worker 存活：后续请求正常完成
        let r2 = tokio::time::timeout(Duration::from_secs(5), rx2)
            .await
            .expect("worker must survive panic and serve next request")
            .expect("channel open");
        assert!(
            r2.is_ok(),
            "post-panic request must succeed: {:?}",
            r2.err()
        );

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s_ in &senders {
            let _ = s_.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// 回归钉（T030/D04）：已取消请求在出队时被丢弃——不触发引擎调用，
    /// 队列槽位释放。
    #[tokio::test]
    async fn test_cancelled_request_skipped_at_dequeue() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        // 短 timeout 通道：取消丢弃路径的条目由 cleanup_expired 按自身
        // timeout 到期回收（worker 收割点生产接线），此处钉回收闭环
        let response_channel = Arc::new(ResponseChannel::with_timeout(Duration::from_millis(80)));

        let flag = queue.cancellations().register("cancelled-1");
        flag.store(true, Ordering::SeqCst);

        queue
            .enqueue(QueuedRequest {
                request_id: "cancelled-1".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "ghost".to_string(),
                    normalize: Some(false),
                }),
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
            })
            .await
            .unwrap();
        assert_eq!(queue.size(), 1);

        // handler 侧注册的通道条目在断连/超时后不再有人接收——
        // register 的 rx 随 handler drop，仅剩 map 条目滞留
        let _rx_dropped = response_channel.register("cancelled-1".to_string()).await;
        drop(_rx_dropped);

        let dequeued = queue.dequeue().await;
        assert!(
            dequeued.is_none(),
            "cancelled request must be dropped, not served"
        );
        assert_eq!(queue.size(), 0, "queue slot must be released");

        // 取消丢弃不产生 complete：条目滞留至自身 timeout，由 worker 收割点的
        // cleanup_expired 回收（无界增长面闭合钉，三路评审 M×3）
        assert_eq!(
            response_channel.pending_count().await,
            1,
            "entry must survive until its own timeout (no complete for cancelled drops)"
        );
        tokio::time::sleep(Duration::from_millis(120)).await;
        response_channel.cleanup_expired().await;
        assert_eq!(
            response_channel.pending_count().await,
            0,
            "expired entry of a cancelled-dropped request must be reclaimed by cleanup"
        );
    }

    /// 回归钉（T032/D07）：shutdown 排空——排队请求立即收到
    /// "server is shutting down" 错误而非悬挂到客户端超时。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_shutdown_drains_queue_with_immediate_error() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 1,
            max_workers: 1,
            batch_wait_ms: 0,
            ..Default::default()
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(SlowEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let manager = WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            config,
            service,
        );
        manager.spawn_worker().await;

        let rx_queued = response_channel.register("drain-me".to_string()).await;
        for id in ["in-flight", "drain-me"] {
            queue
                .enqueue(QueuedRequest {
                    request_id: id.to_string(),
                    request: ServiceRequest::Embed(EmbedRequest {
                        text: format!("text {id}"),
                        normalize: Some(false),
                    }),
                    priority: Priority::Normal,
                    submitted_at: std::time::Instant::now(),
                    timeout: Duration::from_secs(30),
                    source: RequestSource::http("127.0.0.1".to_string()),
                })
                .await
                .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;

        manager.shutdown().await;

        let outcome = tokio::time::timeout(Duration::from_secs(1), rx_queued).await;
        match outcome {
            Ok(Ok(Err(VecboostError::InferenceError(msg)))) => {
                assert!(msg.contains("shutting down"), "got: {msg}");
            }
            Ok(Ok(Ok(_))) => panic!("queued request must be drained with error, got success"),
            Ok(Ok(Err(other))) => panic!("unexpected error type: {:?}", other),
            Ok(Err(_)) => panic!("response channel closed without completing"),
            Err(_) => panic!("queued request must complete promptly during shutdown"),
        }
    }

    /// 回归钉（T006 唤醒不塌缩）：N 个突发请求入队、W 个空闲 worker，
    /// 全部请求必须在 5s 内完成（tokio Notify permit 语义保证不丢唤醒；
    /// 未被唤醒的 worker 由出队臂兜底消费队列）。
    #[tokio::test(flavor = "multi_thread")]
    async fn test_burst_enqueues_all_complete_with_idle_workers() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 3,
            max_workers: 3,
            batch_wait_ms: 0,
            ..Default::default()
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
        assert_eq!(manager.current_workers(), 3);

        let mut rxs = Vec::with_capacity(6);
        for i in 0..6 {
            rxs.push(response_channel.register(format!("burst-{}", i)).await);
        }
        for i in 0..6 {
            queue
                .enqueue(QueuedRequest {
                    request_id: format!("burst-{}", i),
                    request: ServiceRequest::Embed(EmbedRequest {
                        text: format!("burst text {}", i),
                        normalize: Some(false),
                    }),
                    priority: Priority::Normal,
                    submitted_at: std::time::Instant::now(),
                    timeout: Duration::from_secs(30),
                    source: RequestSource::http("127.0.0.1".to_string()),
                })
                .await
                .unwrap();
        }

        for (i, rx) in rxs.into_iter().enumerate() {
            let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
            assert!(
                result.is_ok(),
                "burst request {} not completed within 5s",
                i
            );
            let response = result.unwrap().expect("channel open");
            assert!(
                response.is_ok(),
                "burst request {} must succeed: {:?}",
                i,
                response.err()
            );
        }

        // 清理
        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
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
            let request = QueuedRequest {
                request_id: format!("multi-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: format!("text-{}", i),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
            };
            queue.enqueue(request).await.unwrap();
        }

        manager.spawn_worker().await;

        for (i, rx) in rxs {
            let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
            assert!(result.is_ok(), "response {} should arrive within 5s", i);
            let response_result = result.unwrap().unwrap();
            assert!(response_result.is_ok());
            let ServiceResponse::Embed(response) = response_result.unwrap() else {
                panic!("expected Embed response");
            };
            assert_eq!(response.dimension, 8);
        }

        manager.running.store(false, Ordering::SeqCst);
        let senders = manager.worker_senders.lock().await.clone();
        for s in &senders {
            let _ = s.send(WorkerTask::Shutdown { immediate: true }).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    /// 验证 worker_loop 在运行过程中持续更新 last_active_time。
    ///
    /// 使用 `current_thread` flavor 避免 `multi_thread` 下高并行测试负载导致的
    /// 线程饥饿(scheduler starvation)。在 `current_thread` 中,`tokio::spawn`
    /// 的 worker task 会在每次 `await` 让出控制权时被调度执行。
    #[tokio::test(flavor = "current_thread")]
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

        // Poll for last_active_time refresh. In current_thread runtime, the
        // worker task runs during our sleep().await yield points. Under
        // coverage instrumentation the loop may be slower, so allow 30s.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        #[allow(unused_assignments)]
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
                    "last_active_time was not refreshed within 30s \
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
        let request = QueuedRequest {
            request_id: "test-none-norm".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "hello".to_string(),
                normalize: None,
            }),
            priority: Priority::Normal,
            submitted_at: std::time::Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
        };

        let result = WorkerManager::process_request(&request, &service, None).await;
        assert!(result.is_ok());
        let ServiceResponse::Embed(response) = result.unwrap() else {
            panic!("expected Embed response");
        };
        assert_eq!(response.dimension, 8);
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
            &manager.decision_service,
            &manager.config,
            &manager.running,
            &manager.bg_tasks,
        )
        .await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let health = manager.worker_health.lock().await;
            if health.len() == 1
                && health[0].worker_id == 0
                && !health[0].is_alive
                && manager.current_workers() == 0
            {
                break;
            }
            drop(health);
            if tokio::time::Instant::now() >= deadline {
                let health = manager.worker_health.lock().await;
                panic!(
                    "worker did not exit within 2s (current_workers={}, health_len={}, is_alive={})",
                    manager.current_workers(),
                    health.len(),
                    health.first().map(|h| h.is_alive).unwrap_or(false)
                );
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
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
            drain_timeout_secs: 5,
            scale_check_interval_secs: 1,
            max_batch_size: 8,
            batch_wait_ms: 5,
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
            let request = QueuedRequest {
                request_id: format!("scale-up-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: format!("text-{}", i),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: std::time::Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::http("127.0.0.1".to_string()),
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
            drain_timeout_secs: 5,
            scale_check_interval_secs: 1,
            max_batch_size: 8,
            batch_wait_ms: 5,
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
    async fn test_worker_loop_exits_after_idle_timeout_when_channel_closed() {
        let queue = Arc::new(PriorityRequestQueue::new(100));
        let response_channel = Arc::new(ResponseChannel::new());
        let config = WorkerConfig {
            min_workers: 0,
            max_workers: 2,
            idle_timeout_secs: 1, // 短超时,让 worker 快速回到循环顶检查 running
            ..Default::default()
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let manager = WorkerManager::new(queue, response_channel, config, service);

        manager.spawn_worker().await;
        assert_eq!(manager.current_workers(), 1);

        // 关闭 task channel:清空 senders 使 task_receiver.recv() 返回 None,
        // 同时队列为空 → select! 进入 notify timeout 分支
        {
            let mut senders = manager.worker_senders.lock().await;
            senders.clear();
        }

        // 等待 idle timeout 首次执行(1s timeout)
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // worker 仍在运行(idle_count 未超 MAX_IDLE_COUNT=10 或 worker_id 不大于 min_workers)
        assert_eq!(
            manager.current_workers(),
            1,
            "worker should still be running after idle timeout"
        );

        // 设置 running=false,worker 在下次 timeout 后循环顶部退出
        manager.running.store(false, Ordering::SeqCst);

        // 轮询等待退出(idle_timeout_secs=1,给 5s 余量)
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if manager.current_workers() == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!(
                    "worker did not exit within 5s after running=false (current_workers={})",
                    manager.current_workers()
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
