// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 调度抽象类型(优先级/请求来源/服务请求/队列请求)。
//!
//! 从 pipeline 下沉到 domain:device(continuous_batch)与 pipeline 双向依赖
//! 这些类型,归入 domain 后两者均单向依赖 domain,宏观循环打断。
//! `PriorityRequestQueue`/`PriorityCalculator` 等调度器仍属 pipeline。

use std::time::{Duration, Instant};

use crate::domain::{EmbedRequest, RerankRequest};
use crate::error::VecboostError;
use crate::i18n;

/// 优先级枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Critical = 100,
    High = 75,
    Normal = 50,
    Low = 25,
}

impl Priority {
    pub fn from_score(score: i32) -> Self {
        if score >= 90 {
            Priority::Critical
        } else if score >= 65 {
            Priority::High
        } else if score >= 40 {
            Priority::Normal
        } else {
            Priority::Low
        }
    }

    pub fn as_i32(&self) -> i32 {
        *self as i32
    }
}

/// 请求来源
#[derive(Debug, Clone)]
pub enum RequestSource {
    Http { ip: String },
    Grpc { client_id: String },
    Internal,
}

impl RequestSource {
    pub fn http(ip: String) -> Self {
        RequestSource::Http { ip }
    }

    pub fn grpc(client_id: String) -> Self {
        RequestSource::Grpc { client_id }
    }

    pub fn internal() -> Self {
        RequestSource::Internal
    }
}

/// 服务请求枚举 — 支持嵌入和重排序两种请求类型
#[derive(Debug, Clone)]
pub enum ServiceRequest {
    Embed(EmbedRequest),
    Rerank(RerankRequest),
}

impl ServiceRequest {
    /// 提取嵌入请求，非 Embed 变体时返回错误
    pub fn into_embed(self) -> Result<EmbedRequest, VecboostError> {
        match self {
            ServiceRequest::Embed(req) => Ok(req),
            ServiceRequest::Rerank(_) => Err(VecboostError::InternalError(i18n::tr(
                "queue-type-mismatch",
            ))),
        }
    }
}

/// 队列请求
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    /// 请求 ID
    pub request_id: String,
    /// 服务请求
    pub request: ServiceRequest,
    /// 优先级
    pub priority: Priority,
    /// 提交时间
    pub submitted_at: Instant,
    /// 超时时间
    pub timeout: Duration,
    /// 请求来源
    pub source: RequestSource,
}

// ---- PriorityRequestQueue ----

use log::{debug, warn};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Notify;

/// 优先级老化阈值(默认 5s)—— 必须显著小于请求 SLA(默认 30s),否则
/// 老化跳过在请求可达超时前永不触发,四级优先队列形同虚设。
/// 老化语义=跳级(公平性),不等于超时:被跳过的队首由 [`PriorityRequestQueue::dequeue_expired`]
/// 按各自 SLA 收割,或在下级队列为空时被兜底服务,任何情况下不永久滞留。
const DEFAULT_AGING_THRESHOLD: Duration = Duration::from_secs(5);

/// 请求取消注册表：request_id → 取消标志（T030 断连取消传播）。
///
/// handler 在入队前注册；客户端断连/超时置位；调度器出队时检查，
/// 已取消请求直接丢弃（释放队列槽位、不消耗推理算力）。
/// 释放时机：worker 完成后或调度器丢弃时——不早于最后一次检查。
#[derive(Default)]
pub struct CancellationRegistry {
    flags: std::sync::Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>,
}

impl CancellationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册并返回可克隆的取消标志。
    pub fn register(&self, request_id: &str) -> Arc<std::sync::atomic::AtomicBool> {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.flags
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id.to_string(), Arc::clone(&flag));
        flag
    }

    pub fn is_cancelled(&self, request_id: &str) -> bool {
        self.flags
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(request_id)
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn cancel(&self, request_id: &str) {
        if let Some(f) = self
            .flags
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(request_id)
        {
            f.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn release(&self, request_id: &str) {
        self.flags
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(request_id);
    }
}

/// 优先级请求队列
pub struct PriorityRequestQueue {
    /// 队列: Priority -> 请求队列
    queues: Arc<tokio::sync::RwLock<BTreeMap<Priority, VecDeque<QueuedRequest>>>>,
    /// 最大队列大小
    max_queue_size: usize,
    /// 当前队列大小
    current_size: Arc<AtomicUsize>,
    /// 入队通知——worker 通过 notified() 等待，消除轮询退避
    notify: Arc<Notify>,
    /// 老化跳级阈值（`with_aging_threshold` 可覆盖默认值）
    aging_threshold: Duration,
    /// 取消注册表（出队跳过已取消请求）
    cancellations: Arc<CancellationRegistry>,
}

impl PriorityRequestQueue {
    pub fn new(max_queue_size: usize) -> Self {
        debug!(
            "Creating PriorityRequestQueue with max_size={}",
            max_queue_size
        );

        Self {
            queues: Arc::new(tokio::sync::RwLock::new(BTreeMap::new())),
            max_queue_size,
            current_size: Arc::new(AtomicUsize::new(0)),
            notify: Arc::new(Notify::new()),
            aging_threshold: DEFAULT_AGING_THRESHOLD,
            cancellations: Arc::new(CancellationRegistry::new()),
        }
    }

    /// 取消注册表句柄（handler 注册标志、worker 完成后释放）
    pub fn cancellations(&self) -> Arc<CancellationRegistry> {
        Arc::clone(&self.cancellations)
    }

    /// 覆盖老化跳级阈值（测试与部署调优用）
    pub fn with_aging_threshold(mut self, threshold: Duration) -> Self {
        self.aging_threshold = threshold;
        self
    }

    /// 入队
    pub async fn enqueue(&self, request: QueuedRequest) -> Result<(), VecboostError> {
        // 使用原子操作确保检查和入队的原子性
        loop {
            let current_size = self.current_size.load(Ordering::Acquire);

            if current_size >= self.max_queue_size {
                return Err(VecboostError::RateLimitExceeded(i18n::tr(
                    "queue-full-rejected",
                )));
            }

            match self.current_size.compare_exchange_weak(
                current_size,
                current_size + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }

        let mut queues = self.queues.write().await;

        let priority = request.priority;
        let queue = queues.entry(priority).or_insert_with(VecDeque::new);
        queue.push_back(request);

        self.notify.notify_one();

        debug!(
            "Request enqueued, priority={:?}, queue_size={}",
            priority,
            self.current_size.load(Ordering::Relaxed)
        );

        Ok(())
    }

    /// 出队（按优先级,含老化机制防止低优先级饥饿）
    ///
    /// 队首等待超过老化阈值(默认 5s)的非 Low 优先级被跳过,防止低优先级
    /// 请求永久饥饿。被跳过的请求不会滞留:`dequeue_expired` 按各自 SLA 收割,
    /// 或在所有可服务层级均已空时,兜底弹出最高优先级的 aged 队首（其仍在
    /// 自身 SLA 内,正常服务优于悬挂到超时）。
    pub async fn dequeue(&self) -> Option<QueuedRequest> {
        let mut queues = self.queues.write().await;
        let now = Instant::now();

        // 按优先级从高到低查找
        let mut aged_front: Option<Priority> = None;
        for priority in [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ] {
            if let Some(queue) = queues.get_mut(&priority) {
                // 老化检查:队首等待超阈值 → 记录并跳到下一优先级
                if priority != Priority::Low
                    && let Some(front) = queue.front()
                    && now.duration_since(front.submitted_at) > self.aging_threshold
                {
                    if aged_front.is_none() {
                        aged_front = Some(priority);
                    }
                    continue;
                }
                while let Some(request) = queue.pop_front() {
                    // 已取消请求：丢弃（释放槽位），继续扫描（T030）
                    if self.cancellations.is_cancelled(&request.request_id) {
                        self.current_size.fetch_sub(1, Ordering::Relaxed);
                        self.cancellations.release(&request.request_id);
                        debug!(
                            "Cancelled request {} dropped at dequeue",
                            request.request_id
                        );
                        continue;
                    }
                    let new_size = self.current_size.fetch_sub(1, Ordering::Relaxed) - 1;

                    debug!(
                        "Request dequeued, priority={:?}, queue_size={}",
                        priority, new_size
                    );

                    if queue.is_empty() {
                        queues.remove(&priority);
                    }

                    return Some(request);
                }
            }
        }

        // 兜底:无非 aged 请求可服务时,弹出最高优先级的 aged 队首,避免滞留
        if let Some(priority) = aged_front
            && let Some(queue) = queues.get_mut(&priority)
            && let Some(request) = queue.pop_front()
        {
            let new_size = self.current_size.fetch_sub(1, Ordering::Relaxed) - 1;
            debug!(
                "Aged front served as fallback, priority={:?}, queue_size={}",
                priority, new_size
            );
            if queue.is_empty() {
                queues.remove(&priority);
            }
            return Some(request);
        }

        None
    }

    /// 收割已超过自身 SLA（`QueuedRequest.timeout`）的排队请求。
    ///
    /// 与老化跳级正交:老化是公平性机制（秒级）,收割是 SLA 机制（默认 30s）。
    /// worker 循环周期性调用并对其完成超时响应,保证任何请求在队列中的
    /// 滞留时间有上界,不会出现"队首被跳过后永久搁浅"的泄漏。
    pub async fn dequeue_expired(&self) -> Vec<QueuedRequest> {
        let mut queues = self.queues.write().await;
        let now = Instant::now();
        let mut expired = Vec::new();

        let priorities: Vec<Priority> = queues.keys().copied().collect();
        for priority in priorities {
            if let Some(queue) = queues.get_mut(&priority) {
                let mut remaining = VecDeque::new();
                for req in queue.drain(..) {
                    if now.duration_since(req.submitted_at) >= req.timeout {
                        expired.push(req);
                    } else {
                        remaining.push_back(req);
                    }
                }
                *queue = remaining;
                if queue.is_empty() {
                    queues.remove(&priority);
                }
            }
        }

        if !expired.is_empty() {
            let new_size = self
                .current_size
                .fetch_sub(expired.len(), Ordering::Relaxed)
                - expired.len();
            warn!(
                "Reaped {} expired requests from queue, queue_size={}",
                expired.len(),
                new_size
            );
        }

        expired
    }

    /// 批量出队——取首个请求后继续 try_dequeue 至 max_batch_size。
    ///
    /// 返回至少 1 个请求（调用前须确保队列非空），最多 max_batch_size 个。
    /// 按优先级顺序出队。
    pub async fn dequeue_batch(&self, max_batch_size: usize) -> Vec<QueuedRequest> {
        let mut result = Vec::with_capacity(max_batch_size);
        let mut queues = self.queues.write().await;
        let now = Instant::now();
        let mut aged_front: Option<Priority> = None;

        for priority in [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ] {
            if result.len() >= max_batch_size {
                break;
            }
            if let Some(queue) = queues.get_mut(&priority) {
                while result.len() < max_batch_size {
                    // 老化检查
                    if priority != Priority::Low
                        && let Some(front) = queue.front()
                        && now.duration_since(front.submitted_at) > self.aging_threshold
                    {
                        if aged_front.is_none() {
                            aged_front = Some(priority);
                        }
                        break;
                    }
                    if let Some(request) = queue.pop_front() {
                        if self.cancellations.is_cancelled(&request.request_id) {
                            self.current_size.fetch_sub(1, Ordering::Relaxed);
                            self.cancellations.release(&request.request_id);
                            continue;
                        }
                        self.current_size.fetch_sub(1, Ordering::Relaxed);
                        result.push(request);
                    } else {
                        break;
                    }
                }
                if queue.is_empty() {
                    queues.remove(&priority);
                }
            }
        }

        // 兜底:未凑到任何请求时,弹出最高优先级 aged 队首,避免滞留
        if result.is_empty()
            && let Some(priority) = aged_front
            && let Some(queue) = queues.get_mut(&priority)
            && let Some(request) = queue.pop_front()
        {
            self.current_size.fetch_sub(1, Ordering::Relaxed);
            result.push(request);
            if queue.is_empty() {
                queues.remove(&priority);
            }
        }

        debug!("Batch dequeued: {} requests", result.len());
        result
    }

    /// 获取最高优先级
    pub async fn peek_highest_priority(&self) -> Option<Priority> {
        let queues = self.queues.read().await;

        for priority in [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ] {
            if let Some(queue) = queues.get(&priority)
                && !queue.is_empty()
            {
                return Some(priority);
            }
        }

        None
    }

    /// 获取队列大小
    pub fn size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    /// 获取入队通知引用，供 worker select! 使用
    pub fn notify(&self) -> &Notify {
        &self.notify
    }

    /// 排空队列并返回全部请求（优雅停机善后用，调用方负责完成响应）。
    pub async fn dequeue_all_for_shutdown(&self) -> Vec<QueuedRequest> {
        let mut queues = self.queues.write().await;
        let mut all = Vec::new();
        for queue in queues.values_mut() {
            while let Some(req) = queue.pop_front() {
                self.cancellations.release(&req.request_id);
                all.push(req);
            }
        }
        queues.clear();
        self.current_size.store(0, Ordering::Relaxed);
        if !all.is_empty() {
            warn!("Shutdown drain: {} queued requests evacuated", all.len());
        }
        all
    }

    /// 清空队列
    pub async fn clear(&self) {
        let mut queues = self.queues.write().await;
        let cleared_count = queues.values().map(|q| q.len()).sum::<usize>();

        queues.clear();
        self.current_size.store(0, Ordering::Relaxed);

        warn!("Queue cleared, {} requests discarded", cleared_count);
    }

    /// 获取按优先级分组的队列大小
    pub async fn size_by_priority(&self) -> Vec<(Priority, usize)> {
        let queues = self.queues.read().await;

        queues
            .iter()
            .map(|(priority, queue)| (*priority, queue.len()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_queue_creation() {
        let queue = PriorityRequestQueue::new(100);
        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_enqueue_dequeue() {
        let queue = PriorityRequestQueue::new(100);
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };

        queue.enqueue(request).await.unwrap();
        assert_eq!(queue.size(), 1);

        let dequeued = queue.dequeue().await;
        assert!(dequeued.is_some());
        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = PriorityRequestQueue::new(100);

        for (i, priority) in [
            Priority::Low,
            Priority::Critical,
            Priority::Normal,
            Priority::High,
        ]
        .iter()
        .enumerate()
        {
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: *priority,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            };

            queue.enqueue(request).await.unwrap();
        }

        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Critical);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::High);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Normal);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Low);
    }

    #[tokio::test]
    async fn test_queue_full() {
        let queue = PriorityRequestQueue::new(2);
        let request1 = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        let request2 = QueuedRequest {
            request_id: "test-2".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };

        queue.enqueue(request1).await.unwrap();
        queue.enqueue(request2).await.unwrap();
        let request3 = QueuedRequest {
            request_id: "test-3".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };

        let result = queue.enqueue(request3).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clear() {
        let queue = PriorityRequestQueue::new(100);

        for i in 0..10 {
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            };

            queue.enqueue(request).await.unwrap();
        }

        assert_eq!(queue.size(), 10);

        queue.clear().await;

        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_aging_prevents_low_priority_starvation() {
        // 可配阈值加速测试：50ms 老化窗口 + 80ms 等待
        let queue = PriorityRequestQueue::new(100).with_aging_threshold(Duration::from_millis(50));

        // 入队 Critical 请求(会老化)
        let critical_req = QueuedRequest {
            request_id: "critical-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Critical,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(60),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(critical_req).await.unwrap();

        let low_req = QueuedRequest {
            request_id: "low-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Low,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(60),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(low_req).await.unwrap();

        // 等待超过老化阈值(50ms)
        tokio::time::sleep(Duration::from_millis(80)).await;

        // dequeue 应先返回 Low(Critical 已老化,跳过)
        let dequeued = queue.dequeue().await.unwrap();
        assert_eq!(
            dequeued.priority,
            Priority::Low,
            "aged Critical should be skipped, Low should be dequeued first"
        );
    }

    /// 回归钉（D01 老化反向）：仅剩 aged 队首时 dequeue 必须兜底弹出，
    /// 而非返回 None 使该请求永久搁浅。旧行为：跳过整级且不弹队首 → None。
    #[tokio::test]
    async fn test_aged_front_served_when_no_lower_priority() {
        let queue = PriorityRequestQueue::new(100).with_aging_threshold(Duration::from_millis(50));
        queue
            .enqueue(QueuedRequest {
                request_id: "critical-aged".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Critical,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            })
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;

        let dequeued = queue
            .dequeue()
            .await
            .expect("aged front must be served when no lower priority exists");
        assert_eq!(dequeued.request_id, "critical-aged");
        assert_eq!(
            queue.size(),
            0,
            "queue must be empty after serving aged front"
        );
    }

    /// 回归钉（T008 SLA 收割）：超过自身 timeout 的排队请求被 dequeue_expired
    /// 收割并移出队列，出队计数归零，dequeue 不再返回它。
    #[tokio::test]
    async fn test_dequeue_expired_reaps_past_sla_requests() {
        let queue = PriorityRequestQueue::new(100);
        queue
            .enqueue(QueuedRequest {
                request_id: "short-sla".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_millis(50),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            })
            .await
            .unwrap();
        queue
            .enqueue(QueuedRequest {
                request_id: "long-sla".to_string(),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test-2".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            })
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;

        let expired = queue.dequeue_expired().await;
        assert_eq!(expired.len(), 1, "only the past-SLA request is reaped");
        assert_eq!(expired[0].request_id, "short-sla");
        assert_eq!(queue.size(), 1, "long-SLA request must stay queued");

        // 收割后再 dequeue 只会拿到未过期请求
        let served = queue.dequeue().await.unwrap();
        assert_eq!(served.request_id, "long-sla");
        assert!(queue.dequeue_expired().await.is_empty());
        assert_eq!(queue.size(), 0);
    }

    // ===== peek_highest_priority tests =====

    #[tokio::test]
    async fn test_peek_highest_priority_empty_queue_returns_none() {
        let queue = PriorityRequestQueue::new(100);
        let result = queue.peek_highest_priority().await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_peek_highest_priority_returns_critical() {
        let queue = PriorityRequestQueue::new(100);
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Critical,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(request).await.unwrap();
        let result = queue.peek_highest_priority().await;
        assert_eq!(result, Some(Priority::Critical));
    }

    #[tokio::test]
    async fn test_peek_highest_priority_returns_low() {
        let queue = PriorityRequestQueue::new(100);
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Low,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(request).await.unwrap();
        let result = queue.peek_highest_priority().await;
        assert_eq!(result, Some(Priority::Low));
    }

    #[tokio::test]
    async fn test_peek_highest_priority_after_dequeue() {
        let queue = PriorityRequestQueue::new(100);
        for priority in [Priority::High, Priority::Low] {
            let request = QueuedRequest {
                request_id: format!("test-{:?}", priority),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            };
            queue.enqueue(request).await.unwrap();
        }
        assert_eq!(queue.peek_highest_priority().await, Some(Priority::High));
        queue.dequeue().await.unwrap();
        assert_eq!(queue.peek_highest_priority().await, Some(Priority::Low));
        queue.dequeue().await.unwrap();
        assert!(queue.peek_highest_priority().await.is_none());
    }

    // ===== size_by_priority tests =====

    #[tokio::test]
    async fn test_size_by_priority_empty_queue() {
        let queue = PriorityRequestQueue::new(100);
        let result = queue.size_by_priority().await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_size_by_priority_single_priority() {
        let queue = PriorityRequestQueue::new(100);
        for i in 0..3 {
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            };
            queue.enqueue(request).await.unwrap();
        }
        let result = queue.size_by_priority().await;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (Priority::Normal, 3));
    }

    #[tokio::test]
    async fn test_size_by_priority_multiple_priorities() {
        let queue = PriorityRequestQueue::new(100);
        let priorities_with_counts = [
            (Priority::Critical, 2),
            (Priority::High, 1),
            (Priority::Low, 3),
        ];
        for (priority, count) in priorities_with_counts {
            for i in 0..count {
                let request = QueuedRequest {
                    request_id: format!("test-{:?}-{}", priority, i),
                    request: ServiceRequest::Embed(EmbedRequest {
                        text: "test".to_string(),
                        normalize: Some(true),
                    }),
                    priority,
                    submitted_at: Instant::now(),
                    timeout: Duration::from_secs(30),
                    source: RequestSource::Http {
                        ip: "127.0.0.1".to_string(),
                    },
                };
                queue.enqueue(request).await.unwrap();
            }
        }
        let result = queue.size_by_priority().await;
        let total: usize = result.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 6);
    }

    // ===== dequeue from empty queue =====

    #[tokio::test]
    async fn test_dequeue_empty_queue_returns_none() {
        let queue = PriorityRequestQueue::new(100);
        let result = queue.dequeue().await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_dequeue_all_then_empty() {
        let queue = PriorityRequestQueue::new(100);
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(request).await.unwrap();
        assert!(queue.dequeue().await.is_some());
        assert!(queue.dequeue().await.is_none());
        assert_eq!(queue.size(), 0);
    }

    // ===== clear on empty queue =====

    #[tokio::test]
    async fn test_clear_empty_queue() {
        let queue = PriorityRequestQueue::new(100);
        queue.clear().await;
        assert_eq!(queue.size(), 0);
    }

    // ===== queue size tracking after operations =====

    #[tokio::test]
    async fn test_size_reflects_enqueue_and_dequeue() {
        let queue = PriorityRequestQueue::new(100);
        assert_eq!(queue.size(), 0);
        for i in 0..5 {
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                request: ServiceRequest::Embed(EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                }),
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
            };
            queue.enqueue(request).await.unwrap();
        }
        assert_eq!(queue.size(), 5);
        for _ in 0..3 {
            queue.dequeue().await.unwrap();
        }
        assert_eq!(queue.size(), 2);
    }

    #[tokio::test]
    async fn test_enqueue_max_size_zero_always_rejects() {
        let queue = PriorityRequestQueue::new(0);
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        let result = queue.enqueue(request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_aging_does_not_skip_low_priority() {
        let queue = PriorityRequestQueue::new(100);
        let low_req = QueuedRequest {
            request_id: "low-1".to_string(),
            request: ServiceRequest::Embed(EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            }),
            priority: Priority::Low,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(60),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
        };
        queue.enqueue(low_req).await.unwrap();
        // Low priority does NOT participate in aging skip
        let dequeued = queue.dequeue().await.unwrap();
        assert_eq!(dequeued.priority, Priority::Low);
    }

    // ===== dequeue_batch tests =====

    #[tokio::test]
    async fn test_dequeue_batch_returns_all_when_under_limit() {
        let queue = PriorityRequestQueue::new(100);
        for i in 0..3 {
            queue
                .enqueue(QueuedRequest {
                    request_id: format!("req-{}", i),
                    request: ServiceRequest::Embed(EmbedRequest {
                        text: format!("text-{}", i),
                        normalize: Some(true),
                    }),
                    priority: Priority::Normal,
                    submitted_at: Instant::now(),
                    timeout: Duration::from_secs(30),
                    source: RequestSource::Http {
                        ip: "127.0.0.1".to_string(),
                    },
                })
                .await
                .unwrap();
        }
        let batch = queue.dequeue_batch(5).await;
        assert_eq!(batch.len(), 3, "should return all 3 when under limit");
        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_dequeue_batch_respects_max_limit() {
        let queue = PriorityRequestQueue::new(100);
        for i in 0..8 {
            queue
                .enqueue(QueuedRequest {
                    request_id: format!("req-{}", i),
                    request: ServiceRequest::Embed(EmbedRequest {
                        text: format!("text-{}", i),
                        normalize: Some(true),
                    }),
                    priority: Priority::Normal,
                    submitted_at: Instant::now(),
                    timeout: Duration::from_secs(30),
                    source: RequestSource::Http {
                        ip: "127.0.0.1".to_string(),
                    },
                })
                .await
                .unwrap();
        }
        let batch = queue.dequeue_batch(4).await;
        assert_eq!(batch.len(), 4, "should cap at max_batch_size");
        assert_eq!(queue.size(), 4, "remaining 4 should stay in queue");
    }

    #[tokio::test]
    async fn test_dequeue_batch_empty_queue_returns_empty() {
        let queue = PriorityRequestQueue::new(100);
        let batch = queue.dequeue_batch(5).await;
        assert!(batch.is_empty());
    }
}
