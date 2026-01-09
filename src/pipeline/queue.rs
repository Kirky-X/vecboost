// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

#![allow(clippy::all)]

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{debug, warn};

use super::priority::{Priority, RequestSource};
use crate::domain::EmbedRequest;
use crate::error::AppError;

/// 队列请求
#[derive(Debug)]
pub struct QueuedRequest {
    /// 请求 ID
    pub request_id: String,
    /// 嵌入请求
    pub embed_request: EmbedRequest,
    /// 优先级
    pub priority: Priority,
    /// 提交时间
    pub submitted_at: Instant,
    /// 超时时间
    pub timeout: Duration,
    /// 请求来源
    pub source: RequestSource,
    /// 响应发送器
    pub response_tx: oneshot::Sender<Result<crate::domain::EmbedResponse, AppError>>,
}

/// 优先级请求队列
pub struct PriorityRequestQueue {
    /// 队列: Priority -> 请求队列
    queues: Arc<tokio::sync::RwLock<BTreeMap<Priority, VecDeque<QueuedRequest>>>>,
    /// 最大队列大小
    max_queue_size: usize,
    /// 当前队列大小
    current_size: Arc<AtomicUsize>,
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
        }
    }

    /// 入队
    pub async fn enqueue(&self, request: QueuedRequest) -> Result<(), AppError> {
        // 使用原子操作确保检查和入队的原子性
        loop {
            let current_size = self.current_size.load(Ordering::Acquire);

            if current_size >= self.max_queue_size {
                return Err(AppError::RateLimitExceeded(
                    "Queue is full, request rejected".to_string(),
                ));
            }

            // 尝试原子递增
            match self.current_size.compare_exchange_weak(
                current_size,
                current_size + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // 成功获取槽位，继续入队
                    break;
                }
                Err(_) => {
                    // 失败，重试
                    continue;
                }
            }
        }

        let mut queues = self.queues.write().await;

        let priority = request.priority;
        let queue = queues.entry(priority).or_insert_with(VecDeque::new);
        queue.push_back(request);

        debug!(
            "Request enqueued, priority={:?}, queue_size={}",
            priority,
            self.current_size.load(Ordering::Relaxed)
        );

        Ok(())
    }

    /// 出队（按优先级）
    pub async fn dequeue(&self) -> Option<QueuedRequest> {
        let mut queues = self.queues.write().await;

        // 按优先级从高到低查找
        for priority in [
            Priority::Critical,
            Priority::High,
            Priority::Normal,
            Priority::Low,
        ] {
            if let Some(queue) = queues.get_mut(&priority) {
                if let Some(request) = queue.pop_front() {
                    let new_size = self.current_size.fetch_sub(1, Ordering::Relaxed) - 1;

                    debug!(
                        "Request dequeued, priority={:?}, queue_size={}",
                        priority, new_size
                    );

                    // 清理空的队列
                    if queue.is_empty() {
                        queues.remove(&priority);
                    }

                    return Some(request);
                }
            }
        }

        None
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
            if let Some(queue) = queues.get(&priority) {
                if !queue.is_empty() {
                    return Some(priority);
                }
            }
        }

        None
    }

    /// 获取队列大小
    pub fn size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
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

        let (tx, _rx) = oneshot::channel();
        let request = QueuedRequest {
            request_id: "test-1".to_string(),
            embed_request: EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tx,
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

        // 添加不同优先级的请求
        for (i, priority) in [
            Priority::Low,
            Priority::Critical,
            Priority::Normal,
            Priority::High,
        ]
        .iter()
        .enumerate()
        {
            let (tx, _rx) = oneshot::channel();
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                embed_request: EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                },
                priority: *priority,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
                response_tx: tx,
            };

            queue.enqueue(request).await.unwrap();
        }

        // 验证出队顺序
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Critical);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::High);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Normal);
        assert_eq!(queue.dequeue().await.unwrap().priority, Priority::Low);
    }

    #[tokio::test]
    async fn test_queue_full() {
        let queue = PriorityRequestQueue::new(2);

        let (tx1, _rx1) = oneshot::channel();
        let request1 = QueuedRequest {
            request_id: "test-1".to_string(),
            embed_request: EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tx1,
        };

        let (tx2, _rx2) = oneshot::channel();
        let request2 = QueuedRequest {
            request_id: "test-2".to_string(),
            embed_request: EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tx2,
        };

        queue.enqueue(request1).await.unwrap();
        queue.enqueue(request2).await.unwrap();

        let (tx3, _rx3) = oneshot::channel();
        let request3 = QueuedRequest {
            request_id: "test-3".to_string(),
            embed_request: EmbedRequest {
                text: "test".to_string(),
                normalize: Some(true),
            },
            priority: Priority::Normal,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::Http {
                ip: "127.0.0.1".to_string(),
            },
            response_tx: tx3,
        };

        let result = queue.enqueue(request3).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_clear() {
        let queue = PriorityRequestQueue::new(100);

        // 添加一些请求
        for i in 0..10 {
            let (tx, _rx) = oneshot::channel();
            let request = QueuedRequest {
                request_id: format!("test-{}", i),
                embed_request: EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                },
                priority: Priority::Normal,
                submitted_at: Instant::now(),
                timeout: Duration::from_secs(30),
                source: RequestSource::Http {
                    ip: "127.0.0.1".to_string(),
                },
                response_tx: tx,
            };

            queue.enqueue(request).await.unwrap();
        }

        assert_eq!(queue.size(), 10);

        queue.clear().await;

        assert_eq!(queue.size(), 0);
    }
}
