// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 任务执行事件回放环（近端诊断，单副本视图）。
//!
//! 容量固定 1000，FIFO 逐出最旧条目；按 request_id O(1) 查询。
//! 实现用 `RwLock<VecDeque>` + `HashMap` 索引——语义与环形缓冲等价，
//! 但查询是 O(1) 而非线性扫描；不引入 distributed-redis 依赖。

use crate::audit::TaskExecutionEvent;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

/// 回放环容量（事件条数）。
pub const TASK_EVENT_RING_CAPACITY: usize = 1000;

pub struct TaskEventRing {
    inner: RwLock<RingInner>,
}

#[derive(Default)]
struct RingInner {
    order: VecDeque<String>,
    events: HashMap<String, TaskExecutionEvent>,
}

impl Default for TaskEventRing {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskEventRing {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(RingInner {
                order: VecDeque::with_capacity(TASK_EVENT_RING_CAPACITY),
                events: HashMap::with_capacity(TASK_EVENT_RING_CAPACITY),
            }),
        }
    }

    /// 写入一条事件，超容量时逐出最旧条目（FIFO）。
    pub fn push(&self, event: TaskExecutionEvent) {
        let mut inner = self.inner.write().expect("task event ring poisoned");
        let id = event.request_id.clone();
        // 同一 request_id 重复写入：先移除旧索引，避免顺序队列残留。
        if inner.events.remove(&id).is_some() {
            inner.order.retain(|x| x != &id);
        }
        inner.events.insert(id.clone(), event);
        inner.order.push_back(id);
        while inner.order.len() > TASK_EVENT_RING_CAPACITY {
            if let Some(oldest) = inner.order.pop_front() {
                inner.events.remove(&oldest);
            }
        }
    }

    /// 按 request_id 查询；未命中返回 None。
    pub fn get_by_request_id(&self, request_id: &str) -> Option<TaskExecutionEvent> {
        self.inner
            .read()
            .expect("task event ring poisoned")
            .events
            .get(request_id)
            .cloned()
    }

    /// 当前条目数（诊断用）。
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("task event ring poisoned")
            .events
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::ModelTask;
    use std::sync::Arc;

    fn event(id: &str) -> TaskExecutionEvent {
        TaskExecutionEvent {
            task_type: ModelTask::Decision,
            model_id: "m".to_string(),
            model_sha256_prefix: None,
            input_fingerprint: "fp".to_string(),
            output_confidence: None,
            low_confidence_flagged: false,
            latency_ms: 0.0,
            request_id: id.to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_push_and_get() {
        let ring = TaskEventRing::new();
        ring.push(event("a"));
        assert_eq!(ring.get_by_request_id("a").unwrap().model_id, "m");
        assert!(ring.get_by_request_id("missing").is_none());
    }

    #[test]
    fn test_fifo_eviction_at_capacity() {
        let ring = TaskEventRing::new();
        for i in 0..TASK_EVENT_RING_CAPACITY + 5 {
            ring.push(event(&format!("req-{i}")));
        }
        // 最旧 5 条被逐出
        assert!(ring.get_by_request_id("req-0").is_none());
        assert!(ring.get_by_request_id("req-4").is_none());
        assert!(ring.get_by_request_id("req-5").is_some());
        assert_eq!(ring.len(), TASK_EVENT_RING_CAPACITY);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_push_no_loss_no_panic() {
        let ring = Arc::new(TaskEventRing::new());
        let mut handles = Vec::new();
        for t in 0..8u32 {
            let ring = Arc::clone(&ring);
            handles.push(tokio::spawn(async move {
                for i in 0..100u32 {
                    ring.push(event(&format!("t{t}-{i}")));
                }
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(ring.len(), 800, "8×100 条须全部保留（未超容量）");
    }
}
