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

/// request_id 入环长度上限（字符数）。request_id 吸收客户端 `X-Request-Id`
/// （入站优先），认证调用方可轮换超长 id 挤占 1000 槽位（安全 LOW-2）：
/// 入环处单点收敛到 128 字符，槽位键空间有界。超长 id 收敛为
/// 「119 字符前缀 + `-` + 全文 SHA-256 前 8 hex」，在保留诊断可读性的同时
/// 避免同前缀超长 id 互相覆盖（安全审查 LOW-1）。
pub const REQUEST_ID_MAX_LEN: usize = 128;
const REQUEST_ID_HASH_SUFFIX_LEN: usize = 8;
const REQUEST_ID_PREFIX_LEN: usize = REQUEST_ID_MAX_LEN - REQUEST_ID_HASH_SUFFIX_LEN - 1;

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

/// request_id 收敛（入环与查询共用同一规则）：短 id 原样返回；超长 id 取
/// 119 字符前缀 + `-` + 全文 SHA-256 前 8 hex，总长恰为 128。哈希后缀保证
/// 同前缀超长 id 不碰撞，避免回放错配与审计条目互相覆盖。
fn truncate_request_id(id: &str) -> String {
    if id.chars().count() <= REQUEST_ID_MAX_LEN {
        return id.to_string();
    }
    let digest = crate::utils::hash::sha256_hex(id.as_bytes());
    let prefix: String = id.chars().take(REQUEST_ID_PREFIX_LEN).collect();
    format!("{prefix}-{}", &digest[..REQUEST_ID_HASH_SUFFIX_LEN])
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
    ///
    /// request_id 在入环口截断至 [`REQUEST_ID_MAX_LEN`] 字符（安全 LOW-2 单点
    /// 防御）：回放查询按同规则截断入参，两侧一致。入站 X-Request-Id 摄取处
    /// （sdforge context 中间件）无需同步截断——环层防御已覆盖全部写入源
    /// （审计 sink 与测试注入均经 push），放行层超长头仅占 HTTP 头配额。
    pub fn push(&self, event: TaskExecutionEvent) {
        let mut inner = self.inner.write().expect("task event ring poisoned");
        let id = truncate_request_id(&event.request_id);
        // 同一 request_id 重复写入：先移除旧索引，避免顺序队列残留。
        if inner.events.remove(&id).is_some() {
            inner.order.retain(|x| x != &id);
        }
        let mut event = event;
        event.request_id = id.clone();
        inner.events.insert(id.clone(), event);
        inner.order.push_back(id);
        while inner.order.len() > TASK_EVENT_RING_CAPACITY {
            if let Some(oldest) = inner.order.pop_front() {
                inner.events.remove(&oldest);
            }
        }
    }

    /// 按 request_id 查询；未命中返回 None。入参按入环同规则截断，
    /// 保证「以原超长 id 发起回放」仍能命中已截断存储的条目。
    pub fn get_by_request_id(&self, request_id: &str) -> Option<TaskExecutionEvent> {
        self.inner
            .read()
            .expect("task event ring poisoned")
            .events
            .get(&truncate_request_id(request_id))
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

    /// 安全 LOW-2/LOW-1 回归钉：超长 request_id 入环即收敛到 REQUEST_ID_MAX_LEN
    /// （前缀 + 哈希后缀），事件体内存储值与查询入参两侧同规则——原超长 id 回放
    /// 仍命中；同前缀不同超长 id 不得互相覆盖。
    #[test]
    fn test_push_truncates_oversized_request_id() {
        let ring = TaskEventRing::new();
        let oversized = "x".repeat(REQUEST_ID_MAX_LEN + 37);
        ring.push(event(&oversized));

        let stored = ring
            .get_by_request_id(&oversized)
            .expect("收敛后的 id 必须可回放命中");
        assert_eq!(
            stored.request_id.len(),
            REQUEST_ID_MAX_LEN,
            "入环存储必须收敛到 {REQUEST_ID_MAX_LEN} 字符，got {}",
            stored.request_id.len()
        );
        assert!(
            stored
                .request_id
                .starts_with(&"x".repeat(REQUEST_ID_PREFIX_LEN)),
            "收敛保留前缀"
        );
        assert!(
            stored.request_id.ends_with(&format!(
                "-{}",
                &crate::utils::hash::sha256_hex(oversized.as_bytes())[..REQUEST_ID_HASH_SUFFIX_LEN]
            )),
            "收敛保留全文哈希后缀（防同前缀碰撞）"
        );

        // 碰撞钉：前 128 字符相同、尾部不同的两个超长 id 必须各自命中。
        let other = format!("{}TAIL-A", "x".repeat(REQUEST_ID_MAX_LEN));
        ring.push(event(&other));
        assert_ne!(
            ring.get_by_request_id(&oversized).unwrap().request_id,
            ring.get_by_request_id(&other).unwrap().request_id,
            "同前缀超长 id 必须收敛到不同环键"
        );

        // 边界：恰为上限的 id 不受影响。
        let exact = "y".repeat(REQUEST_ID_MAX_LEN);
        ring.push(event(&exact));
        assert_eq!(
            ring.get_by_request_id(&exact).unwrap().request_id,
            exact,
            "恰好 {REQUEST_ID_MAX_LEN} 字符的 id 必须逐字节保留"
        );
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
