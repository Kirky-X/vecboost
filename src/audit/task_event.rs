// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 任务执行审计事件（`[task_audit]`）。
//!
//! 与 `SecurityEventType` 安全事件类型分离：任务执行是运营/追溯语义，
//! 同走 audit async sink 但不混入安全事件封闭枚举。字段刻意**不含原始
//! 输入文本**——仅 sha256 指纹，避免审计日志成为二次泄露面。

use crate::config::model::ModelTask;
use serde::Serialize;
use time::format_description::well_known::Rfc3339;

/// 单条任务执行审计事件。
#[derive(Debug, Clone, Serialize)]
pub struct TaskExecutionEvent {
    /// 任务类型（decision / translate / detect）。
    pub task_type: ModelTask,
    /// 模型标识（model_id / model_repo）。
    pub model_id: String,
    /// 模型权重 SHA-256 前 12 hex；未设哈希时 None。
    pub model_sha256_prefix: Option<String>,
    /// 规范化输入文本的 sha256 十六进制串（不落原文）。
    pub input_fingerprint: String,
    /// 输出置信度（decision 分数 / detect confidence；translate 为 None）。
    pub output_confidence: Option<f64>,
    /// 置信度低于 `[task_audit] decision_low_confidence` 阈值时 true。
    pub low_confidence_flagged: bool,
    /// 处理耗时（毫秒）。
    pub latency_ms: f64,
    /// 请求 ID（`sdforge::context::current()`；无上下文回退 "req-unknown"）。
    pub request_id: String,
    /// 事件时间戳（RFC3339，可解析；不用 Instant——不可序列化）。
    pub timestamp: String,
}

impl TaskExecutionEvent {
    /// 以当前 UTC 时间构造事件（RFC3339 格式）。
    pub fn with_now(mut self) -> Self {
        self.timestamp = time::OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .expect("RFC3339 format failed");
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TaskExecutionEvent {
        TaskExecutionEvent {
            task_type: ModelTask::Decision,
            model_id: "m1".to_string(),
            model_sha256_prefix: Some("abcdef123456".to_string()),
            input_fingerprint: "ba7816bf8f01".to_string(),
            output_confidence: Some(0.87),
            low_confidence_flagged: false,
            latency_ms: 12.5,
            request_id: "req-1".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    const FIELDS: &[&str] = &[
        "task_type",
        "model_id",
        "model_sha256_prefix",
        "input_fingerprint",
        "output_confidence",
        "low_confidence_flagged",
        "latency_ms",
        "request_id",
        "timestamp",
    ];

    #[test]
    fn test_task_execution_event_serializes_all_fields() {
        let json = serde_json::to_value(sample()).unwrap();
        for field in FIELDS {
            assert!(json.get(field).is_some(), "缺失字段: {field}");
        }
        assert_eq!(json["task_type"], "decision");
        assert_eq!(json["model_id"], "m1");
        assert_eq!(json["request_id"], "req-1");
    }

    #[test]
    fn test_task_execution_event_no_raw_input_field() {
        let json = serde_json::to_value(sample()).unwrap();
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("input"));
        assert!(!obj.contains_key("text"));
        assert!(!obj.contains_key("raw_input"));
        // 仅 fingerprints/prefix，不含原文。
        assert!(obj.contains_key("input_fingerprint"));
        assert!(obj.contains_key("model_sha256_prefix"));
    }

    #[test]
    fn test_timestamp_is_rfc3339_parseable() {
        let event = sample().with_now();
        let parsed = time::OffsetDateTime::parse(&event.timestamp, &Rfc3339);
        assert!(
            parsed.is_ok(),
            "timestamp 须为 RFC3339: {}",
            event.timestamp
        );
    }
}
