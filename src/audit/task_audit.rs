// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 任务审计上下文：把 logger + 回放环 + 低置信阈值聚成一处，
//! 由 main.rs 在服务装配时构造，经 builder 注入 decision/translate/detect 服务。
//! 统一构造 `TaskExecutionEvent`，杜绝多处字段拼装漂移。

use super::logger::AuditLogger;
use super::task_event::TaskExecutionEvent;
use crate::config::model::ModelTask;
use crate::registry::TaskEventRing;
use std::sync::Arc;

pub struct TaskAudit {
    pub logger: Option<Arc<AuditLogger>>,
    pub ring: Arc<TaskEventRing>,
    /// 模型标识（注入时确定）。
    pub model_id: String,
    /// 模型权重 sha256 前 12 hex（未设哈希时 None）。
    pub model_sha256_prefix: Option<String>,
    /// 决策低置信阈值；低于该值标记 low_confidence_flagged。
    pub decision_low_confidence: Option<f64>,
}

impl TaskAudit {
    /// 构造一条任务事件并同时落审计 sink + 回放环。
    pub fn record(
        &self,
        task_type: ModelTask,
        input_text: &str,
        output_confidence: Option<f64>,
        latency_ms: f64,
        request_id: &str,
    ) {
        let low_confidence_flagged = match (output_confidence, self.decision_low_confidence) {
            (Some(conf), Some(threshold)) => conf < threshold,
            _ => false,
        };
        let event = TaskExecutionEvent {
            task_type,
            model_id: self.model_id.clone(),
            model_sha256_prefix: self.model_sha256_prefix.clone(),
            input_fingerprint: crate::utils::hash::sha256_hex(input_text.as_bytes()),
            output_confidence,
            low_confidence_flagged,
            latency_ms,
            request_id: request_id.to_string(),
            timestamp: String::new(),
        }
        .with_now();
        // ring 与审计 sink 同处写入，避免两处漂移。
        self.ring.push(event.clone());
        if let Some(logger) = &self.logger {
            logger.log_task_execution(event);
        }
    }
}

/// 从 kit 读取装配时注入的 TaskAudit（未注入时 None = 无审计）。
pub fn from_kit<S>(kit: &trait_kit::AsyncKit<S>) -> Option<std::sync::Arc<TaskAudit>> {
    kit.config::<std::sync::Arc<TaskAudit>>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audit(threshold: Option<f64>) -> TaskAudit {
        TaskAudit {
            logger: None,
            ring: Arc::new(TaskEventRing::new()),
            model_id: "m".to_string(),
            model_sha256_prefix: None,
            decision_low_confidence: threshold,
        }
    }

    #[test]
    fn test_low_confidence_flag_two_branches() {
        let a = audit(Some(0.5));
        // 0.3 < 0.5 → flagged；0.8 不 flagged。
        a.record(ModelTask::Decision, "low input", Some(0.3), 1.0, "req-low");
        a.record(
            ModelTask::Decision,
            "high input",
            Some(0.8),
            1.0,
            "req-high",
        );
        assert!(
            a.ring
                .get_by_request_id("req-low")
                .unwrap()
                .low_confidence_flagged
        );
        assert!(
            !a.ring
                .get_by_request_id("req-high")
                .unwrap()
                .low_confidence_flagged
        );
    }

    #[test]
    fn test_no_threshold_never_flags() {
        let a = audit(None);
        a.record(ModelTask::Decision, "x", Some(0.0), 1.0, "r");
        assert!(
            !a.ring
                .get_by_request_id("r")
                .unwrap()
                .low_confidence_flagged
        );
    }
}
