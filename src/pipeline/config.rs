// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct PipelineConfig {
    pub enabled: bool,
    pub queue: QueueConfig,
    pub worker: WorkerConfig,
    #[serde(default)]
    pub priority: PriorityConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue: QueueConfig::default(),
            worker: WorkerConfig::default(),
            priority: PriorityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct QueueConfig {
    pub max_queue_size: usize,
    pub enable_priority: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            enable_priority: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct WorkerConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub scale_up_threshold: usize,
    pub scale_down_threshold: usize,
    pub idle_timeout_secs: u64,
    pub scale_check_interval_secs: u64,
    /// 排空拼批最大请求数（worker 单次从队列取出的最大请求数）
    pub max_batch_size: usize,
    /// 时间窗动态拼批等待窗口（毫秒）。首请求到达后继续等待该时长以聚合更多请求；
    /// 0 表示关闭时间窗、严格还原排空式拼批（kill-switch 内建）。
    #[serde(default = "default_batch_wait_ms")]
    #[schemars(default = "default_batch_wait_ms")]
    pub batch_wait_ms: u64,
}

fn default_batch_wait_ms() -> u64 {
    5
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            scale_up_threshold: 100,
            scale_down_threshold: 10,
            idle_timeout_secs: 60,
            scale_check_interval_secs: 5,
            max_batch_size: 8,
            batch_wait_ms: default_batch_wait_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct PriorityConfig {
    pub base_priority: i32,
    pub timeout_boost_factor: f64,
    #[serde(default)]
    pub user_tier_weights: HashMap<String, f64>,
    #[serde(default)]
    pub source_weights: HashMap<String, f64>,
}

impl<'de> Deserialize<'de> for PriorityConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct InnerPriorityConfig {
            base_priority: Option<i32>,
            timeout_boost_factor: Option<f64>,
            user_tier_weights: Option<HashMap<String, f64>>,
            source_weights: Option<HashMap<String, f64>>,
        }

        let inner = InnerPriorityConfig::deserialize(deserializer)?;

        Ok(Self {
            base_priority: inner.base_priority.unwrap_or(50),
            timeout_boost_factor: inner.timeout_boost_factor.unwrap_or(2.0),
            user_tier_weights: inner.user_tier_weights.unwrap_or_default(),
            source_weights: inner.source_weights.unwrap_or_default(),
        })
    }
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            base_priority: 50,
            timeout_boost_factor: 2.0,
            user_tier_weights: [
                ("free".to_string(), 1.0),
                ("pro".to_string(), 1.5),
                ("enterprise".to_string(), 2.0),
            ]
            .into_iter()
            .collect(),
            source_weights: [
                ("http".to_string(), 1.0),
                ("grpc".to_string(), 1.2),
                ("internal".to_string(), 1.5),
            ]
            .into_iter()
            .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default_batch_wait_ms() {
        let cfg = WorkerConfig::default();
        assert_eq!(cfg.batch_wait_ms, 5, "batch_wait_ms default must be 5");
        assert_eq!(cfg.max_batch_size, 8);
    }

    #[test]
    fn test_worker_config_deserialize_missing_uses_default() {
        let cfg: WorkerConfig = toml::from_str(
            "min_workers = 2\nmax_workers = 4\nscale_up_threshold = 100\n\
             scale_down_threshold = 10\nidle_timeout_secs = 60\n\
             scale_check_interval_secs = 5\nmax_batch_size = 8\n",
        )
        .expect("deserialize without batch_wait_ms must succeed");
        assert_eq!(cfg.batch_wait_ms, 5);
    }

    #[test]
    fn test_worker_config_deserialize_zero_is_legal() {
        let cfg: WorkerConfig = toml::from_str(
            "min_workers = 2\nmax_workers = 4\nscale_up_threshold = 100\n\
             scale_down_threshold = 10\nidle_timeout_secs = 60\n\
             scale_check_interval_secs = 5\nmax_batch_size = 8\nbatch_wait_ms = 0\n",
        )
        .expect("batch_wait_ms=0 must be legal (kill-switch)");
        assert_eq!(cfg.batch_wait_ms, 0);
    }

    #[test]
    fn test_worker_config_schema_contains_batch_wait_ms() {
        let schema = schemars::schema_for!(WorkerConfig);
        let json = serde_json::to_value(&schema).expect("schema serializes");
        let text = serde_json::to_string(&json).expect("schema to string");
        assert!(
            text.contains("batch_wait_ms"),
            "generate_schema() output must contain batch_wait_ms"
        );
    }
}
