// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub scale_up_threshold: usize,
    pub scale_down_threshold: usize,
    pub idle_timeout_secs: u64,
    pub scale_check_interval_secs: u64,
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
        }
    }
}

#[derive(Debug, Clone, Serialize)]
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
