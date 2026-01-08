// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};

/// 流水线配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// 是否启用
    pub enabled: bool,
    /// 队列配置
    pub queue: QueueConfig,
    /// Worker 配置
    pub worker: WorkerConfig,
    /// 优先级配置
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

/// 队列配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// 最大队列大小
    pub max_queue_size: usize,
    /// 是否启用优先级
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

/// Worker 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// 最小 Worker 数量
    pub min_workers: usize,
    /// 最大 Worker 数量
    pub max_workers: usize,
    /// 扩容阈值
    pub scale_up_threshold: usize,
    /// 缩容阈值
    pub scale_down_threshold: usize,
    /// 空闲超时（秒）
    pub idle_timeout_secs: u64,
    /// 检查间隔（秒）
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

/// 优先级配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityConfig {
    /// 基础优先级
    pub base_priority: i32,
    /// 超时提升因子
    pub timeout_boost_factor: f64,
    /// 用户等级权重
    pub user_tier_weights: Vec<(String, f64)>,
    /// 来源权重
    pub source_weights: Vec<(String, f64)>,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            base_priority: 50,
            timeout_boost_factor: 2.0,
            user_tier_weights: vec![
                ("free".to_string(), 1.0),
                ("pro".to_string(), 1.5),
                ("enterprise".to_string(), 2.0),
            ],
            source_weights: vec![
                ("http".to_string(), 1.0),
                ("grpc".to_string(), 1.2),
                ("internal".to_string(), 1.5),
            ],
        }
    }
}
