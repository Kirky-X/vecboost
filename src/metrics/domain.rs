// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub inference_count: u64,
    pub total_inference_time_ms: u64,
    pub total_tokens_processed: u64,
    pub current_batch_size: usize,
    pub peak_batch_size: usize,
    pub memory_usage_bytes: u64,
    pub last_inference_time_ms: Option<u64>,
    pub model_name: String,
    pub engine_type: String,
}

impl PerformanceMetrics {
    pub fn new(model_name: String, engine_type: String) -> Self {
        Self {
            model_name,
            engine_type,
            ..Default::default()
        }
    }

    pub fn throughput_tokens_per_sec(&self) -> f64 {
        if self.total_inference_time_ms == 0 {
            0.0
        } else {
            self.total_tokens_processed as f64 / (self.total_inference_time_ms as f64 / 1000.0)
        }
    }

    pub fn average_inference_time_ms(&self) -> f64 {
        if self.inference_count == 0 {
            0.0
        } else {
            self.total_inference_time_ms as f64 / self.inference_count as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceRecord {
    pub timestamp: std::time::Instant,
    pub duration_ms: u64,
    pub batch_size: usize,
    pub tokens_count: usize,
    pub model_name: String,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub active_requests: usize,
    pub queued_requests: usize,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            active_requests: 0,
            queued_requests: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: std::time::Instant,
    pub metrics: PerformanceMetrics,
    pub resource: ResourceUtilization,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    InferenceLatency,
    Throughput,
    MemoryUsage,
    RequestCount,
    ErrorRate,
}

#[derive(Debug, Clone)]
pub struct MetricValue {
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp: std::time::Instant,
    pub labels: HashMap<String, String>,
}
