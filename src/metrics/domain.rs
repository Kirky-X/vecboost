// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f64,
    pub tokens_per_second: f64,
    pub memory_usage_bytes: u64,
    pub peak_memory_bytes: u64,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            inference_time_ms: 0.0,
            tokens_per_second: 0.0,
            memory_usage_bytes: 0,
            peak_memory_bytes: 0,
            batch_size: 0,
            sequence_length: 0,
            timestamp: chrono::DateTime::from_timestamp_nanos(0),
        }
    }
}

impl PerformanceMetrics {
    pub fn new(
        inference_time: Duration,
        token_count: usize,
        memory_bytes: u64,
        peak_memory_bytes: u64,
        batch_size: usize,
        seq_length: usize,
    ) -> Self {
        let inference_time_ms = inference_time.as_secs_f64() * 1000.0;
        let tokens_per_second = if inference_time.as_secs_f64() > 0.0 {
            token_count as f64 / inference_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            inference_time_ms,
            tokens_per_second,
            memory_usage_bytes: memory_bytes,
            peak_memory_bytes,
            batch_size,
            sequence_length: seq_length,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn throughput(&self) -> f64 {
        self.tokens_per_second
    }

    pub fn latency(&self) -> Duration {
        Duration::from_secs_f64(self.inference_time_ms / 1000.0)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceUtilization {
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub gpu_utilization_percent: Option<f64>,
    pub gpu_memory_percent: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            gpu_utilization_percent: None,
            gpu_memory_percent: None,
            timestamp: chrono::DateTime::from_timestamp_nanos(0),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub current: PerformanceMetrics,
    pub average: PerformanceMetrics,
    pub min: PerformanceMetrics,
    pub max: PerformanceMetrics,
    pub sample_count: usize,
    pub collected_at: chrono::DateTime<chrono::Utc>,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            current: PerformanceMetrics::default(),
            average: PerformanceMetrics::default(),
            min: PerformanceMetrics::default(),
            max: PerformanceMetrics::default(),
            sample_count: 0,
            collected_at: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum MetricType {
    InferenceTime,
    Throughput,
    MemoryUsage,
    GpuMemory,
    BatchSize,
    SequenceLength,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricValue {
    pub metric_type: MetricType,
    pub value: f64,
    pub unit: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceRecord {
    pub model_name: String,
    pub input_length: usize,
    pub output_length: usize,
    pub inference_time_ms: f64,
    pub memory_bytes: u64,
    pub success: bool,
    pub error_message: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl InferenceRecord {
    pub fn success(
        model_name: String,
        input_length: usize,
        output_length: usize,
        inference_time_ms: f64,
        memory_bytes: u64,
    ) -> Self {
        Self {
            model_name,
            input_length,
            output_length,
            inference_time_ms,
            memory_bytes,
            success: true,
            error_message: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn failure(model_name: String, input_length: usize, error_message: String) -> Self {
        Self {
            model_name,
            input_length,
            output_length: 0,
            inference_time_ms: 0.0,
            memory_bytes: 0,
            success: false,
            error_message: Some(error_message),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceTestConfig {
    pub concurrent_requests: usize,
    pub total_requests: usize,
    pub warmup_requests: usize,
    pub min_text_length: usize,
    pub max_text_length: usize,
    pub target_qps: Option<f64>,
    pub timeout_seconds: u64,
}

impl Default for PerformanceTestConfig {
    fn default() -> Self {
        Self {
            concurrent_requests: 4,
            total_requests: 100,
            warmup_requests: 10,
            min_text_length: 50,
            max_text_length: 500,
            target_qps: None,
            timeout_seconds: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ThroughputResult {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub total_duration_ms: u64,
    pub qps: f64,
    pub error_rate: f64,
    pub total_tokens_processed: u64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyBenchmarkResult {
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: f64,
    pub std_dev_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_metrics_new() {
        let metrics = PerformanceMetrics::new(
            Duration::from_millis(100),
            512,
            1024 * 1024,
            2 * 1024 * 1024,
            1,
            512,
        );

        assert_eq!(metrics.inference_time_ms, 100.0);
        assert!((metrics.tokens_per_second - 5120.0).abs() < 0.1);
        assert_eq!(metrics.memory_usage_bytes, 1024 * 1024);
        assert_eq!(metrics.peak_memory_bytes, 2 * 1024 * 1024);
        assert_eq!(metrics.batch_size, 1);
        assert_eq!(metrics.sequence_length, 512);
    }

    #[test]
    fn test_performance_metrics_throughput() {
        let metrics = PerformanceMetrics::new(Duration::from_secs(2), 1000, 0, 0, 1, 512);

        assert!((metrics.throughput() - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_performance_metrics_latency() {
        let metrics = PerformanceMetrics::new(Duration::from_millis(150), 512, 0, 0, 1, 512);

        let latency = metrics.latency();
        assert!((latency.as_secs_f64() - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_inference_record_success() {
        let record =
            InferenceRecord::success("test-model".to_string(), 100, 512, 50.0, 1024 * 1024);

        assert!(record.success);
        assert!(record.error_message.is_none());
        assert_eq!(record.model_name, "test-model");
        assert_eq!(record.input_length, 100);
    }

    #[test]
    fn test_inference_record_failure() {
        let record =
            InferenceRecord::failure("test-model".to_string(), 100, "Out of memory".to_string());

        assert!(!record.success);
        assert!(record.error_message.is_some());
        assert_eq!(record.error_message.unwrap(), "Out of memory");
    }
}
