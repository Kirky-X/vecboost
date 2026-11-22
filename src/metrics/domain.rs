// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: time::OffsetDateTime,
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
            timestamp: time::OffsetDateTime::UNIX_EPOCH,
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
            timestamp: time::OffsetDateTime::now_utc(),
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
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: time::OffsetDateTime,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            gpu_utilization_percent: None,
            gpu_memory_percent: None,
            timestamp: time::OffsetDateTime::UNIX_EPOCH,
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
    #[serde(with = "time::serde::rfc3339")]
    pub collected_at: time::OffsetDateTime,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            current: PerformanceMetrics::default(),
            average: PerformanceMetrics::default(),
            min: PerformanceMetrics::default(),
            max: PerformanceMetrics::default(),
            sample_count: 0,
            collected_at: time::OffsetDateTime::now_utc(),
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
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: time::OffsetDateTime,
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
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: time::OffsetDateTime,
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
            timestamp: time::OffsetDateTime::now_utc(),
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
            timestamp: time::OffsetDateTime::now_utc(),
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
    fn test_timestamp_serde_exact_format_pinned_epoch() {
        // 金样本：chrono DateTime<Utc> serde 精确字节格式（chrono→time 迁移前后必须一致）
        let util = ResourceUtilization {
            timestamp: time::OffsetDateTime::UNIX_EPOCH,
            ..Default::default()
        };
        let json = serde_json::to_string(&util).expect("serialize failed");
        assert_eq!(
            json,
            r#"{"cpu_percent":0.0,"memory_percent":0.0,"gpu_utilization_percent":null,"gpu_memory_percent":null,"timestamp":"1970-01-01T00:00:00Z"}"#
        );
    }

    #[test]
    fn test_timestamp_serde_exact_format_pinned_millis() {
        // 金样本：非零小数秒同样钉死（chrono 的 AutoSi 输出毫秒 3 位）
        let ts = time::OffsetDateTime::parse(
            "2023-11-14T22:13:20.123Z",
            &time::format_description::well_known::Rfc3339,
        )
        .expect("parse failed");
        let util = ResourceUtilization {
            timestamp: ts,
            ..Default::default()
        };
        let json = serde_json::to_string(&util).expect("serialize failed");
        assert_eq!(
            json,
            r#"{"cpu_percent":0.0,"memory_percent":0.0,"gpu_utilization_percent":null,"gpu_memory_percent":null,"timestamp":"2023-11-14T22:13:20.123Z"}"#
        );
    }

    #[test]
    fn test_security_event_deserializes_legacy_chrono_rfc3339() {
        // 金样本：chrono 产出的 Z 后缀 RFC3339 字符串必须能被反序列化（迁移前后一致）
        let json = r#"{"timestamp":"1970-01-01T00:00:00Z","event_type":"login_success","user":null,"ip":null,"request_id":null,"user_agent":null,"details":{},"success":true}"#;
        let event: crate::audit::SecurityEvent =
            serde_json::from_str(json).expect("deserialize failed");
        assert_eq!(event.event_type, "login_success");
        assert!(event.success);
    }

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

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.inference_time_ms, 0.0);
        assert_eq!(metrics.tokens_per_second, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 0);
        assert_eq!(metrics.peak_memory_bytes, 0);
        assert_eq!(metrics.batch_size, 0);
        assert_eq!(metrics.sequence_length, 0);
        assert_eq!(metrics.timestamp, time::OffsetDateTime::UNIX_EPOCH);
    }

    #[test]
    fn test_performance_metrics_new_zero_duration() {
        let metrics = PerformanceMetrics::new(Duration::from_secs(0), 100, 1024, 2048, 1, 10);
        assert_eq!(metrics.inference_time_ms, 0.0);
        assert_eq!(metrics.tokens_per_second, 0.0);
        assert_eq!(metrics.memory_usage_bytes, 1024);
        assert_eq!(metrics.peak_memory_bytes, 2048);
    }

    #[test]
    fn test_performance_metrics_throughput_equals_tokens_per_second() {
        let metrics = PerformanceMetrics::new(Duration::from_secs(1), 1000, 0, 0, 1, 100);
        assert!((metrics.throughput() - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_resource_utilization_default() {
        let util = ResourceUtilization::default();
        assert_eq!(util.cpu_percent, 0.0);
        assert_eq!(util.memory_percent, 0.0);
        assert!(util.gpu_utilization_percent.is_none());
        assert!(util.gpu_memory_percent.is_none());
        assert_eq!(util.timestamp, time::OffsetDateTime::UNIX_EPOCH);
    }

    #[test]
    fn test_resource_utilization_with_values() {
        let util = ResourceUtilization {
            cpu_percent: 45.5,
            memory_percent: 60.0,
            gpu_utilization_percent: Some(80.0),
            gpu_memory_percent: Some(50.0),
            timestamp: time::OffsetDateTime::now_utc(),
        };
        assert!((util.cpu_percent - 45.5).abs() < 0.001);
        assert_eq!(util.gpu_utilization_percent, Some(80.0));
    }

    #[test]
    fn test_resource_utilization_serialize() {
        let util = ResourceUtilization::default();
        let json = serde_json::to_string(&util).expect("serialize failed");
        assert!(json.contains("cpu_percent"));
        assert!(json.contains("memory_percent"));
        // 金样本：epoch 时间戳的 serde 字节级格式钉死（RUSTSEC-2026-0543 迁移前后必须一致）
        assert!(
            json.contains("\"timestamp\":\"1970-01-01T00:00:00Z\""),
            "epoch 序列化格式漂移: {json}"
        );
    }

    #[test]
    fn test_metrics_snapshot_default() {
        let snapshot = MetricsSnapshot::default();
        assert_eq!(snapshot.sample_count, 0);
        assert_eq!(snapshot.current.inference_time_ms, 0.0);
        assert_eq!(snapshot.average.inference_time_ms, 0.0);
    }

    #[test]
    fn test_metrics_snapshot_serialize() {
        let snapshot = MetricsSnapshot::default();
        let json = serde_json::to_string(&snapshot).expect("serialize failed");
        assert!(json.contains("sample_count"));
        assert!(json.contains("current"));
        assert!(json.contains("average"));
    }

    #[test]
    fn test_metric_type_equality() {
        assert_eq!(MetricType::InferenceTime, MetricType::InferenceTime);
        assert_ne!(MetricType::InferenceTime, MetricType::Throughput);
        assert_ne!(MetricType::MemoryUsage, MetricType::GpuMemory);
        assert_ne!(MetricType::BatchSize, MetricType::SequenceLength);
    }

    #[test]
    fn test_metric_type_serialize() {
        let json = serde_json::to_string(&MetricType::InferenceTime).expect("serialize");
        assert_eq!(json, "\"InferenceTime\"");
        let json = serde_json::to_string(&MetricType::Throughput).expect("serialize");
        assert_eq!(json, "\"Throughput\"");
    }

    #[test]
    fn test_metric_value_construction() {
        let mv = MetricValue {
            metric_type: MetricType::MemoryUsage,
            value: 1024.0,
            unit: "bytes".to_string(),
            timestamp: time::OffsetDateTime::now_utc(),
        };
        assert_eq!(mv.metric_type, MetricType::MemoryUsage);
        assert_eq!(mv.value, 1024.0);
        assert_eq!(mv.unit, "bytes");
    }

    #[test]
    fn test_metric_value_serialize() {
        let mv = MetricValue {
            metric_type: MetricType::BatchSize,
            value: 32.0,
            unit: "count".to_string(),
            timestamp: time::OffsetDateTime::now_utc(),
        };
        let json = serde_json::to_string(&mv).expect("serialize");
        assert!(json.contains("BatchSize"));
        assert!(json.contains("count"));
    }

    #[test]
    fn test_inference_record_serialize() {
        let record =
            InferenceRecord::success("test-model".to_string(), 100, 512, 50.0, 1024 * 1024);
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("test-model"));
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_inference_record_failure_output_length_zero() {
        let record = InferenceRecord::failure("model".to_string(), 50, "error".to_string());
        assert_eq!(record.output_length, 0);
        assert_eq!(record.inference_time_ms, 0.0);
        assert_eq!(record.memory_bytes, 0);
    }

    #[test]
    fn test_performance_test_config_default() {
        let config = PerformanceTestConfig::default();
        assert_eq!(config.concurrent_requests, 4);
        assert_eq!(config.total_requests, 100);
        assert_eq!(config.warmup_requests, 10);
        assert_eq!(config.min_text_length, 50);
        assert_eq!(config.max_text_length, 500);
        assert!(config.target_qps.is_none());
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_performance_test_config_custom() {
        let config = PerformanceTestConfig {
            concurrent_requests: 8,
            total_requests: 500,
            warmup_requests: 20,
            min_text_length: 10,
            max_text_length: 1000,
            target_qps: Some(100.0),
            timeout_seconds: 120,
        };
        assert_eq!(config.concurrent_requests, 8);
        assert_eq!(config.target_qps, Some(100.0));
    }

    #[test]
    fn test_throughput_result_construction() {
        let result = ThroughputResult {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            total_duration_ms: 10000,
            qps: 9.5,
            error_rate: 0.05,
            total_tokens_processed: 95000,
            tokens_per_second: 9500.0,
        };
        assert_eq!(result.total_requests, 100);
        assert_eq!(result.successful_requests, 95);
        assert_eq!(result.failed_requests, 5);
        assert!((result.qps - 9.5).abs() < 0.001);
    }

    #[test]
    fn test_throughput_result_serialize() {
        let result = ThroughputResult {
            total_requests: 10,
            successful_requests: 10,
            failed_requests: 0,
            total_duration_ms: 1000,
            qps: 10.0,
            error_rate: 0.0,
            total_tokens_processed: 1000,
            tokens_per_second: 1000.0,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("qps"));
        assert!(json.contains("tokens_per_second"));
    }

    #[test]
    fn test_latency_benchmark_result_construction() {
        let result = LatencyBenchmarkResult {
            p50_ms: 10,
            p95_ms: 20,
            p99_ms: 50,
            min_ms: 5,
            max_ms: 100,
            avg_ms: 15.0,
            std_dev_ms: 5.0,
        };
        assert_eq!(result.p50_ms, 10);
        assert_eq!(result.p99_ms, 50);
        assert!((result.avg_ms - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_benchmark_result_serialize() {
        let result = LatencyBenchmarkResult {
            p50_ms: 1,
            p95_ms: 2,
            p99_ms: 3,
            min_ms: 1,
            max_ms: 3,
            avg_ms: 2.0,
            std_dev_ms: 0.5,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("p50_ms"));
        assert!(json.contains("p99_ms"));
        assert!(json.contains("std_dev_ms"));
    }

    #[test]
    fn test_performance_metrics_latency_round_trip() {
        let metrics = PerformanceMetrics::new(Duration::from_millis(250), 100, 0, 0, 1, 50);
        let latency = metrics.latency();
        assert_eq!(latency.as_millis(), 250);
    }

    #[test]
    fn test_inference_record_success_has_no_error() {
        let record = InferenceRecord::success("m".to_string(), 1, 1, 1.0, 1);
        assert!(record.error_message.is_none());
        assert!(record.success);
        assert_eq!(record.output_length, 1);
    }

    #[test]
    fn test_performance_metrics_large_batch() {
        let metrics = PerformanceMetrics::new(
            Duration::from_millis(1000),
            10000,
            1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
            128,
            8192,
        );
        assert!((metrics.inference_time_ms - 1000.0).abs() < 0.1);
        assert_eq!(metrics.batch_size, 128);
        assert_eq!(metrics.sequence_length, 8192);
    }
}
