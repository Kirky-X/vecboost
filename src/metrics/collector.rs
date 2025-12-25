// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::metrics::domain::{
    InferenceRecord, MetricType, MetricValue, MetricsSnapshot, PerformanceMetrics,
    ResourceUtilization,
};
use crate::monitor::MemoryMonitor;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

#[derive(Debug, Clone, Serialize)]
pub struct CollectionConfig {
    pub max_samples: usize,
    pub collection_interval_ms: u64,
    pub enable_gpu_metrics: bool,
    pub enable_memory_tracking: bool,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            max_samples: 1000,
            collection_interval_ms: 1000,
            enable_gpu_metrics: true,
            enable_memory_tracking: true,
        }
    }
}

#[derive(Clone)]
pub struct MetricsCollector {
    config: Arc<CollectionConfig>,
    memory_monitor: Arc<MemoryMonitor>,
    inference_records: Arc<RwLock<VecDeque<InferenceRecord>>>,
    performance_samples: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    metric_history: Arc<RwLock<VecDeque<MetricValue>>>,
    collection_start: Arc<RwLock<Instant>>,
    total_inferences: Arc<RwLock<u64>>,
    total_tokens: Arc<RwLock<u64>>,
    total_errors: Arc<RwLock<u64>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::with_config(CollectionConfig::default())
    }

    pub fn with_config(config: CollectionConfig) -> Self {
        Self {
            config: Arc::new(config),
            memory_monitor: Arc::new(MemoryMonitor::new()),
            inference_records: Arc::new(RwLock::new(VecDeque::new())),
            performance_samples: Arc::new(RwLock::new(VecDeque::new())),
            metric_history: Arc::new(RwLock::new(VecDeque::new())),
            collection_start: Arc::new(RwLock::new(Instant::now())),
            total_inferences: Arc::new(RwLock::new(0)),
            total_tokens: Arc::new(RwLock::new(0)),
            total_errors: Arc::new(RwLock::new(0)),
        }
    }

    pub fn with_memory_monitor(memory_monitor: Arc<MemoryMonitor>) -> Self {
        let mut collector = Self::new();
        collector.memory_monitor = memory_monitor;
        collector
    }

    pub async fn record_inference(
        &self,
        model_name: &str,
        input_length: usize,
        output_length: usize,
        inference_time: Duration,
        memory_bytes: u64,
    ) {
        let record = InferenceRecord::success(
            model_name.to_string(),
            input_length,
            output_length,
            inference_time.as_secs_f64() * 1000.0,
            memory_bytes,
        );

        self.record_inference_full(record.clone()).await;

        let metrics = PerformanceMetrics::new(
            inference_time,
            input_length + output_length,
            memory_bytes,
            self.memory_monitor.get_peak_memory(),
            1,
            input_length + output_length,
        );

        self.add_performance_sample(metrics).await;

        {
            let mut total = self.total_inferences.write().await;
            *total += 1;
        }

        {
            let mut tokens = self.total_tokens.write().await;
            *tokens += (input_length + output_length) as u64;
        }
    }

    pub async fn record_inference_full(&self, record: InferenceRecord) {
        let mut records = self.inference_records.write().await;

        while records.len() >= self.config.max_samples {
            records.pop_front();
        }

        records.push_back(record.clone());

        if !record.success {
            let mut errors = self.total_errors.write().await;
            *errors += 1;
        }
    }

    pub async fn record_error(&self, model_name: &str, input_length: usize, error: &str) {
        let record =
            InferenceRecord::failure(model_name.to_string(), input_length, error.to_string());

        self.record_inference_full(record).await;

        let mut total = self.total_inferences.write().await;
        *total += 1;
    }

    pub async fn record_inference_complete(
        &self,
        model_name: &str,
        duration: std::time::Duration,
        batch_size: usize,
        token_count: usize,
    ) {
        let record = InferenceRecord::success(
            model_name.to_string(),
            batch_size,
            token_count,
            duration.as_secs_f64() * 1000.0,
            self.memory_monitor.get_peak_memory(),
        );

        self.record_inference_full(record).await;

        {
            let mut total = self.total_inferences.write().await;
            *total += 1;
        }

        {
            let mut tokens = self.total_tokens.write().await;
            *tokens += token_count as u64;
        }
    }

    pub async fn record_inference_error(&self, model_name: &str) {
        let record =
            InferenceRecord::failure(model_name.to_string(), 0, "Inference error".to_string());

        self.record_inference_full(record).await;

        let mut total = self.total_inferences.write().await;
        *total += 1;
    }

    async fn add_performance_sample(&self, metrics: PerformanceMetrics) {
        let mut samples = self.performance_samples.write().await;

        while samples.len() >= self.config.max_samples {
            samples.pop_front();
        }

        samples.push_back(metrics);
    }

    pub async fn collect_resource_utilization(&self) -> ResourceUtilization {
        let memory_stats = self.memory_monitor.refresh().await;

        let cpu_usage = sys_info::loadavg()
            .map(|load| load.one * 10.0)
            .unwrap_or(0.0);

        let memory_usage_percent = if memory_stats.total_bytes > 0 {
            (memory_stats.current_bytes as f64 / memory_stats.total_bytes as f64) * 100.0
        } else {
            0.0
        };

        let gpu_utilization = if self.config.enable_gpu_metrics {
            self.memory_monitor
                .get_gpu_stats()
                .await
                .map(|gpu| gpu.utilization_percent)
        } else {
            None
        };

        let gpu_memory = if self.config.enable_gpu_metrics {
            self.memory_monitor.get_gpu_stats().await.map(|gpu| {
                if gpu.total_bytes > 0 {
                    (gpu.used_bytes as f64 / gpu.total_bytes as f64) * 100.0
                } else {
                    0.0
                }
            })
        } else {
            None
        };

        ResourceUtilization {
            cpu_percent: cpu_usage,
            memory_percent: memory_usage_percent,
            gpu_utilization_percent: gpu_utilization,
            gpu_memory_percent: gpu_memory,
            timestamp: chrono::Utc::now(),
        }
    }

    pub async fn record_metric(&self, metric_type: MetricType, value: f64, unit: &str) {
        let metric = MetricValue {
            metric_type,
            value,
            unit: unit.to_string(),
            timestamp: chrono::Utc::now(),
        };

        let mut history = self.metric_history.write().await;

        while history.len() >= self.config.max_samples {
            history.pop_front();
        }

        history.push_back(metric);
    }

    pub async fn get_snapshot(&self) -> MetricsSnapshot {
        let samples = self.performance_samples.read().await;

        if samples.is_empty() {
            return MetricsSnapshot::default();
        }

        let current = samples.back().cloned().unwrap_or_default();

        let count = samples.len() as f64;
        let total_inference_time: f64 = samples.iter().map(|s| s.inference_time_ms).sum();
        let total_throughput: f64 = samples.iter().map(|s| s.tokens_per_second).sum();
        let total_memory: u64 = samples.iter().map(|s| s.memory_usage_bytes).sum();
        let total_peak_memory: u64 = samples.iter().map(|s| s.peak_memory_bytes).sum();
        let total_batch: usize = samples.iter().map(|s| s.batch_size).sum();
        let total_seq: usize = samples.iter().map(|s| s.sequence_length).sum();

        let avg_inference_time = total_inference_time / count;
        let avg_throughput = total_throughput / count;
        let avg_memory = total_memory / samples.len() as u64;
        let avg_peak_memory = total_peak_memory / samples.len() as u64;
        let avg_batch = total_batch / samples.len();
        let avg_seq = total_seq / samples.len();

        let min = samples
            .iter()
            .min_by(|a, b| {
                a.inference_time_ms
                    .partial_cmp(&b.inference_time_ms)
                    .unwrap()
            })
            .cloned()
            .unwrap_or_default();

        let max = samples
            .iter()
            .max_by(|a, b| {
                a.inference_time_ms
                    .partial_cmp(&b.inference_time_ms)
                    .unwrap()
            })
            .cloned()
            .unwrap_or_default();

        MetricsSnapshot {
            current,
            average: PerformanceMetrics {
                inference_time_ms: avg_inference_time,
                tokens_per_second: avg_throughput,
                memory_usage_bytes: avg_memory,
                peak_memory_bytes: avg_peak_memory,
                batch_size: avg_batch,
                sequence_length: avg_seq,
                timestamp: chrono::Utc::now(),
            },
            min,
            max,
            sample_count: samples.len(),
            collected_at: chrono::Utc::now(),
        }
    }

    pub async fn get_inference_records(&self, limit: Option<usize>) -> Vec<InferenceRecord> {
        let records = self.inference_records.read().await;
        let limit = limit.unwrap_or(100);
        records.iter().rev().take(limit).cloned().collect()
    }

    pub async fn get_metrics_history(&self, metric_type: MetricType) -> Vec<MetricValue> {
        let history = self.metric_history.read().await;
        history
            .iter()
            .filter(|m| m.metric_type == metric_type)
            .cloned()
            .collect()
    }

    pub async fn get_summary(&self) -> MetricsSummary {
        let records = self.inference_records.read().await;
        let samples = self.performance_samples.read().await;

        let successful_inferences = records.iter().filter(|r| r.success).count() as u64;
        let failed_inferences = records.iter().filter(|r| !r.success).count() as u64;

        let avg_latency = if !samples.is_empty() {
            samples.iter().map(|s| s.inference_time_ms).sum::<f64>() / samples.len() as f64
        } else {
            0.0
        };

        let avg_throughput = if !samples.is_empty() {
            samples.iter().map(|s| s.tokens_per_second).sum::<f64>() / samples.len() as f64
        } else {
            0.0
        };

        let total_inferences = *self.total_inferences.read().await;
        let total_tokens = *self.total_tokens.read().await;
        let total_errors = *self.total_errors.read().await;

        MetricsSummary {
            total_inferences,
            successful_inferences,
            failed_inferences,
            total_tokens_processed: total_tokens,
            total_errors,
            average_latency_ms: avg_latency,
            average_throughput_tokens_per_sec: avg_throughput,
            collection_duration_seconds: self.collection_duration().await.as_secs(),
            sample_count: samples.len(),
        }
    }

    pub async fn collection_duration(&self) -> Duration {
        self.collection_start.read().await.elapsed()
    }

    pub async fn reset(&self) {
        let mut records = self.inference_records.write().await;
        records.clear();

        let mut samples = self.performance_samples.write().await;
        samples.clear();

        let mut history = self.metric_history.write().await;
        history.clear();

        let mut start = self.collection_start.write().await;
        *start = Instant::now();

        let mut total = self.total_inferences.write().await;
        *total = 0;

        let mut tokens = self.total_tokens.write().await;
        *tokens = 0;

        let mut errors = self.total_errors.write().await;
        *errors = 0;

        self.memory_monitor.reset_peak();

        info!("Metrics collector reset");
    }

    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSummary {
    pub total_inferences: u64,
    pub successful_inferences: u64,
    pub failed_inferences: u64,
    pub total_tokens_processed: u64,
    pub total_errors: u64,
    pub average_latency_ms: f64,
    pub average_throughput_tokens_per_sec: f64,
    pub collection_duration_seconds: u64,
    pub sample_count: usize,
}

impl MetricsSummary {
    pub fn success_rate(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            self.successful_inferences as f64 / self.total_inferences as f64 * 100.0
        }
    }

    pub fn tokens_per_inference(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            self.total_tokens_processed as f64 / self.total_inferences as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let summary = collector.get_summary().await;

        assert_eq!(summary.total_inferences, 0);
        assert_eq!(summary.sample_count, 0);
    }

    #[tokio::test]
    async fn test_record_inference() {
        let collector = MetricsCollector::new();

        collector
            .record_inference(
                "test-model",
                100,
                512,
                Duration::from_millis(50),
                1024 * 1024,
            )
            .await;

        let summary = collector.get_summary().await;
        assert_eq!(summary.total_inferences, 1);
        assert_eq!(summary.successful_inferences, 1);
        assert_eq!(summary.failed_inferences, 0);
    }

    #[tokio::test]
    async fn test_record_error() {
        let collector = MetricsCollector::new();

        collector
            .record_error("test-model", 100, "Out of memory")
            .await;

        let summary = collector.get_summary().await;
        assert_eq!(summary.total_inferences, 1);
        assert_eq!(summary.total_errors, 1);
        assert_eq!(summary.failed_inferences, 1);
    }

    #[tokio::test]
    async fn test_get_snapshot() {
        let collector = MetricsCollector::new();

        for i in 0..5 {
            collector
                .record_inference(
                    "test-model",
                    100,
                    512,
                    Duration::from_millis(50 + i as u64),
                    1024 * 1024,
                )
                .await;
        }

        let snapshot = collector.get_snapshot().await;

        assert_eq!(snapshot.sample_count, 5);
        assert!(snapshot.min.inference_time_ms <= snapshot.average.inference_time_ms);
        assert!(snapshot.max.inference_time_ms >= snapshot.average.inference_time_ms);
    }

    #[tokio::test]
    async fn test_get_inference_records() {
        let collector = MetricsCollector::new();

        for _ in 0..10 {
            collector
                .record_inference(
                    "test-model",
                    100,
                    512,
                    Duration::from_millis(50),
                    1024 * 1024,
                )
                .await;
        }

        let records = collector.get_inference_records(Some(5)).await;
        assert_eq!(records.len(), 5);
    }

    #[tokio::test]
    async fn test_record_metric() {
        let collector = MetricsCollector::new();

        collector
            .record_metric(MetricType::Throughput, 1000.0, "tokens/s")
            .await;

        let history = collector.get_metrics_history(MetricType::Throughput).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].value, 1000.0);
    }

    #[tokio::test]
    async fn test_reset() {
        let collector = MetricsCollector::new();

        collector
            .record_inference(
                "test-model",
                100,
                512,
                Duration::from_millis(50),
                1024 * 1024,
            )
            .await;

        collector.reset().await;

        let summary = collector.get_summary().await;
        assert_eq!(summary.total_inferences, 0);
        assert_eq!(summary.sample_count, 0);
    }

    #[tokio::test]
    async fn test_success_rate() {
        let collector = MetricsCollector::new();

        collector
            .record_inference(
                "test-model",
                100,
                512,
                Duration::from_millis(50),
                1024 * 1024,
            )
            .await;

        collector.record_error("test-model", 100, "Error").await;

        let summary = collector.get_summary().await;
        assert!((summary.success_rate() - 50.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_collection_duration() {
        let collector = MetricsCollector::new();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let duration = collector.collection_duration().await;
        assert!(duration.as_millis() >= 100);
    }
}
