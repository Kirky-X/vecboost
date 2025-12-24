// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::domain::{InferenceRecord, MetricsSnapshot, PerformanceMetrics, ResourceUtilization};

#[derive(Debug, Clone)]
pub struct MetricsCollector {
    inner: Arc<MetricsCollectorInner>,
}

#[derive(Debug)]
struct MetricsCollectorInner {
    global_metrics: RwLock<PerformanceMetrics>,
    model_metrics: RwLock<HashMap<String, ModelMetrics>>,
    inference_records: RwLock<Vec<InferenceRecord>>,
    error_count: AtomicU64,
    request_count: AtomicU64,
    active_requests: AtomicUsize,
    queued_requests: AtomicUsize,
    max_records: usize,
}

#[derive(Debug, Default)]
struct ModelMetrics {
    inference_count: AtomicU64,
    total_inference_time_ms: AtomicU64,
    total_tokens: AtomicU64,
    peak_batch_size: AtomicUsize,
    current_batch_size: AtomicUsize,
    last_inference_time_ms: AtomicU64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsCollectorInner {
                global_metrics: RwLock::new(PerformanceMetrics::default()),
                model_metrics: RwLock::new(HashMap::new()),
                inference_records: RwLock::new(Vec::new()),
                error_count: AtomicU64::new(0),
                request_count: AtomicU64::new(0),
                active_requests: AtomicUsize::new(0),
                queued_requests: AtomicUsize::new(0),
                max_records: 10000,
            }),
        }
    }

    pub fn with_max_records(max_records: usize) -> Self {
        Self {
            inner: Arc::new(MetricsCollectorInner {
                global_metrics: RwLock::new(PerformanceMetrics::default()),
                model_metrics: RwLock::new(HashMap::new()),
                inference_records: RwLock::new(Vec::new()),
                error_count: AtomicU64::new(0),
                request_count: AtomicU64::new(0),
                active_requests: AtomicUsize::new(0),
                queued_requests: AtomicUsize::new(0),
                max_records,
            }),
        }
    }

    pub async fn record_inference_start(&self, model_name: &str, batch_size: usize) {
        self.inner.active_requests.fetch_add(1, Ordering::SeqCst);

        let mut model_metrics = self.inner.model_metrics.write().await;
        let metrics = model_metrics
            .entry(model_name.to_string())
            .or_insert_with(ModelMetrics::default);

        metrics
            .current_batch_size
            .store(batch_size, Ordering::SeqCst);

        let current_peak = metrics.peak_batch_size.load(Ordering::SeqCst);
        if batch_size > current_peak {
            metrics.peak_batch_size.store(batch_size, Ordering::SeqCst);
        }

        self.inner.request_count.fetch_add(1, Ordering::SeqCst);
    }

    pub async fn record_inference_complete(
        &self,
        model_name: &str,
        duration: Duration,
        batch_size: usize,
        tokens_count: usize,
    ) {
        let duration_ms = duration.as_millis() as u64;

        self.inner.active_requests.fetch_sub(1, Ordering::SeqCst);

        let mut global = self.inner.global_metrics.write().await;
        global.inference_count += 1;
        global.total_inference_time_ms += duration_ms;
        global.total_tokens_processed += tokens_count as u64;
        global.current_batch_size = batch_size;
        global.last_inference_time_ms = Some(duration_ms);

        if batch_size > global.peak_batch_size {
            global.peak_batch_size = batch_size;
        }
        drop(global);

        let mut model_metrics = self.inner.model_metrics.write().await;
        if let Some(metrics) = model_metrics.get_mut(model_name) {
            metrics.inference_count.fetch_add(1, Ordering::SeqCst);
            metrics
                .total_inference_time_ms
                .fetch_add(duration_ms, Ordering::SeqCst);
            metrics
                .total_tokens
                .fetch_add(tokens_count as u64, Ordering::SeqCst);
            metrics
                .last_inference_time_ms
                .store(duration_ms, Ordering::SeqCst);
        }
        drop(model_metrics);

        let record = InferenceRecord {
            timestamp: std::time::Instant::now(),
            duration_ms,
            batch_size,
            tokens_count,
            model_name: model_name.to_string(),
            success: true,
        };

        let mut records = self.inner.inference_records.write().await;
        if records.len() >= self.inner.max_records {
            records.remove(0);
        }
        records.push(record);

        debug!(
            model = model_name,
            duration_ms = duration_ms,
            batch_size = batch_size,
            tokens = tokens_count,
            "Inference completed"
        );
    }

    pub async fn record_inference_error(&self, model_name: &str) {
        self.inner.active_requests.fetch_sub(1, Ordering::SeqCst);
        self.inner.error_count.fetch_add(1, Ordering::SeqCst);

        let record = InferenceRecord {
            timestamp: std::time::Instant::now(),
            duration_ms: 0,
            batch_size: 0,
            tokens_count: 0,
            model_name: model_name.to_string(),
            success: false,
        };

        let mut records = self.inner.inference_records.write().await;
        if records.len() >= self.inner.max_records {
            records.remove(0);
        }
        records.push(record);

        warn!(model = model_name, "Inference error recorded");
    }

    pub async fn record_request_queued(&self) {
        self.inner.queued_requests.fetch_add(1, Ordering::SeqCst);
    }

    pub async fn record_request_dequeued(&self) {
        self.inner.queued_requests.fetch_sub(1, Ordering::SeqCst);
    }

    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.inner.global_metrics.read().await.clone()
    }

    pub async fn get_model_metrics(&self, model_name: &str) -> Option<PerformanceMetrics> {
        let model_metrics = self.inner.model_metrics.read().await;
        model_metrics
            .get(model_name)
            .map(|metrics| PerformanceMetrics {
                inference_count: metrics.inference_count.load(Ordering::SeqCst),
                total_inference_time_ms: metrics.total_inference_time_ms.load(Ordering::SeqCst),
                total_tokens_processed: metrics.total_tokens.load(Ordering::SeqCst),
                current_batch_size: metrics.current_batch_size.load(Ordering::SeqCst),
                peak_batch_size: metrics.peak_batch_size.load(Ordering::SeqCst),
                memory_usage_bytes: 0,
                last_inference_time_ms: Some(metrics.last_inference_time_ms.load(Ordering::SeqCst)),
                model_name: model_name.to_string(),
                engine_type: String::new(),
            })
    }

    pub async fn get_all_model_metrics(&self) -> Vec<PerformanceMetrics> {
        let model_metrics = self.inner.model_metrics.read().await;
        model_metrics
            .iter()
            .map(|(name, m)| PerformanceMetrics {
                inference_count: m.inference_count.load(Ordering::SeqCst),
                total_inference_time_ms: m.total_inference_time_ms.load(Ordering::SeqCst),
                total_tokens_processed: m.total_tokens.load(Ordering::SeqCst),
                current_batch_size: m.current_batch_size.load(Ordering::SeqCst),
                peak_batch_size: m.peak_batch_size.load(Ordering::SeqCst),
                memory_usage_bytes: 0,
                last_inference_time_ms: Some(m.last_inference_time_ms.load(Ordering::SeqCst)),
                model_name: name.clone(),
                engine_type: String::new(),
            })
            .collect()
    }

    pub async fn get_resource_utilization(&self) -> ResourceUtilization {
        ResourceUtilization {
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            active_requests: self.inner.active_requests.load(Ordering::SeqCst),
            queued_requests: self.inner.queued_requests.load(Ordering::SeqCst),
        }
    }

    pub async fn get_snapshot(&self) -> MetricsSnapshot {
        let metrics = self.inner.global_metrics.read().await.clone();
        let resource = self.get_resource_utilization().await;

        MetricsSnapshot {
            timestamp: std::time::Instant::now(),
            metrics,
            resource,
        }
    }

    pub async fn get_throughput(&self, window_secs: u64) -> f64 {
        let records = self.inner.inference_records.read().await;
        let now = std::time::Instant::now();

        let window_start = now - Duration::from_secs(window_secs);
        let tokens_in_window: u64 = records
            .iter()
            .filter(|r| r.timestamp >= window_start && r.success)
            .map(|r| r.tokens_count as u64)
            .sum();

        tokens_in_window as f64 / window_secs as f64
    }

    pub async fn get_error_rate(&self) -> f64 {
        let total = self.inner.request_count.load(Ordering::SeqCst);
        if total == 0 {
            return 0.0;
        }
        let errors = self.inner.error_count.load(Ordering::SeqCst);
        errors as f64 / total as f64
    }

    pub async fn get_recent_inference_times(&self, count: usize) -> Vec<u64> {
        let records = self.inner.inference_records.read().await;
        records
            .iter()
            .filter(|r| r.success)
            .rev()
            .take(count)
            .map(|r| r.duration_ms)
            .collect()
    }

    pub async fn get_latency_percentile(&self, percentile: f64, count: usize) -> Option<u64> {
        let times = self.get_recent_inference_times(count).await;
        if times.is_empty() {
            return None;
        }

        let mut sorted = times;
        sorted.sort_unstable();

        let index = ((percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
        Some(sorted[index])
    }

    pub async fn reset(&self) {
        let mut global = self.inner.global_metrics.write().await;
        *global = PerformanceMetrics::default();
        drop(global);

        self.inner.model_metrics.write().await.clear();
        self.inner.inference_records.write().await.clear();
        self.inner.error_count.store(0, Ordering::SeqCst);
        self.inner.request_count.store(0, Ordering::SeqCst);
        self.inner.active_requests.store(0, Ordering::SeqCst);
        self.inner.queued_requests.store(0, Ordering::SeqCst);

        info!("Metrics collector reset");
    }

    pub fn request_count(&self) -> u64 {
        self.inner.request_count.load(Ordering::SeqCst)
    }

    pub fn error_count(&self) -> u64 {
        self.inner.error_count.load(Ordering::SeqCst)
    }

    pub fn active_requests(&self) -> usize {
        self.inner.active_requests.load(Ordering::SeqCst)
    }

    pub fn queued_requests(&self) -> usize {
        self.inner.queued_requests.load(Ordering::SeqCst)
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_record_inference() {
        let collector = MetricsCollector::new();

        collector.record_inference_start("test-model", 4).await;
        collector
            .record_inference_complete("test-model", Duration::from_millis(100), 4, 128)
            .await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.inference_count, 1);
        assert_eq!(metrics.total_inference_time_ms, 100);
        assert_eq!(metrics.total_tokens_processed, 128);
        assert_eq!(metrics.peak_batch_size, 4);
    }

    #[tokio::test]
    async fn test_record_error() {
        let collector = MetricsCollector::new();

        collector.record_inference_start("test-model", 1).await;
        collector.record_inference_error("test-model").await;

        assert_eq!(collector.error_count(), 1);
        assert_eq!(collector.active_requests(), 0);
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let collector = MetricsCollector::new();

        collector
            .record_inference_complete("model1", Duration::from_millis(100), 4, 128)
            .await;
        collector
            .record_inference_complete("model1", Duration::from_millis(100), 4, 128)
            .await;
        collector
            .record_inference_complete("model2", Duration::from_millis(100), 4, 64)
            .await;

        let throughput = collector.get_throughput(60).await;
        assert!(throughput > 0.0);
    }

    #[tokio::test]
    async fn test_error_rate() {
        let collector = MetricsCollector::new();

        for _ in 0..9 {
            collector.record_inference_start("test-model", 1).await;
            collector
                .record_inference_complete("test-model", Duration::from_millis(10), 1, 32)
                .await;
        }

        collector.record_inference_start("test-model", 1).await;
        collector.record_inference_error("test-model").await;

        let error_rate = collector.get_error_rate().await;
        assert!((error_rate - 0.1).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_latency_percentile() {
        let collector = MetricsCollector::new();

        for i in 1..=10 {
            collector
                .record_inference_complete("test-model", Duration::from_millis(i * 10), 1, 32)
                .await;
        }

        let p50 = collector.get_latency_percentile(50.0, 10).await;
        let p90 = collector.get_latency_percentile(90.0, 10).await;

        assert_eq!(p50, Some(50));
        assert_eq!(p90, Some(90));
    }

    #[tokio::test]
    async fn test_model_metrics() {
        let collector = MetricsCollector::new();

        collector.record_inference_start("model-a", 2).await;
        collector
            .record_inference_complete("model-a", Duration::from_millis(50), 2, 64)
            .await;

        collector.record_inference_start("model-a", 2).await;
        collector
            .record_inference_complete("model-a", Duration::from_millis(60), 2, 64)
            .await;

        collector.record_inference_start("model-b", 1).await;
        collector
            .record_inference_complete("model-b", Duration::from_millis(100), 1, 32)
            .await;

        let model_a_metrics = collector.get_model_metrics("model-a").await.unwrap();
        assert_eq!(model_a_metrics.inference_count, 2);
        assert_eq!(model_a_metrics.total_inference_time_ms, 110);

        let model_b_metrics = collector.get_model_metrics("model-b").await.unwrap();
        assert_eq!(model_b_metrics.inference_count, 1);

        let all_metrics = collector.get_all_model_metrics().await;
        assert_eq!(all_metrics.len(), 2);
    }

    #[tokio::test]
    async fn test_reset() {
        let collector = MetricsCollector::new();

        collector.record_inference_start("test-model", 4).await;
        collector
            .record_inference_complete("test-model", Duration::from_millis(100), 4, 128)
            .await;

        assert_eq!(collector.request_count(), 1);

        collector.reset().await;

        let metrics = collector.get_metrics().await;
        assert_eq!(metrics.inference_count, 0);
        assert_eq!(collector.active_requests(), 0);
    }

    #[tokio::test]
    async fn test_request_queue_tracking() {
        let collector = MetricsCollector::new();

        collector.record_request_queued().await;
        collector.record_request_queued().await;
        collector.record_request_dequeued().await;

        assert_eq!(collector.queued_requests(), 1);
    }

    #[tokio::test]
    async fn test_throughput_calculation_empty() {
        let collector = MetricsCollector::new();
        let throughput = collector.get_throughput(60).await;
        assert_eq!(throughput, 0.0);
    }

    #[tokio::test]
    async fn test_latency_percentile_empty() {
        let collector = MetricsCollector::new();
        let p50 = collector.get_latency_percentile(50.0, 10).await;
        assert_eq!(p50, None);
    }
}
