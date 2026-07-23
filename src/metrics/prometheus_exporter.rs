// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Prometheus 指标收集模块
//!
//! 提供核心指标的收集和暴露功能

use prometheus::{
    CounterVec, GaugeVec, HistogramVec, Registry, register_counter_vec_with_registry,
    register_gauge_vec_with_registry, register_histogram_vec_with_registry,
};
use std::sync::Arc;

/// Prometheus 指标收集器
pub struct PrometheusCollector {
    registry: Arc<Registry>,

    // HTTP 请求计数器
    http_requests_total: CounterVec,

    // HTTP 请求延迟直方图
    http_request_duration_seconds: HistogramVec,

    // 活跃连接数
    active_connections: GaugeVec,

    // 批处理大小
    batch_size: HistogramVec,

    // 缓存命中率
    cache_hits: CounterVec,
    cache_misses: CounterVec,
}

impl PrometheusCollector {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Arc::new(Registry::new());

        // HTTP 请求总数（按端点、方法、状态码）
        let http_requests_total = register_counter_vec_with_registry!(
            "http_requests_total",
            "Total number of HTTP requests",
            &["method", "endpoint", "status"],
            registry.clone()
        )?;

        // HTTP 请求延迟（按端点、方法）
        let http_request_duration_seconds = register_histogram_vec_with_registry!(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            &["method", "endpoint"],
            vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            ],
            registry.clone()
        )?;

        // 活跃连接数
        let active_connections = register_gauge_vec_with_registry!(
            "active_connections",
            "Number of active connections",
            &["type"],
            registry.clone()
        )?;

        // 批处理大小
        let batch_size = register_histogram_vec_with_registry!(
            "batch_size",
            "Batch processing size",
            &["operation"],
            vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            registry.clone()
        )?;

        // 缓存命中
        let cache_hits = register_counter_vec_with_registry!(
            "cache_hits_total",
            "Total number of cache hits",
            &["cache_type"],
            registry.clone()
        )?;

        // 缓存未命中
        let cache_misses = register_counter_vec_with_registry!(
            "cache_misses_total",
            "Total number of cache misses",
            &["cache_type"],
            registry.clone()
        )?;

        Ok(Self {
            registry,
            http_requests_total,
            http_request_duration_seconds,
            active_connections,
            batch_size,
            cache_hits,
            cache_misses,
        })
    }

    /// 记录 HTTP 请求
    pub fn record_http_request(&self, method: &str, endpoint: &str, status_code: u16) {
        self.http_requests_total
            .with_label_values(&[method, endpoint, &status_code.to_string()])
            .inc();
    }

    /// 开始计时 HTTP 请求
    pub fn start_http_request_timer(
        &self,
        method: &str,
        endpoint: &str,
    ) -> prometheus::HistogramTimer {
        self.http_request_duration_seconds
            .with_label_values(&[method, endpoint])
            .start_timer()
    }

    /// 更新活跃连接数
    pub fn update_active_connections(&self, connection_type: &str, count: i64) {
        self.active_connections
            .with_label_values(&[connection_type])
            .set(count as f64);
    }

    /// 记录批处理大小
    pub fn record_batch_size(&self, operation: &str, size: f64) {
        self.batch_size
            .with_label_values(&[operation])
            .observe(size);
    }

    /// 记录缓存命中
    pub fn record_cache_hit(&self, cache_type: &str) {
        self.cache_hits.with_label_values(&[cache_type]).inc();
    }

    /// 记录缓存未命中
    pub fn record_cache_miss(&self, cache_type: &str) {
        self.cache_misses.with_label_values(&[cache_type]).inc();
    }

    #[allow(clippy::unnecessary_cast)]
    /// 获取缓存命中率
    pub fn get_cache_hit_rate(&self, cache_type: &str) -> f64 {
        let hits = self.cache_hits.with_label_values(&[cache_type]).get();
        let misses = self.cache_misses.with_label_values(&[cache_type]).get();

        let hits_f64 = hits as f64;
        let misses_f64 = misses as f64;
        let total = hits_f64 + misses_f64;
        if total == 0.0 { 0.0 } else { hits_f64 / total }
    }

    /// 获取注册表（用于暴露指标）
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }
}

impl Default for PrometheusCollector {
    fn default() -> Self {
        Self::new().expect("Failed to create PrometheusCollector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_collector() {
        let collector = PrometheusCollector::new();
        assert!(collector.is_ok());
    }

    #[test]
    fn test_default_creates_collector() {
        let collector = PrometheusCollector::default();
        let registry = collector.registry();
        let families = registry.gather();
        assert!(
            families.is_empty(),
            "gather should return empty when no metrics have been recorded yet"
        );
    }

    #[test]
    fn test_registry_returns_shared_arc() {
        let collector = PrometheusCollector::new().unwrap();
        let registry1 = collector.registry();
        let registry2 = collector.registry();
        assert!(
            Arc::ptr_eq(&registry1, &registry2),
            "registry() should return clones of the same Arc"
        );
    }

    #[test]
    fn test_record_http_request_increments_counter() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_http_request("GET", "/embed", 200);
        collector.record_http_request("GET", "/embed", 200);
        collector.record_http_request("POST", "/embed", 500);

        let families = collector.registry().gather();
        let http_metric = families
            .iter()
            .find(|m| m.name() == "http_requests_total")
            .expect("http_requests_total should be registered");

        let counters = http_metric.get_metric();
        assert_eq!(counters.len(), 2, "should have 2 unique label combinations");

        let get_200: f64 = counters
            .iter()
            .filter(|m| {
                let labels = m.get_label();
                labels.iter().any(|l| l.value() == "GET")
                    && labels.iter().any(|l| l.value() == "200")
            })
            .map(|m| m.get_counter().get_value())
            .sum();
        assert_eq!(get_200, 2.0, "GET 200 counter should be 2");
    }

    #[test]
    fn test_start_http_request_timer_records_observation() {
        let collector = PrometheusCollector::new().unwrap();
        let timer = collector.start_http_request_timer("POST", "/embed");
        timer.observe_duration();

        let families = collector.registry().gather();
        let duration_metric = families
            .iter()
            .find(|m| m.name() == "http_request_duration_seconds")
            .expect("http_request_duration_seconds should be registered");

        let samples = duration_metric.get_metric();
        assert!(!samples.is_empty(), "should have at least one observation");
        assert!(
            samples[0].get_histogram().get_sample_count() >= 1,
            "histogram sample count should be at least 1"
        );
    }

    #[test]
    fn test_update_active_connections_sets_gauge() {
        let collector = PrometheusCollector::new().unwrap();
        collector.update_active_connections("http", 42);
        collector.update_active_connections("grpc", 7);

        let families = collector.registry().gather();
        let gauge_metric = families
            .iter()
            .find(|m| m.name() == "active_connections")
            .expect("active_connections should be registered");

        let gauges = gauge_metric.get_metric();
        assert_eq!(gauges.len(), 2, "should have 2 connection types");

        let http_value: f64 = gauges
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "http"))
            .map(|m| m.get_gauge().get_value())
            .sum();
        assert_eq!(http_value, 42.0, "http connections gauge should be 42");

        let grpc_value: f64 = gauges
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "grpc"))
            .map(|m| m.get_gauge().get_value())
            .sum();
        assert_eq!(grpc_value, 7.0, "grpc connections gauge should be 7");
    }

    #[test]
    fn test_update_active_connections_overwrites_previous_value() {
        let collector = PrometheusCollector::new().unwrap();
        collector.update_active_connections("http", 10);
        collector.update_active_connections("http", 25);

        let families = collector.registry().gather();
        let gauge_metric = families
            .iter()
            .find(|m| m.name() == "active_connections")
            .unwrap();
        let gauge = &gauge_metric.get_metric()[0];
        assert_eq!(
            gauge.get_gauge().get_value(),
            25.0,
            "gauge should reflect the latest set value"
        );
    }

    #[test]
    fn test_record_batch_size_observes_histogram() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_batch_size("embed", 8.0);
        collector.record_batch_size("embed", 16.0);
        collector.record_batch_size("search", 4.0);

        let families = collector.registry().gather();
        let batch_metric = families
            .iter()
            .find(|m| m.name() == "batch_size")
            .expect("batch_size should be registered");

        let histograms = batch_metric.get_metric();
        assert_eq!(histograms.len(), 2, "should have 2 operations");

        let embed_count: u64 = histograms
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "embed"))
            .map(|m| m.get_histogram().get_sample_count())
            .sum();
        assert_eq!(embed_count, 2, "embed operation should have 2 observations");
    }

    #[test]
    fn test_record_cache_hit_and_miss() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_cache_hit("embedding");
        collector.record_cache_hit("embedding");
        collector.record_cache_miss("embedding");

        let families = collector.registry().gather();

        let hits_metric = families
            .iter()
            .find(|m| m.name() == "cache_hits_total")
            .expect("cache_hits_total should be registered");
        let hits_value: f64 = hits_metric
            .get_metric()
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "embedding"))
            .map(|m| m.get_counter().get_value())
            .sum();
        assert_eq!(hits_value, 2.0, "should have 2 cache hits");

        let misses_metric = families
            .iter()
            .find(|m| m.name() == "cache_misses_total")
            .expect("cache_misses_total should be registered");
        let misses_value: f64 = misses_metric
            .get_metric()
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "embedding"))
            .map(|m| m.get_counter().get_value())
            .sum();
        assert_eq!(misses_value, 1.0, "should have 1 cache miss");
    }

    #[test]
    fn test_get_cache_hit_rate_empty() {
        let collector = PrometheusCollector::new().unwrap();
        let rate = collector.get_cache_hit_rate("embedding");
        assert_eq!(rate, 0.0, "hit rate should be 0 when no hits or misses");
    }

    #[test]
    fn test_get_cache_hit_rate_all_hits() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_cache_hit("embedding");
        collector.record_cache_hit("embedding");
        let rate = collector.get_cache_hit_rate("embedding");
        assert_eq!(rate, 1.0, "hit rate should be 1.0 when all hits");
    }

    #[test]
    fn test_get_cache_hit_rate_all_misses() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_cache_miss("embedding");
        collector.record_cache_miss("embedding");
        let rate = collector.get_cache_hit_rate("embedding");
        assert_eq!(rate, 0.0, "hit rate should be 0.0 when all misses");
    }

    #[test]
    fn test_get_cache_hit_rate_mixed() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_cache_hit("embedding");
        collector.record_cache_hit("embedding");
        collector.record_cache_miss("embedding");
        collector.record_cache_miss("embedding");
        let rate = collector.get_cache_hit_rate("embedding");
        assert_eq!(rate, 0.5, "hit rate should be 0.5 for 2 hits / 4 total");
    }

    #[test]
    fn test_get_cache_hit_rate_independent_per_type() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_cache_hit("type_a");
        collector.record_cache_miss("type_a");
        collector.record_cache_hit("type_b");
        collector.record_cache_hit("type_b");

        let rate_a = collector.get_cache_hit_rate("type_a");
        let rate_b = collector.get_cache_hit_rate("type_b");
        assert_eq!(rate_a, 0.5, "type_a should have 0.5 hit rate");
        assert_eq!(rate_b, 1.0, "type_b should have 1.0 hit rate");
    }

    #[test]
    fn test_metrics_export_format() {
        use prometheus::Encoder;
        let collector = PrometheusCollector::new().unwrap();
        collector.record_http_request("GET", "/health", 200);
        collector.update_active_connections("http", 1);

        let encoder = prometheus::TextEncoder::new();
        let families = collector.registry().gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&families, &mut buffer)
            .expect("encode failed");
        let output = String::from_utf8(buffer).expect("output should be valid UTF-8");

        assert!(
            output.contains("http_requests_total"),
            "output should contain http_requests_total"
        );
        assert!(
            output.contains("active_connections"),
            "output should contain active_connections"
        );
        assert!(
            output.contains("# HELP"),
            "output should contain HELP lines"
        );
        assert!(
            output.contains("# TYPE"),
            "output should contain TYPE lines"
        );
    }
}
