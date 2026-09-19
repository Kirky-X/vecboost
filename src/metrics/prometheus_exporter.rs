// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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

    // pipeline 队列深度 / 在途请求(pull 时快照)
    pipeline_queue_depth: prometheus::IntGauge,
    pipeline_in_flight: prometheus::IntGauge,

    // HTTP 请求计数器
    http_requests_total: CounterVec,

    // HTTP 请求延迟直方图
    http_request_duration_seconds: HistogramVec,

    // 活跃连接数
    active_connections: GaugeVec,

    // 批处理大小
    batch_size: HistogramVec,

    // 时间窗拼批指标：批次大小与等待时长
    vecboost_batch_size: HistogramVec,
    vecboost_batch_wait_seconds: HistogramVec,
    // 批内去重率：滚动 gauge
    vecboost_inbatch_dedup_ratio: GaugeVec,
    // 引擎分阶段延迟：tokenize/inference/pool
    vecboost_stage_seconds: HistogramVec,

    // 缓存命中率
    cache_hits: CounterVec,
    cache_misses: CounterVec,

    // 限流决策计数器（limiteron 集成）
    rate_limit_allowed: CounterVec,
    rate_limit_denied: CounterVec,
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

        // pipeline 队列深度与在途请求(pull 时快照,热路径零开销)
        let pipeline_queue_depth = prometheus::register_int_gauge_with_registry!(
            "vecboost_pipeline_queue_depth",
            "Number of requests waiting in the pipeline priority queue",
            registry.clone()
        )?;
        let pipeline_in_flight = prometheus::register_int_gauge_with_registry!(
            "vecboost_pipeline_in_flight_requests",
            "Number of requests currently inside the pipeline (dequeued, awaiting completion)",
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

        // 时间窗拼批批次大小与等待时长（标签 operation 完整）
        let vecboost_batch_size = register_histogram_vec_with_registry!(
            "vecboost_batch_size",
            "Worker time-window batch size",
            &["operation"],
            vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            registry.clone()
        )?;
        let vecboost_batch_wait_seconds = register_histogram_vec_with_registry!(
            "vecboost_batch_wait_seconds",
            "Worker time-window batch wait duration in seconds",
            &["operation"],
            vec![0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            registry.clone()
        )?;
        // 批内去重率滚动 gauge
        let vecboost_inbatch_dedup_ratio = register_gauge_vec_with_registry!(
            "vecboost_inbatch_dedup_ratio",
            "In-batch dedup ratio (n_unique savings)",
            &["operation"],
            registry.clone()
        )?;
        // 引擎分阶段延迟（stage 标签枚举固定为 tokenize|inference|pool）。
        let vecboost_stage_seconds = register_histogram_vec_with_registry!(
            "vecboost_stage_seconds",
            "Engine stage latency in seconds",
            &["stage"],
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
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

        // 限流允许通过的请求数
        let rate_limit_allowed = register_counter_vec_with_registry!(
            "rate_limit_allowed_total",
            "Total number of rate-limited requests allowed",
            &["dimension"],
            registry.clone()
        )?;

        // 限流拒绝的请求数
        let rate_limit_denied = register_counter_vec_with_registry!(
            "rate_limit_denied_total",
            "Total number of rate-limited requests denied",
            &["dimension"],
            registry.clone()
        )?;

        Ok(Self {
            registry,
            pipeline_queue_depth,
            pipeline_in_flight,
            http_requests_total,
            http_request_duration_seconds,
            active_connections,
            batch_size,
            vecboost_batch_size,
            vecboost_batch_wait_seconds,
            vecboost_inbatch_dedup_ratio,
            vecboost_stage_seconds,
            cache_hits,
            cache_misses,
            rate_limit_allowed,
            rate_limit_denied,
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

    /// 在 assemble_batch 返回处埋点。`batch_wait_ms=0` 时调用方传 wait_secs=0。
    pub fn observe_batch(&self, operation: &str, size: usize, wait_secs: f64) {
        self.vecboost_batch_size
            .with_label_values(&[operation])
            .observe(size as f64);
        self.vecboost_batch_wait_seconds
            .with_label_values(&[operation])
            .observe(wait_secs);
    }

    /// 更新批内去重率滚动 gauge。
    pub fn set_dedup_ratio(&self, operation: &str, ratio: f64) {
        self.vecboost_inbatch_dedup_ratio
            .with_label_values(&[operation])
            .set(ratio);
    }

    /// 汇出引擎分阶段延迟快照（drain 语义，来自 StageStats::take）。
    /// 快照聚合了多次调用，直方图按"每次调用均值"观测（秒/计数）；
    /// 计数为 0 的阶段跳过，避免向空桶写入无意义样本。
    pub fn record_stage_snapshot(&self, snapshot: &crate::engine::StageSnapshot) {
        for (stage, total_secs, count) in snapshot.parts() {
            if count > 0 {
                self.vecboost_stage_seconds
                    .with_label_values(&[stage.as_str()])
                    .observe(total_secs / count as f64);
            }
        }
    }

    /// 记录缓存命中
    pub fn record_cache_hit(&self, cache_type: &str) {
        self.cache_hits.with_label_values(&[cache_type]).inc();
    }

    /// 记录缓存未命中
    pub fn record_cache_miss(&self, cache_type: &str) {
        self.cache_misses.with_label_values(&[cache_type]).inc();
    }

    /// 记录限流决策：允许通过
    pub fn record_rate_limit_allowed(&self, dimension: &str) {
        self.rate_limit_allowed
            .with_label_values(&[dimension])
            .inc();
    }

    /// 记录限流决策：拒绝
    pub fn record_rate_limit_denied(&self, dimension: &str) {
        self.rate_limit_denied.with_label_values(&[dimension]).inc();
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

    /// 拉取时更新 pipeline 快照 gauge(队列深度 / 在途请求)
    pub fn set_pipeline_snapshot(&self, queue_depth: i64, in_flight: i64) {
        self.pipeline_queue_depth.set(queue_depth);
        self.pipeline_in_flight.set(in_flight);
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

/// 全局 collector 单例锚点：worker/服务热路径无 kit 状态可拿，
/// 通过进程级 OnceLock 桥接（main 启动时 set 一次）。
static GLOBAL_COLLECTOR: std::sync::OnceLock<std::sync::Arc<PrometheusCollector>> =
    std::sync::OnceLock::new();

/// 设置全局 collector（进程生命周期内一次；重复设置忽略并返回 false）。
pub fn set_global_collector(collector: std::sync::Arc<PrometheusCollector>) -> bool {
    GLOBAL_COLLECTOR.set(collector).is_ok()
}

/// 全局 collector 访问点（未设置时 None —— library 模式/单测环境零指标副作用）。
pub fn global_collector() -> Option<&'static std::sync::Arc<PrometheusCollector>> {
    GLOBAL_COLLECTOR.get()
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
        // 仅常驻的 pipeline 快照 gauge(初值 0)存在于未记录状态
        for family in &families {
            let name = family.name();
            assert!(
                name.starts_with("vecboost_pipeline_"),
                "unexpected always-on family: {name}"
            );
        }
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
            .map(|m| m.get_counter().value())
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
            .map(|m| m.get_gauge().value())
            .sum();
        assert_eq!(http_value, 42.0, "http connections gauge should be 42");

        let grpc_value: f64 = gauges
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "grpc"))
            .map(|m| m.get_gauge().value())
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
            gauge.get_gauge().value(),
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
    fn test_record_stage_snapshot_observes_all_stages_and_skips_empty() {
        use crate::engine::{Stage, StageSnapshot};
        let collector = PrometheusCollector::new().unwrap();
        // 三阶段均有观测：均值 = total/count
        let mut snap = StageSnapshot::default();
        snap.nanos[Stage::Tokenize as usize] = 2_000_000; // 2ms
        snap.counts[Stage::Tokenize as usize] = 2;
        snap.nanos[Stage::Inference as usize] = 5_000_000; // 5ms
        snap.counts[Stage::Inference as usize] = 1;
        snap.nanos[Stage::Pooling as usize] = 1_000_000; // 1ms
        snap.counts[Stage::Pooling as usize] = 1;
        collector.record_stage_snapshot(&snap);

        let families = collector.registry().gather();
        let stage_metric = families
            .iter()
            .find(|m| m.name() == "vecboost_stage_seconds")
            .expect("vecboost_stage_seconds should be registered");
        let labels_seen: Vec<String> = stage_metric
            .get_metric()
            .iter()
            .flat_map(|m| {
                m.get_label()
                    .iter()
                    .map(|l| l.value().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        for expected in ["tokenize", "inference", "pool"] {
            assert!(
                labels_seen.iter().any(|l| l == expected),
                "stage 标签 {expected} 应存在"
            );
        }
        assert_eq!(labels_seen.len(), 3, "stage 标签枚举应恰为三值");

        // 计数为 0 的阶段必须跳过（空快照不产生标签序列；
        // prometheus 对无子序列的 HistogramVec 直接不输出 family）
        let collector2 = PrometheusCollector::new().unwrap();
        collector2.record_stage_snapshot(&StageSnapshot::default());
        let families2 = collector2.registry().gather();
        assert!(
            families2
                .iter()
                .all(|m| m.name() != "vecboost_stage_seconds"),
            "空快照不应产生 vecboost_stage_seconds 输出"
        );
    }

    #[test]
    fn test_observe_batch_records_size_and_wait_with_labels() {
        let collector = PrometheusCollector::new().unwrap();
        collector.observe_batch("worker", 3, 0.005);
        collector.observe_batch("worker", 1, 0.0);

        let families = collector.registry().gather();
        let size_metric = families
            .iter()
            .find(|m| m.name() == "vecboost_batch_size")
            .expect("vecboost_batch_size should be registered");
        let size_hist = &size_metric.get_metric()[0];
        assert_eq!(size_hist.get_histogram().get_sample_count(), 2);
        let labels: Vec<_> = size_hist
            .get_label()
            .iter()
            .map(|l| (l.name().to_string(), l.value().to_string()))
            .collect();
        assert!(
            labels
                .iter()
                .any(|(k, v)| k == "operation" && v == "worker")
        );

        let wait_metric = families
            .iter()
            .find(|m| m.name() == "vecboost_batch_wait_seconds")
            .expect("vecboost_batch_wait_seconds should be registered");
        let wait_hist = &wait_metric.get_metric()[0];
        assert_eq!(wait_hist.get_histogram().get_sample_count(), 2);
        // batch_wait_ms=0 时 wait 观测值为 0 不 panic
        collector.observe_batch("worker-zero", 1, 0.0);
        let families2 = collector.registry().gather();
        assert!(
            families2
                .iter()
                .any(|m| m.name() == "vecboost_batch_wait_seconds")
        );
    }

    #[test]
    fn test_set_dedup_ratio_boundaries() {
        let collector = PrometheusCollector::new().unwrap();
        collector.set_dedup_ratio("embed", 0.0);
        collector.set_dedup_ratio("embed", 0.75);
        let families = collector.registry().gather();
        let metric = families
            .iter()
            .find(|m| m.name() == "vecboost_inbatch_dedup_ratio")
            .expect("vecboost_inbatch_dedup_ratio should be registered");
        let val: f64 = metric.get_metric()[0].get_gauge().value();
        assert!((val - 0.75).abs() < 1e-9);
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
            .map(|m| m.get_counter().value())
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
            .map(|m| m.get_counter().value())
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

    #[test]
    fn test_record_rate_limit_allowed_increments_counter() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_rate_limit_allowed("embedding");
        collector.record_rate_limit_allowed("embedding");
        collector.record_rate_limit_allowed("rerank");

        let families = collector.registry().gather();
        let metric = families
            .iter()
            .find(|m| m.name() == "rate_limit_allowed_total")
            .expect("rate_limit_allowed_total should be registered");

        let counters = metric.get_metric();
        assert_eq!(counters.len(), 2, "should have 2 dimensions");

        let embedding_val: f64 = counters
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "embedding"))
            .map(|m| m.get_counter().value())
            .sum();
        assert_eq!(embedding_val, 2.0);
    }

    #[test]
    fn test_record_rate_limit_denied_increments_counter() {
        let collector = PrometheusCollector::new().unwrap();
        collector.record_rate_limit_denied("embedding");
        collector.record_rate_limit_denied("embedding");
        collector.record_rate_limit_denied("embedding");

        let families = collector.registry().gather();
        let metric = families
            .iter()
            .find(|m| m.name() == "rate_limit_denied_total")
            .expect("rate_limit_denied_total should be registered");

        let counters = metric.get_metric();
        let val: f64 = counters
            .iter()
            .filter(|m| m.get_label().iter().any(|l| l.value() == "embedding"))
            .map(|m| m.get_counter().value())
            .sum();
        assert_eq!(val, 3.0);
    }

    #[cfg(feature = "db")]
    #[test]
    fn test_dbnexus_metrics_append_pattern() {
        // Verify the integration pattern — dbnexus MetricsCollector output
        // can be appended to PrometheusCollector text output
        use prometheus::Encoder;
        let collector = PrometheusCollector::new().unwrap();
        collector.record_http_request("GET", "/health", 200);

        let encoder = prometheus::TextEncoder::new();
        let families = collector.registry().gather();
        let mut buffer = Vec::new();
        encoder.encode(&families, &mut buffer).unwrap();

        // Simulate appending dbnexus metrics (same pattern as endpoint.rs)
        let db_metrics = dbnexus::MetricsCollector::new();
        let db_text = db_metrics.export_prometheus();
        assert!(
            db_text.contains("dbnexus_"),
            "dbnexus metrics should contain dbnexus_ prefixed metrics"
        );
        buffer.extend_from_slice(b"\n");
        buffer.extend_from_slice(db_text.as_bytes());

        let output = String::from_utf8(buffer).unwrap();
        assert!(
            output.contains("http_requests_total"),
            "should have vecboost metrics"
        );
        assert!(
            output.contains("dbnexus_uptime"),
            "should have dbnexus metrics"
        );
    }
}
