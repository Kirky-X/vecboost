// Copyright (c) 2025 VecBoost
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
