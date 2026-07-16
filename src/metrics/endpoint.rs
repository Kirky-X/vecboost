// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Prometheus metrics 端点 — `/metrics`。
//!
//! forge 宏只支持 JSON 响应，Prometheus 导出需要 text/plain，
//! 因此保留手写 handler 作为 HTTP 路由的例外。

use std::net::SocketAddr;

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::Response,
    response::IntoResponse,
};

use crate::rate_limit::is_ip_whitelisted;

/// Prometheus metrics 端点。
///
/// 返回 `text/plain` 格式的 Prometheus 指标。受速率限制 + IP 白名单保护。
pub async fn metrics_endpoint(
    State(app_state): State<crate::VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    use prometheus::Encoder;

    let ip = addr.ip().to_string();

    if app_state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered")
    {
        let ip_whitelist = app_state
            .kit
            .require::<crate::module_registry::IpWhitelistModule>()
            .expect("IpWhitelistModule not registered");

        if !is_ip_whitelisted(&ip, &ip_whitelist) {
            let allowed = app_state
                .kit
                .require::<crate::module_registry::RateLimitModule>()
                .expect("RateLimitModule not registered")
                .check_rate_limit(vec![
                    crate::rate_limit::RateLimitDimension::Global,
                    crate::rate_limit::RateLimitDimension::Ip(ip),
                ])
                .await;
            if !allowed {
                return Response::builder()
                    .status(429)
                    .body(Body::from("Rate limit exceeded"))
                    .unwrap()
                    .into_response();
            }
        }
    }

    let collector_opt = app_state
        .kit
        .require::<crate::module_registry::PrometheusCollectorModule>()
        .expect("PrometheusCollectorModule not registered");
    let prometheus_collector = collector_opt
        .as_ref()
        .expect("PrometheusCollector not configured");
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus_collector.registry().gather();
    let mut buffer = Vec::new();

    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return Response::builder()
            .status(500)
            .body(format!("Failed to encode metrics: {}", e))
            .unwrap()
            .into_response();
    }

    Response::builder()
        .status(200)
        .header("Content-Type", encoder.format_type())
        .body(Body::from(buffer))
        .unwrap()
        .into_response()
}
