// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Prometheus metrics 端点 — `/metrics`。
//!
//! forge 宏只支持 JSON 响应，Prometheus 导出需要 text/plain，
//! 因此保留手写 handler 作为 HTTP 路由的例外。

use std::net::SocketAddr;
#[cfg(feature = "db")]
use std::sync::Arc;

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

    // 提前获取 PrometheusCollector（限流指标记录 + 指标导出共用）
    let collector_opt = app_state
        .kit
        .require::<crate::registry::PrometheusCollectorModule>()
        .unwrap_or_else(|e| {
            log::error!("PrometheusCollectorModule not registered: {}", e);
            None
        });

    if app_state
        .kit
        .config::<crate::registry::RateLimitEnabled>()
        .map(|c| c.0)
        .unwrap_or(false)
    {
        let ip_whitelist = app_state
            .kit
            .require::<crate::registry::IpWhitelistModule>()
            .unwrap_or_else(|e| {
                log::error!("IpWhitelistModule not registered: {}", e);
                Vec::new()
            });

        if !is_ip_whitelisted(&ip, &ip_whitelist) {
            let context = crate::rate_limit::RequestContext {
                client_ip: Some(ip.clone()),
                path: "/metrics".to_string(),
                method: "GET".to_string(),
                ..Default::default()
            };
            let allowed = match app_state
                .kit
                .require::<crate::registry::RateLimitModule>()
            {
                Ok(limiter) => limiter.check_rate_limit(&context).await,
                Err(e) => {
                    // Fail-closed: RateLimitEnabled=true but module unavailable → deny request
                    log::error!(
                        "RateLimitModule not available while RateLimitEnabled=true: {}. Denying request.",
                        e
                    );
                    return Response::builder()
                        .status(500)
                        .body(Body::from("Rate limiter unavailable"))
                        .unwrap()
                        .into_response();
                }
            };
            // 记录限流决策指标
            if let Some(prom) = collector_opt.as_ref() {
                if allowed {
                    prom.record_rate_limit_allowed("metrics_ip");
                } else {
                    prom.record_rate_limit_denied("metrics_ip");
                }
            }
            if !allowed {
                return Response::builder()
                    .status(429)
                    .body(Body::from("Rate limit exceeded"))
                    .unwrap()
                    .into_response();
            }
        }
    }

    let prometheus_collector = match collector_opt.as_ref() {
        Some(c) => c,
        None => {
            return Response::builder()
                .status(500)
                .body(Body::from("PrometheusCollector not configured"))
                .unwrap_or_else(|e| {
                    log::error!("Failed to build error response: {}", e);
                    Response::new(Body::from("Internal error"))
                })
                .into_response();
        }
    };
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

    // T044: Append dbnexus MetricsCollector output (pool/connection/query metrics)
    #[cfg(feature = "db")]
    if let Ok(Some(db_metrics)) = app_state
        .kit
        .config::<Option<Arc<dbnexus::MetricsCollector>>>()
    {
        let db_text = db_metrics.export_prometheus();
        if !db_text.is_empty() {
            buffer.extend_from_slice(b"\n");
            buffer.extend_from_slice(db_text.as_bytes());
        }
    }

    Response::builder()
        .status(200)
        .header("Content-Type", encoder.format_type())
        .body(Body::from(buffer))
        .unwrap()
        .into_response()
}

/// Prometheus HTTP 请求指标记录中间件。
///
/// 自动记录每个请求的方法、路径、状态码和延迟到 PrometheusCollector。
/// 路径经过归一化处理，只保留前 3 段以避免 Prometheus label 基数膨胀。
#[cfg(feature = "http")]
pub async fn metrics_middleware(
    State(app_state): State<crate::VecboostState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = request.method().to_string();
    let path = normalize_metrics_path(request.uri().path());

    let prom_collector = app_state
        .kit
        .require::<crate::registry::PrometheusCollectorModule>()
        .ok()
        .and_then(|opt| opt);

    let _timer = prom_collector
        .as_ref()
        .map(|prom| prom.start_http_request_timer(&method, &path));

    let response = next.run(request).await;

    if let Some(prom) = prom_collector.as_ref() {
        prom.record_http_request(&method, &path, response.status().as_u16());
    }

    response
}

/// 归一化 HTTP 路径，只保留前 4 段以控制 Prometheus label 基数。
///
/// 例如：
/// - `/api/1/embed` → `/api/1/embed`
/// - `/api/1/embed/batch` → `/api/1/embed/batch`
/// - `/api/1/model/switch` → `/api/1/model/switch`
/// - `/health` → `/health`
#[cfg(feature = "http")]
fn normalize_metrics_path(path: &str) -> String {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let prefix = if path.starts_with('/') { "/" } else { "" };
    if segments.len() <= 4 {
        format!("{}{}", prefix, segments.join("/"))
    } else {
        format!("{}{}", prefix, segments[..4].join("/"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_metrics_path_short_paths_unchanged() {
        assert_eq!(normalize_metrics_path("/health"), "/health");
        assert_eq!(normalize_metrics_path("/metrics"), "/metrics");
        assert_eq!(normalize_metrics_path("/api/1/embed"), "/api/1/embed");
        assert_eq!(normalize_metrics_path("/v1/embeddings"), "/v1/embeddings");
    }

    #[test]
    fn test_normalize_metrics_path_four_segments_preserved() {
        assert_eq!(
            normalize_metrics_path("/api/1/embed/batch"),
            "/api/1/embed/batch"
        );
        assert_eq!(
            normalize_metrics_path("/api/1/model/switch"),
            "/api/1/model/switch"
        );
        assert_eq!(
            normalize_metrics_path("/api/1/auth/login"),
            "/api/1/auth/login"
        );
    }

    #[test]
    fn test_normalize_metrics_path_truncates_beyond_four() {
        assert_eq!(
            normalize_metrics_path("/api/1/model/switch/extra"),
            "/api/1/model/switch"
        );
        assert_eq!(
            normalize_metrics_path("/a/b/c/d/e/f"),
            "/a/b/c/d"
        );
    }
}
