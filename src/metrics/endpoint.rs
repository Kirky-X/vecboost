// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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

use crate::i18n;
use crate::rate_limit::is_ip_whitelisted;

/// Prometheus metrics 端点。
///
/// 返回 `text/plain` 格式的 Prometheus 指标。受速率限制 + IP 白名单保护。
pub async fn metrics_endpoint(
    State(app_state): State<crate::VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let ip = addr.ip().to_string();

    // 提前获取 PrometheusCollector（限流指标记录 + 指标导出共用）
    let collector_opt = app_state
        .kit
        .require::<crate::registry::PrometheusCollectorModule>()
        .unwrap_or_else(|e| {
            log::error!(
                "{}",
                i18n::tr_with_args(
                    "log-metrics-collector-unregistered",
                    i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
            None
        });

    if let Some(rejection) =
        enforce_metrics_rate_limit(&app_state, &ip, collector_opt.as_ref()).await
    {
        return rejection;
    }

    let prometheus_collector = match collector_opt.as_ref() {
        Some(c) => c,
        None => {
            return json_error_response(500, &i18n::tr("metrics-collector-missing"));
        }
    };
    snapshot_pipeline_state(&app_state, prometheus_collector);

    let (buffer, content_type) = match encode_metrics_body(&app_state, prometheus_collector) {
        Ok(body) => body,
        Err((status, message)) => return json_error_response(status, &message),
    };

    match Response::builder()
        .status(200)
        .header("Content-Type", content_type)
        .body(Body::from(buffer))
    {
        Ok(resp) => resp.into_response(),
        Err(e) => {
            log::error!(
                "{}",
                i18n::tr_with_args(
                    "log-metrics-response-build-failed",
                    i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
            json_error_response(500, &i18n::tr("metrics-encode-failed"))
        }
    }
}

/// /metrics 限流闸门：RateLimitEnabled=true 时按 IP 白名单边界执行
/// 限流判定并记录决策指标。`Some(resp)` = 拒绝（429）或限流器不可用
/// （fail-closed 500）；`None` = 放行。
async fn enforce_metrics_rate_limit(
    app_state: &crate::VecboostState,
    ip: &str,
    collector: Option<&std::sync::Arc<crate::metrics::prometheus_exporter::PrometheusCollector>>,
) -> Option<axum::response::Response> {
    if !app_state
        .kit
        .config::<crate::registry::RateLimitEnabled>()
        .map(|c| c.0)
        .unwrap_or(false)
    {
        return None;
    }
    let ip_whitelist = app_state
        .kit
        .require::<crate::registry::IpWhitelistModule>()
        .unwrap_or_else(|e| {
            log::error!(
                "{}",
                i18n::tr_with_args(
                    "log-metrics-ip-allowlist-unregistered",
                    i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
            Vec::new()
        });

    if is_ip_whitelisted(ip, &ip_whitelist) {
        return None;
    }
    let context = crate::rate_limit::RequestContext {
        client_ip: Some(ip.to_string()),
        path: "/metrics".to_string(),
        method: "GET".to_string(),
        ..Default::default()
    };
    let allowed = match app_state.kit.require::<crate::registry::RateLimitModule>() {
        Ok(limiter) => limiter.check_rate_limit(&context).await,
        Err(e) => {
            // Fail-closed: RateLimitEnabled=true but module unavailable → deny request
            log::error!(
                "{}",
                i18n::tr_with_args(
                    "log-metrics-ratelimit-unavailable",
                    i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
            return Some(json_error_response(
                500,
                &i18n::tr("metrics-limiter-unavailable"),
            ));
        }
    };
    if let Some(prom) = collector {
        if allowed {
            prom.record_rate_limit_allowed("metrics_ip");
        } else {
            prom.record_rate_limit_denied("metrics_ip");
        }
    }
    if !allowed {
        return Some(json_error_response(429, &i18n::tr("metrics-rate-limited")));
    }
    None
}

/// 拉取时快照 pipeline 状态（队列深度 / 在途请求），热路径零开销、
/// pull 时点才采样。
fn snapshot_pipeline_state(
    app_state: &crate::VecboostState,
    collector: &crate::metrics::prometheus_exporter::PrometheusCollector,
) {
    if let Ok(queue) = app_state
        .kit
        .require::<crate::registry::PipelineQueueModule>()
    {
        let in_flight =
            crate::pipeline::handler::IN_FLIGHT_REQUESTS.load(std::sync::atomic::Ordering::Relaxed);
        collector.set_pipeline_snapshot(queue.size() as i64, in_flight as i64);
    }
}

/// 汇集 collector 自有 registry 与各聚合源（garrison auth 域指标、
/// dbnexus MetricsCollector）为 text/plain 编码体；成功返回
/// `(buffer, Content-Type)`（TextEncoder::format_type 单一事实源），
/// 失败返回 `(status, i18n 文案)`。
fn encode_metrics_body(
    // dbnexus 追加段是唯一消费者；无 db 构建下参数不参与
    #[cfg_attr(not(feature = "db"), allow(unused_variables))] app_state: &crate::VecboostState,
    collector: &crate::metrics::prometheus_exporter::PrometheusCollector,
) -> Result<(Vec<u8>, String), (u16, String)> {
    use prometheus::Encoder;

    let encoder = prometheus::TextEncoder::new();
    #[allow(unused_mut)] // extend 仅在 metrics 聚合 feature 下发生
    let mut metric_families = collector.registry().gather();
    // garrison metrics-prometheus（auth feature）：GarrisonMetrics::new 的 OnceLock
    // 单例把 garrison_* auth 域指标注册在 prometheus default_registry，与上方
    // collector 自有 registry 是两个注册表，此处合并导出（garrison_ 前缀无重名冲突）
    #[cfg(feature = "auth")]
    metric_families.extend(prometheus::default_registry().gather());
    let mut buffer = Vec::new();

    let content_type = encoder.format_type().to_string();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return Err((
            500,
            i18n::tr_with_args(
                "metrics-encode-failed",
                i18n::tr_args(&[("detail", &e.to_string())]),
            )
            .to_string(),
        ));
    }

    // Append dbnexus MetricsCollector output (pool/connection/query metrics)
    #[cfg(feature = "db")]
    if let Ok(Some(db_metrics)) = app_state
        .kit
        .config::<Option<std::sync::Arc<dbnexus::MetricsCollector>>>()
    {
        let db_text = db_metrics.export_prometheus();
        if !db_text.is_empty() {
            buffer.extend_from_slice(b"\n");
            buffer.extend_from_slice(db_text.as_bytes());
        }
    }

    Ok((buffer, content_type))
}

/// Prometheus HTTP 请求指标记录中间件。
///
/// 自动记录每个请求的方法、路径、状态码和延迟到 PrometheusCollector。
/// 路径经过归一化处理，只保留前 4 段以避免 Prometheus label 基数膨胀。
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

/// /metrics 的 4xx/5xx 统一 JSON 错误体(原为纯文本,机器不可读)。
pub fn json_error_response(status: u16, message: &str) -> axum::response::Response {
    let body = serde_json::json!({
        "success": false,
        "error": { "code": "METRICS_ERROR", "message": message }
    });
    let mut resp = axum::response::Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap_or_else(|e| {
            log::error!(
                "{}",
                i18n::tr_with_args(
                    "log-metrics-json-response-build-failed",
                    i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
            axum::response::Response::new(axum::body::Body::from(r#"{"success":false}"#))
        });
    *resp.status_mut() = axum::http::StatusCode::from_u16(status)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    resp
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
        assert_eq!(normalize_metrics_path("/a/b/c/d/e/f"), "/a/b/c/d");
    }

    #[test]
    fn test_normalize_metrics_path_empty_and_root() {
        assert_eq!(normalize_metrics_path(""), "");
        assert_eq!(normalize_metrics_path("/"), "/");
    }
}
