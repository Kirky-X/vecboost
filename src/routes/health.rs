// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Health check related routes
//!
//! Provides API endpoints for health checks, metrics collection, etc.

use axum::{
    body::Body, extract::ConnectInfo, extract::State, response::IntoResponse, response::Response,
};
use std::net::SocketAddr;

/// Basic health check handler
///
/// Returns a simple "OK" response for quick service availability check
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is running normally", body = String)
    ),
    operation_id = "health_check"
)]
pub async fn health_check() -> &'static str {
    "OK"
}

/// Prometheus metrics endpoint
///
/// Returns metrics data in Prometheus format
#[utoipa::path(
    get,
    path = "/metrics",
    tag = "health",
    responses(
        (status = 200, description = "Successfully returned Prometheus metrics", body = String)
    ),
    operation_id = "metrics"
)]
pub async fn metrics_endpoint(
    State(app_state): State<crate::AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    use prometheus::Encoder;

    let ip = addr.ip().to_string();

    // Check if rate limiting is enabled and IP is not whitelisted
    #[allow(clippy::collapsible_if)]
    if app_state.rate_limit_enabled {
        let is_whitelisted = app_state.ip_whitelist.iter().any(|whitelist_ip| {
            // Exact match
            if ip == *whitelist_ip {
                return true;
            }
            // CIDR match (basic implementation)
            if let Some(cidr) = whitelist_ip.strip_suffix("/32") {
                if ip == cidr {
                    return true;
                }
            }
            if let Some(cidr) = whitelist_ip.strip_suffix("/128") {
                if ip == cidr {
                    return true;
                }
            }
            // IPv4 /24 subnet check
            if let Some(cidr) = whitelist_ip.strip_suffix("/24") {
                if ip.starts_with(&format!("{}.", cidr)) {
                    return true;
                }
            }
            false
        });

        if !is_whitelisted {
            // Check both global and IP rate limits for metrics endpoint
            if !app_state
                .rate_limiter
                .check_rate_limit(vec![
                    crate::rate_limit::RateLimitDimension::Global,
                    crate::rate_limit::RateLimitDimension::Ip(ip),
                ])
                .await
            {
                return Response::builder()
                    .status(429)
                    .body(Body::from("Rate limit exceeded"))
                    .unwrap()
                    .into_response();
            }
        }
    }

    let prometheus_collector = app_state.prometheus_collector.as_ref().unwrap();
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

// /// 限流状态查询处理器
// ///
// /// 返回指定维度的限流状态信息
// #[utoipa::path(
//     post,
//     path = "/api/v1/rate-limit/status",
//     tag = "health",
//     request_body = RateLimitStatusRequest,
//     responses(
//         (status = 200, description = "成功返回限流状态", body = Vec<RateLimitStatus>),
//         (status = 400, description = "无效的请求参数")
//     ),
//     operation_id = "rate_limit_status"
// )]
// pub async fn rate_limit_status_handler(
//     State(rate_limiter): State<Arc<vecboost::rate_limit::RateLimiter>>,
// //     Json(request): Json<RateLimitStatusRequest>,
// // ) -> Result<Json<Vec<vecboost::rate_limit::RateLimitStatus>>, crate::error::AppError> {
// //     use vecboost::rate_limit::RateLimitDimension;
// //
// //     let dimension = match request.dimension_type.as_str() {
// //         "global" => RateLimitDimension::Global,
// //         "ip" => {
//             let ip = request.dimension_value.ok_or_else(|| {
//                 crate::error::AppError::ValidationError("IP address is required".to_string())
//             })?;
//             RateLimitDimension::Ip(ip)
//         }
//         "user" => {
//             let username = request.dimension_value.ok_or_else(|| {
//                 crate::error::AppError::ValidationError("Username is required".to_string())
//             })?;
//             RateLimitDimension::User(username)
//         }
//         "apikey" => {
//             let api_key = request.dimension_value.ok_or_else(|| {
//                 crate::error::AppError::ValidationError("API key is required".to_string())
//             })?;
//             RateLimitDimension::ApiKey(api_key)
//         }
//         _ => {
//             return Err(crate::error::AppError::ValidationError(
//                 format!("Invalid dimension type: {}. Must be one of: global, ip, user, apikey", request.dimension_type)
//             ));
//         }
//     };
//
//     let status: vecboost::rate_limit::RateLimitStatus = rate_limiter.get_status(dimension).await;
//     Ok(Json(vec![status]))
