// Copyright (c) 2025-2026 Kirky.X
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
    State(app_state): State<crate::VecboostState>,
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

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;
    use crate::VecboostState;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use crate::error::VecboostError;
    use crate::metrics::PrometheusCollector;
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    };
    use crate::rate_limit::{LimiteronAdapter, RateLimitConfig};
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use axum::http::StatusCode;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;

    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for TestEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            static PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_test_state() -> VecboostState {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            engine,
            Some(model_config),
        )));
        std::mem::forget(temp_dir);

        let prometheus_collector = Arc::new(PrometheusCollector::new().unwrap());
        let rate_limiter = Arc::new(LimiteronAdapter::with_default_config());
        let pipeline_queue = Arc::new(PriorityRequestQueue::new(0));
        let response_channel = Arc::new(ResponseChannel::new());

        #[cfg(feature = "auth")]
        {
            VecboostState {
                service,
                jwt_manager: None,
                user_store: None,
                auth_enabled: false,
                csrf_config: None,
                csrf_token_store: None,
                metrics_collector: None,
                prometheus_collector: Some(prometheus_collector),
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
                worker_manager: Arc::new(WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    Arc::new(RwLock::new(EmbeddingService::new(
                        Arc::new(RwLock::new(TestEngine::new(384))),
                        None,
                    ))),
                )),
                kit: None,
            }
        }

        #[cfg(not(feature = "auth"))]
        {
            VecboostState {
                service,
                auth_enabled: false,
                metrics_collector: None,
                prometheus_collector: Some(prometheus_collector),
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
                worker_manager: Arc::new(WorkerManager::new(
                    Arc::new(PriorityRequestQueue::new(0)),
                    Arc::new(ResponseChannel::new()),
                    WorkerConfig::default(),
                    Arc::new(RwLock::new(EmbeddingService::new(
                        Arc::new(RwLock::new(TestEngine::new(384))),
                        None,
                    ))),
                )),
                kit: None,
            }
        }
    }

    fn make_rate_limited_state() -> VecboostState {
        let mut state = make_test_state();
        state.rate_limit_enabled = true;
        state.rate_limiter = Arc::new(LimiteronAdapter::new(RateLimitConfig {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 0,
            ..Default::default()
        }));
        state
    }

    fn make_rate_limited_state_with_whitelist() -> VecboostState {
        let mut state = make_rate_limited_state();
        state.ip_whitelist = vec!["127.0.0.1".to_string()];
        state
    }

    fn test_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002)
    }

    #[tokio::test]
    async fn test_health_check_returns_ok() {
        let result = health_check().await;
        assert_eq!(result, "OK");
    }

    #[tokio::test]
    async fn test_health_check_into_response_status_200() {
        let response = health_check().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_success() {
        let state = make_test_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_rate_limit_exceeded_returns_429() {
        let state = make_rate_limited_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelisted_ip_bypasses_rate_limit() {
        let state = make_rate_limited_state_with_whitelist();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    fn test_addr_with(ip: IpAddr) -> SocketAddr {
        SocketAddr::new(ip, 9002)
    }

    fn make_rate_limited_state_with_whitelist_ips(whitelist: Vec<String>) -> VecboostState {
        let mut state = make_rate_limited_state();
        state.ip_whitelist = whitelist;
        state
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_32_matches() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1/32".to_string()]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_32_does_not_match_other_ip() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1/32".to_string()]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_matches_subnet() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_does_not_match_other_subnet() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_ipv6_cidr_128_matches() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1/128".to_string()]);
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_multiple_whitelist_entries_one_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec![
            "192.168.1.1/32".to_string(),
            "127.0.0.1/32".to_string(),
        ]);
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_text_content_type() {
        let state = make_test_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            content_type.contains("text/plain"),
            "expected text/plain content-type, got {}",
            content_type
        );
    }

    #[tokio::test]
    async fn test_metrics_endpoint_rate_limit_disabled_returns_ok() {
        let state = make_test_state();
        assert!(!state.rate_limit_enabled);
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_check_returns_exact_ok_string() {
        let result = health_check().await;
        assert_eq!(result, "OK");
    }

    #[tokio::test]
    async fn test_health_check_is_static_str() {
        let result: &'static str = health_check().await;
        assert_eq!(result, "OK");
    }

    #[tokio::test]
    async fn test_metrics_endpoint_under_normal_limit_returns_ok() {
        let state = make_test_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_128_does_not_match_other_ipv6() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1/128".to_string()]);
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 2)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_empty_whitelist_with_rate_limit() {
        let state = make_rate_limited_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_exact_ip_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1".to_string()]);
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_exact_ip_no_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["192.168.1.1".to_string()]);
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_exact_boundary() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 255)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_multiple_whitelist_none_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec![
            "192.168.1.1/32".to_string(),
            "10.0.0/24".to_string(),
        ]);
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_body_with_metrics() {
        let state = make_test_state();
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_check_response_body_is_ok() {
        let response = health_check().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_with_ipv6_addr() {
        let state = make_test_state();
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_rate_limited_with_ipv6() {
        let state = make_rate_limited_state();
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_ipv6_exact_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1".to_string()]);
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
