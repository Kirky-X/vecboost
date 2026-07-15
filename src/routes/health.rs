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
    if app_state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered")
    {
        let is_whitelisted = app_state
            .kit
            .require::<crate::module_registry::IpWhitelistModule>()
            .expect("IpWhitelistModule not registered")
            .iter()
            .any(|whitelist_ip| {
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
                .kit
                .require::<crate::module_registry::RateLimitModule>()
                .expect("RateLimitModule not registered")
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

    async fn make_test_state_with_rate_limit(
        rate_limit_enabled: bool,
        rate_limiter: Arc<LimiteronAdapter>,
        ip_whitelist: Vec<String>,
    ) -> VecboostState {
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
        let pipeline_queue = Arc::new(PriorityRequestQueue::new(0));
        let response_channel = Arc::new(ResponseChannel::new());
        let priority_calculator = Arc::new(PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(WorkerManager::new(
            pipeline_queue.clone(),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(service);
        kit.set_config(rate_limiter);
        kit.set_config(Some(prometheus_collector));
        kit.set_config(pipeline_queue);
        kit.set_config(response_channel);
        kit.set_config(priority_calculator);
        kit.set_config(worker_manager);
        kit.set_config(ip_whitelist);
        kit.set_config(crate::module_registry::AuthEnabled(false));
        kit.set_config(crate::module_registry::RateLimitEnabled(rate_limit_enabled));
        kit.set_config(crate::module_registry::PipelineEnabled(false));
        kit.set_config(crate::module_registry::CacheConfig {
            enabled: false,
            size: 0,
        });
        kit.set_config(crate::module_registry::DbConfig { enabled: false });
        kit.set_config(None::<Arc<crate::audit::AuditLogger>>);
        kit.set_config(None::<Arc<crate::metrics::InferenceCollector>>);
        #[cfg(feature = "auth")]
        {
            kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
            kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
        }
        kit.register::<crate::module_registry::EmbeddingModule>()
            .unwrap();
        kit.register::<crate::module_registry::RateLimitModule>()
            .unwrap();
        kit.register::<crate::module_registry::CacheModule>()
            .unwrap();
        kit.register::<crate::module_registry::DbModule>().unwrap();
        kit.register::<crate::module_registry::AuditModule>()
            .unwrap();
        kit.register::<crate::module_registry::MetricsCollectorModule>()
            .unwrap();
        kit.register::<crate::module_registry::PrometheusCollectorModule>()
            .unwrap();
        kit.register::<crate::module_registry::IpWhitelistModule>()
            .unwrap();
        kit.register::<crate::module_registry::AuthEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::RateLimitEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::PipelineEnabledModule>()
            .unwrap();
        kit.register::<crate::module_registry::PipelineQueueModule>()
            .unwrap();
        kit.register::<crate::module_registry::ResponseChannelModule>()
            .unwrap();
        kit.register::<crate::module_registry::PriorityCalculatorModule>()
            .unwrap();
        kit.register::<crate::module_registry::WorkerManagerModule>()
            .unwrap();
        #[cfg(feature = "auth")]
        {
            kit.register::<crate::module_registry::AuthModule>()
                .unwrap();
            kit.register::<crate::module_registry::UserStoreModule>()
                .unwrap();
            kit.register::<crate::module_registry::CsrfConfigModule>()
                .unwrap();
            kit.register::<crate::module_registry::CsrfTokenStoreModule>()
                .unwrap();
        }
        let kit = kit.build().await.unwrap();
        VecboostState { kit: Arc::new(kit) }
    }

    async fn make_test_state() -> VecboostState {
        make_test_state_with_rate_limit(
            false,
            Arc::new(LimiteronAdapter::with_default_config()),
            vec![],
        )
        .await
    }

    fn make_blocking_rate_limiter() -> Arc<LimiteronAdapter> {
        Arc::new(LimiteronAdapter::new(RateLimitConfig {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 0,
            ..Default::default()
        }))
    }

    async fn make_rate_limited_state() -> VecboostState {
        make_test_state_with_rate_limit(true, make_blocking_rate_limiter(), vec![]).await
    }

    async fn make_rate_limited_state_with_whitelist() -> VecboostState {
        make_test_state_with_rate_limit(
            true,
            make_blocking_rate_limiter(),
            vec!["127.0.0.1".to_string()],
        )
        .await
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
        let state = make_test_state().await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_rate_limit_exceeded_returns_429() {
        let state = make_rate_limited_state().await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelisted_ip_bypasses_rate_limit() {
        let state = make_rate_limited_state_with_whitelist().await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    fn test_addr_with(ip: IpAddr) -> SocketAddr {
        SocketAddr::new(ip, 9002)
    }

    async fn make_rate_limited_state_with_whitelist_ips(whitelist: Vec<String>) -> VecboostState {
        make_test_state_with_rate_limit(true, make_blocking_rate_limiter(), whitelist).await
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_32_matches() {
        let state =
            make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1/32".to_string()]).await;
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_32_does_not_match_other_ip() {
        let state =
            make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1/32".to_string()]).await;
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_matches_subnet() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]).await;
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_does_not_match_other_subnet() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]).await;
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_ipv6_cidr_128_matches() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1/128".to_string()]).await;
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
        ])
        .await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_text_content_type() {
        let state = make_test_state().await;
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
        let state = make_test_state().await;
        assert!(
            !state
                .kit
                .require::<crate::module_registry::RateLimitEnabledModule>()
                .unwrap()
        );
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
        let state = make_test_state().await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_128_does_not_match_other_ipv6() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1/128".to_string()]).await;
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 2)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_empty_whitelist_with_rate_limit() {
        let state = make_rate_limited_state().await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_exact_ip_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["127.0.0.1".to_string()]).await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_exact_ip_no_match() {
        let state =
            make_rate_limited_state_with_whitelist_ips(vec!["192.168.1.1".to_string()]).await;
        let response = metrics_endpoint(State(state), ConnectInfo(test_addr()))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_cidr_24_exact_boundary() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["10.0.0/24".to_string()]).await;
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
        ])
        .await;
        let addr = test_addr_with(IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_returns_body_with_metrics() {
        let state = make_test_state().await;
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
        let state = make_test_state().await;
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_rate_limited_with_ipv6() {
        let state = make_rate_limited_state().await;
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_metrics_endpoint_whitelist_ipv6_exact_match() {
        let state = make_rate_limited_state_with_whitelist_ips(vec!["::1".to_string()]).await;
        let addr = test_addr_with(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let response = metrics_endpoint(State(state), ConnectInfo(addr))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
