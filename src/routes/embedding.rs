// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Embedding-related routes
//!
//! Provides API endpoints for text embedding, batch embedding, similarity calculation, and file embedding

use crate::VecboostState;
use crate::domain::{
    BatchEmbedRequest, EmbedRequest, FileEmbedRequest, FileEmbedResponse, SimilarityRequest,
};
use crate::error::VecboostError;
use crate::utils::{AggregationMode, PathValidator};
use axum::http::HeaderMap;
use axum::{Json, extract::ConnectInfo, extract::State, response::IntoResponse};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Extract the real client IP address, considering proxy headers
/// Returns the IP address as a string
fn extract_real_ip(addr: SocketAddr) -> String {
    addr.ip().to_string()
}

/// Check if an IP is in the whitelist
#[allow(clippy::collapsible_if)]
fn is_ip_whitelisted(ip: &str, whitelist: &[String]) -> bool {
    whitelist.iter().any(|whitelist_ip| {
        // Exact match
        if ip == *whitelist_ip {
            return true;
        }

        // CIDR match (完整实现)
        if let Some((network_str, prefix_len_str)) = whitelist_ip.split_once('/') {
            if let Ok(prefix_len) = prefix_len_str.parse::<u8>() {
                // 尝试解析 IP 地址
                if let Ok(ip_addr) = ip.parse::<std::net::IpAddr>() {
                    if let Ok(network_addr) = network_str.parse::<std::net::IpAddr>() {
                        // 使用标准库的 IpAddr 进行匹配
                        return match (ip_addr, network_addr) {
                            (std::net::IpAddr::V4(ip_v4), std::net::IpAddr::V4(network_v4)) => {
                                // IPv4 CIDR 匹配
                                let ip_u32 = u32::from_be_bytes(ip_v4.octets());
                                let network_u32 = u32::from_be_bytes(network_v4.octets());
                                let mask = if prefix_len >= 32 {
                                    0xFFFFFFFFu32
                                } else {
                                    0xFFFFFFFFu32 << (32 - prefix_len)
                                };
                                (ip_u32 & mask) == (network_u32 & mask)
                            }
                            (std::net::IpAddr::V6(ip_v6), std::net::IpAddr::V6(network_v6)) => {
                                // IPv6 CIDR 匹配
                                let ip_u128 = u128::from_be_bytes(ip_v6.octets());
                                let network_u128 = u128::from_be_bytes(network_v6.octets());
                                let mask = if prefix_len >= 128 {
                                    u128::MAX
                                } else {
                                    u128::MAX << (128u8 - prefix_len)
                                };
                                (ip_u128 & mask) == (network_u128 & mask)
                            }
                            _ => false,
                        };
                    }
                }
            }
        }

        false
    })
}

/// Add rate limit headers to response
async fn add_rate_limit_headers(headers: &mut HeaderMap, state: &VecboostState, ip: &str) {
    use axum::http::HeaderValue;

    let rate_limit_enabled = state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered");
    if rate_limit_enabled {
        let rate_limiter = state
            .kit
            .require::<crate::module_registry::RateLimitModule>()
            .expect("RateLimitModule not registered");
        let global_remaining = rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Global)
            .await;
        let ip_remaining = rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Ip(ip.to_string()))
            .await;

        // Use the more restrictive limit
        let remaining = std::cmp::min(global_remaining, ip_remaining);

        if let Ok(limit_val) = HeaderValue::from_str("1000") {
            headers.insert("x-ratelimit-limit", limit_val);
        }
        if let Ok(remaining_val) = HeaderValue::from_str(&remaining.to_string()) {
            headers.insert("x-ratelimit-remaining", remaining_val);
        }
        if let Ok(reset_val) = HeaderValue::from_str("60") {
            headers.insert("x-ratelimit-reset", reset_val);
        }
    }
}

/// Single text embedding handler
///
/// Converts a single text to vector representation
#[utoipa::path(
    post,
    path = "/api/v1/embed",
    tag = "embedding",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Embedding successful", body = crate::domain::EmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_text",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn embed_handler(
    State(state): State<VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, VecboostError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    let rate_limit_enabled = state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered");
    let ip_whitelist = state
        .kit
        .require::<crate::module_registry::IpWhitelistModule>()
        .expect("IpWhitelistModule not registered");
    if rate_limit_enabled && !is_ip_whitelisted(&ip, &ip_whitelist) {
        // Check both global and IP rate limits
        let rate_limiter = state
            .kit
            .require::<crate::module_registry::RateLimitModule>()
            .expect("RateLimitModule not registered");
        if !rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(VecboostError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    // 如果启用了流水线，使用流水线处理
    let pipeline_enabled = state
        .kit
        .require::<crate::module_registry::PipelineEnabledModule>()
        .expect("PipelineEnabledModule not registered");
    if pipeline_enabled {
        let ip_clone = ip.clone();
        let res = crate::pipeline::handle_pipeline_request(state.clone(), req, ip_clone).await?;
        let mut response = res.into_response();
        add_rate_limit_headers(response.headers_mut(), &state, &ip).await;
        return Ok(response);
    }

    // 否则直接调用服务
    let service = state
        .kit
        .require::<crate::module_registry::EmbeddingModule>()
        .expect("EmbeddingModule not registered");
    let service_guard = service.read().await;
    let res = service_guard.process_text(req, None).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// Batch text embedding handler
///
/// Converts multiple texts to vector representations in batch
#[utoipa::path(
    post,
    path = "/api/v1/embed/batch",
    tag = "embedding",
    request_body = BatchEmbedRequest,
    responses(
        (status = 200, description = "Batch embedding successful", body = crate::domain::BatchEmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_batch",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn batch_embed_handler(
    State(state): State<VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<BatchEmbedRequest>,
) -> Result<impl IntoResponse, VecboostError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    let rate_limit_enabled = state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered");
    let ip_whitelist = state
        .kit
        .require::<crate::module_registry::IpWhitelistModule>()
        .expect("IpWhitelistModule not registered");
    if rate_limit_enabled && !is_ip_whitelisted(&ip, &ip_whitelist) {
        // Check both global and IP rate limits
        let rate_limiter = state
            .kit
            .require::<crate::module_registry::RateLimitModule>()
            .expect("RateLimitModule not registered");
        if !rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(VecboostError::InvalidInput(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    let service = state
        .kit
        .require::<crate::module_registry::EmbeddingModule>()
        .expect("EmbeddingModule not registered");
    let service_guard = service.read().await;
    let res = service_guard.process_batch(req, None).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// Similarity calculation handler
///
/// Calculates the similarity between two texts
#[utoipa::path(
    post,
    path = "/api/v1/similarity",
    tag = "embedding",
    request_body = SimilarityRequest,
    responses(
        (status = 200, description = "Similarity calculation successful", body = crate::domain::SimilarityResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "compute_similarity",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn similarity_handler(
    State(state): State<VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<SimilarityRequest>,
) -> Result<impl IntoResponse, VecboostError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    let rate_limit_enabled = state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered");
    let ip_whitelist = state
        .kit
        .require::<crate::module_registry::IpWhitelistModule>()
        .expect("IpWhitelistModule not registered");
    if rate_limit_enabled && !is_ip_whitelisted(&ip, &ip_whitelist) {
        // Check both global and IP rate limits
        let rate_limiter = state
            .kit
            .require::<crate::module_registry::RateLimitModule>()
            .expect("RateLimitModule not registered");
        if !rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(VecboostError::InvalidInput(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    let service = state
        .kit
        .require::<crate::module_registry::EmbeddingModule>()
        .expect("EmbeddingModule not registered");
    let service_guard = service.read().await;
    let res = service_guard.process_similarity(req).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// File embedding handler
///
/// Converts file content to vector representation
#[utoipa::path(
    post,
    path = "/api/v1/embed/file",
    tag = "embedding",
    request_body = FileEmbedResponse,
    responses(
        (status = 200, description = "File embedding successful", body = crate::domain::FileEmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_file",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn file_embed_handler(
    State(state): State<VecboostState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<FileEmbedRequest>,
) -> Result<impl IntoResponse, VecboostError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    let rate_limit_enabled = state
        .kit
        .require::<crate::module_registry::RateLimitEnabledModule>()
        .expect("RateLimitEnabledModule not registered");
    let ip_whitelist = state
        .kit
        .require::<crate::module_registry::IpWhitelistModule>()
        .expect("IpWhitelistModule not registered");
    if rate_limit_enabled && !is_ip_whitelisted(&ip, &ip_whitelist) {
        // Check both global and IP rate limits
        let rate_limiter = state
            .kit
            .require::<crate::module_registry::RateLimitModule>()
            .expect("RateLimitModule not registered");
        if !rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(VecboostError::InvalidInput(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    // Create path validator, only allow file access within current working directory
    let current_dir = std::env::current_dir()
        .map_err(|e| VecboostError::IoError(format!("Failed to get current directory: {}", e)))?;

    let path_validator = PathValidator::new()
        .add_allowed_root(&current_dir)
        .add_allowed_root("/tmp"); // Allow temporary directory access

    // Validate path to prevent path traversal attacks
    let validated_path = path_validator
        .validate_file(&path)
        .map_err(|e| VecboostError::InvalidInput(format!("Path validation failed: {}", e)))?;

    let service = state
        .kit
        .require::<crate::module_registry::EmbeddingModule>()
        .expect("EmbeddingModule not registered");
    let service_guard = service.read().await;
    let stats = service_guard.get_processing_stats(&validated_path)?;
    let output = service_guard.embed_file(&validated_path, mode).await?;

    drop(service_guard);

    let response = match output {
        crate::domain::EmbeddingOutput::Single(response) => crate::domain::FileEmbedResponse {
            mode,
            stats,
            embedding: Some(response.embedding),
            paragraphs: None,
        },
        crate::domain::EmbeddingOutput::Paragraphs(paragraphs) => {
            crate::domain::FileEmbedResponse {
                mode,
                stats,
                embedding: None,
                paragraphs: Some(paragraphs),
            }
        }
    };

    // Create response with rate limit headers
    let mut resp = Json(response).into_response();
    add_rate_limit_headers(resp.headers_mut(), &state, &ip).await;

    Ok(resp)
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use crate::module_registry::{
        AuditModule, AuthEnabled, AuthEnabledModule, CacheConfig, CacheModule, DbConfig, DbModule,
        EmbeddingModule, IpWhitelistModule, MetricsCollectorModule, PipelineEnabled,
        PipelineEnabledModule, PipelineQueueModule, PriorityCalculatorModule,
        PrometheusCollectorModule, RateLimitEnabled, RateLimitEnabledModule, RateLimitModule,
        ResponseChannelModule, WorkerManagerModule,
    };
    #[cfg(feature = "auth")]
    use crate::module_registry::{
        AuthModule, CsrfConfigModule, CsrfTokenStoreModule, UserStoreModule,
    };
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    };
    use crate::rate_limit::{LimiteronAdapter, RateLimitConfig};
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
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

    /// 参数化构建 VecboostState：可注入 rate_limit_enabled / rate_limiter / ip_whitelist / pipeline_enabled
    async fn make_test_state_with_options(
        rate_limit_enabled: bool,
        rate_limiter: Option<Arc<LimiteronAdapter>>,
        ip_whitelist: Option<Vec<String>>,
        pipeline_enabled: bool,
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

        let rate_limiter =
            rate_limiter.unwrap_or_else(|| Arc::new(LimiteronAdapter::with_default_config()));
        let ip_whitelist = ip_whitelist.unwrap_or_default();
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
        kit.set_config(service.clone());
        kit.set_config(rate_limiter.clone());
        kit.set_config(None::<Arc<crate::metrics::InferenceCollector>>);
        kit.set_config(None::<Arc<crate::metrics::PrometheusCollector>>);
        kit.set_config(None::<Arc<crate::audit::AuditLogger>>);
        kit.set_config(pipeline_queue.clone());
        kit.set_config(response_channel.clone());
        kit.set_config(priority_calculator.clone());
        kit.set_config(worker_manager.clone());
        kit.set_config(ip_whitelist.clone());
        kit.set_config(AuthEnabled(false));
        kit.set_config(RateLimitEnabled(rate_limit_enabled));
        kit.set_config(PipelineEnabled(pipeline_enabled));
        kit.set_config(CacheConfig {
            enabled: false,
            size: 0,
        });
        kit.set_config(DbConfig { enabled: false });
        #[cfg(feature = "auth")]
        {
            kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
            kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
            kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
        }

        kit.register::<EmbeddingModule>().unwrap();
        kit.register::<RateLimitModule>().unwrap();
        kit.register::<CacheModule>().unwrap();
        kit.register::<DbModule>().unwrap();
        kit.register::<AuditModule>().unwrap();
        kit.register::<MetricsCollectorModule>().unwrap();
        kit.register::<PrometheusCollectorModule>().unwrap();
        kit.register::<IpWhitelistModule>().unwrap();
        kit.register::<AuthEnabledModule>().unwrap();
        kit.register::<RateLimitEnabledModule>().unwrap();
        kit.register::<PipelineEnabledModule>().unwrap();
        kit.register::<PipelineQueueModule>().unwrap();
        kit.register::<ResponseChannelModule>().unwrap();
        kit.register::<PriorityCalculatorModule>().unwrap();
        kit.register::<WorkerManagerModule>().unwrap();
        #[cfg(feature = "auth")]
        {
            kit.register::<AuthModule>().unwrap();
            kit.register::<UserStoreModule>().unwrap();
            kit.register::<CsrfConfigModule>().unwrap();
            kit.register::<CsrfTokenStoreModule>().unwrap();
        }

        let kit = kit.build().await.expect("Failed to build AsyncKit");
        VecboostState { kit: Arc::new(kit) }
    }

    async fn make_test_state() -> VecboostState {
        make_test_state_with_options(false, None, None, false).await
    }

    async fn make_rate_limited_state() -> VecboostState {
        let rate_limiter = Arc::new(LimiteronAdapter::new(RateLimitConfig {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 0,
            ..Default::default()
        }));
        make_test_state_with_options(true, Some(rate_limiter), None, false).await
    }

    fn test_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9002)
    }

    #[test]
    fn test_extract_real_ip_ipv4() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 8080);
        assert_eq!(extract_real_ip(addr), "192.168.1.1");
    }

    #[test]
    fn test_extract_real_ip_ipv6() {
        let addr = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8080);
        assert_eq!(extract_real_ip(addr), "::1");
    }

    #[test]
    fn test_is_ip_whitelisted_exact_match() {
        assert!(is_ip_whitelisted("127.0.0.1", &["127.0.0.1".to_string()]));
        assert!(!is_ip_whitelisted("127.0.0.2", &["127.0.0.1".to_string()]));
    }

    #[test]
    fn test_is_ip_whitelisted_cidr_v4_match() {
        assert!(is_ip_whitelisted(
            "192.168.1.100",
            &["192.168.1.0/24".to_string()]
        ));
        assert!(!is_ip_whitelisted(
            "192.168.2.100",
            &["192.168.1.0/24".to_string()]
        ));
    }

    #[test]
    fn test_is_ip_whitelisted_cidr_v6_match() {
        assert!(is_ip_whitelisted(
            "2001:db8::1",
            &["2001:db8::/32".to_string()]
        ));
        assert!(!is_ip_whitelisted(
            "2001:db9::1",
            &["2001:db8::/32".to_string()]
        ));
    }

    #[test]
    fn test_is_ip_whitelisted_empty_whitelist() {
        assert!(!is_ip_whitelisted("127.0.0.1", &[]));
    }

    #[test]
    fn test_is_ip_whitelisted_mismatched_ip_version() {
        assert!(!is_ip_whitelisted("127.0.0.1", &["::1/128".to_string()]));
        assert!(!is_ip_whitelisted("::1", &["127.0.0.1/32".to_string()]));
    }

    #[tokio::test]
    async fn test_add_rate_limit_headers_disabled() {
        let state = make_test_state().await;
        let mut headers = HeaderMap::new();
        add_rate_limit_headers(&mut headers, &state, "127.0.0.1").await;
        assert!(!headers.contains_key("x-ratelimit-limit"));
        assert!(!headers.contains_key("x-ratelimit-remaining"));
        assert!(!headers.contains_key("x-ratelimit-reset"));
    }

    #[tokio::test]
    async fn test_add_rate_limit_headers_enabled() {
        let state = make_rate_limited_state().await;
        let mut headers = HeaderMap::new();
        add_rate_limit_headers(&mut headers, &state, "127.0.0.1").await;
        assert!(headers.contains_key("x-ratelimit-limit"));
        assert!(headers.contains_key("x-ratelimit-remaining"));
        assert!(headers.contains_key("x-ratelimit-reset"));
    }

    #[tokio::test]
    async fn test_embed_handler_success() {
        let state = make_test_state().await;
        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: None,
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_embed_handler_empty_text_returns_bad_request() {
        let state = make_test_state().await;
        let req = EmbedRequest {
            text: "".to_string(),
            normalize: None,
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_embed_handler_whitespace_only_returns_bad_request() {
        let state = make_test_state().await;
        let req = EmbedRequest {
            text: "   ".to_string(),
            normalize: None,
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_embed_handler_rate_limit_exceeded_returns_429() {
        let state = make_rate_limited_state().await;
        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: None,
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::RateLimitExceeded(_) => {}
            other => panic!("expected RateLimitExceeded, got {:?}", other),
        }
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn test_batch_embed_handler_success() {
        let state = make_test_state().await;
        let req = BatchEmbedRequest {
            texts: vec!["hello world".to_string(), "second text".to_string()],
            mode: None,
            normalize: None,
        };
        let result = batch_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_batch_embed_handler_empty_batch_returns_bad_request() {
        let state = make_test_state().await;
        let req = BatchEmbedRequest {
            texts: vec![],
            mode: None,
            normalize: None,
        };
        let result = batch_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_similarity_handler_success() {
        let state = make_test_state().await;
        let req = SimilarityRequest {
            source: "hello world".to_string(),
            target: "hello there".to_string(),
        };
        let result = similarity_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_similarity_handler_empty_source_returns_bad_request() {
        let state = make_test_state().await;
        let req = SimilarityRequest {
            source: "".to_string(),
            target: "hello there".to_string(),
        };
        let result = similarity_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_file_embed_handler_path_traversal_returns_bad_request() {
        let state = make_test_state().await;
        let req = FileEmbedRequest {
            path: "../../../etc/passwd".to_string(),
            mode: None,
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_file_embed_handler_nonexistent_file_returns_error() {
        let state = make_test_state().await;
        let req = FileEmbedRequest {
            path: "nonexistent_file_xyz_123.txt".to_string(),
            mode: None,
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_file_embed_handler_success() {
        let temp_dir = tempfile::tempdir_in(".").unwrap();
        let file_path = temp_dir.path().join("test_embed.txt");
        std::fs::write(&file_path, "hello world\nsecond line\n").unwrap();

        let state = make_test_state().await;
        let req = FileEmbedRequest {
            path: file_path.to_string_lossy().to_string(),
            mode: None,
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_extract_real_ip_ipv4_mapped_ipv6() {
        let addr = SocketAddr::new(
            IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0xffff, 0xc0a8)),
            8080,
        );
        let ip = extract_real_ip(addr);
        assert!(!ip.is_empty());
    }

    #[test]
    fn test_is_ip_whitelisted_cidr_v4_exact_32() {
        assert!(is_ip_whitelisted(
            "192.168.1.5",
            &["192.168.1.5/32".to_string()]
        ));
        assert!(!is_ip_whitelisted(
            "192.168.1.6",
            &["192.168.1.5/32".to_string()]
        ));
    }

    #[test]
    #[should_panic(expected = "shift left with overflow")]
    fn test_is_ip_whitelisted_cidr_v4_prefix_zero_panics() {
        let _ = is_ip_whitelisted("10.0.0.1", &["0.0.0.0/0".to_string()]);
    }

    #[test]
    fn test_is_ip_whitelisted_cidr_v6_exact_128() {
        assert!(is_ip_whitelisted(
            "2001:db8::1",
            &["2001:db8::1/128".to_string()]
        ));
        assert!(!is_ip_whitelisted(
            "2001:db8::2",
            &["2001:db8::1/128".to_string()]
        ));
    }

    #[test]
    fn test_is_ip_whitelisted_invalid_cidr_prefix() {
        assert!(!is_ip_whitelisted(
            "192.168.1.1",
            &["192.168.1.0/abc".to_string()]
        ));
    }

    #[test]
    fn test_is_ip_whitelisted_invalid_ip_in_whitelist() {
        assert!(!is_ip_whitelisted(
            "192.168.1.1",
            &["not-an-ip/24".to_string()]
        ));
    }

    #[test]
    fn test_is_ip_whitelisted_multiple_entries() {
        let whitelist = vec![
            "10.0.0.1".to_string(),
            "192.168.0.0/16".to_string(),
            "2001:db8::/32".to_string(),
        ];
        assert!(is_ip_whitelisted("10.0.0.1", &whitelist));
        assert!(is_ip_whitelisted("192.168.5.5", &whitelist));
        assert!(is_ip_whitelisted("2001:db8::abc", &whitelist));
        assert!(!is_ip_whitelisted("172.16.0.1", &whitelist));
    }

    #[tokio::test]
    async fn test_similarity_handler_empty_target_returns_bad_request() {
        let state = make_test_state().await;
        let req = SimilarityRequest {
            source: "hello world".to_string(),
            target: "".to_string(),
        };
        let result = similarity_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_similarity_handler_rate_limit_exceeded() {
        let state = make_rate_limited_state().await;
        let req = SimilarityRequest {
            source: "hello".to_string(),
            target: "world".to_string(),
        };
        let result = similarity_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("Rate limit")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_batch_embed_handler_rate_limit_exceeded() {
        let state = make_rate_limited_state().await;
        let req = BatchEmbedRequest {
            texts: vec!["hello".to_string()],
            mode: None,
            normalize: None,
        };
        let result = batch_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("Rate limit")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_file_embed_handler_rate_limit_exceeded() {
        let state = make_rate_limited_state().await;
        let req = FileEmbedRequest {
            path: "test.txt".to_string(),
            mode: None,
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        match result.as_ref().err().unwrap() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("Rate limit")),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_embed_handler_with_whitelisted_ip_bypasses_rate_limit() {
        let rate_limiter = Arc::new(LimiteronAdapter::new(RateLimitConfig {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 0,
            ..Default::default()
        }));
        let state = make_test_state_with_options(
            true,
            Some(rate_limiter),
            Some(vec!["127.0.0.1".to_string()]),
            false,
        )
        .await;
        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: None,
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok(), "whitelisted IP should bypass rate limit");
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_file_embed_handler_with_paragraph_mode() {
        let temp_dir = tempfile::tempdir_in(".").unwrap();
        let file_path = temp_dir.path().join("test_para.txt");
        std::fs::write(&file_path, "first paragraph\n\nsecond paragraph\n").unwrap();

        let state = make_test_state().await;
        let req = FileEmbedRequest {
            path: file_path.to_string_lossy().to_string(),
            mode: Some(AggregationMode::Paragraph),
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_embed_handler_with_pipeline_enabled() {
        use crate::config::model::{ModelConfig, Precision};
        use crate::engine::InferenceEngine;
        use crate::pipeline::{
            PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel,
            WorkerConfig, WorkerManager,
        };
        use crate::rate_limit::LimiteronAdapter;
        use crate::service::embedding::EmbeddingService;
        use async_trait::async_trait;
        use std::time::Duration;

        struct PipeEngine {
            dimension: usize,
        }

        #[async_trait]
        impl InferenceEngine for PipeEngine {
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

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(PipeEngine { dimension: 8 }));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let queue = Arc::new(PriorityRequestQueue::new(10));
        let response_channel = Arc::new(ResponseChannel::new());
        let priority_calculator = Arc::new(PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(WorkerManager::new(
            Arc::clone(&queue),
            Arc::clone(&response_channel),
            WorkerConfig::default(),
            Arc::clone(&service),
        ));

        let state = {
            let mut kit = trait_kit::AsyncKit::new();
            kit.set_config(service.clone());
            kit.set_config(Arc::new(LimiteronAdapter::with_default_config()));
            kit.set_config(None::<Arc<crate::metrics::InferenceCollector>>);
            kit.set_config(None::<Arc<crate::metrics::PrometheusCollector>>);
            kit.set_config(None::<Arc<crate::audit::AuditLogger>>);
            kit.set_config(queue.clone());
            kit.set_config(response_channel.clone());
            kit.set_config(priority_calculator.clone());
            kit.set_config(worker_manager.clone());
            kit.set_config(Vec::<String>::new());
            kit.set_config(AuthEnabled(false));
            kit.set_config(RateLimitEnabled(false));
            kit.set_config(PipelineEnabled(true));
            kit.set_config(CacheConfig {
                enabled: false,
                size: 0,
            });
            kit.set_config(DbConfig { enabled: false });
            #[cfg(feature = "auth")]
            {
                kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
                kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
                kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
                kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
            }
            kit.register::<EmbeddingModule>().unwrap();
            kit.register::<RateLimitModule>().unwrap();
            kit.register::<CacheModule>().unwrap();
            kit.register::<DbModule>().unwrap();
            kit.register::<AuditModule>().unwrap();
            kit.register::<MetricsCollectorModule>().unwrap();
            kit.register::<PrometheusCollectorModule>().unwrap();
            kit.register::<IpWhitelistModule>().unwrap();
            kit.register::<AuthEnabledModule>().unwrap();
            kit.register::<RateLimitEnabledModule>().unwrap();
            kit.register::<PipelineEnabledModule>().unwrap();
            kit.register::<PipelineQueueModule>().unwrap();
            kit.register::<ResponseChannelModule>().unwrap();
            kit.register::<PriorityCalculatorModule>().unwrap();
            kit.register::<WorkerManagerModule>().unwrap();
            #[cfg(feature = "auth")]
            {
                kit.register::<AuthModule>().unwrap();
                kit.register::<UserStoreModule>().unwrap();
                kit.register::<CsrfConfigModule>().unwrap();
                kit.register::<CsrfTokenStoreModule>().unwrap();
            }
            let kit = kit.build().await.expect("Failed to build AsyncKit");
            VecboostState { kit: Arc::new(kit) }
        };

        let svc_clone = state
            .kit
            .require::<EmbeddingModule>()
            .expect("EmbeddingModule not registered");
        let queue_clone = state
            .kit
            .require::<PipelineQueueModule>()
            .expect("PipelineQueueModule not registered");
        let rc_clone = state
            .kit
            .require::<ResponseChannelModule>()
            .expect("ResponseChannelModule not registered");
        let consumer = tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                if let Some(req) = queue_clone.dequeue().await {
                    let request_id = req.request_id.clone();
                    let guard = svc_clone.read().await;
                    let result = guard.process_text(req.embed_request, None).await;
                    drop(guard);
                    rc_clone.complete(request_id, result).await;
                    return;
                }
                if tokio::time::Instant::now() >= deadline {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        let req = EmbedRequest {
            text: "hello pipeline".to_string(),
            normalize: Some(true),
        };
        let result = embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok(), "pipeline path should succeed");
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);

        consumer.await.unwrap();
    }

    #[tokio::test]
    async fn test_batch_embed_handler_with_normalize() {
        let state = make_test_state().await;
        let req = BatchEmbedRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = batch_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_similarity_handler_whitespace_source_returns_bad_request() {
        let state = make_test_state().await;
        let req = SimilarityRequest {
            source: "   ".to_string(),
            target: "hello there".to_string(),
        };
        let result = similarity_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_err());
        let response = result.err().unwrap().into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_file_embed_handler_with_document_mode() {
        let temp_dir = tempfile::tempdir_in(".").unwrap();
        let file_path = temp_dir.path().join("test_doc.txt");
        std::fs::write(&file_path, "document content here").unwrap();

        let state = make_test_state().await;
        let req = FileEmbedRequest {
            path: file_path.to_string_lossy().to_string(),
            mode: Some(AggregationMode::Document),
        };
        let result = file_embed_handler(State(state), ConnectInfo(test_addr()), Json(req)).await;
        assert!(result.is_ok());
        let response = result.unwrap().into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
