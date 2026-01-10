// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::extract::FromRef;
use std::sync::Arc;
use tokio::sync::RwLock;

// 公共 API 模块 - 被 main.rs 使用或作为库的公共接口
pub mod audit;
pub mod auth;
pub mod config;
pub mod domain;
pub mod engine;
pub mod grpc;
pub mod metrics;
pub mod pipeline;
pub mod rate_limit;
pub mod routes;
pub mod security;
pub mod service;
pub mod utils;

// 导出 config::app 中的类型
pub use crate::pipeline::config::PriorityConfig;

// 内部实现模块 - 只在 crate 内部使用，不暴露给外部
pub(crate) mod cache;
pub(crate) mod device;
pub mod error; // 暴露给测试文件使用
pub(crate) mod model;
pub(crate) mod monitor;
pub(crate) mod text;

pub use config::app::{AppConfig, AuthConfig, CsrfConfig, RateLimitConfig, ServerConfig};
pub use config::model::ModelConfig;
pub use domain::{EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse};
pub use service::embedding::EmbeddingService;
pub use utils::SimilarityMetric;

/// Application state
///
/// Contains shared state needed by all route handlers
/// Note: This struct is exposed to main.rs and other crates using this library
#[derive(Clone)]
pub struct AppState {
    pub service: Arc<RwLock<EmbeddingService>>,
    pub jwt_manager: Option<Arc<auth::JwtManager>>,
    pub user_store: Option<Arc<auth::UserStore>>,
    pub auth_enabled: bool,
    pub csrf_config: Option<Arc<auth::CsrfConfig>>,
    pub csrf_token_store: Option<Arc<auth::CsrfTokenStore>>,
    pub metrics_collector: Option<Arc<metrics::InferenceCollector>>,
    pub prometheus_collector: Option<Arc<metrics::PrometheusCollector>>,
    pub rate_limiter: Arc<rate_limit::RateLimiter>,
    pub ip_whitelist: Vec<String>,
    pub rate_limit_enabled: bool,
    pub audit_logger: Option<Arc<audit::AuditLogger>>,
    pub pipeline_enabled: bool,
    pub pipeline_queue: Arc<pipeline::PriorityRequestQueue>,
    pub response_channel: Arc<pipeline::ResponseChannel>,
    pub priority_calculator: Arc<pipeline::PriorityCalculator>,
}

impl FromRef<AppState> for Arc<RwLock<EmbeddingService>> {
    fn from_ref(state: &AppState) -> Self {
        state.service.clone()
    }
}

impl FromRef<AppState> for Arc<auth::JwtManager> {
    fn from_ref(state: &AppState) -> Self {
        state
            .jwt_manager
            .clone()
            .expect("JWT manager not available")
    }
}

impl FromRef<AppState> for Arc<auth::UserStore> {
    fn from_ref(state: &AppState) -> Self {
        state.user_store.clone().expect("User store not available")
    }
}

impl FromRef<AppState> for Arc<metrics::InferenceCollector> {
    fn from_ref(state: &AppState) -> Self {
        state
            .metrics_collector
            .clone()
            .expect("Metrics collector not available")
    }
}

impl FromRef<AppState> for Arc<metrics::PrometheusCollector> {
    fn from_ref(state: &AppState) -> Self {
        state
            .prometheus_collector
            .clone()
            .expect("Prometheus collector not available")
    }
}

impl FromRef<AppState> for Arc<rate_limit::RateLimiter> {
    fn from_ref(state: &AppState) -> Self {
        state.rate_limiter.clone()
    }
}

impl FromRef<AppState> for Option<Arc<audit::AuditLogger>> {
    fn from_ref(state: &AppState) -> Self {
        state.audit_logger.clone()
    }
}
