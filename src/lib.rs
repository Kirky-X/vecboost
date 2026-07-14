// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(feature = "http")]
use axum::extract::FromRef;
use std::sync::Arc;
use tokio::sync::RwLock;

// 公共 API 模块 - 被 main.rs 使用或作为库的公共接口
#[cfg(feature = "http")]
pub mod api;
pub mod audit;
#[cfg(feature = "auth")]
pub mod auth;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
#[cfg(feature = "db")]
pub mod db;
pub mod domain;
pub mod engine;
#[cfg(feature = "grpc")]
pub mod grpc;
pub mod metrics;
pub mod module_registry;
pub mod pipeline;
pub mod rate_limit;
pub mod routes;
pub mod security;
pub mod service;
pub mod utils;

// 条件编译模块 — 仅在对应 feature 启用时可见
#[cfg(feature = "inklog")]
pub mod logger;

// 导出 config::app 中的类型
pub use crate::pipeline::PriorityConfig;

// 内部实现模块 - 只在 crate 内部使用，不暴露给外部
pub(crate) mod cache;
pub(crate) mod device;
pub mod error;
pub(crate) mod model;
pub(crate) mod monitor;
pub(crate) mod text;

// 重新导出必要的内部类型（最小化暴露原则）
#[cfg(feature = "db")]
pub use config::app::DatabaseConfig;
pub use config::app::{AppConfig, AuthConfig, CsrfConfig, RateLimitConfig, ServerConfig};
pub use config::model::ModelConfig;
pub use domain::{EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse};
pub use error::VecboostError;
pub use service::embedding::EmbeddingService;
pub use utils::SimilarityMetric;

#[allow(deprecated)]
pub use error::AppError;

/// Application state
///
/// Contains shared state needed by all route handlers
/// Note: This struct is exposed to main.rs and other crates using this library
#[derive(Clone)]
pub struct AppState {
    pub service: Arc<RwLock<EmbeddingService>>,
    #[cfg(feature = "auth")]
    pub jwt_manager: Option<Arc<auth::JwtManager>>,
    #[cfg(feature = "auth")]
    pub user_store: Option<Arc<auth::UserStore>>,
    pub auth_enabled: bool,
    #[cfg(feature = "auth")]
    pub csrf_config: Option<Arc<auth::CsrfConfig>>,
    #[cfg(feature = "auth")]
    pub csrf_token_store: Option<Arc<auth::CsrfTokenStore>>,
    pub metrics_collector: Option<Arc<metrics::InferenceCollector>>,
    pub prometheus_collector: Option<Arc<metrics::PrometheusCollector>>,
    pub rate_limiter: Arc<rate_limit::LimiteronAdapter>,
    pub ip_whitelist: Vec<String>,
    pub rate_limit_enabled: bool,
    pub audit_logger: Option<Arc<audit::AuditLogger>>,
    pub pipeline_enabled: bool,
    pub pipeline_queue: Arc<pipeline::PriorityRequestQueue>,
    pub response_channel: Arc<pipeline::ResponseChannel>,
    pub priority_calculator: Arc<pipeline::PriorityCalculator>,
    pub worker_manager: Arc<pipeline::WorkerManager>,
    /// trait-kit AsyncKit — 模块能力管理中心（D1 集成）
    ///
    /// `AsyncKit<Ready>` 是 `Send + Sync`（基于 `Arc<RwLock>`），可安全存入
    /// `AppState` 并跨线程共享。启动时由 `main.rs` 构建后注入。
    /// 使用 `Option` 以兼容非 http feature 下可能不构建 kit 的场景。
    pub kit: Option<Arc<trait_kit::AsyncKit<trait_kit::AsyncReady>>>,
}

#[cfg(feature = "http")]
impl FromRef<AppState> for Arc<RwLock<EmbeddingService>> {
    fn from_ref(state: &AppState) -> Self {
        state.service.clone()
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
impl FromRef<AppState> for Arc<auth::JwtManager> {
    fn from_ref(state: &AppState) -> Self {
        state
            .jwt_manager
            .clone()
            .expect("JWT manager not available")
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
impl FromRef<AppState> for Arc<auth::UserStore> {
    fn from_ref(state: &AppState) -> Self {
        state.user_store.clone().expect("User store not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<AppState> for Arc<metrics::InferenceCollector> {
    fn from_ref(state: &AppState) -> Self {
        state
            .metrics_collector
            .clone()
            .expect("Metrics collector not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<AppState> for Arc<metrics::PrometheusCollector> {
    fn from_ref(state: &AppState) -> Self {
        state
            .prometheus_collector
            .clone()
            .expect("Prometheus collector not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<AppState> for Arc<rate_limit::LimiteronAdapter> {
    fn from_ref(state: &AppState) -> Self {
        state.rate_limiter.clone()
    }
}

#[cfg(feature = "http")]
impl FromRef<AppState> for Option<Arc<audit::AuditLogger>> {
    fn from_ref(state: &AppState) -> Self {
        state.audit_logger.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use crate::engine::InferenceEngine;
    use crate::pipeline::{PriorityConfig, WorkerConfig};
    use async_trait::async_trait;

    struct MockEngine;

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 384])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_app_state() -> AppState {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));

        let rate_limiter = Arc::new(rate_limit::LimiteronAdapter::with_default_config());

        let pipeline_queue = Arc::new(pipeline::PriorityRequestQueue::new(100));
        let response_channel = Arc::new(pipeline::ResponseChannel::new());
        let priority_calculator =
            Arc::new(pipeline::PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(pipeline::WorkerManager::new(
            pipeline_queue.clone(),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        let metrics_collector = Some(Arc::new(metrics::InferenceCollector::new()));
        let prometheus_collector = Some(Arc::new(
            metrics::PrometheusCollector::new().expect("Failed to create PrometheusCollector"),
        ));

        AppState {
            service,
            #[cfg(feature = "auth")]
            jwt_manager: None,
            #[cfg(feature = "auth")]
            user_store: None,
            auth_enabled: false,
            #[cfg(feature = "auth")]
            csrf_config: None,
            #[cfg(feature = "auth")]
            csrf_token_store: None,
            metrics_collector,
            prometheus_collector,
            rate_limiter,
            ip_whitelist: vec![],
            rate_limit_enabled: false,
            audit_logger: None,
            pipeline_enabled: false,
            pipeline_queue,
            response_channel,
            priority_calculator,
            worker_manager,
            kit: None,
        }
    }

    #[test]
    fn test_app_state_construction() {
        let state = make_app_state();
        assert!(!state.auth_enabled);
        assert!(!state.rate_limit_enabled);
        assert!(!state.pipeline_enabled);
        assert!(state.ip_whitelist.is_empty());
        assert!(state.metrics_collector.is_some());
        assert!(state.prometheus_collector.is_some());
        assert!(state.audit_logger.is_none());
        assert!(state.kit.is_none());
    }

    #[test]
    fn test_app_state_clone_preserves_arcs() {
        let state = make_app_state();
        let cloned = state.clone();
        assert!(Arc::ptr_eq(&state.service, &cloned.service));
        assert!(Arc::ptr_eq(&state.rate_limiter, &cloned.rate_limiter));
        assert!(Arc::ptr_eq(
            state.metrics_collector.as_ref().unwrap(),
            cloned.metrics_collector.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            state.prometheus_collector.as_ref().unwrap(),
            cloned.prometheus_collector.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(&state.pipeline_queue, &cloned.pipeline_queue));
        assert!(Arc::ptr_eq(
            &state.response_channel,
            &cloned.response_channel
        ));
        assert!(Arc::ptr_eq(
            &state.priority_calculator,
            &cloned.priority_calculator
        ));
        assert!(Arc::ptr_eq(&state.worker_manager, &cloned.worker_manager));
    }

    #[test]
    fn test_app_state_field_access_and_mutation() {
        let mut state = make_app_state();
        state.auth_enabled = true;
        state.rate_limit_enabled = true;
        state.pipeline_enabled = true;
        state.ip_whitelist = vec!["127.0.0.1".to_string(), "10.0.0.0/8".to_string()];

        assert!(state.auth_enabled);
        assert!(state.rate_limit_enabled);
        assert!(state.pipeline_enabled);
        assert_eq!(state.ip_whitelist.len(), 2);
        assert_eq!(state.ip_whitelist[0], "127.0.0.1");
    }

    #[test]
    fn test_app_state_auth_fields_default_none() {
        let state = make_app_state();
        #[cfg(feature = "auth")]
        {
            assert!(state.jwt_manager.is_none());
            assert!(state.user_store.is_none());
            assert!(state.csrf_config.is_none());
            assert!(state.csrf_token_store.is_none());
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_service() {
        let state = make_app_state();
        let service: Arc<RwLock<EmbeddingService>> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(&service, &state.service));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_rate_limiter() {
        let state = make_app_state();
        let limiter: Arc<rate_limit::LimiteronAdapter> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(&limiter, &state.rate_limiter));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_metrics_collector() {
        let state = make_app_state();
        let collector: Arc<metrics::InferenceCollector> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(
            &collector,
            state.metrics_collector.as_ref().unwrap()
        ));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_prometheus_collector() {
        let state = make_app_state();
        let collector: Arc<metrics::PrometheusCollector> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(
            &collector,
            state.prometheus_collector.as_ref().unwrap()
        ));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_audit_logger_returns_none() {
        let state = make_app_state();
        let logger: Option<Arc<audit::AuditLogger>> = FromRef::from_ref(&state);
        assert!(logger.is_none());
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_ref_audit_logger_returns_some() {
        let mut state = make_app_state();
        let config = audit::AuditConfig {
            enabled: false,
            ..Default::default()
        };
        state.audit_logger = Some(Arc::new(audit::AuditLogger::new(config)));
        let logger: Option<Arc<audit::AuditLogger>> = FromRef::from_ref(&state);
        assert!(logger.is_some());
        assert!(Arc::ptr_eq(
            logger.as_ref().unwrap(),
            state.audit_logger.as_ref().unwrap()
        ));
    }

    #[cfg(all(feature = "http", feature = "auth"))]
    #[test]
    fn test_from_ref_jwt_manager_with_some() {
        let mut state = make_app_state();
        let secret = "test_secret_key_for_jwt_validation_must_be_long_enough_12345678";
        state.jwt_manager = Some(Arc::new(
            auth::JwtManager::new(secret.to_string()).expect("Failed to create JwtManager"),
        ));
        let manager: Arc<auth::JwtManager> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(&manager, state.jwt_manager.as_ref().unwrap()));
    }

    #[cfg(all(feature = "http", feature = "auth", feature = "db"))]
    #[tokio::test]
    async fn test_from_ref_user_store_with_some() {
        let pool = crate::db::DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create in-memory db pool");
        let mut state = make_app_state();
        state.user_store = Some(Arc::new(auth::UserStore::new(Arc::new(pool))));
        let store: Arc<auth::UserStore> = FromRef::from_ref(&state);
        assert!(Arc::ptr_eq(&store, state.user_store.as_ref().unwrap()));
    }

    #[cfg(all(feature = "http", feature = "auth"))]
    #[test]
    fn test_from_ref_jwt_manager_panics_when_none() {
        let state = make_app_state();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<auth::JwtManager> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when jwt_manager is None"
        );
    }

    #[cfg(all(feature = "http", feature = "auth"))]
    #[test]
    fn test_from_ref_user_store_panics_when_none() {
        let state = make_app_state();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<auth::UserStore> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when user_store is None"
        );
    }

    #[test]
    fn test_app_state_with_auth_enabled_flag() {
        let mut state = make_app_state();
        state.auth_enabled = true;
        assert!(state.auth_enabled);

        state.auth_enabled = false;
        assert!(!state.auth_enabled);
    }

    #[test]
    fn test_app_state_ip_whitelist_mutability() {
        let mut state = make_app_state();
        state.ip_whitelist.push("192.168.1.1".to_string());
        state.ip_whitelist.push("10.0.0.1".to_string());
        assert_eq!(state.ip_whitelist.len(), 2);
        assert!(state.ip_whitelist.contains(&"192.168.1.1".to_string()));
        assert!(state.ip_whitelist.contains(&"10.0.0.1".to_string()));
    }
}
