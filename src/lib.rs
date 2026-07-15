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
/// v0.3.0 D3 重构：所有能力通过 `AsyncKit<Ready>` 查询。
/// 启动时由 `main.rs` 通过 `kit.set_config()` 注入预构建对象 + `kit.register::<M>()`
/// 注册 17 个 Module,`kit.build().await` 后注入到 `VecboostState`。
///
/// 路由 handler 通过 `state.kit.require::<M>().expect("...")` 检索能力,
/// 或通过 Axum `FromRef` 自动注入(`FromRef` impl 也走 `kit.require`)。
#[derive(Clone)]
pub struct VecboostState {
    /// trait-kit AsyncKit — 模块能力管理中心
    ///
    /// `AsyncKit<Ready>` 是 `Send + Sync`(基于 `Arc<RwLock>`),可安全存入
    /// `VecboostState` 并跨线程共享。包含 17 个 Module 的能力查询入口:
    /// - 4 现有:EmbeddingModule/AuthModule/RateLimitModule/AuditModule
    /// - 13 新增:覆盖原 14 字段剩余 13 个(详见 module_registry/mod.rs)
    pub kit: Arc<trait_kit::AsyncKit<trait_kit::AsyncReady>>,
}

#[cfg(feature = "http")]
impl FromRef<VecboostState> for Arc<RwLock<EmbeddingService>> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::EmbeddingModule>()
            .expect("EmbeddingService capability not registered in kit")
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
impl FromRef<VecboostState> for Arc<auth::JwtManager> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::AuthModule>()
            .and_then(|opt| {
                opt.ok_or_else(|| trait_kit::TraitKitError::MissingCapability {
                    key: "jwt_manager (auth disabled at runtime)",
                })
            })
            .expect("JWT manager capability not available")
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
impl FromRef<VecboostState> for Arc<auth::UserStore> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::UserStoreModule>()
            .and_then(|opt| {
                opt.ok_or_else(|| trait_kit::TraitKitError::MissingCapability {
                    key: "user_store (auth disabled at runtime)",
                })
            })
            .expect("UserStore capability not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<VecboostState> for Arc<metrics::InferenceCollector> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::MetricsCollectorModule>()
            .and_then(|opt| {
                opt.ok_or_else(|| trait_kit::TraitKitError::MissingCapability {
                    key: "metrics_collector (not configured)",
                })
            })
            .expect("InferenceCollector capability not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<VecboostState> for Arc<metrics::PrometheusCollector> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::PrometheusCollectorModule>()
            .and_then(|opt| {
                opt.ok_or_else(|| trait_kit::TraitKitError::MissingCapability {
                    key: "prometheus_collector (not configured)",
                })
            })
            .expect("PrometheusCollector capability not available")
    }
}

#[cfg(feature = "http")]
impl FromRef<VecboostState> for Arc<rate_limit::LimiteronAdapter> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::RateLimitModule>()
            .expect("RateLimitModule capability not registered in kit")
    }
}

#[cfg(feature = "http")]
impl FromRef<VecboostState> for Option<Arc<audit::AuditLogger>> {
    fn from_ref(state: &VecboostState) -> Self {
        state
            .kit
            .require::<module_registry::AuditModule>()
            .expect("AuditModule capability not registered in kit")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
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

    /// 参数化构建 VecboostState：可选注入 metrics/prometheus/audit 能力
    ///
    /// `metrics` / `prometheus` / `audit` 为 None 时,Module 仍注册但能力查询返回 None,
    /// 用于测试 FromRef panic 路径。其他能力（service/rate_limiter/pipeline 组件）
    /// 始终注入,因为这些是必需能力（missing config = build 失败）。
    async fn make_app_state_with_options(
        metrics: Option<Arc<metrics::InferenceCollector>>,
        prometheus: Option<Arc<metrics::PrometheusCollector>>,
        audit: Option<Arc<audit::AuditLogger>>,
    ) -> VecboostState {
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

        let mut kit = trait_kit::AsyncKit::new();
        // 注入复杂类型能力（预构建对象）
        kit.set_config(service.clone());
        kit.set_config(rate_limiter.clone());
        kit.set_config(metrics.clone());
        kit.set_config(prometheus.clone());
        kit.set_config(audit.clone());
        kit.set_config(pipeline_queue.clone());
        kit.set_config(response_channel.clone());
        kit.set_config(priority_calculator.clone());
        kit.set_config(worker_manager.clone());
        kit.set_config(Vec::<String>::new());
        // bool newtype 配置（missing = false，与 CacheModule/DbModule 一致）
        kit.set_config(AuthEnabled(false));
        kit.set_config(RateLimitEnabled(false));
        kit.set_config(PipelineEnabled(false));
        kit.set_config(CacheConfig {
            enabled: false,
            size: 0,
        });
        kit.set_config(DbConfig { enabled: false });

        // auth feature 能力（全部 None — 默认禁用）
        #[cfg(feature = "auth")]
        {
            kit.set_config(Option::<Arc<auth::JwtManager>>::None);
            kit.set_config(Option::<Arc<auth::UserStore>>::None);
            kit.set_config(Option::<Arc<auth::CsrfConfig>>::None);
            kit.set_config(Option::<Arc<auth::CsrfTokenStore>>::None);
        }

        // 注册所有 Module（15 个非 auth + 4 个 auth feature）
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

    /// 默认完整 VecboostState：metrics=Some, prometheus=Some, audit=None
    async fn make_app_state() -> VecboostState {
        make_app_state_with_options(
            Some(Arc::new(metrics::InferenceCollector::new())),
            Some(Arc::new(
                metrics::PrometheusCollector::new().expect("Failed to create PrometheusCollector"),
            )),
            None,
        )
        .await
    }

    // -------------------------------------------------------------------------
    // 构建与 Clone 测试
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_app_state_construction() {
        let state = make_app_state().await;
        assert!(state.kit.contains::<EmbeddingModule>());
        assert!(state.kit.contains::<RateLimitModule>());
        assert!(state.kit.contains::<AuditModule>());
        assert!(state.kit.contains::<MetricsCollectorModule>());
        assert!(state.kit.contains::<PrometheusCollectorModule>());
        assert!(state.kit.contains::<IpWhitelistModule>());
        assert!(state.kit.contains::<PipelineQueueModule>());
        assert!(state.kit.contains::<WorkerManagerModule>());
    }

    #[tokio::test]
    async fn test_app_state_clone_preserves_kit_arc() {
        let state = make_app_state().await;
        let cloned = state.clone();
        assert!(Arc::ptr_eq(&state.kit, &cloned.kit));
    }

    #[tokio::test]
    async fn test_app_state_multiple_clones_share_kit() {
        let state = make_app_state().await;
        let clone1 = state.clone();
        let clone2 = state.clone();
        let clone3 = state.clone();
        assert!(Arc::ptr_eq(&state.kit, &clone1.kit));
        assert!(Arc::ptr_eq(&state.kit, &clone2.kit));
        assert!(Arc::ptr_eq(&state.kit, &clone3.kit));
    }

    // -------------------------------------------------------------------------
    // kit.require 能力查询测试
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_kit_require_embedding_service() {
        let state = make_app_state().await;
        let service = state.kit.require::<EmbeddingModule>().unwrap();
        let _guard = service.read().await;
    }

    #[tokio::test]
    async fn test_kit_require_rate_limiter() {
        let state = make_app_state().await;
        let _limiter = state.kit.require::<RateLimitModule>().unwrap();
    }

    #[tokio::test]
    async fn test_kit_require_metrics_collector_returns_some() {
        let state = make_app_state().await;
        let collector = state.kit.require::<MetricsCollectorModule>().unwrap();
        assert!(collector.is_some());
    }

    #[tokio::test]
    async fn test_kit_require_prometheus_collector_returns_some() {
        let state = make_app_state().await;
        let collector = state.kit.require::<PrometheusCollectorModule>().unwrap();
        assert!(collector.is_some());
    }

    #[tokio::test]
    async fn test_kit_require_audit_logger_returns_none() {
        let state = make_app_state().await;
        let logger = state.kit.require::<AuditModule>().unwrap();
        assert!(logger.is_none());
    }

    #[tokio::test]
    async fn test_kit_require_ip_whitelist_empty() {
        let state = make_app_state().await;
        let whitelist = state.kit.require::<IpWhitelistModule>().unwrap();
        assert!(whitelist.is_empty());
    }

    #[tokio::test]
    async fn test_kit_require_bool_flags_default_false() {
        let state = make_app_state().await;
        let auth_enabled = state.kit.require::<AuthEnabledModule>().unwrap();
        let rate_limit_enabled = state.kit.require::<RateLimitEnabledModule>().unwrap();
        let pipeline_enabled = state.kit.require::<PipelineEnabledModule>().unwrap();
        assert!(!auth_enabled);
        assert!(!rate_limit_enabled);
        assert!(!pipeline_enabled);
    }

    #[tokio::test]
    async fn test_kit_require_pipeline_components() {
        let state = make_app_state().await;
        let _queue = state.kit.require::<PipelineQueueModule>().unwrap();
        let _channel = state.kit.require::<ResponseChannelModule>().unwrap();
        let _calculator = state.kit.require::<PriorityCalculatorModule>().unwrap();
        let _manager = state.kit.require::<WorkerManagerModule>().unwrap();
    }

    #[tokio::test]
    async fn test_kit_require_cache_and_db_default_false() {
        let state = make_app_state().await;
        let cache_enabled = state.kit.require::<CacheModule>().unwrap();
        let db_enabled = state.kit.require::<DbModule>().unwrap();
        assert!(!cache_enabled);
        assert!(!db_enabled);
    }

    // -------------------------------------------------------------------------
    // FromRef 测试（http feature）
    // -------------------------------------------------------------------------

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_service() {
        let state = make_app_state().await;
        let service: Arc<RwLock<EmbeddingService>> = FromRef::from_ref(&state);
        let kit_service = state.kit.require::<EmbeddingModule>().unwrap();
        assert!(Arc::ptr_eq(&service, &kit_service));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_rate_limiter() {
        let state = make_app_state().await;
        let limiter: Arc<rate_limit::LimiteronAdapter> = FromRef::from_ref(&state);
        let kit_limiter = state.kit.require::<RateLimitModule>().unwrap();
        assert!(Arc::ptr_eq(&limiter, &kit_limiter));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_metrics_collector() {
        let state = make_app_state().await;
        let collector: Arc<metrics::InferenceCollector> = FromRef::from_ref(&state);
        let kit_collector = state.kit.require::<MetricsCollectorModule>().unwrap();
        assert!(Arc::ptr_eq(&collector, kit_collector.as_ref().unwrap()));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_prometheus_collector() {
        let state = make_app_state().await;
        let collector: Arc<metrics::PrometheusCollector> = FromRef::from_ref(&state);
        let kit_collector = state.kit.require::<PrometheusCollectorModule>().unwrap();
        assert!(Arc::ptr_eq(&collector, kit_collector.as_ref().unwrap()));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_audit_logger_returns_none() {
        let state = make_app_state().await;
        let logger: Option<Arc<audit::AuditLogger>> = FromRef::from_ref(&state);
        assert!(logger.is_none());
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_audit_logger_returns_some() {
        let config = audit::AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let logger = Arc::new(audit::AuditLogger::new(config));
        let state = make_app_state_with_options(
            Some(Arc::new(metrics::InferenceCollector::new())),
            Some(Arc::new(
                metrics::PrometheusCollector::new().expect("Failed to create PrometheusCollector"),
            )),
            Some(logger.clone()),
        )
        .await;
        let extracted: Option<Arc<audit::AuditLogger>> = FromRef::from_ref(&state);
        assert!(extracted.is_some());
        assert!(Arc::ptr_eq(&extracted.unwrap(), &logger));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_service_after_clone() {
        let state = make_app_state().await;
        let cloned = state.clone();
        let service: Arc<RwLock<EmbeddingService>> = FromRef::from_ref(&cloned);
        let kit_service = state.kit.require::<EmbeddingModule>().unwrap();
        assert!(Arc::ptr_eq(&service, &kit_service));
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_rate_limiter_after_clone() {
        let state = make_app_state().await;
        let cloned = state.clone();
        let limiter: Arc<rate_limit::LimiteronAdapter> = FromRef::from_ref(&cloned);
        let kit_limiter = state.kit.require::<RateLimitModule>().unwrap();
        assert!(Arc::ptr_eq(&limiter, &kit_limiter));
    }

    // -------------------------------------------------------------------------
    // FromRef panic 测试（None 能力触发 panic）
    // -------------------------------------------------------------------------

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_metrics_collector_panics_when_none() {
        let state = make_app_state_with_options(None, None, None).await;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<metrics::InferenceCollector> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when metrics_collector is None"
        );
    }

    #[cfg(feature = "http")]
    #[tokio::test]
    async fn test_from_ref_prometheus_collector_panics_when_none() {
        let state = make_app_state_with_options(None, None, None).await;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<metrics::PrometheusCollector> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when prometheus_collector is None"
        );
    }

    #[cfg(all(feature = "http", feature = "auth"))]
    #[tokio::test]
    async fn test_from_ref_jwt_manager_panics_when_none() {
        let state = make_app_state().await;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<auth::JwtManager> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when jwt_manager is None"
        );
    }

    #[cfg(all(feature = "http", feature = "auth"))]
    #[tokio::test]
    async fn test_from_ref_user_store_panics_when_none() {
        let state = make_app_state().await;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Arc<auth::UserStore> = FromRef::from_ref(&state);
        }));
        assert!(
            result.is_err(),
            "from_ref should panic when user_store is None"
        );
    }

    // -------------------------------------------------------------------------
    // VecboostState Send + Sync 编译期断言
    // -------------------------------------------------------------------------

    #[test]
    fn test_vecboost_state_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VecboostState>();
    }

    #[test]
    fn test_vecboost_state_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<VecboostState>();
    }
}
