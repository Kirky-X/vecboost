// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! module_registry 单元测试
//!
//! 基于 trait-kit 0.3 的 `AsyncKit`（`Send + Sync`）。`build()` 是异步的，
//! 测试使用 `#[tokio::test]`。`require` / `config` / `register` 均为同步方法。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;
use trait_kit::AsyncKit;
use trait_kit::prelude::*;

use super::*;
use crate::config::model::{ModelConfig, Precision};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::pipeline::{PriorityConfig, WorkerConfig};
use crate::rate_limit::LimiteronAdapter;
use crate::service::embedding::EmbeddingService;

// ---------------------------------------------------------------------------
// Mock Engine — 用于测试 EmbeddingService 构造
// ---------------------------------------------------------------------------

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

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 循环依赖测试模块
// ---------------------------------------------------------------------------

struct CycleA;
impl ModuleMeta for CycleA {
    const NAME: &'static str = "cycle_a";
    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        static DEPS: &[(&str, std::any::TypeId)] = &[("cycle_b", std::any::TypeId::of::<CycleB>())];
        DEPS
    }
}
impl AsyncAutoBuilder for CycleA {
    type Capability = ();
    type Error = TraitKitError;
    fn build<'a>(
        _kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { Ok(()) })
    }
}

struct CycleB;
impl ModuleMeta for CycleB {
    const NAME: &'static str = "cycle_b";
    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        static DEPS: &[(&str, std::any::TypeId)] = &[("cycle_a", std::any::TypeId::of::<CycleA>())];
        DEPS
    }
}
impl AsyncAutoBuilder for CycleB {
    type Capability = ();
    type Error = TraitKitError;
    fn build<'a>(
        _kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { Ok(()) })
    }
}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

fn make_service() -> Arc<RwLock<EmbeddingService>> {
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(MockEngine));
    Arc::new(RwLock::new(EmbeddingService::new(engine, None)))
}

fn make_rate_limiter() -> Arc<LimiteronAdapter> {
    Arc::new(LimiteronAdapter::with_default_config())
}

// ---------------------------------------------------------------------------
// T013 测试：注册、构建、require（基于 AsyncKit）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_embedding_module_register_and_build() {
    let mut kit = AsyncKit::new();

    let service = make_service();
    kit.set_config(service.clone());
    kit.register::<EmbeddingModule>().unwrap();

    let kit = kit.build().await.unwrap();

    let capability = kit.require::<EmbeddingModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &service));
    assert!(kit.contains::<EmbeddingModule>());
}

#[tokio::test]
async fn test_embedding_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<EmbeddingModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "build should fail when EmbeddingService config is missing"
    );
}

#[tokio::test]
async fn test_rate_limit_module_build() {
    let mut kit = AsyncKit::new();

    let rate_limiter = make_rate_limiter();
    kit.set_config(rate_limiter.clone());
    kit.register::<RateLimitModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<RateLimitModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &rate_limiter));
}

#[tokio::test]
async fn test_cache_module_default_disabled() {
    let mut kit = AsyncKit::new();
    kit.register::<CacheModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<CacheModule>().unwrap();
    assert!(!enabled, "Cache should be disabled by default");
}

#[tokio::test]
async fn test_cache_module_with_config() {
    let mut kit = AsyncKit::new();
    kit.set_config(CacheConfig {
        enabled: true,
        size: 1024,
    });
    kit.register::<CacheModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<CacheModule>().unwrap();
    assert!(enabled, "Cache should be enabled when configured");
}

#[tokio::test]
async fn test_db_module_default_disabled() {
    let mut kit = AsyncKit::new();
    kit.register::<DbModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<DbModule>().unwrap();
    assert!(!enabled, "DB should be disabled by default");
}

#[tokio::test]
async fn test_db_module_with_config() {
    let mut kit = AsyncKit::new();
    kit.set_config(DbConfig { enabled: true });
    kit.register::<DbModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<DbModule>().unwrap();
    assert!(enabled, "DB should be enabled when configured");
}

#[tokio::test]
async fn test_audit_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);
    kit.register::<AuditModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<AuditModule>().unwrap();
    assert!(capability.is_none());
}

#[tokio::test]
async fn test_multiple_modules_build_together() {
    let mut kit = AsyncKit::new();

    let service = make_service();
    let rate_limiter = make_rate_limiter();
    kit.set_config(service.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: true,
        size: 512,
    });
    kit.set_config(DbConfig { enabled: false });
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);

    kit.register::<EmbeddingModule>().unwrap();
    kit.register::<RateLimitModule>().unwrap();
    kit.register::<CacheModule>().unwrap();
    kit.register::<DbModule>().unwrap();
    kit.register::<AuditModule>().unwrap();

    let kit = kit.build().await.unwrap();

    assert!(kit.contains::<EmbeddingModule>());
    assert!(kit.contains::<RateLimitModule>());
    assert!(kit.contains::<CacheModule>());
    assert!(kit.contains::<DbModule>());
    assert!(kit.contains::<AuditModule>());

    let svc = kit.require::<EmbeddingModule>().unwrap();
    assert!(Arc::ptr_eq(&svc, &service));

    let cache_enabled = kit.require::<CacheModule>().unwrap();
    assert!(cache_enabled);

    let db_enabled = kit.require::<DbModule>().unwrap();
    assert!(!db_enabled);
}

#[tokio::test]
async fn test_duplicate_registration_fails() {
    let mut kit = AsyncKit::new();
    kit.set_config(make_service());
    kit.register::<EmbeddingModule>().unwrap();

    let result = kit.register::<EmbeddingModule>();
    assert!(
        result.is_err(),
        "Duplicate registration should return error"
    );
}

// ---------------------------------------------------------------------------
// T013 测试：循环依赖检测
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_circular_dependency_detected() {
    let mut kit = AsyncKit::new();
    kit.register::<CycleA>().unwrap();
    kit.register::<CycleB>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "Circular dependency should be detected and return Err"
    );
}

#[tokio::test]
async fn test_missing_dependency_detected() {
    struct DependentModule;
    impl ModuleMeta for DependentModule {
        const NAME: &'static str = "dependent";
        fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
            static DEPS: &[(&str, std::any::TypeId)] =
                &[("missing_dep", std::any::TypeId::of::<CycleA>())];
            DEPS
        }
    }
    impl AsyncAutoBuilder for DependentModule {
        type Capability = ();
        type Error = TraitKitError;
        fn build<'a>(
            _kit: &'a AsyncKit,
        ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>>
        {
            Box::pin(async move { Ok(()) })
        }
    }

    let mut kit = AsyncKit::new();
    kit.register::<DependentModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "Missing dependency should be detected and return Err"
    );
}

// ---------------------------------------------------------------------------
// AsyncKit Send + Sync 验证（编译期断言）
// ---------------------------------------------------------------------------

#[test]
fn test_async_kit_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AsyncKit>();
    assert_send_sync::<trait_kit::AsyncReady>();
}

#[tokio::test]
async fn test_async_kit_can_be_wrapped_in_arc() {
    let mut kit = AsyncKit::new();
    kit.set_config(make_service());
    kit.register::<EmbeddingModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let kit_arc = Arc::new(kit);

    // 验证 Arc<AsyncKit<Ready>> 可以跨线程克隆，且 require 返回正确的能力
    let kit_clone = Arc::clone(&kit_arc);
    let svc = kit_clone.require::<EmbeddingModule>().unwrap();
    let _guard = svc.read().await;
    // 只需验证能成功获取读写锁即可，不访问私有字段
}

#[tokio::test]
async fn test_rate_limit_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<RateLimitModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "build should fail when RateLimitModule config is missing"
    );
}

#[tokio::test]
async fn test_cache_module_disabled_when_configured_disabled() {
    let mut kit = AsyncKit::new();
    kit.set_config(CacheConfig {
        enabled: false,
        size: 0,
    });
    kit.register::<CacheModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<CacheModule>().unwrap();
    assert!(!enabled, "Cache should be disabled when configured false");
}

#[tokio::test]
async fn test_db_module_disabled_when_configured_disabled() {
    let mut kit = AsyncKit::new();
    kit.set_config(DbConfig { enabled: false });
    kit.register::<DbModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<DbModule>().unwrap();
    assert!(!enabled, "DB should be disabled when configured false");
}

#[tokio::test]
async fn test_audit_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<AuditModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "build should fail when AuditModule config is missing"
    );
}

#[tokio::test]
async fn test_audit_module_with_some_logger() {
    use crate::audit::{AuditConfig, AuditLogger};
    let config = AuditConfig {
        enabled: false,
        ..Default::default()
    };
    let logger = Arc::new(AuditLogger::new(config));
    let mut kit = AsyncKit::new();
    kit.set_config(Some(logger.clone()));
    kit.register::<AuditModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<AuditModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(capability.as_ref().unwrap(), &logger));
}

#[tokio::test]
async fn test_embedding_and_rate_limit_independent_build() {
    let mut kit_a = AsyncKit::new();
    let service_a = make_service();
    kit_a.set_config(service_a.clone());
    kit_a.register::<EmbeddingModule>().unwrap();
    let kit_a = kit_a.build().await.unwrap();

    let mut kit_b = AsyncKit::new();
    let service_b = make_service();
    kit_b.set_config(service_b.clone());
    kit_b.register::<EmbeddingModule>().unwrap();
    let kit_b = kit_b.build().await.unwrap();

    let cap_a = kit_a.require::<EmbeddingModule>().unwrap();
    let cap_b = kit_b.require::<EmbeddingModule>().unwrap();
    assert!(!Arc::ptr_eq(&cap_a, &cap_b));
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_auth_module_with_none_config() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
    kit.register::<AuthModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<AuthModule>().unwrap();
    assert!(capability.is_none());
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_auth_module_with_some_config() {
    use crate::auth::JwtManager;
    let secret = "0123456789abcdef0123456789abcdef01234567".to_string();
    let jwt = Arc::new(JwtManager::new(secret).unwrap());

    let mut kit = AsyncKit::new();
    kit.set_config(Some(jwt.clone()));
    kit.register::<AuthModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<AuthModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(capability.as_ref().unwrap(), &jwt));
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_auth_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<AuthModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "build should fail when AuthModule config is missing"
    );
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_all_modules_build_together_with_auth() {
    let mut kit = AsyncKit::new();

    let service = make_service();
    let rate_limiter = make_rate_limiter();
    kit.set_config(service.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: true,
        size: 256,
    });
    kit.set_config(DbConfig { enabled: true });
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);
    kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);

    kit.register::<EmbeddingModule>().unwrap();
    kit.register::<RateLimitModule>().unwrap();
    kit.register::<CacheModule>().unwrap();
    kit.register::<DbModule>().unwrap();
    kit.register::<AuditModule>().unwrap();
    kit.register::<AuthModule>().unwrap();

    let kit = kit.build().await.unwrap();

    assert!(kit.contains::<EmbeddingModule>());
    assert!(kit.contains::<AuthModule>());
    assert!(kit.contains::<RateLimitModule>());
    assert!(kit.contains::<CacheModule>());
    assert!(kit.contains::<DbModule>());
    assert!(kit.contains::<AuditModule>());

    let auth_cap = kit.require::<AuthModule>().unwrap();
    assert!(auth_cap.is_none());
    let cache_enabled = kit.require::<CacheModule>().unwrap();
    assert!(cache_enabled);
    let db_enabled = kit.require::<DbModule>().unwrap();
    assert!(db_enabled);
}

// ===========================================================================
// v0.3.0 D3 重构：13 个新 Module 测试
//
// 测试覆盖矩阵：
//   - 严格类型（Arc<T>、Option<Arc<T>>、Vec<String>）：注册→构建→require 返回原对象
//   - 严格类型 missing config：build 失败
//   - bool 类型（newtype 派生）：注册→构建→require 返回 bool 值
//   - bool 类型 missing config：默认 false（与 CacheModule/DbModule 一致）
// ===========================================================================

// ---------------------------------------------------------------------------
// 辅助函数 — 新 Module 所需
// ---------------------------------------------------------------------------

fn make_metrics_collector() -> Option<Arc<crate::metrics::InferenceCollector>> {
    Some(Arc::new(crate::metrics::InferenceCollector::new()))
}

fn make_prometheus_collector() -> Option<Arc<crate::metrics::PrometheusCollector>> {
    Some(Arc::new(
        crate::metrics::PrometheusCollector::new().expect("Failed to create PrometheusCollector"),
    ))
}

fn make_pipeline_queue() -> Arc<crate::pipeline::PriorityRequestQueue> {
    Arc::new(crate::pipeline::PriorityRequestQueue::new(100))
}

fn make_response_channel() -> Arc<crate::pipeline::ResponseChannel> {
    Arc::new(crate::pipeline::ResponseChannel::new())
}

fn make_priority_calculator() -> Arc<crate::pipeline::PriorityCalculator> {
    Arc::new(crate::pipeline::PriorityCalculator::new(
        PriorityConfig::default(),
    ))
}

fn make_worker_manager() -> Arc<crate::pipeline::WorkerManager> {
    let queue = make_pipeline_queue();
    let channel = make_response_channel();
    Arc::new(crate::pipeline::WorkerManager::new(
        queue,
        channel,
        WorkerConfig::default(),
        make_service(),
    ))
}

// ---------------------------------------------------------------------------
// AuthEnabledModule (bool from AuthEnabled newtype)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_auth_enabled_module_with_true() {
    let mut kit = AsyncKit::new();
    kit.set_config(AuthEnabled(true));
    kit.register::<AuthEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<AuthEnabledModule>().unwrap();
    assert!(enabled, "AuthEnabledModule should return true");
}

#[tokio::test]
async fn test_auth_enabled_module_with_false() {
    let mut kit = AsyncKit::new();
    kit.set_config(AuthEnabled(false));
    kit.register::<AuthEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<AuthEnabledModule>().unwrap();
    assert!(!enabled, "AuthEnabledModule should return false");
}

#[tokio::test]
async fn test_auth_enabled_module_defaults_false_when_config_missing() {
    // 与 CacheModule/DbModule 一致：missing config 默认 false
    let mut kit = AsyncKit::new();
    kit.register::<AuthEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<AuthEnabledModule>().unwrap();
    assert!(!enabled, "AuthEnabledModule should default to false");
}

// ---------------------------------------------------------------------------
// RateLimitEnabledModule (bool from RateLimitEnabled newtype)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_rate_limit_enabled_module_with_true() {
    let mut kit = AsyncKit::new();
    kit.set_config(RateLimitEnabled(true));
    kit.register::<RateLimitEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<RateLimitEnabledModule>().unwrap();
    assert!(enabled);
}

#[tokio::test]
async fn test_rate_limit_enabled_module_defaults_false_when_config_missing() {
    let mut kit = AsyncKit::new();
    kit.register::<RateLimitEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<RateLimitEnabledModule>().unwrap();
    assert!(!enabled);
}

// ---------------------------------------------------------------------------
// PipelineEnabledModule (bool from PipelineEnabled newtype)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_pipeline_enabled_module_with_true() {
    let mut kit = AsyncKit::new();
    kit.set_config(PipelineEnabled(true));
    kit.register::<PipelineEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<PipelineEnabledModule>().unwrap();
    assert!(enabled);
}

#[tokio::test]
async fn test_pipeline_enabled_module_defaults_false_when_config_missing() {
    let mut kit = AsyncKit::new();
    kit.register::<PipelineEnabledModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let enabled = kit.require::<PipelineEnabledModule>().unwrap();
    assert!(!enabled);
}

// ---------------------------------------------------------------------------
// MetricsCollectorModule (Option<Arc<InferenceCollector>>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_metrics_collector_module_with_some() {
    let collector = make_metrics_collector();
    let mut kit = AsyncKit::new();
    kit.set_config(collector.clone());
    kit.register::<MetricsCollectorModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<MetricsCollectorModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(
        capability.as_ref().unwrap(),
        collector.as_ref().unwrap()
    ));
}

#[tokio::test]
async fn test_metrics_collector_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::metrics::InferenceCollector>>::None);
    kit.register::<MetricsCollectorModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<MetricsCollectorModule>().unwrap();
    assert!(capability.is_none());
}

#[tokio::test]
async fn test_metrics_collector_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<MetricsCollectorModule>().unwrap();

    let result = kit.build().await;
    assert!(
        result.is_err(),
        "build should fail when MetricsCollectorModule config is missing"
    );
}

// ---------------------------------------------------------------------------
// PrometheusCollectorModule (Option<Arc<PrometheusCollector>>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_prometheus_collector_module_with_some() {
    let collector = make_prometheus_collector();
    let mut kit = AsyncKit::new();
    kit.set_config(collector.clone());
    kit.register::<PrometheusCollectorModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<PrometheusCollectorModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(
        capability.as_ref().unwrap(),
        collector.as_ref().unwrap()
    ));
}

#[tokio::test]
async fn test_prometheus_collector_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::metrics::PrometheusCollector>>::None);
    kit.register::<PrometheusCollectorModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<PrometheusCollectorModule>().unwrap();
    assert!(capability.is_none());
}

#[tokio::test]
async fn test_prometheus_collector_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<PrometheusCollectorModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// IpWhitelistModule (Vec<String>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_ip_whitelist_module_with_list() {
    let whitelist = vec!["127.0.0.1".to_string(), "10.0.0.0/8".to_string()];
    let mut kit = AsyncKit::new();
    kit.set_config(whitelist.clone());
    kit.register::<IpWhitelistModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<IpWhitelistModule>().unwrap();
    assert_eq!(capability, whitelist);
}

#[tokio::test]
async fn test_ip_whitelist_module_with_empty_list() {
    let mut kit = AsyncKit::new();
    kit.set_config(Vec::<String>::new());
    kit.register::<IpWhitelistModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<IpWhitelistModule>().unwrap();
    assert!(capability.is_empty());
}

#[tokio::test]
async fn test_ip_whitelist_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<IpWhitelistModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// PipelineQueueModule (Arc<PriorityRequestQueue>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_pipeline_queue_module_register_and_build() {
    let queue = make_pipeline_queue();
    let mut kit = AsyncKit::new();
    kit.set_config(queue.clone());
    kit.register::<PipelineQueueModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<PipelineQueueModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &queue));
}

#[tokio::test]
async fn test_pipeline_queue_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<PipelineQueueModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// ResponseChannelModule (Arc<ResponseChannel>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_response_channel_module_register_and_build() {
    let channel = make_response_channel();
    let mut kit = AsyncKit::new();
    kit.set_config(channel.clone());
    kit.register::<ResponseChannelModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<ResponseChannelModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &channel));
}

#[tokio::test]
async fn test_response_channel_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<ResponseChannelModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// PriorityCalculatorModule (Arc<PriorityCalculator>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_priority_calculator_module_register_and_build() {
    let calc = make_priority_calculator();
    let mut kit = AsyncKit::new();
    kit.set_config(calc.clone());
    kit.register::<PriorityCalculatorModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<PriorityCalculatorModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &calc));
}

#[tokio::test]
async fn test_priority_calculator_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<PriorityCalculatorModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// WorkerManagerModule (Arc<WorkerManager>)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_worker_manager_module_register_and_build() {
    let mgr = make_worker_manager();
    let mut kit = AsyncKit::new();
    kit.set_config(mgr.clone());
    kit.register::<WorkerManagerModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<WorkerManagerModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &mgr));
}

#[tokio::test]
async fn test_worker_manager_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<WorkerManagerModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Auth-gated modules (UserStoreModule, CsrfConfigModule, CsrfTokenStoreModule)
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_user_store_module_with_some() {
    use crate::auth::UserStore;
    let pool = crate::db::DbPool::new("sqlite::memory:").await.unwrap();
    let store = Arc::new(UserStore::new(Arc::new(pool)));
    let mut kit = AsyncKit::new();
    kit.set_config(Some(store.clone()));
    kit.register::<UserStoreModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<UserStoreModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(capability.as_ref().unwrap(), &store));
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_user_store_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
    kit.register::<UserStoreModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<UserStoreModule>().unwrap();
    assert!(capability.is_none());
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_user_store_module_missing_config_fails() {
    let mut kit = AsyncKit::new();
    kit.register::<UserStoreModule>().unwrap();

    let result = kit.build().await;
    assert!(result.is_err());
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_csrf_config_module_with_some() {
    use crate::auth::CsrfConfig;
    let config = Arc::new(CsrfConfig::new(vec!["https://example.com".to_string()]));
    let mut kit = AsyncKit::new();
    kit.set_config(Some(config.clone()));
    kit.register::<CsrfConfigModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<CsrfConfigModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(capability.as_ref().unwrap(), &config));
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_csrf_config_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
    kit.register::<CsrfConfigModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<CsrfConfigModule>().unwrap();
    assert!(capability.is_none());
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_csrf_token_store_module_with_some() {
    use crate::auth::CsrfTokenStore;
    let store = Arc::new(CsrfTokenStore::new());
    let mut kit = AsyncKit::new();
    kit.set_config(Some(store.clone()));
    kit.register::<CsrfTokenStoreModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<CsrfTokenStoreModule>().unwrap();
    assert!(capability.is_some());
    assert!(Arc::ptr_eq(capability.as_ref().unwrap(), &store));
}

#[cfg(feature = "auth")]
#[tokio::test]
async fn test_csrf_token_store_module_with_none() {
    let mut kit = AsyncKit::new();
    kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
    kit.register::<CsrfTokenStoreModule>().unwrap();

    let kit = kit.build().await.unwrap();
    let capability = kit.require::<CsrfTokenStoreModule>().unwrap();
    assert!(capability.is_none());
}

// ---------------------------------------------------------------------------
// 集成测试：所有 17 个 Module 联合构建（验证无 TypeId 冲突）
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_all_seventeen_modules_build_together() {
    let mut kit = AsyncKit::new();

    // 4 个现有 Module
    let service = make_service();
    let rate_limiter = make_rate_limiter();
    kit.set_config(service.clone());
    kit.set_config(rate_limiter.clone());
    kit.set_config(CacheConfig {
        enabled: true,
        size: 256,
    });
    kit.set_config(DbConfig { enabled: false });
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);

    // 13 个新 Module 的 config
    kit.set_config(AuthEnabled(true));
    kit.set_config(RateLimitEnabled(true));
    kit.set_config(PipelineEnabled(false));
    kit.set_config(make_metrics_collector());
    kit.set_config(make_prometheus_collector());
    kit.set_config(vec!["127.0.0.1".to_string()]);
    kit.set_config(make_pipeline_queue());
    kit.set_config(make_response_channel());
    kit.set_config(make_priority_calculator());
    kit.set_config(make_worker_manager());

    #[cfg(feature = "auth")]
    {
        kit.set_config(Option::<Arc<crate::auth::JwtManager>>::None);
        kit.set_config(Option::<Arc<crate::auth::UserStore>>::None);
        kit.set_config(Option::<Arc<crate::auth::CsrfConfig>>::None);
        kit.set_config(Option::<Arc<crate::auth::CsrfTokenStore>>::None);
    }

    // 注册全部
    kit.register::<EmbeddingModule>().unwrap();
    kit.register::<AuthEnabledModule>().unwrap();
    kit.register::<RateLimitEnabledModule>().unwrap();
    kit.register::<PipelineEnabledModule>().unwrap();
    kit.register::<MetricsCollectorModule>().unwrap();
    kit.register::<PrometheusCollectorModule>().unwrap();
    kit.register::<IpWhitelistModule>().unwrap();
    kit.register::<PipelineQueueModule>().unwrap();
    kit.register::<ResponseChannelModule>().unwrap();
    kit.register::<PriorityCalculatorModule>().unwrap();
    kit.register::<WorkerManagerModule>().unwrap();
    kit.register::<CacheModule>().unwrap();
    kit.register::<DbModule>().unwrap();
    kit.register::<AuditModule>().unwrap();
    kit.register::<RateLimitModule>().unwrap();
    #[cfg(feature = "auth")]
    {
        kit.register::<AuthModule>().unwrap();
        kit.register::<UserStoreModule>().unwrap();
        kit.register::<CsrfConfigModule>().unwrap();
        kit.register::<CsrfTokenStoreModule>().unwrap();
    }

    let kit = kit.build().await.unwrap();

    // 验证全部 capability 可检索
    assert!(kit.contains::<EmbeddingModule>());
    assert!(kit.contains::<AuthEnabledModule>());
    assert!(kit.contains::<RateLimitEnabledModule>());
    assert!(kit.contains::<PipelineEnabledModule>());
    assert!(kit.contains::<MetricsCollectorModule>());
    assert!(kit.contains::<PrometheusCollectorModule>());
    assert!(kit.contains::<IpWhitelistModule>());
    assert!(kit.contains::<PipelineQueueModule>());
    assert!(kit.contains::<ResponseChannelModule>());
    assert!(kit.contains::<PriorityCalculatorModule>());
    assert!(kit.contains::<WorkerManagerModule>());
    assert!(kit.contains::<CacheModule>());
    assert!(kit.contains::<DbModule>());
    assert!(kit.contains::<AuditModule>());
    assert!(kit.contains::<RateLimitModule>());
    #[cfg(feature = "auth")]
    {
        assert!(kit.contains::<AuthModule>());
        assert!(kit.contains::<UserStoreModule>());
        assert!(kit.contains::<CsrfConfigModule>());
        assert!(kit.contains::<CsrfTokenStoreModule>());
    }

    // 验证部分 capability 返回正确值
    assert!(kit.require::<AuthEnabledModule>().unwrap());
    assert!(kit.require::<RateLimitEnabledModule>().unwrap());
    assert!(!kit.require::<PipelineEnabledModule>().unwrap());
    let ip_list = kit.require::<IpWhitelistModule>().unwrap();
    assert_eq!(ip_list, vec!["127.0.0.1".to_string()]);
}
