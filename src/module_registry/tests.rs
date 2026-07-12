// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! module_registry 单元测试
//!
//! 注意：trait-kit 基于 RefCell，!Sync，测试在单线程中运行，使用 #[test] 而非 #[tokio::test]。

use std::sync::Arc;
use tokio::sync::RwLock;

use async_trait::async_trait;
use trait_kit::prelude::*;

use super::*;
use crate::config::model::{ModelConfig, Precision};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::rate_limit::{MemoryRateLimitStore, RateLimiter};
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
impl AutoBuilder for CycleA {
    type Capability = ();
    type Error = TraitKitError;
    fn build(_kit: &Kit) -> Result<Self::Capability, Self::Error> {
        Ok(())
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
impl AutoBuilder for CycleB {
    type Capability = ();
    type Error = TraitKitError;
    fn build(_kit: &Kit) -> Result<Self::Capability, Self::Error> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

fn make_service() -> Arc<RwLock<EmbeddingService>> {
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(MockEngine));
    Arc::new(RwLock::new(EmbeddingService::new(engine, None)))
}

fn make_rate_limiter() -> Arc<RateLimiter> {
    Arc::new(RateLimiter::new(Arc::new(MemoryRateLimitStore::new())))
}

// ---------------------------------------------------------------------------
// T013 测试：注册、构建、require
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_module_register_and_build() {
    let mut kit = Kit::new();

    let service = make_service();
    kit.set_config(service.clone());
    kit.register::<EmbeddingModule>().unwrap();

    let kit = kit.build().unwrap();

    let capability = kit.require::<EmbeddingModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &service));
    assert!(kit.contains::<EmbeddingModule>());
}

#[test]
fn test_embedding_module_missing_config_fails() {
    let mut kit = Kit::new();
    kit.register::<EmbeddingModule>().unwrap();

    let result = kit.build();
    assert!(
        result.is_err(),
        "build should fail when EmbeddingService config is missing"
    );
}

#[test]
fn test_rate_limit_module_build() {
    let mut kit = Kit::new();

    let rate_limiter = make_rate_limiter();
    kit.set_config(rate_limiter.clone());
    kit.register::<RateLimitModule>().unwrap();

    let kit = kit.build().unwrap();
    let capability = kit.require::<RateLimitModule>().unwrap();
    assert!(Arc::ptr_eq(&capability, &rate_limiter));
}

#[test]
fn test_cache_module_default_disabled() {
    let mut kit = Kit::new();
    kit.register::<CacheModule>().unwrap();

    let kit = kit.build().unwrap();
    let enabled = kit.require::<CacheModule>().unwrap();
    assert!(!enabled, "Cache should be disabled by default");
}

#[test]
fn test_cache_module_with_config() {
    let mut kit = Kit::new();
    kit.set_config(CacheConfig {
        enabled: true,
        size: 1024,
    });
    kit.register::<CacheModule>().unwrap();

    let kit = kit.build().unwrap();
    let enabled = kit.require::<CacheModule>().unwrap();
    assert!(enabled, "Cache should be enabled when configured");
}

#[test]
fn test_db_module_default_disabled() {
    let mut kit = Kit::new();
    kit.register::<DbModule>().unwrap();

    let kit = kit.build().unwrap();
    let enabled = kit.require::<DbModule>().unwrap();
    assert!(!enabled, "DB should be disabled by default");
}

#[test]
fn test_db_module_with_config() {
    let mut kit = Kit::new();
    kit.set_config(DbConfig { enabled: true });
    kit.register::<DbModule>().unwrap();

    let kit = kit.build().unwrap();
    let enabled = kit.require::<DbModule>().unwrap();
    assert!(enabled, "DB should be enabled when configured");
}

#[test]
fn test_audit_module_with_none() {
    let mut kit = Kit::new();
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);
    kit.register::<AuditModule>().unwrap();

    let kit = kit.build().unwrap();
    let capability = kit.require::<AuditModule>().unwrap();
    assert!(capability.is_none());
}

#[test]
fn test_multiple_modules_build_together() {
    let mut kit = Kit::new();

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

    let kit = kit.build().unwrap();

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

#[test]
fn test_duplicate_registration_fails() {
    let mut kit = Kit::new();
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

#[test]
fn test_circular_dependency_detected() {
    let mut kit = Kit::new();
    kit.register::<CycleA>().unwrap();
    kit.register::<CycleB>().unwrap();

    let result = kit.build();
    assert!(
        result.is_err(),
        "Circular dependency should be detected and return Err"
    );
}

#[test]
fn test_missing_dependency_detected() {
    struct DependentModule;
    impl ModuleMeta for DependentModule {
        const NAME: &'static str = "dependent";
        fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
            static DEPS: &[(&str, std::any::TypeId)] =
                &[("missing_dep", std::any::TypeId::of::<CycleA>())];
            DEPS
        }
    }
    impl AutoBuilder for DependentModule {
        type Capability = ();
        type Error = TraitKitError;
        fn build(_kit: &Kit) -> Result<Self::Capability, Self::Error> {
            Ok(())
        }
    }

    let mut kit = Kit::new();
    kit.register::<DependentModule>().unwrap();

    let result = kit.build();
    assert!(
        result.is_err(),
        "Missing dependency should be detected and return Err"
    );
}
