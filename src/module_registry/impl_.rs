// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! ModuleMeta + AsyncAutoBuilder trait 实现
//!
//! 基于 trait-kit 0.3 的 `AsyncKit`（`Send + Sync`，基于 `Arc<RwLock>`）。
//! 所有模块实现 `AsyncAutoBuilder`，`build` 返回 `Pin<Box<dyn Future>>`，
//! 在 `AsyncKit::build().await` 时按拓扑序异步构造。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use trait_kit::AsyncKit;
use trait_kit::prelude::*;

#[cfg(feature = "auth")]
use super::AuthModule;
#[cfg(feature = "http")]
use super::PrometheusCollectorModule;
use super::RateLimitModule;
use super::{
    AuditModule, CacheConfig, CacheModule, ConfigWatcherModule,
    DbConfig, DbModule, EmbeddingModule, IpWhitelistModule, MetricsCollectorModule,
    PipelineQueueModule, PriorityCalculatorModule, RerankModule,
    ResponseChannelModule, WorkerManagerModule,
};
#[cfg(feature = "auth")]
use super::{CsrfConfigModule};
use crate::audit::AuditLogger;
#[cfg(feature = "auth")]
use crate::auth::GarrisonHandle;
#[cfg(feature = "http")]
use crate::metrics::PrometheusCollector;
use crate::rate_limit::LimiteronAdapter;
use crate::service::embedding::EmbeddingService;
use crate::service::rerank::RerankService;
use crate::{
    metrics::InferenceCollector,
    pipeline::{PriorityCalculator, PriorityRequestQueue, ResponseChannel, WorkerManager},
};

// ---------------------------------------------------------------------------
// EmbeddingModule
// ---------------------------------------------------------------------------

impl ModuleMeta for EmbeddingModule {
    const NAME: &'static str = "embedding";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for EmbeddingModule {
    type Capability = Arc<RwLock<EmbeddingService>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// EmbeddingModule — lifecycle + health (Phase 3)
// ---------------------------------------------------------------------------

impl AsyncLifecycle for EmbeddingModule {
    fn on_ready<'a>(
        _kit: &'a AsyncKit<trait_kit::AsyncReady>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + 'a>> {
        Box::pin(async {
            log::info!("EmbeddingModule: model engine ready");
            Ok(())
        })
    }
}

impl AsyncHealthCheck for EmbeddingModule {
    fn check(_cap: &Self::Capability) -> HealthStatus {
        // EmbeddingService is always operational after successful build
        HealthStatus::Healthy
    }
}

// ---------------------------------------------------------------------------
// RerankModule
// ---------------------------------------------------------------------------

impl ModuleMeta for RerankModule {
    const NAME: &'static str = "rerank";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for RerankModule {
    type Capability = Arc<RwLock<RerankService>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

impl AsyncLifecycle for RerankModule {
    fn on_ready<'a>(
        _kit: &'a AsyncKit<trait_kit::AsyncReady>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + 'a>> {
        Box::pin(async {
            log::info!("RerankModule: rerank service ready");
            Ok(())
        })
    }
}

impl AsyncHealthCheck for RerankModule {
    fn check(_cap: &Self::Capability) -> HealthStatus {
        HealthStatus::Healthy
    }
}

// ---------------------------------------------------------------------------
// AuthModule (auth feature only)
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
impl ModuleMeta for AuthModule {
    const NAME: &'static str = "auth";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

#[cfg(feature = "auth")]
impl AsyncAutoBuilder for AuthModule {
    type Capability = Option<Arc<GarrisonHandle>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// RateLimitModule
// ---------------------------------------------------------------------------

impl ModuleMeta for RateLimitModule {
    const NAME: &'static str = "rate_limit";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for RateLimitModule {
    type Capability = Arc<LimiteronAdapter>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// RateLimitModule — health + lifecycle (Phase 3)
// ---------------------------------------------------------------------------

impl AsyncHealthCheck for RateLimitModule {
    fn check(_cap: &Self::Capability) -> HealthStatus {
        // LimiteronAdapter is always operational after construction
        HealthStatus::Healthy
    }
}

impl AsyncLifecycle for RateLimitModule {
    fn on_ready<'a>(
        _kit: &'a AsyncKit<trait_kit::AsyncReady>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + 'a>> {
        Box::pin(async {
            log::info!("RateLimitModule: on_ready — limiteron rate limiter active");
            Ok(())
        })
    }

    fn on_shutdown<'a>(
        _cap: &'a Self::Capability,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async {
            log::info!("RateLimitModule: on_shutdown — rate limiter shut down");
        })
    }
}

// ---------------------------------------------------------------------------
// CacheModule
// ---------------------------------------------------------------------------

impl ModuleMeta for CacheModule {
    const NAME: &'static str = "cache";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for CacheModule {
    type Capability = bool;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move {
            Ok(kit
                .config::<CacheConfig>()
                .map(|c| c.enabled)
                .unwrap_or(false))
        })
    }
}

// ---------------------------------------------------------------------------
// CacheModule — health (Phase 3)
// ---------------------------------------------------------------------------

impl AsyncHealthCheck for CacheModule {
    fn check(cap: &Self::Capability) -> HealthStatus {
        if *cap {
            HealthStatus::Healthy
        } else {
            HealthStatus::Degraded {
                detail: "cache disabled".into(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DbModule
// ---------------------------------------------------------------------------

impl ModuleMeta for DbModule {
    const NAME: &'static str = "db";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for DbModule {
    type Capability = bool;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { Ok(kit.config::<DbConfig>().map(|c| c.enabled).unwrap_or(false)) })
    }
}

// ---------------------------------------------------------------------------
// AuditModule
// ---------------------------------------------------------------------------

impl ModuleMeta for AuditModule {
    const NAME: &'static str = "audit";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for AuditModule {
    type Capability = Option<Arc<AuditLogger>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// AuditModule — lifecycle (Phase 3)
// ---------------------------------------------------------------------------

impl AsyncLifecycle for AuditModule {
    fn on_shutdown<'a>(
        cap: &'a Self::Capability,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            if let Some(logger) = cap {
                log::info!("AuditModule: flushing audit log before shutdown");
                if let Err(e) = logger.flush().await {
                    log::error!("AuditModule: failed to flush audit log: {}", e);
                }
            }
        })
    }
}

// ===========================================================================
// v0.3.0 D3 重构：覆盖 VecboostState 剩余字段的 13 个 Module 实现
//
// 设计原则与现有 6 个 Module 保持一致：
//   - 复杂类型（Arc<T>、Option<Arc<T>>、Vec<String>）→ `kit.config::<Self::Capability>()`
//     严格模式：missing config = TraitKitError::MissingConfig
//   - bool 类型 → `kit.config::<Newtype>().map(|c| c.0).unwrap_or(false)`
//     宽松模式（与 CacheModule/DbModule 一致）：missing config 默认 false
// ===========================================================================

// ---------------------------------------------------------------------------
// CsrfConfigModule (auth feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "auth")]
impl ModuleMeta for CsrfConfigModule {
    const NAME: &'static str = "csrf_config";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

#[cfg(feature = "auth")]
impl AsyncAutoBuilder for CsrfConfigModule {
    type Capability = Option<Arc<crate::auth::GarrisonCsrfConfig>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// MetricsCollectorModule
// ---------------------------------------------------------------------------

impl ModuleMeta for MetricsCollectorModule {
    const NAME: &'static str = "metrics_collector";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for MetricsCollectorModule {
    type Capability = Option<Arc<InferenceCollector>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// PrometheusCollectorModule（仅 http feature）
// ---------------------------------------------------------------------------

#[cfg(feature = "http")]
impl ModuleMeta for PrometheusCollectorModule {
    const NAME: &'static str = "prometheus_collector";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

#[cfg(feature = "http")]
impl AsyncAutoBuilder for PrometheusCollectorModule {
    type Capability = Option<Arc<PrometheusCollector>>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// IpWhitelistModule
// ---------------------------------------------------------------------------

impl ModuleMeta for IpWhitelistModule {
    const NAME: &'static str = "ip_whitelist";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for IpWhitelistModule {
    type Capability = Vec<String>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// PipelineQueueModule
// ---------------------------------------------------------------------------

impl ModuleMeta for PipelineQueueModule {
    const NAME: &'static str = "pipeline_queue";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for PipelineQueueModule {
    type Capability = Arc<PriorityRequestQueue>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// ResponseChannelModule
// ---------------------------------------------------------------------------

impl ModuleMeta for ResponseChannelModule {
    const NAME: &'static str = "response_channel";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for ResponseChannelModule {
    type Capability = Arc<ResponseChannel>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// PriorityCalculatorModule
// ---------------------------------------------------------------------------

impl ModuleMeta for PriorityCalculatorModule {
    const NAME: &'static str = "priority_calculator";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for PriorityCalculatorModule {
    type Capability = Arc<PriorityCalculator>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// WorkerManagerModule
// ---------------------------------------------------------------------------

impl ModuleMeta for WorkerManagerModule {
    const NAME: &'static str = "worker_manager";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for WorkerManagerModule {
    type Capability = Arc<WorkerManager>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

// ---------------------------------------------------------------------------
// ConfigWatcherModule — confers watch 集成 (Phase 8)
// ---------------------------------------------------------------------------

impl ModuleMeta for ConfigWatcherModule {
    const NAME: &'static str = "config_watcher";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for ConfigWatcherModule {
    type Capability = Arc<confers::watcher::WatcherGuard>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
    }
}

impl AsyncLifecycle for ConfigWatcherModule {
    fn on_ready<'a>(
        _kit: &'a AsyncKit<trait_kit::AsyncReady>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + 'a>> {
        Box::pin(async {
            log::info!("ConfigWatcherModule: config file watcher ready");
            Ok(())
        })
    }

    fn on_shutdown<'a>(
        cap: &'a Self::Capability,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            log::info!("ConfigWatcherModule: shutting down config watcher");
            if let Err(e) = cap.shutdown(std::time::Duration::from_secs(5)).await {
                log::error!("ConfigWatcherModule: error during watcher shutdown: {}", e);
            }
        })
    }
}
