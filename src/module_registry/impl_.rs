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
#[cfg(feature = "limiteron")]
use super::RateLimitModule;
use super::{AuditModule, CacheConfig, CacheModule, DbConfig, DbModule, EmbeddingModule};
use crate::audit::AuditLogger;
#[cfg(feature = "limiteron")]
use crate::rate_limit::LimiteronAdapter;
use crate::service::embedding::EmbeddingService;

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
    type Capability = Option<Arc<crate::auth::JwtManager>>;
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
