// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! ModuleMeta + AutoBuilder trait 实现

use std::sync::Arc;
use tokio::sync::RwLock;
use trait_kit::prelude::*;

#[cfg(feature = "auth")]
use super::AuthModule;
use super::{
    AuditModule, CacheConfig, CacheModule, DbConfig, DbModule, EmbeddingModule, RateLimitModule,
};
use crate::audit::AuditLogger;
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

impl AutoBuilder for EmbeddingModule {
    type Capability = Arc<RwLock<EmbeddingService>>;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        kit.config::<Self::Capability>()
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
impl AutoBuilder for AuthModule {
    type Capability = Option<Arc<crate::auth::JwtManager>>;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        kit.config::<Self::Capability>()
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

impl AutoBuilder for RateLimitModule {
    type Capability = Arc<LimiteronAdapter>;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        kit.config::<Self::Capability>()
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

impl AutoBuilder for CacheModule {
    type Capability = bool;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        Ok(kit
            .config::<CacheConfig>()
            .map(|c| c.enabled)
            .unwrap_or(false))
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

impl AutoBuilder for DbModule {
    type Capability = bool;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        Ok(kit.config::<DbConfig>().map(|c| c.enabled).unwrap_or(false))
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

impl AutoBuilder for AuditModule {
    type Capability = Option<Arc<AuditLogger>>;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        kit.config::<Self::Capability>()
    }
}
