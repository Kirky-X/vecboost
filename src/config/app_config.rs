// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! confers-based configuration loader for VecBoost.
//!
//! This module provides `AppConfig` with `#[derive(Config)]` from the `confers`
//! crate, enabling TOML file loading and environment variable overrides via a
//! unified builder API. The legacy loader in `app.rs` is retained for one
//! release cycle; new code should prefer `load_via_confers` once the `config`
//! feature is stabilized.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use confers::Config;

use super::app::{
    AuditConfig, AuthConfig, EmbeddingConfig, MemoryPoolConfig, ModelConfig, MonitoringConfig,
    RateLimitConfig, ServerConfig,
};
use crate::pipeline::PipelineConfig;

/// Top-level VecBoost configuration loaded via the `confers` crate.
///
/// Field types reuse the existing sub-configuration structs from `app.rs` so
/// that downstream consumers (services, middleware) are unaffected by the
/// loader swap. The `env_prefix` attribute maps environment variables such as
/// `VECBOOST_JWT_SECRET` onto nested fields using confers' standard separator
/// rules.
///
/// Note on `sensitive` fields: confers 0.4 requires `SecretString` /
/// `SecretBytes` types for `#[config(sensitive = true)]`, which is
/// incompatible with the existing `AuthConfig` composite struct defined in
/// `app.rs`. Sensitive-field redaction is therefore deferred until the
/// sub-structs are migrated to secrecy wrappers in a follow-up task.
#[derive(Config, Debug, Clone, Serialize, Deserialize)]
#[config(env_prefix = "VECBOOST_")]
#[serde(default)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub embedding: EmbeddingConfig,
    pub monitoring: MonitoringConfig,
    pub auth: AuthConfig,
    pub rate_limit: RateLimitConfig,
    pub audit: AuditConfig,
    pub memory_pool: MemoryPoolConfig,
    pub pipeline: PipelineConfig,
}

impl AppConfig {
    /// Load configuration via confers using the default `config.toml` path.
    ///
    /// Mirrors the behaviour of `app::AppConfig::load()` while delegating
    /// source merging (file + env) to `confers::ConfigBuilder`.
    pub fn load_via_confers() -> Result<Self, confers::ConfigError> {
        Self::load_via_confers_with_path("config.toml")
    }

    /// Load configuration via confers from an explicit path.
    ///
    /// The path is treated as optional: when the file does not exist, confers
    /// falls back to defaults derived from `Default` plus environment
    /// variables prefixed with `VECBOOST_`.
    pub fn load_via_confers_with_path<P: Into<PathBuf>>(
        path: P,
    ) -> Result<Self, confers::ConfigError> {
        let mut config = confers::ConfigBuilder::<Self>::new()
            .allow_absolute_paths()
            .file_optional(path)
            .env_prefix("VECBOOST_")
            .build()?;
        apply_security_env_overrides(&mut config);
        Ok(config)
    }
}

/// Apply environment variable overrides for sensitive configuration.
///
/// Mirrors the legacy `app::apply_security_env_overrides` behaviour: the
/// flat env var `VECBOOST_JWT_SECRET` is mapped onto `auth.jwt_secret`
/// because confers' nested env mapping would require `VECBOOST_AUTH__JWT_SECRET`.
fn apply_security_env_overrides(config: &mut AppConfig) {
    if let Ok(jwt_secret) = std::env::var("VECBOOST_JWT_SECRET") {
        if !jwt_secret.is_empty() {
            config.auth.jwt_secret = Some(jwt_secret);
        }
    }
    if let Ok(admin_password) = std::env::var("VECBOOST_ADMIN_PASSWORD") {
        if !admin_password.is_empty() {
            config.auth.default_admin_password = Some(admin_password);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;

    #[test]
    fn test_confers_app_config_compiles() {
        let _ = AppConfig::default();
    }
}
