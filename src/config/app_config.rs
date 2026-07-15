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

#[cfg(feature = "db")]
use super::app::DatabaseConfig;
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
    #[cfg(feature = "db")]
    pub database: DatabaseConfig,
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
        super::app::apply_priority_defaults(&mut config.pipeline.priority);
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
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_confers_app_config_compiles() {
        let _ = AppConfig::default();
    }

    #[test]
    fn test_app_config_default_values() {
        let config = AppConfig::default();
        assert!(!config.server.grpc_enabled);
        assert!(config.model.batch_size > 0);
        assert!(config.embedding.max_batch_size > 0);
    }

    #[test]
    fn test_load_via_confers_with_nonexistent_path() {
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        assert!(result.is_ok(), "should fall back to defaults");
        let config = result.unwrap();
        assert!(!config.server.grpc_enabled);
    }

    #[test]
    fn test_load_via_confers_with_valid_toml() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("test_config.toml");
        std::fs::write(
            &config_path,
            r#"
[server]
host = "0.0.0.0"
port = 8080
grpc_enabled = true

[model]
model_repo = "test/repo"
batch_size = 64

[embedding]
max_batch_size = 128
"#,
        )
        .expect("Failed to write config");

        let result = AppConfig::load_via_confers_with_path(&config_path);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.host, "0.0.0.0");
        assert!(config.server.grpc_enabled);
        assert_eq!(config.model.batch_size, 64);
        assert_eq!(config.embedding.max_batch_size, 128);
    }

    #[test]
    fn test_load_via_confers_with_empty_toml() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("empty.toml");
        std::fs::write(&config_path, "").expect("Failed to write empty config");

        let result = AppConfig::load_via_confers_with_path(&config_path);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert!(!config.server.grpc_enabled);
    }

    #[test]
    fn test_load_via_confers_jwt_secret_env_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "test_secret_12345");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed");
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert_eq!(config.auth.jwt_secret.as_deref(), Some("test_secret_12345"));
    }

    #[test]
    fn test_load_via_confers_empty_jwt_secret_ignored() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed");
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(config.auth.jwt_secret.is_none() || config.auth.jwt_secret.as_deref() == Some(""));
    }

    #[test]
    fn test_load_via_confers_admin_password_env_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "admin_pass_123");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed");
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert_eq!(
            config.auth.default_admin_password.as_deref(),
            Some("admin_pass_123")
        );
    }

    #[test]
    fn test_load_via_confers_empty_admin_password_ignored() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed");
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(
            config.auth.default_admin_password.is_none()
                || config.auth.default_admin_password.as_deref() == Some("")
        );
    }

    #[test]
    fn test_load_via_confers_with_partial_toml() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let config_path = temp_dir.path().join("partial.toml");
        std::fs::write(
            &config_path,
            r#"
[server]
port = 9999
"#,
        )
        .expect("Failed to write partial config");

        let result = AppConfig::load_via_confers_with_path(&config_path);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.server.port, 9999);
    }

    #[test]
    fn test_app_config_clone() {
        let config = AppConfig::default();
        let cloned = config.clone();
        assert_eq!(config.server.port, cloned.server.port);
    }

    #[test]
    fn test_app_config_debug_format() {
        let config = AppConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AppConfig"));
    }
}
