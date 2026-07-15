// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

//! confers-based configuration loader for VecBoost.
//!
//! `AppConfig` 通过 `#[derive(Config)]` 从 `confers` crate 派生,统一接管 TOML
//! 文件加载与环境变量覆盖。confers 是必选依赖(`Cargo.toml` 中无 `optional`),
//! 禁止任何手写 `config`/`toml` 解析逻辑。
//!
//! 敏感字段(JWT secret / admin password)的最小长度校验由
//! `app::apply_security_env_overrides` 负责(单一真相源);本模块仅负责
//! confers 加载与默认值填充,不重复实现校验逻辑。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use confers::Config;

#[cfg(feature = "db")]
use super::app::DatabaseConfig;
use super::app::{
    AuditConfig, AuthConfig, ConfigError, EmbeddingConfig, MemoryPoolConfig, ModelConfig,
    MonitoringConfig, RateLimitConfig, ServerConfig, apply_priority_defaults,
    apply_security_env_overrides,
};
use crate::pipeline::PipelineConfig;

/// VecBoost 顶层配置,由 confers 完全接管加载。
///
/// 子结构体复用 `app.rs` 中的定义,确保下游消费者(services / middleware)
/// 不受加载器切换影响。`env_prefix = "VECBOOST_"` 将环境变量如
/// `VECBOOST_JWT_SECRET` 映射到嵌套字段。
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
    /// 通过 confers 从默认路径 `config.toml` 加载配置。
    ///
    /// 文件不存在时回退到 `Default` 实现 + `VECBOOST_` 前缀环境变量。
    /// 敏感字段校验由 `app::apply_security_env_overrides` 执行,
    /// 校验失败时返回 `ConfigError::Message`。
    pub fn load_via_confers() -> Result<Self, ConfigError> {
        Self::load_via_confers_with_path("config.toml")
    }

    /// 通过 confers 从显式路径加载配置。
    ///
    /// 路径是可选的:文件不存在时 confers 回退到 `Default` + 环境变量。
    /// 敏感环境变量(`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`)
    /// 的最小长度校验由 `app::apply_security_env_overrides` 强制执行,
    /// 失败时通过 `?` 显式传播(规则12:错误必须显性化)。
    pub fn load_via_confers_with_path<P: Into<PathBuf>>(path: P) -> Result<Self, ConfigError> {
        let mut config = confers::ConfigBuilder::<Self>::new()
            .allow_absolute_paths()
            .file_optional(path)
            .env_prefix("VECBOOST_")
            .build()?;
        apply_security_env_overrides(&mut config)?;
        apply_priority_defaults(&mut config.pipeline.priority);
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;
    use crate::config::app::test_support::ENV_LOCK;

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
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        assert!(result.is_ok(), "should fall back to defaults");
        let config = result.unwrap();
        assert!(!config.server.grpc_enabled);
    }

    #[test]
    fn test_load_via_confers_with_valid_toml() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
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
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
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
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var(
                "VECBOOST_JWT_SECRET",
                "this-is-a-valid-jwt-secret-32chars!!",
            );
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed with valid JWT secret");
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert_eq!(
            config.auth.jwt_secret.as_deref(),
            Some("this-is-a-valid-jwt-secret-32chars!!")
        );
    }

    #[test]
    fn test_load_via_confers_empty_jwt_secret_rejected() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(
            result.is_err(),
            "empty JWT secret must be rejected by apply_security_env_overrides"
        );
    }

    #[test]
    fn test_load_via_confers_short_jwt_secret_rejected() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "tooshort");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(
            result.is_err(),
            "JWT secret shorter than 32 chars must be rejected"
        );
    }

    #[test]
    fn test_load_via_confers_admin_password_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "SuperSecurePass123!");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        let config = result.expect("load should succeed with valid admin password");
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert_eq!(
            config.auth.default_admin_password.as_deref(),
            Some("SuperSecurePass123!")
        );
    }

    #[test]
    fn test_load_via_confers_empty_admin_password_rejected() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(
            result.is_err(),
            "empty admin password must be rejected by apply_security_env_overrides"
        );
    }

    #[test]
    fn test_load_via_confers_short_admin_password_rejected() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "short");
        }
        let result = AppConfig::load_via_confers_with_path("/nonexistent/config.toml");
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(
            result.is_err(),
            "admin password shorter than 12 chars must be rejected"
        );
    }

    #[test]
    fn test_load_via_confers_with_partial_toml() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
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
