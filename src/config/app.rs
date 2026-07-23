// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 配置子结构体定义(数据结构层)。
//!
//! `AppConfig` 的 confers 加载逻辑见 `app_config.rs`(confers 完全接管配置加载,
//! 禁止手写 config/toml 解析)。本文件只保留子结构体定义、默认值实现、
//! 安全环境变量覆盖和优先级默认值。

#![allow(clippy::all)]

use serde::{Deserialize, Serialize};

pub use crate::pipeline::{PipelineConfig, PriorityConfig, QueueConfig, WorkerConfig};

// 注:AppConfig 定义已迁移至 app_config.rs(由 confers #[derive(Config)] 接管)。
// 本文件保留所有子结构体定义,供 app_config.rs 引用。

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub grpc_host: Option<String>,
    pub grpc_port: Option<u16>,
    pub grpc_enabled: bool,
    pub workers: Option<usize>,
    pub timeout: Option<u64>,
    /// gRPC server max concurrent streams per connection.
    pub grpc_max_connections: Option<usize>,
    /// gRPC request timeout in seconds (applies to streaming RPCs).
    pub grpc_timeout_seconds: Option<u64>,
    /// Whether gRPC server requires authentication (secure default: true).
    /// Set to false only for development/test environments behind network isolation.
    pub grpc_require_auth: Option<bool>,
    /// Allowed root directories for `grpc_embed_file` path validation.
    /// When empty, falls back to current working directory (with sensitive-dir check).
    pub grpc_allowed_roots: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    pub model_repo: String,
    pub model_revision: String,
    pub model_path: Option<String>,
    pub use_gpu: bool,
    pub batch_size: usize,
    pub expected_dimension: Option<usize>,
    pub max_sequence_length: Option<usize>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub default_aggregation: String,
    pub similarity_metric: String,
    pub cache_enabled: bool,
    pub cache_size: usize,
    pub max_batch_size: usize,
    pub max_text_length: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct MonitoringConfig {
    pub memory_limit_mb: Option<usize>,
    pub memory_warning_threshold: Option<f64>,
    pub metrics_enabled: bool,
    pub log_level: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: Option<String>,
    pub token_expiration_hours: Option<i64>,
    pub default_admin_username: Option<String>,
    pub default_admin_password: Option<String>,
    pub security: SecurityConfig,
    pub csrf: CsrfConfig,
    /// Trusted proxy CIDRs for X-Forwarded-For trust boundary.
    ///
    /// When non-empty, `X-Forwarded-For` / `X-Real-IP` headers are honored only if
    /// the `ConnectInfo` peer IP matches a CIDR entry — preventing clients outside
    /// the trust boundary from spoofing their IP via XFF.
    /// When empty, XFF is honored unconditionally (legacy v0.3.0–v0.3.2 behavior,
    /// kept for backward compatibility).
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
}

/// Rate limiting configuration
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Global requests per minute
    pub global_requests_per_minute: u64,
    /// IP requests per minute
    pub ip_requests_per_minute: u64,
    /// User requests per minute
    pub user_requests_per_minute: u64,
    /// API Key requests per minute
    pub api_key_requests_per_minute: u64,
    /// Window size in seconds
    pub window_secs: u64,
    /// IP whitelist (these IPs bypass rate limiting)
    pub ip_whitelist: Vec<String>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            global_requests_per_minute: 1000,
            ip_requests_per_minute: 100,
            user_requests_per_minute: 200,
            api_key_requests_per_minute: 500,
            window_secs: 60,
            ip_whitelist: vec![],
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub storage_type: String,
    pub encryption_key: Option<String>,
    pub key_file_path: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct CsrfConfig {
    pub enabled: bool,
    pub allowed_origins: Option<Vec<String>>,
    pub token_validation_enabled: bool,
    pub token_expiration_secs: Option<u64>,
    pub allow_same_origin: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            storage_type: "environment".to_string(),
            encryption_key: None,
            key_file_path: None,
        }
    }
}

impl Default for CsrfConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_origins: None,
            token_validation_enabled: false,
            token_expiration_secs: Some(3600),
            allow_same_origin: true,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_file_path: String,
    pub log_level: String,
    pub max_file_size_mb: usize,
    pub max_files: usize,
}

/// 数据库配置（dbnexus，需启用 `db` feature）
#[cfg(feature = "db")]
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// 数据库连接 URL
    /// SQLite: "sqlite:vecboost.db" / "sqlite::memory:"
    /// PostgreSQL: "postgres://user:pass@localhost:5432/vecboost"
    pub url: String,
    /// 连接池大小
    pub max_connections: u32,
    /// 连接超时（秒）
    pub connect_timeout_secs: u64,
}

#[cfg(feature = "db")]
impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "sqlite:vecboost.db".to_string(),
            max_connections: 10,
            connect_timeout_secs: 5,
        }
    }
}

/// 内存池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct MemoryPoolConfig {
    /// 是否启用内存池
    pub enabled: bool,
    /// Tensor 池配置
    pub tensor_pool: TensorPoolConfig,
    /// 缓冲区池配置
    pub buffer_pool: BufferPoolConfig,
    /// 模型权重池配置
    pub model_pool: ModelPoolConfig,
    /// CUDA 池配置
    pub cuda_pool: CudaPoolConfig,
}

/// Tensor 池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct TensorPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 最大批量大小
    pub max_batch_size: usize,
    /// 最大序列长度
    pub max_sequence_length: usize,
    /// 每种形状的池大小
    pub pool_size_per_shape: usize,
    /// 启动时预分配
    pub preallocate_on_startup: bool,
}

/// 缓冲区池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct BufferPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 文本缓冲区大小列表
    pub text_buffer_sizes: Vec<usize>,
    /// 向量缓冲区大小列表
    pub vector_buffer_sizes: Vec<usize>,
    /// 每种大小的池大小
    pub pool_size_per_size: usize,
}

/// 模型权重池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct ModelPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 最大内存（MB）
    pub max_memory_mb: usize,
    /// 是否缓存模型
    pub cache_models: bool,
}

/// CUDA 池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(default)]
pub struct CudaPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 最大内存（MB）
    pub max_memory_mb: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_file_path: "logs/audit.log".to_string(),
            log_level: "info".to_string(),
            max_file_size_mb: 100,
            max_files: 10,
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        }
    }
}

impl Default for TensorPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 128,
            max_sequence_length: 8192,
            pool_size_per_shape: 4,
            preallocate_on_startup: true,
        }
    }
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            text_buffer_sizes: vec![16, 32, 64, 128, 256],
            vector_buffer_sizes: vec![16, 32, 64, 128, 256],
            pool_size_per_size: 8,
        }
    }
}

impl Default for ModelPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: 8192,
            cache_models: true,
        }
    }
}

impl Default for CudaPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: 4096,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            grpc_host: None,
            grpc_port: Some(50051),
            grpc_enabled: false,
            workers: None,
            timeout: Some(30),
            grpc_max_connections: Some(1000),
            grpc_timeout_seconds: Some(30),
            // Secure default: require auth unless explicitly disabled.
            // Callers must opt-out via config.toml `[server] grpc_require_auth = false`.
            grpc_require_auth: Some(true),
            grpc_allowed_roots: None,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_repo: "BAAI/bge-m3".to_string(),
            model_revision: "main".to_string(),
            model_path: None,
            use_gpu: false,
            batch_size: 32,
            expected_dimension: Some(1024),
            max_sequence_length: Some(8192),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            default_aggregation: "mean".to_string(),
            similarity_metric: "cosine".to_string(),
            cache_enabled: true,
            cache_size: 1024,
            max_batch_size: 64,
            max_text_length: 8192,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: Some(4096),
            memory_warning_threshold: Some(0.8),
            metrics_enabled: true,
            log_level: Some("info".to_string()),
        }
    }
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            jwt_secret: None,
            token_expiration_hours: Some(24),
            default_admin_username: None,
            default_admin_password: None,
            security: SecurityConfig::default(),
            csrf: CsrfConfig::default(),
            trusted_proxies: Vec::new(),
        }
    }
}

pub(crate) fn apply_priority_defaults(priority: &mut PriorityConfig) {
    if priority.user_tier_weights.is_empty() {
        priority.user_tier_weights.insert("free".to_string(), 1.0);
        priority.user_tier_weights.insert("basic".to_string(), 1.5);
        priority
            .user_tier_weights
            .insert("premium".to_string(), 2.0);
        priority
            .user_tier_weights
            .insert("enterprise".to_string(), 3.0);
    }
    if priority.source_weights.is_empty() {
        priority.source_weights.insert("http".to_string(), 1.0);
        priority.source_weights.insert("grpc".to_string(), 1.2);
        priority.source_weights.insert("internal".to_string(), 1.5);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration error: {0}")]
    Message(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("confers error: {0}")]
    Confers(#[from] confers::ConfigError),
}

/// Apply environment variable overrides for sensitive configuration
///
/// This function handles the priority of configuration sources:
/// 1. Environment variables (highest priority)
/// 2. Configuration file
/// 3. Default values (lowest priority)
///
/// For sensitive values like JWT secrets and passwords, environment variables
/// are required for production deployments. Validates minimum length constraints.
pub(crate) fn apply_security_env_overrides(
    cfg: &mut super::app_config::AppConfig,
) -> Result<(), ConfigError> {
    use std::env;

    const MIN_JWT_SECRET_LENGTH: usize = 32;
    const MIN_PASSWORD_LENGTH: usize = 12;

    // Handle JWT secret with environment variable
    if let Ok(jwt_secret) = env::var("VECBOOST_JWT_SECRET") {
        if jwt_secret.is_empty() {
            return Err(ConfigError::Message(
                "VECBOOST_JWT_SECRET cannot be empty".to_string(),
            ));
        }
        if jwt_secret.len() < MIN_JWT_SECRET_LENGTH {
            return Err(ConfigError::Message(format!(
                "VECBOOST_JWT_SECRET must be at least {} characters (current: {})",
                MIN_JWT_SECRET_LENGTH,
                jwt_secret.len()
            )));
        }
        cfg.auth.jwt_secret = Some(jwt_secret);
    }

    // Handle default admin password with environment variable
    if let Ok(admin_password) = env::var("VECBOOST_ADMIN_PASSWORD") {
        if admin_password.is_empty() {
            return Err(ConfigError::Message(
                "VECBOOST_ADMIN_PASSWORD cannot be empty".to_string(),
            ));
        }
        if admin_password.len() < MIN_PASSWORD_LENGTH {
            return Err(ConfigError::Message(format!(
                "VECBOOST_ADMIN_PASSWORD must be at least {} characters (current: {})",
                MIN_PASSWORD_LENGTH,
                admin_password.len()
            )));
        }
        cfg.auth.default_admin_password = Some(admin_password);
    }

    Ok(())
}

/// 全局 env var 测试锁。
///
/// 串行化所有使用 `std::env::set_var` / `remove_var` 的测试,避免并行测试
/// 之间的环境变量污染。crate 内所有 env-touching 测试必须通过此锁,
/// 不允许在其它模块再定义独立的 ENV_LOCK,否则会破坏串行化保证。
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::app_config::AppConfig;
    use std::sync::Mutex;

    fn env_lock() -> &'static Mutex<()> {
        &super::test_support::ENV_LOCK
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3000);
        assert!(!config.grpc_enabled);
        assert_eq!(config.grpc_port, Some(50051));
        assert_eq!(config.timeout, Some(30));
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.model_repo, "BAAI/bge-m3");
        assert_eq!(config.model_revision, "main");
        assert!(!config.use_gpu);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.expected_dimension, Some(1024));
        assert_eq!(config.max_sequence_length, Some(8192));
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.default_aggregation, "mean");
        assert_eq!(config.similarity_metric, "cosine");
        assert!(config.cache_enabled);
        assert_eq!(config.cache_size, 1024);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_text_length, 8192);
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert_eq!(config.memory_limit_mb, Some(4096));
        assert_eq!(config.memory_warning_threshold, Some(0.8));
        assert!(config.metrics_enabled);
    }

    #[test]
    fn test_auth_config_default() {
        let config = AuthConfig::default();
        assert!(!config.enabled);
        assert!(config.jwt_secret.is_none());
        assert_eq!(config.token_expiration_hours, Some(24));
        assert!(config.trusted_proxies.is_empty());
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert!(config.enabled);
        assert_eq!(config.log_file_path, "logs/audit.log");
        assert_eq!(config.log_level, "info");
        assert_eq!(config.max_file_size_mb, 100);
        assert_eq!(config.max_files, 10);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.global_requests_per_minute, 1000);
        assert_eq!(config.ip_requests_per_minute, 100);
        assert_eq!(config.user_requests_per_minute, 200);
        assert_eq!(config.api_key_requests_per_minute, 500);
        assert_eq!(config.window_secs, 60);
        assert!(config.ip_whitelist.is_empty());
    }

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert_eq!(config.storage_type, "environment");
        assert!(config.encryption_key.is_none());
        assert!(config.key_file_path.is_none());
    }

    #[test]
    fn test_csrf_config_default() {
        let config = CsrfConfig::default();
        assert!(!config.enabled);
        assert!(config.allowed_origins.is_none());
        assert!(!config.token_validation_enabled);
        assert_eq!(config.token_expiration_secs, Some(3600));
        assert!(config.allow_same_origin);
    }

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();
        assert!(config.enabled);
        assert!(config.tensor_pool.enabled);
        assert!(config.buffer_pool.enabled);
        assert!(config.model_pool.enabled);
        assert!(config.cuda_pool.enabled);
        assert_eq!(config.tensor_pool.max_batch_size, 128);
        assert_eq!(config.tensor_pool.max_sequence_length, 8192);
        assert_eq!(config.model_pool.max_memory_mb, 8192);
        assert_eq!(config.cuda_pool.max_memory_mb, 4096);
    }

    #[test]
    fn test_app_config_default() {
        let config = AppConfig::default();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.model.model_repo, "BAAI/bge-m3");
        assert!(config.embedding.cache_enabled);
        assert!(config.audit.enabled);
        assert!(config.rate_limit.enabled);
        assert!(config.memory_pool.enabled);
    }

    #[test]
    fn test_apply_priority_defaults_with_empty_weights() {
        let mut cfg = AppConfig::default();
        cfg.pipeline.priority.user_tier_weights.clear();
        cfg.pipeline.priority.source_weights.clear();
        apply_priority_defaults(&mut cfg.pipeline.priority);
        assert_eq!(cfg.pipeline.priority.user_tier_weights.len(), 4);
        assert!(cfg.pipeline.priority.user_tier_weights.contains_key("free"));
        assert!(
            cfg.pipeline
                .priority
                .user_tier_weights
                .contains_key("premium")
        );
        assert_eq!(cfg.pipeline.priority.source_weights.len(), 3);
        assert!(cfg.pipeline.priority.source_weights.contains_key("http"));
    }

    #[test]
    fn test_apply_priority_defaults_preserves_existing() {
        let mut cfg = AppConfig::default();
        cfg.pipeline
            .priority
            .user_tier_weights
            .insert("custom".to_string(), 5.0);
        apply_priority_defaults(&mut cfg.pipeline.priority);
        assert!(
            cfg.pipeline
                .priority
                .user_tier_weights
                .contains_key("custom")
        );
        assert_eq!(
            cfg.pipeline.priority.user_tier_weights.get("custom"),
            Some(&5.0)
        );
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::Message("display test".to_string());
        assert_eq!(format!("{}", err), "Configuration error: display test");
    }

    #[test]
    fn test_config_error_io_variant_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let err = ConfigError::Io(io_err);
        assert!(format!("{}", err).contains("IO error"));
        assert!(format!("{}", err).contains("missing file"));
    }

    #[test]
    fn test_tensor_pool_config_default() {
        let config = TensorPoolConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_batch_size, 128);
        assert_eq!(config.max_sequence_length, 8192);
        assert_eq!(config.pool_size_per_shape, 4);
        assert!(config.preallocate_on_startup);
    }

    #[test]
    fn test_buffer_pool_config_default() {
        let config = BufferPoolConfig::default();
        assert!(config.enabled);
        assert_eq!(config.text_buffer_sizes, vec![16, 32, 64, 128, 256]);
        assert_eq!(config.vector_buffer_sizes, vec![16, 32, 64, 128, 256]);
        assert_eq!(config.pool_size_per_size, 8);
    }

    #[test]
    fn test_model_pool_config_default() {
        let config = ModelPoolConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_memory_mb, 8192);
        assert!(config.cache_models);
    }

    #[test]
    fn test_cuda_pool_config_default() {
        let config = CudaPoolConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_memory_mb, 4096);
    }

    #[test]
    fn test_app_config_default_full_structure() {
        let config = AppConfig::default();
        // 验证所有子配置默认值
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 3000);
        assert!(!config.server.grpc_enabled);
        assert_eq!(config.server.grpc_port, Some(50051));
        assert_eq!(config.server.timeout, Some(30));

        assert_eq!(config.model.model_repo, "BAAI/bge-m3");
        assert_eq!(config.model.batch_size, 32);
        assert!(!config.model.use_gpu);

        assert_eq!(config.embedding.default_aggregation, "mean");
        assert_eq!(config.embedding.similarity_metric, "cosine");

        assert_eq!(config.monitoring.memory_limit_mb, Some(4096));
        assert!(config.monitoring.metrics_enabled);

        assert!(!config.auth.enabled);
        assert_eq!(config.auth.token_expiration_hours, Some(24));
        assert!(!config.auth.csrf.enabled);
        assert!(config.auth.csrf.allow_same_origin);

        assert!(config.audit.enabled);
        assert!(config.rate_limit.enabled);
        assert!(config.memory_pool.enabled);
    }

    #[test]
    fn test_apply_security_env_overrides_no_env_vars_is_noop() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        // 确保不设置环境变量时,函数不修改配置
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }

        let mut cfg = AppConfig::default();
        cfg.auth.jwt_secret = None;
        cfg.auth.default_admin_password = None;
        let result = apply_security_env_overrides(&mut cfg);
        assert!(result.is_ok());
        assert!(cfg.auth.jwt_secret.is_none());
        assert!(cfg.auth.default_admin_password.is_none());
    }

    #[test]
    fn test_apply_security_env_overrides_empty_jwt_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "");
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("VECBOOST_JWT_SECRET cannot be empty"));
    }

    #[test]
    fn test_apply_security_env_overrides_short_jwt_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", "tooshort");
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("at least 32 characters"));
    }

    #[test]
    fn test_apply_security_env_overrides_valid_jwt_applied() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let secret = "this-is-a-valid-jwt-secret-32chars!!".to_string();
        unsafe {
            std::env::set_var("VECBOOST_JWT_SECRET", &secret);
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_JWT_SECRET");
        }
        assert!(result.is_ok());
        assert_eq!(cfg.auth.jwt_secret, Some(secret));
    }

    #[test]
    fn test_apply_security_env_overrides_empty_password_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "");
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("VECBOOST_ADMIN_PASSWORD cannot be empty"));
    }

    #[test]
    fn test_apply_security_env_overrides_short_password_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", "short");
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("at least 12 characters"));
    }

    #[test]
    fn test_apply_security_env_overrides_valid_password_applied() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let password = "SuperSecurePass123!".to_string();
        unsafe {
            std::env::set_var("VECBOOST_ADMIN_PASSWORD", &password);
        }
        let mut cfg = AppConfig::default();
        let result = apply_security_env_overrides(&mut cfg);
        unsafe {
            std::env::remove_var("VECBOOST_ADMIN_PASSWORD");
        }
        assert!(result.is_ok());
        assert_eq!(cfg.auth.default_admin_password, Some(password));
    }

    #[test]
    fn test_apply_priority_defaults_idempotent() {
        let mut cfg = AppConfig::default();
        // 第一次应用
        apply_priority_defaults(&mut cfg.pipeline.priority);
        let tier_count = cfg.pipeline.priority.user_tier_weights.len();
        let source_count = cfg.pipeline.priority.source_weights.len();
        // 第二次应用不应改变(non-empty 时跳过)
        apply_priority_defaults(&mut cfg.pipeline.priority);
        assert_eq!(cfg.pipeline.priority.user_tier_weights.len(), tier_count);
        assert_eq!(cfg.pipeline.priority.source_weights.len(), source_count);
    }

    #[test]
    fn test_apply_priority_defaults_source_weights_values() {
        let mut cfg = AppConfig::default();
        cfg.pipeline.priority.source_weights.clear();
        apply_priority_defaults(&mut cfg.pipeline.priority);
        assert_eq!(cfg.pipeline.priority.source_weights.get("http"), Some(&1.0));
        assert_eq!(cfg.pipeline.priority.source_weights.get("grpc"), Some(&1.2));
        assert_eq!(
            cfg.pipeline.priority.source_weights.get("internal"),
            Some(&1.5)
        );
    }

    #[test]
    fn test_rate_limit_config_with_whitelist() {
        let mut config = RateLimitConfig::default();
        config.ip_whitelist = vec!["127.0.0.1".to_string(), "10.0.0.0/8".to_string()];
        assert_eq!(config.ip_whitelist.len(), 2);
        assert!(config.ip_whitelist.contains(&"127.0.0.1".to_string()));
    }

    #[test]
    fn test_app_config_clone_preserves_values() {
        let original = AppConfig::default();
        let cloned = original.clone();
        assert_eq!(original.server.port, cloned.server.port);
        assert_eq!(original.model.model_repo, cloned.model.model_repo);
        assert_eq!(
            original.rate_limit.window_secs,
            cloned.rate_limit.window_secs
        );
    }

    #[test]
    fn test_config_error_from_message() {
        let err = ConfigError::Message("custom error".to_string());
        assert!(err.to_string().contains("custom error"));
    }

    #[test]
    fn test_database_config_default_values() {
        #[cfg(feature = "db")]
        {
            let config = DatabaseConfig::default();
            assert_eq!(config.url, "sqlite:vecboost.db");
            assert_eq!(config.max_connections, 10);
            assert_eq!(config.connect_timeout_secs, 5);
        }
    }

    #[test]
    fn test_server_config_custom_values() {
        let config = ServerConfig {
            host: "localhost".to_string(),
            port: 4000,
            grpc_host: Some("0.0.0.0".to_string()),
            grpc_port: Some(6000),
            grpc_enabled: true,
            workers: Some(8),
            timeout: Some(60),
            grpc_max_connections: Some(500),
            grpc_timeout_seconds: Some(120),
            grpc_require_auth: Some(false),
            grpc_allowed_roots: Some(vec!["/data".to_string()]),
        };
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 4000);
        assert_eq!(config.grpc_host, Some("0.0.0.0".to_string()));
        assert_eq!(config.grpc_port, Some(6000));
        assert!(config.grpc_enabled);
        assert_eq!(config.workers, Some(8));
        assert_eq!(config.timeout, Some(60));
        assert_eq!(config.grpc_max_connections, Some(500));
        assert_eq!(config.grpc_timeout_seconds, Some(120));
        assert_eq!(config.grpc_require_auth, Some(false));
        assert_eq!(config.grpc_allowed_roots, Some(vec!["/data".to_string()]));
    }

    #[test]
    fn test_model_config_custom_values() {
        let config = ModelConfig {
            model_repo: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_revision: "v1.0".to_string(),
            model_path: Some("/path/to/model".to_string()),
            use_gpu: true,
            batch_size: 128,
            expected_dimension: Some(384),
            max_sequence_length: Some(512),
        };
        assert_eq!(config.model_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert!(config.use_gpu);
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.expected_dimension, Some(384));
    }

    #[test]
    fn test_embedding_config_custom_values() {
        let config = EmbeddingConfig {
            default_aggregation: "max".to_string(),
            similarity_metric: "dot".to_string(),
            cache_enabled: false,
            cache_size: 512,
            max_batch_size: 32,
            max_text_length: 4096,
        };
        assert_eq!(config.default_aggregation, "max");
        assert!(!config.cache_enabled);
        assert_eq!(config.cache_size, 512);
        assert_eq!(config.max_text_length, 4096);
    }

    #[test]
    fn test_auth_config_custom_values() {
        let config = AuthConfig {
            enabled: true,
            jwt_secret: Some("my-secret-key-at-least-32-chars!!".to_string()),
            token_expiration_hours: Some(48),
            default_admin_username: Some("admin".to_string()),
            default_admin_password: Some("MyPassword123!".to_string()),
            security: SecurityConfig::default(),
            csrf: CsrfConfig {
                enabled: true,
                allowed_origins: Some(vec!["https://example.com".to_string()]),
                token_validation_enabled: true,
                token_expiration_secs: Some(7200),
                allow_same_origin: false,
            },
            trusted_proxies: vec!["10.0.0.0/8".to_string()],
        };
        assert!(config.enabled);
        assert!(config.jwt_secret.is_some());
        assert_eq!(config.token_expiration_hours, Some(48));
        assert!(config.csrf.enabled);
        assert!(!config.csrf.allow_same_origin);
    }

    #[test]
    fn test_csrf_config_custom_values() {
        let config = CsrfConfig {
            enabled: true,
            allowed_origins: Some(vec![
                "https://example.com".to_string(),
                "https://app.example.com".to_string(),
            ]),
            token_validation_enabled: true,
            token_expiration_secs: Some(1800),
            allow_same_origin: false,
        };
        assert!(config.enabled);
        assert_eq!(config.allowed_origins.unwrap().len(), 2);
        assert!(config.token_validation_enabled);
    }

    #[test]
    fn test_security_config_custom_values() {
        let config = SecurityConfig {
            storage_type: "file".to_string(),
            encryption_key: Some("encryption-key".to_string()),
            key_file_path: Some("/path/to/key".to_string()),
        };
        assert_eq!(config.storage_type, "file");
        assert!(config.encryption_key.is_some());
        assert!(config.key_file_path.is_some());
    }

    #[test]
    fn test_audit_config_custom_values() {
        let config = AuditConfig {
            enabled: false,
            log_file_path: "/var/log/audit.log".to_string(),
            log_level: "debug".to_string(),
            max_file_size_mb: 200,
            max_files: 5,
        };
        assert!(!config.enabled);
        assert_eq!(config.log_file_path, "/var/log/audit.log");
        assert_eq!(config.log_level, "debug");
        assert_eq!(config.max_file_size_mb, 200);
        assert_eq!(config.max_files, 5);
    }

    #[test]
    fn test_memory_pool_config_custom_values() {
        let config = MemoryPoolConfig {
            enabled: false,
            tensor_pool: TensorPoolConfig {
                enabled: false,
                max_batch_size: 64,
                max_sequence_length: 4096,
                pool_size_per_shape: 2,
                preallocate_on_startup: false,
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        assert!(!config.enabled);
        assert!(!config.tensor_pool.enabled);
        assert_eq!(config.tensor_pool.max_batch_size, 64);
    }
}
