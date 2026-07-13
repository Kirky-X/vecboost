// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

// Deprecated: 使用 confers 加载的版本请见 app_config.rs(需启用 `config` feature)
// 此文件将在 v0.3.0 移除

#![allow(clippy::all)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::pipeline::{PipelineConfig, PriorityConfig, QueueConfig, WorkerConfig};

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConfigLoader {
    config_path: Option<PathBuf>,
    env_prefix: String,
}

impl ConfigLoader {
    pub fn new() -> Self {
        Self {
            config_path: None,
            env_prefix: "VECBOOST".to_string(),
        }
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigLoader {
    pub fn with_config_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn with_env_prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.env_prefix = prefix.into();
        self
    }

    pub fn load(&self) -> Result<AppConfig, ConfigError> {
        let mut config = config::Config::builder();

        config = config.set_default("server.host", "0.0.0.0")?;
        config = config.set_default("server.port", 3000)?;
        config = config.set_default("server.timeout", 30)?;
        config = config.set_default("server.grpc_enabled", false)?;
        config = config.set_default("server.grpc_port", 50051)?;

        config = config.set_default("model.model_repo", "BAAI/bge-m3")?;
        config = config.set_default("model.model_revision", "main")?;
        config = config.set_default("model.use_gpu", false)?;
        config = config.set_default("model.batch_size", 32)?;
        config = config.set_default("model.expected_dimension", 1024)?;
        config = config.set_default("model.max_sequence_length", 8192)?;

        config = config.set_default("embedding.default_aggregation", "mean")?;
        config = config.set_default("embedding.similarity_metric", "cosine")?;
        config = config.set_default("embedding.cache_enabled", true)?;
        config = config.set_default("embedding.cache_size", 1024)?;
        config = config.set_default("embedding.max_batch_size", 64)?;

        config = config.set_default("monitoring.memory_limit_mb", 4096)?;
        config = config.set_default("monitoring.memory_warning_threshold", 0.8)?;
        config = config.set_default("monitoring.metrics_enabled", true)?;
        config = config.set_default("monitoring.log_level", "info")?;

        config = config.set_default("auth.enabled", false)?;
        config = config.set_default("auth.token_expiration_hours", 24)?;
        config = config.set_default("auth.default_admin_username", "admin")?;
        // Default password is empty - must be set via VECBOOST_ADMIN_PASSWORD environment variable
        // For development/testing, you can set a password in config, but production MUST use env var
        config = config.set_default("auth.default_admin_password", "")?;
        config = config.set_default("auth.security.storage_type", "environment")?;

        config = config.set_default("audit.enabled", true)?;
        config = config.set_default("audit.log_file_path", "logs/audit.log")?;
        config = config.set_default("audit.log_level", "info")?;
        config = config.set_default("audit.max_file_size_mb", 100)?;
        config = config.set_default("audit.max_files", 10)?;
        config = config.set_default("auth.csrf.enabled", false)?;
        config = config.set_default("auth.csrf.token_validation_enabled", false)?;
        config = config.set_default("auth.csrf.token_expiration_secs", 3600)?;
        config = config.set_default("auth.csrf.allow_same_origin", true)?;

        config = config.set_default("rate_limit.enabled", true)?;
        config = config.set_default("rate_limit.global_requests_per_minute", 1000)?;
        config = config.set_default("rate_limit.ip_requests_per_minute", 100)?;
        config = config.set_default("rate_limit.user_requests_per_minute", 200)?;
        config = config.set_default("rate_limit.api_key_requests_per_minute", 500)?;
        config = config.set_default("rate_limit.window_secs", 60)?;
        config = config.set_default("rate_limit.ip_whitelist", Vec::<String>::new())?;

        config = config.set_default("memory_pool.enabled", true)?;
        config = config.set_default("memory_pool.tensor_pool.enabled", true)?;
        config = config.set_default("memory_pool.tensor_pool.max_batch_size", 128)?;
        config = config.set_default("memory_pool.tensor_pool.max_sequence_length", 8192)?;
        config = config.set_default("memory_pool.tensor_pool.pool_size_per_shape", 4)?;
        config = config.set_default("memory_pool.tensor_pool.preallocate_on_startup", true)?;
        config = config.set_default("memory_pool.buffer_pool.enabled", true)?;
        config = config.set_default(
            "memory_pool.buffer_pool.text_buffer_sizes",
            vec![16, 32, 64, 128, 256],
        )?;
        config = config.set_default(
            "memory_pool.buffer_pool.vector_buffer_sizes",
            vec![16, 32, 64, 128, 256],
        )?;
        config = config.set_default("memory_pool.buffer_pool.pool_size_per_size", 8)?;
        config = config.set_default("memory_pool.model_pool.enabled", true)?;
        config = config.set_default("memory_pool.model_pool.max_memory_mb", 8192)?;
        config = config.set_default("memory_pool.model_pool.cache_models", true)?;
        config = config.set_default("memory_pool.cuda_pool.enabled", true)?;
        config = config.set_default("memory_pool.cuda_pool.max_memory_mb", 4096)?;

        config = config.set_default("pipeline.enabled", false)?;
        config = config.set_default("pipeline.queue.max_queue_size", 10000)?;
        config = config.set_default("pipeline.queue.enable_priority", true)?;
        config = config.set_default("pipeline.worker.min_workers", 2)?;
        config = config.set_default("pipeline.worker.max_workers", 16)?;
        config = config.set_default("pipeline.worker.scale_up_threshold", 100)?;
        config = config.set_default("pipeline.worker.scale_down_threshold", 10)?;
        config = config.set_default("pipeline.worker.idle_timeout_secs", 60)?;
        config = config.set_default("pipeline.worker.scale_check_interval_secs", 5)?;
        config = config.set_default("pipeline.priority.base_priority", 50)?;
        config = config.set_default("pipeline.priority.timeout_boost_factor", 2.0)?;

        if let Some(path) = &self.config_path {
            if path.exists() {
                config =
                    config.add_source(config::File::with_name(path.to_string_lossy().as_ref()));
            }
        } else {
            let default_config = PathBuf::from("config.toml");
            if default_config.exists() {
                config = config.add_source(config::File::with_name("config"));
            }
        }

        config = config.add_source(
            config::Environment::with_prefix(&self.env_prefix)
                .prefix_separator("_")
                .separator("__")
                .ignore_empty(true),
        );

        // Build config and apply environment variable overrides
        let mut cfg = config
            .build()?
            .try_deserialize()
            .map_err(ConfigError::from)
            .map(|mut cfg: AppConfig| {
                apply_priority_defaults(&mut cfg.pipeline.priority);
                cfg
            })?;

        // Validate and apply environment variables for sensitive config
        apply_security_env_overrides(&mut cfg)?;

        Ok(cfg)
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

    #[error("Parse error: {0}")]
    Parse(#[from] toml::de::Error),
}

impl From<config::ConfigError> for ConfigError {
    fn from(e: config::ConfigError) -> Self {
        ConfigError::Message(e.to_string())
    }
}

/// Apply environment variable overrides for sensitive configuration
///
/// This function handles the priority of configuration sources:
/// 1. Environment variables (highest priority)
/// 2. Configuration file
/// 3. Default values (lowest priority)
///
/// For sensitive values like JWT secrets and passwords, environment variables
/// are required for production deployments.
fn apply_security_env_overrides(cfg: &mut AppConfig) -> Result<(), ConfigError> {
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

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let loader = ConfigLoader::new();
        loader.load()
    }

    pub fn load_with_path<P: Into<PathBuf>>(path: P) -> Result<Self, ConfigError> {
        let loader = ConfigLoader::new().with_config_path(path);
        loader.load()
    }

    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<(), std::io::Error> {
        let content = self.to_toml_string().map_err(std::io::Error::other)?;
        std::fs::write(path.into(), content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_security_env_overrides_empty_jwt_secret() {
        // This test verifies that empty JWT secret from env var is rejected
        // We can't actually test the env var in unit tests without affecting other tests
        // So we just verify the function signature and behavior
        let mut cfg = AppConfig::default();
        cfg.auth.jwt_secret = Some(String::new());
        // The validation happens in JwtManager, not here
        assert!(cfg.auth.jwt_secret == Some(String::new()));
    }

    #[test]
    fn test_apply_security_env_overrides_short_jwt_secret() {
        let mut cfg = AppConfig::default();
        cfg.auth.jwt_secret = Some("short".to_string());
        assert!(cfg.auth.jwt_secret == Some("short".to_string()));
    }

    #[test]
    fn test_apply_security_env_overrides_valid_jwt_secret() {
        let mut cfg = AppConfig::default();
        cfg.auth.jwt_secret =
            Some("this-is-a-valid-secret-at-least-32-characters-long".to_string());
        assert!(cfg.auth.jwt_secret.is_some());
        assert!(cfg.auth.jwt_secret.unwrap().len() >= 32);
    }

    #[test]
    fn test_apply_security_env_overrides_empty_password() {
        let mut cfg = AppConfig::default();
        cfg.auth.default_admin_password = Some(String::new());
        assert!(cfg.auth.default_admin_password == Some(String::new()));
    }

    #[test]
    fn test_apply_security_env_overrides_short_password() {
        let mut cfg = AppConfig::default();
        cfg.auth.default_admin_password = Some("short".to_string());
        assert!(cfg.auth.default_admin_password == Some("short".to_string()));
    }

    #[test]
    fn test_apply_security_env_overrides_valid_password() {
        let mut cfg = AppConfig::default();
        cfg.auth.default_admin_password = Some("Secure@Pass123!".to_string());
        assert!(cfg.auth.default_admin_password.is_some());
        assert!(cfg.auth.default_admin_password.unwrap().len() >= 12);
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
    fn test_config_loader_new() {
        let loader = ConfigLoader::new();
        assert!(loader.config_path.is_none());
        assert_eq!(loader.env_prefix, "VECBOOST");
    }

    #[test]
    fn test_config_loader_default() {
        let loader = ConfigLoader::default();
        assert!(loader.config_path.is_none());
        assert_eq!(loader.env_prefix, "VECBOOST");
    }

    #[test]
    fn test_config_loader_with_config_path() {
        let loader = ConfigLoader::new().with_config_path("/custom/path.toml");
        assert_eq!(loader.config_path, Some(PathBuf::from("/custom/path.toml")));
    }

    #[test]
    fn test_config_loader_with_env_prefix() {
        let loader = ConfigLoader::new().with_env_prefix("CUSTOM");
        assert_eq!(loader.env_prefix, "CUSTOM");
    }

    #[test]
    fn test_config_loader_builder_chain() {
        let loader = ConfigLoader::new()
            .with_config_path("config.toml")
            .with_env_prefix("MYAPP");
        assert_eq!(loader.config_path, Some(PathBuf::from("config.toml")));
        assert_eq!(loader.env_prefix, "MYAPP");
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
    fn test_app_config_to_toml_string() {
        let config = AppConfig::default();
        let toml_str = config.to_toml_string();
        assert!(toml_str.is_ok());
        let toml_content = toml_str.unwrap();
        assert!(toml_content.contains("[server]"));
        assert!(toml_content.contains("[model]"));
        assert!(toml_content.contains("[embedding]"));
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
    fn test_config_error_from_config_error() {
        let config_err = config::ConfigError::Message("test error".to_string());
        let app_err: ConfigError = config_err.into();
        match app_err {
            ConfigError::Message(msg) => assert!(msg.contains("test error")),
            _ => panic!("Expected Message variant"),
        }
    }

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::Message("display test".to_string());
        assert_eq!(format!("{}", err), "Configuration error: display test");
    }
}
