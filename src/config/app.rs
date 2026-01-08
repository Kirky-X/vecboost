// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
}

#[derive(Debug, Deserialize, Clone, Serialize)]
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
pub struct EmbeddingConfig {
    pub default_aggregation: String,
    pub similarity_metric: String,
    pub cache_enabled: bool,
    pub cache_size: usize,
    pub max_batch_size: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct MonitoringConfig {
    pub memory_limit_mb: Option<usize>,
    pub memory_warning_threshold: Option<f64>,
    pub metrics_enabled: bool,
    pub log_level: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
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
pub struct SecurityConfig {
    pub storage_type: String,
    pub encryption_key: Option<String>,
    pub key_file_path: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
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
pub struct AuditConfig {
    pub enabled: bool,
    pub log_file_path: String,
    pub log_level: String,
    pub max_file_size_mb: usize,
    pub max_files: usize,
}

/// 内存池配置
#[derive(Debug, Deserialize, Clone, Serialize)]
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
        config = config.set_default("auth.default_admin_password", "admin123")?;
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

        config.build()?.try_deserialize().map_err(ConfigError::from)
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

/// 流水线配置
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct PipelineConfig {
    /// 是否启用流水线
    pub enabled: bool,
    /// 队列配置
    pub queue: QueueConfig,
    /// Worker 配置
    pub worker: WorkerConfig,
    /// 优先级配置
    pub priority: PriorityConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            queue: QueueConfig::default(),
            worker: WorkerConfig::default(),
            priority: PriorityConfig::default(),
        }
    }
}

/// 队列配置
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct QueueConfig {
    /// 最大队列大小
    pub max_queue_size: usize,
    /// 是否启用优先级
    pub enable_priority: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            enable_priority: true,
        }
    }
}

/// Worker 配置
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct WorkerConfig {
    /// 最小 Worker 数量
    pub min_workers: usize,
    /// 最大 Worker 数量
    pub max_workers: usize,
    /// 扩容阈值
    pub scale_up_threshold: usize,
    /// 缩容阈值
    pub scale_down_threshold: usize,
    /// 空闲超时（秒）
    pub idle_timeout_secs: u64,
    /// 检查间隔（秒）
    pub scale_check_interval_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            scale_up_threshold: 100,
            scale_down_threshold: 10,
            idle_timeout_secs: 60,
            scale_check_interval_secs: 5,
        }
    }
}

/// 优先级配置
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct PriorityConfig {
    /// 基础优先级
    pub base_priority: i32,
    /// 超时提升因子
    pub timeout_boost_factor: f64,
    /// 用户等级权重
    pub user_tier_weights: Vec<(String, f64)>,
    /// 来源权重
    pub source_weights: Vec<(String, f64)>,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            base_priority: 50,
            timeout_boost_factor: 2.0,
            user_tier_weights: vec![
                ("free".to_string(), 1.0),
                ("basic".to_string(), 1.5),
                ("premium".to_string(), 2.0),
                ("enterprise".to_string(), 3.0),
            ],
            source_weights: vec![
                ("http".to_string(), 1.0),
                ("grpc".to_string(), 1.2),
                ("internal".to_string(), 1.5),
            ],
        }
    }
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
