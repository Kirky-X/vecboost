// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 配置子结构体定义(数据结构层)。
//!
//! `AppConfig` 的 confers 加载逻辑见 `app_config.rs`(confers 完全接管配置加载,
//! 禁止手写 config/toml 解析)。本文件只保留子结构体定义、默认值实现、
//! 安全环境变量覆盖和优先级默认值。

#![allow(clippy::derivable_impls)]

use crate::error::VecboostError;
use serde::{Deserialize, Serialize};

// 注:AppConfig 定义已迁移至 app_config.rs(由 confers #[derive(Config)] 接管)。
// 本文件保留所有子结构体定义,供 app_config.rs 引用。

#[derive(Debug, Deserialize, Clone, Serialize, garde::Validate, schemars::JsonSchema)]
#[serde(default)]
pub struct ServerConfig {
    #[garde(skip)]
    pub host: String,
    #[garde(range(min = 1, max = 65535))]
    pub port: u16,
    #[garde(skip)]
    pub grpc_host: Option<String>,
    #[garde(skip)]
    pub grpc_port: Option<u16>,
    #[garde(skip)]
    pub grpc_enabled: bool,
    #[garde(skip)]
    pub workers: Option<usize>,
    #[garde(skip)]
    pub timeout: Option<u64>,
    /// gRPC server max concurrent streams per connection.
    #[garde(skip)]
    pub grpc_max_connections: Option<usize>,
    /// gRPC request timeout in seconds (applies to streaming RPCs).
    #[garde(skip)]
    pub grpc_timeout_seconds: Option<u64>,
    /// HTTP 请求体大小上限(MiB);默认 5。
    #[garde(range(min = 1, max = 1024))]
    pub body_limit_mb: u32,
    /// HTTP 全局请求超时秒数(TimeoutLayer);默认 60。
    #[garde(range(min = 1, max = 3600))]
    pub request_timeout_seconds: u64,
    /// Whether gRPC server requires authentication (secure default: true).
    /// Set to false only for development/test environments behind network isolation.
    #[garde(skip)]
    pub grpc_require_auth: Option<bool>,
    /// Allowed root directories for `grpc_embed_file` path validation.
    /// When empty, falls back to current working directory (with sensitive-dir check).
    #[garde(skip)]
    pub grpc_allowed_roots: Option<Vec<String>>,
    /// 启用 CORS 中间件（默认关闭）。
    #[garde(skip)]
    #[serde(default)]
    pub cors_enabled: bool,
    /// CORS 允许的来源列表；含 "*" 或为空时允许任意来源。
    #[garde(skip)]
    #[serde(default)]
    pub cors_allow_origins: Vec<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize, garde::Validate, schemars::JsonSchema)]
#[serde(default)]
pub struct ModelConfig {
    #[garde(length(min = 1))]
    pub model_repo: String,
    #[garde(skip)]
    pub model_revision: String,
    #[garde(skip)]
    pub model_path: Option<String>,
    #[garde(skip)]
    pub use_gpu: bool,
    #[garde(range(min = 1, max = 1024))]
    pub batch_size: usize,
    #[garde(skip)]
    pub expected_dimension: Option<usize>,
    #[garde(skip)]
    pub max_sequence_length: Option<usize>,
    /// GGUF 量化模型开关：true 且 model_path 以 `.gguf` 结尾时
    /// 选用量化引擎；默认 false。对应运行时 `ModelConfig.quantized`。
    #[garde(skip)]
    #[serde(default)]
    pub quantized: bool,
    /// 最大驻留模型数：对应 `ModelManager::with_residency`；
    /// None = 不限制（现状行为）。
    #[garde(skip)]
    pub max_resident_models: Option<usize>,
    /// 驻留内存预算 MB：超预算按 LFRU 硬驱逐；None = 不限制。
    #[garde(skip)]
    pub resident_memory_budget_mb: Option<u64>,
}

#[derive(Debug, Deserialize, Clone, Serialize, garde::Validate, schemars::JsonSchema)]
#[serde(default)]
pub struct EmbeddingConfig {
    #[garde(skip)]
    pub default_aggregation: String,
    #[garde(skip)]
    pub similarity_metric: String,
    #[garde(skip)]
    pub cache_enabled: bool,
    #[garde(skip)]
    pub cache_size: usize,
    /// embedding 缓存 WAL 持久文件路径（None = 纯内存，默认）。
    #[garde(skip)]
    pub persist_path: Option<String>,
    /// WAL 超过该字节数触发启动紧凑化（None = 1 GiB）。
    #[garde(skip)]
    pub persist_max_bytes: Option<u64>,
    #[garde(range(min = 1))]
    pub max_batch_size: usize,
    #[garde(range(min = 1))]
    pub max_text_length: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize, garde::Validate, schemars::JsonSchema)]
#[serde(default)]
pub struct RerankConfig {
    #[garde(range(min = 1))]
    pub max_documents_per_query: usize,
    #[garde(range(min = 1))]
    pub max_query_length: usize,
    #[garde(skip)]
    pub cache_enabled: bool,
    #[garde(skip)]
    pub cache_size: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct MonitoringConfig {
    pub memory_limit_mb: Option<usize>,
    pub memory_warning_threshold: Option<f64>,
    pub metrics_enabled: bool,
    pub log_level: Option<String>,
}

/// 日志配置（[logging] 段）。
///
/// `VECBOOST_LOG_LEVEL` 环境变量优先级高于 `level` 字段。
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct LoggingConfig {
    /// 日志级别: trace, debug, info, warn, error
    pub level: String,
    /// 是否输出到控制台（CLI/MCP 模式下强制关闭，stdout 需承载结果/协议消息）
    pub console: bool,
    /// 日志文件路径（空字符串表示不写入文件）
    pub file_path: String,
    /// 日志文件轮转大小 (MB)
    pub rotation_size_mb: u64,
    /// 保留的日志文件数量
    pub max_files: u32,
    /// 文件日志采样配置（低于阈值的记录按 N 取 1 写入文件）
    pub sampling: LogSamplingConfig,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            console: true,
            file_path: "logs/vecboost.log".to_string(),
            rotation_size_mb: 100,
            max_files: 10,
            sampling: LogSamplingConfig::default(),
        }
    }
}

/// 文件日志采样配置（inklog Sampler/SamplingSink）。
///
/// 采样只作用于文件 sink：≥ `min_level` 的记录直接放行，低于阈值的记录按
/// `sample_every_n` 取 1 写入；命中 `keyword_whitelist` 的记录豁免采样。
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct LogSamplingConfig {
    /// 启用文件日志采样
    pub enabled: bool,
    /// 级别阈值（trace/debug/info/warn/error/fatal）
    pub min_level: String,
    /// 低于阈值的记录 N 取 1（1 表示全放行）
    pub sample_every_n: u64,
    /// 豁免关键词（大小写不敏感子串匹配 message/target）
    pub keyword_whitelist: Vec<String>,
}

impl Default for LogSamplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_level: "info".to_string(),
            sample_every_n: 1,
            keyword_whitelist: vec![],
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct AuthConfig {
    pub enabled: bool,
    #[serde(
        default,
        serialize_with = "crate::config::encryption::encrypted_option::serialize",
        deserialize_with = "crate::config::encryption::encrypted_option::deserialize"
    )]
    pub jwt_secret: Option<String>,
    pub token_expiration_hours: Option<i64>,
    /// Token 有效期秒数覆盖（R-4 审计建议：亚小时粒度，供 E2E 过期测试与
    /// 调试用）。设置且 >0 时优先于 `token_expiration_hours`；生产配置不建议使用。
    #[serde(default)]
    pub token_expiration_seconds: Option<i64>,
    pub default_admin_username: Option<String>,
    #[serde(
        default,
        serialize_with = "crate::config::encryption::encrypted_option::serialize",
        deserialize_with = "crate::config::encryption::encrypted_option::deserialize"
    )]
    pub default_admin_password: Option<String>,
    pub csrf: CsrfConfig,
    /// Trusted proxy CIDRs for X-Forwarded-For trust boundary.
    ///
    /// When non-empty, `X-Forwarded-For` / `X-Real-IP` headers are honored only if
    /// the `ConnectInfo` peer IP matches a CIDR entry — preventing clients outside
    /// the trust boundary from spoofing their IP via XFF.
    /// When empty, XFF is IGNORED and the direct peer IP is used (secure default).
    /// To trust forwarded headers behind a reverse proxy, list your proxy CIDRs here
    /// (e.g. `["10.0.0.0/8"]`; `["0.0.0.0/0"]` restores the legacy v0.3.0–v0.3.2
    /// trust-all behavior).
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
}

/// Rate limiting configuration
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
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
    /// Inject IETF RateLimit-* headers (limit/remaining/reset/policy) into
    /// HTTP responses; rejected requests additionally carry Retry-After
    pub headers_enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            global_requests_per_minute: DEFAULT_GLOBAL_RPM,
            ip_requests_per_minute: DEFAULT_IP_RPM,
            user_requests_per_minute: DEFAULT_USER_RPM,
            api_key_requests_per_minute: DEFAULT_API_KEY_RPM,
            window_secs: DEFAULT_WINDOW_SECS,
            ip_whitelist: vec![],
            headers_enabled: false,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct CsrfConfig {
    pub enabled: bool,
}

impl Default for CsrfConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_file_path: String,
    pub log_level: String,
    pub max_file_size_mb: usize,
    pub max_files: usize,
}

/// 语义缓存配置
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct SemanticCacheConfig {
    /// 是否启用语义缓存
    pub enabled: bool,
    /// trigram Jaccard 相似度阈值（0.0-1.0）
    pub similarity_threshold: f32,
    /// 语义索引最大条目数
    pub capacity: usize,
    /// 向量比较模式：exact（默认）| i8 | binary；
    /// i8/binary 用 vquant 粗筛 + 原始向量复验，仅内部比较路径。
    #[serde(default = "default_comparison_mode")]
    pub comparison_mode: String,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            capacity: DEFAULT_SEMANTIC_CACHE_CAPACITY,
            comparison_mode: default_comparison_mode(),
        }
    }
}

fn default_comparison_mode() -> String {
    "exact".to_string()
}

/// 设备与硬件感知配置（`[device]` 段）。
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct DeviceConfig {
    /// 硬件感知启动规划（planner）：true 时用探测计划填充**未显式配置**
    /// 的字段（显式值优先），计划全文进启动日志；默认 false（零计划行为）。
    pub auto_plan: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self { auto_plan: false }
    }
}

/// GPU 显存分页配置
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct MemoryPagingConfig {
    /// 是否启用显存分页
    pub enabled: bool,
    /// GPU 显存预算（字节，0 = 自动检测）
    pub gpu_memory_budget_bytes: u64,
    /// LRU-K 的 K 值
    pub lru_k: usize,
    /// 预取深度
    pub prefetch_depth: usize,
}

impl Default for MemoryPagingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gpu_memory_budget_bytes: 0,
            lru_k: DEFAULT_LRU_K,
            prefetch_depth: DEFAULT_PREFETCH_DEPTH,
        }
    }
}

/// 数据库配置（dbnexus，需启用 `db` feature）
#[cfg(feature = "db")]
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
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
            url: DEFAULT_DB_URL.to_string(),
            max_connections: DEFAULT_DB_MAX_CONNECTIONS,
            connect_timeout_secs: DEFAULT_DB_CONNECT_TIMEOUT_SECS,
        }
    }
}

/// 内存池配置
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
#[serde(default)]
pub struct MemoryPoolConfig {
    /// 是否启用内存池
    pub enabled: bool,
    /// 缓冲区池配置
    pub buffer_pool: BufferPoolConfig,
    /// 模型权重池配置
    pub model_pool: ModelPoolConfig,
    /// CUDA 池配置
    pub cuda_pool: CudaPoolConfig,
}

/// 缓冲区池配置
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
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
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
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
#[derive(Debug, Deserialize, Clone, Serialize, schemars::JsonSchema)]
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
            log_level: DEFAULT_AUDIT_LOG_LEVEL.to_string(),
            max_file_size_mb: DEFAULT_AUDIT_MAX_FILE_SIZE_MB,
            max_files: DEFAULT_AUDIT_MAX_FILES,
        }
    }
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        }
    }
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            text_buffer_sizes: vec![16, 32, 64, 128, 256],
            vector_buffer_sizes: vec![16, 32, 64, 128, 256],
            pool_size_per_size: DEFAULT_POOL_SIZE_PER_SIZE,
        }
    }
}

/// 模型池默认最大内存 (8 GB)
const DEFAULT_MODEL_POOL_MAX_MEMORY_MB: usize = 8192;
/// CUDA 池默认最大内存 (4 GB)
const DEFAULT_CUDA_POOL_MAX_MEMORY_MB: usize = 4096;

impl Default for ModelPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: DEFAULT_MODEL_POOL_MAX_MEMORY_MB,
            cache_models: true,
        }
    }
}

impl Default for CudaPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: DEFAULT_CUDA_POOL_MAX_MEMORY_MB,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            grpc_host: None,
            grpc_port: Some(DEFAULT_GRPC_PORT),
            grpc_enabled: false,
            workers: None,
            timeout: Some(DEFAULT_TIMEOUT_SECS),
            grpc_max_connections: Some(DEFAULT_GRPC_MAX_CONNECTIONS),
            grpc_timeout_seconds: Some(DEFAULT_TIMEOUT_SECS),
            body_limit_mb: 5,
            request_timeout_seconds: 60,
            // Secure default: require auth unless explicitly disabled.
            // Callers must opt-out via config/config.toml `[server] grpc_require_auth = false`.
            grpc_require_auth: Some(true),
            grpc_allowed_roots: None,
            cors_enabled: false,
            cors_allow_origins: Vec::new(),
        }
    }
}

/// 模型配置默认值
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_EXPECTED_DIMENSION: usize = 1024;
const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 8192;

/// 嵌入配置默认值
const DEFAULT_CACHE_SIZE: usize = 1024;
const DEFAULT_MAX_BATCH_SIZE: usize = 64;
const DEFAULT_MAX_TEXT_LENGTH: usize = 8192;

/// 重排配置默认值
const DEFAULT_MAX_DOCUMENTS_PER_QUERY: usize = 100;
const DEFAULT_MAX_QUERY_LENGTH: usize = 8192;

/// 监控配置默认值
const DEFAULT_MEMORY_LIMIT_MB: usize = 4096;
const DEFAULT_MEMORY_WARNING_THRESHOLD: f64 = 0.8;

/// 审计配置默认值
const DEFAULT_AUDIT_LOG_LEVEL: &str = "info";
const DEFAULT_AUDIT_MAX_FILE_SIZE_MB: usize = 100;
const DEFAULT_AUDIT_MAX_FILES: usize = 10;

/// 限流配置默认值
const DEFAULT_GLOBAL_RPM: u64 = 1000;
const DEFAULT_IP_RPM: u64 = 100;
const DEFAULT_USER_RPM: u64 = 200;
const DEFAULT_API_KEY_RPM: u64 = 500;
const DEFAULT_WINDOW_SECS: u64 = 60;

/// 认证配置默认值
const DEFAULT_TOKEN_EXPIRATION_HOURS: i64 = 24;

/// 内存分页配置默认值
const DEFAULT_LRU_K: usize = 2;
const DEFAULT_PREFETCH_DEPTH: usize = 2;

/// 语义缓存配置默认值
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.7;
const DEFAULT_SEMANTIC_CACHE_CAPACITY: usize = 10000;

/// 数据库配置默认值（需 `db` feature）
#[cfg(feature = "db")]
const DEFAULT_DB_URL: &str = "sqlite:vecboost.db";
#[cfg(feature = "db")]
const DEFAULT_DB_MAX_CONNECTIONS: u32 = 10;
#[cfg(feature = "db")]
const DEFAULT_DB_CONNECT_TIMEOUT_SECS: u64 = 5;

/// 缓冲区池配置默认值
const DEFAULT_POOL_SIZE_PER_SIZE: usize = 8;

/// 服务器配置默认值
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: u16 = 9002;
const DEFAULT_GRPC_PORT: u16 = 50051;
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_GRPC_MAX_CONNECTIONS: usize = 1000;

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_repo: "BAAI/bge-m3".to_string(),
            model_revision: "main".to_string(),
            model_path: None,
            use_gpu: false,
            batch_size: DEFAULT_BATCH_SIZE,
            expected_dimension: Some(DEFAULT_EXPECTED_DIMENSION),
            max_sequence_length: Some(DEFAULT_MAX_SEQUENCE_LENGTH),
            quantized: false,
            max_resident_models: None,
            resident_memory_budget_mb: None,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            default_aggregation: "mean".to_string(),
            similarity_metric: "cosine".to_string(),
            cache_enabled: true,
            cache_size: DEFAULT_CACHE_SIZE,
            persist_path: None,
            persist_max_bytes: None,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_text_length: DEFAULT_MAX_TEXT_LENGTH,
        }
    }
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            max_documents_per_query: DEFAULT_MAX_DOCUMENTS_PER_QUERY,
            max_query_length: DEFAULT_MAX_QUERY_LENGTH,
            cache_enabled: true,
            cache_size: DEFAULT_CACHE_SIZE,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            memory_limit_mb: Some(DEFAULT_MEMORY_LIMIT_MB),
            memory_warning_threshold: Some(DEFAULT_MEMORY_WARNING_THRESHOLD),
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
            token_expiration_hours: Some(DEFAULT_TOKEN_EXPIRATION_HOURS),
            token_expiration_seconds: None,
            default_admin_username: None,
            default_admin_password: None,
            // 修订:CSRF 默认关闭。本 API 为纯 Bearer 认证(garrison
            // is_read_cookie=false),无 Cookie 则无 CSRF 攻击面;默认开启反而令
            // 标准 API 客户端(无 Origin 头的服务器间调用)被 403 拒绝(实测)。
            // 需要浏览器会话场景时显式设 csrf.enabled=true。
            csrf: CsrfConfig::default(),
            trusted_proxies: Vec::new(),
        }
    }
}

pub(crate) fn apply_priority_defaults(priority: &mut crate::pipeline::PriorityConfig) {
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
) -> Result<(), VecboostError> {
    use std::env;

    const MIN_JWT_SECRET_LENGTH: usize = 32;
    const MIN_PASSWORD_LENGTH: usize = 12;

    // Handle JWT secret with environment variable
    if let Ok(jwt_secret) = env::var("VECBOOST_JWT_SECRET") {
        if jwt_secret.is_empty() {
            return Err(VecboostError::ConfigError(crate::i18n::tr(
                "config-jwt-empty",
            )));
        }
        if jwt_secret.len() < MIN_JWT_SECRET_LENGTH {
            return Err(VecboostError::ConfigError(crate::i18n::tr_with_args(
                "config-jwt-length",
                crate::i18n::tr_args(&[("min", &MIN_JWT_SECRET_LENGTH.to_string())]),
            )));
        }
        cfg.auth.jwt_secret = Some(jwt_secret);
    }

    // Handle default admin password with environment variable
    if let Ok(admin_password) = env::var("VECBOOST_ADMIN_PASSWORD") {
        if admin_password.is_empty() {
            return Err(VecboostError::ConfigError(crate::i18n::tr(
                "config-password-empty",
            )));
        }
        if admin_password.len() < MIN_PASSWORD_LENGTH {
            return Err(VecboostError::ConfigError(crate::i18n::tr_with_args(
                "config-password-length",
                crate::i18n::tr_args(&[("min", &MIN_PASSWORD_LENGTH.to_string())]),
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
    /// AuthConfig 默认 CSRF 开启(配合挂载条件实现“跟随 auth”)
    #[test]
    fn authconfig_default_csrf_disabled() {
        assert!(!AuthConfig::default().csrf.enabled);
    }

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
        assert_eq!(config.port, 9002);
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
        assert!(!config.headers_enabled, "限流响应头注入缺省关闭");
    }

    #[test]
    fn test_rate_limit_headers_enabled_parses_from_toml() {
        let parsed: RateLimitConfig =
            toml::from_str("headers_enabled = true\nip_whitelist = []").expect("parse");
        assert!(parsed.headers_enabled);
        // 缺省段反序列化仍成立（serde(default)）
        let bare: RateLimitConfig = toml::from_str("enabled = true").expect("parse");
        assert!(!bare.headers_enabled);
    }

    #[test]
    fn test_logging_sampling_config_parses_from_toml() {
        let parsed: LoggingConfig = toml::from_str(
            r#"
            level = "debug"
            [sampling]
            enabled = true
            min_level = "warn"
            sample_every_n = 10
            keyword_whitelist = ["health"]
            "#,
        )
        .expect("parse");
        assert!(parsed.sampling.enabled);
        assert_eq!(parsed.sampling.min_level, "warn");
        assert_eq!(parsed.sampling.sample_every_n, 10);
        assert_eq!(
            parsed.sampling.keyword_whitelist,
            vec!["health".to_string()]
        );
        // 缺省：关闭
        let bare = LoggingConfig::default();
        assert!(!bare.sampling.enabled);
        assert_eq!(bare.sampling.sample_every_n, 1);
    }

    #[test]
    fn test_csrf_config_default() {
        let config = CsrfConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();
        assert!(config.enabled);
        assert!(config.buffer_pool.enabled);
        assert!(config.model_pool.enabled);
        assert!(config.cuda_pool.enabled);
        assert_eq!(config.model_pool.max_memory_mb, 8192);
        assert_eq!(config.cuda_pool.max_memory_mb, 4096);
    }

    #[test]
    fn test_app_config_default() {
        let config = AppConfig::default();
        assert_eq!(config.server.port, 9002);
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
        // VecboostError::ConfigError 的 Display 走 i18n 键 error-config
        let err = VecboostError::ConfigError("display test".to_string());
        assert!(format!("{}", err).contains("display test"));
    }

    #[test]
    fn test_config_error_io_variant_display() {
        // ConfigError 已并入 VecboostError(IoError 变体承载 IO 语义)
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let err = VecboostError::IoError(io_err.to_string());
        assert!(err.error_detail().contains("missing file"));
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
        assert_eq!(config.server.port, 9002);
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
        // 修订:CSRF 默认关闭(纯 Bearer API 无 Cookie 攻击面)
        assert!(!config.auth.csrf.enabled);

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
        crate::i18n::init();
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
        assert!(
            err_msg.contains("VECBOOST_JWT_SECRET"),
            "应包含变量名: {err_msg}"
        );
    }

    #[test]
    fn test_apply_security_env_overrides_short_jwt_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        crate::i18n::init();
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
        assert!(
            err_msg.contains("VECBOOST_JWT_SECRET") && err_msg.contains("32"),
            "应包含变量名与最小长度: {err_msg}"
        );
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
        crate::i18n::init();
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
        assert!(
            err_msg.contains("VECBOOST_ADMIN_PASSWORD"),
            "应包含变量名: {err_msg}"
        );
    }

    #[test]
    fn test_apply_security_env_overrides_short_password_rejected() {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        crate::i18n::init();
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
        assert!(
            err_msg.contains("VECBOOST_ADMIN_PASSWORD") && err_msg.contains("12"),
            "应包含变量名与最小长度: {err_msg}"
        );
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
        let config = RateLimitConfig {
            ip_whitelist: vec!["127.0.0.1".to_string(), "10.0.0.0/8".to_string()],
            ..Default::default()
        };
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
        let err = VecboostError::ConfigError("custom error".to_string());
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
            body_limit_mb: 5,
            request_timeout_seconds: 60,
            grpc_require_auth: Some(false),
            grpc_allowed_roots: Some(vec!["/data".to_string()]),
            cors_enabled: true,
            cors_allow_origins: vec!["https://example.com".to_string()],
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
            quantized: false,
            max_resident_models: None,
            resident_memory_budget_mb: None,
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
            persist_path: Some("data/cache.wal".to_string()),
            persist_max_bytes: Some(1024 * 1024),
            max_batch_size: 32,
            max_text_length: 4096,
        };
        assert_eq!(config.default_aggregation, "max");
        assert!(!config.cache_enabled);
        assert_eq!(config.cache_size, 512);
        assert_eq!(config.persist_path.as_deref(), Some("data/cache.wal"));
        assert_eq!(config.persist_max_bytes, Some(1024 * 1024));
        assert_eq!(config.max_text_length, 4096);
    }

    #[test]
    fn test_auth_config_custom_values() {
        let config = AuthConfig {
            enabled: true,
            jwt_secret: Some("my-secret-key-at-least-32-chars!!".to_string()),
            token_expiration_hours: Some(48),
            token_expiration_seconds: None,
            default_admin_username: Some("admin".to_string()),
            default_admin_password: Some("MyPassword123!".to_string()),
            csrf: CsrfConfig { enabled: true },
            trusted_proxies: vec!["10.0.0.0/8".to_string()],
        };
        assert!(config.enabled);
        assert!(config.jwt_secret.is_some());
        assert_eq!(config.token_expiration_hours, Some(48));
        assert!(config.csrf.enabled);
    }

    #[test]
    fn test_csrf_config_custom_values() {
        let config = CsrfConfig { enabled: true };
        assert!(config.enabled);
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
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        assert!(!config.enabled);
    }
}
