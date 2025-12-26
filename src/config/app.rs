// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub embedding: EmbeddingConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
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

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            embedding: EmbeddingConfig::default(),
            monitoring: MonitoringConfig::default(),
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

        if let Some(path) = &self.config_path {
            if path.exists() {
                config = config.add_source(config::File::with_name(path.to_string_lossy().as_ref()));
            }
        } else {
            let default_config = PathBuf::from("config.toml");
            if default_config.exists() {
                config = config.add_source(config::File::with_name("config"));
            }
        }

        config = config.add_source(config::Environment::with_prefix(&self.env_prefix)
            .prefix_separator("_")
            .separator("__")
            .ignore_empty(true));

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

impl AppConfig {
    pub fn load() -> Result<Self, ConfigError> {
        let loader = ConfigLoader::new();
        loader.load()
    }

    pub fn load_with_path<P: Into<PathBuf>>(path: P) -> Result<Self, ConfigError> {
        let loader = ConfigLoader::new().with_config_path(path);
        loader.load()
    }

    pub fn load_env_only() -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();

        let host = env::var("VECBOOST_SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = env::var("VECBOOST_SERVER_PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse()
            .unwrap_or(3000);

        let model_repo = env::var("VECBOOST_MODEL_REPO").unwrap_or_else(|_| "BAAI/bge-m3".to_string());
        let model_revision = env::var("VECBOOST_MODEL_REVISION").unwrap_or_else(|_| "main".to_string());
        let use_gpu = env::var("VECBOOST_MODEL_USE_GPU")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);
        let batch_size = env::var("VECBOOST_MODEL_BATCH_SIZE")
            .unwrap_or_else(|_| "32".to_string())
            .parse()
            .unwrap_or(32);
        let expected_dimension = env::var("VECBOOST_MODEL_DIMENSION")
            .unwrap_or_else(|_| "".to_string())
            .parse()
            .ok();

        Ok(AppConfig {
            server: ServerConfig { host, port, workers: None, timeout: Some(30) },
            model: ModelConfig {
                model_repo,
                model_revision,
                model_path: None,
                use_gpu,
                batch_size,
                expected_dimension,
                max_sequence_length: Some(8192),
            },
            embedding: EmbeddingConfig {
                default_aggregation: "mean".to_string(),
                similarity_metric: "cosine".to_string(),
                cache_enabled: true,
                cache_size: 1024,
                max_batch_size: 64,
            },
            monitoring: MonitoringConfig {
                memory_limit_mb: Some(4096),
                memory_warning_threshold: Some(0.8),
                metrics_enabled: true,
                log_level: Some("info".to_string()),
            },
        })
    }

    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<(), std::io::Error> {
        let content = self.to_toml_string().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path.into(), content)
    }
}
