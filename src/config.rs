// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub model_repo: String,
    pub model_revision: String,
    pub use_gpu: bool,
    pub batch_size: usize,
    pub expected_dimension: Option<usize>,
}

impl AppConfig {
    pub fn load() -> Result<Self, config::ConfigError> {
        dotenvy::dotenv().ok();

        let host = env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = env::var("PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse()
            .unwrap_or(3000);

        let model_repo = env::var("MODEL_REPO").unwrap_or_else(|_| "BAAI/bge-m3".to_string());
        let model_revision = env::var("MODEL_REVISION").unwrap_or_else(|_| "main".to_string());
        let use_gpu = env::var("USE_GPU")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);
        let batch_size = env::var("BATCH_SIZE")
            .unwrap_or_else(|_| "32".to_string())
            .parse()
            .unwrap_or(32);
        let expected_dimension = env::var("MODEL_DIMENSION")
            .unwrap_or_else(|_| "".to_string())
            .parse()
            .ok();

        Ok(AppConfig {
            server: ServerConfig { host, port },
            model: ModelConfig {
                model_repo,
                model_revision,
                use_gpu,
                batch_size,
                expected_dimension,
            },
        })
    }
}
