// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::config::model::{EngineType, ModelConfig};
use crate::error::AppError;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

#[async_trait]
pub trait ModelLoader: Send + Sync {
    async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, AppError>;
    async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, AppError>;
    async fn is_model_cached(&self, config: &ModelConfig) -> bool;
}

pub trait LoadedModel: Send + Sync {
    fn name(&self) -> &str;
    fn path(&self) -> &Path;
    fn engine_type(&self) -> EngineType;
    fn reload(&self) -> Result<(), AppError>;
}

struct CandleModel {
    path: PathBuf,
    name: String,
}

#[allow(dead_code)]
struct OnnxModel {
    path: PathBuf,
    name: String,
}

impl LoadedModel for CandleModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn engine_type(&self) -> EngineType {
        EngineType::Candle
    }

    fn reload(&self) -> Result<(), AppError> {
        info!("Reloading Candle model from: {}", self.path.display());
        Ok(())
    }
}

impl LoadedModel for OnnxModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn engine_type(&self) -> EngineType {
        #[cfg(feature = "onnx")]
        {
            EngineType::Onnx
        }
        #[cfg(not(feature = "onnx"))]
        {
            unreachable!()
        }
    }

    fn reload(&self) -> Result<(), AppError> {
        info!("Reloading ONNX model from: {}", self.path.display());
        Ok(())
    }
}

pub struct LocalModelLoader {
    cache_dir: PathBuf,
}

impl LocalModelLoader {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }
}

#[async_trait]
impl ModelLoader for LocalModelLoader {
    async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, AppError> {
        if !config.model_path.exists() {
            return Err(AppError::NotFound(format!(
                "Model not found at: {}",
                config.model_path.display()
            )));
        }

        let model: Arc<dyn LoadedModel> = match config.engine_type {
            EngineType::Candle => Arc::new(CandleModel {
                path: config.model_path.clone(),
                name: config.name.clone(),
            }),
            #[cfg(feature = "onnx")]
            EngineType::Onnx => Arc::new(OnnxModel {
                path: config.model_path.clone(),
                name: config.name.clone(),
            }),
        };

        Ok(model)
    }

    async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, AppError> {
        if config.model_path.exists() {
            return Ok(config.model_path.clone());
        }

        let local_path = self.cache_dir.join(&config.name);
        if local_path.exists() {
            return Ok(local_path);
        }

        Err(AppError::NotFound(format!(
            "Model not found: {} (checked: {} and {})",
            config.name,
            config.model_path.display(),
            local_path.display()
        )))
    }

    async fn is_model_cached(&self, config: &ModelConfig) -> bool {
        config.model_path.exists() || self.cache_dir.join(&config.name).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_local_model_loader_creation() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        assert!(loader.cache_dir.exists());
    }

    #[test]
    fn test_local_model_loader_not_cached() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());

        let config = ModelConfig {
            name: "nonexistent-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/nonexistent/path"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(loader.is_model_cached(&config));
        assert!(!result);
    }

    #[test]
    fn test_candle_model_properties() {
        let path = PathBuf::from("/test/model");
        let model = CandleModel {
            path: path.clone(),
            name: "test-candle".to_string(),
        };

        assert_eq!(model.name(), "test-candle");
        assert_eq!(model.path(), path.as_path());
        assert_eq!(model.engine_type(), EngineType::Candle);
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_onnx_model_properties() {
        let path = PathBuf::from("/test/model");
        let model = OnnxModel {
            path: path.clone(),
            name: "test-onnx".to_string(),
        };

        assert_eq!(model.name(), "test-onnx");
        assert_eq!(model.path(), path.as_path());
        assert_eq!(model.engine_type(), EngineType::Onnx);
    }
}
