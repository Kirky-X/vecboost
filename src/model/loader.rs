// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::config::model::{EngineType, ModelConfig};
use crate::error::VecboostError;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

#[async_trait]
pub trait ModelLoader: Send + Sync {
    async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError>;
    async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError>;
    async fn is_model_cached(&self, config: &ModelConfig) -> bool;
}

pub trait LoadedModel: Send + Sync {
    fn name(&self) -> &str;
    fn path(&self) -> &Path;
    fn engine_type(&self) -> EngineType;
    fn reload(&self) -> Result<(), VecboostError>;
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

    fn reload(&self) -> Result<(), VecboostError> {
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

    fn reload(&self) -> Result<(), VecboostError> {
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
    async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        if !config.model_path.exists() {
            return Err(VecboostError::NotFound(format!(
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

    async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError> {
        if config.model_path.exists() {
            return Ok(config.model_path.clone());
        }

        let local_path = self.cache_dir.join(&config.name);
        if local_path.exists() {
            return Ok(local_path);
        }

        Err(VecboostError::NotFound(format!(
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

    #[test]
    fn test_candle_model_reload_succeeds() {
        let path = PathBuf::from("/test/model");
        let model = CandleModel {
            path,
            name: "test-candle".to_string(),
        };
        assert!(model.reload().is_ok());
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_onnx_model_reload_succeeds() {
        let path = PathBuf::from("/test/model");
        let model = OnnxModel {
            path,
            name: "test-onnx".to_string(),
        };
        assert!(model.reload().is_ok());
    }

    fn make_config(name: &str, model_path: PathBuf) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        }
    }

    #[tokio::test]
    async fn test_load_candle_success() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("candle-model");
        std::fs::create_dir_all(&model_path).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("candle-model", model_path.clone());

        let model = loader.load(&config).await.unwrap();
        assert_eq!(model.name(), "candle-model");
        assert_eq!(model.path(), model_path.as_path());
        assert_eq!(model.engine_type(), EngineType::Candle);
    }

    #[tokio::test]
    #[cfg(feature = "onnx")]
    async fn test_load_onnx_success() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("onnx-model");
        std::fs::create_dir_all(&model_path).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let mut config = make_config("onnx-model", model_path.clone());
        config.engine_type = EngineType::Onnx;

        let model = loader.load(&config).await.unwrap();
        assert_eq!(model.name(), "onnx-model");
        assert_eq!(model.path(), model_path.as_path());
        assert_eq!(model.engine_type(), EngineType::Onnx);
    }

    #[tokio::test]
    async fn test_load_nonexistent_path_fails() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("missing", PathBuf::from("/nonexistent/model"));

        let result = loader.load(&config).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            VecboostError::NotFound(msg) => assert!(msg.contains("Model not found")),
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_get_model_path_from_config_path() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("direct-model");
        std::fs::create_dir_all(&model_path).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("direct-model", model_path.clone());

        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, model_path);
    }

    #[tokio::test]
    async fn test_get_model_path_from_cache_dir() {
        let cache_dir = tempdir().unwrap();
        let cached_model = cache_dir.path().join("cached-model");
        std::fs::create_dir_all(&cached_model).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("cached-model", PathBuf::from("/nonexistent/direct"));

        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, cached_model);
    }

    #[tokio::test]
    async fn test_get_model_path_not_found_anywhere() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("missing", PathBuf::from("/nonexistent/direct"));

        let result = loader.get_model_path(&config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::NotFound(msg) => {
                assert!(msg.contains("Model not found"));
                assert!(msg.contains("missing"));
            }
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_is_model_cached_true_via_config_path() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("cached-via-config");
        std::fs::create_dir_all(&model_path).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("any", model_path);

        assert!(loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_is_model_cached_true_via_cache_dir() {
        let cache_dir = tempdir().unwrap();
        let cached_model = cache_dir.path().join("in-cache-dir");
        std::fs::create_dir_all(&cached_model).unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("in-cache-dir", PathBuf::from("/nonexistent/direct"));

        assert!(loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_load_candle_model_with_file_path() {
        let cache_dir = tempdir().unwrap();
        let model_file = cache_dir.path().join("model.safetensors");
        std::fs::write(&model_file, "dummy").unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("file-model", model_file.clone());

        let model = loader.load(&config).await.unwrap();
        assert_eq!(model.name(), "file-model");
        assert_eq!(model.path(), model_file.as_path());
    }

    #[tokio::test]
    async fn test_get_model_path_prefers_config_path_over_cache() {
        let cache_dir = tempdir().unwrap();
        let config_path = cache_dir.path().join("config-model");
        std::fs::create_dir_all(&config_path).unwrap();
        let cached_path = cache_dir.path().join("dual-model");
        std::fs::create_dir_all(&cached_path).unwrap();

        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("dual-model", config_path.clone());

        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, config_path);
    }

    #[tokio::test]
    async fn test_is_model_cached_both_paths_exist() {
        let cache_dir = tempdir().unwrap();
        let config_path = cache_dir.path().join("config-path");
        std::fs::create_dir_all(&config_path).unwrap();
        let cached_path = cache_dir.path().join("both-model");
        std::fs::create_dir_all(&cached_path).unwrap();

        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("both-model", config_path);

        assert!(loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_candle_model_reload_ok() {
        let path = PathBuf::from("/test/candle/reload");
        let model = CandleModel {
            path,
            name: "reload-test".to_string(),
        };
        assert!(model.reload().is_ok());
    }

    #[test]
    #[cfg(feature = "onnx")]
    fn test_onnx_model_reload_ok() {
        let path = PathBuf::from("/test/onnx/reload");
        let model = OnnxModel {
            path,
            name: "onnx-reload-test".to_string(),
        };
        assert!(model.reload().is_ok());
    }

    #[tokio::test]
    async fn test_load_multiple_models_independently() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());

        let path1 = cache_dir.path().join("model-1");
        let path2 = cache_dir.path().join("model-2");
        std::fs::create_dir_all(&path1).unwrap();
        std::fs::create_dir_all(&path2).unwrap();

        let config1 = make_config("model-1", path1.clone());
        let config2 = make_config("model-2", path2.clone());

        let model1 = loader.load(&config1).await.unwrap();
        let model2 = loader.load(&config2).await.unwrap();

        assert_ne!(model1.name(), model2.name());
        assert_ne!(model1.path(), model2.path());
    }

    #[tokio::test]
    async fn test_get_model_path_error_contains_both_paths() {
        let cache_dir = tempdir().unwrap();
        let loader = LocalModelLoader::new(cache_dir.path().to_path_buf());
        let config = make_config("missing-model", PathBuf::from("/nonexistent/one"));

        let result = loader.get_model_path(&config).await;
        let err = result.expect_err("should error");
        match err {
            VecboostError::NotFound(msg) => {
                assert!(msg.contains("/nonexistent/one"));
                assert!(msg.contains("missing-model"));
            }
            other => panic!("expected NotFound, got {:?}", other),
        }
    }
}
