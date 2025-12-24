// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use crate::model::config::{EngineType, ModelConfig};
use crate::model::loader::{LoadedModel, LocalModelLoader, ModelLoader};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use tracing::warn;

#[derive(Clone)]
pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, Arc<dyn LoadedModel>>>>,
    loader: Arc<dyn ModelLoader>,
    default_config: ModelConfig,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::with_loader(
            Arc::new(LocalModelLoader::new(PathBuf::from("models"))) as Arc<dyn ModelLoader>
        )
    }

    pub fn with_loader(loader: Arc<dyn ModelLoader>) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            loader,
            default_config: ModelConfig::default(),
        }
    }

    pub async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, AppError> {
        let model_name = config.name.clone();

        if let Some(existing) = self.get(&model_name).await {
            info!(
                "Model {} already loaded, reusing existing instance",
                model_name
            );
            return Ok(existing);
        }

        info!("Loading model: {} from {:?}", model_name, config.model_path);
        let model = self.loader.load(config).await?;

        let mut models = self.models.write().await;
        models.insert(model_name.clone(), Arc::clone(&model));

        info!("Model {} loaded successfully", model_name);
        Ok(model)
    }

    pub async fn get(&self, name: &str) -> Option<Arc<dyn LoadedModel>> {
        let models = self.models.read().await;
        models.get(name).map(Arc::clone)
    }

    pub async fn unload(&self, name: &str) -> Result<(), AppError> {
        let mut models = self.models.write().await;

        if let Some(model) = models.remove(name) {
            info!("Model {} unloaded successfully", name);
            drop(model);
            Ok(())
        } else {
            warn!("Model {} not found for unloading", name);
            Err(AppError::NotFound(format!("Model not found: {}", name)))
        }
    }

    pub async fn unload_all(&self) {
        let mut models = self.models.write().await;
        let model_names: Vec<String> = models.keys().cloned().collect();

        for name in model_names {
            if let Some(model) = models.remove(&name) {
                info!("Unloaded model: {}", name);
                drop(model);
            }
        }

        info!("All models unloaded");
    }

    pub async fn reload(&self, name: &str) -> Result<Arc<dyn LoadedModel>, AppError> {
        let config = {
            let models = self.models.read().await;
            let model = models
                .get(name)
                .ok_or_else(|| AppError::NotFound(format!("Model not found: {}", name)))?;

            ModelConfig {
                name: name.to_string(),
                engine_type: model.engine_type(),
                model_path: model.path().to_path_buf(),
                tokenizer_path: None,
                device: crate::model::config::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
            }
        };

        self.unload(name).await?;
        self.load(&config).await
    }

    pub async fn count(&self) -> usize {
        self.models.read().await.len()
    }

    pub async fn is_loaded(&self, name: &str) -> bool {
        self.models.read().await.contains_key(name)
    }

    pub async fn list_loaded(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    pub async fn stats(&self) -> ModelStats {
        let models = self.models.read().await;

        let candle_count = models
            .values()
            .filter(|m| m.engine_type() == EngineType::Candle)
            .count();

        let onnx_count = models
            .values()
            .filter(|m| m.engine_type() == EngineType::Onnx)
            .count();

        let mut total_model_size = 0;
        for model in models.values() {
            if let Ok(metadata) = std::fs::metadata(model.path()) {
                total_model_size += metadata.len();
            }
        }

        ModelStats {
            total_models: models.len(),
            candle_models: candle_count,
            onnx_models: onnx_count,
            total_size_bytes: total_model_size,
        }
    }

    pub fn set_default_config(&mut self, config: ModelConfig) {
        self.default_config = config;
    }

    pub async fn load_default(&self) -> Result<Arc<dyn LoadedModel>, AppError> {
        self.load(&self.default_config).await
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ModelStats {
    pub total_models: usize,
    pub candle_models: usize,
    pub onnx_models: usize,
    pub total_size_bytes: u64,
}

impl ModelStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn format_size(&self) -> String {
        if self.total_size_bytes < 1024 {
            format!("{} B", self.total_size_bytes)
        } else if self.total_size_bytes < 1024 * 1024 {
            format!("{} KB", self.total_size_bytes / 1024)
        } else if self.total_size_bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", self.total_size_bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!(
                "{:.2} GB",
                self.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn create_test_model_file(path: &PathBuf) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "test model content").unwrap();
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_with_loader() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_load_unload() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("test-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::model::config::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        };

        let _model = manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("test-model").await);

        manager.unload("test-model").await.unwrap();
        assert_eq!(manager.count().await, 0);
        assert!(!manager.is_loaded("test-model").await);
    }

    #[tokio::test]
    async fn test_model_manager_list_loaded() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config1 = ModelConfig {
            name: "model-1".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("model-1"),
            tokenizer_path: None,
            device: crate::model::config::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        };

        let config2 = ModelConfig {
            name: "model-2".to_string(),
            engine_type: EngineType::Onnx,
            model_path: cache_dir.path().join("model-2"),
            tokenizer_path: None,
            device: crate::model::config::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        };

        fs::create_dir_all(&config1.model_path).unwrap();
        fs::create_dir_all(&config2.model_path).unwrap();

        manager.load(&config1).await.unwrap();
        manager.load(&config2).await.unwrap();

        let loaded = manager.list_loaded().await;
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(&"model-1".to_string()));
        assert!(loaded.contains(&"model-2".to_string()));
    }

    #[tokio::test]
    async fn test_model_manager_stats() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let stats = manager.stats().await;
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.candle_models, 0);
        assert_eq!(stats.onnx_models, 0);
    }

    #[tokio::test]
    async fn test_model_manager_unload_all() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("test-model"),
            tokenizer_path: None,
            device: crate::model::config::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        };

        fs::create_dir_all(&config.model_path).unwrap();
        manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);

        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_unload_nonexistent() {
        let manager = ModelManager::new();
        let result = manager.unload("nonexistent").await;
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("not found"));
        }
    }

    #[tokio::test]
    async fn test_model_stats_format_size() {
        let stats = ModelStats {
            total_models: 1,
            candle_models: 1,
            onnx_models: 0,
            total_size_bytes: 1024,
        };

        assert_eq!(stats.format_size(), "1 KB");
    }
}
