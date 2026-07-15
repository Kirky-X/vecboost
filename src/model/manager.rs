// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::config::model::{EngineType, ModelConfig};
use crate::error::VecboostError;
use crate::model::loader::{LoadedModel, LocalModelLoader, ModelLoader};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::info;
use tracing::warn;

const DEFAULT_MODEL_LOAD_TIMEOUT_SECS: u64 = 300;

#[derive(Clone)]
pub struct ModelManager {
    models: Arc<RwLock<HashMap<String, Arc<dyn LoadedModel>>>>,
    loader: Arc<dyn ModelLoader>,
    default_config: ModelConfig,
    timeout_duration: Duration,
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
            timeout_duration: Duration::from_secs(DEFAULT_MODEL_LOAD_TIMEOUT_SECS),
        }
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_duration = Duration::from_secs(timeout_secs);
        self
    }

    pub async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        let model_name = config.name.clone();

        if let Some(existing) = self.get(&model_name).await {
            info!(
                "Model {} already loaded, reusing existing instance",
                model_name
            );
            return Ok(existing);
        }

        info!(
            "Loading model: {} from {:?} (timeout: {:?})",
            model_name, config.model_path, self.timeout_duration
        );

        let load_future = self.loader.load(config);
        match timeout(self.timeout_duration, load_future).await {
            Ok(Ok(model)) => {
                let mut models = self.models.write().await;
                models.insert(model_name.clone(), Arc::clone(&model));

                info!("Model {} loaded successfully", model_name);
                Ok(model)
            }
            Ok(Err(e)) => {
                warn!("Model {} loading failed: {}", model_name, e);
                Err(VecboostError::ModelLoadError(format!(
                    "Failed to load model {}: {}",
                    model_name, e
                )))
            }
            Err(_) => {
                warn!(
                    "Model {} loading timed out after {:?}",
                    model_name, self.timeout_duration
                );
                Err(VecboostError::ModelLoadError(format!(
                    "Model loading timed out after {} seconds: {}",
                    self.timeout_duration.as_secs(),
                    model_name
                )))
            }
        }
    }

    pub async fn get(&self, name: &str) -> Option<Arc<dyn LoadedModel>> {
        let models = self.models.read().await;
        models.get(name).map(Arc::clone)
    }

    pub async fn unload(&self, name: &str) -> Result<(), VecboostError> {
        let mut models = self.models.write().await;

        if let Some(model) = models.remove(name) {
            info!("Model {} unloaded successfully", name);
            drop(model);
            Ok(())
        } else {
            warn!("Model {} not found for unloading", name);
            Err(VecboostError::NotFound(format!(
                "Model not found: {}",
                name
            )))
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

    pub async fn reload(&self, name: &str) -> Result<Arc<dyn LoadedModel>, VecboostError> {
        let config = {
            let models = self.models.read().await;
            let model = models
                .get(name)
                .ok_or_else(|| VecboostError::NotFound(format!("Model not found: {}", name)))?;

            ModelConfig {
                name: name.to_string(),
                engine_type: model.engine_type(),
                model_path: model.path().to_path_buf(),
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
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
            .filter(|_m| {
                #[cfg(feature = "onnx")]
                {
                    _m.engine_type() == EngineType::Onnx
                }
                #[cfg(not(feature = "onnx"))]
                {
                    false
                }
            })
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

    pub async fn load_default(&self) -> Result<Arc<dyn LoadedModel>, VecboostError> {
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
    use async_trait::async_trait;
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    struct SlowModelLoader {
        delay_ms: u64,
    }

    impl SlowModelLoader {
        fn new(delay_ms: u64) -> Self {
            Self { delay_ms }
        }
    }

    #[async_trait]
    impl ModelLoader for SlowModelLoader {
        async fn load(&self, config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;

            let model: Arc<dyn LoadedModel> = Arc::new(CandleModel {
                path: config.model_path.clone(),
                name: config.name.clone(),
            });

            Ok(model)
        }

        async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError> {
            Ok(config.model_path.clone())
        }

        async fn is_model_cached(&self, _config: &ModelConfig) -> bool {
            true
        }
    }

    struct CandleModel {
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
            Ok(())
        }
    }

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
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
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
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let config2 = ModelConfig {
            name: "model-2".to_string(),
            #[cfg(feature = "onnx")]
            engine_type: EngineType::Onnx,
            #[cfg(not(feature = "onnx"))]
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("model-2"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        fs::create_dir_all(&config1.model_path).unwrap();
        fs::create_dir_all(&config2.model_path).unwrap();

        manager.load(&config1).await.unwrap();
        #[cfg(feature = "onnx")]
        manager.load(&config2).await.unwrap();

        let loaded = manager.list_loaded().await;
        #[cfg(feature = "onnx")]
        {
            assert_eq!(loaded.len(), 2);
            assert!(loaded.contains(&"model-1".to_string()));
            assert!(loaded.contains(&"model-2".to_string()));
        }
        #[cfg(not(feature = "onnx"))]
        {
            assert_eq!(loaded.len(), 1);
            assert!(loaded.contains(&"model-1".to_string()));
        }
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
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
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

    #[tokio::test]
    async fn test_model_manager_with_timeout() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(1);

        assert_eq!(manager.timeout_duration.as_secs(), 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_timeout_success() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(30);

        let config = ModelConfig {
            name: "timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;
        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_reuses_existing() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader).with_timeout(30);

        let config = ModelConfig {
            name: "reuse-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("reuse-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let first = manager.load(&config).await.unwrap();
        let second = manager.load(&config).await.unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_model_manager_load_timeout_failure() {
        let cache_dir = tempdir().unwrap();
        let slow_loader = Arc::new(SlowModelLoader::new(2000));
        let manager = ModelManager::with_loader(slow_loader).with_timeout(1);

        let config = ModelConfig {
            name: "slow-timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("slow-timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;

        assert!(result.is_err());
        match result {
            Err(e) => {
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("timed out"),
                    "Expected 'timed out' in error message, got: {}",
                    error_msg
                );
            }
            Ok(_) => panic!("Expected error, but got success"),
        }
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_model_manager_load_success_after_timeout() {
        let cache_dir = tempdir().unwrap();
        let slow_loader = Arc::new(SlowModelLoader::new(100));
        let manager = ModelManager::with_loader(slow_loader).with_timeout(5);

        let config = ModelConfig {
            name: "success-after-timeout-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("success-after-timeout-test-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        fs::create_dir_all(&config.model_path).unwrap();

        let result = manager.load(&config).await;

        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_get_returns_none_for_unknown_model() {
        let manager = ModelManager::new();
        assert!(manager.get("does-not-exist").await.is_none());
    }

    #[tokio::test]
    async fn test_get_returns_loaded_model() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("get-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "get-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let loaded = manager.load(&config).await.unwrap();
        let fetched = manager.get("get-model").await;
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().name(), loaded.name());
    }

    #[tokio::test]
    async fn test_is_loaded_returns_false_for_unknown() {
        let manager = ModelManager::new();
        assert!(!manager.is_loaded("unknown").await);
    }

    #[tokio::test]
    async fn test_default_creates_empty_manager() {
        let manager = ModelManager::default();
        assert_eq!(manager.count().await, 0);
        assert!(manager.list_loaded().await.is_empty());
    }

    #[tokio::test]
    async fn test_unload_all_on_empty_manager_is_noop() {
        let manager = ModelManager::new();
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_unload_all_removes_multiple_models() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        for name in ["multi-1", "multi-2", "multi-3"] {
            let model_path = cache_dir.path().join(name);
            create_test_model_file(&model_path);
            let config = ModelConfig {
                name: name.to_string(),
                engine_type: EngineType::Candle,
                model_path,
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
            };
            manager.load(&config).await.unwrap();
        }

        assert_eq!(manager.count().await, 3);
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
        assert!(manager.list_loaded().await.is_empty());
    }

    #[tokio::test]
    async fn test_reload_existing_model_succeeds() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("reload-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "reload-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let _ = manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);

        let reloaded = manager.reload("reload-model").await;
        assert!(reloaded.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("reload-model").await);
    }

    #[tokio::test]
    async fn test_reload_unknown_model_returns_not_found() {
        let manager = ModelManager::new();
        let result = manager.reload("no-such-model").await;
        assert!(result.is_err());
        match result.err().unwrap() {
            VecboostError::NotFound(msg) => {
                assert!(msg.contains("no-such-model"));
            }
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    struct FailingLoader;

    #[async_trait]
    impl ModelLoader for FailingLoader {
        async fn load(&self, _config: &ModelConfig) -> Result<Arc<dyn LoadedModel>, VecboostError> {
            Err(VecboostError::ModelLoadError(
                "intentional failure".to_string(),
            ))
        }

        async fn get_model_path(&self, config: &ModelConfig) -> Result<PathBuf, VecboostError> {
            Ok(config.model_path.clone())
        }

        async fn is_model_cached(&self, _config: &ModelConfig) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn test_load_with_loader_failure_returns_model_load_error() {
        let loader = Arc::new(FailingLoader);
        let manager = ModelManager::with_loader(loader);

        let cache_dir = tempdir().unwrap();
        let config = ModelConfig {
            name: "fail-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("fail-model"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let result = manager.load(&config).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("fail-model"));
                assert!(msg.contains("intentional failure"));
            }
            other => panic!("expected ModelLoadError, got {:?}", other),
        }
        assert_eq!(manager.count().await, 0);
    }

    #[tokio::test]
    async fn test_load_default_uses_default_config() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("default-test-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let mut manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "default-test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        manager.set_default_config(config);

        let result = manager.load_default().await;
        assert!(result.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("default-test-model").await);
    }

    #[tokio::test]
    async fn test_stats_counts_candle_models() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        for name in ["stat-1", "stat-2"] {
            let model_path = cache_dir.path().join(name);
            create_test_model_file(&model_path);
            let config = ModelConfig {
                name: name.to_string(),
                engine_type: EngineType::Candle,
                model_path,
                tokenizer_path: None,
                device: crate::config::model::DeviceType::Cpu,
                max_batch_size: 32,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: false,
                model_sha256: None,
            };
            manager.load(&config).await.unwrap();
        }

        let stats = manager.stats().await;
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.candle_models, 2);
        assert_eq!(stats.onnx_models, 0);
        assert!(stats.total_size_bytes > 0);
    }

    #[tokio::test]
    async fn test_concurrent_loads_distinct_models() {
        let cache_dir = tempdir().unwrap();
        let cache_root = cache_dir.path().to_path_buf();

        let mut handles = Vec::new();
        let loader = Arc::new(LocalModelLoader::new(cache_root.clone()));
        let manager = std::sync::Arc::new(ModelManager::with_loader(loader));

        for i in 0..5 {
            let mgr = std::sync::Arc::clone(&manager);
            let root = cache_root.clone();
            handles.push(tokio::spawn(async move {
                let name = format!("conc-{}", i);
                let model_path = root.join(&name);
                fs::create_dir_all(model_path.parent().unwrap()).unwrap();
                fs::write(&model_path, "content").unwrap();

                let config = ModelConfig {
                    name: name.clone(),
                    engine_type: EngineType::Candle,
                    model_path,
                    tokenizer_path: None,
                    device: crate::config::model::DeviceType::Cpu,
                    max_batch_size: 32,
                    pooling_mode: None,
                    expected_dimension: None,
                    memory_limit_bytes: None,
                    oom_fallback_enabled: false,
                    model_sha256: None,
                };
                mgr.load(&config).await.unwrap();
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(manager.count().await, 5);
        let loaded = manager.list_loaded().await;
        for i in 0..5 {
            assert!(loaded.contains(&format!("conc-{}", i)));
        }
    }

    #[test]
    fn test_model_stats_total_size_mb() {
        let stats = ModelStats {
            total_models: 1,
            candle_models: 1,
            onnx_models: 0,
            total_size_bytes: 2 * 1024 * 1024,
        };
        assert_eq!(stats.total_size_mb(), 2.0);
    }

    #[test]
    fn test_model_stats_format_size_bytes() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 512,
        };
        assert_eq!(stats.format_size(), "512 B");
    }

    #[test]
    fn test_model_stats_format_size_kb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 2048,
        };
        assert_eq!(stats.format_size(), "2 KB");
    }

    #[test]
    fn test_model_stats_format_size_mb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 1024 * 1024,
        };
        assert!(stats.format_size().ends_with("MB"));
    }

    #[test]
    fn test_model_stats_format_size_gb() {
        let stats = ModelStats {
            total_models: 0,
            candle_models: 0,
            onnx_models: 0,
            total_size_bytes: 1024 * 1024 * 1024,
        };
        assert!(stats.format_size().ends_with("GB"));
    }

    #[tokio::test]
    async fn test_load_already_loaded_returns_same_instance() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("reuse-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "reuse-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let first = manager.load(&config).await.unwrap();
        let second = manager.load(&config).await.unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(manager.count().await, 1);
    }

    #[tokio::test]
    async fn test_unload_then_load_recreates_model() {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join("recreate-model");
        create_test_model_file(&model_path);

        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let config = ModelConfig {
            name: "recreate-model".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };

        let _first = manager.load(&config).await.unwrap();
        manager.unload("recreate-model").await.unwrap();
        assert_eq!(manager.count().await, 0);

        let second = manager.load(&config).await;
        assert!(second.is_ok());
        assert_eq!(manager.count().await, 1);
        assert!(manager.is_loaded("recreate-model").await);
    }

    #[tokio::test]
    async fn test_slow_model_loader_get_model_path_and_cached() {
        let loader = SlowModelLoader::new(10);
        let config = ModelConfig {
            name: "slow-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/slow/path"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, PathBuf::from("/slow/path"));
        assert!(loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_slow_model_loader_load_succeeds() {
        let cache_dir = tempdir().unwrap();
        let loader = SlowModelLoader::new(10);
        let config = ModelConfig {
            name: "slow-load".to_string(),
            engine_type: EngineType::Candle,
            model_path: cache_dir.path().join("slow-load"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        let model = loader.load(&config).await.unwrap();
        assert_eq!(model.name(), "slow-load");
        assert_eq!(model.engine_type(), EngineType::Candle);
        assert!(model.reload().is_ok());
    }

    #[tokio::test]
    async fn test_failing_loader_get_model_path_and_cached() {
        let loader = FailingLoader;
        let config = ModelConfig {
            name: "fail-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/fail/path"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        let path = loader.get_model_path(&config).await.unwrap();
        assert_eq!(path, PathBuf::from("/fail/path"));
        assert!(!loader.is_model_cached(&config).await);
    }

    #[tokio::test]
    async fn test_unload_all_on_manager_with_single_model_logs() {
        let cache_dir = tempdir().unwrap();
        let loader = Arc::new(LocalModelLoader::new(cache_dir.path().to_path_buf()));
        let manager = ModelManager::with_loader(loader);

        let model_path = cache_dir.path().join("unload-single");
        create_test_model_file(&model_path);
        let config = ModelConfig {
            name: "unload-single".to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        manager.load(&config).await.unwrap();
        assert_eq!(manager.count().await, 1);
        manager.unload_all().await;
        assert_eq!(manager.count().await, 0);
    }
}
