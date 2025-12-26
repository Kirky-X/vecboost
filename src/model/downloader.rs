// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    HuggingFace,
    ModelScope,
}

#[derive(Debug, Clone)]
pub struct ModelDownloadConfig {
    pub source: ModelSource,
    pub repo_id: String,
    pub revision: String,
    pub cache_dir: Option<PathBuf>,
    pub file_patterns: Vec<String>,
}

impl Default for ModelDownloadConfig {
    fn default() -> Self {
        Self {
            source: ModelSource::HuggingFace,
            repo_id: "BAAI/bge-m3".to_string(),
            revision: "main".to_string(),
            cache_dir: None,
            file_patterns: vec![
                "config.json".to_string(),
                "tokenizer.json".to_string(),
                "model.safetensors".to_string(),
                "model.onnx".to_string(),
            ],
        }
    }
}

#[derive(Clone)]
pub struct DownloadProgress {
    pub file_name: String,
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub percentage: f64,
}

impl std::fmt::Debug for DownloadProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadProgress")
            .field("file_name", &self.file_name)
            .field("total_bytes", &self.total_bytes)
            .field("downloaded_bytes", &self.downloaded_bytes)
            .field("percentage", &self.percentage)
            .finish()
    }
}

pub struct ModelDownloader {
    config: ModelDownloadConfig,
    hf_api: Option<Api>,
    progress: Arc<Mutex<DownloadProgress>>,
}

impl ModelDownloader {
    pub fn new(config: ModelDownloadConfig) -> Result<Self, AppError> {
        let hf_api = if config.source == ModelSource::HuggingFace {
            let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
            Some(api)
        } else {
            None
        };

        Ok(Self {
            config,
            hf_api,
            progress: Arc::new(Mutex::new(DownloadProgress {
                file_name: String::new(),
                total_bytes: 0,
                downloaded_bytes: 0,
                percentage: 0.0,
            })),
        })
    }

    pub fn with_huggingface(repo_id: &str, revision: &str) -> Result<Self, AppError> {
        let config = ModelDownloadConfig {
            source: ModelSource::HuggingFace,
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn with_modelscope(repo_id: &str, revision: &str) -> Result<Self, AppError> {
        let config = ModelDownloadConfig {
            source: ModelSource::ModelScope,
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            ..Default::default()
        };
        Self::new(config)
    }

    pub async fn download(&self) -> Result<Vec<PathBuf>, AppError> {
        match self.config.source {
            ModelSource::HuggingFace => self.download_from_huggingface().await,
            ModelSource::ModelScope => self.download_from_modelscope().await,
        }
    }

    async fn download_from_huggingface(&self) -> Result<Vec<PathBuf>, AppError> {
        let Some(ref api) = self.hf_api else {
            return Err(AppError::ModelLoadError(
                "HuggingFace API not initialized".to_string(),
            ));
        };

        let repo = api.repo(Repo::new(
            self.config.repo_id.clone(),
            RepoType::Model,
        ));

        let mut downloaded_files = Vec::new();

        for pattern in &self.config.file_patterns {
            info!("Downloading {}...", pattern);

            let file_path = repo
                .get(pattern)
                .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

            downloaded_files.push(file_path);
            info!("Downloaded: {}", pattern);
        }

        Ok(downloaded_files)
    }

    async fn download_from_modelscope(&self) -> Result<Vec<PathBuf>, AppError> {
        let ms_url = format!(
            "https://modelscope.cn/api/v1/models/{}/repo?Revision={}",
            self.config.repo_id,
            self.config.revision
        );

        info!("ModelScope URL: {}", ms_url);

        let client = reqwest::Client::new();

        let mut downloaded_files = Vec::new();

        for pattern in &self.config.file_patterns {
            info!("Attempting to download {} from ModelScope...", pattern);

            let file_url = format!(
                "https://modelscope.cn/api/v1/models/{}/raw/main/{}?Revision={}",
                self.config.repo_id, pattern, self.config.revision
            );

            let response = client
                .get(&file_url)
                .send()
                .await
                .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

            if response.status().is_success() {
                let cache_dir = self.config.cache_dir.clone().unwrap_or_else(|| {
                    PathBuf::from(home::home_dir().unwrap_or_default()).join(".cache/vecboost/models")
                });

                let file_path = cache_dir.join(&self.config.repo_id.replace('/', "_")).join(pattern);

                if let Some(parent) = file_path.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                }

                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

                std::fs::write(&file_path, &bytes)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

                let display_path = file_path.clone();
                downloaded_files.push(file_path);
                info!("Downloaded: {} to {:?}", pattern, display_path);
            } else {
                info!("File {} not found on ModelScope, skipping...", pattern);
            }
        }

        if downloaded_files.is_empty() {
            return Err(AppError::ModelLoadError(
                "No files could be downloaded from ModelScope".to_string(),
            ));
        }

        Ok(downloaded_files)
    }

    pub async fn get_progress(&self) -> DownloadProgress {
        let guard = self.progress.lock().await;
        DownloadProgress {
            file_name: guard.file_name.clone(),
            total_bytes: guard.total_bytes,
            downloaded_bytes: guard.downloaded_bytes,
            percentage: guard.percentage,
        }
    }

    pub fn set_cache_dir(&mut self, path: PathBuf) {
        self.config.cache_dir = Some(path);
    }

    pub fn add_file_pattern(&mut self, pattern: &str) {
        self.config.file_patterns.push(pattern.to_string());
    }

    pub fn get_config(&self) -> &ModelDownloadConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_download_config_default() {
        let config = ModelDownloadConfig::default();
        assert_eq!(config.source, ModelSource::HuggingFace);
        assert_eq!(config.repo_id, "BAAI/bge-m3");
        assert!(config.file_patterns.contains(&"config.json".to_string()));
    }

    #[test]
    fn test_model_source_equality() {
        let hf = ModelSource::HuggingFace;
        let ms = ModelSource::ModelScope;
        assert_eq!(hf, ModelSource::HuggingFace);
        assert_eq!(ms, ModelSource::ModelScope);
        assert_ne!(hf, ms);
    }

    #[test]
    fn test_download_progress() {
        let progress = DownloadProgress {
            file_name: "model.safetensors".to_string(),
            total_bytes: 1024,
            downloaded_bytes: 512,
            percentage: 50.0,
        };
        assert_eq!(progress.percentage, 50.0);
    }
}
