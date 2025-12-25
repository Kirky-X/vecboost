// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::model::{DeviceType, ModelConfig};
use crate::error::AppError;
use crate::monitor::MemoryMonitor;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
use tokenizers::{PaddingParams, Tokenizer};

pub struct CandleEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    memory_monitor: Option<Arc<MemoryMonitor>>,
}

impl CandleEngine {
    pub fn new(config: &ModelConfig) -> Result<Self, AppError> {
        let device = if config.device == DeviceType::Cuda && candle_core::utils::cuda_is_available()
        {
            tracing::info!("Using CUDA GPU");
            Device::new_cuda(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else {
            tracing::info!("Using CPU");
            Device::Cpu
        };

        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(
            config.model_path.to_string_lossy().into_owned(),
            RepoType::Model,
        ));

        tracing::info!("Downloading/Loading model files...");
        let config_filename = repo
            .get("config.json")
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let weights_filename = repo
            .get("model.safetensors")
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        // Load Config
        let config_content = std::fs::read_to_string(config_filename)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        // Load Tokenizer
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // Load Model
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)
        }
        .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let memory_monitor = if device.is_cuda() {
            Some(Arc::new(MemoryMonitor::new()))
        } else {
            None
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            memory_monitor,
        })
    }

    fn forward_pass(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AppError::TokenizationError(e.to_string()))?;

        let token_ids = Tensor::new(tokens.get_ids(), &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let token_type_ids = token_ids
            .zeros_like()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, None)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        if let Some(ref _monitor) = self.memory_monitor {
            #[cfg(feature = "cuda")]
            _monitor.update_gpu_memory_from_candle().await;
        }

        // Use CLS pooling (first token) for simplicity and reliability
        let cls_embedding = embeddings
            .get(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .get(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let vec = cls_embedding
            .to_vec1::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        Ok(vec)
    }
}

impl InferenceEngine for CandleEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        self.forward_pass(text)
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.forward_pass(text)?);
        }
        Ok(results)
    }
}
