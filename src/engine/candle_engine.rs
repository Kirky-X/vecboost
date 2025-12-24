// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::ModelConfig;
use crate::error::AppError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

pub struct CandleEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEngine {
    pub fn new(config: &ModelConfig) -> Result<Self, AppError> {
        let device = if config.use_gpu && candle_core::utils::cuda_is_available() {
            tracing::info!("Using CUDA GPU");
            Device::new_cuda(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else {
            tracing::info!("Using CPU");
            Device::Cpu
        };

        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));

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

        Ok(Self {
            model,
            tokenizer,
            device,
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

        // Perform Mean Pooling (taking the first token [CLS] is another strategy, but mean is common for BGE)
        // Here we implement CLS pooling for simplicity as BGE-M3 often works well with it,
        // but for strict correctness with BGE, we usually do Mean Pooling over attention mask.
        // Let's do CLS (index 0) for this implementation to keep it fast and standard for BERT-likes.
        let (_n_sentence, _n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

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
