// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::model::{DeviceType, ModelConfig, Precision};
use crate::error::AppError;
use crate::monitor::MemoryMonitor;
use crate::text::CachedTokenizer;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
use tokenizers::Tokenizer as HfTokenizer;

pub struct CandleEngine {
    model: BertModel,
    tokenizer: CachedTokenizer,
    device: Device,
    precision: Precision,
    memory_monitor: Option<Arc<MemoryMonitor>>,
}

impl CandleEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, AppError> {
        let device = if config.device == DeviceType::Cuda && candle_core::utils::cuda_is_available()
        {
            tracing::info!("Using CUDA GPU");
            Device::new_cuda(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else {
            tracing::info!("Using CPU");
            Device::Cpu
        };

        let dtype = match precision {
            Precision::Fp32 => DType::F32,
            Precision::Fp16 => {
                if device.is_cuda() {
                    tracing::info!("Using FP16 precision");
                    DType::F16
                } else {
                    tracing::warn!("FP16 not supported on CPU, falling back to FP32");
                    DType::F32
                }
            }
            Precision::Int8 => {
                tracing::warn!("INT8 precision not fully supported, falling back to FP32");
                DType::F32
            }
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

        let config_content = std::fs::read_to_string(config_filename)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let hf_tokenizer = HfTokenizer::from_file(tokenizer_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let tokenizer = CachedTokenizer::new(hf_tokenizer, bert_config.max_position_embeddings, 2048);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)
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
            precision,
            memory_monitor,
        })
    }

    async fn forward_pass(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .await
            .map_err(|e| AppError::TokenizationError(e.to_string()))?;

        let token_ids = Tensor::new(encoding.get_ids(), &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let embeddings = self
            .model
            .forward(&token_ids, &attention_mask, None)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        if let Some(ref _monitor) = self.memory_monitor {
            #[cfg(feature = "cuda")]
            _monitor.update_gpu_memory_from_candle().await;
        }

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

    fn precision(&self) -> Precision {
        self.precision.clone()
    }

    fn supports_mixed_precision(&self) -> bool {
        self.device.is_cuda()
    }
}
