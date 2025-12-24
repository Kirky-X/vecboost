// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::ModelConfig;
use crate::error::AppError;
use hf_hub::{api::sync::Api, Repo, RepoType};
use ndarray::Array1;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::{PaddingParams, Tokenizer};

pub struct OnnxEngine {
    session: Session,
    tokenizer: Tokenizer,
    hidden_size: usize,
    max_input_length: usize,
}

impl OnnxEngine {
    pub fn new(config: &ModelConfig) -> Result<Self, AppError> {
        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(config.model_repo.clone(), RepoType::Model));

        tracing::info!("Downloading/Loading ONNX model files...");
        let onnx_filename = repo
            .get("model.onnx")
            .or_else(|_| repo.get("model_quantized.onnx"))
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        tracing::info!("Initializing ONNX Runtime session...");
        let session = Session::builder()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .commit_from_file(onnx_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        tracing::info!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vocab_size = tokenizer.get_vocab_size(false);
        let hidden_size = 768;
        let max_input_length = std::cmp::min(vocab_size, 512);

        tracing::info!(
            "ONNX Engine initialized: hidden_size={}, max_input_length={}, vocab_size={}",
            hidden_size,
            max_input_length,
            vocab_size
        );

        Ok(Self {
            session,
            tokenizer,
            hidden_size,
            max_input_length,
        })
    }

    fn forward_pass(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| AppError::TokenizationError(e.to_string()))?;

        let input_ids: Vec<i64> = tokens
            .get_ids()
            .iter()
            .take(self.max_input_length)
            .map(|&id| id as i64)
            .collect();

        let attention_mask: Vec<i64> = tokens
            .get_attention_mask()
            .iter()
            .take(self.max_input_length)
            .map(|&v| v as i64)
            .collect();

        let attention_mask_clone = attention_mask.clone();
        let input_ids_array = Array1::from(input_ids);
        let attention_mask_array = Array1::from(attention_mask);

        let input_ids_tensor = Tensor::from_array(input_ids_array.into_dyn())
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.into_dyn())
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let last_hidden_state = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let seq_len = attention_mask_clone.iter().filter(|&&v| v == 1).count();
        if seq_len == 0 {
            return Err(AppError::InferenceError(
                "Empty sequence after mask".to_string(),
            ));
        }

        let mut weighted_sum = vec![0.0f32; self.hidden_size];
        let mut mask_sum = 0.0f32;

        for (seq_idx, &mask_val) in attention_mask_clone.iter().enumerate().take(seq_len) {
            if mask_val == 1 {
                for h in 0..self.hidden_size {
                    let token_embedding = last_hidden_state[[seq_idx, h]];
                    weighted_sum[h] += token_embedding * mask_val as f32;
                    mask_sum += mask_val as f32;
                }
            }
        }

        if mask_sum > 0.0 {
            for h in 0..self.hidden_size {
                weighted_sum[h] /= mask_sum;
            }
        }

        Ok(weighted_sum)
    }
}

impl InferenceEngine for OnnxEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        self.forward_pass(text)
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.forward_pass(text)?);
        }
        Ok(results)
    }
}
