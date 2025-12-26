// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::model::{DeviceType, ModelConfig, Precision};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::AppError;
use crate::monitor::MemoryMonitor;
use crate::text::{CachedTokenizer, Encoding};
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
    memory_limit_controller: Option<Arc<MemoryLimitController>>,
    fallback_triggered: bool,
    device_type: DeviceType,
}

impl CandleEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, AppError> {
        Self::with_device(config, precision, config.device.clone())
    }

    pub fn with_device(config: &ModelConfig, precision: Precision, device_type: DeviceType) -> Result<Self, AppError> {
        let device = if device_type == DeviceType::Cuda && candle_core::utils::cuda_is_available()
        {
            tracing::info!("Using CUDA GPU");
            Device::new_cuda(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else if device_type == DeviceType::Metal && candle_core::utils::metal_is_available()
        {
            tracing::info!("Using Metal GPU");
            Device::new_metal(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else if matches!(device_type, DeviceType::Amd | DeviceType::OpenCL)
        {
            tracing::warn!("Candle engine does not natively support AMD GPUs. AMD GPU support requires ROCm-enabled Candle build or ONNX Runtime. Falling back to CPU.");
            Device::Cpu
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
                    tracing::warn!("FP16 not supported on non-CUDA devices, falling back to FP32");
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

        let memory_monitor = if device.is_cuda() || device.is_metal() {
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
            memory_limit_controller: None,
            fallback_triggered: false,
            device_type,
        })
    }

    pub fn set_memory_limit_controller(&mut self, controller: Arc<MemoryLimitController>) {
        self.memory_limit_controller = Some(controller);
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    pub fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    pub fn check_memory_pressure(&self, threshold_percent: u64) -> bool {
        if let Some(ref monitor) = self.memory_monitor {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let stats = monitor.get_memory_stats().await;
                let usage_percent = if stats.total_bytes > 0 {
                    (stats.current_bytes * 100) / stats.total_bytes
                } else {
                    0
                };
                usage_percent >= threshold_percent
            })
        } else {
            false
        }
    }

    pub async fn check_memory_limit_and_fallback(&mut self, config: &ModelConfig) -> Result<bool, AppError> {
        if self.fallback_triggered {
            return Ok(false);
        }

        if let Some(ref controller) = self.memory_limit_controller {
            let status = controller.check_limit().await;

            if status == MemoryLimitStatus::Exceeded {
                tracing::warn!("Memory limit exceeded, attempting fallback to CPU");
                self.try_fallback_to_cpu(config).await?;
                return Ok(true);
            } else if status == MemoryLimitStatus::Critical {
                tracing::warn!("Memory limit critical, checking memory pressure for fallback");
                if self.check_memory_pressure(90) {
                    self.try_fallback_to_cpu(config).await?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    pub async fn update_memory_limit(&self, used_bytes: u64) {
        if let Some(ref controller) = self.memory_limit_controller {
            controller.update_usage(used_bytes).await;
        }
    }

    pub async fn get_memory_status(&self) -> Option<MemoryLimitStatus> {
        if let Some(ref controller) = self.memory_limit_controller {
            Some(controller.check_limit().await)
        } else {
            None
        }
    }

    pub async fn update_gpu_memory(&self) {
        if let Some(ref _monitor) = self.memory_monitor {
            #[cfg(feature = "cuda")]
            _monitor.update_gpu_memory_from_candle().await;
            #[cfg(feature = "metal")]
            _monitor.update_gpu_memory_from_metal().await;
        }
    }

    fn create_padded_batch_tensor(
        &self,
        encodings: &[Encoding],
        max_seq_len: usize,
        is_input_ids: bool,
    ) -> Result<Tensor, AppError> {
        let batch_size = encodings.len();
        let mut batch_data = vec![0i64; batch_size * max_seq_len];

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let data = if is_input_ids {
                encoding.get_ids()
            } else {
                encoding.get_attention_mask()
            };

            for (seq_idx, &value) in data.iter().enumerate().take(max_seq_len) {
                batch_data[batch_idx * max_seq_len + seq_idx] = value as i64;
            }
        }

        Tensor::new(batch_data.as_slice(), &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .reshape(&[batch_size, max_seq_len])
            .map_err(|e| AppError::InferenceError(e.to_string()))
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

        self.update_gpu_memory().await;

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

    async fn forward_pass_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = texts.len();
        tracing::debug!("Processing batch of {} texts with true batch inference", batch_size);

        let mut encodings: Vec<Encoding> = Vec::with_capacity(batch_size);
        let mut max_seq_len = 0;

        for &text in texts {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .await
                .map_err(|e| AppError::TokenizationError(e.to_string()))?;

            let seq_len = encoding.get_ids().len();
            if seq_len > max_seq_len {
                max_seq_len = seq_len;
            }
            encodings.push(encoding);
        }

        let batch_input_ids = self.create_padded_batch_tensor(&encodings, max_seq_len, true)?;
        let batch_attention_mask = self.create_padded_batch_tensor(&encodings, max_seq_len, false)?;

        let embeddings = self
            .model
            .forward(&batch_input_ids, &batch_attention_mask, None)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        self.update_gpu_memory().await;

        let mut results = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let cls_embedding = embeddings
                .get(batch_idx)
                .map_err(|e| AppError::InferenceError(e.to_string()))?
                .get(0)
                .map_err(|e| AppError::InferenceError(e.to_string()))?;

            let vec = cls_embedding
                .to_vec1::<f32>()
                .map_err(|e| AppError::InferenceError(e.to_string()))?;

            results.push(vec);
        }

        Ok(results)
    }

    pub async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), AppError> {
        if self.fallback_triggered {
            return Ok(());
        }

        tracing::info!("Attempting fallback from GPU to CPU due to memory pressure");

        self.memory_monitor = None;
        self.device = Device::Cpu;
        self.device_type = DeviceType::Cpu;
        self.fallback_triggered = true;

        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(
            config.model_path.to_string_lossy().into_owned(),
            RepoType::Model,
        ));

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

        self.tokenizer = CachedTokenizer::new(hf_tokenizer, bert_config.max_position_embeddings, 2048);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &self.device)
        }
        .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        self.model = BertModel::load(vb, &bert_config)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        tracing::info!("Successfully fell back to CPU");
        Ok(())
    }
}

impl InferenceEngine for CandleEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        let config = ModelConfig::default();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        rt.block_on(async {
            if let Err(e) = self.check_memory_limit_and_fallback(&config).await {
                tracing::error!("Memory limit check failed: {}", e);
            }
            self.forward_pass(text).await
        })
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let config = ModelConfig::default();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        rt.block_on(async {
            if let Err(e) = self.check_memory_limit_and_fallback(&config).await {
                tracing::error!("Memory limit check failed: {}", e);
            }
            self.forward_pass_batch(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice()).await
        })
    }

    fn precision(&self) -> Precision {
        self.precision.clone()
    }

    fn supports_mixed_precision(&self) -> bool {
        self.device.is_cuda() || self.device.is_metal()
    }
}
