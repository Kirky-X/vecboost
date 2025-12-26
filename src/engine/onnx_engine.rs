// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::{InferenceEngine, Precision};
use crate::config::model::{DeviceType, ModelConfig};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::AppError;
use crate::monitor::MemoryMonitor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use ndarray::Array1;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::sync::Arc;
use tokenizers::{PaddingParams, Tokenizer};

pub struct OnnxEngine {
    session: Session,
    tokenizer: Tokenizer,
    hidden_size: usize,
    max_input_length: usize,
    precision: Precision,
    memory_monitor: Option<Arc<MemoryMonitor>>,
    memory_limit_controller: Option<Arc<MemoryLimitController>>,
    fallback_triggered: bool,
    device_type: DeviceType,
    supports_cuda: bool,
}

impl OnnxEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, AppError> {
        Self::with_device(config, precision, config.device.clone())
    }

    pub fn with_device(config: &ModelConfig, precision: Precision, device_type: DeviceType) -> Result<Self, AppError> {
        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(
            config.model_path.to_string_lossy().into_owned(),
            RepoType::Model,
        ));

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

        let supports_cuda = device_type == DeviceType::Cuda;
        let supports_amd = matches!(device_type, DeviceType::Amd | DeviceType::OpenCL);

        tracing::info!("Initializing ONNX Runtime session...");
        let session = Session::builder()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let session = if supports_cuda {
            tracing::info!("Attempting to configure CUDA execution provider for ONNX Runtime");
            let cuda_provider = ort::execution_providers::CUDA::default();
            session.with_execution_provider(cuda_provider)
        } else if supports_amd {
            tracing::info!("Attempting to configure ROCm execution provider for ONNX Runtime");
            #[cfg(feature = "rocm")]
            {
                let rocm_provider = ort::execution_providers::ROCM::default();
                session.with_execution_provider(rocm_provider)
            }
            #[cfg(not(feature = "rocm"))]
            {
                tracing::warn!("ROCM execution provider not available. ONNX Runtime AMD support requires rocm feature flag. Using CPU execution provider.");
                session
            }
        } else {
            session
        };

        let session = session
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

        let actual_precision = match precision {
            Precision::Fp16 => {
                if supports_cuda || supports_amd {
                    tracing::info!("Using FP16 precision with GPU acceleration");
                    Precision::Fp16
                } else {
                    tracing::warn!("FP16 not supported without GPU acceleration, falling back to FP32");
                    Precision::Fp32
                }
            }
            _ => {
                tracing::info!("Using {} precision", precision);
                precision
            }
        };

        tracing::info!(
            "ONNX Engine initialized: hidden_size={}, max_input_length={}, vocab_size={}, precision={:?}",
            hidden_size,
            max_input_length,
            vocab_size,
            actual_precision
        );

        let memory_monitor = if supports_cuda || supports_amd {
            Some(Arc::new(MemoryMonitor::new()))
        } else {
            None
        };

        Ok(Self {
            session,
            tokenizer,
            hidden_size,
            max_input_length,
            precision: actual_precision,
            memory_monitor,
            memory_limit_controller: None,
            fallback_triggered: false,
            device_type,
            supports_cuda,
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
                tracing::warn!("Memory limit exceeded for ONNX engine, attempting fallback to CPU");
                self.try_fallback_to_cpu(config).await?;
                return Ok(true);
            } else if status == MemoryLimitStatus::Critical {
                tracing::warn!("Memory limit critical for ONNX engine, checking memory pressure for fallback");
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
        if let Some(ref monitor) = self.memory_monitor {
            #[cfg(feature = "onnx")]
            monitor.update_gpu_memory_from_ort().await;
        }
    }

    fn extract_embedding_from_output(
        outputs: &ort::outputs::OutputTensor,
        attention_mask: &[i64],
        hidden_size: usize,
    ) -> Result<Vec<f32>, AppError> {
        let outputs_map = outputs.as_hash_map();
        let last_hidden_state_array = outputs_map
            .get("last_hidden_state")
            .ok_or_else(|| AppError::InferenceError("Missing last_hidden_state in outputs".to_string()))?;

        let last_hidden_state = last_hidden_state_array
            .try_extract_array::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let seq_len = attention_mask.iter().filter(|&&v| v == 1).count();
        if seq_len == 0 {
            return Err(AppError::InferenceError(
                "Empty sequence after mask".to_string(),
            ));
        }

        let mut weighted_sum = vec![0.0f32; hidden_size];
        let mut mask_sum = 0.0f32;

        for (seq_idx, &mask_val) in attention_mask.iter().enumerate().take(seq_len) {
            if mask_val == 1 {
                for h in 0..hidden_size {
                    let token_embedding = last_hidden_state[[seq_idx, h]];
                    weighted_sum[h] += token_embedding * mask_val as f32;
                    mask_sum += mask_val as f32;
                }
            }
        }

        if mask_sum > 0.0 {
            for h in 0..hidden_size {
                weighted_sum[h] /= mask_sum;
            }
        }

        Ok(weighted_sum)
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

        Self::extract_embedding_from_output(&outputs, &attention_mask_clone, self.hidden_size)
    }

    pub async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), AppError> {
        if self.fallback_triggered {
            return Ok(());
        }

        tracing::info!("Attempting fallback from GPU to CPU for ONNX engine");

        self.memory_monitor = None;
        self.device_type = DeviceType::Cpu;
        self.supports_cuda = false;
        self.fallback_triggered = true;

        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(
            config.model_path.to_string_lossy().into_owned(),
            RepoType::Model,
        ));

        let onnx_filename = repo
            .get("model.onnx")
            .or_else(|_| repo.get("model_quantized.onnx"))
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        self.session = Session::builder()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .commit_from_file(onnx_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        self.precision = Precision::Fp32;

        tracing::info!("Successfully fell back to CPU for ONNX engine");
        Ok(())
    }
}

impl InferenceEngine for OnnxEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        let config = ModelConfig::default();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        rt.block_on(async {
            if let Err(e) = self.check_memory_limit_and_fallback(&config).await {
                tracing::error!("Memory limit check failed: {}", e);
            }
            self.forward_pass(text)
        })
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let config = ModelConfig::default();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        rt.block_on(async {
            if let Err(e) = self.check_memory_limit_and_fallback(&config).await {
                tracing::error!("Memory limit check failed: {}", e);
            }
            self.forward_pass_batch(texts).await
        })
    }

        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(batch_size);
        let mut all_attention_masks: Vec<Vec<i64>> = Vec::with_capacity(batch_size);
        let mut max_seq_len = 0;

        for text in texts {
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

            if input_ids.len() > max_seq_len {
                max_seq_len = input_ids.len();
            }

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
        }

        let padded_batch_size = all_input_ids.len();
        let mut batch_input_ids = vec![0i64; padded_batch_size * max_seq_len];
        let mut batch_attention_mask = vec![0i64; padded_batch_size * max_seq_len];

        for (batch_idx, (input_ids, attention_mask)) in all_input_ids
            .iter()
            .zip(all_attention_masks.iter())
            .enumerate()
        {
            for (seq_idx, (&id, &mask)) in input_ids.iter().zip(attention_mask.iter()).enumerate() {
                let pos = batch_idx * max_seq_len + seq_idx;
                batch_input_ids[pos] = id;
                batch_attention_mask[pos] = mask;
            }
        }

        let input_ids_array =
            Array1::from_shape_vec((padded_batch_size, max_seq_len), batch_input_ids)
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
        let attention_mask_array =
            Array1::from_shape_vec((padded_batch_size, max_seq_len), batch_attention_mask)
                .map_err(|e| AppError::InferenceError(e.to_string()))?;

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

        let mut results = Vec::with_capacity(padded_batch_size);
        for batch_idx in 0..padded_batch_size {
            let attention_mask = &all_attention_masks[batch_idx];
            let start_idx = batch_idx * max_seq_len;

            let mut weighted_sum = vec![0.0f32; self.hidden_size];
            let mut mask_sum = 0.0f32;

            for seq_idx in 0..max_seq_len {
                let mask_val = attention_mask[seq_idx];
                if mask_val == 1 {
                    for h in 0..self.hidden_size {
                        let token_embedding = last_hidden_state[[start_idx + seq_idx, h]];
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

            results.push(weighted_sum);
        }

        Ok(results)
    }

    fn precision(&self) -> Precision {
        self.precision.clone()
    }

    fn supports_mixed_precision(&self) -> bool {
        self.memory_monitor.is_some()
    }
}
