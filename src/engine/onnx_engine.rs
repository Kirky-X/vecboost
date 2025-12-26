// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::{InferenceEngine, Precision};
use crate::config::model::{DeviceType, ModelConfig};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::AppError;
use crate::monitor::MemoryMonitor;
use crate::utils::hash::verify_sha256;
use hf_hub::{Repo, RepoType, api::sync::Api};
use ndarray::{Array1, Array2};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, Tokenizer};

pub struct OnnxEngine {
    session: Arc<Mutex<Session>>,
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

    pub fn with_device(
        config: &ModelConfig,
        precision: Precision,
        device_type: DeviceType,
    ) -> Result<Self, AppError> {
        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();

        let (onnx_filename, tokenizer_filename) = if is_local_path {
            tracing::info!("Using local model path: {:?}", model_path);
            let onnx_filename = if model_path.join("model_quantized.onnx").exists() {
                model_path.join("model_quantized.onnx")
            } else if model_path.join("model.onnx").exists() {
                model_path.join("model.onnx")
            } else {
                return Err(AppError::ModelLoadError(format!(
                    "No ONNX model found in {:?}",
                    model_path
                )));
            };
            let tokenizer_filename = if model_path.join("tokenizer.json").exists() {
                model_path.join("tokenizer.json")
            } else {
                let parent = model_path.parent().and_then(|p| p.parent());
                let cache_path = parent.ok_or_else(|| {
                    AppError::ModelLoadError(format!(
                        "Cannot determine HuggingFace cache path from {:?}",
                        model_path
                    ))
                })?;
                let cache_tokenizer = cache_path.join("tokenizer.json");
                if cache_tokenizer.exists() {
                    tracing::info!(
                        "Using tokenizer from HuggingFace cache: {:?}",
                        cache_tokenizer
                    );
                    cache_tokenizer
                } else {
                    return Err(AppError::ModelLoadError(format!(
                        "Tokenizer not found at {:?} or {:?}",
                        model_path.join("tokenizer.json"),
                        cache_tokenizer
                    )));
                }
            };
            (onnx_filename, tokenizer_filename)
        } else {
            tracing::info!("Using HuggingFace Hub for model: {:?}", model_path);
            let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
            let repo = api.repo(Repo::new(
                model_path.to_string_lossy().into_owned(),
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
            (onnx_filename, tokenizer_filename)
        };

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
            #[cfg(feature = "cuda")]
            {
                let cuda_provider =
                    ort::execution_providers::CUDAExecutionProvider::default().build();
                session
                    .with_execution_providers([cuda_provider])
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!(
                    "CUDA execution provider not available. ONNX Runtime CUDA support requires cuda feature flag. Using CPU execution provider."
                );
                session
            }
        } else if supports_amd {
            tracing::info!("Attempting to configure ROCm execution provider for ONNX Runtime");
            #[cfg(feature = "rocm")]
            {
                let rocm_provider =
                    ort::execution_providers::ROCMExecutionProvider::default().build();
                session
                    .with_execution_providers([rocm_provider])
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            }
            #[cfg(not(feature = "rocm"))]
            {
                tracing::warn!(
                    "ROCM execution provider not available. ONNX Runtime AMD support requires rocm feature flag. Using CPU execution provider."
                );
                session
            }
        } else {
            session
        };

        if let Some(ref expected_hash) = config.model_sha256 {
            tracing::info!("Verifying model file SHA256 hash...");
            let is_valid = verify_sha256(&onnx_filename, expected_hash)
                .map_err(|e| AppError::ModelLoadError(format!("Failed to verify SHA256: {}", e)))?;

            if !is_valid {
                return Err(AppError::ModelLoadError(format!(
                    "Model file SHA256 verification failed. Expected: {}, File: {:?}",
                    expected_hash, onnx_filename
                )));
            }

            tracing::info!("Model file SHA256 verification passed");
        }

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
        let hidden_size = config.expected_dimension.unwrap_or(1024);
        tracing::info!(
            "Using hidden_size from configuration: {:?}",
            config.expected_dimension
        );
        tracing::info!("Final hidden_size value: {}", hidden_size);
        let max_input_length = std::cmp::min(vocab_size, 512);

        let actual_precision = match precision {
            Precision::Fp16 => {
                if supports_cuda || supports_amd {
                    tracing::info!("Using FP16 precision with GPU acceleration");
                    Precision::Fp16
                } else {
                    tracing::warn!(
                        "FP16 not supported without GPU acceleration, falling back to FP32"
                    );
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
            session: Arc::new(Mutex::new(session)),
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
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.block_on(async {
                    let stats = monitor.get_memory_stats().await;
                    let usage_percent = if stats.total_bytes > 0 {
                        (stats.current_bytes * 100) / stats.total_bytes
                    } else {
                        0
                    };
                    usage_percent >= threshold_percent
                })
            } else {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap_or_else(|_| {
                        tracing::error!("Failed to create Tokio runtime for memory check");
                        std::process::exit(1);
                    });
                rt.block_on(async {
                    let stats = monitor.get_memory_stats().await;
                    let usage_percent = if stats.total_bytes > 0 {
                        (stats.current_bytes * 100) / stats.total_bytes
                    } else {
                        0
                    };
                    usage_percent >= threshold_percent
                })
            }
        } else {
            false
        }
    }

    pub async fn check_memory_limit_and_fallback(
        &mut self,
        config: &ModelConfig,
    ) -> Result<bool, AppError> {
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
                tracing::warn!(
                    "Memory limit critical for ONNX engine, checking memory pressure for fallback"
                );
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

    fn forward_pass(&self, text: &str) -> Result<Vec<f32>, AppError> {
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

        let input_ids_array = Array1::from(input_ids);
        let attention_mask_array = Array1::from(attention_mask.clone());

        let input_ids_tensor = Tensor::from_array(input_ids_array.into_dyn())
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.into_dyn())
            .map_err(|e: ort::Error| AppError::InferenceError(e.to_string()))?;

        let last_hidden_state = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
            let output_array = outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| AppError::InferenceError(e.to_string()))?
                .to_owned();
            tracing::debug!("ONNX model output shape: {:?}", output_array.shape());
            output_array
        };

        let seq_len = attention_mask.iter().filter(|&&v| v == 1).count();
        if seq_len == 0 {
            return Err(AppError::InferenceError(
                "Empty sequence after mask".to_string(),
            ));
        }

        let mut weighted_sum = vec![0.0f32; self.hidden_size];
        let mut mask_sum = 0.0f32;

        for (seq_idx, &mask_val) in attention_mask.iter().enumerate().take(seq_len) {
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

        let new_session = Session::builder()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?
            .commit_from_file(onnx_filename)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        *session_guard = new_session;
        drop(session_guard);

        self.precision = Precision::Fp32;

        tracing::info!("Successfully fell back to CPU for ONNX engine");
        Ok(())
    }

    fn forward_pass_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(batch_size);
        let mut all_attention_masks: Vec<Vec<i64>> = Vec::with_capacity(batch_size);
        let mut max_seq_len = 0;

        for text in texts {
            let text_ref: &str = text.as_str();
            let tokens = self
                .tokenizer
                .encode(text_ref, true)
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
            Array2::from_shape_vec((padded_batch_size, max_seq_len), batch_input_ids)
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
        let attention_mask_array =
            Array2::from_shape_vec((padded_batch_size, max_seq_len), batch_attention_mask)
                .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let input_ids_tensor = Tensor::from_array(input_ids_array.into_dyn())
            .map_err(|e| AppError::InferenceError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.into_dyn())
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let last_hidden_state = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| AppError::InferenceError(e.to_string()))?;
            outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| AppError::InferenceError(e.to_string()))?
                .to_owned()
        };

        let mut results = Vec::with_capacity(padded_batch_size);
        for batch_idx in 0..padded_batch_size {
            let attention_mask = &all_attention_masks[batch_idx];
            let actual_seq_len = attention_mask.len();
            tracing::debug!(
                "batch_idx: {}, max_seq_len: {}, actual_seq_len: {}",
                batch_idx,
                max_seq_len,
                actual_seq_len
            );

            let mut weighted_sum = vec![0.0f32; self.hidden_size];
            let mut mask_sum = 0.0f32;

            let effective_max_seq = std::cmp::min(max_seq_len, actual_seq_len);
            for seq_idx in 0..effective_max_seq {
                let mask_val = attention_mask[seq_idx];
                if mask_val == 1 {
                    for h in 0..self.hidden_size {
                        let token_embedding = last_hidden_state[[batch_idx, seq_idx, h]];
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
}

impl InferenceEngine for OnnxEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        self.forward_pass(text)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        self.forward_pass_batch(texts)
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        self.memory_monitor.is_some()
    }
}
