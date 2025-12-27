// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::model::{DeviceType, ModelConfig, Precision};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::AppError;
use crate::model::recovery::{ModelRecovery, RecoveryConfig};
use crate::monitor::MemoryMonitor;
use crate::text::{CachedTokenizer, Encoding};
use crate::utils::hash::{check_model_integrity, verify_sha256};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
use futures::executor::block_on;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(target_os = "macos")]
use tokenizers::Tokenizer as HfTokenizer;

#[cfg(not(target_os = "macos"))]
type HfTokenizer = crate::text::Tokenizer;

#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    Bert,
    XlmRoberta,
}

#[derive(Deserialize)]
struct ModelConfigJson {
    pub architectures: Option<Vec<String>>,
    pub model_type: Option<String>,
}

impl ModelConfigJson {
    pub fn get_architecture(&self) -> ModelArchitecture {
        if let Some(arch) = &self.architectures {
            for a in arch {
                if a.contains("XLMRoberta") || a.contains("xlm_roberta") {
                    return ModelArchitecture::XlmRoberta;
                }
            }
        }
        if let Some(t) = &self.model_type
            && (t.contains("xlm-roberta") || t.contains("xlm_roberta"))
        {
            return ModelArchitecture::XlmRoberta;
        }
        ModelArchitecture::Bert
    }
}

enum ModelWrapper {
    Bert(BertModel),
    XlmRoberta(XLMRobertaModel),
}

pub struct CandleEngine {
    model: ModelWrapper,
    tokenizer: CachedTokenizer,
    device: Device,
    precision: Precision,
    memory_monitor: Option<Arc<MemoryMonitor>>,
    memory_limit_controller: Option<Arc<MemoryLimitController>>,
    fallback_triggered: bool,
    device_type: DeviceType,
    model_architecture: ModelArchitecture,
}

impl CandleEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, AppError> {
        Self::with_device(config, precision, config.device.clone())
    }

    pub fn with_device(
        config: &ModelConfig,
        precision: Precision,
        device_type: DeviceType,
    ) -> Result<Self, AppError> {
        let device = if device_type == DeviceType::Cuda && candle_core::utils::cuda_is_available() {
            tracing::info!("Using CUDA GPU");
            Device::new_cuda(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else if device_type == DeviceType::Metal && candle_core::utils::metal_is_available() {
            tracing::info!("Using Metal GPU");
            Device::new_metal(0).map_err(|e| AppError::InferenceError(e.to_string()))?
        } else if matches!(device_type, DeviceType::Amd | DeviceType::OpenCL) {
            tracing::warn!(
                "Candle engine does not natively support AMD GPUs. AMD GPU support requires ROCm-enabled Candle build or ONNX Runtime. Falling back to CPU."
            );
            Device::Cpu
        } else {
            tracing::info!("Using CPU");
            Device::Cpu
        };

        let _dtype = match precision {
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

        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();

        let (config_filename, tokenizer_filename, weights_filename): (
            std::path::PathBuf,
            std::path::PathBuf,
            std::path::PathBuf,
        ) = if is_local_path {
            tracing::info!("Loading model from local path: {:?}", model_path);
            let config_path = model_path.join("config.json");
            let tokenizer_path = model_path.join("tokenizer.json");
            let weights_path = model_path.join("model.safetensors");
            let alt_weights_path = model_path.join("pytorch_model.bin");

            let weights_filename = if weights_path.exists() {
                weights_path
            } else if alt_weights_path.exists() {
                alt_weights_path
            } else {
                return Err(AppError::ModelLoadError(
                    "No model weights file found (model.safetensors or pytorch_model.bin)"
                        .to_string(),
                ));
            };

            (config_path, tokenizer_path, weights_filename)
        } else {
            tracing::info!(
                "Downloading/Loading model from HuggingFace Hub: {:?}",
                model_path
            );
            let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
            let repo = api.repo(Repo::new(
                model_path.to_string_lossy().into_owned(),
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

            (config_filename, tokenizer_filename, weights_filename)
        };

        let config_content = std::fs::read_to_string(&config_filename)?;
        let model_config_json: ModelConfigJson = serde_json::from_str(&config_content)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let model_architecture = model_config_json.get_architecture();

        tracing::info!("Detected model architecture: {:?}", model_architecture);

        let (bert_config, xlm_config) = match &model_architecture {
            ModelArchitecture::Bert => {
                let bert_config: BertConfig = serde_json::from_str(&config_content)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                (Some(bert_config), None)
            }
            ModelArchitecture::XlmRoberta => {
                let xlm_config: XlmRobertaConfig = serde_json::from_str(&config_content)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                (None, Some(xlm_config))
            }
        };

        let hf_tokenizer = HfTokenizer::from_file(tokenizer_filename.to_string_lossy().as_ref())
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let max_position_embeddings = match &model_architecture {
            ModelArchitecture::Bert => bert_config.as_ref().unwrap().max_position_embeddings,
            ModelArchitecture::XlmRoberta => xlm_config.as_ref().unwrap().max_position_embeddings,
        };

        let tokenizer = CachedTokenizer::new(hf_tokenizer, max_position_embeddings, 2048);

        let is_pytorch = weights_filename.to_string_lossy().ends_with(".bin");

        let config_str = config_filename.to_string_lossy().to_string();
        let tokenizer_str = tokenizer_filename.to_string_lossy().to_string();
        let weights_str = weights_filename.to_string_lossy().to_string();

        let files_to_check = vec![
            (config_str.clone(), None),
            (tokenizer_str.clone(), None),
            (weights_str.clone(), config.model_sha256.clone()),
        ];

        let min_sizes = {
            let mut sizes = HashMap::new();
            sizes.insert(config_str.clone(), 100);
            sizes.insert(tokenizer_str.clone(), 1000);
            sizes.insert(weights_str.clone(), 1024 * 1024);
            sizes
        };

        tracing::info!("Checking model file integrity...");
        let integrity_report = check_model_integrity(&config.name, files_to_check, Some(min_sizes))
            .map_err(|e| AppError::ModelIntegrityError(format!("Integrity check failed: {}", e)))?;

        if !integrity_report.overall_valid {
            tracing::error!("Model file integrity check failed!");
            for check in &integrity_report.files_checked {
                if !check.is_valid {
                    tracing::error!(
                        "  File: {}, Error: {}",
                        check.file_path,
                        check.error_message.as_deref().unwrap_or("Unknown error")
                    );
                }
            }

            tracing::info!("Attempting automatic recovery of corrupted files...");
            let recovery_config = RecoveryConfig::default();
            let recovery = ModelRecovery::new(recovery_config);

            let repo_id_str = if !is_local_path {
                Some(config.model_path.to_string_lossy().to_string())
            } else {
                None
            };

            let repo_id = repo_id_str.as_deref();

            let recovery_result = recovery
                .recover_corrupted_files(
                    &config.name,
                    model_path,
                    repo_id,
                    &integrity_report.corrupted_files,
                )
                .map_err(|e| AppError::ModelIntegrityError(format!("Recovery failed: {}", e)))?;

            if recovery_result.success {
                tracing::info!("Successfully recovered all corrupted files");
                tracing::info!("Re-running integrity check after recovery...");

                let files_to_check = vec![
                    (config_str.clone(), None),
                    (tokenizer_str.clone(), None),
                    (weights_str.clone(), config.model_sha256.clone()),
                ];

                let min_sizes = {
                    let mut sizes = HashMap::new();
                    sizes.insert(config_str, 100);
                    sizes.insert(tokenizer_str, 1000);
                    sizes.insert(weights_str, 1024 * 1024);
                    sizes
                };

                let recovery_integrity_report =
                    check_model_integrity(&config.name, files_to_check, Some(min_sizes)).map_err(
                        |e| {
                            AppError::ModelIntegrityError(format!(
                                "Post-recovery integrity check failed: {}",
                                e
                            ))
                        },
                    )?;

                if !recovery_integrity_report.overall_valid {
                    return Err(AppError::ModelFileCorrupted(format!(
                        "Model files still corrupted after recovery. Corrupted files: {:?}",
                        recovery_integrity_report.corrupted_files
                    )));
                }

                tracing::info!("Post-recovery integrity check passed");
            } else {
                return Err(AppError::ModelFileCorrupted(format!(
                    "Failed to recover corrupted files after {} attempts. Corrupted files: {:?}",
                    recovery_result.attempts, recovery_result.failed_files
                )));
            }
        }

        tracing::info!("Model file integrity check passed");

        if let Some(ref expected_hash) = config.model_sha256 {
            tracing::info!("Verifying model file SHA256 hash...");
            let is_valid = verify_sha256(&weights_filename, expected_hash)
                .map_err(|e| AppError::ModelLoadError(format!("Failed to verify SHA256: {}", e)))?;

            if !is_valid {
                return Err(AppError::ModelFileCorrupted(format!(
                    "Model file SHA256 verification failed. Expected: {}, File: {:?}",
                    expected_hash, weights_filename
                )));
            }

            tracing::info!("Model file SHA256 verification passed");
        }

        let dtype = DType::F32;
        let vb: VarBuilder = if is_pytorch {
            tracing::info!("Loading PyTorch model weights from: {:?}", weights_filename);

            let file_size = std::fs::metadata(&weights_filename)
                .map_err(|e| AppError::ModelLoadError(e.to_string()))?
                .len();

            if file_size > 2 * 1024 * 1024 * 1024 {
                tracing::warn!(
                    "PyTorch model file is large ({} GB). Large PyTorch files may have loading issues.",
                    file_size as f64 / 1024.0 / 1024.0 / 1024.0
                );
                tracing::info!("Consider converting to safetensors format for better performance:");
                tracing::info!(
                    "  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{}').save_pretrained('./model_converted')\"",
                    config.model_path.to_string_lossy()
                );
                tracing::info!("  Then use './model_converted' as the model_path in config.toml");
            }

            let mut varmap = VarMap::new();
            match varmap.load(&weights_filename) {
                Ok(_) => {
                    tracing::info!("PyTorch weights loaded successfully");
                }
                Err(e) => {
                    tracing::error!("Failed to load PyTorch weights: {}", e);
                    tracing::error!(
                        "This is often due to large model files or incompatible formats."
                    );
                    tracing::error!("Please convert the model to safetensors format:");
                    tracing::error!("  pip install optimum");
                    tracing::error!(
                        "  optimum-cli export onnx --model {} --task feature-extraction ./model_safetensors",
                        config.model_path.to_string_lossy()
                    );
                    return Err(AppError::ModelLoadError(format!(
                        "Failed to load PyTorch weights: {}. Please convert the model to safetensors format.",
                        e
                    )));
                }
            }
            VarBuilder::from_varmap(&varmap, dtype, &device)
        } else {
            tracing::info!(
                "Loading safetensors model weights from: {:?}",
                weights_filename
            );
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device) };
            vb.map_err(|e| AppError::ModelLoadError(e.to_string()))?
        };

        if is_pytorch {
            tracing::info!("Loaded PyTorch model weights successfully");
        } else {
            tracing::info!("Loaded safetensors model weights successfully");
        }

        let model = match &model_architecture {
            ModelArchitecture::Bert => {
                let config = bert_config.ok_or_else(|| {
                    AppError::ModelLoadError("Bert config is required for Bert model".to_string())
                })?;
                let bert_model = BertModel::load(vb, &config)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                ModelWrapper::Bert(bert_model)
            }
            ModelArchitecture::XlmRoberta => {
                let config = xlm_config.ok_or_else(|| {
                    AppError::ModelLoadError("XLM-RoBERTa config is required".to_string())
                })?;
                let xlm_model = XLMRobertaModel::new(&config, vb)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                ModelWrapper::XlmRoberta(xlm_model)
            }
        };

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
            model_architecture,
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

    pub async fn check_memory_pressure(&self, threshold_percent: u64) -> bool {
        if let Some(ref monitor) = self.memory_monitor {
            let stats = monitor.get_memory_stats().await;
            let usage_percent = if stats.total_bytes > 0 {
                (stats.current_bytes * 100) / stats.total_bytes
            } else {
                0
            };
            usage_percent >= threshold_percent
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
                tracing::warn!("Memory limit exceeded, attempting fallback to CPU");
                self.try_fallback_to_cpu(config).await?;
                return Ok(true);
            } else if status == MemoryLimitStatus::Critical {
                tracing::warn!("Memory limit critical, checking memory pressure for fallback");
                if self.check_memory_pressure(90).await {
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn create_type_ids_tensor(
        &self,
        encodings: &[Encoding],
        max_seq_len: usize,
    ) -> Result<Tensor, AppError> {
        let batch_size = encodings.len();
        let mut batch_data = vec![0i64; batch_size * max_seq_len];

        for (batch_idx, encoding) in encodings.iter().enumerate() {
            for (seq_idx, &value) in encoding.type_ids.iter().enumerate().take(max_seq_len) {
                batch_data[batch_idx * max_seq_len + seq_idx] = value as i64;
            }
        }

        Tensor::new(batch_data.as_slice(), &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .reshape(&[batch_size, max_seq_len])
            .map_err(|e| AppError::InferenceError(e.to_string()))
    }

    async fn forward_pass(&self, text: &str) -> Result<Vec<f32>, AppError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .await
            .map_err(|e| AppError::TokenizationError(e.to_string()))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        tracing::debug!("Token IDs: {:?}", ids);
        tracing::debug!("Max token ID: {}", ids.iter().max().copied().unwrap_or(0));
        tracing::debug!("Attention mask: {:?}", attention_mask);

        let vocab_size = 250002;
        let max_id = ids.iter().max().copied().unwrap_or(0);
        if max_id >= vocab_size {
            tracing::warn!(
                "Token ID {} exceeds vocab_size {}, clamping",
                max_id,
                vocab_size
            );
        }

        let max_len = self.tokenizer.max_length().min(512);

        let ids_slice: Vec<u32> = ids
            .iter()
            .take(max_len)
            .map(|&id| if id >= vocab_size { vocab_size - 1 } else { id })
            .collect();

        let mask_slice: Vec<u32> = attention_mask
            .iter()
            .take(max_len)
            .map(|&id| if id >= vocab_size { vocab_size - 1 } else { id })
            .collect();

        let token_ids = Tensor::new(ids_slice, &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let attention_mask_tensor = Tensor::new(mask_slice, &self.device)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        let embeddings = match &self.model {
            ModelWrapper::Bert(bert_model) => bert_model
                .forward(&token_ids, &attention_mask_tensor, None)
                .map_err(|e| AppError::InferenceError(e.to_string()))?,
            ModelWrapper::XlmRoberta(xlm_model) => {
                let type_ids_slice: Vec<u32> =
                    encoding.type_ids.iter().take(max_len).cloned().collect();
                let token_type_ids = Tensor::new(type_ids_slice, &self.device)
                    .map_err(|e| AppError::InferenceError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| AppError::InferenceError(e.to_string()))?;
                xlm_model
                    .forward(
                        &token_ids,
                        &attention_mask_tensor,
                        &token_type_ids,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| AppError::InferenceError(e.to_string()))?
            }
        };

        tracing::debug!("Embeddings shape: {:?}", embeddings.shape());
        tracing::debug!("Embeddings dims: {}", embeddings.dims().len());
        tracing::debug!("Embeddings dims array: {:?}", embeddings.dims());

        self.update_gpu_memory().await;

        let embedding_result: Tensor;
        let dims = embeddings.dims();
        tracing::debug!("Processing embedding with {} dimensions", dims.len());

        if dims.len() == 1 {
            tracing::debug!("1D embedding, using directly");
            embedding_result = embeddings.clone();
        } else if dims.len() == 2 {
            if dims[0] == 1 && dims[1] > 1 {
                tracing::debug!("2D embedding [1, hidden_size], extracting batch 0");
                embedding_result = embeddings
                    .get(0)
                    .map_err(|e| AppError::InferenceError(format!("Failed to get batch 0: {}", e)))?
                    .clone();
            } else {
                tracing::debug!(
                    "2D embedding [seq_len, hidden_size], extracting CLS token (index 0)"
                );
                embedding_result = embeddings
                    .get(0)
                    .map_err(|e| AppError::InferenceError(format!("Failed to get token 0: {}", e)))?
                    .clone();
            }
        } else if dims.len() == 3 {
            tracing::debug!("3D embedding [batch, seq_len, hidden], extracting batch 0, token 0");
            embedding_result = embeddings
                .get(0)
                .map_err(|e| AppError::InferenceError(format!("Failed to get batch 0: {}", e)))?
                .get(0)
                .map_err(|e| AppError::InferenceError(format!("Failed to get token 0: {}", e)))?
                .clone();
        } else {
            return Err(AppError::InferenceError(format!(
                "Unsupported embedding dimensions: {} (shape: {:?})",
                dims.len(),
                embeddings.shape()
            )));
        }

        tracing::debug!("Final embedding shape: {:?}", embedding_result.shape());

        let vec = embedding_result
            .to_vec1::<f32>()
            .map_err(|e| AppError::InferenceError(e.to_string()))?;

        Ok(vec)
    }

    async fn forward_pass_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            "Processing batch of {} texts using sequential processing fallback",
            texts.len()
        );

        let mut results = Vec::with_capacity(texts.len());
        for (i, &text) in texts.iter().enumerate() {
            tracing::debug!("Batch[{}] processing: '{}' (len={})", i, text, text.len());
            let embedding = self.forward_pass(text).await?;
            tracing::debug!("Batch[{}] embedding vector len: {}", i, embedding.len());
            results.push(embedding);
        }

        tracing::debug!(
            "Batch processing completed, {} embeddings generated",
            results.len()
        );
        Ok(results)
    }
}

#[async_trait]
impl InferenceEngine for CandleEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, AppError> {
        block_on(async { self.forward_pass(text).await })
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let texts_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        block_on(async { self.forward_pass_batch(&texts_refs).await })
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        self.device.is_cuda()
    }

    fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), AppError> {
        self.try_fallback_to_cpu_impl(config).await
    }
}

impl CandleEngine {
    async fn try_fallback_to_cpu_impl(&mut self, config: &ModelConfig) -> Result<(), AppError> {
        if self.fallback_triggered {
            return Ok(());
        }

        tracing::info!("Attempting fallback from GPU to CPU due to memory pressure");

        self.memory_monitor = None;
        self.device = Device::Cpu;
        self.device_type = DeviceType::Cpu;
        self.fallback_triggered = true;

        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();

        let (config_filename, tokenizer_filename, weights_filename): (
            std::path::PathBuf,
            std::path::PathBuf,
            std::path::PathBuf,
        ) = if is_local_path {
            tracing::info!(
                "Loading model from local path for fallback: {:?}",
                model_path
            );
            let config_path = model_path.join("config.json");
            let tokenizer_path = model_path.join("tokenizer.json");
            let weights_path = model_path.join("model.safetensors");
            let alt_weights_path = model_path.join("pytorch_model.bin");

            let weights_filename = if weights_path.exists() {
                weights_path
            } else if alt_weights_path.exists() {
                alt_weights_path
            } else {
                return Err(AppError::ModelLoadError(
                    "No model weights file found during fallback".to_string(),
                ));
            };

            (config_path, tokenizer_path, weights_filename)
        } else {
            tracing::info!(
                "Loading model from HuggingFace Hub for fallback: {:?}",
                model_path
            );
            let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
            let repo = api.repo(Repo::new(
                model_path.to_string_lossy().into_owned(),
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

            (config_filename, tokenizer_filename, weights_filename)
        };

        let config_content = std::fs::read_to_string(config_filename)?;
        let model_config_json: ModelConfigJson = serde_json::from_str(&config_content)
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let fallback_architecture = model_config_json.get_architecture();

        tracing::info!(
            "Fallback: Detected model architecture: {:?}",
            fallback_architecture
        );

        let (bert_config, xlm_config) = match &fallback_architecture {
            ModelArchitecture::Bert => {
                let bert_config: BertConfig = serde_json::from_str(&config_content)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                (Some(bert_config), None)
            }
            ModelArchitecture::XlmRoberta => {
                let xlm_config: XlmRobertaConfig = serde_json::from_str(&config_content)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                (None, Some(xlm_config))
            }
        };

        let hf_tokenizer = HfTokenizer::from_file(tokenizer_filename.to_string_lossy().as_ref())
            .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        let max_position_embeddings = match &fallback_architecture {
            ModelArchitecture::Bert => bert_config.as_ref().unwrap().max_position_embeddings,
            ModelArchitecture::XlmRoberta => xlm_config.as_ref().unwrap().max_position_embeddings,
        };

        self.tokenizer = CachedTokenizer::new(hf_tokenizer, max_position_embeddings, 2048);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &self.device)
        }
        .map_err(|e| AppError::ModelLoadError(e.to_string()))?;

        self.model = match &fallback_architecture {
            ModelArchitecture::Bert => {
                let config = bert_config.ok_or_else(|| {
                    AppError::ModelLoadError("Bert config is required".to_string())
                })?;
                let bert_model = BertModel::load(vb, &config)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                ModelWrapper::Bert(bert_model)
            }
            ModelArchitecture::XlmRoberta => {
                let config = xlm_config.ok_or_else(|| {
                    AppError::ModelLoadError("XLM-RoBERTa config is required".to_string())
                })?;
                let xlm_model = XLMRobertaModel::new(&config, vb)
                    .map_err(|e| AppError::ModelLoadError(e.to_string()))?;
                ModelWrapper::XlmRoberta(xlm_model)
            }
        };

        self.model_architecture = fallback_architecture;

        tracing::info!("Successfully fell back to CPU");
        Ok(())
    }
}
