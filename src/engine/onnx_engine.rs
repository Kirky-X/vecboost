// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use super::{InferenceEngine, Precision};
use crate::config::model::{DeviceType, ModelConfig};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::VecboostError;
use crate::monitor::MemoryMonitor;
use crate::utils::hash::verify_sha256;
use crate::utils::hf_hub::build_hf_repo;
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokenizers::{PaddingParams, PaddingStrategy};

/// ONNX 引擎序列截断上限，与 tokenizer 层 max_length=512 对齐
pub(crate) const ONNX_MAX_INPUT_LENGTH: usize = 512;

/// 原地 L2 归一化（零/近零向量静默跳过，与 candle 引擎同语义）。
/// 引擎出口契约要求归一化输出（T026——ONNX 两条路径此前均缺失）。
fn l2_normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// 本地 bundle 探测（spec R-model-bundle-fetch-002，embedding 引擎口径）。
///
/// onnx 文件名优先级：`model_quantized.onnx` → `model.onnx` → 目录内唯一
/// `*.onnx`（官方 laya bundle 为 `laya.onnx`），多个候选时显性报错并列出
/// 全部候选文件名。注意：决策管线（decision.rs `MODEL_CANDIDATES`）对同一
/// 目录按 model.onnx 优先的固定清单探测——优先级分歧是任务协议约定
/// （embedding 侧要求 quantized 优先；决策侧要求 fp32 主模型优先），两处
/// 均有测试钉，改动须同步评估另一侧。
///
/// tokenizer 解析委托 [`super::local_bundle::resolve_tokenizer_path`]
/// （与决策管线共用的单一事实源）。
fn resolve_local_bundle(
    model_dir: &Path,
    tokenizer_override: Option<&Path>,
) -> Result<(PathBuf, PathBuf), VecboostError> {
    let onnx_path = if model_dir.join("model_quantized.onnx").is_file() {
        model_dir.join("model_quantized.onnx")
    } else if model_dir.join("model.onnx").is_file() {
        model_dir.join("model.onnx")
    } else {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .map_err(|e| {
                VecboostError::ModelLoadError(format!(
                    "No ONNX model found in {model_dir:?}: cannot read directory ({e})"
                ))
            })?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "onnx"))
            .collect();
        candidates.sort();
        match candidates.as_slice() {
            [] => {
                return Err(VecboostError::ModelLoadError(format!(
                    "No ONNX model found in {model_dir:?}"
                )));
            }
            [only] => only.clone(),
            many => {
                let names: Vec<String> = many
                    .iter()
                    .map(|p| {
                        p.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect();
                return Err(VecboostError::ModelLoadError(format!(
                    "Multiple ONNX candidates in {model_dir:?}: {}. Set \
                     [model].model_path to a directory containing exactly one \
                     ONNX file",
                    names.join(", ")
                )));
            }
        }
    };

    let tokenizer_path =
        super::local_bundle::resolve_tokenizer_path(model_dir, tokenizer_override)?;
    Ok((onnx_path, tokenizer_path))
}

pub struct OnnxEngine {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    hidden_size: usize,
    max_input_length: usize,
    precision: Precision,
    memory_monitor: Option<Arc<MemoryMonitor>>,
    memory_limit_controller: Option<Arc<MemoryLimitController>>,
    fallback_triggered: bool,
    fallback_lock: Arc<Mutex<()>>, // 保护降级过程的互斥锁
    device_type: DeviceType,
    supports_cuda: bool,
    model_name: String,
}

impl OnnxEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, VecboostError> {
        Self::with_device(config, precision, config.device.clone())
    }

    pub fn with_device(
        config: &ModelConfig,
        precision: Precision,
        device_type: DeviceType,
    ) -> Result<Self, VecboostError> {
        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();

        let (onnx_filename, tokenizer_filename) = if is_local_path {
            log::info!("Using local model path: {:?}", model_path);
            resolve_local_bundle(model_path, config.tokenizer_path.as_deref())?
        } else {
            log::info!("Using HuggingFace Hub for model: {:?}", model_path);
            let repo_id = model_path.to_string_lossy().into_owned();
            let repo = build_hf_repo(&repo_id)?;

            log::info!("Downloading/Loading ONNX model files...");
            let onnx_filename = repo
                .download_file()
                .filename("model.onnx")
                .send()
                .or_else(|_| repo.download_file().filename("model_quantized.onnx").send())
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

            let tokenizer_filename = repo
                .download_file()
                .filename("tokenizer.json")
                .send()
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
            (onnx_filename, tokenizer_filename)
        };

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        let supports_cuda = device_type == DeviceType::Cuda;
        let supports_amd = matches!(device_type, DeviceType::Amd | DeviceType::OpenCL);

        log::info!("Initializing ONNX Runtime session...");
        let session = Session::builder()
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        let mut session = if supports_cuda {
            log::info!("Attempting to configure CUDA execution provider for ONNX Runtime");
            #[cfg(feature = "cuda")]
            {
                let cuda_provider =
                    ort::execution_providers::CUDAExecutionProvider::default().build();
                session
                    .with_execution_providers([cuda_provider])
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                log::warn!(
                    "CUDA execution provider not available. ONNX Runtime CUDA support requires cuda feature flag. Using CPU execution provider."
                );
                session
            }
        } else if supports_amd {
            log::info!(
                "AMD GPU detected but ROCm execution provider is not configured in this build. Using CPU execution provider."
            );
            session
        } else {
            session
        };

        if let Some(ref expected_hash) = config.model_sha256 {
            log::info!("Verifying model file SHA256 hash...");
            let is_valid = verify_sha256(&onnx_filename, expected_hash).map_err(|e| {
                VecboostError::ModelLoadError(format!("Failed to verify SHA256: {}", e))
            })?;

            if !is_valid {
                return Err(VecboostError::ModelLoadError(format!(
                    "Model file SHA256 verification failed. Expected: {}, File: {:?}",
                    expected_hash, onnx_filename
                )));
            }

            log::info!("Model file SHA256 verification passed");
        }

        let session = session
            .commit_from_file(onnx_filename)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        log::info!("Loading tokenizer...");
        #[allow(unused_mut)]
        let mut tokenizer = Tokenizer::from_file(&tokenizer_filename)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        // 全平台启用 padding 配置
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vocab_size = tokenizer.get_vocab_size(true);
        let hidden_size = config.expected_dimension.unwrap_or(1024);
        log::info!(
            "Using hidden_size from configuration: {:?}",
            config.expected_dimension
        );
        log::info!("Final hidden_size value: {}", hidden_size);
        // 序列截断上限与 tokenizer 层一致取 512；
        // 词表大小是 embedding 数量，与序列长度是两个量纲，不可混用
        let max_input_length = ONNX_MAX_INPUT_LENGTH;

        let actual_precision = match precision {
            Precision::Fp16 => {
                if supports_cuda || supports_amd {
                    log::info!("Using FP16 precision with GPU acceleration");
                    Precision::Fp16
                } else {
                    log::warn!("FP16 not supported without GPU acceleration, falling back to FP32");
                    Precision::Fp32
                }
            }
            _ => {
                log::info!("Using {} precision", precision);
                precision
            }
        };

        log::info!(
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
            fallback_lock: Arc::new(Mutex::new(())),
            device_type,
            supports_cuda,
            model_name: config.name.clone(),
        })
    }

    pub fn set_memory_limit_controller(&mut self, controller: Arc<MemoryLimitController>) {
        self.memory_limit_controller = Some(controller);
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    pub fn get_model_name(&self) -> &str {
        &self.model_name
    }

    pub fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    pub fn check_memory_pressure(&self, threshold_percent: u64) -> bool {
        if let Some(ref monitor) = self.memory_monitor {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.block_on(async {
                    let stats = monitor.get_memory_stats().await;
                    let usage_percent = (stats.current_bytes * 100)
                        .checked_div(stats.total_bytes)
                        .unwrap_or(0);
                    usage_percent >= threshold_percent
                })
            } else {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        log::error!("Failed to create Tokio runtime for memory check: {}", e);
                        return false;
                    }
                };
                rt.block_on(async {
                    let stats = monitor.get_memory_stats().await;
                    let usage_percent = (stats.current_bytes * 100)
                        .checked_div(stats.total_bytes)
                        .unwrap_or(0);
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
    ) -> Result<bool, VecboostError> {
        if self.fallback_triggered {
            return Ok(false);
        }

        if let Some(ref controller) = self.memory_limit_controller {
            let status = controller.check_limit().await;

            if status == MemoryLimitStatus::Exceeded {
                log::warn!("Memory limit exceeded for ONNX engine, attempting fallback to CPU");
                self.try_fallback_to_cpu(config).await?;
                return Ok(true);
            } else if status == MemoryLimitStatus::Critical {
                log::warn!(
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

    fn forward_pass(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| VecboostError::TokenizationError(e.to_string()))?;

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
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.into_dyn())
            .map_err(|e: ort::Error| VecboostError::InferenceError(e.to_string()))?;

        let last_hidden_state = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            let output_array = outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .to_owned();
            log::debug!("ONNX model output shape: {:?}", output_array.shape());
            output_array
        };

        let seq_len = attention_mask.iter().filter(|&&v| v == 1).count();
        if seq_len == 0 {
            return Err(VecboostError::InferenceError(
                "Empty sequence after mask".to_string(),
            ));
        }

        let mut flat = Vec::with_capacity(attention_mask.len() * self.hidden_size);
        for seq_idx in 0..attention_mask.len() {
            for h in 0..self.hidden_size {
                flat.push(last_hidden_state[[seq_idx, h]]);
            }
        }
        let mut pooled = onnx_pool_mean(&flat, &attention_mask, self.hidden_size);
        l2_normalize_in_place(&mut pooled);
        Ok(pooled)
    }

    pub async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        // 使用互斥锁确保只有一个线程执行降级
        let _lock = self.fallback_lock.lock().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to acquire fallback lock: {}", e))
        })?;

        // 双重检查：获取锁后再次检查是否已经降级
        if self.fallback_triggered {
            return Ok(());
        }

        log::info!("Attempting fallback from GPU to CPU for ONNX engine");

        self.memory_monitor = None;
        self.device_type = DeviceType::Cpu;
        self.supports_cuda = false;
        self.fallback_triggered = true;

        let repo_id = config.model_path.to_string_lossy().into_owned();
        let repo = build_hf_repo(&repo_id)?;

        let onnx_filename = repo
            .download_file()
            .filename("model.onnx")
            .send()
            .or_else(|_| repo.download_file().filename("model_quantized.onnx").send())
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        let new_session = Session::builder()
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .with_intra_threads(num_threads)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .commit_from_file(onnx_filename)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        *session_guard = new_session;
        drop(session_guard);

        self.precision = Precision::Fp32;

        log::info!("Successfully fell back to CPU for ONNX engine");
        Ok(())
    }

    fn forward_pass_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
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
                .map_err(|e| VecboostError::TokenizationError(e.to_string()))?;

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
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let attention_mask_array =
            Array2::from_shape_vec((padded_batch_size, max_seq_len), batch_attention_mask)
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        let input_ids_tensor = Tensor::from_array(input_ids_array.into_dyn())
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array.into_dyn())
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        let last_hidden_state = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .to_owned()
        };

        let mut results = Vec::with_capacity(padded_batch_size);
        for batch_idx in 0..padded_batch_size {
            let attention_mask = &all_attention_masks[batch_idx];
            let actual_seq_len = attention_mask.len();
            log::debug!(
                "batch_idx: {}, max_seq_len: {}, actual_seq_len: {}",
                batch_idx,
                max_seq_len,
                actual_seq_len
            );

            let effective_max_seq = std::cmp::min(max_seq_len, actual_seq_len);
            let mut flat = Vec::with_capacity(effective_max_seq * self.hidden_size);
            for seq_idx in 0..effective_max_seq {
                for h in 0..self.hidden_size {
                    flat.push(last_hidden_state[[batch_idx, seq_idx, h]]);
                }
            }
            let mut pooled = onnx_pool_mean(
                &flat,
                &attention_mask[..effective_max_seq],
                self.hidden_size,
            );
            l2_normalize_in_place(&mut pooled);
            results.push(pooled);
        }

        Ok(results)
    }
}

/// 对单个样本的扁平 hidden states `[seq_len, hidden_size]` 做 mean pooling。
/// mask 每 token 计一次——除数是 token 计数，与 hidden_size 无关；
/// mask=0 的 token（含前导 padding）整行排除。
pub(crate) fn onnx_pool_mean(hidden: &[f32], mask: &[i64], hidden_size: usize) -> Vec<f32> {
    let mut weighted_sum = vec![0.0f32; hidden_size];
    let mut mask_sum = 0.0f32;
    for (seq_idx, &mask_val) in mask.iter().enumerate() {
        if mask_val == 1 {
            mask_sum += 1.0;
            let row = &hidden[seq_idx * hidden_size..(seq_idx + 1) * hidden_size];
            for (h, &v) in row.iter().enumerate() {
                weighted_sum[h] += v;
            }
        }
    }
    if mask_sum > 0.0 {
        weighted_sum.iter_mut().for_each(|v| *v /= mask_sum);
    }
    weighted_sum
}

#[async_trait]
impl InferenceEngine for OnnxEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        self.forward_pass(text)
    }

    fn attach_memory_limit_controller(
        &mut self,
        controller: std::sync::Arc<crate::device::memory_limit::MemoryLimitController>,
    ) {
        self.set_memory_limit_controller(controller);
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        self.forward_pass_batch(texts)
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        self.memory_monitor.is_some()
    }

    fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        OnnxEngine::try_fallback_to_cpu(self, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{EngineType, ModelConfig};
    use std::path::PathBuf;

    /// 回归钉：mean 池化除数是 token 计数，不随 hidden_size 放大。
    /// 旧实现 mask_sum 累加写在 hidden 维循环内，除数被放大 hidden_size 倍。
    #[test]
    fn onnx_pool_mean_divides_by_token_count_not_hidden() {
        let hidden = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mask = vec![1, 1];
        let out = onnx_pool_mean(&hidden, &mask, 4);
        assert_eq!(out, vec![0.5, 0.5, 0.0, 0.0]);
    }

    /// 前导 padding（mask=0）的 token 必须被排除，不污染均值。
    #[test]
    fn onnx_pool_mean_excludes_masked_out_tokens() {
        let hidden = vec![100.0, 100.0, 5.0, 7.0];
        let mask = vec![0, 1];
        let out = onnx_pool_mean(&hidden, &mask, 2);
        assert_eq!(out, vec![5.0, 7.0]);
    }

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-onnx".to_string(),
            engine_type: EngineType::Onnx,
            model_path: PathBuf::from("/nonexistent/onnx-model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        }
    }

    /// 验证 OnnxEngine::new 在空目录上返回 ModelLoadError
    #[test]
    fn test_onnx_engine_new_returns_error_for_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::new(&config, Precision::Fp32);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError, got {:?}",
                e
            );
        }
    }

    /// 验证 with_device 在空目录(CPU)上返回带特定消息的 ModelLoadError
    #[test]
    fn test_onnx_engine_with_device_empty_dir_cpu() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("No ONNX model found"),
                "Expected 'No ONNX model found', got: {}",
                msg
            );
        }
    }

    /// 验证 FP16 精度在空目录上仍返回错误
    #[test]
    fn test_onnx_engine_with_device_fp16_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp16, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 INT8 精度在空目录上仍返回错误
    #[test]
    fn test_onnx_engine_with_device_int8_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Int8, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 CUDA 设备类型在空目录上返回错误(覆盖 supports_cuda 分支)
    #[test]
    fn test_onnx_engine_with_device_cuda_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cuda);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError, got {:?}",
                e
            );
        }
    }

    /// 验证 AMD 设备类型在空目录上返回错误(覆盖 supports_amd 分支)
    #[test]
    fn test_onnx_engine_with_device_amd_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Amd);
        assert!(result.is_err());
    }

    /// 验证 OpenCL 设备类型在空目录上返回错误
    #[test]
    fn test_onnx_engine_with_device_opencl_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::OpenCL);
        assert!(result.is_err());
    }

    /// 验证存在 model.onnx 但缺失 tokenizer.json 时返回 "Tokenizer not found" 错误
    #[test]
    fn test_onnx_engine_with_model_but_no_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Tokenizer not found"),
                "Expected 'Tokenizer not found', got: {}",
                msg
            );
        }
    }

    /// 验证存在 model_quantized.onnx(优先于 model.onnx)但缺失 tokenizer.json 时返回错误
    #[test]
    fn test_onnx_engine_prefers_quantized_model_but_no_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("model_quantized.onnx"),
            b"fake quantized",
        )
        .expect("Failed to write fake model_quantized.onnx");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Tokenizer not found"),
                "Expected 'Tokenizer not found', got: {}",
                msg
            );
        }
    }

    /// 验证 `Precision` 的 Display 实现
    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::Fp32.to_string(), "fp32");
        assert_eq!(Precision::Fp16.to_string(), "fp16");
        assert_eq!(Precision::Int8.to_string(), "int8");
    }

    /// 验证 `DeviceType` 的序列化形式(serde rename)
    #[test]
    fn test_device_type_serialization() {
        assert_eq!(
            serde_json::to_string(&DeviceType::Cpu).expect("serialize Cpu"),
            "\"cpu\""
        );
        assert_eq!(
            serde_json::to_string(&DeviceType::Amd).expect("serialize Amd"),
            "\"amd\""
        );
        assert_eq!(
            serde_json::to_string(&DeviceType::OpenCL).expect("serialize OpenCL"),
            "\"opencl\""
        );
    }

    /// 验证 model.onnx + tokenizer.json 同时存在时,在 commit_from_file 阶段失败
    /// (fake ONNX 文件无法被 ONNX Runtime 解析)
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_model_and_tokenizer_fails_at_session() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from session commit, got {:?}",
                e
            );
        }
    }

    /// 验证 model_quantized.onnx 优先于 model.onnx 被选择,且在 session 阶段失败
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_quantized_preferred_with_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("model_quantized.onnx"),
            b"fake quantized",
        )
        .expect("Failed to write fake model_quantized.onnx");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from session commit, got {:?}",
                e
            );
        }
    }

    /// 回归钉：HuggingFace cache 父目录（parent's parent）的 tokenizer.json
    /// 不再被本地分支探测——spec R-model-bundle-fetch-002 三级顺序
    /// （tokenizer_path → 根目录 → tokenizer/ 子目录）皆缺即显性报错。
    /// 探测失败发生在 Session 构造之前，不触发 ort 环境。
    #[test]
    fn test_onnx_engine_cache_tokenizer_no_longer_consulted() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let cache_dir = temp_dir.path();
        let intermediate = cache_dir.join("intermediate");
        let model_dir = intermediate.join("model_dir");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model_dir");
        std::fs::write(model_dir.join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(cache_dir.join("tokenizer.json"), b"{}")
            .expect("Failed to write cache tokenizer.json");

        let mut config = test_config();
        config.model_path = model_dir;

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(msg.contains("Tokenizer not found"), "got: {msg}");
        } else {
            panic!("Expected ModelLoadError: cache-parent tokenizer must not be consulted");
        }
    }

    /// 验证设置 model_sha256 时,SHA256 校验失败返回 ModelLoadError
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_sha256_verification_mismatch() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();
        config.model_sha256 =
            Some("0000000000000000000000000000000000000000000000000000000000000000".to_string());

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("SHA256 verification failed"),
                "Expected SHA256 verification failure, got: {}",
                msg
            );
        }
    }

    /// 验证 CUDA 设备类型在 model.onnx + tokenizer.json 存在时,
    /// 进入 CUDA 配置分支(非 cuda feature 下走 warn 路径)后在 commit_from_file 失败
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_cuda_device_with_model_and_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cuda);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from CUDA path, got {:?}",
                e
            );
        }
    }

    /// 验证 AMD 设备类型在 model.onnx + tokenizer.json 存在时,
    /// 进入 AMD 配置分支后在 commit_from_file 失败
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_amd_device_with_model_and_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Amd);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from AMD path, got {:?}",
                e
            );
        }
    }

    /// 验证 OpenCL 设备类型在 model.onnx + tokenizer.json 存在时,
    /// 进入 AMD/OpenCL 配置分支后在 commit_from_file 失败
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_opencl_device_with_model_and_tokenizer() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::OpenCL);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from OpenCL path, got {:?}",
                e
            );
        }
    }

    /// 验证 FP16 + CUDA 设备在 model.onnx + tokenizer.json 存在时进入 CUDA 分支后失败
    #[ignore = "ort crate environment cleanup crashes test runner when Session::builder() is called"]
    #[test]
    fn test_onnx_engine_fp16_cuda_device_with_model() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");
        std::fs::write(temp_dir.path().join("tokenizer.json"), b"{}")
            .expect("Failed to write fake tokenizer.json");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp16, DeviceType::Cuda);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError from FP16+CUDA path, got {:?}",
                e
            );
        }
    }

    /// 三级 tokenizer 探测皆缺时返回 "Tokenizer not found" 错误
    /// （错误消息列出检查过的路径：根目录与 tokenizer/ 子目录）
    #[test]
    fn test_onnx_engine_tokenizer_missing_reports_checked_paths() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Tokenizer not found"),
                "Expected 'Tokenizer not found', got: {}",
                msg
            );
            assert!(
                msg.contains("tokenizer.json"),
                "错误必须列出检查过的 tokenizer 路径清单, got: {}",
                msg
            );
        }
    }

    /// 嵌套目录中的 bundle 同样走三级探测，皆缺时报 "Tokenizer not found"
    /// 且消息含 tokenizer/ 子目录候选
    #[test]
    fn test_onnx_engine_nested_bundle_tokenizer_missing_lists_subdir_candidate() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let intermediate = temp_dir.path().join("intermediate");
        let model_dir = intermediate.join("model_dir");
        std::fs::create_dir_all(&model_dir).expect("Failed to create model_dir");
        std::fs::write(model_dir.join("model.onnx"), b"fake onnx")
            .expect("Failed to write fake model.onnx");

        let mut config = test_config();
        config.model_path = model_dir;

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Tokenizer not found"),
                "Expected 'Tokenizer not found', got: {}",
                msg
            );
            assert!(
                msg.contains("tokenizer"),
                "清单必须含 tokenizer/ 子目录候选, got: {}",
                msg
            );
        }
    }

    /// 验证仅存在 model_quantized.onnx(无 model.onnx)且无 tokenizer 时返回错误
    #[test]
    fn test_onnx_engine_only_quantized_no_model_onnx() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(
            temp_dir.path().join("model_quantized.onnx"),
            b"fake quantized",
        )
        .expect("Failed to write fake model_quantized.onnx");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Tokenizer not found") || msg.contains("Cannot determine"),
                "Expected tokenizer/cache error, got: {}",
                msg
            );
        }
    }

    /// 验证 FP16 + AMD 设备在空目录上返回错误(覆盖 FP16 + supports_amd 分支)
    #[test]
    fn test_onnx_engine_fp16_amd_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp16, DeviceType::Amd);
        assert!(result.is_err());
    }

    /// 验证 FP16 + OpenCL 设备在空目录上返回错误(覆盖 FP16 + supports_amd 分支)
    #[test]
    fn test_onnx_engine_fp16_opencl_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp16, DeviceType::OpenCL);
        assert!(result.is_err());
    }

    /// 验证 INT8 + CUDA 设备在空目录上返回错误
    #[test]
    fn test_onnx_engine_int8_cuda_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Int8, DeviceType::Cuda);
        assert!(result.is_err());
    }

    /// 验证 INT8 + AMD 设备在空目录上返回错误
    #[test]
    fn test_onnx_engine_int8_amd_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Int8, DeviceType::Amd);
        assert!(result.is_err());
    }

    /// 验证 model_path 指向文件(非目录)时进入 HuggingFace Hub 下载路径并失败
    #[test]
    fn test_onnx_engine_model_path_is_file_not_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("not_a_dir");
        std::fs::write(&file_path, b"file").expect("Failed to write file");

        let mut config = test_config();
        config.model_path = file_path;

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 Metal 设备类型在空目录上返回错误
    #[test]
    fn test_onnx_engine_metal_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = OnnxEngine::with_device(&config, Precision::Fp32, DeviceType::Metal);
        assert!(result.is_err());
    }

    // =========================================================================
    // resolve_local_bundle 本地 bundle 探测测试（离线，不经 Session 构造）
    // =========================================================================

    fn write_file(dir: &std::path::Path, rel: &str, contents: &[u8]) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent dir");
        }
        std::fs::write(path, contents).expect("write test file");
    }

    /// 官方 laya bundle 形态：`laya.onnx`（非 model.onnx 命名）被识别
    #[test]
    fn resolve_local_bundle_recognizes_laya_onnx() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "laya.onnx", b"fake");
        write_file(dir.path(), "tokenizer/tokenizer.json", b"{}");

        let (onnx, tokenizer) =
            resolve_local_bundle(dir.path(), None).expect("laya bundle recognized");
        assert_eq!(onnx, dir.path().join("laya.onnx"));
        assert_eq!(tokenizer, dir.path().join("tokenizer/tokenizer.json"));
    }

    /// tokenizer 子目录形态被识别（根目录无 tokenizer.json 时）
    #[test]
    fn resolve_local_bundle_recognizes_tokenizer_subdir() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "model.onnx", b"fake");
        write_file(dir.path(), "tokenizer/tokenizer.json", b"{}");

        let (onnx, tokenizer) =
            resolve_local_bundle(dir.path(), None).expect("subdir tokenizer recognized");
        assert_eq!(onnx, dir.path().join("model.onnx"));
        assert_eq!(tokenizer, dir.path().join("tokenizer/tokenizer.json"));
    }

    /// 多个非 model 命名的 onnx 候选时显性报错并列出全部候选文件名
    #[test]
    fn resolve_local_bundle_multiple_onnx_candidates_error() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "laya.onnx", b"fake");
        write_file(dir.path(), "other.onnx", b"fake");

        let err = resolve_local_bundle(dir.path(), None).unwrap_err();
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("laya.onnx"), "candidate list missing: {msg}");
                assert!(msg.contains("other.onnx"), "candidate list missing: {msg}");
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 优先级钉：model_quantized.onnx > model.onnx > 唯一 *.onnx
    #[test]
    fn resolve_local_bundle_onnx_priority_order() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "model_quantized.onnx", b"fake");
        write_file(dir.path(), "model.onnx", b"fake");
        write_file(dir.path(), "tokenizer.json", b"{}");

        let (onnx, _) = resolve_local_bundle(dir.path(), None).expect("quantized recognized");
        assert_eq!(onnx, dir.path().join("model_quantized.onnx"));

        let dir2 = tempfile::tempdir().expect("temp dir");
        write_file(dir2.path(), "model.onnx", b"fake");
        write_file(dir2.path(), "laya.onnx", b"fake");
        write_file(dir2.path(), "tokenizer.json", b"{}");
        let (onnx2, _) = resolve_local_bundle(dir2.path(), None).expect("model.onnx picked");
        assert_eq!(onnx2, dir2.path().join("model.onnx"));
    }

    /// tokenizer_path 显式配置优先于根目录与子目录
    #[test]
    fn resolve_local_bundle_tokenizer_override_wins() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "model.onnx", b"fake");
        write_file(dir.path(), "tokenizer.json", b"{}");
        write_file(dir.path(), "tokenizer/tokenizer.json", b"{}");
        write_file(dir.path(), "custom/tok.json", b"{}");
        let override_path = dir.path().join("custom/tok.json");

        let (_, tokenizer) =
            resolve_local_bundle(dir.path(), Some(&override_path)).expect("override recognized");
        assert_eq!(tokenizer, override_path);
    }

    /// tokenizer_path 已配置但路径不存在时显性报错，禁止静默回落
    #[test]
    fn resolve_local_bundle_missing_tokenizer_override_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "model.onnx", b"fake");
        write_file(dir.path(), "tokenizer.json", b"{}");
        let missing = dir.path().join("nowhere/tok.json");

        let err = resolve_local_bundle(dir.path(), Some(&missing)).unwrap_err();
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(
                    msg.contains("nowhere/tok.json") || msg.contains("nowhere"),
                    "error must name the configured override path, got: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 三级 tokenizer 探测皆缺时显性报错
    #[test]
    fn resolve_local_bundle_no_tokenizer_anywhere_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_file(dir.path(), "model.onnx", b"fake");

        let err = resolve_local_bundle(dir.path(), None).unwrap_err();
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("Tokenizer not found"), "got: {msg}");
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 目录无任何 onnx 文件时报 No ONNX model found（既有口径不回退）
    #[test]
    fn resolve_local_bundle_no_onnx_error() {
        let dir = tempfile::tempdir().expect("temp dir");

        let err = resolve_local_bundle(dir.path(), None).unwrap_err();
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("No ONNX model found"), "got: {msg}");
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }
}
