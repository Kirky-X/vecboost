// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::manual_checked_ops, clippy::identity_op)]
use super::InferenceEngine;
use super::{Stage, StageSnapshot, StageStats};
use crate::config::model::{DeviceType, ModelConfig, Precision};
use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
use crate::error::VecboostError;
use crate::model::recovery::{ModelRecovery, RecoveryConfig};
use crate::monitor::MemoryMonitor;
use crate::text::{Encoding, SyncTokenizer};
use crate::utils::hash::{check_model_integrity, verify_sha256};
use crate::utils::hf_hub::build_hf_repo;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
// 全平台统一使用 tokenizers crate
use tokenizers::Tokenizer as HfTokenizer;

/// PyTorch 大文件警告阈值（2 GB）
const PYTORCH_LARGE_FILE_WARN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// 每种精度每个参数占用的字节数
fn bytes_per_param(precision: Precision) -> u64 {
    match precision {
        Precision::Fp32 => 4,
        Precision::Fp16 | Precision::Bf16 => 2,
        Precision::Int8 => 1,
    }
}

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

/// 分阶段计时守卫：作用域退出（含 `?` 提前返回）时自动累加耗时。
/// 无锁快路径：底层为原子 fetch_add。
struct StageTimer<'a> {
    stats: &'a StageStats,
    stage: Stage,
    start: std::time::Instant,
}

impl<'a> StageTimer<'a> {
    fn new(stats: &'a StageStats, stage: Stage) -> Self {
        Self {
            stats,
            stage,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for StageTimer<'_> {
    fn drop(&mut self) {
        self.stats.record(self.stage, self.start.elapsed());
    }
}

pub struct CandleEngine {
    model: ModelWrapper,
    tokenizer: SyncTokenizer,
    device: Device,
    precision: Precision,
    memory_monitor: Option<Arc<MemoryMonitor>>,
    memory_limit_controller: Option<Arc<MemoryLimitController>>,
    fallback_triggered: bool,
    device_type: DeviceType,
    model_architecture: ModelArchitecture,
    use_quantization: bool, // INT8 量化未实现，始终为 false（保留字段供未来实现）
    _model_name: String,
    /// 模型隐藏层大小（从 config.hidden_size 读取）
    hidden_size: usize,
    /// 分阶段延迟累加器：tokenize/inference/pooling，无锁原子累加。
    stage_stats: Arc<StageStats>,
    /// 模型参数数量估算（从 config 计算）
    parameter_count: u64,
    /// 汇聚模式(Auto 在构造时已解析为具体模式)
    pooling_mode: crate::config::model::PoolingMode,
    /// 词表大小(从 config.json vocab_size 读取,替代硬编码 250002)
    vocab_size: usize,
}

/// 从 BERT/RoBERTa config 估算参数数量。
///
/// 公式：embeddings + layers * (attention + ffn + norms)
/// - embeddings = vocab * h + max_pos * h + type_vocab * h + layer_norm
/// - attention per layer = 4 * h² + 2 * h (Q/K/V/O + bias)
/// - ffn per layer = 2 * h * intermediate + 2 * h
/// - norms per layer = 2 * 2 * h (two LayerNorms, weight + bias each)
fn estimate_bert_params(
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    _num_attention_heads: usize,
) -> u64 {
    let h = hidden_size as u64;
    let v = vocab_size as u64;
    let layers = num_hidden_layers as u64;
    let intermediate = intermediate_size as u64;

    // Embedding layer
    let embedding_params = v * h + 512 * h + 2 * h + h; // word + pos + token_type + layer_norm

    // Per-layer parameters
    let attention_params = 4 * h * h + 2 * h; // Q/K/V/O projections + biases
    let ffn_params = 2 * h * intermediate + 2 * h; // up + down projections + biases
    let norm_params = 4 * h; // two LayerNorms (weight + bias each)
    let per_layer = attention_params + ffn_params + norm_params;

    // Pooler (dense + tanh)
    let pooler_params = h * h + h;

    embedding_params + layers * per_layer + pooler_params
}

impl CandleEngine {
    pub fn new(config: &ModelConfig, precision: Precision) -> Result<Self, VecboostError> {
        Self::with_device(config, precision, config.device.clone())
    }

    pub fn with_device(
        config: &ModelConfig,
        precision: Precision,
        device_type: DeviceType,
    ) -> Result<Self, VecboostError> {
        let device = resolve_compute_device(&device_type)?;
        // 确定计算数据类型（FP16/BF16 按设备映射；INT8 量化未实现，诚实回退 FP32）
        let (dtype, use_quantization) = resolve_compute_dtype(&precision, device.is_cuda());

        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();
        let (config_filename, tokenizer_filename, weights_filename) =
            resolve_model_files(config, model_path, is_local_path)?;

        let config_content = std::fs::read_to_string(&config_filename)?;
        let parsed = parse_model_config(&config_content)?;

        let model_architecture = parsed.architecture;
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-detected-model-architecture",
                crate::i18n::tr_args(&[("a1", &format!("{:?}", model_architecture))])
            )
        );

        let hf_tokenizer = HfTokenizer::from_file(tokenizer_filename.to_string_lossy().as_ref())
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        let tokenizer = SyncTokenizer::new(hf_tokenizer, parsed.max_position_embeddings);

        ensure_model_integrity(
            config,
            model_path,
            is_local_path,
            &[
                config_filename.clone(),
                tokenizer_filename.clone(),
                weights_filename.clone(),
            ],
        )?;

        let is_pytorch = weights_filename.to_string_lossy().ends_with(".bin");
        let mut varmap = VarMap::new();
        let vb = load_weights_varbuilder(
            is_pytorch,
            &weights_filename,
            &mut varmap,
            dtype,
            &device,
            config,
        )?;

        let model = match &model_architecture {
            ModelArchitecture::Bert => {
                let config = parsed.bert_config.ok_or_else(|| {
                    VecboostError::ModelLoadError(crate::i18n::tr("engine-bert-config-required"))
                })?;
                let bert_model = BertModel::load(vb, &config)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                ModelWrapper::Bert(bert_model)
            }
            ModelArchitecture::XlmRoberta => {
                let config = parsed.xlm_config.ok_or_else(|| {
                    VecboostError::ModelLoadError(crate::i18n::tr("engine-xlm-config-required"))
                })?;
                let xlm_model = XLMRobertaModel::new(&config, vb)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                ModelWrapper::XlmRoberta(xlm_model)
            }
        };

        let memory_monitor = if device.is_cuda() || device.is_metal() {
            Some(Arc::new(MemoryMonitor::new()))
        } else {
            None
        };

        // 解析 pooling 模式——Auto 按模型名推断为具体模式
        let resolved_pooling = match config.pooling_mode.clone().unwrap_or_default() {
            crate::config::model::PoolingMode::Auto => super::infer_pooling_mode(&config.name),
            other => other,
        };
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-pooling-mode-model",
                crate::i18n::tr_args(&[
                    ("a1", &format!("{:?}", resolved_pooling)),
                    ("a2", &(config.name).to_string())
                ])
            )
        );

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
            use_quantization,
            _model_name: config.name.clone(),
            hidden_size: parsed.hidden_size,
            parameter_count: parsed.parameter_count,
            pooling_mode: resolved_pooling,
            vocab_size: parsed.vocab_size,
            stage_stats: Arc::new(StageStats::default()),
        })
    }

    pub fn set_memory_limit_controller(&mut self, controller: Arc<MemoryLimitController>) {
        self.memory_limit_controller = Some(controller);
    }

    /// 分阶段延迟快照（非破坏性，供诊断）。
    pub fn stage_snapshot(&self) -> StageSnapshot {
        self.stage_stats.snapshot()
    }

    /// 取出并清零分阶段延迟（指标汇出用）。
    pub fn take_stage_snapshot(&self) -> StageSnapshot {
        self.stage_stats.take()
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    pub fn get_model_name(&self) -> &str {
        &self._model_name
    }

    pub fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    pub async fn check_memory_pressure(&self, threshold_percent: u64) -> bool {
        if let Some(ref monitor) = self.memory_monitor {
            let stats = monitor.get_memory_stats().await;
            let usage_percent = if stats.total_bytes > 0 {
                // 使用 u128 中间值避免 current_bytes * 100 溢出 u64
                ((stats.current_bytes as u128 * 100) / stats.total_bytes as u128) as u64
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
    ) -> Result<bool, VecboostError> {
        // 如果已触发 fallback，检查是否可以恢复
        if self.fallback_triggered {
            // 内存压力解除后，尝试恢复到 GPU
            if !self.check_memory_pressure(50).await {
                log::info!(
                    "{}",
                    crate::i18n::tr("log-engine-memory-pressure-cleared-resetting")
                );
                self.fallback_triggered = false;
            }
            return Ok(false);
        }

        if let Some(ref controller) = self.memory_limit_controller {
            let status = controller.check_limit().await;

            if status == MemoryLimitStatus::Exceeded {
                log::warn!(
                    "{}",
                    crate::i18n::tr("log-engine-memory-limit-exceeded-attempting")
                );
                self.try_fallback_to_cpu(config).await?;
                return Ok(true);
            } else if status == MemoryLimitStatus::Critical {
                log::warn!(
                    "{}",
                    crate::i18n::tr("log-engine-memory-limit-critical-checking")
                );
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

    // 纯同步 forward_pass——归一化已由 forward_pass_inner 出口统一完成
    fn forward_pass(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        self.forward_pass_inner(text)
    }

    fn forward_pass_inner(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        // 分阶段埋点：tokenize（守卫 Drop 时累加，`?` 提前返回亦覆盖）。
        let encoding = {
            let _timer = StageTimer::new(&self.stage_stats, Stage::Tokenize);
            self.tokenizer
                .encode(text, true)
                .map_err(|e| VecboostError::TokenizationError(e.to_string()))?
        };

        // 从 config.json 读取的 vocab_size(替代硬编码 250002)
        let vocab_size = self.vocab_size as u32;
        let max_len = self.tokenizer.max_length().min(512);
        let (token_ids, token_type_ids, attention_mask_tensor, mask_for_pooling) =
            self.prepare_single_input(&encoding, vocab_size, max_len)?;

        let embeddings = {
            let _timer = StageTimer::new(&self.stage_stats, Stage::Inference);
            self.run_model_forward(
                &token_ids,
                &token_type_ids,
                &attention_mask_tensor,
                &encoding,
                max_len,
            )?
        };

        // pooling 计时守卫驻留至函数返回，覆盖全部三个出口。
        let _pool_timer = StageTimer::new(&self.stage_stats, Stage::Pooling);
        self.pool_single_embedding(&embeddings, mask_for_pooling)
    }

    /// 单条输入张量化：vocab 越界 clamp（warn 一次）+ max_len 截断，产出
    /// ids/type_ids/mask 三张量与池化用 mask 副本。
    fn prepare_single_input(
        &self,
        encoding: &Encoding,
        vocab_size: u32,
        max_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Vec<u32>), VecboostError> {
        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        log::debug!("Token IDs: {:?}", ids);
        log::debug!("Max token ID: {}", ids.iter().max().copied().unwrap_or(0));
        log::debug!("Attention mask: {:?}", attention_mask);

        let max_id = ids.iter().max().copied().unwrap_or(0);
        if max_id >= vocab_size {
            log::warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-engine-token-id-exceeds-vocab",
                    crate::i18n::tr_args(&[
                        ("a1", &(max_id).to_string()),
                        ("a2", &(vocab_size).to_string())
                    ])
                )
            );
        }

        let ids_slice: Vec<u32> = ids
            .iter()
            .take(max_len)
            .map(|&id| if id >= vocab_size { vocab_size - 1 } else { id })
            .collect();

        let mask_slice: Vec<u32> = attention_mask
            .iter()
            .take(max_len)
            // attention_mask 值为 0/1,不需要 vocab 截断
            .copied()
            .collect();

        let token_ids = Tensor::new(ids_slice, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        // 保留 mask 副本供 pooling 函数使用
        let mask_for_pooling = mask_slice.clone();
        let attention_mask_tensor = Tensor::new(mask_slice, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        // 修复：candle BertModel::forward 签名为
        // (input_ids, token_type_ids, attention_mask: Option)——旧实现把
        // attention_mask 误传到 token_type_ids 槽位、mask 传 None，
        // 导致 segment embedding 错误（单条）且批内 padding 完全无隔离（批量）。
        let type_ids_slice: Vec<u32> = encoding.type_ids.iter().take(max_len).cloned().collect();
        let token_type_ids = Tensor::new(type_ids_slice, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        Ok((
            token_ids,
            token_type_ids,
            attention_mask_tensor,
            mask_for_pooling,
        ))
    }

    /// 模型前向（单条）：BERT 臂 (ids, type_ids, Some(mask))；XLM-R 臂
    /// 签名不同（mask 居第二位）且内部重建 type_ids（重复构建原样保留）。
    fn run_model_forward(
        &self,
        token_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask_tensor: &Tensor,
        encoding: &Encoding,
        max_len: usize,
    ) -> Result<Tensor, VecboostError> {
        match &self.model {
            ModelWrapper::Bert(bert_model) => bert_model
                .forward(token_ids, token_type_ids, Some(attention_mask_tensor))
                .map_err(|e| VecboostError::InferenceError(e.to_string())),
            ModelWrapper::XlmRoberta(xlm_model) => {
                let type_ids_slice: Vec<u32> =
                    encoding.type_ids.iter().take(max_len).cloned().collect();
                let token_type_ids = Tensor::new(type_ids_slice, &self.device)
                    .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
                xlm_model
                    .forward(
                        token_ids,
                        attention_mask_tensor,
                        &token_type_ids,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| VecboostError::InferenceError(e.to_string()))
            }
        }
    }

    /// 单条池化：1D 输出（架构内建单向量）早退；3D/2D 展平后按
    /// pooling_mode 汇聚，出口 L2 归一化（与批量路径、量化引擎一致）。
    fn pool_single_embedding(
        &self,
        embeddings: &Tensor,
        mask_for_pooling: Vec<u32>,
    ) -> Result<Vec<f32>, VecboostError> {
        log::debug!("Embeddings shape: {:?}", embeddings.shape());
        log::debug!("Embeddings dims: {:?}", embeddings.dims());

        // 移除 update_gpu_memory().await——GPU 监控由独立后台任务负责

        let dims = embeddings.dims();
        let hidden_dim = self.hidden_size;

        // 按 pooling_mode 汇聚——将 [seq_len, hidden_dim] 展平数据 + mask 传入纯函数
        let seq_hidden: Vec<f32> = if dims.len() == 3 {
            // [batch, seq_len, hidden] → 取 batch 0 的 [seq_len, hidden]
            let batch0 = embeddings
                .get(0)
                .map_err(|e| VecboostError::InferenceError(format!("get batch 0: {}", e)))?;
            // cast to f32 if needed
            let cast = if batch0.dtype() == DType::F32 {
                batch0
            } else {
                batch0
                    .to_dtype(DType::F32)
                    .map_err(|e| VecboostError::InferenceError(format!("cast to f32: {}", e)))?
            };
            cast.to_vec2::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .into_iter()
                .flatten()
                .collect()
        } else if dims.len() == 2 {
            // [seq_len, hidden] 直接展平
            let cast = if embeddings.dtype() == DType::F32 {
                embeddings.clone()
            } else {
                embeddings
                    .to_dtype(DType::F32)
                    .map_err(|e| VecboostError::InferenceError(format!("cast to f32: {}", e)))?
            };
            cast.to_vec2::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .into_iter()
                .flatten()
                .collect()
        } else if dims.len() == 1 {
            // 1D: 模型已输出单向量(某些特殊架构),直接返回
            let mut vec = embeddings
                .to_vec1::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            l2_normalize_in_place(&mut vec);
            return Ok(vec);
        } else {
            return Err(VecboostError::InferenceError(format!(
                "Unsupported embedding dimensions: {} (shape: {:?})",
                dims.len(),
                embeddings.shape()
            )));
        };

        let vec = match self.pooling_mode {
            crate::config::model::PoolingMode::Cls => {
                pool_cls(&seq_hidden, &mask_for_pooling, hidden_dim)
            }
            crate::config::model::PoolingMode::Mean => {
                pool_mean(&seq_hidden, &mask_for_pooling, hidden_dim)
            }
            crate::config::model::PoolingMode::Max => {
                pool_max(&seq_hidden, &mask_for_pooling, hidden_dim)
            }
            crate::config::model::PoolingMode::Auto => {
                unreachable!("Auto pooling should have been resolved at construction time")
            }
        };

        log::debug!("Final embedding dim: {}", vec.len());
        // 引擎出口契约：L2 归一化（与批量路径、量化引擎一致，见 trait 文档）
        let mut vec = vec;
        l2_normalize_in_place(&mut vec);
        Ok(vec)
    }

    fn forward_pass_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, VecboostError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        log::debug!(
            "Processing batch of {} texts using optimized batch processing",
            texts.len()
        );

        // 批量编码所有文本——同步直通编码（spawn_blocking 上下文，无异步缓存）
        let encodings: Vec<Encoding> = {
            let _timer = StageTimer::new(&self.stage_stats, Stage::Tokenize);
            let mut encodings = Vec::with_capacity(texts.len());
            for &text in texts {
                let encoding = self
                    .tokenizer
                    .encode(text, true)
                    .map_err(|e| VecboostError::TokenizationError(e.to_string()))?;
                encodings.push(encoding);
            }
            encodings
        };

        let max_seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.tokenizer.max_length());

        if max_seq_len == 0 {
            return Err(VecboostError::InferenceError(
                "Empty token sequence: all inputs produced zero tokens after truncation"
                    .to_string(),
            ));
        }

        let batch_size = texts.len();
        let (token_ids, token_type_ids, attention_mask_tensor, pooling_mask) =
            self.build_batch_tensors(&encodings, batch_size, max_seq_len)?;

        let embeddings = {
            let _timer = StageTimer::new(&self.stage_stats, Stage::Inference);
            match (&self.model, &self.model_architecture) {
                (ModelWrapper::Bert(bert_model), ModelArchitecture::Bert) => bert_model
                    .forward(&token_ids, &token_type_ids, Some(&attention_mask_tensor))
                    .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                (ModelWrapper::XlmRoberta(xlm_model), ModelArchitecture::XlmRoberta) => xlm_model
                    .forward(
                        &token_ids,
                        &attention_mask_tensor,
                        &token_type_ids,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                _ => {
                    return Err(VecboostError::InferenceError(format!(
                        "Model architecture mismatch for batch processing: {:?}",
                        self.model_architecture
                    )));
                }
            }
        };

        // pooling 计时守卫驻留至函数返回，覆盖正常与 fallback 出口。
        let _pool_timer = StageTimer::new(&self.stage_stats, Stage::Pooling);
        let mut results =
            self.pool_batch_embeddings(&embeddings, &pooling_mask, batch_size, max_seq_len)?;

        log::debug!(
            "Batch processing completed, {} embeddings generated",
            results.len()
        );
        // 引擎出口 L2 归一化（平台契约 normalize: Some(true)）
        for v in results.iter_mut() {
            l2_normalize_in_place(v);
        }
        Ok(results)
    }

    /// 批量输入张量化：三组 [batch, max_seq_len] 平面缓冲（ids/mask/
    /// type_ids）按行填充后成张量；mask 克隆一份留给 Mean/Max 池化构建
    /// 逐样本 mask。
    fn build_batch_tensors(
        &self,
        encodings: &[Encoding],
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Vec<i64>), VecboostError> {
        let mut batch_ids = vec![0i64; batch_size * max_seq_len];
        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            for (seq_idx, &id) in ids.iter().enumerate().take(max_seq_len) {
                batch_ids[batch_idx * max_seq_len + seq_idx] = id as i64;
            }
        }

        // 直接动态分配 token_ids 张量
        // （Candle Tensor 不可变，张量池取出后仍需重建，无实际收益）
        let token_ids = Tensor::new(batch_ids, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape(&[batch_size, max_seq_len])
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        let mut batch_mask = vec![0i64; batch_size * max_seq_len];
        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let mask = encoding.get_attention_mask();
            for (seq_idx, &m) in mask.iter().enumerate().take(max_seq_len) {
                batch_mask[batch_idx * max_seq_len + seq_idx] = m as i64;
            }
        }

        // 动态分配 attention_mask 张量（Candle Tensor 不可变，池化无收益）
        let pooling_mask = batch_mask.clone();
        let attention_mask_tensor = Tensor::new(batch_mask, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape(&[batch_size, max_seq_len])
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        // 修复：构建批量 token_type_ids 并将 attention_mask
        // 以 Some(...) 正确传入（旧实现把 mask 传到 type_ids 槽位、mask 传 None，
        // 导致批内 padding 无隔离，短序列向量被长序列污染，cos 仅 ~0.59）。
        let mut batch_type_ids = vec![0i64; batch_size * max_seq_len];
        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let type_ids = encoding.get_type_ids();
            for (seq_idx, &tid) in type_ids.iter().enumerate().take(max_seq_len) {
                batch_type_ids[batch_idx * max_seq_len + seq_idx] = tid as i64;
            }
        }
        let token_type_ids = Tensor::new(batch_type_ids, &self.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape(&[batch_size, max_seq_len])
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        Ok((
            token_ids,
            token_type_ids,
            attention_mask_tensor,
            pooling_mask,
        ))
    }

    /// 批量池化：CLS 走 narrow 单次提取全部 token（一次 kernel 传输替代
    /// 逐样本 N 次 GPU kernel + DMA，见鲲鹏文档「理论计算极限」）；
    /// Mean/Max 逐样本取 [seq_len, hidden] 并携带各自 attention mask 传入
    /// pool_* 纯函数，与单条路径（forward_pass）行为一致；2D 输出
    /// （架构内建池化）逐行取出。归一化由调用方统一执行。
    fn pool_batch_embeddings(
        &self,
        embeddings: &Tensor,
        pooling_mask: &[i64],
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<Vec<Vec<f32>>, VecboostError> {
        let hidden_size = self.hidden_size;

        // 按引擎配置的 pooling_mode 汇聚
        if embeddings.dims().len() == 3 {
            return match &self.pooling_mode {
                crate::config::model::PoolingMode::Cls => Self::pool_batch_cls(embeddings),
                mode @ (crate::config::model::PoolingMode::Mean
                | crate::config::model::PoolingMode::Max) => self.pool_batch_mean_max(
                    embeddings,
                    mode,
                    pooling_mask,
                    batch_size,
                    max_seq_len,
                    hidden_size,
                ),
                crate::config::model::PoolingMode::Auto => {
                    unreachable!("Auto pooling should have been resolved at construction time")
                }
            };
        }

        if embeddings.dims().len() == 2 {
            // 模型已输出 [batch, hidden]（架构内建池化），逐行取出即可
            return Self::flatten_batch_rows(embeddings);
        }

        Err(VecboostError::InferenceError(format!(
            "Unsupported batch embedding dimensions: {} (shape: {:?})",
            embeddings.dims().len(),
            embeddings.shape()
        )))
    }

    /// [batch, hidden] 输出（架构内建池化）逐行取出；f16/bf16 先 cast f32。
    fn flatten_batch_rows(embeddings: &Tensor) -> Result<Vec<Vec<f32>>, VecboostError> {
        let cast = if embeddings.dtype() == DType::F32 {
            embeddings.clone()
        } else {
            embeddings.to_dtype(DType::F32).map_err(|e| {
                VecboostError::InferenceError(format!("Failed to cast to f32: {}", e))
            })?
        };
        cast.to_vec2::<f32>().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to convert batch to vec2: {}", e))
        })
    }

    /// CLS 批量池化：narrow + squeeze + to_vec2 单次提取所有 CLS token，
    /// 替代逐样本 get(i).get(0).to_vec1() 的 N 次 GPU kernel + N 次 DMA 传输。
    /// 对应鲲鹏文档「理论计算极限」：减少全局内存访问延迟（400-600 clocks）。
    /// f16/bf16 输出先 cast 到 f32。
    fn pool_batch_cls(embeddings: &Tensor) -> Result<Vec<Vec<f32>>, VecboostError> {
        let cls_all = embeddings
            .narrow(1, 0, 1)
            .map_err(|e| VecboostError::InferenceError(format!("Failed to narrow CLS dim: {}", e)))?
            .squeeze(1)
            .map_err(|e| {
                VecboostError::InferenceError(format!("Failed to squeeze CLS dim: {}", e))
            })?;
        // 需要根据实际 dtype 转换：模型可能输出 f16/bf16，需先 cast 到 f32
        let cls_f32 = if cls_all.dtype() == DType::F32 {
            cls_all
        } else {
            cls_all.to_dtype(DType::F32).map_err(|e| {
                VecboostError::InferenceError(format!("Failed to cast CLS to f32: {}", e))
            })?
        };
        cls_f32.to_vec2::<f32>().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to convert CLS batch to vec2: {}", e))
        })
    }

    /// Mean/Max 批量池化：逐样本取 [seq_len, hidden] 并携带各自 attention
    /// mask 传入 pool_* 纯函数，与单条路径（forward_pass）行为一致。
    #[allow(clippy::too_many_arguments)]
    fn pool_batch_mean_max(
        &self,
        embeddings: &Tensor,
        mode: &crate::config::model::PoolingMode,
        pooling_mask: &[i64],
        batch_size: usize,
        max_seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<Vec<f32>>, VecboostError> {
        let mut out = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let sample = embeddings.get(i).map_err(|e| {
                VecboostError::InferenceError(format!("Failed to get batch {}: {}", i, e))
            })?;
            let cast = if sample.dtype() == DType::F32 {
                sample
            } else {
                sample.to_dtype(DType::F32).map_err(|e| {
                    VecboostError::InferenceError(format!(
                        "Failed to cast sample {} to f32: {}",
                        i, e
                    ))
                })?
            };
            let seq_hidden: Vec<f32> = cast
                .to_vec2::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .into_iter()
                .flatten()
                .collect();
            let mask: Vec<u32> = pooling_mask[i * max_seq_len..(i + 1) * max_seq_len]
                .iter()
                .map(|&m| m as u32)
                .collect();
            let vec = match mode {
                crate::config::model::PoolingMode::Mean => {
                    pool_mean(&seq_hidden, &mask, hidden_size)
                }
                crate::config::model::PoolingMode::Max => pool_max(&seq_hidden, &mask, hidden_size),
                _ => unreachable!("matched Mean/Max above"),
            };
            out.push(vec);
        }
        Ok(out)
    }
}

#[async_trait]
impl InferenceEngine for CandleEngine {
    // 纯同步实现——移除 block_in_place+block_on，由调用方 spawn_blocking 包装
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
        let texts_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.forward_pass_batch(&texts_refs)
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

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        self.try_fallback_to_cpu_impl(config).await
    }

    fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        self.tokenizer.count_tokens(text)
    }

    fn take_stage_snapshot(&self) -> Option<StageSnapshot> {
        Some(self.take_stage_snapshot())
    }
}

impl CandleEngine {
    /// 检查是否启用了 INT8 量化
    pub fn uses_quantization(&self) -> bool {
        self.use_quantization
    }

    /// 获取内存使用估算（基于精度和量化）
    pub fn estimate_memory_usage(&self, batch_size: usize, sequence_length: usize) -> u64 {
        let hidden_size = self.get_hidden_size();
        let num_params = self.estimate_parameter_count();

        // 基础模型大小（MB）
        let model_size_mb = match (&self.precision, self.use_quantization) {
            (Precision::Fp32, _) => num_params * bytes_per_param(Precision::Fp32) / (1024 * 1024),
            (Precision::Fp16, _) => num_params * bytes_per_param(Precision::Fp16) / (1024 * 1024),
            (Precision::Bf16, _) => num_params * bytes_per_param(Precision::Bf16) / (1024 * 1024),
            (Precision::Int8, true) => {
                num_params * bytes_per_param(Precision::Int8) / (1024 * 1024)
            }
            (Precision::Int8, false) => {
                num_params * bytes_per_param(Precision::Fp32) / (1024 * 1024)
            }
        };

        // 激活值大小（MB）
        let activation_size_mb = batch_size * sequence_length * hidden_size * 4 / (1024 * 1024);

        model_size_mb + activation_size_mb as u64
    }

    /// 估算模型参数数量
    fn estimate_parameter_count(&self) -> u64 {
        self.parameter_count
    }

    /// 获取隐藏层大小
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }

    async fn try_fallback_to_cpu_impl(
        &mut self,
        config: &ModelConfig,
    ) -> Result<(), VecboostError> {
        if self.fallback_triggered {
            return Ok(());
        }

        log::info!(
            "{}",
            crate::i18n::tr("log-engine-attempting-fallback-from-gpu")
        );

        self.memory_monitor = None;
        self.device = Device::Cpu;
        self.device_type = DeviceType::Cpu;
        self.fallback_triggered = true;

        let model_path = &config.model_path;
        let is_local_path = model_path.exists() && model_path.is_dir();
        let (config_filename, tokenizer_filename, weights_filename) =
            resolve_model_files(config, model_path, is_local_path)?;

        let config_content = std::fs::read_to_string(config_filename)?;
        let model_config_json: ModelConfigJson = serde_json::from_str(&config_content)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        let fallback_architecture = model_config_json.get_architecture();

        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-fallback-detected-model-architecture",
                crate::i18n::tr_args(&[("a1", &format!("{:?}", fallback_architecture))])
            )
        );

        let (bert_config, xlm_config) = match &fallback_architecture {
            ModelArchitecture::Bert => {
                let bert_config: BertConfig = serde_json::from_str(&config_content)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                (Some(bert_config), None)
            }
            ModelArchitecture::XlmRoberta => {
                let xlm_config: XlmRobertaConfig = serde_json::from_str(&config_content)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                (None, Some(xlm_config))
            }
        };

        let hf_tokenizer = HfTokenizer::from_file(tokenizer_filename.to_string_lossy().as_ref())
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        let max_position_embeddings = match (&fallback_architecture, &bert_config, &xlm_config) {
            (ModelArchitecture::Bert, Some(bert), _) => bert.max_position_embeddings,
            (ModelArchitecture::XlmRoberta, _, Some(xlm)) => xlm.max_position_embeddings,
            _ => {
                return Err(VecboostError::ModelLoadError(format!(
                    "Invalid configuration for architecture: {:?}",
                    fallback_architecture
                )));
            }
        };

        self.tokenizer = SyncTokenizer::new(hf_tokenizer, max_position_embeddings);

        // 保留原始精度对应的 dtype（CPU 上 FP16/BF16 不可用，回退到 FP32）
        let compute_dtype = match self.precision {
            Precision::Fp32 | Precision::Int8 => DType::F32,
            Precision::Fp16 | Precision::Bf16 => {
                log::warn!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-engine-fallback-cpu-supported-cpu",
                        crate::i18n::tr_args(&[("a1", &format!("{:?}", self.precision))])
                    )
                );
                DType::F32
            }
        };
        // SAFETY: Same as the initialization path — memory-maps safetensors weight file.
        // The file path comes from a validated model config and the mmap is read-only.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], compute_dtype, &self.device)
        }
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

        self.model = match &fallback_architecture {
            ModelArchitecture::Bert => {
                let config = bert_config.ok_or_else(|| {
                    VecboostError::ModelLoadError(crate::i18n::tr("engine-bert-config-required"))
                })?;
                let bert_model = BertModel::load(vb, &config)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                ModelWrapper::Bert(bert_model)
            }
            ModelArchitecture::XlmRoberta => {
                let config = xlm_config.ok_or_else(|| {
                    VecboostError::ModelLoadError(crate::i18n::tr("engine-xlm-config-required"))
                })?;
                let xlm_model = XLMRobertaModel::new(&config, vb)
                    .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
                ModelWrapper::XlmRoberta(xlm_model)
            }
        };

        self.model_architecture = fallback_architecture;
        // 回退时禁用量化
        self.use_quantization = false;

        log::info!(
            "{}",
            crate::i18n::tr("log-engine-successfully-fell-back-cpu")
        );
        Ok(())
    }
}

// ============================================================================
// Pooling 数学纯函数(与 candle Tensor 解耦,可独立单测)
// ============================================================================

/// 加载期解析产物：架构判定、架构专属 config、规模元数据与位置上限。
struct ParsedModelConfig {
    architecture: ModelArchitecture,
    bert_config: Option<BertConfig>,
    xlm_config: Option<XlmRobertaConfig>,
    hidden_size: usize,
    vocab_size: usize,
    parameter_count: u64,
    max_position_embeddings: usize,
}

/// 设备落位：CUDA/Metal 可用即用之；AMD/OpenCL 显性告知不原生支持并
/// 回退 CPU（ROCm 级 Candle 构建或 ONNX Runtime 才有 AMD 路径）。
///
/// CUDA 的**静默降级已禁止**（2026-10-18）：用户显式配 `device = "cuda"` 而构建缺
/// `candle-cuda`（`cuda_is_available()` 恒假）时，原先会打一句 INFO 后落 CPU，
/// 等于把 CPU 数字记到 GPU 档名下。现改为启动期 `ConfigError` 显性拒绝，
/// 与 `startup-engine-candle-not-compiled`（engine_type 未编译）同口径。
fn resolve_compute_device(device_type: &DeviceType) -> Result<Device, VecboostError> {
    if *device_type == DeviceType::Cuda {
        if !candle_core::utils::cuda_is_available() {
            return Err(VecboostError::ConfigError(crate::i18n::tr(
                "engine-candle-cuda-unavailable",
            )));
        }
        log::info!("{}", crate::i18n::tr("log-engine-using-cuda-gpu"));
        return Device::new_cuda(0).map_err(|e| VecboostError::InferenceError(e.to_string()));
    }
    if *device_type == DeviceType::Metal && candle_core::utils::metal_is_available() {
        log::info!("{}", crate::i18n::tr("log-engine-using-metal-gpu"));
        Ok(Device::new_metal(0).map_err(|e| VecboostError::InferenceError(e.to_string()))?)
    } else if matches!(device_type, DeviceType::Amd | DeviceType::OpenCL) {
        log::warn!(
            "{}",
            crate::i18n::tr("log-engine-candle-engine-does-natively")
        );
        Ok(Device::Cpu)
    } else {
        log::info!("{}", crate::i18n::tr("log-engine-using-cpu"));
        Ok(Device::Cpu)
    }
}

/// 计算数据类型映射：FP16/BF16 仅 CUDA 生效（其余设备诚实回退 FP32）；
/// INT8 量化未实现，一律 FP32 且 use_quantization 恒 false。
fn resolve_compute_dtype(precision: &Precision, is_cuda: bool) -> (DType, bool) {
    let compute_dtype = match (&precision, is_cuda) {
        (Precision::Int8, _) => {
            log::warn!(
                "{}",
                crate::i18n::tr("log-engine-int8-quantization-yet-implemented")
            );
            DType::F32
        }
        (Precision::Fp16, true) => {
            log::info!("{}", crate::i18n::tr("log-engine-using-fp16-precision"));
            DType::F16
        }
        (Precision::Fp16, false) => {
            log::warn!("{}", crate::i18n::tr("log-engine-fp16-supported-non-cuda"));
            DType::F32
        }
        (Precision::Bf16, true) => {
            log::info!(
                "{}",
                crate::i18n::tr("log-engine-using-bf16-precision-brain")
            );
            DType::BF16
        }
        (Precision::Bf16, false) => {
            log::warn!("{}", crate::i18n::tr("log-engine-bf16-supported-non-cuda"));
            DType::F32
        }
        (Precision::Fp32, _) => {
            log::info!("{}", crate::i18n::tr("log-engine-using-fp32-precision"));
            DType::F32
        }
    };
    let use_quantization = false;
    if matches!(precision, Precision::Int8) {
        (DType::F32, use_quantization)
    } else {
        (compute_dtype, use_quantization)
    }
}

/// 模型三文件（config/tokenizer/weights）落位：本地路径做路径遍历攻击
/// 检测（`..`/`~` 直接拒绝）与权重文件嗅探；远程 repo 走 build_hf_repo
/// 的 vuln-0009 校验与下载。
fn resolve_model_files(
    config: &ModelConfig,
    model_path: &std::path::Path,
    is_local_path: bool,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), VecboostError> {
    // 安全增强：本地路径进行路径遍历攻击检测，直接拒绝
    if is_local_path {
        let model_path_str = model_path.to_string_lossy();
        if model_path_str.contains("..") || model_path_str.contains('~') {
            return Err(VecboostError::ModelLoadError(format!(
                "Rejected model path due to potential path traversal attack: {:?}. \
                     Path must not contain '..' or '~'.",
                model_path
            )));
        }

        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-loading-model-from-local",
                crate::i18n::tr_args(&[("a1", &format!("{:?}", model_path))])
            )
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
            return Err(VecboostError::ModelLoadError(
                "No model weights file found (model.safetensors or pytorch_model.bin)".to_string(),
            ));
        };

        return Ok((config_path, tokenizer_path, weights_filename));
    }

    // vuln-0009 修复:repo_id 格式校验由 build_hf_repo 统一执行，防止恶意配置注入
    let repo_id = model_path.to_string_lossy().into_owned();

    // vuln-0009 加固:远程下载时若未设置 model_sha256,记录警告
    // .bin(pickle)格式有代码执行风险,model_sha256 是完整性校验的最后防线
    if config.model_sha256.is_none() {
        log::warn!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-loading-model-from-remote",
                crate::i18n::tr_args(&[("a1", &(repo_id).to_string())])
            )
        );
    }

    log::info!(
        "{}",
        crate::i18n::tr_with_args(
            "log-engine-downloading-loading-model-from",
            crate::i18n::tr_args(&[("a1", &format!("{:?}", model_path))])
        )
    );
    let repo = build_hf_repo(&repo_id)?;

    let config_filename = repo
        .download_file()
        .filename("config.json")
        .send()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    let tokenizer_filename = repo
        .download_file()
        .filename("tokenizer.json")
        .send()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    let weights_filename = repo
        .download_file()
        .filename("model.safetensors")
        .send()
        .or_else(|_| {
            log::warn!(
                "{}",
                crate::i18n::tr("log-engine-model-safetensors-unavailable-falling")
            );
            repo.download_file().filename("pytorch_model.bin").send()
        })
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

    Ok((config_filename, tokenizer_filename, weights_filename))
}

/// config.json 解析：架构判定 → 架构专属 config 反序列化 → 规模元数据
/// （hidden/vocab/参数估算，在 config 被模型构造消费前提取）与位置上限。
fn parse_model_config(config_content: &str) -> Result<ParsedModelConfig, VecboostError> {
    let model_config_json: ModelConfigJson = serde_json::from_str(config_content)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    let architecture = model_config_json.get_architecture();

    let (bert_config, xlm_config) = match &architecture {
        ModelArchitecture::Bert => {
            let bert_config: BertConfig = serde_json::from_str(config_content)
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
            (Some(bert_config), None)
        }
        ModelArchitecture::XlmRoberta => {
            let xlm_config: XlmRobertaConfig = serde_json::from_str(config_content)
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
            (None, Some(xlm_config))
        }
    };

    // 从 config 提取 hidden_size、vocab_size 和估算参数数量
    let (hidden_size, vocab_size, parameter_count) = match (&bert_config, &xlm_config) {
        (Some(bc), _) => {
            let params = estimate_bert_params(
                bc.vocab_size,
                bc.hidden_size,
                bc.num_hidden_layers,
                bc.intermediate_size,
                bc.num_attention_heads,
            );
            (bc.hidden_size, bc.vocab_size, params)
        }
        (_, Some(xc)) => {
            let params = estimate_bert_params(
                xc.vocab_size,
                xc.hidden_size,
                xc.num_hidden_layers,
                xc.intermediate_size,
                xc.num_attention_heads,
            );
            (xc.hidden_size, xc.vocab_size, params)
        }
        _ => (768, 30522, 110_000_000), // fallback: BERT-base vocab
    };

    let max_position_embeddings = match (&architecture, &bert_config, &xlm_config) {
        (ModelArchitecture::Bert, Some(bert), _) => bert.max_position_embeddings,
        (ModelArchitecture::XlmRoberta, _, Some(xlm)) => xlm.max_position_embeddings,
        _ => {
            return Err(VecboostError::ModelLoadError(format!(
                "Invalid configuration for architecture: {:?}",
                architecture
            )));
        }
    };

    Ok(ParsedModelConfig {
        architecture,
        bert_config,
        xlm_config,
        hidden_size,
        vocab_size,
        parameter_count,
        max_position_embeddings,
    })
}

/// 完整性校验 + 损坏自动恢复 + 恢复后复检 + 可选 SHA256 终验；任何
/// 一环失败即显性 ModelLoadError/ModelFileCorrupted，禁止带病加载。
fn ensure_model_integrity(
    config: &ModelConfig,
    model_path: &std::path::Path,
    is_local_path: bool,
    filenames: &[std::path::PathBuf; 3],
) -> Result<(), VecboostError> {
    let [config_filename, tokenizer_filename, weights_filename] = filenames;
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

    log::info!(
        "{}",
        crate::i18n::tr("log-engine-checking-model-file-integrity")
    );
    let integrity_report = check_model_integrity(&config.name, files_to_check, Some(min_sizes))
        .map_err(|e| {
            VecboostError::ModelIntegrityError(format!("Integrity check failed: {}", e))
        })?;

    if !integrity_report.overall_valid {
        log::error!(
            "{}",
            crate::i18n::tr("log-engine-model-file-integrity-check-2")
        );
        for check in &integrity_report.files_checked {
            if !check.is_valid {
                log::error!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-engine-file-error",
                        crate::i18n::tr_args(&[
                            ("a1", &(check.file_path).to_string()),
                            (
                                "a2",
                                (check.error_message.as_deref().unwrap_or("Unknown error"))
                            )
                        ])
                    )
                );
            }
        }

        log::info!(
            "{}",
            crate::i18n::tr("log-engine-attempting-automatic-recovery-corrupted")
        );
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
            .map_err(|e| VecboostError::ModelIntegrityError(format!("Recovery failed: {}", e)))?;

        if recovery_result.success {
            log::info!(
                "{}",
                crate::i18n::tr("log-engine-successfully-recovered-all-corrupted")
            );
            log::info!(
                "{}",
                crate::i18n::tr("log-engine-re-running-integrity-check")
            );

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
                        VecboostError::ModelIntegrityError(format!(
                            "Post-recovery integrity check failed: {}",
                            e
                        ))
                    },
                )?;

            if !recovery_integrity_report.overall_valid {
                return Err(VecboostError::ModelFileCorrupted(format!(
                    "Model files still corrupted after recovery. Corrupted files: {:?}",
                    recovery_integrity_report.corrupted_files
                )));
            }

            log::info!(
                "{}",
                crate::i18n::tr("log-engine-post-recovery-integrity-check")
            );
        } else {
            return Err(VecboostError::ModelFileCorrupted(format!(
                "Failed to recover corrupted files after {} attempts. Corrupted files: {:?}",
                recovery_result.attempts, recovery_result.failed_files
            )));
        }
    }

    log::info!(
        "{}",
        crate::i18n::tr("log-engine-model-file-integrity-check")
    );

    verify_model_sha256(config, weights_filename)?;

    Ok(())
}

/// 配置声明 model_sha256 时对权重文件做 SHA256 终验（完整性校验链
/// 最后一环，防 tampering/supply-chain）。不匹配即 ModelFileCorrupted。
fn verify_model_sha256(
    config: &ModelConfig,
    weights_filename: &std::path::PathBuf,
) -> Result<(), VecboostError> {
    let Some(ref expected_hash) = config.model_sha256 else {
        return Ok(());
    };
    log::info!(
        "{}",
        crate::i18n::tr("log-engine-verifying-model-file-sha256")
    );
    let is_valid = verify_sha256(weights_filename, expected_hash)
        .map_err(|e| VecboostError::ModelLoadError(format!("Failed to verify SHA256: {}", e)))?;
    if !is_valid {
        return Err(VecboostError::ModelFileCorrupted(format!(
            "Model file SHA256 verification failed. Expected: {}, File: {:?}",
            expected_hash, weights_filename
        )));
    }
    log::info!(
        "{}",
        crate::i18n::tr("log-engine-model-file-sha256-verification")
    );
    Ok(())
}

/// 权重 → VarBuilder：PyTorch .bin 走 VarMap 全量加载（大文件提示转换
/// safetensors），safetensors 走 mmap 零拷贝路径。
fn load_weights_varbuilder<'a>(
    is_pytorch: bool,
    weights_filename: &'a std::path::PathBuf,
    // VarBuilder<'a> 的保活锚：pytorch 分支的 from_varmap 借用调用方持有的
    // VarMap（局部 VarMap 会在本函数结束时 drop，无法满足 'a）
    varmap: &'a mut VarMap,
    dtype: DType,
    device: &Device,
    config: &ModelConfig,
) -> Result<VarBuilder<'a>, VecboostError> {
    if is_pytorch {
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-engine-loading-pytorch-model-weights",
                crate::i18n::tr_args(&[("a1", &format!("{:?}", weights_filename))])
            )
        );

        let file_size = std::fs::metadata(weights_filename)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
            .len();

        if file_size > PYTORCH_LARGE_FILE_WARN_BYTES {
            log::warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-engine-pytorch-model-file-large",
                    crate::i18n::tr_args(&[(
                        "a1",
                        &(file_size as f64 / 1024.0 / 1024.0 / 1024.0).to_string()
                    )])
                )
            );
            log::info!(
                "{}",
                crate::i18n::tr("log-engine-consider-converting-safetensors-format")
            );
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-engine-python-c-from-transformers",
                    crate::i18n::tr_args(&[("a1", (config.model_path.to_string_lossy()).as_ref())])
                )
            );
            log::info!("{}", crate::i18n::tr("log-engine-then-use-model-converted"));
        }

        match varmap.load(weights_filename) {
            Ok(_) => {
                log::info!(
                    "{}",
                    crate::i18n::tr("log-engine-pytorch-weights-loaded-successfully")
                );
            }
            Err(e) => {
                log::error!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-engine-failed-load-pytorch-weights",
                        crate::i18n::tr_args(&[("a1", &(e).to_string())])
                    )
                );
                log::error!("{}", crate::i18n::tr("log-engine-this-often-due-large"));
                log::error!(
                    "{}",
                    crate::i18n::tr("log-engine-please-convert-model-safetensors")
                );
                log::error!("{}", crate::i18n::tr("log-engine-pip-install-optimum"));
                log::error!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-engine-optimum-cli-export-onnx",
                        crate::i18n::tr_args(&[(
                            "a1",
                            (config.model_path.to_string_lossy()).as_ref()
                        )])
                    )
                );
                return Err(VecboostError::ModelLoadError(format!(
                    "Failed to load PyTorch weights: {}. Please convert the model to safetensors format.",
                    e
                )));
            }
        }
        return Ok(VarBuilder::from_varmap(varmap, dtype, device));
    }

    log::info!(
        "{}",
        crate::i18n::tr_with_args(
            "log-engine-loading-safetensors-model-weights",
            crate::i18n::tr_args(&[("a1", &format!("{:?}", weights_filename))])
        )
    );
    // SAFETY: `VarBuilder::from_mmaped_safetensors` memory-maps model weight files
    // from disk. This is safe because:
    // 1. The file path is validated to exist before this call.
    // 2. The mmap is read-only and managed by candle's internal lifetime tracking.
    // 3. The OS handles page faults; no manual pointer manipulation is needed.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, device) };
    let vb = vb.map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    log::info!(
        "{}",
        crate::i18n::tr("log-engine-loaded-safetensors-model-weights")
    );
    Ok(vb)
}

/// CLS pooling:取序列第一个 token 的 hidden state。
///
/// `hidden` 为 `[seq_len × hidden_dim]` 行优先展平向量。
/// 引擎出口 L2 归一化（平台契约 normalize: Some(true) 要求所有 embed
/// 输出为单位向量）。近零向量（退化输出）保持原样，交由上游判空。
fn l2_normalize_in_place(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

pub(crate) fn pool_cls(hidden: &[f32], _mask: &[u32], hidden_dim: usize) -> Vec<f32> {
    hidden[..hidden_dim].to_vec()
}

/// Mean pooling:以 attention_mask 为权重的加权和 ÷ mask 元素总和。
///
/// `hidden`: `[seq_len × hidden_dim]` 行优先展平。
/// `mask`: 长度 `seq_len`,值为 0/1。仅 mask=1 的位置参与累加。
pub(crate) fn pool_mean(hidden: &[f32], mask: &[u32], hidden_dim: usize) -> Vec<f32> {
    let seq_len = mask.len();
    let mut result = vec![0.0_f32; hidden_dim];
    let mut mask_sum = 0.0_f32;
    for (t, &m) in mask.iter().enumerate().take(seq_len) {
        if m == 0 {
            continue;
        }
        mask_sum += 1.0;
        let offset = t * hidden_dim;
        for (d, &h) in hidden[offset..offset + hidden_dim].iter().enumerate() {
            result[d] += h;
        }
    }
    if mask_sum > 0.0 {
        for v in &mut result {
            *v /= mask_sum;
        }
    }
    result
}

/// Max pooling:mask 内逐维取最大。
///
/// `hidden`: `[seq_len × hidden_dim]` 行优先展平。
/// `mask`: 长度 `seq_len`。mask=0 的位置不参与 max 比较。
pub(crate) fn pool_max(hidden: &[f32], mask: &[u32], hidden_dim: usize) -> Vec<f32> {
    let seq_len = mask.len();
    let mut result = vec![f32::NEG_INFINITY; hidden_dim];
    let mut any_active = false;
    for (t, &m) in mask.iter().enumerate().take(seq_len) {
        if m == 0 {
            continue;
        }
        any_active = true;
        let offset = t * hidden_dim;
        for (d, &h) in hidden[offset..offset + hidden_dim].iter().enumerate() {
            if h > result[d] {
                result[d] = h;
            }
        }
    }
    if !any_active {
        result.fill(0.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{EngineType, ModelConfig};
    use std::path::PathBuf;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-candle".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/nonexistent/model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(768),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        }
    }

    /// 验证 `tokio::task::block_in_place(|| Handle::current().block_on(...))` 模式
    /// 在 multi-thread runtime 下不 panic。
    ///
    /// 此前 `embed`/`embed_batch` 使用 `futures::executor::block_on`,在 Tokio 异步上下文中
    /// 会阻塞当前 worker 线程并可能死锁(若 future 需要 Tokio 资源)。
    /// 改用 `block_in_place` 后,Tokio 会将其他任务调度到其他 worker,避免死锁。
    ///
    /// 注:CandleEngine 构造需要真实模型文件,无法在单元测试中实例化;
    /// 此测试验证 block_in_place 调用模式本身的正确性——这是 修复的核心。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_block_in_place_pattern_completes_without_panic() {
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // 模拟 forward_pass 内部的异步操作
                tokio::task::yield_now().await;
                42
            })
        });
        assert_eq!(
            result, 42,
            "block_in_place pattern must complete without panic in multi-thread runtime"
        );
    }

    /// 验证 block_in_place 内部的 block_on 可以正确 await Tokio 异步原语
    /// (如 tokio::sync::RwLock)。这模拟了 forward_pass_batch 中 pool.write().await 的场景。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_block_in_place_with_tokio_rwlock_does_not_deadlock() {
        let lock = std::sync::Arc::new(tokio::sync::RwLock::new(0i32));
        let lock_clone = std::sync::Arc::clone(&lock);

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut guard = lock_clone.write().await;
                *guard = 42;
                *guard
            })
        });

        assert_eq!(
            result, 42,
            "block_in_place must allow Tokio RwLock acquisition"
        );
        // 验证锁已释放
        let guard = lock.try_read();
        assert!(
            guard.is_ok(),
            "RwLock must be released after block_on completes"
        );
    }

    /// 验证 `ModelConfigJson::get_architecture` 从 architectures 字段识别 XLM-RoBERTa
    #[test]
    fn test_get_architecture_xlm_roberta_from_architectures() {
        let config = ModelConfigJson {
            architectures: Some(vec!["XLMRobertaForMaskedLM".to_string()]),
            model_type: None,
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 architectures 字段中的小写 "xlm_roberta" 也能被识别
    #[test]
    fn test_get_architecture_xlm_roberta_from_lowercase_arch() {
        let config = ModelConfigJson {
            architectures: Some(vec!["xlm_roberta".to_string()]),
            model_type: None,
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 model_type 字段为 "xlm-roberta" 时识别为 XLM-RoBERTa
    #[test]
    fn test_get_architecture_xlm_roberta_from_model_type_hyphen() {
        let config = ModelConfigJson {
            architectures: None,
            model_type: Some("xlm-roberta".to_string()),
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 model_type 字段为 "xlm_roberta" 时识别为 XLM-RoBERTa
    #[test]
    fn test_get_architecture_xlm_roberta_from_model_type_underscore() {
        let config = ModelConfigJson {
            architectures: None,
            model_type: Some("xlm_roberta".to_string()),
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证空字段时默认返回 Bert
    #[test]
    fn test_get_architecture_bert_default_when_all_none() {
        let config = ModelConfigJson {
            architectures: None,
            model_type: None,
        };
        assert!(matches!(config.get_architecture(), ModelArchitecture::Bert));
    }

    /// 验证 architectures 仅含 Bert 类项时返回 Bert
    #[test]
    fn test_get_architecture_bert_when_no_xlm_in_architectures() {
        let config = ModelConfigJson {
            architectures: Some(vec!["BertForMaskedLM".to_string()]),
            model_type: Some("bert".to_string()),
        };
        assert!(matches!(config.get_architecture(), ModelArchitecture::Bert));
    }

    /// 验证 architectures 字段优先级高于 model_type
    #[test]
    fn test_get_architecture_architectures_takes_precedence_over_model_type() {
        let config = ModelConfigJson {
            architectures: Some(vec!["XLMRobertaModel".to_string()]),
            model_type: Some("bert".to_string()),
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 ModelArchitecture 的 Clone 行为
    #[test]
    fn test_model_architecture_clone_preserves_variant() {
        let bert = ModelArchitecture::Bert;
        let bert_clone = bert.clone();
        assert!(matches!(bert_clone, ModelArchitecture::Bert));

        let xlm = ModelArchitecture::XlmRoberta;
        let xlm_clone = xlm.clone();
        assert!(matches!(xlm_clone, ModelArchitecture::XlmRoberta));
    }

    /// 验证 CandleEngine::new 在空目录上返回 ModelLoadError(不依赖网络)
    #[test]
    fn test_candle_engine_new_returns_error_for_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::new(&config, Precision::Fp32);
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
    fn test_candle_engine_with_device_empty_dir_cpu() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("No model weights file found"),
                "Expected 'No model weights file found', got: {}",
                msg
            );
        }
    }

    /// 验证 FP16 精度在空目录上仍返回错误(覆盖 dtype 分支)
    #[test]
    fn test_candle_engine_with_device_fp16_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp16, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 INT8 精度在空目录上仍返回错误(覆盖 INT8 量化分支)
    #[test]
    fn test_candle_engine_with_device_int8_empty_dir() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Int8, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 AMD 设备类型会回退到 CPU 并在空目录上返回 ModelLoadError
    #[test]
    fn test_candle_engine_with_device_amd_falls_back_to_cpu() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Amd);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError, got {:?}",
                e
            );
        }
    }

    /// 验证 OpenCL 设备类型同样回退到 CPU 并在空目录上返回错误
    #[test]
    fn test_candle_engine_with_device_opencl_falls_back_to_cpu() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::OpenCL);
        assert!(result.is_err());
    }

    /// 验证存在 model.safetensors 但缺失 config.json 时返回错误
    #[test]
    fn test_candle_engine_with_safetensors_but_no_config() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.safetensors"), b"fake")
            .expect("Failed to write fake safetensors");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = e.to_string();
            assert!(
                msg.contains("config.json")
                    || msg.contains("No such file")
                    || matches!(
                        e,
                        VecboostError::IoError(_) | VecboostError::ModelLoadError(_)
                    ),
                "Expected IO/ModelLoadError related to config.json, got: {}",
                msg
            );
        }
    }

    /// 验证存在 pytorch_model.bin 但缺失 config.json 时返回错误
    #[test]
    fn test_candle_engine_with_pytorch_bin_but_no_config() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("pytorch_model.bin"), b"fake")
            .expect("Failed to write fake pytorch_model.bin");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
    }

    /// 验证 `Precision` 的 Display 实现
    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::Fp32.to_string(), "fp32");
        assert_eq!(Precision::Fp16.to_string(), "fp16");
        assert_eq!(Precision::Bf16.to_string(), "bf16");
        assert_eq!(Precision::Int8.to_string(), "int8");
    }

    /// 验证 `DeviceType` 的序列化形式(serde rename)
    #[test]
    fn test_device_type_serialization() {
        let cpu_json = serde_json::to_string(&DeviceType::Cpu).expect("serialize Cpu");
        assert_eq!(cpu_json, "\"cpu\"");

        let amd_json = serde_json::to_string(&DeviceType::Amd).expect("serialize Amd");
        assert_eq!(amd_json, "\"amd\"");

        let opencl_json = serde_json::to_string(&DeviceType::OpenCL).expect("serialize OpenCL");
        assert_eq!(opencl_json, "\"opencl\"");
    }

    /// 验证 architectures 为空 Vec 时回退到 model_type 判断
    #[test]
    fn test_get_architecture_empty_architectures_vec_falls_to_model_type() {
        let config = ModelConfigJson {
            architectures: Some(vec![]),
            model_type: Some("xlm-roberta".to_string()),
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 architectures 列表中 XLM-RoBERTa 出现在 Bert 之后时仍能识别
    #[test]
    fn test_get_architecture_xlm_after_bert_in_list() {
        let config = ModelConfigJson {
            architectures: Some(vec![
                "BertForMaskedLM".to_string(),
                "XLMRobertaForMaskedLM".to_string(),
            ]),
            model_type: None,
        };
        assert!(matches!(
            config.get_architecture(),
            ModelArchitecture::XlmRoberta
        ));
    }

    /// 验证 architectures 和 model_type 均无 XLM 关键字时返回 Bert
    #[test]
    fn test_get_architecture_bert_with_non_xlm_model_type() {
        let config = ModelConfigJson {
            architectures: Some(vec!["BertModel".to_string()]),
            model_type: Some("roberta".to_string()),
        };
        assert!(matches!(config.get_architecture(), ModelArchitecture::Bert));
    }

    /// 验证 ModelArchitecture 的 Debug 派生输出正确
    #[test]
    fn test_model_architecture_debug_format() {
        let bert_debug = format!("{:?}", ModelArchitecture::Bert);
        assert!(bert_debug.contains("Bert"));

        let xlm_debug = format!("{:?}", ModelArchitecture::XlmRoberta);
        assert!(xlm_debug.contains("XlmRoberta"));
    }

    /// 验证路径包含 ".." 时直接拒绝加载（路径遍历攻击防护）
    #[test]
    fn test_candle_engine_path_with_dotdot_traversal_rejected() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let sub_dir = temp_dir.path().join("sub");
        std::fs::create_dir(&sub_dir).expect("Failed to create sub dir");
        let model_path = sub_dir.join("..").join("sub");

        assert!(
            model_path.exists(),
            "Path with .. must resolve to existing dir"
        );
        assert!(model_path.is_dir(), "Resolved path must be a directory");

        let mut config = test_config();
        config.model_path = model_path;

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("path traversal"),
                "Expected path traversal rejection, got: {}",
                msg
            );
        }
    }

    /// 验证 config.json 为非 JSON 文本时返回 ModelLoadError
    #[test]
    fn test_candle_engine_malformed_config_json() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.safetensors"), b"fake")
            .expect("Failed to write fake safetensors");
        std::fs::write(temp_dir.path().join("config.json"), b"not valid json {{{")
            .expect("Failed to write malformed config");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError for malformed JSON, got {:?}",
                e
            );
        }
    }

    /// 验证 config.json 为空文件时返回 ModelLoadError
    #[test]
    fn test_candle_engine_empty_config_json() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.safetensors"), b"fake")
            .expect("Failed to write fake safetensors");
        std::fs::write(temp_dir.path().join("config.json"), b"")
            .expect("Failed to write empty config");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError for empty JSON, got {:?}",
                e
            );
        }
    }

    /// 验证 config.json 为合法 JSON 但非对象(如数字)时返回 ModelLoadError
    #[test]
    fn test_candle_engine_non_object_config_json() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        std::fs::write(temp_dir.path().join("model.safetensors"), b"fake")
            .expect("Failed to write fake safetensors");
        std::fs::write(temp_dir.path().join("config.json"), b"42")
            .expect("Failed to write non-object config");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError for non-object JSON, got {:?}",
                e
            );
        }
    }

    /// 验证 model_sha256 设置但模型文件不存在时返回错误(覆盖 SHA256 设置路径)
    #[test]
    fn test_candle_engine_with_sha256_but_no_weights() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

        let mut config = test_config();
        config.model_path = temp_dir.path().to_path_buf();
        config.model_sha256 = Some("abc123".to_string());

        let result = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu);
        assert!(result.is_err());
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("No model weights file found"),
                "Expected weights error when SHA256 set but no weights, got: {}",
                msg
            );
        }
    }

    // ===== 真实模型集成测试 =====

    const REAL_MODEL_PATH: &str = "models/BAAI-bge-small-en-v1.5";

    fn real_model_config() -> ModelConfig {
        ModelConfig {
            name: "bge-small-en".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(REAL_MODEL_PATH),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        }
    }

    fn require_real_model() -> bool {
        // 权重文件(而非仅配置)存在才可加载;仓库通常只含 tokenizer/config,
        // GGUF 量化产物不满足 CandleEngine 的 safetensors 加载路径。
        let has_weights = ["model.safetensors", "pytorch_model.bin"]
            .iter()
            .any(|w| std::path::Path::new(&format!("{REAL_MODEL_PATH}/{w}")).exists());
        if !has_weights {
            eprintln!(
                "Skipping test: model weights not found at {REAL_MODEL_PATH}/(model.safetensors|pytorch_model.bin)"
            );
            return false;
        }
        let path = format!("{REAL_MODEL_PATH}/config.json");
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping test: model files not found at {REAL_MODEL_PATH}");
            return false;
        }
        true
    }

    /// 验证 CandleEngine::new 成功加载真实 BERT 模型并完成单文本推理、
    /// 同时覆盖 precision/device_type/supports_mixed_precision/is_fallback_triggered/
    /// uses_quantization/estimate_memory_usage/内存方法/embed_batch 等核心路径。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_load_and_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");

        assert_eq!(*engine.precision(), Precision::Fp32);
        assert_eq!(engine.device_type(), DeviceType::Cpu);
        assert!(!engine.supports_mixed_precision());
        assert!(!engine.is_fallback_triggered());
        assert!(!engine.uses_quantization());

        let result = engine.embed("hello world");
        assert!(result.is_ok(), "embed failed: {:?}", result.err());
        let embedding = result.unwrap();
        assert_eq!(
            embedding.len(),
            384,
            "Expected 384-dim embedding, got {}",
            embedding.len()
        );
        let non_zero = embedding.iter().filter(|v| **v != 0.0).count();
        assert!(non_zero > 0, "Embedding should have non-zero values");

        let result2 = engine.embed("hello world");
        assert!(result2.is_ok());
        let embedding2 = result2.unwrap();
        for (a, b) in embedding.iter().zip(embedding2.iter()) {
            assert!((a - b).abs() < 1e-5, "Same text should give same embedding");
        }

        let empty_result = engine.embed("");
        assert!(
            empty_result.is_err(),
            "embed empty string should return error"
        );
        assert!(matches!(
            empty_result.unwrap_err(),
            VecboostError::TokenizationError(_)
        ));

        let mem = engine.estimate_memory_usage(1, 128);
        assert!(
            mem > 0,
            "estimate_memory_usage should return positive value"
        );

        assert!(
            !engine.check_memory_pressure(90).await,
            "check_memory_pressure should be false without monitor"
        );
        assert_eq!(
            engine.get_memory_status().await,
            None,
            "get_memory_status should be None without controller"
        );
        engine.update_memory_limit(1024).await;
        engine.update_gpu_memory().await;

        let empty_batch: Vec<String> = vec![];
        let batch_result = engine.embed_batch(&empty_batch);
        assert!(batch_result.is_ok());
        assert_eq!(batch_result.unwrap().len(), 0);

        let texts: Vec<String> = vec!["hello world".to_string(), "machine learning".to_string()];
        let batch_result = engine.embed_batch(&texts);
        assert!(
            batch_result.is_ok(),
            "embed_batch failed: {:?}",
            batch_result.err()
        );
        let embeddings = batch_result.unwrap();
        assert_eq!(embeddings.len(), 2);
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(
                emb.len(),
                384,
                "Batch embedding {} should be 384-dim, got {}",
                i,
                emb.len()
            );
        }
    }

    /// 验证 try_fallback_to_cpu 在 CPU 引擎上仍能重新加载模型并继续推理
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_try_fallback_to_cpu() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let mut engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");

        assert!(!engine.is_fallback_triggered());

        let fallback_result = engine.try_fallback_to_cpu(&config).await;
        assert!(
            fallback_result.is_ok(),
            "try_fallback_to_cpu failed: {:?}",
            fallback_result.err()
        );
        assert!(engine.is_fallback_triggered());
        assert_eq!(engine.device_type(), DeviceType::Cpu);
        assert!(
            !engine.uses_quantization(),
            "use_quantization should be disabled after fallback"
        );

        let result = engine.embed("post fallback text");
        assert!(
            result.is_ok(),
            "embed after fallback failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().len(), 384);

        let second_fallback = engine.try_fallback_to_cpu(&config).await;
        assert!(second_fallback.is_ok(), "Repeated fallback should be no-op");
    }

    /// 验证 set_memory_limit_controller 后内存状态查询与回退检查路径
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_with_memory_controller() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let mut engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");

        assert_eq!(engine.get_memory_status().await, None);

        let controller = Arc::new(MemoryLimitController::new());
        engine.set_memory_limit_controller(controller);

        let status = engine.get_memory_status().await;
        assert!(status.is_some(), "get_memory_status should return Some");
        assert_eq!(status.unwrap(), MemoryLimitStatus::Ok);

        let fallback_result = engine.check_memory_limit_and_fallback(&config).await;
        assert!(
            fallback_result.is_ok(),
            "check_memory_limit_and_fallback failed: {:?}",
            fallback_result.err()
        );
        assert!(
            !fallback_result.unwrap(),
            "Should not fallback when under limit"
        );

        engine.update_memory_limit(1024 * 1024).await;
        assert_eq!(
            engine.get_memory_status().await,
            Some(MemoryLimitStatus::Ok)
        );
    }

    /// 验证 FP16 精度在 CPU 上回退到 FP32 后仍可正常推理
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_loads_fp16_cpu() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine = CandleEngine::new(&config, Precision::Fp16)
            .expect("Failed to load real model with FP16");

        assert_eq!(*engine.precision(), Precision::Fp16);
        assert!(
            !engine.supports_mixed_precision(),
            "CPU should not support mixed precision"
        );

        let result = engine.embed("fp16 precision test");
        assert!(
            result.is_ok(),
            "embed with FP16 on CPU failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().len(), 384);
    }

    /// 验证 INT8 精度在 CPU 上回退 FP32，use_quantization 始终为 false
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_loads_int8_cpu() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine = CandleEngine::new(&config, Precision::Int8)
            .expect("Failed to load real model with INT8");

        assert_eq!(*engine.precision(), Precision::Int8);
        assert!(
            !engine.uses_quantization(),
            "INT8 not implemented, use_quantization should be false"
        );

        let result = engine.embed("int8 precision test");
        assert!(
            result.is_ok(),
            "embed with INT8 on CPU failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().len(), 384);
    }

    /// 验证不同 PoolingMode 配置下引擎均能加载并推理
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_loads_with_all_pooling_modes() {
        if !require_real_model() {
            return;
        }
        use crate::config::model::PoolingMode;
        for (name, mode) in [
            ("Mean", PoolingMode::Mean),
            ("Max", PoolingMode::Max),
            ("Cls", PoolingMode::Cls),
        ] {
            let mut config = real_model_config();
            config.pooling_mode = Some(mode.clone());
            let engine = CandleEngine::new(&config, Precision::Fp32)
                .unwrap_or_else(|e| panic!("Failed to load with {} pooling: {:?}", name, e));
            let result = engine.embed("pooling mode test");
            assert!(
                result.is_ok(),
                "embed with {} pooling failed: {:?}",
                name,
                result.err()
            );
            assert_eq!(result.unwrap().len(), 384);
        }
    }

    /// 验证 with_device 显式传入 Cpu 设备时行为与 new() 一致
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_with_device_cpu_no_pool() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine = CandleEngine::with_device(&config, Precision::Fp32, DeviceType::Cpu)
            .expect("Failed to load real model via with_device");

        assert_eq!(engine.device_type(), DeviceType::Cpu);
        let result = engine.embed("with_device test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 384);
    }

    /// 验证 PyTorch 模型加载路径(覆盖 is_pytorch 分支)
    /// HuggingFace pytorch_model.bin 使用 pickle 格式,candle VarMap::load 不兼容,
    /// 返回 ModelLoadError 提示转换为 safetensors 格式
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[cfg(unix)]
    async fn test_real_model_pytorch_bin_loading() {
        if !require_real_model() {
            return;
        }
        // 该测试依赖源目录存在 pytorch_model.bin（HF pickle 权重）——
        // 仅 safetensors 的 checkout 无此文件，跳过而非误报
        if !std::path::Path::new(&format!("{REAL_MODEL_PATH}/pytorch_model.bin")).exists() {
            eprintln!("Skipping test: pytorch_model.bin not present in {REAL_MODEL_PATH}");
            return;
        }
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let real_path = std::path::Path::new(REAL_MODEL_PATH)
            .canonicalize()
            .expect("Failed to canonicalize real model path");
        for file in &["config.json", "tokenizer.json", "pytorch_model.bin"] {
            let src = real_path.join(file);
            let dst = temp_dir.path().join(file);
            std::os::unix::fs::symlink(&src, &dst)
                .unwrap_or_else(|e| panic!("Failed to symlink {}: {}", file, e));
        }
        let mut config = real_model_config();
        config.model_path = temp_dir.path().to_path_buf();
        let result = CandleEngine::new(&config, Precision::Fp32);
        assert!(
            result.is_err(),
            "PyTorch loading should fail with candle VarMap"
        );
        if let Err(VecboostError::ModelLoadError(msg)) = result {
            assert!(
                msg.contains("Failed to load PyTorch weights"),
                "Expected PyTorch load error, got: {}",
                msg
            );
        }
    }

    /// 验证内存超限时触发 CPU 回退(覆盖 Exceeded 分支)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_memory_limit_exceeded_triggers_fallback() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let mut engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let controller = Arc::new(MemoryLimitController::with_config(
            crate::device::memory_limit::MemoryLimitConfig {
                limit_bytes: 1024,
                warning_threshold_percent: 80,
                critical_threshold_percent: 90,
            },
        ));
        controller.update_usage(2048).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Exceeded);
        engine.set_memory_limit_controller(controller);
        let fallback = engine.check_memory_limit_and_fallback(&config).await;
        assert!(
            fallback.is_ok(),
            "check_memory_limit_and_fallback failed: {:?}",
            fallback.err()
        );
        assert!(
            fallback.unwrap(),
            "Should trigger fallback when memory exceeded"
        );
        assert!(
            engine.is_fallback_triggered(),
            "fallback_triggered flag must be set"
        );
    }

    /// 验证 Critical 状态在 CPU 上不触发回退(因 check_memory_pressure 返回 false)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_memory_limit_critical_no_fallback_on_cpu() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let mut engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let controller = Arc::new(MemoryLimitController::with_config(
            crate::device::memory_limit::MemoryLimitConfig {
                limit_bytes: 1024,
                warning_threshold_percent: 80,
                critical_threshold_percent: 90,
            },
        ));
        controller.update_usage(922).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Critical);
        engine.set_memory_limit_controller(controller);
        let fallback = engine.check_memory_limit_and_fallback(&config).await;
        assert!(
            fallback.is_ok(),
            "check_memory_limit_and_fallback failed: {:?}",
            fallback.err()
        );
        assert!(
            !fallback.unwrap(),
            "Should NOT trigger fallback on CPU with Critical status"
        );
        assert!(
            !engine.is_fallback_triggered(),
            "fallback_triggered should remain false"
        );
    }

    /// 验证大批量推理(4+ 文本)覆盖批量处理循环路径
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_large_batch_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let texts: Vec<String> = vec![
            "first sentence".to_string(),
            "second sentence".to_string(),
            "third sentence".to_string(),
            "fourth sentence".to_string(),
        ];
        let result = engine.embed_batch(&texts);
        assert!(result.is_ok(), "large batch failed: {:?}", result.err());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 4);
        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(emb.len(), 384, "Embedding {} should be 384-dim", i);
        }
    }

    /// 验证 estimate_memory_usage 在不同精度下的返回值(覆盖所有 precision 分支)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_estimate_memory_usage_all_precisions() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();

        let engine_fp32 =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load FP32 model");
        let mem_fp32 = engine_fp32.estimate_memory_usage(1, 128);
        assert!(mem_fp32 > 0, "FP32 memory estimate should be positive");

        let engine_fp16 =
            CandleEngine::new(&config, Precision::Fp16).expect("Failed to load FP16 model");
        let mem_fp16 = engine_fp16.estimate_memory_usage(1, 128);
        assert!(mem_fp16 > 0, "FP16 memory estimate should be positive");
        assert!(mem_fp16 < mem_fp32, "FP16 should use less memory than FP32");

        let engine_int8 =
            CandleEngine::new(&config, Precision::Int8).expect("Failed to load INT8 model");
        let mem_int8 = engine_int8.estimate_memory_usage(1, 128);
        assert!(mem_int8 > 0, "INT8 memory estimate should be positive");
        // INT8 量化未实现，回退 FP32，内存估算与 FP32 相同
        assert_eq!(
            mem_int8, mem_fp32,
            "INT8 not implemented, should fall back to FP32 memory usage"
        );

        let mem_batch = engine_fp32.estimate_memory_usage(8, 256);
        assert!(
            mem_batch > mem_fp32,
            "Larger batch/seq should use more memory"
        );
    }

    /// 验证长文本推理覆盖 tokenization 长序列路径
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_long_text_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let long_text = "This is a very long text used for testing. ".repeat(20);
        let result = engine.embed(&long_text);
        assert!(result.is_ok(), "long text embed failed: {:?}", result.err());
        assert_eq!(result.unwrap().len(), 384);
    }

    /// 验证批量推理与单条推理结果的一致性
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_batch_vs_single_consistency() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let text1 = "hello world";
        let text2 = "machine learning";
        let single1 = engine.embed(text1).expect("single embed 1 failed");
        let single2 = engine.embed(text2).expect("single embed 2 failed");
        let batch: Vec<String> = vec![text1.to_string(), text2.to_string()];
        let batch_result = engine.embed_batch(&batch).expect("batch embed failed");
        assert_eq!(batch_result.len(), 2);
        for (i, (single, batch_emb)) in [single1, single2]
            .iter()
            .zip(batch_result.iter())
            .enumerate()
        {
            for (a, b) in single.iter().zip(batch_emb.iter()) {
                assert!(
                    (a - b).abs() < 1e-3,
                    "Batch vs single mismatch at index {} (diff > 1e-3)",
                    i
                );
            }
        }
    }

    /// 验证 Mean pooling 模型(MiniLM)批量推理与单条推理一致。
    /// 回归钉：批量路径曾硬编码 CLS 提取，Mean/Max 模型批量结果错误。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_mean_pooling_batch_vs_single_consistency() {
        const MEAN_MODEL_PATH: &str = "models/all-MiniLM-L6-v2";
        let has_weights =
            std::path::Path::new(&format!("{MEAN_MODEL_PATH}/model.safetensors")).exists();
        if !has_weights {
            eprintln!("Skipping test: model weights not found at {MEAN_MODEL_PATH}");
            return;
        }
        use crate::config::model::PoolingMode;
        let config = ModelConfig {
            name: "all-MiniLM-L6-v2".to_string(),
            model_path: PathBuf::from(MEAN_MODEL_PATH),
            pooling_mode: Some(PoolingMode::Mean),
            ..real_model_config()
        };
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load MiniLM model");
        let text = "vector search quality matters";
        let single = engine.embed(text).expect("single embed failed");
        let batch = engine
            .embed_batch(&[text.to_string()])
            .expect("batch embed failed");
        assert_eq!(batch.len(), 1);
        let batch_emb = &batch[0];
        let dot: f32 = single
            .iter()
            .zip(batch_emb.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot >= 0.999,
            "Mean pooling batch vs single cosine {dot} < 0.999 (batch path likely pooling CLS)"
        );
    }

    /// 验证 Unicode 文本(中文+emoji)推理覆盖 tokenizer 多字节路径
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_unicode_text_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp32).expect("Failed to load real model");
        let result = engine.embed("你好世界 🌍 machine learning");
        assert!(result.is_ok(), "unicode embed failed: {:?}", result.err());
        assert_eq!(result.unwrap().len(), 384);
    }

    /// 验证 FP16 精度下的批量推理路径(CPU 回退到 FP32)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_fp16_batch_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Fp16).expect("Failed to load FP16 model");
        let texts: Vec<String> = vec!["fp16 batch test".to_string(), "second text".to_string()];
        let result = engine.embed_batch(&texts);
        assert!(
            result.is_ok(),
            "FP16 batch embed failed: {:?}",
            result.err()
        );
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    /// 验证 INT8 精度下的批量推理路径
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_real_model_int8_batch_embed() {
        if !require_real_model() {
            return;
        }
        let config = real_model_config();
        let engine =
            CandleEngine::new(&config, Precision::Int8).expect("Failed to load INT8 model");
        let texts: Vec<String> = vec!["int8 batch test".to_string(), "second text".to_string()];
        let result = engine.embed_batch(&texts);
        assert!(
            result.is_ok(),
            "INT8 batch embed failed: {:?}",
            result.err()
        );
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    fn estimate_bert_params_base() {
        // BERT-Base: vocab=30522, hidden=768, layers=12, intermediate=3072, heads=12
        let params = estimate_bert_params(30522, 768, 12, 3072, 12);
        // BERT-Base 约 110M 参数，允许 10% 偏差
        assert!(
            params > 100_000_000 && params < 120_000_000,
            "BERT-Base params should be ~110M, got {}",
            params
        );
    }

    #[test]
    fn estimate_bert_params_large() {
        // BERT-Large: vocab=30522, hidden=1024, layers=24, intermediate=4096, heads=16
        let params = estimate_bert_params(30522, 1024, 24, 4096, 16);
        // BERT-Large 约 340M 参数
        assert!(
            params > 300_000_000 && params < 380_000_000,
            "BERT-Large params should be ~340M, got {}",
            params
        );
    }

    #[test]
    fn estimate_bert_params_xlm_roberta_base() {
        // XLM-RoBERTa-Base: vocab=250002, hidden=768, layers=12, intermediate=3072, heads=12
        let params = estimate_bert_params(250002, 768, 12, 3072, 12);
        // XLM-RoBERTa-Base 约 270M（主要因大 vocab）
        assert!(
            params > 250_000_000 && params < 300_000_000,
            "XLM-RoBERTa-Base params should be ~270M, got {}",
            params
        );
    }

    // ========================================================================
    // Pooling 数学单元测试(合成 hidden states + attention mask)
    // ========================================================================

    /// 合成 [seq_len=3, hidden_dim=4] 的 hidden states + mask
    fn synthetic_pooling_data() -> (Vec<f32>, Vec<u32>, usize) {
        let hidden_dim = 4;
        // seq_len=3, hidden_dim=4 → 12 个 f32
        // token 0 (CLS): [1.0, 2.0, 3.0, 4.0]
        // token 1:       [5.0, 6.0, 7.0, 8.0]
        // token 2 (pad): [0.0, 0.0, 0.0, 0.0]
        let hidden = vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            0.0, 0.0, 0.0, 0.0, //
        ];
        let mask = vec![1, 1, 0]; // token 2 是 padding
        (hidden, mask, hidden_dim)
    }

    #[test]
    fn pool_cls_returns_first_token() {
        let (hidden, mask, dim) = synthetic_pooling_data();
        let result = pool_cls(&hidden, &mask, dim);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pool_mean_mask_weighted_average() {
        let (hidden, mask, dim) = synthetic_pooling_data();
        let result = pool_mean(&hidden, &mask, dim);
        // mask=[1,1,0]: 仅 token 0 和 token 1 参与,mean_sum=2
        // dim 0: (1.0+5.0)/2 = 3.0
        // dim 1: (2.0+6.0)/2 = 4.0
        // dim 2: (3.0+7.0)/2 = 5.0
        // dim 3: (4.0+8.0)/2 = 6.0
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn pool_max_mask_element_wise_max() {
        let (hidden, mask, dim) = synthetic_pooling_data();
        let result = pool_max(&hidden, &mask, dim);
        // mask=[1,1,0]: 仅 token 0 和 token 1 参与
        // dim 0: max(1.0,5.0) = 5.0
        // dim 1: max(2.0,6.0) = 6.0
        // dim 2: max(3.0,7.0) = 7.0
        // dim 3: max(4.0,8.0) = 8.0
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn pool_mean_all_masked_out_returns_zero() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0]; // 全部 padding
        let result = pool_mean(&hidden, &mask, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pool_max_all_masked_out_returns_zero() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0];
        let result = pool_max(&hidden, &mask, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pool_mean_single_token_equals_that_token() {
        let hidden = vec![10.0, 20.0, 30.0];
        let mask = vec![1];
        let result = pool_mean(&hidden, &mask, 3);
        assert_eq!(result, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn pool_max_negative_values() {
        let hidden_dim = 2;
        // token 0: [-5.0, -1.0], token 1: [-3.0, -9.0]
        let hidden = vec![-5.0, -1.0, -3.0, -9.0];
        let mask = vec![1, 1];
        let result = pool_max(&hidden, &mask, hidden_dim);
        // max(-5,-3)=-3, max(-1,-9)=-1
        assert_eq!(result, vec![-3.0, -1.0]);
    }

    // ========================================================================
    // Auto pooling 推断测试
    // ========================================================================

    #[test]
    fn infer_pooling_minilm_returns_mean() {
        use crate::config::model::PoolingMode;
        assert_eq!(
            crate::engine::infer_pooling_mode("all-MiniLM-L6-v2"),
            PoolingMode::Mean
        );
        assert_eq!(
            crate::engine::infer_pooling_mode("paraphrase-MiniLM-L12-v2"),
            PoolingMode::Mean
        );
    }

    #[test]
    fn infer_pooling_e5_returns_mean() {
        use crate::config::model::PoolingMode;
        assert_eq!(
            crate::engine::infer_pooling_mode("multilingual-e5-small"),
            PoolingMode::Mean
        );
        assert_eq!(
            crate::engine::infer_pooling_mode("intfloat/e5-small-v2"),
            PoolingMode::Mean
        );
    }

    #[test]
    fn infer_pooling_gte_returns_mean() {
        use crate::config::model::PoolingMode;
        assert_eq!(
            crate::engine::infer_pooling_mode("Alibaba-NLP/gte-Qwen2-1.5B-instruct"),
            PoolingMode::Mean
        );
    }

    #[test]
    fn infer_pooling_bge_returns_cls() {
        use crate::config::model::PoolingMode;
        assert_eq!(
            crate::engine::infer_pooling_mode("BAAI/bge-small-en-v1.5"),
            PoolingMode::Cls
        );
        assert_eq!(
            crate::engine::infer_pooling_mode("BAAI/bge-large-zh-v1.5"),
            PoolingMode::Cls
        );
    }

    #[test]
    fn infer_pooling_unknown_returns_cls() {
        use crate::config::model::PoolingMode;
        // 未知模型家族 → Cls (带 warn)
        assert_eq!(
            crate::engine::infer_pooling_mode("some-unknown-model"),
            PoolingMode::Cls
        );
    }

    #[test]
    fn infer_pooling_case_insensitive() {
        use crate::config::model::PoolingMode;
        assert_eq!(
            crate::engine::infer_pooling_mode("BGE-Small"),
            PoolingMode::Cls
        );
        assert_eq!(
            crate::engine::infer_pooling_mode("MINILM-v2"),
            PoolingMode::Mean
        );
        assert_eq!(
            crate::engine::infer_pooling_mode("E5-Large"),
            PoolingMode::Mean
        );
    }

    // -- StageTimer --

    #[test]
    fn stage_timer_records_on_scope_exit() {
        let stats = StageStats::default();
        {
            let _timer = StageTimer::new(&stats, Stage::Inference);
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let snap = stats.snapshot();
        assert_eq!(
            snap.counts[Stage::Inference as usize],
            1,
            "作用域退出应记录"
        );
        assert!(snap.nanos[Stage::Inference as usize] > 0, "耗时应大于 0");
    }

    #[test]
    fn stage_timer_records_on_early_return_path() {
        // 模拟 `?` 提前返回：守卫 Drop 仍须累加（无锁快路径不漏计）
        fn fallible(stats: &StageStats) -> Result<(), ()> {
            let _timer = StageTimer::new(stats, Stage::Tokenize);
            Err(())
        }
        let stats = StageStats::default();
        let _ = fallible(&stats);
        assert_eq!(
            stats.snapshot().counts[Stage::Tokenize as usize],
            1,
            "提前返回路径也应记录"
        );
    }

    /// 回归钉（2026-10-18）：candle 引擎显式配 `device=cuda` 而构建缺 `candle-cuda` 时，
    /// 必须**启动期 ConfigError 显性拒绝**，不得静默落 CPU（旧行为只打 INFO，会把
    /// CPU 数字记到 GPU 档位名下）。仅在无 `candle-cuda` 的构建下断言（有该特性时
    /// 结果取决于本机 GPU，不属本测试范围）。
    #[cfg(not(feature = "candle-cuda"))]
    #[test]
    fn resolve_compute_device_rejects_cuda_without_candle_cuda() {
        let err = resolve_compute_device(&DeviceType::Cuda)
            .expect_err("缺 candle-cuda 时必须显性拒绝，不得静默落 CPU");
        assert!(
            matches!(err, VecboostError::ConfigError(_)),
            "错误类型应为 ConfigError，got {err:?}"
        );
    }
}
