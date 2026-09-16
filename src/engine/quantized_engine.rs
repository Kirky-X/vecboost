// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! GGUF 量化推理引擎（feature `quantized-gguf`）。
//!
//! 路由规则：`model_path` 以 `.gguf` 结尾（大小写不敏感）**且**
//! `ModelConfig.quantized=true` 时选用量化引擎；否则维持 safetensors 路径不变。
//! 路由判定单一真源在 `super::factory::should_use_quantized_engine`（常编译）。
//!
//! 实现方式（收敛落地）：**加载期反量化桥**。candle-transformers 0.11
//! 没有 `quantized_bert` 模型，但 `candle-core::quantized` 提供完整的 GGUF
//! 读/写（v2）与 k-quants（Q8_0/Q4_K 等）张量。本引擎：
//! - `write_gguf_from_safetensors`：把 safetensors 模型按 llama.cpp BERT 命名
//!   约定写出 GGUF（二维权重按目标 dtype 量化，一维 LayerNorm/bias 保留 F32）；
//! - `QuantizedCandleEngine::load`：读 GGUF → 逐张量反量化为 f32 → 复用
//!   candle `BertModel` 前向。
//!
//! 语义边界（诚实记录）：收益是**存储与加载体积**（Q8_0 ≈ 1/4、Q4_K ≈ 1/7），
//! 运行期计算与 fp32 等价（反量化后走同一内核），不宣称算力加速。

use crate::config::model::{PoolingMode, Precision};
use crate::error::VecboostError;
use async_trait::async_trait;
use candle_core::quantized::gguf_file::{self, Value as GgufValue};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use super::candle_engine::{infer_pooling_mode, pool_cls, pool_max, pool_mean};
use super::{InferenceEngine, ModelConfig};

/// GGUF 文件魔数（"GGUF" little-endian）。
pub const GGUF_MAGIC: u32 = 0x46554747;

/// tokenizer 上下文窗口上限（与 fp32 路径 forward_pass 的 512 一致）。
const MAX_LEN: usize = 512;

/// 校验 GGUF 文件头魔数。返回 `Ok(())` 表示文件存在且魔数合法。
pub fn validate_gguf_magic(path: &Path) -> Result<(), VecboostError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| {
        VecboostError::ModelLoadError(format!("无法打开 GGUF 文件 {}: {}", path.display(), e))
    })?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| {
        VecboostError::ModelLoadError(format!("GGUF 文件过短 {}: {}", path.display(), e))
    })?;
    let value = u32::from_le_bytes(magic);
    if value != GGUF_MAGIC {
        return Err(VecboostError::ModelLoadError(format!(
            "GGUF 魔数校验失败 {}: 期望 0x{:08X}，实际 0x{:08X}",
            path.display(),
            GGUF_MAGIC,
            value
        )));
    }
    Ok(())
}

/// candle BERT 变量名 → llama.cpp GGUF 张量名（写出约定）。
/// 未列出的键（如 `embeddings.position_ids` 缓冲）由调用方过滤。
pub fn candle_to_gguf(name: &str) -> Option<String> {
    const FIXED: &[(&str, &str)] = &[
        ("embeddings.word_embeddings.weight", "token_embd.weight"),
        (
            "embeddings.position_embeddings.weight",
            "position_embd.weight",
        ),
        (
            "embeddings.token_type_embeddings.weight",
            "token_types.weight",
        ),
        ("embeddings.LayerNorm.weight", "position_embd_norm.weight"),
        ("embeddings.LayerNorm.bias", "position_embd_norm.bias"),
        ("encoder.layer_norm.weight", "output_norm.weight"),
        ("encoder.layer_norm.bias", "output_norm.bias"),
    ];
    if let Some((_, g)) = FIXED.iter().find(|(c, _)| *c == name) {
        return Some((*g).to_string());
    }
    let rest = name.strip_prefix("encoder.layer.")?;
    let (idx, tail) = rest.split_once('.')?;
    let gguf_tail = match tail {
        "attention.self.query.weight" => "attn_q.weight",
        "attention.self.query.bias" => "attn_q.bias",
        "attention.self.key.weight" => "attn_k.weight",
        "attention.self.key.bias" => "attn_k.bias",
        "attention.self.value.weight" => "attn_v.weight",
        "attention.self.value.bias" => "attn_v.bias",
        "attention.output.dense.weight" => "attn_out.weight",
        "attention.output.dense.bias" => "attn_out.bias",
        "attention.output.LayerNorm.weight" => "attn_output_norm.weight",
        "attention.output.LayerNorm.bias" => "attn_output_norm.bias",
        "intermediate.dense.weight" => "ffn_up.weight",
        "intermediate.dense.bias" => "ffn_up.bias",
        "output.dense.weight" => "ffn_down.weight",
        "output.dense.bias" => "ffn_down.bias",
        "output.LayerNorm.weight" => "attn_norm.weight",
        "output.LayerNorm.bias" => "attn_norm.bias",
        _ => return None,
    };
    Some(format!("blk.{idx}.{gguf_tail}"))
}

/// llama.cpp GGUF 张量名 → candle BERT 变量名（写出约定的逆映射，
/// 兼容 llama.cpp 新旧两种 final-LN 命名：`output_norm` / `output.output_norm`）。
pub fn gguf_to_candle(name: &str) -> Option<String> {
    const FIXED: &[(&str, &str)] = &[
        ("token_embd.weight", "embeddings.word_embeddings.weight"),
        (
            "position_embd.weight",
            "embeddings.position_embeddings.weight",
        ),
        (
            "token_types.weight",
            "embeddings.token_type_embeddings.weight",
        ),
        ("position_embd_norm.weight", "embeddings.LayerNorm.weight"),
        ("position_embd_norm.bias", "embeddings.LayerNorm.bias"),
        ("output_norm.weight", "encoder.layer_norm.weight"),
        ("output_norm.bias", "encoder.layer_norm.bias"),
        ("output.output_norm.weight", "encoder.layer_norm.weight"),
        ("output.output_norm.bias", "encoder.layer_norm.bias"),
    ];
    if let Some((_, c)) = FIXED.iter().find(|(g, _)| *g == name) {
        return Some((*c).to_string());
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let candle_tail = match tail {
        "attn_q.weight" => "attention.self.query.weight",
        "attn_q.bias" => "attention.self.query.bias",
        "attn_k.weight" => "attention.self.key.weight",
        "attn_k.bias" => "attention.self.key.bias",
        "attn_v.weight" => "attention.self.value.weight",
        "attn_v.bias" => "attention.self.value.bias",
        "attn_out.weight" => "attention.output.dense.weight",
        "attn_out.bias" => "attention.output.dense.bias",
        "attn_output_norm.weight" => "attention.output.LayerNorm.weight",
        "attn_output_norm.bias" => "attention.output.LayerNorm.bias",
        "ffn_up.weight" => "intermediate.dense.weight",
        "ffn_up.bias" => "intermediate.dense.bias",
        "ffn_down.weight" => "output.dense.weight",
        "ffn_down.bias" => "output.dense.bias",
        "attn_norm.weight" => "output.LayerNorm.weight",
        "attn_norm.bias" => "output.LayerNorm.bias",
        _ => return None,
    };
    Some(format!("encoder.layer.{idx}.{candle_tail}"))
}

/// 从模型目录（含 model.safetensors 与 config.json）写出 GGUF（v2）。
///
/// 二维 `.weight` 按目标 `dtype` 量化；一维 LayerNorm/bias 保留 F32
/// （与 llama.cpp 转换器惯例一致，量化 LayerNorm 会显著放大误差）。
/// 写出统计：`f16_fallback` 记录目标 dtype 因块大小不整除而回退 F16 的张量数
/// （candle 0.11 量化不做 padding，Q4_K 块 256 对 384 维等不整除；诚实计数）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufWriteStats {
    pub total_tensors: usize,
    pub f16_fallback: usize,
}

pub fn write_gguf_from_safetensors(
    model_dir: &Path,
    out_path: &Path,
    dtype: GgmlDType,
) -> Result<GgufWriteStats, VecboostError> {
    let weights_path = model_dir.join("model.safetensors");
    let tensors = candle_core::safetensors::load(&weights_path, &Device::Cpu).map_err(|e| {
        VecboostError::ModelLoadError(format!(
            "safetensors 加载失败 {}: {e}",
            weights_path.display()
        ))
    })?;
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| VecboostError::ModelLoadError(format!("config.json 不可读: {e}")))?,
    )
    .map_err(|e| VecboostError::ModelLoadError(format!("config.json 解析失败: {e}")))?;
    let u64_field = |key: &str| {
        config
            .get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| VecboostError::ModelLoadError(format!("config.json 缺少 {key}")))
    };
    let layers = u64_field("num_hidden_layers")? as u32;
    let heads = u64_field("num_attention_heads")? as u32;
    let hidden = u64_field("hidden_size")? as u32;
    let intermediate = u64_field("intermediate_size")? as u32;
    let max_pos = u64_field("max_position_embeddings")? as u32;

    let mut converted: Vec<(String, QTensor)> = Vec::new();
    let mut f16_fallback = 0usize;
    let mut total_quantized = 0usize;
    for (name, tensor) in &tensors {
        // 跳过非参数缓冲与 pooler 头（candle BertModel 无 pooler，池化在应用层）
        if name == "__metadata__"
            || name == "embeddings.position_ids"
            || name.starts_with("pooler.")
        {
            continue;
        }
        let Some(gguf_name) = candle_to_gguf(name) else {
            return Err(VecboostError::ModelLoadError(format!(
                "safetensors 张量 {name} 无 GGUF 映射（架构不支持或命名漂移）"
            )));
        };
        let is_2d_weight = tensor.dims().len() == 2 && name.ends_with(".weight");
        let q = if is_2d_weight {
            total_quantized += 1;
            match QTensor::quantize(tensor, dtype) {
                Ok(q) => q,
                Err(_) => {
                    // 块大小不整除等量化不可行 → F16 兜底（半精度存储，无块约束）
                    f16_fallback += 1;
                    QTensor::quantize(tensor, GgmlDType::F16).map_err(|e| {
                        VecboostError::ModelLoadError(format!(
                            "张量 {name} 目标 dtype 与 F16 兜底均失败: {e}"
                        ))
                    })?
                }
            }
        } else {
            QTensor::quantize(tensor, GgmlDType::F32)
                .map_err(|e| VecboostError::ModelLoadError(format!("张量 {name} 量化失败: {e}")))?
        };
        converted.push((gguf_name, q));
    }
    let stats = GgufWriteStats {
        total_tensors: total_quantized,
        f16_fallback,
    };

    let dir_name = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let metadata = vec![
        (
            "general.architecture",
            GgufValue::String("bert".to_string()),
        ),
        ("general.name", GgufValue::String(dir_name)),
        ("bert.block_count", GgufValue::U32(layers)),
        ("bert.attention.head_count", GgufValue::U32(heads)),
        ("bert.embedding_length", GgufValue::U32(hidden)),
        ("bert.feed_forward_length", GgufValue::U32(intermediate)),
        ("bert.context_length", GgufValue::U32(max_pos)),
        ("bert.attention.causal", GgufValue::Bool(false)),
    ];
    let mrefs: Vec<(&str, &GgufValue)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let trefs: Vec<(&str, &QTensor)> = converted.iter().map(|(k, v)| (k.as_str(), v)).collect();

    let file = std::fs::File::create(out_path).map_err(|e| {
        VecboostError::ModelLoadError(format!("GGUF 输出不可写 {}: {e}", out_path.display()))
    })?;
    let mut writer = BufWriter::new(file);
    gguf_file::write(&mut writer, &mrefs, &trefs)
        .map_err(|e| VecboostError::ModelLoadError(format!("GGUF 写入失败: {e}")))?;
    writer
        .flush()
        .map_err(|e| VecboostError::ModelLoadError(format!("GGUF 输出 flush 失败: {e}")))?;
    Ok(stats)
}

/// GGUF 量化 candle 引擎（Q8_0/Q4_K，加载期反量化桥）。
pub struct QuantizedCandleEngine {
    model_path: PathBuf,
    tokenizer: crate::text::tokenizer::CachedTokenizer,
    model: candle_transformers::models::bert::BertModel,
    pooling: PoolingMode,
    hidden_size: usize,
    vocab_size: usize,
    precision: Precision,
}

impl QuantizedCandleEngine {
    /// 加载 GGUF：魔数/架构校验 → 逐张量反量化 f32 → BertModel 装配。
    /// tokenizer 从 GGUF 同目录的 tokenizer.json 加载。
    pub fn load(path: &Path) -> Result<Self, VecboostError> {
        validate_gguf_magic(path)?;
        let file = std::fs::File::open(path).map_err(|e| {
            VecboostError::ModelLoadError(format!("GGUF 打开失败 {}: {e}", path.display()))
        })?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| VecboostError::ModelLoadError(format!("GGUF 头解析失败: {e}")))?;

        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_default();
        if arch != "bert" {
            return Err(VecboostError::ModelLoadError(format!(
                "GGUF 架构 {arch:?} 不受支持（仅 bert；文件: {}）",
                path.display()
            )));
        }
        let meta_u32 = |key: &str| -> Result<usize, VecboostError> {
            content
                .metadata
                .get(key)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| VecboostError::ModelLoadError(format!("GGUF 元数据缺少 {key}")))
        };
        let num_layers = meta_u32("bert.block_count")?;
        let num_heads = meta_u32("bert.attention.head_count")?;
        let hidden = meta_u32("bert.embedding_length")?;
        let intermediate = meta_u32("bert.feed_forward_length")?;
        let max_pos = meta_u32("bert.context_length").unwrap_or(512);

        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::with_capacity(content.tensor_infos.len());
        for (name, info) in &content.tensor_infos {
            let qt = info
                .read(&mut reader, content.tensor_data_offset, &dev)
                .map_err(|e| {
                    VecboostError::ModelLoadError(format!("GGUF 张量 {name} 读取失败: {e}"))
                })?;
            let tensor = qt.dequantize(&dev).map_err(|e| {
                VecboostError::ModelLoadError(format!("GGUF 张量 {name} 反量化失败: {e}"))
            })?;
            let candle_name = gguf_to_candle(name).ok_or_else(|| {
                VecboostError::ModelLoadError(format!("GGUF 张量 {name} 无 candle 映射"))
            })?;
            if map.insert(candle_name.clone(), tensor).is_some() {
                return Err(VecboostError::ModelLoadError(format!(
                    "GGUF 张量 {name} 映射冲突（{candle_name} 重复）"
                )));
            }
        }

        let vocab_size = map
            .get("embeddings.word_embeddings.weight")
            .ok_or_else(|| {
                VecboostError::ModelLoadError("GGUF 缺少 token_embd.weight".to_string())
            })?
            .dim(0)
            .map_err(|e| VecboostError::ModelLoadError(format!("token_embd 维度读取失败: {e}")))?;
        let type_vocab_size = map
            .get("embeddings.token_type_embeddings.weight")
            .map(|t| t.dim(0).unwrap_or(2))
            .unwrap_or(2);

        let config = candle_transformers::models::bert::Config {
            vocab_size,
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            intermediate_size: intermediate,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: max_pos,
            type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Default::default(),
            use_cache: false,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        };
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &dev);
        let model = candle_transformers::models::bert::BertModel::load(vb, &config)
            .map_err(|e| VecboostError::ModelLoadError(format!("GGUF BERT 装配失败: {e}")))?;

        // tokenizer 与 pooling 约定沿用 fp32 路径：GGUF 同目录的 tokenizer.json，
        // 池化按模型目录名推断（bge→Cls / MiniLM→Mean / e5→Mean）。
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let tok_path = parent.join("tokenizer.json");
        let hf = tokenizers::Tokenizer::from_file(tok_path.as_os_str()).map_err(|e| {
            VecboostError::ModelLoadError(format!(
                "GGUF 同目录 tokenizer.json 加载失败（{}）: {e}",
                tok_path.display()
            ))
        })?;
        let tokenizer = crate::text::tokenizer::CachedTokenizer::new(hf, MAX_LEN, 1024);
        let dir_name = parent
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let pooling = match infer_pooling_mode(&dir_name) {
            PoolingMode::Auto => PoolingMode::Cls, // 推断失败按 BERT 惯例 Cls
            other => other,
        };

        Ok(Self {
            model_path: path.to_path_buf(),
            tokenizer,
            model,
            pooling,
            hidden_size: hidden,
            vocab_size,
            precision: Precision::Fp32,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// 单文本前向：与 fp32 CandleEngine::forward_pass 相同的
    /// tokenize → forward → pooling 链路（不含归一化，行为对齐 fp32 引擎）。
    fn forward_one(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        let encoding = self
            .tokenizer
            .encode_sync(text, true)
            .map_err(|e| VecboostError::TokenizationError(e.to_string()))?;
        let out = self.forward_encoded(
            encoding.get_ids(),
            encoding.get_attention_mask(),
            &encoding.type_ids,
        )?;
        Ok(out)
    }

    fn forward_encoded(
        &self,
        ids: &[u32],
        mask: &[u32],
        type_ids: &[u32],
    ) -> Result<Vec<f32>, VecboostError> {
        let clamp = |v: &u32| {
            if *v >= self.vocab_size as u32 {
                self.vocab_size as u32 - 1
            } else {
                *v
            }
        };
        let ids: Vec<u32> = ids.iter().take(MAX_LEN).map(clamp).collect();
        let mask: Vec<u32> = mask.iter().take(MAX_LEN).copied().collect();
        let mask_for_pooling = mask.clone();
        let type_ids: Vec<u32> = type_ids.iter().take(MAX_LEN).copied().collect();

        let token_ids = Tensor::new(ids, &self.model.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let token_type_ids = Tensor::new(type_ids, &self.model.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let attention = Tensor::new(mask, &self.model.device)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention))
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        // forward 输出 [1, seq_len, hidden]（rank 3），取 batch 0 展平
        let seq_hidden: Vec<f32> = hidden
            .to_dtype(DType::F32)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .to_vec3::<f32>()
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| VecboostError::InferenceError("空 batch".to_string()))?
            .into_iter()
            .flatten()
            .collect();
        let pooled = match self.pooling {
            PoolingMode::Cls => pool_cls(&seq_hidden, &mask_for_pooling, self.hidden_size),
            PoolingMode::Mean => pool_mean(&seq_hidden, &mask_for_pooling, self.hidden_size),
            PoolingMode::Max => pool_max(&seq_hidden, &mask_for_pooling, self.hidden_size),
            PoolingMode::Auto => unreachable!("Auto 已在 load 时解析"),
        };
        Ok(pooled)
    }
}

impl QuantizedCandleEngine {
    /// 批量前向：右侧 padding 对齐后一次 forward，逐样本按各自 mask 池化。
    fn forward_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, VecboostError> {
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| self.tokenizer.encode_sync(t, true))
            .collect::<Result<_, _>>()
            .map_err(|e| VecboostError::TokenizationError(e.to_string()))?;
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_LEN))
            .max()
            .unwrap_or(1)
            .max(1);

        let mut all_ids = Vec::with_capacity(encodings.len() * seq_len);
        let mut all_types = Vec::with_capacity(encodings.len() * seq_len);
        let mut all_masks = Vec::with_capacity(encodings.len() * seq_len);
        let mut per_sample_masks = Vec::with_capacity(encodings.len());
        let clamp = |v: &u32| {
            if *v >= self.vocab_size as u32 {
                self.vocab_size as u32 - 1
            } else {
                *v
            }
        };
        for e in &encodings {
            let ids: Vec<u32> = e.get_ids().iter().take(MAX_LEN).map(clamp).collect();
            let types: Vec<u32> = e.type_ids.iter().take(MAX_LEN).copied().collect();
            let mut mask: Vec<u32> = e
                .get_attention_mask()
                .iter()
                .take(MAX_LEN)
                .copied()
                .collect();
            let pad = seq_len - ids.len();
            all_ids.extend(ids.iter().copied());
            all_ids.extend(std::iter::repeat(0u32).take(pad));
            all_types.extend(types);
            all_types.extend(std::iter::repeat(0u32).take(pad));
            mask.extend(std::iter::repeat(0u32).take(pad));
            per_sample_masks.push(mask.clone());
            all_masks.extend(mask);
        }

        let dev = &self.model.device;
        let token_ids = Tensor::new(all_ids, dev)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape((encodings.len(), seq_len))
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let token_type_ids = Tensor::new(all_types, dev)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape((encodings.len(), seq_len))
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let attention = Tensor::new(all_masks, dev)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .reshape((encodings.len(), seq_len))
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;

        let hidden = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention))
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
        let flat: Vec<f32> = hidden
            .to_dtype(DType::F32)
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .to_vec3::<f32>()
            .map_err(|e| VecboostError::InferenceError(e.to_string()))?
            .into_iter()
            .flatten()
            .flatten()
            .collect();

        Ok(encodings
            .iter()
            .zip(per_sample_masks)
            .enumerate()
            .map(|(b, (_, mask))| {
                let start = b * seq_len * self.hidden_size;
                let slice = &flat[start..start + seq_len * self.hidden_size];
                match self.pooling {
                    PoolingMode::Cls => pool_cls(slice, &mask, self.hidden_size),
                    PoolingMode::Mean => pool_mean(slice, &mask, self.hidden_size),
                    PoolingMode::Max => pool_max(slice, &mask, self.hidden_size),
                    PoolingMode::Auto => unreachable!("Auto 已在 load 时解析"),
                }
            })
            .collect())
    }
}

#[async_trait]
impl InferenceEngine for QuantizedCandleEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        self.forward_one(text)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.forward_batch(&refs)
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        self.tokenizer.count_tokens(text)
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::io::Write;

    fn write_gguf_magic_only(path: &Path) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
    }

    #[test]
    fn test_routing_matrix() {
        use super::super::factory::should_use_quantized_engine;
        let cases = [
            ("model.gguf", true, true),
            ("model.GGUF", true, true),
            ("model.safetensors", true, false),
            ("model.gguf", false, false),
            ("model", true, false),
            ("model.safetensors", false, false),
        ];
        for (p, q, expected) in cases {
            assert_eq!(
                should_use_quantized_engine(&PathBuf::from(p), q),
                expected,
                "路由矩阵: path={p} quantized={q}"
            );
        }
    }

    #[test]
    fn test_validate_gguf_magic_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q8_0.gguf");
        write_gguf_magic_only(&path);
        assert!(validate_gguf_magic(&path).is_ok());
    }

    #[test]
    fn test_validate_gguf_magic_rejects_bad_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"NOPE........").unwrap();
        assert!(validate_gguf_magic(&path).is_err());
        assert!(validate_gguf_magic(&dir.path().join("missing.gguf")).is_err());
    }

    #[test]
    fn test_name_mapping_roundtrip() {
        let cases = [
            "embeddings.word_embeddings.weight",
            "encoder.layer.6.attention.self.query.weight",
            "encoder.layer.3.output.LayerNorm.bias",
            "encoder.layer_norm.weight",
        ];
        for c in cases {
            let g = candle_to_gguf(c).unwrap();
            assert_eq!(gguf_to_candle(&g).unwrap(), c, "{c} 往返");
        }
        // llama.cpp 新旧 final-LN 命名兼容
        assert_eq!(
            gguf_to_candle("output.output_norm.weight").unwrap(),
            "encoder.layer_norm.weight"
        );
        assert!(candle_to_gguf("embeddings.position_ids").is_none());
    }

    /// 端到端：微型 BERT → GGUF(Q8_0) → 引擎加载 → 推理输出形状/有限性。
    #[test]
    fn test_tiny_model_end_to_end() {
        let (vocab, hidden, heads, layers, ffn, max_pos, type_vocab) =
            (32usize, 8usize, 2usize, 1usize, 16usize, 16usize, 2usize);
        let dev = Device::Cpu;
        let rand = |shape: (usize, usize)| Tensor::rand(0.0f32, 0.5f32, shape, &dev).unwrap();
        let ones_1d = |n: usize| Tensor::ones(n, DType::F32, &dev).unwrap();

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "embeddings.word_embeddings.weight".into(),
            rand((vocab, hidden)),
        );
        t.insert(
            "embeddings.position_embeddings.weight".into(),
            rand((max_pos, hidden)),
        );
        t.insert(
            "embeddings.token_type_embeddings.weight".into(),
            rand((type_vocab, hidden)),
        );
        t.insert("embeddings.LayerNorm.weight".into(), ones_1d(hidden));
        t.insert("embeddings.LayerNorm.bias".into(), ones_1d(hidden));
        for suffix in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ] {
            let (i, o) = if suffix == "output.dense" {
                (ffn, hidden)
            } else {
                (hidden, hidden)
            };
            // intermediate.dense 是 [ffn, hidden]，其余 [hidden, hidden]
            let (i, o) = if suffix == "intermediate.dense" {
                (hidden, ffn)
            } else {
                (i, o)
            };
            t.insert(format!("encoder.layer.0.{suffix}.weight"), rand((o, i)));
            t.insert(format!("encoder.layer.0.{suffix}.bias"), ones_1d(o));
        }
        t.insert(
            "encoder.layer.0.attention.output.LayerNorm.weight".into(),
            ones_1d(hidden),
        );
        t.insert(
            "encoder.layer.0.attention.output.LayerNorm.bias".into(),
            ones_1d(hidden),
        );
        t.insert(
            "encoder.layer.0.output.LayerNorm.weight".into(),
            ones_1d(hidden),
        );
        t.insert(
            "encoder.layer.0.output.LayerNorm.bias".into(),
            ones_1d(hidden),
        );
        t.insert("encoder.layer_norm.weight".into(), ones_1d(hidden));
        t.insert("encoder.layer_norm.bias".into(), ones_1d(hidden));

        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("tiny-bert");
        std::fs::create_dir(&model_dir).unwrap();
        candle_core::safetensors::save(&t, &model_dir.join("model.safetensors")).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{"vocab_size": {vocab}, "hidden_size": {hidden}, "num_hidden_layers": {layers},
                    "num_attention_heads": {heads}, "intermediate_size": {ffn},
                    "max_position_embeddings": {max_pos}, "type_vocab_size": {type_vocab}}}"#
            ),
        )
        .unwrap();

        // 最小 WordPiece tokenizer（PAD/UNK/hello/world），与 GGUF 同目录供引擎加载
        std::fs::write(
            model_dir.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
                "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
                "decoder":null,
                "model":{"type":"WordPiece","unk_token":"UNK","continuing_subword_prefix":"",
                "max_input_chars_per_word":100,"vocab":{"PAD":0,"UNK":1,"hello":2,"world":3}}}"#,
        )
        .unwrap();
        let gguf_path = model_dir.join("tiny-q8_0.gguf");
        let stats = write_gguf_from_safetensors(&model_dir, &gguf_path, GgmlDType::Q8_0).unwrap();
        assert!(
            stats.f16_fallback > 0,
            "8 维权重对 32 块不整除，应回退 F16: {stats:?}"
        );

        let engine = QuantizedCandleEngine::load(&gguf_path).unwrap();
        let v = engine.embed("hello world").unwrap();
        assert_eq!(v.len(), hidden, "输出维度应等于 hidden_size");
        assert!(v.iter().all(|x| x.is_finite()), "输出应全部有限");
        let batch = engine
            .embed_batch(&["hello world".to_string(), "tiny".to_string()])
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|v| v.len() == hidden));
    }
}
