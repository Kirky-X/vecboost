// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Candle 原生 Laya 决策头（`convaiinnovations/laya` PyTorch checkpoint 的
//! 纯 Rust 前向，去 onnxruntime 运行时依赖）。
//!
//! 结构 = ModernBERT-large 骨干（candle-transformers `models::modernbert`）
//! + 决策头（官方 `rl_common.py::DecisionModel`）：type_emb 按题广播，
//!   2 层 norm_first TransformerEncoderLayer，marker 位置 gather，
//!   option scorer（LayerNorm→Linear→GELU→Linear），act/escalate 头。
//!
//! # 键形态实测基线
//!
//! 权重键名以官方 checkpoint 实测为准（`model.safetensors` header 206 张量，
//! repo 提交 `55cf4c4`，经 safetensors header HTTP Range 实读），非臆测：
//!
//! - 骨干 `encoder/` 前缀（HF transformers 5.0 命名，全程无 bias）；第 0 层
//!   **无 `attn_norm`**（上游与 embedding norm 融合导出，candle 骨干的
//!   `Option<LayerNorm>` 惰性加载恰好兼容）；RoPE 无持久张量（运行时构造）。
//! - 决策头为 PyTorch `nn.TransformerEncoderLayer` 原生命名（`head.layers.*`
//!   的 fused `self_attn.in_proj_weight`、`norm_first` 语义、FFN 激活 ReLU）。
//! - 权重 F16 存储（仅 `temperature` 为 F32），经 VarBuilder 统一 upcast F32。
//!
//! 预处理/后处理/温度校准与 onnx [`super::decision::DecisionPipeline`] 同源
//! （同一预处理产物、同一后处理函数），对齐闸门见
//! `tests/candle_decision_parity.rs`（`|Δlogit| ≤ 1e-4`）。

use std::collections::BTreeSet;
use std::path::Path;

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{VarBuilder, embedding, layer_norm, linear};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use tokenizers::Tokenizer;

use crate::config::model::{DecisionParams, DeviceType, ModelConfig, ModelTask, Precision};
use crate::domain::{DecisionRequest, DecisionResponse};
use crate::engine::decision_protocol::{
    BatchTensors, FixedMarkers, TemperatureCalibration, answer_for_question, build_question_row,
    collate_batch, encode_ids, qtype_code, state_text,
};
use crate::error::VecboostError;

/// 决策头层数（官方 rl_agent_config.json `head_layers`）
pub(crate) const HEAD_LAYERS: usize = 2;
/// act/escalate 头输出类数（n_act = len(act_costs) + 1）
pub(crate) const ACT_CLASSES: usize = 2;
/// qtype 类型嵌入词表宽（choice/score/noul 三分支）
pub(crate) const QTYPE_VOCAB: usize = 3;
/// act 头特征宽：pooled [CLS] + (top1, top1−top2, 归一化熵, 有效 marker 占比)
pub(crate) const ACT_FEATS: usize = 4;
/// act 头隐层宽（官方 nn.Sequential(nn.Linear(d+4, 256), ...)）
pub(crate) const ACT_HIDDEN: usize = 256;
/// PyTorch `nn.LayerNorm` 默认 eps（决策头 norm1/norm2/scorer.0 均未显式
/// 指定，取默认值；骨干 eps 走 candle Config.layer_norm_eps）
const PYTORCH_LAYER_NORM_EPS: f64 = 1e-5;
/// pad marker 位的 logit 填充值（官方 `logits.masked_fill(~marker_mask, -1e4)`）
const MASKED_LOGIT: f32 = -1e4;

/// 官方 checkpoint 的必需键全集（提交基线 `55cf4c4` 实测 206 张量）。
///
/// 模板按层数展开：骨干层 `i > 0` 才有 `attn_norm`（第 0 层融合导出）；
/// 决策头为 PyTorch `nn.TransformerEncoderLayer` 原生命名。
pub(crate) fn required_keys(num_hidden_layers: usize, head_layers: usize) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    keys.insert("encoder.embeddings.tok_embeddings.weight".to_string());
    keys.insert("encoder.embeddings.norm.weight".to_string());
    keys.insert("encoder.final_norm.weight".to_string());
    for i in 0..num_hidden_layers {
        keys.insert(format!("encoder.layers.{i}.attn.Wqkv.weight"));
        keys.insert(format!("encoder.layers.{i}.attn.Wo.weight"));
        if i > 0 {
            keys.insert(format!("encoder.layers.{i}.attn_norm.weight"));
        }
        keys.insert(format!("encoder.layers.{i}.mlp.Wi.weight"));
        keys.insert(format!("encoder.layers.{i}.mlp.Wo.weight"));
        keys.insert(format!("encoder.layers.{i}.mlp_norm.weight"));
    }
    for j in 0..head_layers {
        let base = format!("head.layers.{j}.");
        for name in [
            "self_attn.in_proj_weight",
            "self_attn.in_proj_bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ] {
            keys.insert(format!("{base}{name}"));
        }
    }
    keys.insert("type_emb.weight".to_string());
    for name in [
        "scorer.0.weight",
        "scorer.0.bias",
        "scorer.1.weight",
        "scorer.1.bias",
        "scorer.3.weight",
        "scorer.3.bias",
        "act_head.0.weight",
        "act_head.0.bias",
        "act_head.2.weight",
        "act_head.2.bias",
        "temperature",
    ] {
        keys.insert(name.to_string());
    }
    keys
}

/// 骨干 + 决策头权重集合（`load` 一次构建，前向只读）。
pub(crate) struct CandleLayaWeights {
    encoder: ModernBert,
    /// norm_first TransformerEncoderLayer 决策头层
    head: Vec<DecisionHeadLayer>,
    /// qtype → 偏置向量 [QTYPE_VOCAB, hidden]，逐题广播加到全部位置
    type_emb: candle_nn::Embedding,
    /// option scorer：LayerNorm → Linear(hidden→hidden) → GELU → Linear(hidden→1)
    scorer_norm: candle_nn::LayerNorm,
    scorer_hidden: candle_nn::Linear,
    scorer_out: candle_nn::Linear,
    /// act/escalate 头：Linear(hidden+4→ACT_HIDDEN) → GELU → Linear(ACT_HIDDEN→ACT_CLASSES)。
    /// 生产热路径不消费（见 forward_act 注释），加载保留为 R-001 键映射完整性
    #[allow(dead_code)]
    act_hidden: candle_nn::Linear,
    #[allow(dead_code)]
    act_out: candle_nn::Linear,
    /// checkpoint 自带的 per-qtype 温度 [QTYPE_VOCAB]（F32）。加载即校验
    /// 键存在与 F32 upcast（键映射完整性）；运行时校准以 bundle
    /// `laya_config.json` 的 cardinality 分桶为准（decision.rs 同源）——
    /// checkpoint 缓冲是训练侧 fitted 产物，官方 API 推理路径亦不消费它。
    #[allow(dead_code)]
    temperature: Tensor,
    /// 前向张量落位设备（与权重一致）
    device: Device,
}

impl CandleLayaWeights {
    /// 从 VarBuilder 加载：先对 [`required_keys`] 做存在性校验（缺任一键即
    /// `ModelLoadError` 列出全部缺失键清单，禁止静默跳过），再构建各模块。
    ///
    /// 校验针对 checkpoint 原生键名（`encoder/` 前缀）；骨干加载经前缀
    /// 重映射（`encoder.` → candle ModernBert 的 `model.`），决策头键名
    /// 原生即 candle 侧路径。
    pub(crate) fn load(
        vb: &VarBuilder<'_>,
        config: &ModernBertConfig,
    ) -> Result<Self, VecboostError> {
        let missing: Vec<String> = required_keys(config.num_hidden_layers, HEAD_LAYERS)
            .into_iter()
            .filter(|key| !vb.contains_tensor(key))
            .collect();
        if !missing.is_empty() {
            return Err(VecboostError::ModelLoadError(format!(
                "Laya checkpoint is missing {} required tensor(s): {}",
                missing.len(),
                missing.join(", ")
            )));
        }
        // rename 方向：candle 查询名 → checkpoint 名（骨干查询 model.* →
        // encoder.*；决策头/校准查询名不含 model. 前缀，不受影响）
        let renamed = vb
            .clone()
            .rename_f(|key| key.replacen("model.", "encoder.", 1));
        let encoder =
            ModernBert::load(renamed, config).map_err(|e| model_load("modernbert backbone", e))?;
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head = (0..HEAD_LAYERS)
            .map(|j| DecisionHeadLayer::load(&vb.pp(format!("head.layers.{j}")), hidden, num_heads))
            .collect::<Result<Vec<_>, VecboostError>>()?;
        let type_emb = embedding(QTYPE_VOCAB, hidden, vb.pp("type_emb"))
            .map_err(|e| model_load("type_emb", e))?;
        let scorer_norm = layer_norm(hidden, PYTORCH_LAYER_NORM_EPS, vb.pp("scorer.0"))
            .map_err(|e| model_load("scorer.0", e))?;
        let scorer_hidden =
            linear(hidden, hidden, vb.pp("scorer.1")).map_err(|e| model_load("scorer.1", e))?;
        let scorer_out =
            linear(hidden, 1, vb.pp("scorer.3")).map_err(|e| model_load("scorer.3", e))?;
        let act_hidden = linear(hidden + ACT_FEATS, ACT_HIDDEN, vb.pp("act_head.0"))
            .map_err(|e| model_load("act_head.0", e))?;
        let act_out = linear(ACT_HIDDEN, ACT_CLASSES, vb.pp("act_head.2"))
            .map_err(|e| model_load("act_head.2", e))?;
        let temperature = vb
            .get(QTYPE_VOCAB, "temperature")
            .map_err(|e| model_load("temperature", e))?
            .to_dtype(DType::F32)
            .map_err(|e| model_load("temperature upcast", e))?;
        Ok(Self {
            encoder,
            head,
            type_emb,
            scorer_norm,
            scorer_hidden,
            scorer_out,
            act_hidden,
            act_out,
            temperature,
            device: vb.device().clone(),
        })
    }

    /// 生产前向：只产出未校准 logits `[B, N]`（pad 位 [`MASKED_LOGIT`]）。
    /// act/escalate 头无 wire 消费者（onnx 对拍侧同无），热路径不付其
    /// softmax 主机往返、特征提取与线性层的代价。
    pub(crate) fn forward(&self, batch: &BatchTensors) -> Result<Tensor, VecboostError> {
        self.forward_hidden(batch).map(|(logits, _)| logits)
    }

    /// 完整前向（官方 `rl_common.py::DecisionModel.forward` 的逐算子实现，
    /// eval 语义）：modernbert 骨干 → type_emb 按题广播 → norm_first 决策头
    /// 层 → marker gather → option scorer → pad marker 位 -1e4。输入即 onnx
    /// 管线同一 [`BatchTensors`] 产物（同源预处理），返回未校准 logits 与
    /// 决策头层处理后的隐藏态 `[B, T, D]`（含 final_norm 与 type_emb 加法）；
    /// act 头经 [`Self::forward_act`] 消费两者。
    pub(crate) fn forward_hidden(
        &self,
        batch: &BatchTensors,
    ) -> Result<(Tensor, Tensor), VecboostError> {
        let dev = &self.device;
        let (b, t, n) = (batch.batch_size, batch.seq_len, batch.max_markers);
        let input_ids: Vec<u32> = batch
            .input_ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| token_range(id)))
            .collect::<Result<_, VecboostError>>()?;
        let attention: Vec<u32> = batch
            .attention_mask
            .iter()
            .map(|&m| u32::try_from(m).map_err(|_| token_range(m)))
            .collect::<Result<_, VecboostError>>()?;
        let qtype: Vec<u32> = batch
            .qtype
            .iter()
            .map(|&q| u32::try_from(q).map_err(|_| token_range(q)))
            .collect::<Result<_, VecboostError>>()?;
        let marker_pos: Vec<u32> = batch
            .marker_pos
            .iter()
            .map(|&p| {
                u32::try_from(p).map_err(|_| token_range(p)).and_then(|v| {
                    if v as usize >= t {
                        Err(VecboostError::InferenceError(format!(
                            "marker position {p} out of range for seq_len {t}"
                        )))
                    } else {
                        Ok(v)
                    }
                })
            })
            .collect::<Result<_, VecboostError>>()?;

        let xs = Tensor::from_vec(input_ids, (b, t), dev).map_err(fwd)?;
        let mask = Tensor::from_vec(attention, (b, t), dev).map_err(fwd)?;
        // 骨干输出含 final_norm（HF ModernBert.last_hidden_state 语义）
        let mut h = self.encoder.forward(&xs, &mask).map_err(fwd)?;
        // type_emb(qtype)[:, None, :]：逐题同一向量广播到全部位置
        let qtype_t = Tensor::from_vec(qtype, b, dev).map_err(fwd)?;
        let type_bias = self
            .type_emb
            .forward(&qtype_t)
            .map_err(fwd)?
            .unsqueeze(1)
            .map_err(fwd)?;
        h = h.broadcast_add(&type_bias).map_err(fwd)?;

        // src_key_padding_mask → additive bias [B, 1, 1, T]（pad key 位 f32::MIN，
        // softmax 减 max 后严格 0——ModernBert 内部同款数值口径）
        let pad_bias = mask
            .to_dtype(DType::F32)
            .map_err(fwd)?
            .affine(-1.0, 1.0)
            .map_err(fwd)?
            .affine(f32::MIN as f64, 0.0)
            .map_err(fwd)?
            .reshape((b, 1, 1, t))
            .map_err(fwd)?;
        for layer in &self.head {
            h = layer.forward(&h, &pad_bias)?;
        }

        // marker 位置 gather：clamp(min=0) 语义由 collate 保证 pad 位 0，
        // gather 后经 masked_fill 覆盖，不进 logits。candle/torch gather
        // 索引须与 lhs 同维数——idx 沿 hidden 维 expand（官方参考实现同款）
        let (_, _, d) = h.dims3().map_err(fwd)?;
        let idx = Tensor::from_vec(marker_pos, (b, n), dev)
            .map_err(fwd)?
            .unsqueeze(2)
            .map_err(fwd)?
            .expand((b, n, d))
            .map_err(fwd)?
            .contiguous()
            .map_err(fwd)?;
        let gathered = h.contiguous().map_err(fwd)?.gather(&idx, 1).map_err(fwd)?;
        let scores = gathered
            .apply(&self.scorer_norm)
            .map_err(fwd)?
            .apply(&self.scorer_hidden)
            .map_err(fwd)?
            .gelu_erf()
            .map_err(fwd)?
            .apply(&self.scorer_out)
            .map_err(fwd)?
            .squeeze(2)
            .map_err(fwd)?;
        let marker_mask: Vec<u8> = batch.marker_mask.iter().map(|&m| u8::from(m)).collect();
        let mask_t = Tensor::from_vec(marker_mask, (b, n), dev).map_err(fwd)?;
        let masked = Tensor::full(MASKED_LOGIT, (b, n), dev).map_err(fwd)?;
        let logits = mask_t.where_cond(&scores, &masked).map_err(fwd)?;
        Ok((logits, h))
    }

    /// act/escalate 头（官方公式）：pooled [CLS] + 分布统计特征（top1/
    /// top1−top2/归一化熵/有效 marker 占比）。特征从 softmax 概率按官方
    /// 公式以 f32 逐位重算（与 PyTorch 数值一致；detach 在推理期无语义）。
    /// act 输出不进入 wire 契约（onnx 对拍侧同无）——生产热路径不调用
    /// （评审定案：softmax D2H 同步 + 特征提取 + act 线性层是无消费者开销），
    /// 权重仍按 R-001 键映射完整加载；唯一现役消费者是单元测试，未来
    /// act 概率进 API 时即插即用。
    #[allow(dead_code)]
    pub(crate) fn forward_act(
        &self,
        logits: &Tensor,
        h: &Tensor,
        marker_mask: &[bool],
    ) -> Result<Tensor, VecboostError> {
        let dev = logits.device();
        let b = logits.dim(0).map_err(fwd)?;
        let n = logits.dim(1).map_err(fwd)?;
        // softmax 强制 D2H 同步点：仅 act 消费路径支付
        let probs = softmax_last_dim(logits)
            .map_err(fwd)?
            .to_vec2::<f32>()
            .map_err(fwd)?;
        let mut feats: Vec<f32> = Vec::with_capacity(b * ACT_FEATS);
        for (row_p, row_mask) in probs.iter().zip(marker_mask.chunks(n)) {
            // clamp(min=2) 与官方一致；N=1（单选项 choice）时官方 topk(2)
            // 直接崩溃，此处 top2 以 0 退化（该组合 domain 侧不拒绝，前向
            // 必须显性存活而非 panic）
            let k = row_mask.iter().filter(|&&m| m).count().max(2);
            let entropy: f32 =
                -row_p.iter().map(|&p| p * p.max(1e-9).ln()).sum::<f32>() / (k as f32).ln();
            let mut sorted = row_p.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top1 = sorted[0];
            let top2 = sorted.get(1).copied().unwrap_or(0.0);
            // 255 为官方 rl_common.py DecisionModel.forward 的硬编码归一化
            // 常数（k/255.0 有效 marker 占比缩放，训练侧选择而非本仓推导）；
            // act 特征整体无独立对照面，已登记 decision_protocol P0 待校准清单
            feats.extend([top1, top1 - top2, entropy, k as f32 / 255.0]);
        }
        let feats_t = Tensor::from_vec(feats, (b, ACT_FEATS), dev).map_err(fwd)?;
        // pooled = h[:, 0]：决策头层处理后的 [CLS] 位
        let pooled = h.narrow(1, 0, 1).map_err(fwd)?.squeeze(1).map_err(fwd)?;
        let act_in = Tensor::cat(&[&pooled, &feats_t], 1).map_err(fwd)?;
        act_in
            .apply(&self.act_hidden)
            .map_err(fwd)?
            .gelu_erf()
            .map_err(fwd)?
            .apply(&self.act_out)
            .map_err(fwd)
    }
}

/// 加载期 candle 错误 → 显性 ModelLoadError（点名模块）
fn model_load(what: &str, e: candle_core::Error) -> VecboostError {
    VecboostError::ModelLoadError(format!("Laya decision head load failed at {what}: {e}"))
}

/// 前向期 candle 错误 → 显性 InferenceError
fn fwd(e: candle_core::Error) -> VecboostError {
    VecboostError::InferenceError(e.to_string())
}

/// i64 → u32 转换越界的显性错误（协议 id 域非负；越界即上游张量构造 bug）
fn token_range(value: i64) -> VecboostError {
    VecboostError::InferenceError(format!(
        "token id / qtype {value} outside u32 protocol range"
    ))
}

/// norm_first TransformerEncoderLayer（PyTorch `nn.TransformerEncoderLayer`
/// batch_first + norm_first + ReLU 的 eval 语义；dropout 仅训练期生效）。
struct DecisionHeadLayer {
    norm1: candle_nn::LayerNorm,
    /// fused qkv 投影 [3D, D] + bias（PyTorch MultiheadAttention in_proj）
    in_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    norm2: candle_nn::LayerNorm,
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    hidden: usize,
    num_heads: usize,
}

impl DecisionHeadLayer {
    fn load(vb: &VarBuilder<'_>, hidden: usize, num_heads: usize) -> Result<Self, VecboostError> {
        Ok(Self {
            norm1: layer_norm(hidden, PYTORCH_LAYER_NORM_EPS, vb.pp("norm1"))
                .map_err(|e| model_load("head norm1", e))?,
            // in_proj_weight/in_proj_bias 是 PyTorch MHA 的参数名（非模块
            // 路径），无法经 pp("in_proj")+linear() 取用，手工构造 Linear
            in_proj: {
                let w = vb
                    .get((hidden * 3, hidden), "self_attn.in_proj_weight")
                    .map_err(|e| model_load("head in_proj_weight", e))?;
                let b = vb
                    .get(hidden * 3, "self_attn.in_proj_bias")
                    .map_err(|e| model_load("head in_proj_bias", e))?;
                candle_nn::Linear::new(w, Some(b))
            },
            out_proj: linear(hidden, hidden, vb.pp("self_attn.out_proj"))
                .map_err(|e| model_load("head out_proj", e))?,
            norm2: layer_norm(hidden, PYTORCH_LAYER_NORM_EPS, vb.pp("norm2"))
                .map_err(|e| model_load("head norm2", e))?,
            linear1: linear(hidden, hidden * 4, vb.pp("linear1"))
                .map_err(|e| model_load("head linear1", e))?,
            linear2: linear(hidden * 4, hidden, vb.pp("linear2"))
                .map_err(|e| model_load("head linear2", e))?,
            hidden,
            num_heads,
        })
    }

    /// norm_first 前向：`x + attn(norm1(x), key_padding)` → `x + ffn(norm2(x))`。
    /// 注意 FFN 激活是 ReLU（`nn.TransformerEncoderLayer` 默认，非 GELU）。
    fn forward(&self, xs: &Tensor, pad_bias: &Tensor) -> Result<Tensor, VecboostError> {
        let (b, t, d) = xs.dims3().map_err(fwd)?;
        debug_assert_eq!(d, self.hidden);
        let head_dim = self.hidden / self.num_heads;

        // fused in_proj → [3, B, H, T, hd]（ModernBert attention 同款拆分）
        let qkv = xs
            .apply(&self.norm1)
            .map_err(fwd)?
            .apply(&self.in_proj)
            .map_err(fwd)?
            .reshape((b, t, 3, self.num_heads, head_dim))
            .map_err(fwd)?
            .permute((2, 0, 3, 1, 4))
            .map_err(fwd)?;
        let q = qkv.get(0).map_err(fwd)?;
        let k = qkv.get(1).map_err(fwd)?;
        let v = qkv.get(2).map_err(fwd)?;
        let q = (q * (head_dim as f64).powf(-0.5)).map_err(fwd)?;
        let att = q
            .matmul(&k.transpose(D::Minus2, D::Minus1).map_err(fwd)?)
            .map_err(fwd)?
            .broadcast_add(pad_bias)
            .map_err(fwd)?;
        let att = softmax_last_dim(&att).map_err(fwd)?;
        let ctx = att
            .matmul(&v)
            .map_err(fwd)?
            .transpose(1, 2)
            .map_err(fwd)?
            .reshape((b, t, self.hidden))
            .map_err(fwd)?;
        let attn_out = ctx.apply(&self.out_proj).map_err(fwd)?;
        let x = (xs + attn_out).map_err(fwd)?;
        let ffn = x
            .apply(&self.norm2)
            .map_err(fwd)?
            .apply(&self.linear1)
            .map_err(fwd)?
            .relu()
            .map_err(fwd)?
            .apply(&self.linear2)
            .map_err(fwd)?;
        (x + ffn).map_err(fwd)
    }
}

/// checkpoint 权重文件的固定文件名（docs/USER_GUIDE.md 规定的落位布局）
const WEIGHTS_FILE: &str = "model.safetensors";
/// 骨干配置相对路径（HF repo `encoder/config.json`）
const ENCODER_CONFIG_FILE: &str = "encoder/config.json";

/// Candle 原生决策引擎：`task=decision + engine_type=candle` 的分派目标。
///
/// 消费官方 PyTorch checkpoint（`models/laya-pytorch/` 布局：根目录
/// `model.safetensors` + `encoder/config.json` + `tokenizer/`，获取步骤见
/// docs/USER_GUIDE.md「Candle 原生决策路径」），无 onnxruntime
/// 运行时依赖。预处理/后处理/校准与 onnx [`super::decision::DecisionPipeline`]
/// 共享同一协议层（`decision_protocol`），温度校准同口径（缺
/// `laya_config.json` 回退 1.2）。
pub(crate) struct CandleDecisionEngine {
    weights: CandleLayaWeights,
    tokenizer: Tokenizer,
    /// score/noul 固定 marker 的预编码 id（加载期一次，热路径复用）
    fixed_markers: FixedMarkers,
    calibration: TemperatureCalibration,
    /// per-checkpoint 序列预算（加载期定值，热路径逐题消费）
    params: DecisionParams,
    precision: Precision,
}

impl CandleDecisionEngine {
    /// 从模型目录加载：权重 + 骨干配置 + tokenizer + 校准表。
    ///
    /// 加载顺序与 [`super::decision::DecisionPipeline::load`] 同款：参数
    /// 防御性复检 → 目录/资产探测（显性报错）→ sha256（配置时）→ tokenizer
    /// 清洗（清 truncation/padding）→ 固定 marker 预编码 → 校准表 → 权重。
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        // decision_params 理论恒经合法路径构造（预设表解析期校验 + 启动
        // fail-fast + switch_override 复检），字段全 pub 可库内直构非法值
        // ——加载期防御性复检，先于资产探测显性拒绝（decision.rs 同款）
        let params = match config.decision_params {
            Some(p) => DecisionParams::new(p.head_max_len, p.state_max_tokens)?,
            None => DecisionParams::default(),
        };
        let bundle_dir = &config.model_path;
        if !bundle_dir.is_dir() {
            return Err(VecboostError::ModelLoadError(format!(
                "candle decision model path is not a directory: {}",
                bundle_dir.display()
            )));
        }
        let weights_file = bundle_dir.join(WEIGHTS_FILE);
        if !weights_file.is_file() {
            return Err(VecboostError::ModelLoadError(format!(
                "Laya PyTorch checkpoint not found: {}（获取步骤见 \
                 docs/USER_GUIDE.md「Candle 原生决策路径」；若已有 receptron/laya-onnx \
                 ONNX bundle，可设置 engine_type = \"onnx\" 走 ort 管线沿用既有资产）",
                weights_file.display()
            )));
        }
        let encoder_config_file = bundle_dir.join(ENCODER_CONFIG_FILE);
        let encoder_config = parse_encoder_config(&encoder_config_file)?;

        if let Some(ref expected_hash) = config.model_sha256 {
            log::info!("Verifying candle decision checkpoint SHA256 hash...");
            let is_valid = crate::utils::hash::verify_sha256(&weights_file, expected_hash)
                .map_err(|e| {
                    VecboostError::ModelLoadError(format!("Failed to verify SHA256: {e}"))
                })?;
            if !is_valid {
                return Err(VecboostError::ModelLoadError(format!(
                    "Checkpoint SHA256 verification failed. Expected: {expected_hash}, File: {:?}",
                    weights_file
                )));
            }
        }

        let tokenizer_file = super::local_bundle::resolve_tokenizer_path(
            bundle_dir,
            config.tokenizer_path.as_deref(),
        )?;
        // 决策协议自管特殊 token 拼接与 collate 填充：资产自带的
        // truncation/padding 配置必须清除（decision.rs 同款，防序列被静默
        // pad/截断的协议漂移）
        let mut tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(|e| {
            VecboostError::ModelLoadError(format!(
                "Failed to load tokenizer {}: {e}",
                tokenizer_file.display()
            ))
        })?;
        tokenizer.with_truncation(None).map_err(|e| {
            VecboostError::ModelLoadError(format!("Failed to clear truncation: {e}"))
        })?;
        tokenizer.with_padding(None);

        let fixed_markers = FixedMarkers::new(&tokenizer)?;
        // tokenizer ↔ encoder config 跨资产一致性：tokenizer 可产出的最大
        // token id = get_vocab_size(true)-1，越过骨干词表宽即 embedding 越界
        // ——bundle 拿错 tokenizer（如 MiniLM 配 laya checkpoint）不拦就会拖到
        // 首个决策请求才在 index_select 深处以 InvalidIndex 显形，错误面远离
        // 根因；model_sha256 仅覆盖 model.safetensors，两个可信资产间的交叉
        // 一致性只能在此拦（parse_encoder_config 同款加载期语义校验口径）
        let tokenizer_vocab = tokenizer.get_vocab_size(true);
        if tokenizer_vocab > encoder_config.vocab_size {
            return Err(VecboostError::ModelLoadError(format!(
                "{}: tokenizer vocab size ({tokenizer_vocab}) exceeds encoder \
                 config vocab_size ({}) — mismatched tokenizer/encoder assets \
                 (model_sha256 covers model.safetensors only)",
                tokenizer_file.display(),
                encoder_config.vocab_size
            )));
        }
        let calibration = TemperatureCalibration::from_bundle(bundle_dir)?;
        let device = resolve_device(&config.device)?;
        // SAFETY: `VarBuilder::from_mmaped_safetensors` memory-maps model weight files
        // read-only; the checkpoint is a local, immutable asset during the engine
        // lifetime (candle_engine.rs 同款）
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], DType::F32, &device) }
                .map_err(|e| {
                    VecboostError::ModelLoadError(format!("Failed to mmap checkpoint: {e}"))
                })?;
        let weights = CandleLayaWeights::load(&vb, &encoder_config)?;

        log::info!(
            "Candle decision engine initialized: bundle={}, hidden={}, layers={}, \
             calibration_buckets={}",
            bundle_dir.display(),
            encoder_config.hidden_size,
            encoder_config.num_hidden_layers,
            calibration.temperatures.len()
        );
        Ok(Self {
            weights,
            tokenizer,
            fixed_markers,
            calibration,
            params,
            precision: Precision::Fp32,
        })
    }

    /// 决策主链路：校验 → 逐题预处理 → collate → candle 前向 → 按行
    /// softmax+温度校准 → 三类后处理（协议层与 onnx 管线同源）。
    ///
    /// logits 绝不 L2 归一化（decision.rs 同一红线：归一化毁掉 softmax 前
    /// 的概率语义）；act 头输出不进入 wire 契约（onnx 对拍侧同无）。
    pub(crate) fn decide_impl(
        &self,
        req: &DecisionRequest,
    ) -> Result<DecisionResponse, VecboostError> {
        let started = std::time::Instant::now();
        let logits_rows = self.candle_logits(req)?;
        let mut answers = Vec::with_capacity(logits_rows.len());
        for (b, question) in req.questions.iter().enumerate() {
            // 行长度 = 该题有效 marker 数（candle_logits 已做前缀切片）
            let temperature = self.calibration.temperature_for(logits_rows[b].len());
            answers.push(answer_for_question(question, &logits_rows[b], temperature)?);
        }
        let elapsed = started.elapsed();
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.observe_decision_seconds(elapsed.as_secs_f64());
            collector.record_batch_size("decision", req.questions.len() as f64);
        }
        Ok(DecisionResponse {
            answers,
            processing_time_ms: elapsed.as_millis(),
        })
    }

    /// 校验 → 逐题预处理 → collate → candle 前向 → 逐行有效 marker 前缀的
    /// **未校准** logits（温度施加前，pad 位 masked_fill 值被前缀切片剔除）。
    /// 决策主链路（[`Self::decide_impl`]）与对齐闸门（`decide_logits` trait
    /// 出口）的共享实现；决策埋点仅在 decide_impl，诊断直调不污染指标。
    pub(crate) fn candle_logits(
        &self,
        req: &DecisionRequest,
    ) -> Result<Vec<Vec<f32>>, VecboostError> {
        // trait 契约：实现方自行调用 validate（crate 内直调同样被覆盖）
        req.validate()?;
        let state = state_text(&req.state);
        let state_ids = encode_ids(&self.tokenizer, &state)?;
        let mut rows = Vec::with_capacity(req.questions.len());
        let mut qtype_codes = Vec::with_capacity(req.questions.len());
        for question in &req.questions {
            rows.push(build_question_row(
                &self.tokenizer,
                question,
                &state_ids,
                &self.fixed_markers,
                self.params,
            )?);
            qtype_codes.push(qtype_code(&question.qtype));
        }
        let batch = collate_batch(&rows, &qtype_codes)?;
        let logits = self.weights.forward(&batch)?;
        let logits = logits.to_vec2::<f32>().map_err(fwd)?;
        if logits.len() != batch.batch_size
            || logits
                .first()
                .is_some_and(|row| row.len() != batch.max_markers)
        {
            return Err(VecboostError::InferenceError(format!(
                "unexpected candle logits shape {:?}, expected [{}, {}]",
                (logits.len(), logits.first().map(Vec::len).unwrap_or(0)),
                batch.batch_size,
                batch.max_markers
            )));
        }
        // 逐行有效 marker 前缀（pad 位 -1e4 为掩码填充值，非模型输出语义）
        Ok(req
            .questions
            .iter()
            .enumerate()
            .map(|(b, _)| logits[b][..rows[b].marker_pos.len()].to_vec())
            .collect())
    }
}

/// 设备解析（candle_engine.rs 同款惯例）：Cuda/Metal 可用即用、创建失败
/// 传播 Err（显性拒绝加载，用户可显式改 device=cpu）；AMD/OpenCL 显性
/// warn 回退 CPU，其余 CPU。
fn resolve_device(device_type: &DeviceType) -> Result<Device, VecboostError> {
    if *device_type == DeviceType::Cuda && candle_core::utils::cuda_is_available() {
        log::info!("Candle decision engine using CUDA GPU");
        Device::new_cuda(0).map_err(|e| VecboostError::InferenceError(e.to_string()))
    } else if *device_type == DeviceType::Metal && candle_core::utils::metal_is_available() {
        log::info!("Candle decision engine using Metal GPU");
        Device::new_metal(0).map_err(|e| VecboostError::InferenceError(e.to_string()))
    } else {
        if matches!(device_type, DeviceType::Amd | DeviceType::OpenCL) {
            log::warn!(
                "Candle engine does not natively support AMD GPUs; \
                 candle decision engine falling back to CPU"
            );
        }
        Ok(Device::Cpu)
    }
}

/// 解析官方 repo 的 `encoder/config.json` → candle ModernBert Config。
///
/// 兼容两种 rope theta 布局：transformers 4.x 顶层平铺
/// `global_rope_theta`/`local_rope_theta`，5.x（官方 checkpoint 实测格式）
/// 嵌套 `rope_parameters.{full_attention,sliding_attention}.rope_theta`；
/// 皆缺显性报错（不臆测默认值）。
fn parse_encoder_config(path: &Path) -> Result<ModernBertConfig, VecboostError> {
    let missing = |field: &str| {
        VecboostError::ModelLoadError(format!(
            "{}: missing or invalid field `{field}` for ModernBERT encoder config",
            path.display()
        ))
    };
    let raw = std::fs::read_to_string(path).map_err(|e| {
        VecboostError::ModelLoadError(format!(
            "failed to read encoder config {}: {}（官方 repo convaiinnovations/laya 的 \
             encoder/config.json，获取步骤见 docs/USER_GUIDE.md）",
            path.display(),
            e
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        VecboostError::ModelFileCorrupted(format!("failed to parse {}: {e}", path.display()))
    })?;
    let usize_field = |field: &str| {
        value
            .get(field)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| missing(field))
    };
    let rope_theta = |keys: &[&str]| -> Result<f64, VecboostError> {
        let mut cursor = &value;
        for key in keys {
            cursor = cursor.get(key).ok_or_else(|| missing(&keys.join(".")))?;
        }
        cursor.as_f64().ok_or_else(|| missing(&keys.join(".")))
    };
    let global_rope_theta = rope_theta(&["global_rope_theta"])
        .or_else(|_| rope_theta(&["rope_parameters", "full_attention", "rope_theta"]))?;
    let local_rope_theta = rope_theta(&["local_rope_theta"])
        .or_else(|_| rope_theta(&["rope_parameters", "sliding_attention", "rope_theta"]))?;
    // 语义校验（加载期显性失败而非请求期 panic）：num_attention_heads=0 会让
    // DecisionHeadLayer 的 head_dim = hidden / num_heads 在请求期 usize 除零；
    // hidden=0 / 不整除同为 config 与结构不相容。encoder/config.json 不在
    // model_sha256 校验内，错配资产只能在此拦（decision.rs 全文显性失败口径）
    let hidden_size = usize_field("hidden_size")?;
    let num_attention_heads = usize_field("num_attention_heads")?;
    if num_attention_heads == 0 {
        return Err(VecboostError::ModelLoadError(format!(
            "{}: num_attention_heads must be >= 1, got 0",
            path.display()
        )));
    }
    if hidden_size == 0 {
        return Err(VecboostError::ModelLoadError(format!(
            "{}: hidden_size must be >= 1, got 0",
            path.display()
        )));
    }
    if hidden_size % num_attention_heads != 0 {
        return Err(VecboostError::ModelLoadError(format!(
            "{}: hidden_size ({hidden_size}) must be divisible by \
             num_attention_heads ({num_attention_heads})",
            path.display()
        )));
    }
    Ok(ModernBertConfig {
        vocab_size: usize_field("vocab_size")?,
        hidden_size,
        num_hidden_layers: usize_field("num_hidden_layers")?,
        num_attention_heads,
        intermediate_size: usize_field("intermediate_size")?,
        max_position_embeddings: usize_field("max_position_embeddings")?,
        layer_norm_eps: value
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| missing("layer_norm_eps"))?,
        pad_token_id: usize_field("pad_token_id")? as u32,
        global_attn_every_n_layers: usize_field("global_attn_every_n_layers")?,
        global_rope_theta,
        local_attention: usize_field("local_attention")?,
        local_rope_theta,
        classifier_config: None,
    })
}

#[async_trait::async_trait]
impl super::InferenceEngine for CandleDecisionEngine {
    /// 决策主链路唯一 trait 入口：委托固有方法 [`CandleDecisionEngine::decide_impl`]
    /// （固有方法名与 trait 方法无同名遮蔽，漏覆盖即继承默认 UnsupportedTask）。
    fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
        self.decide_impl(req)
    }

    /// 对齐/诊断出口：委托 [`CandleDecisionEngine::candle_logits`]
    fn decide_logits(&self, req: &DecisionRequest) -> Result<Vec<Vec<f32>>, VecboostError> {
        self.candle_logits(req)
    }

    /// 决策引擎不产向量（诚实语义，DecisionPipeline 同款）
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "candle decision engine does not produce embeddings".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "candle decision engine does not produce embeddings".to_string(),
        ))
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    /// 默认实现依赖 `embed`（bi-encoder rerank）——决策引擎不能谎报
    fn supports_rerank(&self) -> bool {
        false
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        task == ModelTask::Decision
    }

    /// candle 路径无运行时 EP 降级：设备在加载期解析定死（权重 mmap 绑定
    /// 设备，无 DecisionPipeline 式 Session 重建入口）。返回显性 Err 而非
    /// 假成功 Ok——调用方对 Ok 一律打 "Successfully fell back to CPU" 并
    /// 重试，GPU OOM 活跃路径上那会变成永不动状态的无限重试梯；Err 走
    /// 调用方 "Failed to fallback" 诚实分支，客户端即刻收到有界 OutOfMemory。
    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Err(VecboostError::inference_error(
            "candle decision engine does not support runtime device fallback \
             (device is fixed at load time); restart with device=cpu instead"
                .to_string(),
        ))
    }
}

/// 合成 checkpoint fixture（cfg(test)）：factory 分派测试与 parity 测试共享。
/// 键形态与 [`required_keys`] 同模板，小维度（内存可忽略）。
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    /// 小维度合成 checkpoint 的骨干 config（与官方 1024/28 层同构）。
    pub(crate) fn tiny_config() -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: 20,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 4,
            max_position_embeddings: 64,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160_000.0,
            local_attention: 4,
            local_rope_theta: 10_000.0,
            classifier_config: None,
        }
    }

    /// 按 required_keys 模板合成官方键形态的最小 safetensors 文件。
    /// F16 存储（与官方一致，temperature 除外），形状按官方真实维度（norm/
    /// bias 为 1D）按 tiny config 推导；`omit` 模拟上游缺键。返回落位路径。
    pub(crate) fn synthetic_checkpoint(dir: &Path, omit: &[&str]) -> PathBuf {
        let cfg = tiny_config();
        let (v, d, i) = (cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size);
        let f16 = |shape: (usize, usize)| Tensor::zeros(shape, DType::F16, &Device::Cpu).unwrap();
        let f16_1d = |n: usize| Tensor::zeros(n, DType::F16, &Device::Cpu).unwrap();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, t: Tensor| {
            if !omit.contains(&name) {
                tensors.insert(name.to_string(), t);
            }
        };
        put("encoder.embeddings.tok_embeddings.weight", f16((v, d)));
        put("encoder.embeddings.norm.weight", f16_1d(d));
        put("encoder.final_norm.weight", f16_1d(d));
        for layer in 0..cfg.num_hidden_layers {
            put(
                &format!("encoder.layers.{layer}.attn.Wqkv.weight"),
                f16((3 * d, d)),
            );
            put(
                &format!("encoder.layers.{layer}.attn.Wo.weight"),
                f16((d, d)),
            );
            if layer > 0 {
                put(
                    &format!("encoder.layers.{layer}.attn_norm.weight"),
                    f16_1d(d),
                );
            }
            put(
                &format!("encoder.layers.{layer}.mlp.Wi.weight"),
                f16((2 * i, d)),
            );
            put(
                &format!("encoder.layers.{layer}.mlp.Wo.weight"),
                f16((d, i)),
            );
            put(
                &format!("encoder.layers.{layer}.mlp_norm.weight"),
                f16_1d(d),
            );
        }
        for j in 0..HEAD_LAYERS {
            let base = format!("head.layers.{j}.");
            put(&format!("{base}self_attn.in_proj_weight"), f16((3 * d, d)));
            put(&format!("{base}self_attn.in_proj_bias"), f16_1d(3 * d));
            put(&format!("{base}self_attn.out_proj.weight"), f16((d, d)));
            put(&format!("{base}self_attn.out_proj.bias"), f16_1d(d));
            put(&format!("{base}linear1.weight"), f16((4 * d, d)));
            put(&format!("{base}linear1.bias"), f16_1d(4 * d));
            put(&format!("{base}linear2.weight"), f16((d, 4 * d)));
            put(&format!("{base}linear2.bias"), f16_1d(d));
            put(&format!("{base}norm1.weight"), f16_1d(d));
            put(&format!("{base}norm1.bias"), f16_1d(d));
            put(&format!("{base}norm2.weight"), f16_1d(d));
            put(&format!("{base}norm2.bias"), f16_1d(d));
        }
        put("type_emb.weight", f16((QTYPE_VOCAB, d)));
        put("scorer.0.weight", f16_1d(d));
        put("scorer.0.bias", f16_1d(d));
        put("scorer.1.weight", f16((d, d)));
        put("scorer.1.bias", f16_1d(d));
        put("scorer.3.weight", f16((1, d)));
        put("scorer.3.bias", f16_1d(1));
        put("act_head.0.weight", f16((ACT_HIDDEN, d + ACT_FEATS)));
        put("act_head.0.bias", f16_1d(ACT_HIDDEN));
        put("act_head.2.weight", f16((ACT_CLASSES, ACT_HIDDEN)));
        put("act_head.2.bias", f16_1d(ACT_CLASSES));
        put(
            "temperature",
            Tensor::zeros(QTYPE_VOCAB, DType::F32, &Device::Cpu).unwrap(),
        );
        let path = dir.join(WEIGHTS_FILE);
        candle_core::safetensors::save(&tensors, &path).expect("save synthetic checkpoint");
        path
    }

    /// 组装完整 candle 决策 bundle（权重 + encoder config（官方 5.0 嵌套
    /// rope_parameters 格式）+ WordLevel tokenizer.json），返回目录路径。
    /// tokenizer 词表覆盖协议所需词元（特殊 token/score 等级/noul 与测试
    /// 选项），未知词统一落 [UNK]——编码只需确定性，不需真实词汇。
    pub(crate) fn write_bundle(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir).expect("create bundle dir");
        synthetic_checkpoint(dir, &[]);
        let cfg = tiny_config();
        let encoder_config = serde_json::json!({
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "intermediate_size": cfg.intermediate_size,
            "max_position_embeddings": cfg.max_position_embeddings,
            "layer_norm_eps": cfg.layer_norm_eps,
            "pad_token_id": cfg.pad_token_id,
            "global_attn_every_n_layers": cfg.global_attn_every_n_layers,
            "local_attention": cfg.local_attention,
            "rope_parameters": {
                "full_attention": {"rope_theta": cfg.global_rope_theta, "rope_type": "default"},
                "sliding_attention": {"rope_theta": cfg.local_rope_theta, "rope_type": "default"}
            }
        });
        let encoder_dir = dir.join("encoder");
        std::fs::create_dir_all(&encoder_dir).expect("create encoder dir");
        std::fs::write(encoder_dir.join("config.json"), encoder_config.to_string())
            .expect("write encoder config");
        let vocab: serde_json::Map<String, serde_json::Value> = [
            "[CLS]", "[SEP]", "[MASK]", "[UNK]", "choice", "question", ":", "pick", "one", "score",
            "noul", "beach", "mountain", "false", "true", "0", "1", "2", "3", "4",
        ]
        .iter()
        .enumerate()
        .map(|(i, token)| ((*token).to_string(), serde_json::json!(i as u32)))
        .collect();
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "[UNK]"
            }
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string())
            .expect("write tokenizer");
        dir.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{synthetic_checkpoint, tiny_config};
    use super::*;

    /// 实测基线钉：官方 checkpoint（206 张量）的键数与代表键形态。
    /// 来源：safetensors header HTTP Range 实读（repo 提交 55cf4c4）。
    #[test]
    fn test_required_keys_matches_measured_checkpoint_shape() {
        let keys = required_keys(28, 2);
        assert_eq!(
            keys.len(),
            206,
            "28 层骨干 + 2 层决策头的必需键总数必须等于实测 206 张量"
        );
        // 骨干：第 0 层无 attn_norm（上游融合导出），1..28 有
        assert!(
            !keys.contains("encoder.layers.0.attn_norm.weight"),
            "第 0 层 attn_norm 必须不在必需清单（实测该键不存在）"
        );
        assert!(keys.contains("encoder.layers.1.attn_norm.weight"));
        assert!(keys.contains("encoder.layers.27.attn.Wqkv.weight"));
        assert!(
            !keys.contains("encoder.layers.0.attn.Wqkv.bias"),
            "transformers 5.0 导出骨干全程无 bias"
        );
        // 决策头：PyTorch TransformerEncoderLayer 原生命名（fused in_proj_weight）
        assert!(keys.contains("head.layers.0.self_attn.in_proj_weight"));
        assert!(keys.contains("head.layers.1.self_attn.out_proj.bias"));
        assert!(keys.contains("head.layers.1.linear2.weight"));
        // scorer/act/温度
        assert!(keys.contains("scorer.0.weight"));
        assert!(keys.contains("scorer.3.bias"));
        assert!(keys.contains("act_head.0.weight"));
        assert!(keys.contains("temperature"));
        assert!(
            !keys.contains("model.embeddings.tok_embeddings.weight"),
            "必需键用 checkpoint 原生命名（encoder/ 前缀），非 candle 内部路径"
        );
    }

    #[test]
    fn test_weights_load_maps_official_key_subset() {
        let cfg = tiny_config();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = synthetic_checkpoint(dir.path(), &[]);
        let raw = std::fs::read(&path).expect("read checkpoint");
        let vb = VarBuilder::from_buffered_safetensors(raw, DType::F32, &Device::Cpu)
            .expect("varbuilder");
        let weights = CandleLayaWeights::load(&vb, &cfg).expect("load weights");
        assert_eq!(weights.head.len(), HEAD_LAYERS);
        assert_eq!(
            weights.temperature.dims(),
            &[QTYPE_VOCAB],
            "per-qtype 温度缓冲必须按 [3] 加载"
        );
    }

    #[test]
    fn test_weights_load_missing_keys_reported_together() {
        let cfg = tiny_config();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = synthetic_checkpoint(
            dir.path(),
            &[
                "encoder.embeddings.tok_embeddings.weight",
                "head.layers.1.linear1.bias",
                "scorer.3.weight",
            ],
        );
        let raw = std::fs::read(&path).expect("read checkpoint");
        let vb = VarBuilder::from_buffered_safetensors(raw, DType::F32, &Device::Cpu)
            .expect("varbuilder");
        let err = match CandleLayaWeights::load(&vb, &cfg) {
            Err(e) => e,
            Ok(_) => panic!("缺键必须报错"),
        };
        let detail = err.error_detail();
        assert!(
            matches!(err, VecboostError::ModelLoadError(_)),
            "缺键必须 ModelLoadError，got {err:?}"
        );
        for missing in [
            "encoder.embeddings.tok_embeddings.weight",
            "head.layers.1.linear1.bias",
            "scorer.3.weight",
        ] {
            assert!(
                detail.contains(missing),
                "错误必须一次性列出全部缺失键（含 {missing}），detail={detail}"
            );
        }
    }

    #[test]
    fn test_weights_load_upcasts_f16_storage() {
        let cfg = tiny_config();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = synthetic_checkpoint(dir.path(), &[]);
        let raw = std::fs::read(&path).expect("read checkpoint");
        let vb = VarBuilder::from_buffered_safetensors(raw, DType::F32, &Device::Cpu)
            .expect("varbuilder");
        let weights = CandleLayaWeights::load(&vb, &cfg).expect("load weights");
        assert_eq!(weights.temperature.dtype(), DType::F32);
    }

    /// 前向 wire 契约：BatchTensors（onnx 管线同一 collate 产物）进出——
    /// logits [B, N] pad 位 -1e4、act [B, 2]、确定性。
    #[test]
    fn test_forward_contract_shapes_and_masked_pad() {
        let cfg = tiny_config();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = synthetic_checkpoint(dir.path(), &[]);
        let raw = std::fs::read(&path).expect("read checkpoint");
        let vb = VarBuilder::from_buffered_safetensors(raw, DType::F32, &Device::Cpu)
            .expect("varbuilder");
        let weights = CandleLayaWeights::load(&vb, &cfg).expect("load weights");

        // B=2、seq_len=6、max_markers=3：行 1 两 marker 无 pad（qtype=0），
        // 行 2 三 marker 且带行尾 pad（qtype=2）——混合题型按行对位
        let batch = BatchTensors {
            input_ids: vec![5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 0, 0],
            attention_mask: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            marker_pos: vec![1, 2, 0, 1, 2, 3],
            marker_mask: vec![true, true, false, true, true, true],
            qtype: vec![0, 2],
            batch_size: 2,
            seq_len: 6,
            max_markers: 3,
        };
        let (logits, h) = weights.forward_hidden(&batch).expect("forward hidden");
        assert_eq!(logits.dims(), &[2, 3], "logits 宽度必须为 [B, max_markers]");
        assert_eq!(h.dims(), &[2, 6, 8], "隐藏态必须为 [B, seq_len, hidden]");
        let logits_rows = logits.to_vec2::<f32>().expect("logits to vec");
        assert_eq!(
            logits_rows[0][2], MASKED_LOGIT,
            "pad marker 位（行 1 第 3 marker）必须为 masked_fill 填充值"
        );
        assert!(
            logits_rows[0][..2].iter().all(|v| v.is_finite())
                && logits_rows[1].iter().all(|v| v.is_finite()),
            "有效 marker 位 logits 必须有限"
        );

        // 生产 forward 只产 logits（act 无 wire 消费者，热路径不付其开销）；
        // act 头为独立组合方法
        let logits_only = weights.forward(&batch).expect("forward");
        assert_eq!(
            logits.to_vec2::<f32>().expect("logits"),
            logits_only.to_vec2::<f32>().expect("forward logits"),
            "forward 必须与 forward_hidden 的 logits 逐位一致"
        );
        let act = weights
            .forward_act(&logits, &h, &batch.marker_mask)
            .expect("forward act");
        assert_eq!(act.dims(), &[2, ACT_CLASSES], "act 头输出 [B, 2]");
        let act_rows = act.to_vec2::<f32>().expect("act to vec");
        assert!(act_rows.iter().flatten().all(|v| v.is_finite()));

        let logits_again = weights.forward(&batch).expect("forward again");
        assert_eq!(
            logits.to_vec2::<f32>().expect("logits"),
            logits_again.to_vec2::<f32>().expect("logits again"),
            "同输入前向必须确定"
        );
    }

    /// encoder/config.json 语义校验：num_attention_heads=0（请求期 usize
    /// 除零 panic 源）与不整除组合必须在加载期显性 ModelLoadError 点名字段。
    #[test]
    fn test_parse_encoder_config_rejects_invalid_head_geometry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let write_cfg = |hidden: usize, heads: usize| {
            let cfg = serde_json::json!({
                "vocab_size": 16, "hidden_size": hidden, "num_hidden_layers": 2,
                "num_attention_heads": heads, "intermediate_size": 4,
                "max_position_embeddings": 64, "layer_norm_eps": 1e-5,
                "pad_token_id": 0, "global_attn_every_n_layers": 3,
                "local_attention": 4,
                "rope_parameters": {
                    "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                }
            });
            let path = dir.path().join(format!("config_{hidden}_{heads}.json"));
            std::fs::write(&path, cfg.to_string()).expect("write config");
            path
        };
        let err = parse_encoder_config(&write_cfg(8, 0)).expect_err("0 头必须拒绝");
        assert!(
            matches!(err, VecboostError::ModelLoadError(_))
                && err.error_detail().contains("num_attention_heads"),
            "0 头必须 ModelLoadError 点名字段（把请求期除零 panic 变加载期显性失败），got {err:?}"
        );
        let err = parse_encoder_config(&write_cfg(0, 2)).expect_err("hidden=0 必须拒绝");
        assert!(
            err.error_detail().contains("hidden_size"),
            "hidden=0 必须点名 hidden_size，got {err:?}"
        );
        let err = parse_encoder_config(&write_cfg(10, 4)).expect_err("不整除必须拒绝");
        let detail = err.error_detail();
        assert!(
            detail.contains("10") && detail.contains("4") && detail.contains("divisible"),
            "不整除必须点名实际值，detail={detail}"
        );
        // 合法组合（整除）不受误伤
        parse_encoder_config(&write_cfg(8, 2)).expect("8/2 整除必须通过");
    }

    /// 加载期 tokenizer ↔ encoder config 词表交叉校验：tokenizer 词表宽超过
    /// 骨干词表（bundle 拿错 tokenizer）必须在 load 显性 ModelLoadError 点名
    /// 两个数值，而非首个决策请求期 embedding index_select InvalidIndex；
    /// 词表相等（合成 bundle 20/20）不得误伤。
    #[test]
    fn test_load_rejects_tokenizer_vocab_larger_than_encoder_vocab() {
        use super::test_support::write_bundle;
        use std::path::PathBuf;
        let config_for = |model_path: PathBuf| ModelConfig {
            name: "test-candle-vocab".to_string(),
            engine_type: crate::config::model::EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: ModelTask::Decision,
            quantized: false,
            decision_params: None,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        // 基线：词表相等（tokenizer 20 / encoder 20）合法加载
        let bundle = write_bundle(&dir.path().join("ok"));
        CandleDecisionEngine::load(&config_for(bundle)).expect("词表相等必须通过");
        // 错配：encoder 词表缩到 15，tokenizer 可产 id 15..19 越界
        let bundle = write_bundle(&dir.path().join("mismatched"));
        let config_path = bundle.join("encoder/config.json");
        let mut raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).expect("read config"))
                .expect("parse config");
        raw["vocab_size"] = serde_json::json!(15);
        std::fs::write(&config_path, raw.to_string()).expect("rewrite config");
        let err = match CandleDecisionEngine::load(&config_for(bundle)) {
            Err(e) => e,
            Ok(_) => panic!("词表错配必须拒绝加载"),
        };
        assert!(
            matches!(err, VecboostError::ModelLoadError(_)),
            "词表错配必须 ModelLoadError，got {err:?}"
        );
        let detail = err.error_detail();
        assert!(
            detail.contains("20") && detail.contains("15"),
            "错误必须点名 tokenizer 词表与 encoder vocab_size 两个数值，detail={detail}"
        );
    }
}
