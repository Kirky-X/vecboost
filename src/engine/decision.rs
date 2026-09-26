// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Laya 决策管线（文档 temp/laya-vecboost-feasibility.md §2.1/§4.2/§9.6）：
//! 预处理（序列构造 / marker_pos / qtype 编码）→ 5 张量 ONNX 推理
//! → 按题 softmax + per-cardinality 温度校准 → choice/score/noul 后处理。
//!
//! 本 mod 为 `pub(crate)`：对外唯一入口是
//! [`crate::engine::InferenceEngine::decide`]（经 `AnyEngine::Decision`）。
//!
//! 序列协议（§2.1）：
//! `[CLS] question [SEP] [MASK] option1 [MASK] option2 ... [SEP] state [SEP]`
//!
//! # P0 待校准项（与 receptron/laya 参考实现对照后固化，禁止漂移被静默接受）
//! - head 前缀文本格式（`"{qtype} question: {instructions}"`，§4.2 伪代码）
//! - marker 前置 option 拼接（`" " + opt`）
//! - noul marker 顺序（固定 `[false, true]`）
//! - score 等级数（domain 契约禁 score 携带 options，管线内固定 5 级）
//!
//! 以上均需在拿到真实 bundle 跑 P0 数值对照（概率差 <1e-4）后确认。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::config::model::{DeviceType, ModelConfig, ModelTask, Precision};
use crate::domain::{
    DecisionAnswer, DecisionAnswerBody, DecisionQuestion, DecisionRequest, DecisionResponse,
    QuestionType,
};
use crate::error::VecboostError;
use crate::utils::validator::input::echo;
use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::InferenceEngine;

/// head（qtype 前缀 + instructions）+ 全部 options（含各自 [MASK]）的 token 预算
/// （英文 checkpoint 口径，文档 §2.1；超限直接 InvalidInput，风险前置校验）
pub(crate) const HEAD_MAX_LEN: usize = 192;
/// state 部分 token 截断上限（文档 §2.1：截断后约 256）
pub(crate) const STATE_MAX_TOKENS: usize = 256;
/// noul 固定两 marker（[false, true] 顺序）
pub(crate) const NOUL_MARKERS: usize = 2;
/// score 等级数：domain 契约（`DecisionQuestion::validate`）禁止 score 携带
/// options，等级空间无法由请求定义，管线内固定 5 级（P0 待校准项）
pub(crate) const SCORE_LEVELS: usize = 5;
/// 校准温度钳制下界（文档 §9.6：加载时钳制 [0.5, 5]）
const MIN_TEMPERATURE: f32 = 0.5;
/// 校准温度钳制上界
const MAX_TEMPERATURE: f32 = 5.0;
/// 校准缺桶回退温度（文档 §9.6：样本 <10 条回退 1.2；加载缺桶同口径）
pub(crate) const DEFAULT_TEMPERATURE: f32 = 1.2;

/// bundle 内模型文件探测顺序（任务协议：model.onnx → model_quantized.onnx
/// → laya.onnx → laya_int8.onnx）
const MODEL_CANDIDATES: [&str; 4] = [
    "model.onnx",
    "model_quantized.onnx",
    "laya.onnx",
    "laya_int8.onnx",
];

/// 问题类型 → 模型输入 qtype 编码（§2.2：0=choice, 1=score, 2=noul）
pub(crate) fn qtype_code(qtype: &QuestionType) -> i64 {
    match qtype {
        QuestionType::Choice => 0,
        QuestionType::Score => 1,
        QuestionType::Noul => 2,
    }
}

/// 问题类型 → head 文本中的类型词（与 wire 契约同 lowercase 口径）
fn qtype_text(qtype: &QuestionType) -> &'static str {
    match qtype {
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
        QuestionType::Noul => "noul",
    }
}

/// 文本编码为 token id 序列（`add_special_tokens=false`——特殊 token 由
/// 管线手工拼接，见 [`build_question_row`]）
fn encode_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, VecboostError> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| VecboostError::TokenizationError(e.to_string()))?
        .get_ids()
        .iter()
        .map(|&id| id as i64)
        .collect())
}

/// 查特殊 token id：ModernBERT/BERT 协议依赖 [CLS]/[SEP]/[MASK]，
/// 词表缺失即 tokenizer 资产与协议不符，显性报错
fn special_id(tokenizer: &Tokenizer, token: &str) -> Result<i64, VecboostError> {
    tokenizer
        .token_to_id(token)
        .map(|id| id as i64)
        .ok_or_else(|| VecboostError::TokenizationError(format!("tokenizer vocab missing {token}")))
}

/// state 文本化：字符串原样借用（零拷贝），其余 serde_json 紧凑序列化（§4.2）
pub(crate) fn state_text(state: &serde_json::Value) -> std::borrow::Cow<'_, str> {
    match state {
        serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s),
        other => std::borrow::Cow::Owned(other.to_string()),
    }
}

/// 单题预处理产物：input_ids 已含 [CLS]/[SEP]/[MASK] 手工拼接
/// （encode(add_special_tokens=false)，特殊 token 由本管线自管）。
/// attention_mask 恒为行前缀全 1（无行内 padding），由 [`collate_batch`]
/// 直接生成，不落字段。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QuestionRow {
    pub input_ids: Vec<i64>,
    /// 每个选项 [MASK] 在 input_ids 中的位置（与 options/marker 顺序对齐）
    pub marker_pos: Vec<i64>,
}

/// score/noul 固定 marker 的预编码 id 与序列协议特殊 token id（只依赖
/// tokenizer、请求间恒定）。加载期解析/编码一次，热路径每题零重复
/// encode 与词表查询（choice 的 marker 文本为请求侧 options，请求相关，
/// 无法预编码）。
#[derive(Debug, Clone)]
pub(crate) struct FixedMarkers {
    pub(crate) cls: i64,
    pub(crate) sep: i64,
    pub(crate) mask: i64,
    score: Vec<Vec<i64>>,
    noul: Vec<Vec<i64>>,
}

impl FixedMarkers {
    pub(crate) fn new(tokenizer: &Tokenizer) -> Result<Self, VecboostError> {
        let cls = special_id(tokenizer, "[CLS]")?;
        let sep = special_id(tokenizer, "[SEP]")?;
        let mask = special_id(tokenizer, "[MASK]")?;
        // score=等级索引文本；noul=固定 [false, true]（P0 待校准项）
        let score = (0..SCORE_LEVELS)
            .map(|i| encode_ids(tokenizer, &format!(" {i}")))
            .collect::<Result<Vec<_>, _>>()?;
        let noul = ["false", "true"]
            .iter()
            .map(|t| encode_ids(tokenizer, &format!(" {t}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            cls,
            sep,
            mask,
            score,
            noul,
        })
    }

    #[cfg(test)]
    fn score_len(&self) -> usize {
        self.score.len()
    }

    #[cfg(test)]
    fn noul_len(&self) -> usize {
        self.noul.len()
    }
}

/// 单题序列构造（§4.2）：
/// `[CLS] + head + [SEP]`，随后逐选项 `[MASK] + option_tokens`，
/// 末尾 `[SEP] + state(截断) + [SEP]`。
///
/// `state_ids` 由调用方对整个请求编码一次后传入（同一 state 的多题共享，
/// 消除逐题重复 tokenize）；本函数内截断到 `state_max_tokens`（防御性，
/// 未截断的调用方也安全）。
///
/// head + 全部 options（含各自 [MASK]）超过 `head_max_len` 时报
/// `InvalidInput`（§2.1 风险前置校验）。
pub(crate) fn build_question_row(
    tokenizer: &Tokenizer,
    question: &DecisionQuestion,
    state_ids: &[i64],
    fixed: &FixedMarkers,
    head_max_len: usize,
    state_max_tokens: usize,
) -> Result<QuestionRow, VecboostError> {
    let cls_id = fixed.cls;
    let sep_id = fixed.sep;
    let mask_id = fixed.mask;

    let head_text = format!(
        "{} question: {}",
        qtype_text(&question.qtype),
        question.instructions
    );
    let head_ids = encode_ids(tokenizer, &head_text)?;

    // choice 的 marker 文本为请求侧 options（借用不深拷贝，现场编码）；
    // 逐 option 累积预算，超 head_max_len 即提前拒绝（超预算请求不白跑
    // 剩余 option 的 tokenize，拒绝路径受 128KB validate 上界封顶）
    let mut choice_marker_ids: Vec<Vec<i64>> = Vec::new();
    let marker_ids: Vec<&[i64]> = match question.qtype {
        QuestionType::Choice => {
            if question.options.is_empty() {
                return Err(VecboostError::invalid_input(format!(
                    "choice question {} requires at least one option",
                    echo(&question.name)
                )));
            }
            let mut budget = head_ids.len();
            for opt in &question.options {
                let ids = encode_ids(tokenizer, &format!(" {}", opt.as_str()))?;
                budget += 1 + ids.len();
                if budget > head_max_len {
                    return Err(VecboostError::invalid_input(format!(
                        "question {} head+options exceeds head_max_len {head_max_len}",
                        echo(&question.name)
                    )));
                }
                choice_marker_ids.push(ids);
            }
            choice_marker_ids.iter().map(|v| v.as_slice()).collect()
        }
        QuestionType::Score => fixed.score.iter().map(|v| v.as_slice()).collect(),
        QuestionType::Noul => fixed.noul.iter().map(|v| v.as_slice()).collect(),
    };

    let options_budget: usize = marker_ids.iter().map(|ids| 1 + ids.len()).sum();
    let head_budget = head_ids.len() + options_budget;
    if head_budget > head_max_len {
        return Err(VecboostError::invalid_input(format!(
            "question {} head+options requires {head_budget} tokens > head_max_len {head_max_len}",
            echo(&question.name)
        )));
    }

    let state_ids = &state_ids[..state_ids.len().min(state_max_tokens)];

    let mut input_ids =
        Vec::with_capacity(1 + head_ids.len() + 1 + options_budget + 1 + state_ids.len() + 1);
    let mut marker_pos = Vec::with_capacity(marker_ids.len());
    input_ids.push(cls_id);
    input_ids.extend_from_slice(&head_ids);
    input_ids.push(sep_id);
    for ids in &marker_ids {
        marker_pos.push(input_ids.len() as i64);
        input_ids.push(mask_id);
        input_ids.extend_from_slice(ids);
    }
    input_ids.push(sep_id);
    input_ids.extend_from_slice(state_ids);
    input_ids.push(sep_id);
    Ok(QuestionRow {
        input_ids,
        marker_pos,
    })
}

/// collate 后的 5 张量平面 buffer（行主序，形状由
/// `[batch_size, seq_len]` / `[batch_size, max_markers]` / `[batch_size]` 给出）
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BatchTensors {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_markers: usize,
}

/// 按批 collate：pad 到 batch 内最长序列（pad id=0、mask=0）；
/// marker_pos 二维 [B,N]（N=batch 内最大选项数，pad 位 0 且 marker_mask=false）。
/// 多题混合 choice/score/noul 时三张量按行对位（行内 marker 数可不同）。
pub(crate) fn collate_batch(
    rows: &[QuestionRow],
    qtype_codes: &[i64],
) -> Result<BatchTensors, VecboostError> {
    if rows.is_empty() {
        return Err(VecboostError::invalid_input(
            "collate requires at least one question row".to_string(),
        ));
    }
    if rows.len() != qtype_codes.len() {
        return Err(VecboostError::invalid_input(format!(
            "collate rows/qtype length mismatch: {} rows vs {} qtype codes",
            rows.len(),
            qtype_codes.len()
        )));
    }
    let seq_len = rows
        .iter()
        .fold(0usize, |acc, r| acc.max(r.input_ids.len()));
    let max_markers = rows
        .iter()
        .fold(0usize, |acc, r| acc.max(r.marker_pos.len()));

    let mut input_ids = vec![0i64; rows.len() * seq_len];
    let mut attention_mask = vec![0i64; rows.len() * seq_len];
    let mut marker_pos = vec![0i64; rows.len() * max_markers];
    let mut marker_mask = vec![false; rows.len() * max_markers];

    for (b, row) in rows.iter().enumerate() {
        input_ids[b * seq_len..b * seq_len + row.input_ids.len()].copy_from_slice(&row.input_ids);
        // 行前缀全 1（无行内 padding）、pad 位 0——attention_mask 在此直接
        // 生成，QuestionRow 不携带恒全 1 的冗余字段
        let row_mask = &mut attention_mask[b * seq_len..b * seq_len + row.input_ids.len()];
        row_mask.fill(1);
        for (n, &pos) in row.marker_pos.iter().enumerate() {
            marker_pos[b * max_markers + n] = pos;
            marker_mask[b * max_markers + n] = true;
        }
    }
    Ok(BatchTensors {
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype: qtype_codes.to_vec(),
        batch_size: rows.len(),
        seq_len,
        max_markers,
    })
}

/// 数值稳定 softmax（减 max）+ 温度缩放：softmax(logits / T)。
/// 温度增大分布更平（单调性）；T ≤ 0（含 NaN）显式拒绝。
pub(crate) fn softmax_with_temperature(
    logits: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, VecboostError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(VecboostError::invalid_input(format!(
            "temperature must be a positive finite number, got {temperature}"
        )));
    }
    if logits.is_empty() {
        return Err(VecboostError::invalid_input(
            "softmax requires non-empty logits".to_string(),
        ));
    }
    if logits.iter().any(|l| !l.is_finite()) {
        return Err(VecboostError::inference_error(
            "logits contain non-finite values; refusing to emit NaN probabilities".to_string(),
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    // 减 max 后最大项 exp(0)=1，sum ∈ [1, len] 恒有限，不会除零；就地归一化
    for e in &mut exps {
        *e /= sum;
    }
    Ok(exps)
}

fn clamp_temperature(t: f32) -> f32 {
    t.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE)
}

/// per-cardinality 温度校准表（文档 §9.6：logit 上叠加温度再 softmax，
/// 未校准概率过度自信——ECE 0.466 vs 校准后 0.081）。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TemperatureCalibration {
    temperatures: HashMap<usize, f32>,
}

impl TemperatureCalibration {
    /// 从 bundle 目录读 `laya_config.json` 的 `temperature_by_options`
    /// 分桶温度（cardinality → temperature）。
    ///
    /// - 文件缺失：`log::warn` 显性记录 + 空表兜底（缺桶全部回退
    ///   [`DEFAULT_TEMPERATURE`]），不静默（规则 11）
    /// - 文件存在但 JSON 损坏 / 字段形态不符：`ModelFileCorrupted` 显性报错
    /// - 加载即钳制 [0.5, 5]（文档 §9.6）
    ///
    /// # P0 待校准项
    /// 字段名 `temperature_by_options` 以真实 bundle 为准，P0 对照后固化。
    pub(crate) fn from_bundle(bundle_dir: &Path) -> Result<Self, VecboostError> {
        let path = bundle_dir.join("laya_config.json");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(_) => {
                log::warn!(
                    "calibration file missing at {}: probabilities will be overconfident \
                     (uncalibrated ECE 0.466 vs 0.081 calibrated, model card); \
                     falling back to temperature {DEFAULT_TEMPERATURE} for all cardinalities",
                    path.display()
                );
                return Ok(Self {
                    temperatures: HashMap::new(),
                });
            }
        };
        let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            VecboostError::model_file_corrupted(format!("failed to parse {}: {e}", path.display()))
        })?;
        let Some(table) = value.get("temperature_by_options") else {
            log::warn!(
                "{} has no temperature_by_options field; falling back to \
                 temperature {DEFAULT_TEMPERATURE} for all cardinalities",
                path.display()
            );
            return Ok(Self {
                temperatures: HashMap::new(),
            });
        };
        let map = table.as_object().ok_or_else(|| {
            VecboostError::model_file_corrupted(format!(
                "{}: temperature_by_options must be an object of cardinality → temperature, \
                 got {}",
                path.display(),
                table
            ))
        })?;
        let mut temperatures = HashMap::with_capacity(map.len());
        for (key, value) in map {
            let cardinality: usize = key.parse().map_err(|_| {
                VecboostError::model_file_corrupted(format!(
                    "{:?}: non-integer cardinality key {key:?} in temperature_by_options",
                    path.display()
                ))
            })?;
            let temperature = value.as_f64().ok_or_else(|| {
                VecboostError::model_file_corrupted(format!(
                    "{:?}: temperature for cardinality {key} is not a number",
                    path.display()
                ))
            })? as f32;
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(VecboostError::model_file_corrupted(format!(
                    "{:?}: temperature for cardinality {key} must be positive finite, \
                     got {temperature}",
                    path.display()
                )));
            }
            temperatures.insert(cardinality, clamp_temperature(temperature));
        }
        Ok(Self { temperatures })
    }

    /// 查指定 cardinality（该题 marker 数）的温度；缺桶回退 1.2。
    pub(crate) fn temperature_for(&self, cardinality: usize) -> f32 {
        self.temperatures
            .get(&cardinality)
            .copied()
            .unwrap_or(DEFAULT_TEMPERATURE)
    }
}

/// choice 后处理：argmax + 完整概率表（选项名 → 校准后概率）。
/// logits 长度必须与 options 一致，重复选项名显式拒绝（静默合并即丢概率）。
/// 错误回显统一走公共 `echo`（64 字符截断+控制字符字面量化，防日志注入）
pub(crate) fn choice_answer(
    question_name: &str,
    options: &[String],
    logits: &[f32],
    temperature: f32,
) -> Result<DecisionAnswer, VecboostError> {
    let name = echo(question_name);
    if options.is_empty() {
        return Err(VecboostError::invalid_input(format!(
            "choice question {name} requires at least one option"
        )));
    }
    if options.len() != logits.len() {
        return Err(VecboostError::inference_error(format!(
            "choice question {name}: got {} logits for {} options",
            logits.len(),
            options.len()
        )));
    }
    let mut seen = HashSet::with_capacity(options.len());
    for option in options {
        if !seen.insert(option.as_str()) {
            return Err(VecboostError::invalid_input(format!(
                "choice question {name} has duplicate option {:?}: \
                 probability table keys would silently merge",
                echo(option)
            )));
        }
    }
    let probs = softmax_with_temperature(logits, temperature)?;
    let mut best_idx = 0;
    let mut best_p = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best_idx = i;
        }
    }
    let probabilities: BTreeMap<String, f32> = options.iter().cloned().zip(probs).collect();
    Ok(DecisionAnswer {
        question: question_name.to_string(),
        answer: DecisionAnswerBody::Choice {
            index: best_idx,
            option: options[best_idx].clone(),
            probabilities,
        },
    })
}

/// score 后处理：期望等级 Σ(i·p_i) + 完整分布（等级索引字符串 → 概率，
/// 与训练侧 gold 分布键格式一致，0 起）。
pub(crate) fn score_answer(
    question_name: &str,
    logits: &[f32],
    temperature: f32,
) -> Result<DecisionAnswer, VecboostError> {
    let probs = softmax_with_temperature(logits, temperature)?;
    let expected: f32 = probs
        .iter()
        .enumerate()
        .map(|(level, &p)| level as f32 * p)
        .sum();
    let distribution: BTreeMap<String, f32> = probs
        .iter()
        .enumerate()
        .map(|(level, &p)| (level.to_string(), p))
        .collect();
    Ok(DecisionAnswer {
        question: question_name.to_string(),
        answer: DecisionAnswerBody::Score {
            expected,
            distribution,
        },
    })
}

/// noul 后处理：固定 [false, true] 两 marker，取 P(true)。
pub(crate) fn noul_answer(
    question_name: &str,
    logits: &[f32],
    temperature: f32,
) -> Result<DecisionAnswer, VecboostError> {
    if logits.len() != NOUL_MARKERS {
        return Err(VecboostError::inference_error(format!(
            "noul question {}: expected exactly {NOUL_MARKERS} logits \
             ([false, true] marker order), got {}",
            echo(question_name),
            logits.len()
        )));
    }
    let probs = softmax_with_temperature(logits, temperature)?;
    Ok(DecisionAnswer {
        question: question_name.to_string(),
        answer: DecisionAnswerBody::Noul { p_true: probs[1] },
    })
}

/// 按题型分发后处理
pub(crate) fn answer_for_question(
    question: &DecisionQuestion,
    logits: &[f32],
    temperature: f32,
) -> Result<DecisionAnswer, VecboostError> {
    match question.qtype {
        QuestionType::Choice => {
            choice_answer(&question.name, &question.options, logits, temperature)
        }
        QuestionType::Score => score_answer(&question.name, logits, temperature),
        QuestionType::Noul => noul_answer(&question.name, logits, temperature),
    }
}

/// Laya 决策管线：bundle 自持（模型 + tokenizer + 可选校准表）+ ort Session。
pub(crate) struct DecisionPipeline {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    calibration: TemperatureCalibration,
    /// score/noul 固定 marker 的预编码 id（加载期一次，热路径复用）
    fixed_markers: FixedMarkers,
    precision: Precision,
    /// 已探明的本地 bundle 目录（try_fallback_to_cpu 重建 CPU Session 用，
    /// 无网络依赖）
    bundle_dir: PathBuf,
    fallback_triggered: bool,
    fallback_lock: Arc<Mutex<()>>,
}

/// bundle 内模型文件探测（顺序见 [`MODEL_CANDIDATES`]）；
/// 全 miss 报 `ModelLoadError` 含尝试路径清单（onnx_engine.rs 同款显性化）
fn probe_model_file(bundle_dir: &Path) -> Result<PathBuf, VecboostError> {
    let mut tried = Vec::with_capacity(MODEL_CANDIDATES.len());
    for name in MODEL_CANDIDATES {
        let candidate = bundle_dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(candidate.display().to_string());
    }
    Err(VecboostError::ModelLoadError(format!(
        "No Laya ONNX model found in {}; tried: {}",
        bundle_dir.display(),
        tried.join(", ")
    )))
}

/// tokenizer 探测：显式路径 → bundle 根 tokenizer.json → tokenizer/ 子目录；
/// 全 miss 报含尝试路径清单的 `ModelLoadError`
fn probe_tokenizer_file(
    bundle_dir: &Path,
    explicit: Option<&Path>,
) -> Result<PathBuf, VecboostError> {
    let mut candidates = Vec::with_capacity(3);
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    }
    candidates.push(bundle_dir.join("tokenizer.json"));
    candidates.push(bundle_dir.join("tokenizer").join("tokenizer.json"));
    let mut tried = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
        tried.push(candidate.display().to_string());
    }
    Err(VecboostError::ModelLoadError(format!(
        "Tokenizer not found for decision bundle {}; tried: {}",
        bundle_dir.display(),
        tried.join(", ")
    )))
}

/// 探测命中的模型文件名 → 对外精度标签：`*_quantized`/`*_int8` 候选命中时
/// 报 Int8，其余 Fp32（supports_mixed_precision 恒 false 已诚实，仅加载期
/// 标签与探测结果一致，避免 quantized/int8 bundle 对外失真报 Fp32）
fn precision_for_model_file(model_file: &Path) -> Precision {
    let name = model_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if name.contains("quantized") || name.contains("int8") {
        Precision::Int8
    } else {
        Precision::Fp32
    }
}

/// Session 构建：Level3 图优化 + intra threads + CUDA EP 分支
/// （onnx_engine.rs:128-159 同模式）
fn build_session(model_file: &Path, device: &DeviceType) -> Result<Session, VecboostError> {
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    log::info!("Initializing ONNX Runtime session for decision pipeline...");
    let builder = Session::builder()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_intra_threads(num_threads)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

    let mut session = if *device == DeviceType::Cuda {
        log::info!("Attempting to configure CUDA execution provider for decision pipeline");
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!(
                "CUDA execution provider requires the cuda feature flag; \
                 using CPU execution provider for decision pipeline"
            );
            builder
        }
    } else {
        builder
    };

    session
        .commit_from_file(model_file)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))
}

impl DecisionPipeline {
    /// 从 bundle 目录加载：模型按 [`MODEL_CANDIDATES`] 探测，tokenizer 按
    /// `config.tokenizer_path` 显式 → bundle 根 `tokenizer.json` →
    /// `tokenizer/tokenizer.json` 子目录探测（装配侧 `tokenizer_path: None`
    /// 不影响 fallback 链）；全 miss 报 `ModelLoadError` 含尝试路径清单。
    /// 校准表缺失 warn + 空表兜底；损坏报 `ModelFileCorrupted`。
    ///
    /// # 完整性校验覆盖边界（威胁模型声明）
    /// `config.model_sha256` 仅校验探测命中的主模型文件。bundle 内
    /// tokenizer.json 与 laya_config.json 温度校准表**不受 sha256 校验**——
    /// 它们与本管线同目录读取，威胁模型将 bundle 目录视为可信本地资产；
    /// 能写 bundle 目录的攻击者无需替换模型即可经篡改校准温度（分布尖锐化/
    /// 操纵置信呈现）或词表（分词漂移）改变下游语义。部署上以目录权限而非
    /// 文件哈希作为该资产的边界；bundle 清单化校验待 config 契约扩展任务组。
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        let bundle_dir = config.model_path.clone();
        if !bundle_dir.is_dir() {
            return Err(VecboostError::ModelLoadError(format!(
                "decision bundle path is not a directory: {}",
                bundle_dir.display()
            )));
        }
        let model_file = probe_model_file(&bundle_dir)?;

        if let Some(ref expected_hash) = config.model_sha256 {
            log::info!("Verifying decision model file SHA256 hash...");
            let is_valid =
                crate::utils::hash::verify_sha256(&model_file, expected_hash).map_err(|e| {
                    VecboostError::ModelLoadError(format!("Failed to verify SHA256: {e}"))
                })?;
            if !is_valid {
                return Err(VecboostError::ModelLoadError(format!(
                    "Model file SHA256 verification failed. Expected: {expected_hash}, File: {:?}",
                    model_file
                )));
            }
        }

        let tokenizer_file = probe_tokenizer_file(&bundle_dir, config.tokenizer_path.as_deref())?;
        // bundle tokenizer.json 常自带 truncation/padding 配置（如 MiniLM 的
        // fixed-128 padding）——决策协议自管特殊 token 拼接与 collate 填充，
        // 两者必须清除，否则序列被静默 pad/截断（协议漂移）
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
        let session = build_session(&model_file, &config.device)?;
        let calibration = TemperatureCalibration::from_bundle(&bundle_dir)?;
        let precision = precision_for_model_file(&model_file);

        log::info!(
            "Decision pipeline initialized: bundle={}, model={:?}, precision={:?}, calibration_buckets={}",
            bundle_dir.display(),
            model_file,
            precision,
            calibration.temperatures.len()
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            calibration,
            fixed_markers,
            precision,
            bundle_dir,
            fallback_triggered: false,
            fallback_lock: Arc::new(Mutex::new(())),
        })
    }

    /// 决策管线本体：校验 → 逐题预处理 → collate → 5 张量推理 →
    /// 按行 softmax+温度校准 → 三类后处理。
    ///
    /// logits **绝不经过 `l2_normalize_in_place`**——embedding 出口的
    /// 归一化契约不适用于决策 logits（docDrift：对 logits 归一化会让
    /// softmax 前的概率分布整体塌缩到单位球面，校准与概率语义全毁）。
    pub(crate) fn decision(
        &self,
        req: &DecisionRequest,
    ) -> Result<DecisionResponse, VecboostError> {
        let started = std::time::Instant::now();
        // trait 契约：实现方须假定请求已过 validate 或自行调用——
        // 管线自行调用（crate 内直调场景同样被输入防线覆盖）
        req.validate()?;

        // 同一请求内 state 全题共享：只 encode 一次（32 题上界下消除 31 次
        // 重复 tokenize）。取前 256 token 由 build_question_row 内防御性切片
        // 完成——tokenizers 0.23.2 的 encode 无编码期截断（normalize→
        // pre_tokenize→tokenize 全量完成后 post_process 才丢弃尾部），编码期
        // 截断与后置切片逐 token 等价；64KB 上界的全量 tokenize 为毫秒级、
        // 相对单次推理非瓶颈，如需消除须字符级前缀粗剪（预剪点须对齐
        // pre-token 边界才保证等价），待真实负载数据立项后再做
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
                HEAD_MAX_LEN,
                STATE_MAX_TOKENS,
            )?);
            qtype_codes.push(qtype_code(&question.qtype));
        }
        let batch = collate_batch(&rows, &qtype_codes)?;

        let input_ids = Array2::from_shape_vec((batch.batch_size, batch.seq_len), batch.input_ids)
            .map_err(|e| VecboostError::InferenceError(format!("input_ids shape: {e}")))?;
        let attention_mask =
            Array2::from_shape_vec((batch.batch_size, batch.seq_len), batch.attention_mask)
                .map_err(|e| VecboostError::InferenceError(format!("attention_mask shape: {e}")))?;
        let marker_pos =
            Array2::from_shape_vec((batch.batch_size, batch.max_markers), batch.marker_pos)
                .map_err(|e| VecboostError::InferenceError(format!("marker_pos shape: {e}")))?;
        let marker_mask =
            Array2::from_shape_vec((batch.batch_size, batch.max_markers), batch.marker_mask)
                .map_err(|e| VecboostError::InferenceError(format!("marker_mask shape: {e}")))?;
        let qtype = ndarray::Array1::from(batch.qtype);

        // marker_mask dtype 为 bool（§2.2）。若与真实导出图不符，
        // P0 数值对照时改 i64 并在此留注释
        let logits = {
            let mut session_guard = self
                .session
                .lock()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            let outputs = session_guard
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(input_ids.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "attention_mask" => Tensor::from_array(attention_mask.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "marker_pos" => Tensor::from_array(marker_pos.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "marker_mask" => Tensor::from_array(marker_mask.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                    "qtype" => Tensor::from_array(qtype.into_dyn())
                        .map_err(|e| VecboostError::InferenceError(e.to_string()))?,
                ])
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?;
            // get 而非 Index：Index 缺名时 panic（ort output.rs:189-192），
            // 与全文件显性失败口径不符——缺名属 bundle 模型资产与协议不符，
            // 显性报错并列出实际输出名清单
            let logits_value = outputs.get("logits").ok_or_else(|| {
                let names: Vec<&str> = outputs.iter().map(|(k, _)| k).collect();
                VecboostError::InferenceError(format!(
                    "decision model has no `logits` output; actual outputs: {names:?}"
                ))
            })?;
            logits_value
                .try_extract_array::<f32>()
                .map_err(|e| VecboostError::InferenceError(e.to_string()))?
                .to_owned()
        };

        let logits_shape = logits.shape().to_vec();
        // §2.2 dense gather 协议：logits 宽度恰为 [B, N]（N=batch 内最大
        // marker 数）。等值校验一次到位——宽度异常偏大（如误导出的
        // [B, seq_len] 图）在此显性报错，而非静默取前 N 个 logit 错答
        if logits.ndim() != 2
            || logits_shape[0] != batch.batch_size
            || logits_shape[1] != batch.max_markers
        {
            return Err(VecboostError::InferenceError(format!(
                "unexpected logits shape {logits_shape:?}, expected [{}, {}]",
                batch.batch_size, batch.max_markers
            )));
        }
        let logits_2d = logits
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| VecboostError::InferenceError(format!("logits dims: {e}")))?;

        let mut answers = Vec::with_capacity(batch.batch_size);
        for (b, question) in req.questions.iter().enumerate() {
            let marker_count = rows[b].marker_pos.len();
            // [B,N] 行主序 C-contiguous：行前缀切片必连续，零分配借用；
            // 非连续属数组构造异常，显性报错而非 panic
            let row = logits_2d.row(b);
            let row_view = row.slice(ndarray::s![..marker_count]);
            let row_logits = row_view.to_slice().ok_or_else(|| {
                VecboostError::InferenceError("logits row slice is not contiguous".to_string())
            })?;
            let temperature = self.calibration.temperature_for(marker_count);
            answers.push(answer_for_question(question, row_logits, temperature)?);
        }
        let elapsed = started.elapsed();
        // 埋点：决策链路调用时延直方图（histogram _count 即调用计数）与
        // 批大小。collector 未设置时零开销跳过。
        // Stage 三值豁免口径：Stage 枚举固定 tokenize/inference/pool（指标
        // 标签稳定性），语义属 embedding 管线分阶段；决策管线不强行映射，
        // take_stage_snapshot 恒 None，观测走本处的独立 decision 指标
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
}

#[async_trait::async_trait]
impl InferenceEngine for DecisionPipeline {
    /// 决策主链路唯一 trait 入口：委托固有方法 [`DecisionPipeline::decision`]。
    /// 漏覆盖则继承 trait 默认 `Err(UnsupportedTask)`——`/api/1/decisions`
    /// 恒 400 且服务层经 trait 调用走不到管线（全部离线测试盲区）。
    /// 固有方法名为 `decision()`，与 trait 方法无同名遮蔽，方法解析自然进管线。
    fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
        self.decision(req)
    }

    /// 决策引擎不产向量（诚实语义）：task=decision 时 embedding/rerank
    /// 端点以 UnsupportedTask 400 显性拒绝
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "decision pipeline does not produce embeddings".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "decision pipeline does not produce embeddings".to_string(),
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

    fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        task == ModelTask::Decision
    }

    /// 用已探明的本地 bundle 路径重建 CPU Session（onnx_engine.rs:419-471
    /// 同模式，无 HF 网络依赖）
    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        let _lock = self.fallback_lock.lock().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to acquire fallback lock: {e}"))
        })?;
        // 双重检查：获取锁后再次确认未降级
        if self.fallback_triggered {
            return Ok(());
        }
        log::info!("Attempting fallback to CPU for decision pipeline");
        let model_file = probe_model_file(&self.bundle_dir)?;
        let session = build_session(&model_file, &DeviceType::Cpu)?;
        let mut session_guard = self
            .session
            .lock()
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        *session_guard = session;
        drop(session_guard);
        // 与加载路径单一事实源：降级复用同一 bundle（可能为 quantized/int8 图），
        // 标签按探测文件名映射，不得硬编码 Fp32 失真（onnx_engine 降级真下载
        // Fp32 model.onnx，其硬编码在那边语义成立，此处不同）
        self.precision = precision_for_model_file(&model_file);
        // 置位于 session 替换成功之后：probe/build 失败时状态位保持 false，
        // 后续 OOM 降级尝试不会被双重检查恒短路掩盖
        self.fallback_triggered = true;
        log::info!("Successfully fell back to CPU for decision pipeline");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试：接口驱动（TDD red 阶段先行）。测试 tokenizer 资产按
// models/all-MiniLM-L6-v2/tokenizer.json 存在性守卫 SKIP（离线硬约束）。
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 本地 tokenizer 资产守卫：缺失时 SKIP（不硬失败——离线约束）。
    /// 与生产 load 同款清洗：资产自带的 truncation/padding 会污染协议
    /// （MiniLM tokenizer.json 为 fixed-128 padding，不清除则每条序列 pad 到 128）
    fn mini_lm_tokenizer() -> Option<Tokenizer> {
        let path = Path::new("models/all-MiniLM-L6-v2/tokenizer.json");
        if !path.exists() {
            eprintln!("Skipping test: tokenizer asset not found at {path:?}");
            return None;
        }
        let mut tokenizer = match Tokenizer::from_file(path) {
            Ok(t) => t,
            Err(e) => panic!("本地 tokenizer 资产必须可解析: {e}"),
        };
        tokenizer.with_truncation(None).expect("clear truncation");
        tokenizer.with_padding(None);
        Some(tokenizer)
    }

    fn choice_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "destination".to_string(),
            qtype: QuestionType::Choice,
            instructions: "pick one".to_string(),
            options: vec!["beach".to_string(), "mountain".to_string()],
        }
    }

    fn score_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "urgency".to_string(),
            qtype: QuestionType::Score,
            instructions: "rate urgency".to_string(),
            options: vec![],
        }
    }

    fn noul_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "churn_risk".to_string(),
            qtype: QuestionType::Noul,
            instructions: "state your p(true)".to_string(),
            options: vec![],
        }
    }

    /// 与生产 decision() 同口径的行输入：state 编码一次 + 加载期 FixedMarkers
    fn row_inputs(tok: &Tokenizer, state: &str) -> (Vec<i64>, FixedMarkers) {
        let state_ids = encode_ids(tok, state).expect("state ids");
        let fixed = FixedMarkers::new(tok).expect("fixed markers");
        (state_ids, fixed)
    }

    // ── qtype 编码 ──

    #[test]
    fn test_qtype_codes_match_wire_contract() {
        // §2.2：0=choice, 1=score, 2=noul——编码漂移即推理全错
        assert_eq!(qtype_code(&QuestionType::Choice), 0);
        assert_eq!(qtype_code(&QuestionType::Score), 1);
        assert_eq!(qtype_code(&QuestionType::Noul), 2);
    }

    // ── ① 预处理：序列协议结构 ──

    #[test]
    fn test_build_question_row_protocol_layout() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        let q = choice_q();
        let (state_ids, fixed) = row_inputs(&tok, "billed twice");
        let row = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .expect("row");

        let cls_id = tok.token_to_id("[CLS]").expect("[CLS] in vocab") as i64;
        let sep_id = tok.token_to_id("[SEP]").expect("[SEP] in vocab") as i64;
        let mask_id = tok.token_to_id("[MASK]").expect("[MASK] in vocab") as i64;

        // 期望序列：[CLS] head [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
        // （head/option/state 均 encode(add_special_tokens=false)）
        let head_ids: Vec<i64> = tok
            .encode("choice question: pick one", false)
            .expect("head encode")
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();
        let opt0_ids: Vec<i64> = tok
            .encode(" beach", false)
            .expect("opt0")
            .get_ids()
            .to_vec()
            .iter()
            .map(|&i| i as i64)
            .collect();
        let opt1_ids: Vec<i64> = tok
            .encode(" mountain", false)
            .expect("opt1")
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();
        let expected_state_ids: Vec<i64> = tok
            .encode("billed twice", false)
            .expect("state encode")
            .get_ids()
            .iter()
            .map(|&i| i as i64)
            .collect();

        let mut expected = vec![cls_id];
        expected.extend(&head_ids);
        expected.push(sep_id);
        expected.push(mask_id);
        expected.extend(&opt0_ids);
        expected.push(mask_id);
        expected.extend(&opt1_ids);
        expected.push(sep_id);
        expected.extend(&expected_state_ids);
        expected.push(sep_id);

        assert_eq!(row.input_ids, expected, "序列协议布局必须精确匹配");
        // marker_pos 指向两个 [MASK]
        assert_eq!(row.marker_pos.len(), 2);
        assert_eq!(row.marker_pos[0] as usize, 1 + head_ids.len() + 1);
        assert_eq!(
            row.marker_pos[1] as usize,
            1 + head_ids.len() + 1 + 1 + opt0_ids.len()
        );
        for &p in &row.marker_pos {
            assert_eq!(
                row.input_ids[p as usize], mask_id,
                "最强协议断言：每个 marker_pos 处必为 [MASK]"
            );
        }
    }

    #[test]
    fn test_build_question_row_state_string_used_verbatim() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // state 字符串原样编码，不做 JSON 引号包装
        let (state_ids, fixed) = row_inputs(&tok, "hello world");
        let row = build_question_row(
            &tok,
            &choice_q(),
            &state_ids,
            &fixed,
            HEAD_MAX_LEN,
            STATE_MAX_TOKENS,
        )
        .expect("row");
        let tail =
            &row.input_ids[row.input_ids.len() - state_ids.len() - 1..row.input_ids.len() - 1];
        assert_eq!(tail, &state_ids[..], "字符串 state 必须原文进入序列尾部");
    }

    #[test]
    fn test_build_question_row_non_string_state_compact_json() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // 非字符串 state 走 serde_json 紧凑序列化（无空格）
        let state_json = serde_json::json!({"topic": "vacation"});
        let compact = state_json.to_string();
        assert!(
            !compact.contains(' '),
            "serde_json to_string 必须紧凑: {compact}"
        );
        let (state_ids, fixed) = row_inputs(&tok, &compact);
        let row = build_question_row(
            &tok,
            &choice_q(),
            &state_ids,
            &fixed,
            HEAD_MAX_LEN,
            STATE_MAX_TOKENS,
        )
        .expect("row");
        assert_eq!(
            &row.input_ids[row.input_ids.len() - state_ids.len() - 1..row.input_ids.len() - 1],
            &state_ids[..]
        );
    }

    // ── ② head+options 超 head_max_len → InvalidInput ──

    #[test]
    fn test_build_question_row_rejects_head_over_budget() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // head（前缀 + instructions）自身已超预算
        let mut q = choice_q();
        q.instructions = "word ".repeat(200);
        let (state_ids, fixed) = row_inputs(&tok, "state");
        let err = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "head 超预算必须 InvalidInput，got {err:?}"
        );
    }

    #[test]
    fn test_build_question_row_rejects_options_over_budget() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // head 合法，但 head + options 总量超预算
        let mut q = choice_q();
        q.instructions = "pick one".to_string();
        q.options = (0..40)
            .map(|i| format!("option number {i} with several tokens here"))
            .collect();
        let (state_ids, fixed) = row_inputs(&tok, "state");
        let err = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "options 超预算必须 InvalidInput，got {err:?}"
        );
    }

    #[test]
    fn test_build_question_row_budget_boundary_passes() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // 恰好在预算内的短问题必须通过
        let (state_ids, fixed) = row_inputs(&tok, "state");
        build_question_row(
            &tok,
            &choice_q(),
            &state_ids,
            &fixed,
            HEAD_MAX_LEN,
            STATE_MAX_TOKENS,
        )
        .expect("合法问题必须通过预算检查");
    }

    #[test]
    fn test_build_question_row_noul_two_markers_score_five() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        let (state_ids, fixed) = row_inputs(&tok, "state");
        assert_eq!(
            fixed.score_len(),
            SCORE_LEVELS,
            "加载期预编码 5 组 score marker"
        );
        assert_eq!(
            fixed.noul_len(),
            NOUL_MARKERS,
            "加载期预编码 2 组 noul marker"
        );
        let noul_row = build_question_row(
            &tok,
            &noul_q(),
            &state_ids,
            &fixed,
            HEAD_MAX_LEN,
            STATE_MAX_TOKENS,
        )
        .expect("noul row");
        assert_eq!(
            noul_row.marker_pos.len(),
            NOUL_MARKERS,
            "noul 固定 [false, true] 两 marker"
        );
        let score_row = build_question_row(
            &tok,
            &score_q(),
            &state_ids,
            &fixed,
            HEAD_MAX_LEN,
            STATE_MAX_TOKENS,
        )
        .expect("score row");
        assert_eq!(
            score_row.marker_pos.len(),
            SCORE_LEVELS,
            "score 固定 5 级 marker"
        );
    }

    // ── ③ state 超长截断 ──

    #[test]
    fn test_build_question_row_truncates_state() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        let long_state = "word ".repeat(600);
        let q = choice_q();
        let (state_ids, fixed) = row_inputs(&tok, &long_state);
        assert!(
            state_ids.len() > STATE_MAX_TOKENS,
            "前置条件：原始 state 超长"
        );
        let row = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .expect("row");
        let head_len = tok
            .encode("choice question: pick one", false)
            .expect("head")
            .get_ids()
            .len();
        let opt_len: usize = [" beach", " mountain"]
            .iter()
            .map(|o| tok.encode(*o, false).expect("opt").get_ids().len())
            .sum();
        // 固定结构：1([CLS]) + head + 1([SEP]) + Σ(1+opt) + 1([SEP]) + state' + 1([SEP])
        let expected_len = 1 + head_len + 1 + 2 + opt_len + 1 + STATE_MAX_TOKENS + 1;
        assert_eq!(
            row.input_ids.len(),
            expected_len,
            "state 必须截断到 {STATE_MAX_TOKENS} token"
        );
        assert_eq!(
            &row.input_ids[expected_len - 1 - STATE_MAX_TOKENS..expected_len - 1],
            &state_ids[..STATE_MAX_TOKENS],
            "截断取前缀"
        );
    }

    // ── ④ collate：多题混合按行对位 ──

    #[test]
    fn test_collate_batch_mixed_qtypes_row_alignment() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // 3 题混合：choice(3 opt) / noul(2) / score(5)——N = 5
        let mut choice3 = choice_q();
        choice3.options = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let (state_ids, fixed) = row_inputs(&tok, "state");
        let rows = [
            build_question_row(
                &tok,
                &choice3,
                &state_ids,
                &fixed,
                HEAD_MAX_LEN,
                STATE_MAX_TOKENS,
            )
            .unwrap(),
            build_question_row(
                &tok,
                &noul_q(),
                &state_ids,
                &fixed,
                HEAD_MAX_LEN,
                STATE_MAX_TOKENS,
            )
            .unwrap(),
            build_question_row(
                &tok,
                &score_q(),
                &state_ids,
                &fixed,
                HEAD_MAX_LEN,
                STATE_MAX_TOKENS,
            )
            .unwrap(),
        ];
        let batch = collate_batch(&rows, &[0, 2, 1]).expect("batch");

        assert_eq!(batch.batch_size, 3);
        assert_eq!(
            batch.seq_len,
            rows.iter().map(|r| r.input_ids.len()).max().unwrap()
        );
        assert_eq!(batch.max_markers, 5, "N 取 batch 内最大 marker 数");

        // qtype 按行对位
        assert_eq!(batch.qtype, vec![0, 2, 1]);

        let mask_id = tok.token_to_id("[MASK]").expect("[MASK]") as i64;
        for (b, row) in rows.iter().enumerate() {
            // 行前缀原样保留（input_ids 按行对位；attention_mask 行前缀全 1
            // 由 collate 直接生成——QuestionRow 不携带恒全 1 冗余字段）
            assert_eq!(
                &batch.input_ids[b * batch.seq_len..b * batch.seq_len + row.input_ids.len()],
                &row.input_ids[..]
            );
            assert_eq!(
                &batch.attention_mask[b * batch.seq_len..b * batch.seq_len + row.input_ids.len()],
                &vec![1i64; row.input_ids.len()][..],
                "attention_mask 行前缀必须全 1"
            );
            // 行尾 padding：pad id=0、mask=0
            for s in row.input_ids.len()..batch.seq_len {
                assert_eq!(batch.input_ids[b * batch.seq_len + s], 0);
                assert_eq!(batch.attention_mask[b * batch.seq_len + s], 0);
            }
            // marker 维：有效位对位且指向 [MASK]，pad 位 marker_mask=false
            for n in 0..batch.max_markers {
                let idx = b * batch.max_markers + n;
                if n < row.marker_pos.len() {
                    assert!(batch.marker_mask[idx], "有效 marker 必须置位");
                    assert_eq!(batch.marker_pos[idx], row.marker_pos[n]);
                    assert_eq!(
                        batch.input_ids[b * batch.seq_len + row.marker_pos[n] as usize],
                        mask_id,
                        "风险#7：混合题型 batch 的三张量必须按行对位"
                    );
                } else {
                    assert!(!batch.marker_mask[idx], "pad 位 marker_mask 必须 false");
                    assert_eq!(batch.marker_pos[idx], 0);
                }
            }
        }
    }

    #[test]
    fn test_collate_batch_rejects_empty_rows() {
        let err = collate_batch(&[], &[]).unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "空 batch 必须显式拒绝"
        );
    }

    #[test]
    fn test_collate_batch_rejects_len_mismatch() {
        let row = QuestionRow {
            input_ids: vec![1, 2, 3],
            marker_pos: vec![2],
        };
        let err = collate_batch(std::slice::from_ref(&row), &[]).unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "rows/qtype 数不匹配必须显式拒绝，got {err:?}"
        );
    }

    // ── ⑤ softmax + 温度 ──

    #[test]
    fn test_softmax_sums_to_one() {
        let p = softmax_with_temperature(&[1.0, 2.0, 3.0], 1.0).expect("softmax");
        assert_eq!(p.len(), 3);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "和必须为 1，sum={sum}");
        assert!(p[2] > p[1] && p[1] > p[0], "logit 越大概率越大");
    }

    #[test]
    fn test_softmax_higher_temperature_flattens() {
        // 单调性：温度增大分布更平（max 概率下降）
        let cold = softmax_with_temperature(&[2.0, 1.0], 1.0).expect("cold");
        let hot = softmax_with_temperature(&[2.0, 1.0], 10.0).expect("hot");
        assert!(
            cold[0] > hot[0],
            "温度升高必须更平：cold_max={} hot_max={}",
            cold[0],
            hot[0]
        );
        // T→∞ 时趋于均匀
        let flat = softmax_with_temperature(&[2.0, 1.0], 1e6).expect("flat");
        assert!((flat[0] - 0.5).abs() < 1e-3, "极大温度应近均匀，p={flat:?}");
    }

    #[test]
    fn test_softmax_numeric_stability_at_large_logits() {
        // 减 max 数值稳定：e^1000 会溢出，减 max 后不产生 NaN/inf
        let p = softmax_with_temperature(&[1000.0, 1001.0], 1.0).expect("softmax");
        assert!(p.iter().all(|x| x.is_finite()), "必须数值稳定，p={p:?}");
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_temperature_hand_computed_golden() {
        // 手算 golden：softmax([0, ln3], T=1) = [0.25, 0.75]
        let ln3 = 3.0f32.ln();
        let p = softmax_with_temperature(&[0.0, ln3], 1.0).expect("softmax");
        assert!((p[0] - 0.25).abs() < 1e-6, "p[0]={}", p[0]);
        assert!((p[1] - 0.75).abs() < 1e-6, "p[1]={}", p[1]);
        // 除以温度等价于缩放 logits：softmax([0, ln3], T=2) = softmax([0, ln3/2])
        let direct = softmax_with_temperature(&[0.0, ln3 / 2.0], 1.0).expect("direct");
        let via_t = softmax_with_temperature(&[0.0, ln3], 2.0).expect("via T");
        for (a, b) in direct.iter().zip(&via_t) {
            assert!((a - b).abs() < 1e-6, "温度语义必须是 logits/T");
        }
    }

    #[test]
    fn test_softmax_rejects_non_positive_temperature() {
        for t in [0.0, -1.0, f32::NAN] {
            let err = softmax_with_temperature(&[1.0, 2.0], t).unwrap_err();
            assert!(
                matches!(err, VecboostError::InvalidInput(_)),
                "T={t} 必须显式拒绝（除零/NaN 不得静默），got {err:?}"
            );
        }
    }

    #[test]
    fn test_softmax_rejects_empty_logits() {
        let err = softmax_with_temperature(&[], 1.0).unwrap_err();
        assert!(matches!(err, VecboostError::InvalidInput(_)));
    }

    // ── ⑥ 校准：查表 / 钳制 / 缺桶回退 / 损坏显性报错 ──

    #[test]
    fn test_calibration_lookup_and_fallback() {
        // 钳制语义归 from_bundle（加载即钳制，见下方 clamps_on_load 测试）；
        // 此处只验证查表与缺桶回退的读取语义
        let cal = TemperatureCalibration {
            temperatures: HashMap::from([(2usize, 0.8f32), (5usize, 3.0f32)]),
        };
        assert!((cal.temperature_for(2) - 0.8).abs() < 1e-6);
        assert!((cal.temperature_for(5) - 3.0).abs() < 1e-6);
        assert!(
            (cal.temperature_for(3) - DEFAULT_TEMPERATURE).abs() < 1e-6,
            "缺桶必须回退 1.2"
        );
    }

    #[test]
    fn test_calibration_from_bundle_parses_laya_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("laya_config.json"),
            r#"{"temperature_by_options": {"2": 0.8, "5": 3.0}}"#,
        )
        .expect("write config");
        let cal = TemperatureCalibration::from_bundle(dir.path()).expect("parse");
        assert!((cal.temperature_for(2) - 0.8).abs() < 1e-6);
        assert!((cal.temperature_for(5) - 3.0).abs() < 1e-6);
        assert!(
            (cal.temperature_for(4) - DEFAULT_TEMPERATURE).abs() < 1e-6,
            "缺桶回退 1.2"
        );
    }

    #[test]
    fn test_calibration_from_bundle_clamps_on_load() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("laya_config.json"),
            r#"{"temperature_by_options": {"3": 0.05, "6": 100}}"#,
        )
        .expect("write config");
        let cal = TemperatureCalibration::from_bundle(dir.path()).expect("parse");
        assert!(
            (cal.temperature_for(3) - MIN_TEMPERATURE).abs() < 1e-6,
            "加载即钳制下界（§9.6）"
        );
        assert!(
            (cal.temperature_for(6) - MAX_TEMPERATURE).abs() < 1e-6,
            "加载即钳制上界（§9.6）"
        );
    }

    #[test]
    fn test_calibration_missing_file_warns_and_falls_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cal = TemperatureCalibration::from_bundle(dir.path()).expect("缺文件兜底不得报错");
        assert!(
            (cal.temperature_for(2) - DEFAULT_TEMPERATURE).abs() < 1e-6,
            "校准缺失回退 1.2（未校准 ECE 0.466 vs 校准后 0.081，不得静默当已校准）"
        );
    }

    #[test]
    fn test_calibration_corrupted_json_reports_model_file_corrupted() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("laya_config.json"), "{not json").expect("write");
        let err = TemperatureCalibration::from_bundle(dir.path()).unwrap_err();
        assert!(
            matches!(err, VecboostError::ModelFileCorrupted(_)),
            "JSON 损坏必须 ModelFileCorrupted 而非吞掉，got {err:?}"
        );
    }

    #[test]
    fn test_calibration_wrong_field_shape_reports_model_file_corrupted() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("laya_config.json"),
            r#"{"temperature_by_options": 3}"#,
        )
        .expect("write");
        let err = TemperatureCalibration::from_bundle(dir.path()).unwrap_err();
        assert!(
            matches!(err, VecboostError::ModelFileCorrupted(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_calibration_non_numeric_value_reports_model_file_corrupted() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("laya_config.json"),
            r#"{"temperature_by_options": {"2": "hot"}}"#,
        )
        .expect("write");
        let err = TemperatureCalibration::from_bundle(dir.path()).unwrap_err();
        assert!(
            matches!(err, VecboostError::ModelFileCorrupted(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_calibration_non_integer_key_reports_model_file_corrupted() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("laya_config.json"),
            r#"{"temperature_by_options": {"two": 1.0}}"#,
        )
        .expect("write");
        let err = TemperatureCalibration::from_bundle(dir.path()).unwrap_err();
        assert!(
            matches!(err, VecboostError::ModelFileCorrupted(_)),
            "got {err:?}"
        );
    }

    // ── ⑦ 三类后处理（手算 golden）──

    #[test]
    fn test_choice_answer_argmax_and_full_probability_table() {
        // 手算：softmax([1, 3], T=1)：e^1=2.71828, e^3=20.08553 → p=[0.11920, 0.88080]
        let options = vec!["beach".to_string(), "mountain".to_string()];
        let answer = choice_answer("destination", &options, &[1.0, 3.0], 1.0).expect("answer");
        assert_eq!(answer.question, "destination");
        match answer.answer {
            DecisionAnswerBody::Choice {
                index,
                option,
                probabilities,
            } => {
                assert_eq!(index, 1, "argmax 应命中 mountain");
                assert_eq!(option, "mountain");
                assert_eq!(probabilities.len(), 2, "完整概率表");
                assert!(
                    (probabilities["beach"] - 0.11920).abs() < 1e-4,
                    "{:?}",
                    probabilities
                );
                assert!((probabilities["mountain"] - 0.88080).abs() < 1e-4);
                let sum: f32 = probabilities.values().sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
            other => panic!("choice 型必须产出 Choice 体，got {other:?}"),
        }
    }

    #[test]
    fn test_choice_answer_tie_takes_first_index() {
        // 平局取最小索引（严格大于才更新 argmax）
        let options = vec!["a".to_string(), "b".to_string()];
        let answer = choice_answer("q", &options, &[1.0, 1.0], 1.0).expect("answer");
        match answer.answer {
            DecisionAnswerBody::Choice { index, .. } => assert_eq!(index, 0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_choice_answer_length_mismatch_is_error() {
        let options = vec!["a".to_string(), "b".to_string()];
        let err = choice_answer("q", &options, &[1.0, 2.0, 3.0], 1.0).unwrap_err();
        assert!(
            matches!(err, VecboostError::InferenceError(_)),
            "logits 数与 options 数不符必须显式报错，got {err:?}"
        );
    }

    #[test]
    fn test_choice_answer_duplicate_option_names_rejected() {
        // 重复选项名会使 BTreeMap 键合并、静默丢概率——显式拒绝
        let options = vec!["same".to_string(), "same".to_string()];
        let err = choice_answer("q", &options, &[1.0, 2.0], 1.0).unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "重复选项名必须显式拒绝，got {err:?}"
        );
    }

    #[test]
    fn test_choice_answer_error_echo_is_sanitized() {
        // 测试钉：重复选项错误的 name/option 回显必须经公共 echo——
        // 换行字面量化（单行化，防日志注入）+ 超长原文截断。
        // domain validate 现已前置拒绝重复 option（defense in depth：
        // 引擎层防御臂直调可达，回显同样不得裸内插）
        let long_evil = format!("evil\n{}\tsame", "x".repeat(200));
        let options = vec![long_evil.clone(), long_evil];
        let err = choice_answer("q", &options, &[1.0, 2.0], 1.0).unwrap_err();
        let detail = err.error_detail();
        assert!(
            !detail.contains('\n') && !detail.contains('\t'),
            "回显必须单行化（\\n\\t 字面量化），detail={detail:?}"
        );
        assert!(
            detail.len() < 300,
            "name/option 原文（200+ 字符）必须被截断，len={}",
            detail.len()
        );
        assert!(detail.contains("duplicate option"));
    }

    #[test]
    fn test_score_answer_expected_and_distribution() {
        // 手算 golden：logits [0, ln3], T=1 → p=[0.25, 0.75]
        // expected = 0*0.25 + 1*0.75 = 0.75
        let ln3 = 3.0f32.ln();
        let answer = score_answer("urgency", &[0.0, ln3], 1.0).expect("answer");
        assert_eq!(answer.question, "urgency");
        match answer.answer {
            DecisionAnswerBody::Score {
                expected,
                distribution,
            } => {
                assert!(
                    (expected - 0.75).abs() < 1e-5,
                    "期望等级手算 0.75，got {expected}"
                );
                assert_eq!(distribution.len(), 2, "完整分布");
                assert!((distribution["0"] - 0.25).abs() < 1e-5);
                assert!((distribution["1"] - 0.75).abs() < 1e-5);
            }
            other => panic!("score 型必须产出 Score 体，got {other:?}"),
        }
    }

    #[test]
    fn test_score_answer_distribution_sums_to_one() {
        let p = score_answer("urgency", &[0.1, 0.4, 0.2, 0.9, 0.3], 1.2).expect("answer");
        match p.answer {
            DecisionAnswerBody::Score {
                expected,
                distribution,
            } => {
                let sum: f32 = distribution.values().sum();
                assert!((sum - 1.0).abs() < 1e-6, "分布和必须为 1，sum={sum}");
                assert!(
                    (0.0..=4.0).contains(&expected),
                    "期望等级必须落在等级空间内"
                );
                assert_eq!(distribution.len(), SCORE_LEVELS);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_noul_answer_p_true_hand_computed() {
        // 手算 golden：logits [0, ln3], T=1 → P(true)=0.75；逆序 → 0.25
        let ln3 = 3.0f32.ln();
        let answer = noul_answer("churn_risk", &[0.0, ln3], 1.0).expect("answer");
        assert_eq!(answer.question, "churn_risk");
        match answer.answer {
            DecisionAnswerBody::Noul { p_true } => {
                assert!(
                    (p_true - 0.75).abs() < 1e-5,
                    "P(true) 手算 0.75，got {p_true}"
                );
            }
            other => panic!("noul 型必须产出 Noul 体，got {other:?}"),
        }
        let flipped = noul_answer("churn_risk", &[ln3, 0.0], 1.0).expect("flipped");
        match flipped.answer {
            DecisionAnswerBody::Noul { p_true } => {
                assert!(
                    (p_true - 0.25).abs() < 1e-5,
                    "marker 顺序 [false,true]，逆序得 0.25"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_noul_answer_rejects_wrong_marker_count() {
        for logits in [&[1.0][..], &[1.0, 2.0, 3.0][..]] {
            let err = noul_answer("q", logits, 1.0).unwrap_err();
            assert!(
                matches!(err, VecboostError::InferenceError(_)),
                "noul 固定两 marker，got {err:?}"
            );
        }
    }

    // ── 防回归钉①：decide 必须作为 trait 方法对 DecisionPipeline 可解析 ──
    //
    // 如实声明局限：fn 引用无法区分默认实现与覆盖——「漏覆盖 → 继承默认
    // Err(UnsupportedTask)」由 bundle 守卫测试（运行级）与 P0 example
    // （在线级）兜住，本钉只防 trait 接入整体缺失（编译级）。

    #[test]
    fn test_compile_pin_decide_is_trait_method_on_pipeline() {
        let _: fn(&DecisionPipeline, &DecisionRequest) -> Result<DecisionResponse, VecboostError> =
            <DecisionPipeline as InferenceEngine>::decide;
    }

    // ── 防回归钉②（运行级，bundle 就绪守卫，离线 SKIP）：
    // DecisionPipeline 经 &dyn InferenceEngine 调 decide 必须走真实管线
    // （钉死「漏覆盖 decide → 默认 UnsupportedTask」的分发回归）。──

    #[test]
    fn test_bundle_guard_decide_via_trait_object() {
        let bundle = Path::new("models/laya");
        if !bundle.is_dir() {
            eprintln!(
                "Skipping test: laya bundle not found at {:?}（真 bundle 环境下本测试钉死 DecisionPipeline 级分发链）",
                bundle
            );
            return;
        }
        let mut config = test_config();
        config.model_path = bundle.to_path_buf();
        let pipeline = DecisionPipeline::load(&config).expect("bundle 就绪时必须可加载");
        let engine: &dyn InferenceEngine = &pipeline;
        // task=decision 时 embedding 端点必须显性 UnsupportedTask（诚实语义）
        match engine.embed("text") {
            Err(VecboostError::UnsupportedTask(_)) => {}
            other => panic!("决策引擎 embed 必须显性 UnsupportedTask，got {other:?}"),
        }
        assert!(
            engine.supports_task(crate::config::model::ModelTask::Decision),
            "决策引擎必须自报支持 Decision"
        );
        assert!(
            !engine.supports_rerank(),
            "决策引擎不得谎报 rerank 能力（默认实现依赖 embed）"
        );
        let req: DecisionRequest = serde_json::from_str(
            r#"{"state":"billed twice","questions":[
                {"name":"department","qtype":"choice","instructions":"which?","options":["billing","other"]},
                {"name":"urgency","qtype":"score","instructions":"how urgent?"},
                {"name":"churn_risk","qtype":"noul","instructions":"churning?"}
            ]}"#,
        )
        .expect("guard request");
        match engine.decide(&req) {
            Ok(resp) => assert_eq!(resp.answers.len(), 3, "三题型各一答案"),
            Err(e) => panic!("bundle 就绪时 decide 必须进真实管线而非报错：{e:?}"),
        }
    }

    // ── 加载类：bundle 探测失败显性 ModelLoadError（含尝试路径清单）──

    #[test]
    fn test_decision_pipeline_load_empty_dir_reports_tried_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.model_path = dir.path().to_path_buf();
        match DecisionPipeline::load(&config) {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("model.onnx") && msg.contains("laya.onnx"),
                    "错误必须列出尝试的模型路径清单，msg={msg}"
                );
            }
            Err(other) => panic!("期望 ModelLoadError，got {other:?}"),
            Ok(_) => panic!("空 bundle 必须加载失败"),
        }
    }

    #[test]
    fn test_decision_pipeline_load_model_without_tokenizer_lists_tokenizer_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.onnx"), b"fake onnx").expect("write model");
        let mut config = test_config();
        config.model_path = dir.path().to_path_buf();
        match DecisionPipeline::load(&config) {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("tokenizer.json"),
                    "错误必须列出尝试的 tokenizer 路径清单，msg={msg}"
                );
                assert!(
                    msg.contains("tokenizer") && msg.contains("tokenizer.json"),
                    "清单必须含 tokenizer/ 子目录候选，msg={msg}"
                );
            }
            Err(other) => panic!("期望 ModelLoadError，got {other:?}"),
            // 正常情况在 fake onnx 的 session 构建前即失败（探测先于 Session::builder，
            // 不触发 ort 环境崩溃问题）
            Ok(_) => panic!("缺 tokenizer 必须加载失败"),
        }
    }

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-decision".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: PathBuf::from("/nonexistent/laya"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: crate::config::model::ModelTask::Decision,
            quantized: false,
        }
    }

    // ── 错误回显防线：引擎层回显必须走公共 echo（截断+字面量化）──

    #[test]
    fn test_error_echo_truncates_long_name() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // name 超长（直调纯函数不过 domain validate）+ head 超预算触发回显：
        // detail 必须被 64 字符截断，不得携带原文全文
        let mut q = choice_q();
        q.name = "n".repeat(500);
        q.instructions = "word ".repeat(200);
        let (state_ids, fixed) = row_inputs(&tok, "state");
        let err = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .unwrap_err();
        let detail = err.error_detail();
        assert!(detail.len() < 200, "回显必须截断，len={}", detail.len());
        assert!(
            !detail.contains(&"n".repeat(100)),
            "原文不得完整进入 detail"
        );
    }

    #[test]
    fn test_error_echo_literalizes_control_chars() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        // name 含换行：回显必须单行化（日志注入面），\n 字面量化为 \\n
        let mut q = choice_q();
        q.instructions = "word ".repeat(200);
        q.name = "evil\nFAKE LOG LINE".to_string();
        let (state_ids, fixed) = row_inputs(&tok, "state");
        let err = build_question_row(&tok, &q, &state_ids, &fixed, HEAD_MAX_LEN, STATE_MAX_TOKENS)
            .unwrap_err();
        let detail = err.error_detail();
        assert!(!detail.contains('\n'), "回显必须单行化，detail={detail:?}");
        assert!(detail.contains("\\n"), "换行应字面量化，detail={detail:?}");
    }

    // ── precision 标签与探测文件一致 ──

    #[test]
    fn test_precision_for_model_file_names() {
        assert_eq!(
            precision_for_model_file(Path::new("bundle/model.onnx")),
            Precision::Fp32
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/laya.onnx")),
            Precision::Fp32
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/model_quantized.onnx")),
            Precision::Int8
        );
        assert_eq!(
            precision_for_model_file(Path::new("bundle/laya_int8.onnx")),
            Precision::Int8
        );
    }

    #[test]
    fn test_marker_constants_consistent() {
        assert_eq!(NOUL_MARKERS, 2);
        assert_eq!(SCORE_LEVELS, 5);
        assert_eq!(MODEL_CANDIDATES.len(), 4);
    }
}
