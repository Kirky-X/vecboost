// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Laya 决策协议层：预处理（序列构造 / marker_pos / qtype 编码）、collate、
//! 按题 softmax + per-cardinality 温度校准、choice/score/noul 三类后处理。
//!
//! 纯 Rust、无推理后端依赖，供 onnx 管线（[`super::decision`]）与 candle
//! 原生决策头（[`super::candle_decision`]）共享——两路输入/输出同源是对齐
//! 闸门（`tests/candle_decision_parity.rs`）的前提。
//!
//! 序列协议（可行性文档 §2.1）：
//! `[CLS] question [SEP] [MASK] option1 [MASK] option2 ... [SEP] state [SEP]`
//!
//! # P0 待校准项（与 receptron/laya 参考实现对照后固化，禁止漂移被静默接受）
//! - head 前缀文本格式（`"{qtype} question: {instructions}"`，§4.2 伪代码）
//! - marker 前置 option 拼接（`" " + opt`）
//! - noul marker 顺序（固定 `[false, true]`）
//! - score 等级数（domain 契约禁 score 携带 options，管线内固定 5 级）
//! - candle 决策头 act 特征四元组（`candle_decision.rs`：top1/top1−top2/
//!   归一化熵/有效 marker 占比 ÷255）——官方 `rl_common.py` 手写复刻，无
//!   独立对照面（onnx bundle 图无 act 输出，对齐闸门只覆盖 decision
//!   logits），待真实 act 参照面出现时闭环
//!
//! 以上均需在拿到真实 bundle 跑 P0 数值对照（概率差 <1e-4）后确认。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::config::model::DecisionParams;
use crate::domain::{DecisionAnswer, DecisionAnswerBody, DecisionQuestion, QuestionType};
use crate::error::VecboostError;
use crate::utils::validator::input::echo;
use tokenizers::Tokenizer;

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
pub(crate) fn encode_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, VecboostError> {
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
/// head + 全部 options（含各自 [MASK]）超过 `params.head_max_len` 时报
/// `InvalidInput`（§2.1 风险前置校验；报错消息含实际生效的 head_max_len 数值，
/// per-checkpoint 参数漂移可从错误面直接观察）。
pub(crate) fn build_question_row(
    tokenizer: &Tokenizer,
    question: &DecisionQuestion,
    state_ids: &[i64],
    fixed: &FixedMarkers,
    params: DecisionParams,
) -> Result<QuestionRow, VecboostError> {
    let head_max_len = params.head_max_len;
    let state_max_tokens = params.state_max_tokens;
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
    pub(crate) temperatures: HashMap<usize, f32>,
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
    if options.is_empty() {
        return Err(VecboostError::invalid_input(format!(
            "choice question {} requires at least one option",
            echo(question_name)
        )));
    }
    if options.len() != logits.len() {
        return Err(VecboostError::inference_error(format!(
            "choice question {}: got {} logits for {} options",
            echo(question_name),
            logits.len(),
            options.len()
        )));
    }
    let mut seen = HashSet::with_capacity(options.len());
    for option in options {
        if !seen.insert(option.as_str()) {
            return Err(VecboostError::invalid_input(format!(
                "choice question {} has duplicate option {:?}: \
                 probability table keys would silently merge",
                echo(question_name),
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

// ─────────────────────────────────────────────────────────────────────────────
// 测试：接口驱动。测试 tokenizer 资产按
// models/all-MiniLM-L6-v2/tokenizer.json 存在性守卫 SKIP（离线硬约束）。
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{HEAD_MAX_LEN, MAX_DECISION_SEQ_LEN, STATE_MAX_TOKENS};

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

    /// 与生产管线同口径的行输入：state 编码一次 + 加载期 FixedMarkers
    fn row_inputs(tok: &Tokenizer, state: &str) -> (Vec<i64>, FixedMarkers) {
        let state_ids = encode_ids(tok, state).expect("state ids");
        let fixed = FixedMarkers::new(tok).expect("fixed markers");
        (state_ids, fixed)
    }

    // ── DecisionParams 解析：缺省口径 / 非法值显性拒绝 ──

    #[test]
    fn test_decision_params_default_matches_constants() {
        let p = DecisionParams::default();
        assert_eq!(p.head_max_len, HEAD_MAX_LEN, "缺省必须与 HEAD_MAX_LEN 同源");
        assert_eq!(
            p.state_max_tokens, STATE_MAX_TOKENS,
            "缺省必须与 STATE_MAX_TOKENS 同源"
        );
        // 英文 checkpoint 口径数值钉：漂移即序列协议变更，须显性确认
        assert_eq!(p.head_max_len, 192);
        assert_eq!(p.state_max_tokens, 256);
    }

    #[test]
    fn test_decision_params_new_accepts_multilingual_profile() {
        // laya-multilingual 口径（head 256 / state 256）：合法且逐字段保真
        let p = DecisionParams::new(256, 256).expect("multilingual 口径合法");
        assert_eq!(p.head_max_len, 256);
        assert_eq!(p.state_max_tokens, 256);
    }

    #[test]
    fn test_decision_params_rejects_zero() {
        for (head, state) in [(0usize, 256usize), (192, 0), (0, 0)] {
            let err = DecisionParams::new(head, state).unwrap_err();
            assert!(
                matches!(err, VecboostError::ConfigError(_)),
                "head={head} state={state} 必须解析期显性拒绝（禁止静默钳制），got {err:?}"
            );
        }
    }

    #[test]
    fn test_decision_params_rejects_over_hard_limit() {
        for (head, state) in [
            (MAX_DECISION_SEQ_LEN + 1, 256usize),
            (192, MAX_DECISION_SEQ_LEN + 1),
        ] {
            let err = DecisionParams::new(head, state).unwrap_err();
            assert!(
                matches!(err, VecboostError::ConfigError(_)),
                "head={head} state={state} 超硬上限必须解析期显性拒绝，got {err:?}"
            );
        }
    }

    #[test]
    fn test_decision_params_error_message_names_field_and_value() {
        let err = DecisionParams::new(0, 256).unwrap_err();
        let detail = err.error_detail();
        assert!(
            detail.contains("head_max_len") && detail.contains('0'),
            "错误必须点名字段与实际值，detail={detail}"
        );
        let err = DecisionParams::new(192, MAX_DECISION_SEQ_LEN + 1).unwrap_err();
        let detail = err.error_detail();
        assert!(
            detail.contains("state_max_tokens"),
            "错误必须点名 state_max_tokens，detail={detail}"
        );
    }

    #[test]
    fn test_decision_params_combined_budget_rejected() {
        use crate::config::model::DECISION_ROW_FIXED_TOKENS;
        // 两字段各自合规（=单字段上限）但组合超窗：启动期显性拒绝而非延迟到
        // 请求期以推理错误暴露（单字段口径不保证行总长 ≤ 位置编码窗宽）
        let err = DecisionParams::new(MAX_DECISION_SEQ_LEN, MAX_DECISION_SEQ_LEN).unwrap_err();
        assert!(
            matches!(err, VecboostError::ConfigError(_)),
            "组合超窗必须解析期显性拒绝，got {err:?}"
        );
        assert!(
            err.error_detail().contains("combined sequence budget"),
            "错误必须点名组合口径，detail={}",
            err.error_detail()
        );
        // 边界钉：head + state + 固定 marker 恰达窗宽合法，超出 1 token 拒绝
        let at_window = MAX_DECISION_SEQ_LEN - 4000 - DECISION_ROW_FIXED_TOKENS;
        let ok = DecisionParams::new(4000, at_window).expect("组合恰达窗宽必须合法");
        assert_eq!(
            ok.head_max_len + ok.state_max_tokens + DECISION_ROW_FIXED_TOKENS,
            MAX_DECISION_SEQ_LEN
        );
        let err = DecisionParams::new(4000, at_window + 1).unwrap_err();
        assert!(
            matches!(err, VecboostError::ConfigError(_)),
            "组合超窗 1 token 也必须拒绝，got {err:?}"
        );
    }

    #[test]
    fn test_decision_params_effective_on_budget_behavior() {
        let Some(tok) = mini_lm_tokenizer() else {
            return;
        };
        let q = choice_q();
        let (state_ids, fixed) = row_inputs(&tok, "state");
        // 同一请求在收紧预算(8)下超限拒绝，报错消息携带实际生效的 8——
        // 钉死「报错含实际生效 head_max_len」的参数化语义（拒绝静默用编译期常量）
        let err = build_question_row(
            &tok,
            &q,
            &state_ids,
            &fixed,
            DecisionParams::new(8, 8).expect("收紧口径合法"),
        )
        .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "收紧预算必须拒绝，got {err:?}"
        );
        assert!(
            err.error_detail().contains("head_max_len 8"),
            "报错必须含实际生效的 head_max_len 数值，detail={}",
            err.error_detail()
        );
        // 同一请求在缺省英文口径下通过——参数真实生效而非常量直传
        build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
            .expect("同一请求在缺省预算下必须通过");
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
        let row = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
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
            DecisionParams::default(),
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
            DecisionParams::default(),
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
        let err = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
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
        let err = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
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
            DecisionParams::default(),
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
            DecisionParams::default(),
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
            DecisionParams::default(),
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
        let row = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
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
                DecisionParams::default(),
            )
            .unwrap(),
            build_question_row(
                &tok,
                &noul_q(),
                &state_ids,
                &fixed,
                DecisionParams::default(),
            )
            .unwrap(),
            build_question_row(
                &tok,
                &score_q(),
                &state_ids,
                &fixed,
                DecisionParams::default(),
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

    #[test]
    fn test_marker_constants_consistent() {
        assert_eq!(NOUL_MARKERS, 2);
        assert_eq!(SCORE_LEVELS, 5);
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
        let err = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
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
        let err = build_question_row(&tok, &q, &state_ids, &fixed, DecisionParams::default())
            .unwrap_err();
        let detail = err.error_detail();
        assert!(!detail.contains('\n'), "回显必须单行化，detail={detail:?}");
        assert!(detail.contains("\\n"), "换行应字面量化，detail={detail:?}");
    }
}
