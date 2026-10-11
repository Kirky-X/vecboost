// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! ONNX seq2seq 翻译引擎（Marian/opus-mt，encoder + decoder_model_merged
//! 双 Session 贪心解码）。
//!
//! 张量契约（T013 首日实测，notes/u2-inspection-raw.txt）：
//! - encoder：inputs `input_ids`/`attention_mask` (Int64[-1,-1]) →
//!   output `last_hidden_state` (F32[-1,-1,512])
//! - decoder_model_merged：inputs `input_ids`/`encoder_attention_mask`/
//!   `encoder_hidden_states`/`use_cache_branch` (Bool[1]——If 节点分支选择，
//!   缺失即 `Missing Input: use_cache_branch` 硬错) + `past_key_values.N.
//!   {decoder,encoder}.{key,value}` ×24 (F32[-1,8,-1,64]) → outputs `logits`
//!   (F32[-1,-1,V]) + `present.N.*` ×24。
//!
//! # 缓存分支禁用（实测缺陷）
//! `use_cache_branch=true` 的缓存分支在 ORT 1.27(py)/1.28(dylib) 下必败：
//! 其交叉注意力 `encoder_attn/Reshape_4` 的目标 shape 从 decoder hidden 取
//! 序列维（缓存步恒为 1）而非 encoder 源长，实抓
//! `ReshapeHelper: dimension with value zero exceeds...`；present→past 回喂、
//! 空张量占位、反语义三种喂法均复现，属图导出缺陷而非调用口径。故贪心解码
//! 恒走无缓存分支（`use_cache_branch=false` + 空张量占位），每步对已生成
//! 序列全量重算、取末位 logits——O(L²) 成本有界于 `max_new_tokens`，
//! 正确性优先（端到端实测译文正确）。
//!
//! tokenizer shim（T013 U1 换路，语义保持）：Xenova Marian bundle 的
//! `normalizer.precompiled_charsmap = null` 使 tokenizers 0.23.2 在
//! Precompiled 反序列化分支 panic（不可捕获）——加载前在内存中移除该
//! no-op normalizer（等价 identity，id 序列与净副本一致），不改磁盘 bundle。

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use super::session::{SessionOpts, build_session, precision_for_model_file, probe_model_file};
use crate::config::model::{
    DeviceType, ModelConfig, ModelTask, OutputFix, Precision, TranslationParams,
};
use crate::domain::{TranslateRequest, TranslateResponse};
use crate::error::VecboostError;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::InferenceEngine;

/// encoder 候选（fp32 主模型优先，与 decision 探测口径同构）
const ENCODER_CANDIDATES: [&str; 2] = [
    "onnx/encoder_model.onnx",
    "onnx/encoder_model_quantized.onnx",
];
/// decoder 候选（merged 变体唯一可自回归：past_key_values 输入 + present 输出）
const DECODER_CANDIDATES: [&str; 2] = [
    "onnx/decoder_model_merged.onnx",
    "onnx/decoder_model_merged_quantized.onnx",
];
/// Xenova Marian 词表维度（U3 实测 logits 尾维；形状断言用，防图错位）
#[cfg(test)]
const VOCAB_DIM_EN_ZH: usize = 65001;

/// 全局 per-Session intra 线程预算（评审 PERF-4）：多 pair 路由表下每对
/// 2 个 Session 各自全核 intra 线程会过订阅 CPU（16 对 × 2 Session × 全核），
/// 启动期按 pair 数分摊（`available_parallelism / pairs`，下限 1）；0 =
/// 未设置（单 pair/非路由场景回退全核）。encoder/decoder 串行执行，同 pair
/// 有效并行度即该值而非 2×。
static SESSION_INTRA_THREADS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 启动期设置全局 per-Session 线程预算（main.rs pair 循环前调用）
pub fn set_session_intra_threads(threads: usize) {
    SESSION_INTRA_THREADS.store(threads.max(1), std::sync::atomic::Ordering::Relaxed);
}

/// past 占位输入名静态表（评审 PERF-2：消除每步 24 次 format! 堆分配）
static PAST_TENSOR_NAMES: &[(&str, &str, &str)] = &[
    ("past_key_values.0.decoder.key", "", ""),
    ("past_key_values.0.decoder.value", "", ""),
    ("past_key_values.0.encoder.key", "", ""),
    ("past_key_values.0.encoder.value", "", ""),
    ("past_key_values.1.decoder.key", "", ""),
    ("past_key_values.1.decoder.value", "", ""),
    ("past_key_values.1.encoder.key", "", ""),
    ("past_key_values.1.encoder.value", "", ""),
    ("past_key_values.2.decoder.key", "", ""),
    ("past_key_values.2.decoder.value", "", ""),
    ("past_key_values.2.encoder.key", "", ""),
    ("past_key_values.2.encoder.value", "", ""),
    ("past_key_values.3.decoder.key", "", ""),
    ("past_key_values.3.decoder.value", "", ""),
    ("past_key_values.3.encoder.key", "", ""),
    ("past_key_values.3.encoder.value", "", ""),
    ("past_key_values.4.decoder.key", "", ""),
    ("past_key_values.4.decoder.value", "", ""),
    ("past_key_values.4.encoder.key", "", ""),
    ("past_key_values.4.encoder.value", "", ""),
    ("past_key_values.5.decoder.key", "", ""),
    ("past_key_values.5.decoder.value", "", ""),
    ("past_key_values.5.encoder.key", "", ""),
    ("past_key_values.5.encoder.value", "", ""),
];

/// logits 图契约校验（评审 SEC-2）：batch=1、[1, dec_len, V] 形状、数据长度
/// 一致。不符返回含实际形状的消息（换导出仓库/量化变体时快速定位）。
fn validate_logits_shape(shape: &[i64], dec_len: i64, data_len: usize) -> Result<(), String> {
    if shape.len() != 3 {
        return Err(format!(
            "expected rank-3 logits [1, dec_len, V], got shape {shape:?}"
        ));
    }
    if shape[0] != 1 {
        return Err(format!(
            "expected batch=1 (greedy decode), got shape {shape:?}"
        ));
    }
    if shape[1] != dec_len {
        return Err(format!("expected dec_len {dec_len}, got shape {shape:?}"));
    }
    let vocab = shape[2] as usize;
    if data_len != dec_len as usize * vocab {
        return Err(format!(
            "data length {data_len} != dec_len {dec_len} × vocab {vocab}"
        ));
    }
    if vocab == 0 {
        return Err(format!("vocab dimension is zero, shape {shape:?}"));
    }
    Ok(())
}

/// 双 Session 束（encoder 单次前向 + decoder 自回归，单锁串行化整链）
pub(crate) struct TranslateSessions {
    encoder: Session,
    decoder: Session,
}

/// 构建一对 encoder/decoder Session（主路径与并行 worker 共用）。
fn build_translate_sessions(
    encoder_file: &Path,
    decoder_file: &Path,
    intra_threads: usize,
    device: &DeviceType,
) -> Result<TranslateSessions, VecboostError> {
    let encoder = build_session(
        encoder_file,
        &SessionOpts {
            intra_threads,
            device: device.clone(),
            extra_config: &[],
        },
    )?;
    let decoder = build_session(
        decoder_file,
        &SessionOpts {
            intra_threads,
            device: device.clone(),
            extra_config: &[],
        },
    )?;
    Ok(TranslateSessions { encoder, decoder })
}

/// 句级并行 worker 数（含主 Session）：`TRANSLATION_PARALLEL_WORKERS` 覆盖，
/// 缺省 4，钳 1..=8。1 = 关闭并行（整段直解）。worker 各自独占 Session 集，
/// 内存随 worker 数线性增长（int8 418M 每套约 0.5GB 级）。
const DEFAULT_TRANSLATION_PARALLEL_WORKERS: usize = 4;
const MAX_TRANSLATION_PARALLEL_WORKERS: usize = 8;

fn translation_parallel_workers() -> usize {
    std::env::var("TRANSLATION_PARALLEL_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TRANSLATION_PARALLEL_WORKERS)
        .clamp(1, MAX_TRANSLATION_PARALLEL_WORKERS)
}

/// 惰性补建数量（wsl2-oom-hardening D2）：worker=1 并行关闭恒 0；
/// 否则 max(0, workers-1-current)，已有超量不裁剪。
fn missing_worker_sets(current: usize, worker_count: usize) -> usize {
    worker_count.saturating_sub(1).saturating_sub(current)
}

/// 长输入并行阈值：受保护文本字符数 ≥ 此值且句数 ≥2 才走句级并行（短文本
/// 整段直解保留跨句上下文）。paper-translate-quality P7：论文摘要 1k+ 字符
/// 整段贪心 150s+，并行后时延 ÷ worker 数。
const PARALLEL_MIN_CHARS: usize = 600;

/// generation_config.json 解析结果（贪心解码锚点）。
///
/// `num_beams` 等 beam 参数不消费：本引擎恒贪心（argmax），与配置文件
/// 的 num_beams=4 无关（bundle 该字段面向 transformers 生成器，语义
/// 不适用于贪心实现）。
struct GenerationConfig {
    decoder_start_token_id: u32,
    eos_token_id: u32,
}

/// 语言 token 风格（m2m_100 家族内二分；load 期以 tokenizer 探测判定）。
/// Marian pair 专用 bundle 无语言 token。
#[derive(Debug, Clone, PartialEq)]
enum LangTokenStyle {
    /// M2M100：`__{iso}__`（vocab 内建百语言码，token_to_id 探测验证）
    M2m,
    /// NLLB-200：`{iso639-3}_{script}` 显式映射表
    Nllb,
}

/// m2m_100 家族图几何：past 交叉注意力 KV 名表按层数动态生成、占位形状
/// 按（头数, 头维）实例化。字段读自 bundle config.json（load 期 fail-fast）。
/// 实测：M2M100-418M 与 NLLB-200-distilled-600M 均 12 层 / 16 头 / 头维 64。
#[derive(Debug)]
struct M2mGraph {
    past_names: Vec<String>,
    head_shape: [i64; 4],
}

impl M2mGraph {
    fn from_config(config: &serde_json::Value) -> Result<Self, VecboostError> {
        let field = |name: &str| -> Result<i64, VecboostError> {
            config.get(name).and_then(|v| v.as_i64()).ok_or_else(|| {
                VecboostError::ModelLoadError(format!(
                    "bundle config.json missing or invalid numeric field '{name}'"
                ))
            })
        };
        let decoder_layers = field("decoder_layers")?;
        let heads = field("encoder_attention_heads")?;
        let d_model = field("d_model")?;
        if decoder_layers <= 0 || heads <= 0 || d_model <= 0 || d_model % heads != 0 {
            return Err(VecboostError::ModelLoadError(format!(
                "invalid m2m graph geometry: decoder_layers={decoder_layers}, \
                 heads={heads}, d_model={d_model}"
            )));
        }
        let head_dim = d_model / heads;
        let past_names = (0..decoder_layers)
            .flat_map(|i| {
                [
                    format!("past_key_values.{i}.decoder.key"),
                    format!("past_key_values.{i}.decoder.value"),
                    format!("past_key_values.{i}.encoder.key"),
                    format!("past_key_values.{i}.encoder.value"),
                ]
            })
            .collect();
        Ok(Self {
            past_names,
            head_shape: [1, heads, 0, head_dim],
        })
    }
}

/// 模型家族（load 期按 bundle config.json `model_type` 分派；图输入/输出
/// 命名两族同构——input_ids / use_cache_branch / encoder_attention_mask /
/// encoder_hidden_states / past_key_values.N.{decoder,encoder}.{key,value} /
/// last_hidden_state / logits，实测 Xenova 导出）。
#[derive(Debug)]
enum Family {
    /// Helsinki Marian：6 层静态名表；tokenizer post_processor 自带
    /// `</s>`（T015：add_special_tokens 必须 true）
    Marian,
    /// Meta M2M100 / NLLB-200（同 m2m_100 架构）：源序列显式构造
    /// `[src_lang] + ids + [eos]`，decoder 起始 `[decoder_start, tgt_lang]`
    M2m {
        graph: M2mGraph,
        lang_style: LangTokenStyle,
    },
}

/// NLLB 语言码映射（ISO 639-1 及引擎契约的 639-3 等价码 →
/// NLLB-200 语言 token）。未收录码显性拒绝（不猜测）。
fn nllb_lang_token(iso: &str) -> Option<&'static str> {
    match iso {
        "en" => Some("eng_Latn"),
        "zh" | "zho" | "cmn" => Some("zho_Hans"),
        "yue" => Some("yue_Hant"),
        "fr" => Some("fra_Latn"),
        "de" => Some("deu_Latn"),
        "es" => Some("spa_Latn"),
        "pt" => Some("por_Latn"),
        "it" => Some("ita_Latn"),
        "ru" => Some("rus_Cyrl"),
        "ja" => Some("jpn_Jpan"),
        "ko" => Some("kor_Hang"),
        "ar" => Some("arb_Arab"),
        "nl" => Some("nld_Latn"),
        "pl" => Some("pol_Latn"),
        "tr" => Some("tur_Latn"),
        "vi" => Some("vie_Latn"),
        "th" => Some("tha_Thai"),
        "id" => Some("ind_Latn"),
        _ => None,
    }
}

impl Family {
    /// 从 bundle config.json + tokenizer 探测家族（load 期；未支持
    /// model_type 显性 ModelLoadError）
    fn detect(config: &serde_json::Value, tokenizer: &Tokenizer) -> Result<Self, VecboostError> {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match model_type {
            "marian" => Ok(Family::Marian),
            "m2m_100" => {
                let graph = M2mGraph::from_config(config)?;
                let lang_style = if tokenizer.token_to_id("__en__").is_some() {
                    LangTokenStyle::M2m
                } else if tokenizer.token_to_id("eng_Latn").is_some() {
                    LangTokenStyle::Nllb
                } else {
                    return Err(VecboostError::ModelLoadError(
                        "m2m_100 bundle tokenizer exposes no known language tokens \
                         (neither '__en__' nor 'eng_Latn')"
                            .to_string(),
                    ));
                };
                Ok(Family::M2m { graph, lang_style })
            }
            other => Err(VecboostError::ModelLoadError(format!(
                "unsupported translation model_type '{other}'; supported: marian, \
                 m2m_100 (M2M100 / NLLB)"
            ))),
        }
    }

    /// 语言码 → 语言 token id（请求期；未支持码 InvalidInput 显性拒绝）
    fn lang_token_id(&self, tokenizer: &Tokenizer, iso: &str) -> Result<u32, VecboostError> {
        let token: String = match self {
            Family::Marian => {
                return Err(VecboostError::InvalidInput(
                    "marian bundles are pair-specific and take no language token".to_string(),
                ));
            }
            Family::M2m {
                lang_style: LangTokenStyle::M2m,
                ..
            } => format!("__{iso}__"),
            Family::M2m {
                lang_style: LangTokenStyle::Nllb,
                ..
            } => nllb_lang_token(iso)
                .ok_or_else(|| {
                    VecboostError::InvalidInput(format!(
                        "language code '{iso}' is not supported by this NLLB bundle; \
                         supported examples: en, zh, yue, fr, de, es, pt, it, ru, ja, ko, \
                         ar, nl, pl, tr, vi, th, id"
                    ))
                })?
                .to_string(),
        };
        tokenizer.token_to_id(&token).ok_or_else(|| {
            VecboostError::InvalidInput(format!(
                "language token '{token}' not present in bundle vocabulary"
            ))
        })
    }
}

/// ONNX seq2seq 翻译引擎：bundle 自持（encoder/decoder/tokenizer/
/// generation_config）。
pub(crate) struct OnnxTranslationEngine {
    sessions: Arc<Mutex<TranslateSessions>>,
    /// 句级并行 worker 的附加 Session 集（workers-1 个；ort rc.13 禁并发 Run
    /// 同一 Session，每 worker 独占一套）。惰性加载（wsl2-oom-hardening D2）：
    /// init 期为空、idle 驻留仅主集一份，首次长输入并行请求按需补建。
    /// 空 = 并行关闭（整段直解路径）。
    parallel_sessions: Mutex<Vec<Arc<Mutex<TranslateSessions>>>>,
    /// init 期已探明的 bundle 内 encoder/decoder 路径（惰性补建 worker 集复用）
    encoder_file: PathBuf,
    decoder_file: PathBuf,
    /// init 期装配设备；fallback 已触发后惰性补建改走 Cpu（与主集重建同语义）
    session_device: DeviceType,
    tokenizer: Tokenizer,
    decoder_start_token_id: u32,
    eos_token_id: u32,
    /// 模型家族（ Marian / M2M100 / NLLB，load 期探测）
    family: Family,
    /// 解码图静态契约（past KV 名表 + 空占位形状；贪心循环消费）
    decode_graph: DecodeGraph,
    /// per-checkpoint 翻译预算（加载期定值，热路径消费）
    params: TranslationParams,
    /// 译文输出修复表（加载期定值，出口按声明序字面替换；paper-translate-quality）
    output_fixes: Vec<OutputFix>,
    precision: Precision,
    /// 已探明的本地 bundle 目录（try_fallback_to_cpu 重建 CPU Session 用）
    bundle_dir: PathBuf,
    fallback_triggered: bool,
    fallback_lock: Arc<Mutex<()>>,
    /// 最近一次成功解码的生成 token 数（评审 converge T033：质量观测通道，
    /// wire 五键契约不扩——e2e 质量输出经引擎级查询读取）
    output_tokens_last: std::sync::atomic::AtomicU64,
}

/// 解码图静态契约：past KV 输入名表 + 空占位张量形状（seq_len=0）
struct DecodeGraph {
    past_names: Vec<String>,
    head_shape: [i64; 4],
}

impl DecodeGraph {
    fn for_marian() -> Self {
        Self {
            past_names: PAST_TENSOR_NAMES
                .iter()
                .map(|(name, _, _)| (*name).to_string())
                .collect(),
            head_shape: [1, 8, 0, 64],
        }
    }

    fn for_m2m(graph: &M2mGraph) -> Self {
        Self {
            past_names: graph.past_names.clone(),
            head_shape: graph.head_shape,
        }
    }
}

// probe_model_file / precision_for_model_file 本体已收编至 [`super::session`]（单一事实源）。

/// tokenizer 加载（T013 U1 换路 shim，见模块文档）：
/// ① 内存内移除 null `precompiled_charsmap` 的 Precompiled normalizer
/// （等价 identity 归一化）；② 过滤 added_tokens 中 content 不在 model
/// vocab 的条目（Xenova m2m100 导出含越词表 token——`Token 'ଙ' out of
/// vocabulary` 解析期即失败）；均不改磁盘 bundle，其余字段原样反序列化。
fn load_tokenizer(tokenizer_file: &Path) -> Result<Tokenizer, VecboostError> {
    let raw = std::fs::read_to_string(tokenizer_file).map_err(|e| {
        VecboostError::ModelLoadError(format!(
            "Failed to read tokenizer {}: {e}",
            tokenizer_file.display()
        ))
    })?;
    let mut value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        VecboostError::ModelFileCorrupted(format!(
            "tokenizer {} is not valid JSON: {e}",
            tokenizer_file.display()
        ))
    })?;
    let mut shims: Vec<&str> = Vec::new();
    if value["normalizer"]["type"] == "Precompiled"
        && value["normalizer"]["precompiled_charsmap"].is_null()
    {
        if let Some(obj) = value.as_object_mut() {
            obj.remove("normalizer");
        }
        shims.push("removed no-op Precompiled normalizer (null precompiled_charsmap)");
    }
    // 词表事实源先提取为 owned 集合（BPE 对象 / wordlevel 列表），避免与
    // 后续可变借用冲突。注意：只过滤 merges、不动 added_tokens——特殊
    // token（`</s>`/`<pad>` 等）合法地仅存在于 added_tokens 而不在 vocab
    // （marian 词表实测），过滤 added_tokens 会误删 EOS 锚点；BPE 构建
    // 的词表成员校验只发生在 merges（m2m100 的 546 个坏 merge 全在此）
    let vocab_keys: std::collections::HashSet<String> = match value["model"].get("vocab") {
        Some(serde_json::Value::Object(map)) => map.keys().cloned().collect(),
        Some(serde_json::Value::Array(list)) => list
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new().into_iter().collect(),
    };
    if !vocab_keys.is_empty()
        && let Some(merges) = value["model"]
            .get_mut("merges")
            .and_then(|v| v.as_array_mut())
    {
        let before = merges.len();
        merges.retain(|merge| {
            let parts: Vec<&str> = match merge {
                // 旧格式 "a b" 单字符串；新格式 ["a","b"] 二元组
                serde_json::Value::String(line) => line.split(' ').collect(),
                serde_json::Value::Array(list) => list.iter().filter_map(|v| v.as_str()).collect(),
                _ => return true,
            };
            !parts.is_empty()
                && parts.iter().all(|part| vocab_keys.contains(*part))
                && vocab_keys.contains(&parts.concat())
        });
        if merges.len() != before {
            shims.push("filtered merge entries referencing tokens missing from vocab");
        }
    }
    for shim in &shims {
        log::info!(
            "tokenizer {}: {shim} (in-memory; tokenizers 0.23 parse contract)",
            tokenizer_file.display()
        );
    }
    Tokenizer::from_str(&value.to_string()).map_err(|e| {
        VecboostError::ModelLoadError(format!(
            "Failed to load tokenizer {}: {e}",
            tokenizer_file.display()
        ))
    })
}

impl GenerationConfig {
    /// 解析 generation_config.json：decoder_start_token_id / eos_token_id
    /// 必填读取；缺项回落 tokenizer 特殊 token（Marian 口径 `</s>` 为 EOS、
    /// `pad` 作 decoder start）并 warn（显性降级，非静默）。
    fn from_bundle(bundle_dir: &Path, tokenizer: &Tokenizer) -> Result<Self, VecboostError> {
        let path = bundle_dir.join("generation_config.json");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) => {
                log::warn!(
                    "translation bundle {}: generation_config.json unreadable ({e}); \
                     falling back to tokenizer special tokens for decode anchors",
                    bundle_dir.display()
                );
                return Self::from_tokenizer_fallback(bundle_dir, tokenizer);
            }
        };
        let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            VecboostError::ModelFileCorrupted(format!(
                "generation_config.json in {} is not valid JSON: {e}",
                bundle_dir.display()
            ))
        })?;
        match (
            value["decoder_start_token_id"].as_u64(),
            value["eos_token_id"].as_u64(),
        ) {
            (Some(start), Some(eos)) => Ok(Self {
                decoder_start_token_id: start as u32,
                eos_token_id: eos as u32,
            }),
            _ => {
                log::warn!(
                    "translation bundle {}: generation_config.json missing decoder_start_token_id/eos_token_id; \
                     falling back to tokenizer special tokens for decode anchors",
                    bundle_dir.display()
                );
                Self::from_tokenizer_fallback(bundle_dir, tokenizer)
            }
        }
    }

    fn from_tokenizer_fallback(
        bundle_dir: &Path,
        tokenizer: &Tokenizer,
    ) -> Result<Self, VecboostError> {
        let eos = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<eos>"))
            .ok_or_else(|| {
                VecboostError::ModelLoadError(format!(
                    "translation bundle {}: no generation_config.json anchors and tokenizer \
                     exposes no </s>/<eos> special token — greedy decode cannot stop",
                    bundle_dir.display()
                ))
            })?;
        let start = tokenizer
            .token_to_id("<pad>")
            .or_else(|| tokenizer.token_to_id("pad"))
            .unwrap_or(eos);
        Ok(Self {
            decoder_start_token_id: start,
            eos_token_id: eos,
        })
    }
}

// Session 构建、probe/precision 已全部收编至 [`super::session`]（单一事实源）；
// 线程预算 `SESSION_INTRA_THREADS` 由各调用点读入 `SessionOpts.intra_threads`。

/// ort 调用错误 → InferenceError（wire 由上层 to_api_error 统一脱敏）
fn inference_err(step: &str) -> impl Fn(ort::Error) -> VecboostError + '_ {
    move |e: ort::Error| VecboostError::InferenceError(format!("translation {step} failed: {e}"))
}

/// encoder 单次前向：last_hidden_state → (shape, data) owned 返回
///（输出名 ≠ 输入名，U2 清单实测；owned 化避免视图借用跨函数存活）
fn run_encoder(
    encoder: &mut Session,
    source_ids: &[i64],
) -> Result<(Vec<i64>, Vec<f32>), VecboostError> {
    let source_len = source_ids.len() as i64;
    let outputs = encoder
        .run(ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, source_len], source_ids.to_vec()))
                .map_err(inference_err("encoder input_ids"))?,
            "attention_mask" => Tensor::from_array(
                (vec![1i64, source_len], vec![1i64; source_len as usize])
            )
            .map_err(inference_err("encoder attention_mask"))?,
        ])
        .map_err(inference_err("encoder run"))?;
    // 显性取键（审查 SEC-2 延伸）：缺键是图契约损坏，显性 InferenceError
    // 而非 HashMap Index panic
    let enc_output = outputs.get("last_hidden_state").ok_or_else(|| {
        VecboostError::InferenceError(
            "encoder graph output 'last_hidden_state' missing".to_string(),
        )
    })?;
    let (enc_shape, enc_data) = enc_output
        .try_extract_tensor::<f32>()
        .map_err(inference_err("encoder last_hidden_state"))?;
    Ok((enc_shape.to_vec(), enc_data.to_vec()))
}

/// 末位分布 argmax（total_cmp 全序，评审 PERF-3/ARC-6 语义不变）。EOS
/// 抑制：`exclude` 提供时该索引不参与——未达 min_new_tokens 窗口前排除
/// EOS（HF min_new_tokens 同语义，NLLB 质量评审实测贪心提前截断丢句）。
/// 全部被排除（vocab==1 病态图）返回 None，调用方回退含 EOS 的正常 argmax。
fn argmax_index(row: &[f32], exclude: Option<usize>) -> Option<usize> {
    row.iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

/// 贪心解码循环：每步对 [初始序列 + 已生成] 全量重算、取末位 logits argmax
///（无缓存分支，依据见模块文档「缓存分支禁用」）；EOS 停止、
/// `max_new_tokens` 硬顶、`min_new_tokens` 窗口内抑制 EOS。past 输入以
/// 零长占位张量（图输入必填，else 分支不消费值）；初始序列与 past 名表/
/// 形状按模型家族注入。
//
// 9 参为贪心解码热循环「编码器输出 / past / init」的自然分组，保持扁平签名
// 以免波及两处调用点；用 #[expect]（非 #[allow]）显式钉住，参数若降至阈值内
// 会反向告警提示可移除（设计文档 D4）。
#[expect(clippy::too_many_arguments)]
fn greedy_decode(
    decoder: &mut Session,
    enc_dims: &[i64],
    enc_data: &[f32],
    source_len: usize,
    max_new_tokens: usize,
    min_new_tokens: usize,
    decoder_init: &[i64],
    past_names: &[String],
    head_shape: [i64; 4],
    eos_token_id: u32,
) -> Result<Vec<u32>, VecboostError> {
    // 评审 PERF-2：循环不变输入循环外构造一次，经 TensorRef 视图借用调用方
    // 切片（源 512 token≈1MB）。注意：每步必须以 `&*view` 引用传入（命中
    // From<&Value> → SessionInputValue::View 零拷贝），不可 .clone()——
    // ValueRef 无 Clone，经 Deref 解析到 owned Value::clone（ort copy.rs）
    // 是深拷贝且内部运行 identity ORT session，热路径每步双份 memcpy
    let enc_hidden_view = ort::value::TensorRef::from_array_view((enc_dims, enc_data))
        .map_err(inference_err("encoder hidden view"))?;
    let attention_mask: Vec<i64> = vec![1; source_len];
    let mask_view = ort::value::TensorRef::from_array_view((
        [1i64, source_len as i64],
        attention_mask.as_slice(),
    ))
    .map_err(inference_err("attention mask view"))?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut decoder_ids: Vec<i64> = decoder_init.to_vec();
    for _step in 0..max_new_tokens {
        let dec_len = decoder_ids.len() as i64;
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, dec_len], decoder_ids.clone()))
                .map_err(inference_err("decoder input_ids"))?,
            "use_cache_branch" => Tensor::from_array((vec![1usize], vec![false]))
                .map_err(inference_err("decoder use_cache_branch"))?,
            "encoder_attention_mask" => &*mask_view,
            "encoder_hidden_states" => &*enc_hidden_view,
        ];
        // 静态名表：消除每步 format! 堆分配（家族 load 期生成，Marian 24 /
        // m2m 12 层 48）；空占位张量零数据元素，每步重建成本可忽略
        for name in past_names {
            inputs.push((
                Cow::Borrowed(name.as_str()),
                Tensor::from_array((head_shape, Vec::<f32>::new()))
                    .map_err(inference_err("decoder past placeholder"))?
                    .into(),
            ));
        }
        let outputs = decoder.run(inputs).map_err(inference_err("decoder step"))?;

        // 显性取键（审查 SEC-2 延伸）：缺键是图契约损坏，显性 InferenceError
        // 而非 HashMap Index panic
        let logits_output = outputs.get("logits").ok_or_else(|| {
            VecboostError::InferenceError("decoder graph output 'logits' missing".to_string())
        })?;
        let (logits_shape, logits_data) = logits_output
            .try_extract_tensor::<f32>()
            .map_err(inference_err("decoder logits"))?;
        // 评审 SEC-2：图契约运行时校验（release 生效，替代仅 debug 的
        // debug_assert）——换导出仓库/量化变体时显性 InferenceError
        // 附实际形状，而非 index-out-of-bounds panic
        if let Err(msg) = validate_logits_shape(logits_shape, dec_len, logits_data.len()) {
            return Err(VecboostError::InferenceError(format!(
                "decoder logits contract mismatch: {msg}"
            )));
        }
        // 末位 token 的分布（logits [1, dec_len, V]，取最后一行）；
        // EOS 抑制窗口内排除 EOS（vocab==1 病态图回退正常 argmax 即停止）
        let vocab = logits_shape[2] as usize;
        let last_row = &logits_data[(dec_len as usize - 1) * vocab..dec_len as usize * vocab];
        // 评审 PERF-3/ARC-6：total_cmp 全序比较（NaN 排序确定不 panic、
        // 不中毒锁；与 candle_decision 的容忍先例同语义强度）
        let exclude = (generated.len() < min_new_tokens).then_some(eos_token_id as usize);
        let best = argmax_index(last_row, exclude)
            .or_else(|| argmax_index(last_row, None))
            .expect("logits 非空");
        let best = best as u32;

        if best == eos_token_id {
            break;
        }
        generated.push(best);
        decoder_ids.push(best as i64);
    }
    Ok(generated)
}

impl OnnxTranslationEngine {
    /// 从 bundle 目录加载：encoder/decoder 按 [`ENCODER_CANDIDATES`]/
    /// [`DECODER_CANDIDATES`] 探测，tokenizer.json 必填（bundle 根，
    /// 经 `resolve_tokenizer_path` 共享契约：显式路径 → 根 → 子目录）。
    pub(crate) fn load(config: &ModelConfig) -> Result<Self, VecboostError> {
        // translation_params 理论恒经合法路径构造（TranslationParams::new
        // 校验），字段全 pub 可库内直构——加载期防御性复检（decision_params
        // 同款先例），先于资产探测显性拒绝
        let params = match config.translation_params {
            Some(ref p) => {
                TranslationParams::new(p.max_source_tokens, p.max_new_tokens, p.min_new_tokens)?
            }
            None => TranslationParams::default(),
        };
        let bundle_dir = config.model_path.clone();
        if !bundle_dir.is_dir() {
            return Err(VecboostError::ModelLoadError(format!(
                "translation bundle path is not a directory: {}",
                bundle_dir.display()
            )));
        }
        let encoder_file = probe_model_file(&bundle_dir, &ENCODER_CANDIDATES, "encoder")?;
        let decoder_file = probe_model_file(&bundle_dir, &DECODER_CANDIDATES, "decoder")?;

        if let Some(ref expected_hash) = config.model_sha256 {
            // 口径同 decision：sha256 仅覆盖探测命中的主模型（此处双模型，
            // 任一失配即拒绝）
            for model_file in [&encoder_file, &decoder_file] {
                let is_valid = crate::utils::hash::verify_sha256(model_file, expected_hash)
                    .map_err(|e| {
                        VecboostError::ModelLoadError(format!("Failed to verify SHA256: {e}"))
                    })?;
                if !is_valid {
                    return Err(VecboostError::ModelLoadError(format!(
                        "Model file SHA256 verification failed. Expected: {expected_hash}, File: {model_file:?}"
                    )));
                }
            }
        }

        let tokenizer_file = super::local_bundle::resolve_tokenizer_path(
            &bundle_dir,
            config.tokenizer_path.as_deref(),
        )?;
        let mut tokenizer = load_tokenizer(&tokenizer_file)?;
        // 翻译协议自管窗口（encode 后二次校验 max_source_tokens），
        // bundle 内 truncation/padding 配置必须清除，否则序列被静默截断/
        // 填充（decision 同款防线）
        tokenizer.with_truncation(None).map_err(|e| {
            VecboostError::ModelLoadError(format!("Failed to clear truncation: {e}"))
        })?;
        tokenizer.with_padding(None);

        let generation = GenerationConfig::from_bundle(&bundle_dir, &tokenizer)?;
        // 家族探测：bundle config.json model_type 分派（marian / m2m_100），
        // 未支持类型启动期 fail-fast（多厂商模型支持：Meta M2M100/NLLB）
        let bundle_config: serde_json::Value = {
            let path = bundle_dir.join("config.json");
            let raw = std::fs::read_to_string(&path).map_err(|e| {
                VecboostError::ModelLoadError(format!(
                    "Failed to read bundle config {}: {e}",
                    path.display()
                ))
            })?;
            serde_json::from_str(&raw).map_err(|e| {
                VecboostError::ModelFileCorrupted(format!(
                    "bundle config {} is not valid JSON: {e}",
                    path.display()
                ))
            })?
        };
        let family = Family::detect(&bundle_config, &tokenizer)?;
        let decode_graph = match &family {
            Family::Marian => DecodeGraph::for_marian(),
            Family::M2m { graph, .. } => DecodeGraph::for_m2m(graph),
        };
        let intra_threads = SESSION_INTRA_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        let sessions =
            build_translate_sessions(&encoder_file, &decoder_file, intra_threads, &config.device)?;
        // 句级并行 worker 集：惰性加载（wsl2-oom-hardening D2）。ORT 对标准
        // .onnx 每 Session 独占一份权重堆拷贝（跨 Session 不共享），此前 init
        // 预载 4 pair × 4 worker 全 FP32 曾致 ~30G+ 常驻 idle；现 idle 驻留
        // 收敛为每 pair 一套主集，首次长输入并行请求经 ensure_parallel_sessions
        // 按需补建。P7 时延契约不变（worker 数来源与缺省均不动）。
        let parallel_workers = translation_parallel_workers();
        log::info!(
            "translation parallel workers={parallel_workers} (lazy; extra session sets \
             build on first long-parallel request)"
        );
        let precision = precision_for_model_file(&decoder_file);

        log::info!(
            "Translation engine initialized: bundle={}, encoder={:?}, decoder={:?}, \
             precision={:?}, max_source_tokens={}, max_new_tokens={}, \
             decoder_start_token_id={}, eos_token_id={}, past_tensors={}",
            bundle_dir.display(),
            encoder_file,
            decoder_file,
            precision,
            params.max_source_tokens,
            params.max_new_tokens,
            generation.decoder_start_token_id,
            generation.eos_token_id,
            decode_graph.past_names.len(),
        );

        Ok(Self {
            sessions: Arc::new(Mutex::new(sessions)),
            parallel_sessions: Mutex::new(Vec::new()),
            encoder_file,
            decoder_file,
            session_device: config.device.clone(),
            tokenizer,
            decoder_start_token_id: generation.decoder_start_token_id,
            eos_token_id: generation.eos_token_id,
            family,
            decode_graph,
            output_fixes: params.output_fixes.clone(),
            params,
            precision,
            bundle_dir,
            fallback_triggered: false,
            fallback_lock: Arc::new(Mutex::new(())),
            output_tokens_last: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// 最近一次成功解码的生成 token 数（`translate` 成功后更新；质量观测
    /// 通道，wire 契约不变）
    pub fn last_output_tokens(&self) -> u64 {
        self.output_tokens_last
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 翻译本体：两遍策略——第一遍整段直解（最大上下文：实体消歧、句间衔接
    /// 最佳）；经确定性退化检测（4-gram 重复循环 / 压满预算不出 EOS，见
    /// [`is_degenerate_generation`]）命中时第二遍按句切分逐句解码救援（消除
    /// 上下文相关失效，带全局输出预算与句数上限）。Session 锁临界区仅含
    /// 推理（tokenize/detok/切分均在锁外，架构审查 L1）；之后按目标语言
    /// 拼接（CJK 直接相连）→ CJK 标点归一 + 字间空格清理。
    pub(crate) fn translate(
        &self,
        req: &TranslateRequest,
    ) -> Result<TranslateResponse, VecboostError> {
        let started = std::time::Instant::now();
        // 源语言解析：显式码优先；缺省时引擎侧 whatlang 兜底（m2m 家族
        // 需要具体语言码构造 src token；Marian 不消费，维持 None 直通）
        let source_language = match (&self.family, req.source_language.as_deref()) {
            (Family::M2m { .. }, Some(iso)) => iso.to_string(),
            (Family::M2m { .. }, None) => {
                let detection = crate::text::detect_language(&req.text, None);
                if !detection.is_reliable || detection.language.is_empty() {
                    return Err(VecboostError::InvalidInput(crate::i18n::tr(
                        "translate-detect-failed",
                    )));
                }
                detection.language
            }
            (Family::Marian, _) => String::new(),
        };
        // 译前 token 保护（R-001 设计）：URL/LaTeX 转义/arXiv·DOI 标识符/全大写
        // 缩写替换为占位符，译文出口按槽位回填——修复 seq2seq 对这类 token 的
        // 实测丢失（论文评测：DTI 缩写与 arXiv:1908.05156 被整段丢弃）。占位
        // 吞没风险由真 bundle 存活率单测钉住；未命中显性 warn，不静默丢内容。
        let (protected_text, protected_slots) = protect_tokens(&req.text);
        // 长输入句级并行（P7）：字符数与句数双达阈值且并行未关闭时走并行；
        // 短文本/单句走整段直解（保留跨句上下文）。decoder_init 两路共用。
        // worker 集惰性加载后「是否并行」以旋钮为准（=1 关闭），附加集由
        // decode_sentences_parallel 按需补建
        let sentences = split_source_sentences(&protected_text);
        let use_parallel = translation_parallel_workers() > 1
            && sentences.len() >= 2
            && protected_text.chars().count() >= PARALLEL_MIN_CHARS;

        // decoder 初始序列：Marian [start]；m2m [start, tgt_lang]（目标语言
        // token 由解码器强制作为首个生成位；未支持码显性拒绝）
        let decoder_init: Vec<i64> = match &self.family {
            Family::Marian => vec![self.decoder_start_token_id as i64],
            Family::M2m { .. } => {
                let tgt_token = self
                    .family
                    .lang_token_id(&self.tokenizer, &req.target_language)?;
                vec![self.decoder_start_token_id as i64, tgt_token as i64]
            }
        };

        let (pieces, generated_total) = if use_parallel {
            // 并行路径在 worker 内直接 detok 产出文本
            self.decode_sentences_parallel(&sentences, &source_language, &decoder_init)?
        } else {
            // 源序列构造（家族分派）+ 预算校验（锁外：tokenize 无 Session 依赖）
            let source_ids = match &self.family {
                Family::Marian => encode_source_ids(&self.tokenizer, &protected_text)?,
                Family::M2m { .. } => {
                    let src_token = self
                        .family
                        .lang_token_id(&self.tokenizer, &source_language)?;
                    let mut ids = vec![src_token as i64];
                    ids.extend(encode_raw_ids(&self.tokenizer, &protected_text)?);
                    ids.push(self.eos_token_id as i64);
                    ids
                }
            };
            check_token_budget(source_ids.len(), self.params.max_source_tokens, "text")?;

            // 第一遍整段直解。评审 PERF-3：panic 后恢复中毒锁（into_inner 取回
            // 内层数据）——每步解码输入独立构造、Session 无跨请求可变状态，恢复
            // 安全；否则单次 panic 即该 pair 永久 "lock poisoned" 直至重启
            let (generated, degenerate) = {
                let mut sessions = self
                    .sessions
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let (enc_dims, enc_data) = run_encoder(&mut sessions.encoder, &source_ids)?;
                let generated = greedy_decode(
                    &mut sessions.decoder,
                    &enc_dims,
                    &enc_data,
                    source_ids.len(),
                    self.params.max_new_tokens,
                    self.params.min_new_tokens,
                    &decoder_init,
                    &self.decode_graph.past_names,
                    self.decode_graph.head_shape,
                    self.eos_token_id,
                )?;
                // 退化判据 token 级（空生成 = 立即 EOS，同判退化转救援；
                // min_new_tokens 抑制窗口内不计 EOS 故空生成仅在窗口为 0 时可达）
                let degenerate = generated.is_empty()
                    || is_degenerate_generation(
                        &generated,
                        generated.len() >= self.params.max_new_tokens,
                    );
                (generated, degenerate)
            };

            // 第二遍救援：按句切分逐句解码（run1 评审实测：长段重复循环消除；
            // 代价是句间上下文丢失，仅作退化救援路径）。救援带全局输出 token
            // 预算与句数上限（审查 F2：无界救援按句数放大解码步数并占死 pair
            // 信号量），耗尽即截断并显性告警，禁止静默丢句
            let (generated_sets, generated_total) = if !degenerate {
                let generated_total = generated.len();
                (vec![generated], generated_total)
            } else {
                let sentences = split_source_sentences(&protected_text);
                log::warn!(
                    "translation whole-text decode degenerated ({} tokens generated, \
                 {} sentences); retrying per-sentence with output budget {}",
                    generated.len(),
                    sentences.len(),
                    self.params.max_new_tokens
                );
                if sentences.len() > MAX_RESCUE_SENTENCES {
                    log::warn!(
                        "rescue sentence count {} exceeds cap {MAX_RESCUE_SENTENCES}; \
                     remaining sentences dropped",
                        sentences.len()
                    );
                }
                let sentence_ids_list = sentences
                    .iter()
                    .take(MAX_RESCUE_SENTENCES)
                    .map(|sentence| encode_source_ids(&self.tokenizer, sentence))
                    .collect::<Result<Vec<_>, VecboostError>>()?;
                for sentence_ids in &sentence_ids_list {
                    check_token_budget(
                        sentence_ids.len(),
                        self.params.max_source_tokens,
                        "sentence",
                    )?;
                }
                let generated_sets = {
                    let mut sessions = self
                        .sessions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    let mut generated_sets = Vec::with_capacity(sentence_ids_list.len());
                    let mut rescue_budget = self.params.max_new_tokens;
                    let mut produced: usize = 0;
                    for sentence_ids in &sentence_ids_list {
                        if rescue_budget == 0 {
                            log::warn!(
                                "rescue output budget exhausted after {produced} tokens; \
                             remaining sentences dropped"
                            );
                            break;
                        }
                        let (sentence_dims, sentence_data) =
                            run_encoder(&mut sessions.encoder, sentence_ids)?;
                        let generated = greedy_decode(
                            &mut sessions.decoder,
                            &sentence_dims,
                            &sentence_data,
                            sentence_ids.len(),
                            rescue_budget,
                            // 抑制窗口不得超剩余预算（max ≥ min 不变式；预算
                            // 递减后窗口随之收缩，压满即按退化判据显性暴露）
                            self.params.min_new_tokens.min(rescue_budget),
                            &decoder_init,
                            &self.decode_graph.past_names,
                            self.decode_graph.head_shape,
                            self.eos_token_id,
                        )?;
                        rescue_budget = rescue_budget.saturating_sub(generated.len());
                        produced += generated.len();
                        generated_sets.push(generated);
                    }
                    generated_sets
                };
                let total: usize = generated_sets.iter().map(|g| g.len()).sum();
                (generated_sets, total)
            };
            // detok 统一锁外（架构审查 L1：临界区仅含推理）
            let mut pieces: Vec<String> = Vec::with_capacity(generated_sets.len());
            for generated in &generated_sets {
                let text = decode_output_text(&self.tokenizer, generated)?;
                if !text.is_empty() {
                    pieces.push(text);
                }
            }
            (pieces, generated_total)
        };

        self.output_tokens_last
            .store(generated_total as u64, std::sync::atomic::Ordering::Relaxed);
        let joined = pieces.join(translation_joiner(&req.target_language));
        let translation = normalize_cjk_punctuation(joined, &req.target_language);
        // 中/日目标额外清除字间/标点旁半角空格（Marian 解码系统性伪影，
        // 质量评审 run1 实测：“取代 ” ， 后者”“。 许多”“ — —”）。韩语
        // 不参与（空格有辨义作用），见 is_zh_ja_target 文档
        let translation = if is_zh_ja_target(&req.target_language) {
            collapse_cjk_spaces(&translation)
        } else {
            translation
        };
        // zh 目标繁→简归一化（m2m100 实测繁简混排："人類製作複雜"）
        let translation = if req.target_language.starts_with("zh") {
            simplify_chinese_output(translation)
        } else {
            translation
        };
        // 占位回填置于 CJK 归一化之后、符号修复之前：回填的原文（URL /
        // arXiv:ID / \% 等）随后统一经 repair 清创，半角形态不被 normalize
        // 本地化；output_fixes 仍可对回填后的内容生效
        let translation = backfill_protected(translation, &protected_slots);
        // 符号损伤确定性修复（URL 全角冒号 / LaTeX 转义反斜杠，论文评测实测）
        let translation = repair_symbol_damage(translation);
        // config 驱动译文修复表（按声明序字面替换）
        let translation = apply_output_fixes(translation, &self.output_fixes);
        Ok(TranslateResponse {
            translation,
            source_language: req.source_language.clone(),
            target_language: req.target_language.clone(),
            detected_source_language: None,
            processing_time_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// 惰性补建句级并行 worker 附加集（锁内顺序构建；首个并行请求付一次性
    /// 构建耗时，后续请求复用快照，无重复构建）。
    fn ensure_parallel_sessions(&self, worker_count: usize) -> Result<(), VecboostError> {
        let mut sets = self
            .parallel_sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let missing = missing_worker_sets(sets.len(), worker_count);
        if missing == 0 {
            return Ok(());
        }
        let device = if self.fallback_triggered {
            DeviceType::Cpu
        } else {
            self.session_device.clone()
        };
        let intra_threads = SESSION_INTRA_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        for _ in 0..missing {
            let sessions = build_translate_sessions(
                &self.encoder_file,
                &self.decoder_file,
                intra_threads,
                &device,
            )?;
            sets.push(Arc::new(Mutex::new(sessions)));
            log::info!(
                "translation parallel worker session set lazy-loaded ({}/{} extra sets ready)",
                sets.len(),
                worker_count - 1
            );
        }
        Ok(())
    }

    /// 句级并行解码（P7 长输入路径）：按句切分源文，worker 各领一句解码
    /// （每 worker 独占 Session 集——ort rc.13 禁并发 Run 同一 Session），
    /// 在 worker 内直接 detok 产出文本，按句序返回。任一句失败即整请求显性
    /// 失败（禁静默丢句）；句数超 MAX_RESCUE_SENTENCES 截断并 warn。
    fn decode_sentences_parallel(
        &self,
        sentences: &[String],
        source_language: &str,
        decoder_init: &[i64],
    ) -> Result<(Vec<String>, usize), VecboostError> {
        if sentences.len() > MAX_RESCUE_SENTENCES {
            log::warn!(
                "parallel sentence count {} exceeds cap {MAX_RESCUE_SENTENCES};                  remaining sentences dropped",
                sentences.len()
            );
        }
        let sentences: Vec<&String> = sentences.iter().take(MAX_RESCUE_SENTENCES).collect();
        let sentences = &sentences;
        self.ensure_parallel_sessions(translation_parallel_workers())?;
        // 快照 Arc 列表：锁只护惰性构建，不跨解码持有
        let parallel_snapshot: Vec<Arc<Mutex<TranslateSessions>>> = self
            .parallel_sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let pieces: std::sync::Mutex<Vec<(usize, String, usize)>> =
            std::sync::Mutex::new(Vec::with_capacity(sentences.len()));
        let failure: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        // worker 0 = 主 Session 集；1.. = 并行附加集
        let worker_sessions: Vec<&Arc<Mutex<TranslateSessions>>> = std::iter::once(&self.sessions)
            .chain(parallel_snapshot.iter())
            .collect();
        std::thread::scope(|scope| {
            for sessions in worker_sessions {
                let next = &next;
                let pieces = &pieces;
                let failure = &failure;
                scope.spawn(move || {
                    loop {
                        if failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .is_some()
                        {
                            return;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if i >= sentences.len() {
                            return;
                        }
                        let sentence = sentences[i];
                        // 家族分派句编码（与整段路径同构）
                        let ids = match self.sentence_ids(sentence, source_language) {
                            Ok(ids) => ids,
                            Err(e) => {
                                *failure
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                    Some(format!("sentence {i} encode: {e}"));
                                return;
                            }
                        };
                        if let Err(e) =
                            check_token_budget(ids.len(), self.params.max_source_tokens, "sentence")
                        {
                            *failure
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(format!("sentence {i} budget: {e}"));
                            return;
                        }
                        let decoded = {
                            let mut guard = sessions
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            (|| -> Result<(String, usize), VecboostError> {
                                let (dims, data) = run_encoder(&mut guard.encoder, &ids)?;
                                let generated = greedy_decode(
                                    &mut guard.decoder,
                                    &dims,
                                    &data,
                                    ids.len(),
                                    self.params.max_new_tokens,
                                    self.params.min_new_tokens,
                                    decoder_init,
                                    &self.decode_graph.past_names,
                                    self.decode_graph.head_shape,
                                    self.eos_token_id,
                                )?;
                                let text = decode_output_text(&self.tokenizer, &generated)?;
                                Ok((text, generated.len()))
                            })()
                        };
                        match decoded {
                            Ok((text, tokens)) => pieces
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .push((i, text, tokens)),
                            Err(e) => {
                                *failure
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                    Some(format!("sentence {i} decode: {e}"));
                                return;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failure
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Err(VecboostError::InferenceError(format!(
                "parallel translation failed: {e}"
            )));
        }
        let mut done = pieces
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        done.sort_by_key(|(i, _, _)| *i);
        let total: usize = done.iter().map(|(_, _, t)| *t).sum();
        Ok((done.into_iter().map(|(_, text, _)| text).collect(), total))
    }

    /// 单句源序列（家族分派，与整段路径同构）：Marian 直编；m2m 显式
    /// src lang token + eos 包裹。
    fn sentence_ids(
        &self,
        sentence: &str,
        source_language: &str,
    ) -> Result<Vec<i64>, VecboostError> {
        match &self.family {
            Family::Marian => encode_source_ids(&self.tokenizer, sentence),
            Family::M2m { .. } => {
                let src_token = self
                    .family
                    .lang_token_id(&self.tokenizer, source_language)?;
                let mut ids = vec![src_token as i64];
                ids.extend(encode_raw_ids(&self.tokenizer, sentence)?);
                ids.push(self.eos_token_id as i64);
                Ok(ids)
            }
        }
    }
}

/// 救援遍句数上限（审查 F2：句数无界时按句放大解码前向次数并占死 pair
/// 信号量；64 句覆盖正常段落，超出按截断显性告警）
const MAX_RESCUE_SENTENCES: usize = 64;

/// zh 目标输出繁→简归一化（paper-translate-quality：m2m100 en→zh 实测
/// 繁简混排——"人類可以努力製作高複雜的指示"）。
fn simplify_chinese_output(text: String) -> String {
    character_converter::traditional_to_simplified(&text).into_owned()
}

/// 符号损伤确定性修复：seq2seq 模型对符号 token 的实测损伤（论文评测）——
/// URL 协议分隔符冒号被本地化（"https：//"）与源文 LaTeX 转义忠实拷贝
/// （"90\%"）。仅处理无歧义模式：全角冒号仅出现在 "：//" 语境即归位；
/// 反斜杠仅清理 `\%`/`\$`/`\&` 三种转义，`\pi` 等命令不动。
fn repair_symbol_damage(text: String) -> String {
    text.replace("：//", "://")
        .replace("\\%", "%")
        .replace("\\$", "$")
        .replace("\\&", "&")
}

/// config 驱动译文修复表：from→to 字面全量替换，按声明序应用
/// （空 from 跳过，防误配置把整个译文替换掉）。
fn apply_output_fixes(mut text: String, fixes: &[OutputFix]) -> String {
    for fix in fixes {
        if !fix.from.is_empty() {
            text = text.replace(&fix.from, &fix.to);
        }
    }
    text
}

// ── 译前 token 保护与译后回填（R-001）────────────────────────────────────
//
// seq2seq 模型对 URL / LaTeX 转义 / arXiv·DOI 标识符 / 全大写缩写有实测丢失
// （论文评测：DTI 与 arXiv:1908.05156 被整段丢弃，出口修复无从恢复不存在
// 的内容）。译前把这些 token 替换为占位符 ⟦Pn⟧、译后按槽位回填原文，是
// 唯一能让源 token 穿过解码器的机制。占位吞没风险（模型不吐出占位符）由
// 真 bundle 存活率测试钉住；未命中显性 warn，绝不静默。

/// 保护槽位：占位符 ↔ 原文。
struct ProtectedSlot {
    placeholder: String,
    original: String,
}

/// 单请求保护条数上限：防病态输入把源文几乎全部变成占位符。
const PROTECT_MAX_SLOTS: usize = 64;

/// 全大写缩写停用词（与评测 acronyms() 同口径）：冠词/介词/系动词不保护。
const PROTECT_ACRONYM_STOPWORDS: &[&str] = &[
    "THE", "AND", "FOR", "WE", "IN", "ON", "TO", "OF", "IS", "ARE", "IT",
];

/// 四个保护模式（常量编译一次）：URL / arXiv·DOI 标识符 / LaTeX 转义 /
/// 全大写缩写。
fn protect_patterns() -> &'static (regex::Regex, regex::Regex, regex::Regex, regex::Regex) {
    use std::sync::OnceLock;
    static P: OnceLock<(regex::Regex, regex::Regex, regex::Regex, regex::Regex)> = OnceLock::new();
    P.get_or_init(|| {
        (
            regex::Regex::new(r"https?://[^\s]+").expect("const regex: url"),
            regex::Regex::new(r"(?:[Aa]rXiv)\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?|10\.\d{4,9}/[^\s]+")
                .expect("const regex: identifier"),
            regex::Regex::new(r"\\[%$&]").expect("const regex: latex escape"),
            regex::Regex::new(r"\b[A-Z][A-Z0-9]{1,}\b").expect("const regex: acronym"),
        )
    })
}

/// 回填匹配两级：严格带括号形态（`⟦Z1⟧`）与裸核心形态（`Z1`）——实测
/// m2m100 会丢弃 ⟦⟧ 括号但完整拷贝 "Z1" 核心（见真 bundle 存活率测试）。
/// Z+数字在学术文本中几乎不出现，裸匹配误配风险可忽略；槽位消费一次即标记
/// 已恢复，重复出现不二次注入。
fn placeholder_res() -> &'static (regex::Regex, regex::Regex) {
    use std::sync::OnceLock;
    static P: OnceLock<(regex::Regex, regex::Regex)> = OnceLock::new();
    P.get_or_init(|| {
        (
            regex::Regex::new("⟦\\s*Z\\s*(\\d+)\\s*⟧").expect("const regex: placeholder strict"),
            regex::Regex::new("Z(\\d+)").expect("const regex: placeholder bare"),
        )
    })
}

/// 区间重叠判定：半开区间 [s,e) 与已占用集合是否存在交叠。
fn spans_overlap(claimed: &[(usize, usize)], s: usize, e: usize) -> bool {
    claimed.iter().any(|&(cs, ce)| s < ce && cs < e)
}

/// 译前保护：按 URL > 标识符 > LaTeX 转义 > 缩写优先级占用区间（URL 内的
/// 缩写等不再二次保护），返回占位符化文本与槽位表（按出现序编号）。
fn protect_tokens(text: &str) -> (String, Vec<ProtectedSlot>) {
    let (url_re, ident_re, latex_re, acronym_re) = protect_patterns();
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for re in [url_re, ident_re, latex_re] {
        for m in re.find_iter(text) {
            if spans_overlap(&claimed, m.start(), m.end()) {
                continue;
            }
            claimed.push((m.start(), m.end()));
            spans.push((m.start(), m.end()));
        }
    }
    for m in acronym_re.find_iter(text) {
        if spans_overlap(&claimed, m.start(), m.end())
            || PROTECT_ACRONYM_STOPWORDS.contains(&m.as_str())
            || m.as_str().len() < 2
        {
            continue;
        }
        claimed.push((m.start(), m.end()));
        spans.push((m.start(), m.end()));
    }
    spans.sort_by_key(|&(s, _)| s);
    spans.truncate(PROTECT_MAX_SLOTS);
    if spans.is_empty() {
        return (text.to_string(), Vec::new());
    }
    let mut out = String::with_capacity(text.len());
    let mut slots = Vec::with_capacity(spans.len());
    let mut last = 0usize;
    for (i, (s, e)) in spans.iter().enumerate() {
        out.push_str(&text[last..*s]);
        let placeholder = format!("⟦Z{}⟧", i + 1);
        out.push_str(&placeholder);
        slots.push(ProtectedSlot {
            placeholder,
            original: text[*s..*e].to_string(),
        });
        last = *e;
    }
    out.push_str(&text[last..]);
    (out, slots)
}

/// 译后回填：占位符替换回原文；未命中（模型吞位）逐条 warn 显性暴露。
/// 两级匹配：先严格括号形态，后对仍未恢复的槽位试裸核心形态（实测模型
/// 丢括号保核心）；每个槽位只注入一次。
fn backfill_protected(text: String, slots: &[ProtectedSlot]) -> String {
    if slots.is_empty() {
        return text;
    }
    let (strict_re, bare_re) = placeholder_res();
    let mut recovered = vec![false; slots.len()];
    // 槽位索引 → 待替换区间（形态及命中外延由覆盖写控制：从后向前替换避免
    // 位移失效；先严格后裸核，重复命中同槽不二次注入）
    let mut replacements: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, slot_idx)
    for re in [strict_re, bare_re] {
        for m in re.find_iter(&text) {
            let digits: String = m.as_str().chars().filter(char::is_ascii_digit).collect();
            let Ok(idx) = digits.parse::<usize>() else {
                continue;
            };
            if idx == 0 || idx > slots.len() || recovered[idx - 1] {
                continue;
            }
            replacements.push((m.start(), m.end(), idx - 1));
            recovered[idx - 1] = true;
        }
    }
    replacements.sort_by_key(|&(s, _, _)| std::cmp::Reverse(s));
    let mut out = text;
    for (s, e, slot_idx) in replacements {
        out.replace_range(s..e, &slots[slot_idx].original);
    }
    for (i, slot) in slots.iter().enumerate() {
        if !recovered[i] {
            log::warn!(
                "protected token not recovered in translation (placeholder {}): {:?}",
                slot.placeholder,
                slot.original
            );
        }
    }
    out
}

/// 生成退化检测：opus-mt 长段整解的实测失效模式是重复循环（run1 评审：
/// 「伟大的替代理论 — — 一个伟大的替代理论 — —」）与压满解码预算不出
/// EOS。任一 4-gram 重复 ≥3 次或压满预算即判定退化。确定性规则，零模型
/// 参与（规则7：确定性决策不得交给模型）。
fn is_degenerate_generation(generated: &[u32], hit_budget: bool) -> bool {
    if hit_budget {
        return true;
    }
    if generated.len() < 8 {
        return false;
    }
    let mut seen: std::collections::HashMap<&[u32], usize> = std::collections::HashMap::new();
    for window in generated.windows(4) {
        let count = seen.entry(window).or_insert(0);
        *count += 1;
        if *count >= 3 {
            return true;
        }
    }
    false
}

/// 源文本编码（Marian：post_processor 自带 `</s>`——T015 实测缺它解码
/// 退化为重复循环，add_special_tokens 恒 true）
fn encode_source_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, VecboostError> {
    let encoding = tokenizer.encode(text, true).map_err(|e| {
        VecboostError::InferenceError(format!("translation tokenization failed: {e}"))
    })?;
    Ok(encoding.get_ids().iter().map(|&i| i as i64).collect())
}

/// 原始编码（m2m 家族：语言 token 与 eos 由引擎显式构造，
/// post_processor 的英文硬编码模板不适用非英语源）
fn encode_raw_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>, VecboostError> {
    let encoding = tokenizer.encode(text, false).map_err(|e| {
        VecboostError::InferenceError(format!("translation tokenization failed: {e}"))
    })?;
    Ok(encoding.get_ids().iter().map(|&i| i as i64).collect())
}

/// 源序列解码（跳过特殊 token + trim）
fn decode_output_text(tokenizer: &Tokenizer, generated: &[u32]) -> Result<String, VecboostError> {
    Ok(tokenizer
        .decode(generated, true)
        .map_err(|e| {
            VecboostError::InferenceError(format!("translation detokenization failed: {e}"))
        })?
        .trim()
        .to_string())
}

/// 源序列预算二次校验（显性拒绝，禁止静默截断——截断会丢源语义致译文错译）
fn check_token_budget(
    source_len: usize,
    max_source_tokens: usize,
    scope: &str,
) -> Result<(), VecboostError> {
    if source_len > max_source_tokens {
        // InvalidInput（to_api_error 映射 400）：design D2 承诺超限冒泡为
        // 400 语义；服务层预判为第一道，此处为引擎兜底（口径一致）
        return Err(VecboostError::InvalidInput(format!(
            "source {scope} exceeds token budget: {source_len} tokens > \
             {max_source_tokens} (max_source_tokens); shorten the text or \
             raise the pair's max_source_tokens limit"
        )));
    }
    Ok(())
}

/// CJK 目标语言半角标点归一：Marian 译文系统性带半角标点（质量评审
/// golden 抽验：zh 句含半角 `,` `?`，模型固有输出特性）；目标语言命中
/// 中/日/韩码族（639-1 及等价 639-3：zh/zho/cmn/yue、ja/jpn、ko/kor，
/// 配置面 target 为 `^[a-z]{2,3}$` 格式校验，两码空间均可配置到达）时
/// 归一为全角。数字两侧同时夹数字时不归一（千分位 `3,000`/时间 `12:30`
/// 误转即语义错误）；句号不归一（`.`→`。` 对小数/缩写有误伤风险，且
/// 模型句末已输出全角「。」）。
///
/// 引擎层后处理（TemperatureCalibration 同款先例）：golden 闸门、质量
/// 输出与四协议面度量/交付同一译文，服务层不重复处理。
fn normalize_cjk_punctuation(text: String, target_language: &str) -> String {
    if !matches!(
        target_language,
        "zh" | "zho" | "cmn" | "yue" | "ja" | "jpn" | "ko" | "kor"
    ) {
        return text;
    }
    // 快速路径：无半角目标标点（模型已输出全角/纯文本句）零分配直返
    if !text
        .bytes()
        .any(|b| matches!(b, b',' | b';' | b':' | b'?' | b'!'))
    {
        return text;
    }
    let chars: Vec<char> = text.chars().collect();
    // 映射为 1B ASCII → 3B UTF-8，最坏全句映射：len × 3 为容量上界
    let mut out = String::with_capacity(text.len() * 3);
    for (i, &ch) in chars.iter().enumerate() {
        let mapped = match ch {
            ',' => '，',
            ';' => '；',
            ':' => '：',
            '?' => '？',
            '!' => '！',
            _ => ch,
        };
        let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
        let next_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
        if mapped != ch && prev_digit && next_digit {
            out.push(ch);
        } else {
            out.push(mapped);
        }
    }
    out
}

/// 句终标点连串后吸收的闭引号/闭括号（句界归属前句）
const SENTENCE_CLOSERS: [char; 8] = ['"', '\'', '”', '’', ')', ']', '】', '」'];

/// 缩写守卫表（小写、去点后比对）：句点命中即不切分，误切丢语义
const SENTENCE_ABBREVIATIONS: [&str; 27] = [
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie", "al", "no", "fig",
    "inc", "ltd", "co", "dept", "univ", "approx", "cf", "pp", "vol", "us", "uk", "eu",
];

/// 判定句点前片段是否缩写或单字母首字母（"U.S." / "e.g." / "X. Kendi"）
fn is_abbreviation_or_initial(chars: &[char]) -> bool {
    let mut reversed = String::new();
    for &c in chars.iter().rev() {
        if c.is_alphanumeric() || c == '.' {
            reversed.push(c.to_ascii_lowercase());
        } else {
            break;
        }
    }
    let word: String = reversed.chars().rev().collect();
    let word = word.replace('.', "");
    if word.is_empty() {
        return false;
    }
    if word.chars().count() == 1 && word.chars().next().is_some_and(char::is_alphabetic) {
        return true;
    }
    SENTENCE_ABBREVIATIONS.contains(&word.as_str())
}

/// 按句切分源文本：opus-mt 以句对训练，多句整段直解在长输入上实测产生
/// 重复循环、从句并装错误与实体退化（质量评审 run1），逐句解码为确定性
/// 前置手段。句界 = 句终标点（`.!?…`）连串 + 紧随闭引号/闭括号 + 后随
/// 空白/文末；缩写、单字母首字母、小数与「标点后紧跟实字符/数字」不切。
/// 无句终标点的文本单片段返回。
fn split_source_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        if !matches!(chars[i], '.' | '!' | '?' | '…') {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && matches!(chars[j], '.' | '!' | '?' | '…') {
            j += 1;
        }
        while j < chars.len() && SENTENCE_CLOSERS.contains(&chars[j]) {
            j += 1;
        }
        // 边界不成立：标点后紧跟实字符（小数 `3.5`、域名等）——前移续扫
        if j < chars.len() && !chars[j].is_whitespace() {
            i = j;
            continue;
        }
        let next = chars[j..]
            .iter()
            .position(|c| !c.is_whitespace())
            .map_or(chars.len(), |p| j + p);
        // 编号列表（"… . 5)"）后随数字：保守不切分
        if next < chars.len() && chars[next].is_ascii_digit() {
            i = j;
            continue;
        }
        if matches!(chars[i], '.') && is_abbreviation_or_initial(&chars[start..i]) {
            i = j;
            continue;
        }
        let piece: String = chars[start..j].iter().collect();
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            pieces.push(trimmed.to_string());
        }
        start = next;
        i = next;
    }
    if start < chars.len() {
        let tail: String = chars[start..].iter().collect();
        let trimmed = tail.trim();
        if !trimmed.is_empty() {
            pieces.push(trimmed.to_string());
        }
    }
    if pieces.is_empty() {
        let whole = text.trim();
        if !whole.is_empty() {
            return vec![whole.to_string()];
        }
    }
    pieces
}

/// 分句译文拼接符：中/日句末自带句读直接相连；韩语与其余语言以空格相连
fn translation_joiner(target_language: &str) -> &'static str {
    if is_zh_ja_target(target_language) {
        ""
    } else {
        " "
    }
}

/// 中/日目标语言码族（639-1 及等价 639-3）：分句拼接零分隔与字间空格清理
/// 共用判据（架构审查 M2：单一事实源）。注意 `normalize_cjk_punctuation`
/// 为刻意超集（含韩语——标点归一对韩语适用；空格清理不适用，韩语空格
/// 有辨义作用）。
fn is_zh_ja_target(target_language: &str) -> bool {
    matches!(target_language, "zh" | "zho" | "cmn" | "yue" | "ja" | "jpn")
}

/// 中日文排印字符判定（清理判定用）：CJK 统一表意及扩展、假名、CJK/全角
/// 标点、破折号/弯引号/省略号/间隔号。韩文不参与（空格有辨义作用）。
fn is_cjk_typography_char(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F   // CJK 标点（、。〈〉《》「」『』【】〔〕）
        | 0x3040..=0x30FF // 平假名/片假名
        | 0x3400..=0x4DBF // CJK 扩展 A
        | 0x4E00..=0x9FFF // CJK 统一表意
        | 0xF900..=0xFAFF // CJK 兼容表意
        | 0xFF00..=0xFFEF // 全角形式（，！？：；）
        | 0x00B7          // 间隔号 ·
        | 0x2014          // 破折号 —
        | 0x2018..=0x201F // 弯引号
        | 0x2026) // 省略号 …
}

/// 清除中日译文中的字间/标点旁半角空格（Marian 解码系统性伪影，质量评审
/// 实测：“取代 ” ， 后者”“。 许多”“— —”）；词内拉丁空格（"Elon Musk"）
/// 保留。引擎层后处理（normalize 同款先例）：golden 闸门、质量输出与
/// 四协议面交付同一译文。
fn collapse_cjk_spaces(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (k, &ch) in chars.iter().enumerate() {
        if ch == ' ' {
            let prev = k > 0 && is_cjk_typography_char(chars[k - 1]);
            let next = k + 1 < chars.len() && is_cjk_typography_char(chars[k + 1]);
            if prev || next {
                continue;
            }
        }
        out.push(ch);
    }
    out
}

use std::borrow::Cow;

#[async_trait::async_trait]
impl InferenceEngine for OnnxTranslationEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        // 诚实语义：seq2seq 翻译引擎不产向量，embed 显性拒绝（非静默空实现）
        Err(VecboostError::unsupported_task(
            "translation engine does not produce embeddings; use an embedding engine".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::unsupported_task(
            "translation engine does not produce embeddings; use an embedding engine".to_string(),
        ))
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        task == ModelTask::Translate
    }

    fn supports_rerank(&self) -> bool {
        false
    }

    fn decide(
        &self,
        _req: &crate::domain::DecisionRequest,
    ) -> Result<crate::domain::DecisionResponse, VecboostError> {
        Err(VecboostError::unsupported_task(
            "translation engine does not support decision inference".to_string(),
        ))
    }

    fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        // 精确覆盖：bytes/4 估算对 CJK→en 翻译预算判定偏差过大
        Ok(self
            .tokenizer
            .encode(text, true)
            .map_err(|e| VecboostError::TokenizationError(format!("count_tokens failed: {e}")))?
            .get_ids()
            .len())
    }

    /// 用已探明的本地 bundle 路径重建 CPU Session（decision.rs 同模式，
    /// 无 HF 网络依赖）
    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        let _lock = self.fallback_lock.lock().map_err(|e| {
            VecboostError::InferenceError(format!("Failed to acquire fallback lock: {e}"))
        })?;
        if self.fallback_triggered {
            return Ok(());
        }
        log::info!("Attempting fallback to CPU for translation engine");
        let encoder_file = probe_model_file(&self.bundle_dir, &ENCODER_CANDIDATES, "encoder")?;
        let decoder_file = probe_model_file(&self.bundle_dir, &DECODER_CANDIDATES, "decoder")?;
        let intra_threads = SESSION_INTRA_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        let encoder = build_session(
            &encoder_file,
            &SessionOpts {
                intra_threads,
                device: DeviceType::Cpu,
                extra_config: &[],
            },
        )?;
        let decoder = build_session(
            &decoder_file,
            &SessionOpts {
                intra_threads,
                device: DeviceType::Cpu,
                extra_config: &[],
            },
        )?;
        {
            let mut guard = self
                .sessions
                .lock()
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
            guard.encoder = encoder;
            guard.decoder = decoder;
        }
        self.precision = precision_for_model_file(&decoder_file);
        self.fallback_triggered = true;
        log::info!("Successfully fell back to CPU for translation engine");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试：探测/契约面（离线可跑）。真机贪心链路由 T025 E2E 覆盖。
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn opus_bundle() -> Option<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/opus-mt-en-zh");
        dir.is_dir().then_some(dir)
    }

    fn m2m100_bundle() -> Option<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/m2m100_418M");
        dir.is_dir().then_some(dir)
    }

    // ── 惰性 worker 集：补建数量纯函数（离线可测） ─────────────────────

    #[test]
    fn missing_worker_sets_computes_delta() {
        assert_eq!(missing_worker_sets(0, 4), 3);
        assert_eq!(missing_worker_sets(1, 4), 2);
        assert_eq!(missing_worker_sets(3, 4), 0);
        // worker=1 并行关闭：永不补建
        assert_eq!(missing_worker_sets(0, 1), 0);
        // 已有超量不裁剪
        assert_eq!(missing_worker_sets(5, 4), 0);
    }

    // ── 探测：资产缺失显性报错（离线可跑） ──────────────────────────────

    #[test]
    fn test_probe_missing_assets_reports_model_load_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = probe_model_file(dir.path(), &ENCODER_CANDIDATES, "encoder").unwrap_err();
        assert!(matches!(err, VecboostError::ModelLoadError(_)));
        let detail = err.error_detail();
        assert!(detail.contains("No encoder model found"), "err={detail}");
        // 尝试路径清单显性化（onnx_engine 同款）
        assert!(detail.contains("onnx/encoder_model.onnx"), "err={detail}");
        assert!(
            detail.contains("onnx/encoder_model_quantized.onnx"),
            "err={detail}"
        );
    }

    #[test]
    fn test_probe_prefers_fp32_over_quantized() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("onnx")).unwrap();
        // 只落 quantized → 命中 quantized
        std::fs::write(dir.path().join("onnx/encoder_model_quantized.onnx"), b"x").unwrap();
        let hit = probe_model_file(dir.path(), &ENCODER_CANDIDATES, "encoder").unwrap();
        assert!(hit.ends_with("encoder_model_quantized.onnx"));
        // 再落 fp32 → fp32 优先
        std::fs::write(dir.path().join("onnx/encoder_model.onnx"), b"x").unwrap();
        let hit = probe_model_file(dir.path(), &ENCODER_CANDIDATES, "encoder").unwrap();
        assert!(hit.ends_with("onnx/encoder_model.onnx"));
    }

    // ── tokenizer shim：null normalizer 移除（真 bundle 在位时） ────────

    // 评审 NEW-3：图契约校验纯函数六分支单测（漂移场景无真机覆盖，校验
    // 逻辑自身回归必须由单测钉住）
    #[test]
    fn test_validate_logits_shape_contract() {
        // 合法：[1, dec_len=3, V=256]，数据 3×256
        validate_logits_shape(&[1, 3, 256], 3, 3 * 256).unwrap();

        // 漂移一：rank ≠ 3
        let err = validate_logits_shape(&[1, 256], 3, 256).unwrap_err();
        assert!(
            err.contains("rank-3") && err.contains("[1, 256]"),
            "err={err}"
        );

        // 漂移二：batch ≠ 1（贪心解码恒 batch=1）
        let err = validate_logits_shape(&[2, 3, 256], 3, 2 * 3 * 256).unwrap_err();
        assert!(
            err.contains("batch=1") && err.contains("[2, 3, 256]"),
            "err={err}"
        );

        // 漂移三：dec_len 不符
        let err = validate_logits_shape(&[1, 5, 256], 3, 5 * 256).unwrap_err();
        assert!(
            err.contains("dec_len 3") && err.contains("[1, 5, 256]"),
            "err={err}"
        );

        // 漂移四：数据长度短少
        let err = validate_logits_shape(&[1, 3, 256], 3, 3 * 256 - 1).unwrap_err();
        assert!(err.contains("data length"), "err={err}");

        // 漂移五：vocab 维度为零
        let err = validate_logits_shape(&[1, 3, 0], 3, 0).unwrap_err();
        assert!(err.contains("vocab dimension is zero"), "err={err}");
    }

    #[test]
    fn test_tokenizer_shim_loads_xenova_marian() {
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        // from_file 对同文件会 panic（T013 红灯）——shim 后必须可加载且可用
        let encoding = tokenizer.encode("Hello", false).unwrap();
        assert!(!encoding.get_ids().is_empty());
        assert_eq!(tokenizer.get_vocab_size(true), VOCAB_DIM_EN_ZH);
    }

    // ── generation_config 解析 ──────────────────────────────────────────

    #[test]
    fn test_generation_config_parses_real_bundle() {
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        let config = GenerationConfig::from_bundle(&dir, &tokenizer).unwrap();
        // T013 实测值：decoder_start_token_id=65000(pad)、eos_token_id=0
        assert_eq!(config.decoder_start_token_id, 65000);
        assert_eq!(config.eos_token_id, 0);
    }

    #[test]
    fn test_generation_config_missing_falls_back_to_tokenizer() {
        // 无 generation_config.json 的空目录 + 真 tokenizer → 回落 </s>
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        let empty = tempfile::tempdir().unwrap();
        let config = GenerationConfig::from_bundle(empty.path(), &tokenizer).unwrap();
        assert_eq!(config.eos_token_id, tokenizer.token_to_id("</s>").unwrap());
    }

    #[test]
    fn test_generation_config_partial_keys_falls_back_without_panic() {
        // D2 回归：generation_config.json 存在但缺 decoder_start_token_id →
        // 穷尽 match 落 `_` 分支走 tokenizer 兜底，绝不触及旧实现里被
        // guard 保证安全的 unwrap（离线红线：不 panic、不硬失败）。
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        let partial = tempfile::tempdir().unwrap();
        std::fs::write(
            partial.path().join("generation_config.json"),
            r#"{"eos_token_id": 0}"#,
        )
        .unwrap();
        // 缺 decoder_start → `_` 兜底：decoder_start 取 `<pad>` 特殊 token
        // （Marian 口径，R-translation-decode-004），确证走兜底分支且全程无 panic/Err。
        let config = GenerationConfig::from_bundle(partial.path(), &tokenizer)
            .expect("缺 decoder_start 应落 tokenizer 兜底而非 panic/Err");
        assert_eq!(
            config.decoder_start_token_id,
            tokenizer.token_to_id("<pad>").unwrap()
        );
    }

    // ── 真机贪心链路（bundle 守卫；T025 E2E 做逐句 golden，此处钉实现级
    //    可译性：加载→编码→解码→detok 全链不炸且译文非空） ─────────────

    #[test]
    fn test_translate_real_bundle_end_to_end() {
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !std::path::Path::new(&dylib).is_file() {
            eprintln!("SKIP: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let config = ModelConfig {
            name: "opus-smoke".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: dir,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Translate,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        };
        let engine = OnnxTranslationEngine::load(&config).expect("bundle 就绪时必须可加载");
        // 评审 ARC-8：supports_task 真实断言（替代已删的同义反复恒真测试）
        assert!(engine.supports_task(ModelTask::Translate));
        assert!(!engine.supports_task(ModelTask::Embedding));
        assert!(!engine.supports_task(ModelTask::Decision));
        let req = TranslateRequest {
            text: "Hello, my dog is cute".to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        };
        // trait 分发面（引擎经 AnyEngine 由 T016 接线；此处直接调固有方法）
        let resp = OnnxTranslationEngine::translate(&engine, &req).expect("贪心解码必须成功");
        assert!(!resp.translation.trim().is_empty(), "译文非空");
        assert!(
            !resp.translation.contains("</s>") && !resp.translation.contains("<pad>"),
            "特殊 token 不得泄漏: {:?}",
            resp.translation
        );
        println!("SMOKE translate(en→zh): {:?}", resp.translation);
        assert_eq!(resp.target_language, "zh");
        assert!(resp.processing_time_ms > 0 || !resp.translation.is_empty());
        // 评审 converge T033：生成 token 数透出通道（>0 断言）
        assert!(
            engine.last_output_tokens() > 0,
            "成功解码后 last_output_tokens 必须 > 0"
        );
    }

    // ── 域行为（真机 bundle 守卫测试内断言 supports_task；mock-free 的
    //    形式主义测试已按评审 ARC-8 删除）────────────────────────────────

    #[test]
    fn test_repair_symbol_damage_url_colon_and_latex_escape() {
        // URL 协议分隔符被本地化（论文评测实测）→ 归位半角
        assert_eq!(
            repair_symbol_damage("代码见 https：//github.com/x/y".to_string()),
            "代码见 https://github.com/x/y"
        );
        // LaTeX 转义反斜杠清理（源文 "90\%" 被忠实拷贝进译文）
        assert_eq!(
            repair_symbol_damage("准确率达 90\\%".to_string()),
            "准确率达 90%"
        );
        assert_eq!(repair_symbol_damage("成本 \\$5".to_string()), "成本 $5");
        // 非转义命令不动（\pi 等保留）
        assert_eq!(
            repair_symbol_damage("\\pi 近似 3.14".to_string()),
            "\\pi 近似 3.14"
        );
    }

    // ── 译前保护 / 译后回填（R-001）────────────────────────────────────

    #[test]
    fn test_protect_tokens_roundtrip_all_classes() {
        let text = "We study drug-target interaction (DTI) prediction, see \
                    arXiv:1908.05156 and https://github.com/x/y with 90\\% gain, \
                    DOI 10.1234/abcd.5678 for GPT and LLM workloads.";
        let (protected, slots) = protect_tokens(text);
        // 五类 token 全部入槽：DTI / arXiv / URL / \% / DOI / GPT / LLM
        let originals: Vec<&str> = slots.iter().map(|s| s.original.as_str()).collect();
        for expect in [
            "DTI",
            "arXiv:1908.05156",
            "https://github.com/x/y",
            "\\%",
            "10.1234/abcd.5678",
            "GPT",
            "LLM",
        ] {
            assert!(
                originals.contains(&expect),
                "missing {expect} in {originals:?}"
            );
        }
        // 原文不再直现（占位符已替换）
        assert!(!protected.contains("DTI"));
        assert!(!protected.contains("github.com"));
        assert!(!protected.contains("1908.05156"));
        // 占位符编号连续且唯一
        let ph: Vec<&str> = slots.iter().map(|s| s.placeholder.as_str()).collect();
        assert_eq!(ph[0], "⟦Z1⟧");
        assert_eq!(ph[ph.len() - 1], format!("⟦Z{}⟧", slots.len()).as_str());
        // 回填还原（译文字序中占位符按同序出现）
        let translated = slots
            .iter()
            .map(|s| s.placeholder.clone())
            .collect::<Vec<_>>()
            .join("，");
        assert_eq!(backfill_protected(translated, &slots), originals.join("，"));
    }

    #[test]
    fn test_protect_tokens_skips_stopwords_and_overlaps() {
        // 停用词（WE/THE 等）不占槽；URL 内的缩写（NLPXucan 非全大写，取
        // 全大写段）不因 URL 已占用而重复保护
        let text = "WE visit THE site https://EXAMPLE.com/ABC for data.";
        let (protected, slots) = protect_tokens(text);
        let originals: Vec<&str> = slots.iter().map(|s| s.original.as_str()).collect();
        assert!(!originals.contains(&"WE") && !originals.contains(&"THE"));
        // URL 整体一条（内部 ABC 不单独立槽）
        assert_eq!(originals, vec!["https://EXAMPLE.com/ABC"]);
        assert_eq!(protected, "WE visit THE site ⟦Z1⟧ for data.");
    }

    #[test]
    fn test_backfill_protected_tolerates_spacing_and_missing() {
        let slots = vec![ProtectedSlot {
            placeholder: "⟦Z1⟧".to_string(),
            original: "arXiv:1908.05156".to_string(),
        }];
        // 模型丢弃括号只留核心（实测 m2m100 行为）仍可回填
        assert_eq!(
            backfill_protected("版本为 Z1。".to_string(), &slots),
            "版本为 arXiv:1908.05156。"
        );
        // 带括号且内插空白形态
        assert_eq!(
            backfill_protected("版本为 ⟦ Z1 ⟧。".to_string(), &slots),
            "版本为 arXiv:1908.05156。"
        );
        // 重复命中同槽只注入一次
        assert_eq!(
            backfill_protected("Z1 与 Z1。".to_string(), &slots),
            "arXiv:1908.05156 与 Z1。"
        );
        // 缺失占位符：原文不注入、文本原样透出（warn 由实现承担）
        assert_eq!(
            backfill_protected("版本号丢失。".to_string(), &slots),
            "版本号丢失。"
        );
        // 空槽位零开销
        assert_eq!(backfill_protected("原文".to_string(), &[]), "原文");
    }

    #[test]
    fn test_protect_tokens_empty_input() {
        let (text, slots) = protect_tokens("");
        assert!(text.is_empty());
        assert!(slots.is_empty());
    }

    /// 真 bundle 存活率（R-001 验收核心）：m2m100 en→zh 必须把占位符吐回，
    /// DTI 缩写与 arXiv 标识符经保护后出现在译文中。资产缺失即 SKIP。
    #[test]
    fn test_protect_tokens_real_bundle_survival() {
        let Some(dir) = m2m100_bundle() else {
            eprintln!("SKIP: models/m2m100_418M 缺失（离线环境）");
            return;
        };
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !std::path::Path::new(&dylib).is_file() {
            eprintln!("SKIP: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let config = ModelConfig {
            name: "m2m100-protect".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: dir,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Translate,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        };
        let engine = OnnxTranslationEngine::load(&config).expect("bundle 就绪时必须可加载");
        let req = TranslateRequest {
            text: "We benchmark drug-target interaction (DTI) prediction baselines. \
                   The protocol is described in arXiv:1908.05156, and code is at \
                   https://github.com/example/repo. Accuracy improved by 90\\% over \
                   the GPT-4 baseline."
                .to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        };
        let resp = OnnxTranslationEngine::translate(&engine, &req).expect("贪心解码必须成功");
        println!("SMOKE protect(en→zh): {:?}", resp.translation);
        // 占位符不得泄漏（未回填的 ⟦Pn⟧ 说明回填失败，等同内容丢失）
        assert!(
            !resp.translation.contains('⟦') && !resp.translation.contains('⟧'),
            "placeholder leaked: {:?}",
            resp.translation
        );
        // 受保护 token 必须存活于译文（丢失即模型吞位，本测试红）
        assert!(
            resp.translation.contains("DTI"),
            "DTI not protected through decode: {:?}",
            resp.translation
        );
        assert!(
            resp.translation.contains("arXiv:1908.05156"),
            "arXiv id not protected through decode: {:?}",
            resp.translation
        );
        assert!(
            resp.translation.contains("github.com"),
            "URL not protected through decode: {:?}",
            resp.translation
        );
        // LaTeX 转义回填后经 repair 清创（模型可能在数字与占位符间插空格，
        // 评测口径按数字/百分号分别命中）
        assert!(
            resp.translation.contains('%') && resp.translation.contains("90"),
            "LaTeX escape not backfilled: {:?}",
            resp.translation
        );
    }

    /// 并行失败路径（R-009 验收）：长输入中夹入超预算句（>max_source_tokens）
    /// → 整请求显性 Err，禁静默丢句 Partial。
    #[test]
    fn test_parallel_failure_is_explicit_not_partial() {
        let Some(dir) = m2m100_bundle() else {
            eprintln!("SKIP: models/m2m100_418M 缺失（离线环境）");
            return;
        };
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !std::path::Path::new(&dylib).is_file() {
            eprintln!("SKIP: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let config = ModelConfig {
            name: "m2m100-parallel-fail".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: dir,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Translate,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        };
        let engine = OnnxTranslationEngine::load(&config).expect("bundle 就绪时必须可加载");
        // 首句正常但整体超 600 字符阈值触发并行；次句 3600 连续字符（无句读）
        // 单句必超 512 source token 预算 → budget 失败必须显性暴露
        let mut text = String::from("We study graph neural networks for drug discovery. ");
        text.push_str(&"molecule ".repeat(600));
        let req = TranslateRequest {
            text,
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        };
        let err = OnnxTranslationEngine::translate(&engine, &req)
            .expect_err("超预算句必须显性失败，禁丢句 Partial");
        assert!(
            matches!(err, VecboostError::InferenceError(_)),
            "并行失败应为 InferenceError（禁静默丢句）: {err:?}"
        );
        let dbg = format!("{err:?}");
        assert!(
            dbg.contains("budget") || dbg.contains("token"),
            "错误应指向预算违规: {dbg}"
        );
    }

    #[test]
    fn test_simplify_chinese_output_normalizes_variants() {
        assert_eq!(
            simplify_chinese_output("人類製作複雜的指示".to_string()),
            "人类制作复杂的指示"
        );
        assert_eq!(simplify_chinese_output("下游任务".to_string()), "下游任务");
    }

    #[test]
    fn test_apply_output_fixes_ordered_literal_replacement() {
        let fixes = vec![
            OutputFix {
                from: "下流任务".to_string(),
                to: "下游任务".to_string(),
            },
            OutputFix {
                from: "精细的".to_string(),
                to: "微调的".to_string(),
            },
        ];
        assert_eq!(
            apply_output_fixes("NLP下流任务与精细的模型".to_string(), &fixes),
            "NLP下游任务与微调的模型"
        );
        // 空 from 跳过（防误配置全量替换）
        let bad = vec![OutputFix {
            from: String::new(),
            to: "X".to_string(),
        }];
        assert_eq!(apply_output_fixes("保持原样".to_string(), &bad), "保持原样");
        // 空表行为不变
        assert_eq!(apply_output_fixes("不变".to_string(), &[]), "不变");
    }

    #[test]
    fn test_cjk_punctuation_normalized_for_zh() {
        // 质量评审（golden 抽验）：Marian zh 输出系统性带半角标点（, ? !）
        assert_eq!(
            normalize_cjk_punctuation("她步行到市场,买了面包,对吗?".to_string(), "zh"),
            "她步行到市场，买了面包，对吗？"
        );
        assert_eq!(
            normalize_cjk_punctuation("注意:风险;小心!".to_string(), "zh"),
            "注意：风险；小心！"
        );
    }

    #[test]
    fn test_cjk_punctuation_digit_context_kept_halfwidth() {
        // 数字上下文不归一：千分位/时间冒号误转即语义错误
        assert_eq!(
            normalize_cjk_punctuation("共 3,000 台,12:30 出发,别迟到".to_string(), "zh"),
            "共 3,000 台，12:30 出发，别迟到"
        );
    }

    #[test]
    fn test_cjk_punctuation_applies_to_cjk_code_family() {
        // 639-1 与等价 639-3 码同口径（配置面两码空间均可到达）
        assert_eq!(
            normalize_cjk_punctuation("テスト,です?".to_string(), "jpn"),
            "テスト，です？"
        );
        assert_eq!(
            normalize_cjk_punctuation("테스트,입니까?".to_string(), "kor"),
            "테스트，입니까？"
        );
        assert_eq!(
            normalize_cjk_punctuation("广东话,测试!".to_string(), "yue"),
            "广东话，测试！"
        );
    }

    #[test]
    fn test_non_cjk_target_untouched() {
        assert_eq!(
            normalize_cjk_punctuation("Hello, world? Wait: yes; ok!".to_string(), "en"),
            "Hello, world? Wait: yes; ok!"
        );
        assert_eq!(
            normalize_cjk_punctuation("Hola, mundo? Sí: claro!".to_string(), "es"),
            "Hola, mundo? Sí: claro!"
        );
    }

    #[test]
    fn test_cjk_punctuation_period_and_empty_untouched() {
        // 句号不归一：模型已输出全角「。」，`.`→`。` 对小数/缩写有误伤风险
        assert_eq!(
            normalize_cjk_punctuation("版本 3.5 发布了。".to_string(), "zh"),
            "版本 3.5 发布了。"
        );
        assert_eq!(normalize_cjk_punctuation(String::new(), "zh"), "");
    }

    // ── 按句切分（离线纯函数；长难段质量调优的确定性前置） ──────────────

    #[test]
    fn test_split_multi_sentence() {
        assert_eq!(
            split_source_sentences("Rust is fast. It is safe. It is fun."),
            ["Rust is fast.", "It is safe.", "It is fun."]
        );
    }

    #[test]
    fn test_split_question_exclamation() {
        assert_eq!(
            split_source_sentences("Really? Yes! Fine."),
            ["Really?", "Yes!", "Fine."]
        );
    }

    #[test]
    fn test_split_abbreviation_guard() {
        // 缩写不切分：Dr. / U.S. / e.g.
        assert_eq!(
            split_source_sentences("Dr. Smith joined the U.S. delegation in 2024. He spoke first."),
            [
                "Dr. Smith joined the U.S. delegation in 2024.",
                "He spoke first."
            ]
        );
        assert_eq!(
            split_source_sentences("Agents, e.g. Claude, automate work. Risks remain."),
            ["Agents, e.g. Claude, automate work.", "Risks remain."]
        );
    }

    #[test]
    fn test_split_single_letter_initial_not_boundary() {
        assert_eq!(
            split_source_sentences("Ibram X. Kendi wrote it. Sales followed."),
            ["Ibram X. Kendi wrote it.", "Sales followed."]
        );
    }

    #[test]
    fn test_split_decimal_not_boundary() {
        // 小数点后紧跟数字不切分
        assert_eq!(
            split_source_sentences("Revenue rose 12.5 percent. Costs fell."),
            ["Revenue rose 12.5 percent.", "Costs fell."]
        );
    }

    #[test]
    fn test_split_closing_quote_boundary() {
        // 句终标点 + 闭引号归前句
        assert_eq!(
            split_source_sentences("He said, “go all out.” Then he left."),
            ["He said, “go all out.”", "Then he left."]
        );
    }

    #[test]
    fn test_split_no_terminal_punct_single_piece() {
        assert_eq!(
            split_source_sentences("AI should be no different"),
            ["AI should be no different"]
        );
        assert_eq!(split_source_sentences(""), Vec::<String>::new());
    }

    // ── 生成退化检测（离线纯函数；两遍策略的切换判据） ──────────────────

    #[test]
    fn test_degenerate_repeated_ngram_detected() {
        // 任一 4-gram 重复 ≥3 次 → 退化（重复循环）
        assert!(is_degenerate_generation(
            &[7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10],
            false
        ));
        assert!(is_degenerate_generation(
            &[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            false
        ));
    }

    #[test]
    fn test_degenerate_budget_hit_detected() {
        assert!(is_degenerate_generation(&[1, 2, 3, 4, 5, 6, 7, 8], true));
    }

    #[test]
    fn test_normal_generation_not_degenerate() {
        // 无重复 4-gram 的正常序列（含 2 次重复不足判据）
        assert!(!is_degenerate_generation(
            &[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 9, 9, 9, 9],
            false
        ));
        // 短序列（<8 token）不判定
        assert!(!is_degenerate_generation(&[1, 2, 3, 4, 1, 2, 3, 4], false));
        assert!(!is_degenerate_generation(&[], false));
    }

    #[test]
    fn test_check_token_budget_within_and_over() {
        // 预算内通过；超限 InvalidInput 且消息含 scope 与上限数值
        check_token_budget(511, 512, "text").unwrap();
        let err = check_token_budget(513, 512, "sentence").unwrap_err();
        assert!(matches!(err, VecboostError::InvalidInput(_)));
        let detail = err.error_detail();
        assert!(detail.contains("sentence"), "err={detail}");
        assert!(
            detail.contains("513") && detail.contains("512"),
            "err={detail}"
        );
    }

    // ── 多厂商模型家族（M2M100/NLLB 支持的离线面） ─────────────────────

    #[test]
    fn test_m2m_graph_spec_from_config() {
        let config: serde_json::Value = serde_json::json!({
            "model_type": "m2m_100",
            "decoder_layers": 12,
            "encoder_attention_heads": 16,
            "d_model": 1024,
        });
        let graph = M2mGraph::from_config(&config).unwrap();
        assert_eq!(graph.past_names.len(), 12 * 4);
        assert!(
            graph
                .past_names
                .contains(&"past_key_values.11.decoder.key".to_string()),
            "末层 decoder.key 必须在名表"
        );
        assert!(
            graph
                .past_names
                .contains(&"past_key_values.0.encoder.value".to_string())
        );
        assert_eq!(graph.head_shape, [1, 16, 0, 64]);
    }

    #[test]
    fn test_m2m_graph_spec_rejects_invalid_geometry() {
        let missing: serde_json::Value = serde_json::json!({ "model_type": "m2m_100" });
        assert!(matches!(
            M2mGraph::from_config(&missing),
            Err(VecboostError::ModelLoadError(_))
        ));
        let indivisible: serde_json::Value = serde_json::json!({
            "decoder_layers": 12, "encoder_attention_heads": 16, "d_model": 1000,
        });
        assert!(matches!(
            M2mGraph::from_config(&indivisible),
            Err(VecboostError::ModelLoadError(_))
        ));
    }

    #[test]
    fn test_nllb_lang_token_mapping() {
        assert_eq!(nllb_lang_token("en"), Some("eng_Latn"));
        assert_eq!(nllb_lang_token("zh"), Some("zho_Hans"));
        assert_eq!(nllb_lang_token("zho"), Some("zho_Hans"));
        assert_eq!(nllb_lang_token("cmn"), Some("zho_Hans"));
        assert_eq!(nllb_lang_token("yue"), Some("yue_Hant"));
        assert_eq!(nllb_lang_token("ja"), Some("jpn_Jpan"));
        assert_eq!(nllb_lang_token("xx"), None);
    }

    // ── EOS 抑制窗口（NLLB 提前截断的确定性修复） ───────────────────────

    #[test]
    fn test_argmax_index_excludes_eos_under_min_window() {
        // eos(2) 是全局最优：无排除时选中；窗口内排除后选次优(5)
        let row = [0.1f32, 0.2, 9.0, 0.3, 0.4, 5.0];
        assert_eq!(argmax_index(&row, None), Some(2));
        assert_eq!(argmax_index(&row, Some(2)), Some(5));
        // eos 非最优时排除无影响
        assert_eq!(argmax_index(&row, Some(5)), Some(2));
        // NaN 排序确定（total_cmp 全序：正 NaN > 数值，平局取末位——
        // 与既有贪心 argmax 语义一致，不 panic 即为本测试目标）
        let nan_row = [f32::NAN, 1.0, f32::NAN];
        assert_eq!(argmax_index(&nan_row, None), Some(2));
        // vocab==1 且排除唯一索引 → None（调用方回退）
        assert_eq!(argmax_index(&[1.0], Some(0)), None);
    }

    #[test]
    fn test_family_detect_rejects_unknown_model_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        // 最小合法 tokenizer（tokenizers 0.23 反序列化契约）
        std::fs::write(
            &path,
            r#"{"version":"1.0","added_tokens":[],"normalizer":null,
                "pre_tokenizer":null,"post_processor":null,"decoder":null,
                "model":{"type":"BPE","vocab":{},"merges":[]}}"#,
        )
        .unwrap();
        let tokenizer = load_tokenizer(&path).unwrap();
        let config: serde_json::Value = serde_json::json!({ "model_type": "some_future_arch" });
        let err = Family::detect(&config, &tokenizer).unwrap_err();
        assert!(matches!(err, VecboostError::ModelLoadError(_)));
        let detail = err.error_detail();
        assert!(detail.contains("some_future_arch"), "err={detail}");
    }

    #[test]
    fn test_load_tokenizer_shim_keeps_specials_and_filters_bad_merges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        // shim 契约：added_tokens 一律保留（`</s>` 等特殊 token 合法地不在
        // vocab——误删即丢 EOS 锚点）；仅 merges 按部件+结果越词表过滤
        std::fs::write(
            &path,
            r#"{"version":"1.0","added_tokens":[
                 {"id":0,"content":"</s>","single_word":false,"lstrip":false,
                  "rstrip":false,"normalized":false,"special":true},
                 {"id":128022,"content":"__en__","single_word":false,"lstrip":false,
                  "rstrip":false,"normalized":false,"special":true}],
                "normalizer":null,"pre_tokenizer":null,"post_processor":null,
                "decoder":null,
                "model":{"type":"BPE","vocab":{"h":0,"ello":1,"hello":2},
                         "merges":["h ello","hello x","ଙ h"]}}"#,
        )
        .unwrap();
        let tokenizer = load_tokenizer(&path).expect("shim 后必须可加载");
        // 特殊/语言 token 不在 vocab 也必须存活（手写 id 与 BPE vocab id 空间
        // 冲突会被解析器重排，故只断存在性，R-translation-decode-003）
        assert!(
            tokenizer.token_to_id("</s>").is_some(),
            "`</s>` 不得被 shim 误删"
        );
        assert!(
            tokenizer.token_to_id("__en__").is_some(),
            "`__en__` 不得被 shim 误删"
        );
    }

    #[test]
    fn test_load_tokenizer_marian_specials_survive() {
        // bundle 回归：marian 词表 dict 不含 `</s>`（仅 added_tokens 承载），
        // shim 后必须存活——generation_config 兜底与 decode 都依赖它
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let tokenizer = load_tokenizer(&dir.join("tokenizer.json")).unwrap();
        assert!(
            tokenizer.token_to_id("</s>").is_some(),
            "`</s>` 不得被 shim 误删"
        );
    }

    #[test]
    fn test_translate_m2m_real_bundle_end_to_end() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/m2m100_418M");
        if !dir.is_dir() {
            eprintln!("SKIP: models/m2m100_418M 缺失（离线环境）");
            return;
        }
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !std::path::Path::new(&dylib).is_file() {
            eprintln!("SKIP: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let config = ModelConfig {
            name: "m2m-smoke".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: dir,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Translate,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        };
        let engine = OnnxTranslationEngine::load(&config).expect("m2m bundle 必须可加载");
        let req = TranslateRequest {
            text: "Hello, my dog is cute".to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        };
        let resp = OnnxTranslationEngine::translate(&engine, &req).expect("m2m 贪心解码必须成功");
        assert!(!resp.translation.trim().is_empty(), "译文非空");
        assert!(
            !resp.translation.contains("__") && !resp.translation.contains("</s>"),
            "语言/特殊 token 不得泄漏: {:?}",
            resp.translation
        );
        println!("SMOKE m2m100 translate(en→zh): {:?}", resp.translation);
    }

    // ── CJK 字间空格清理（离线纯函数；Marian 解码伪影） ─────────────────

    #[test]
    fn test_collapse_spaces_between_cjk_chars() {
        // 评审 run1 实测伪影：标点旁/字间半角空格
        assert_eq!(
            collapse_cjk_spaces("取代 ” ， 后者由精英策划"),
            "取代”，后者由精英策划"
        );
        assert_eq!(
            collapse_cjk_spaces("《思想链 》 与 《 实践 》"),
            "《思想链》与《实践》"
        );
        assert_eq!(collapse_cjk_spaces("。 许多公司"), "。许多公司");
    }

    #[test]
    fn test_collapse_keeps_latin_inner_spaces() {
        // 词内拉丁空格保留；CJK↔拉丁边界空格按百度基准口径清除
        assert_eq!(
            collapse_cjk_spaces("Elon Musk 是该党的坚定支持者"),
            "Elon Musk是该党的坚定支持者"
        );
        assert_eq!(collapse_cjk_spaces("埃隆 · 马斯克"), "埃隆·马斯克");
    }

    #[test]
    fn test_collapse_em_dash_run() {
        assert_eq!(
            collapse_cjk_spaces("伟大的替代理论 — — 一个现象"),
            "伟大的替代理论——一个现象"
        );
    }

    #[test]
    fn test_collapse_ascii_only_untouched() {
        assert_eq!(
            collapse_cjk_spaces("Hello, my dog is cute"),
            "Hello, my dog is cute"
        );
    }

    // ── 真机多句链路（bundle 守卫；钉住按句切分后的 e2e 可译性） ─────────

    #[test]
    fn test_translate_multi_sentence_real_bundle() {
        let Some(dir) = opus_bundle() else {
            eprintln!("SKIP: models/opus-mt-en-zh 缺失（离线环境）");
            return;
        };
        let dylib = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
        if !std::path::Path::new(&dylib).is_file() {
            eprintln!("SKIP: onnxruntime dylib 缺失（{dylib}）");
            return;
        }
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            // SAFETY: 测试进程内单线程时点回填一次
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &dylib);
            }
        }
        let config = ModelConfig {
            name: "opus-smoke-multi".to_string(),
            engine_type: crate::config::model::EngineType::Onnx,
            model_path: dir,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: ModelTask::Translate,
            quantized: false,
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        };
        let engine = OnnxTranslationEngine::load(&config).expect("bundle 就绪时必须可加载");
        let req = TranslateRequest {
            text: "Rust is a systems programming language. It guarantees memory safety \
                   without a garbage collector."
                .to_string(),
            source_language: Some("en".to_string()),
            target_language: "zh".to_string(),
        };
        let resp = OnnxTranslationEngine::translate(&engine, &req).expect("分句贪心解码必须成功");
        assert!(!resp.translation.trim().is_empty(), "译文非空");
        assert!(
            !resp.translation.contains("</s>") && !resp.translation.contains("<pad>"),
            "特殊 token 不得泄漏: {:?}",
            resp.translation
        );
        // 两个句片段均成功解码（无句被空译文吞掉）
        assert!(engine.last_output_tokens() > 0);
        println!(
            "SMOKE multi-sentence translate(en→zh): {:?}",
            resp.translation
        );
    }
}
