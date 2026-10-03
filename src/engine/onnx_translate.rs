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

use crate::config::model::{DeviceType, ModelConfig, ModelTask, Precision, TranslationParams};
use crate::domain::{TranslateRequest, TranslateResponse};
use crate::error::VecboostError;
use ort::session::{Session, builder::GraphOptimizationLevel};
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

/// generation_config.json 解析结果（贪心解码锚点）。
///
/// `num_beams` 等 beam 参数不消费：本引擎恒贪心（argmax），与配置文件
/// 的 num_beams=4 无关（bundle 该字段面向 transformers 生成器，语义
/// 不适用于贪心实现）。
struct GenerationConfig {
    decoder_start_token_id: u32,
    eos_token_id: u32,
}

/// ONNX seq2seq 翻译引擎：bundle 自持（encoder/decoder/tokenizer/
/// generation_config）。
pub(crate) struct OnnxTranslationEngine {
    sessions: Arc<Mutex<TranslateSessions>>,
    tokenizer: Tokenizer,
    decoder_start_token_id: u32,
    eos_token_id: u32,
    /// per-checkpoint 翻译预算（加载期定值，热路径消费）
    params: TranslationParams,
    precision: Precision,
    /// 已探明的本地 bundle 目录（try_fallback_to_cpu 重建 CPU Session 用）
    bundle_dir: PathBuf,
    fallback_triggered: bool,
    fallback_lock: Arc<Mutex<()>>,
    /// 最近一次成功解码的生成 token 数（评审 converge T033：质量观测通道，
    /// wire 五键契约不扩——e2e 质量输出经引擎级查询读取）
    output_tokens_last: std::sync::atomic::AtomicU64,
}

/// bundle 内模型文件探测：候选按序首中，全 miss 报 `ModelLoadError`
/// 含尝试路径清单（decision.rs::probe_model_file 同款显性化）
fn probe_model_file(
    bundle_dir: &Path,
    candidates: &[&str],
    role: &str,
) -> Result<PathBuf, VecboostError> {
    let mut tried = Vec::with_capacity(candidates.len());
    for name in candidates {
        let candidate = bundle_dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(candidate.display().to_string());
    }
    Err(VecboostError::ModelLoadError(format!(
        "No {role} model found in translation bundle {}; tried: {}",
        bundle_dir.display(),
        tried.join(", ")
    )))
}

/// tokenizer 加载（T013 U1 换路 shim，见模块文档）：
/// 内存内移除 null `precompiled_charsmap` 的 Precompiled normalizer
/// （等价 identity 归一化），不改磁盘 bundle；其余字段原样反序列化。
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
    let shim_applied = value["normalizer"]["type"] == "Precompiled"
        && value["normalizer"]["precompiled_charsmap"].is_null();
    if shim_applied {
        if let Some(obj) = value.as_object_mut() {
            obj.remove("normalizer");
        }
        log::info!(
            "tokenizer {}: removed no-op Precompiled normalizer (null precompiled_charsmap; \
             tokenizers 0.23 rejects null charsmap, T013 first-day finding)",
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
        let start = value["decoder_start_token_id"].as_u64();
        let eos = value["eos_token_id"].as_u64();
        if start.is_none() || eos.is_none() {
            log::warn!(
                "translation bundle {}: generation_config.json missing decoder_start_token_id/eos_token_id; \
                 falling back to tokenizer special tokens for decode anchors",
                bundle_dir.display()
            );
            return Self::from_tokenizer_fallback(bundle_dir, tokenizer);
        }
        Ok(Self {
            decoder_start_token_id: start.unwrap() as u32,
            eos_token_id: eos.unwrap() as u32,
        })
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
        let start = tokenizer.token_to_id("pad").unwrap_or(eos);
        Ok(Self {
            decoder_start_token_id: start,
            eos_token_id: eos,
        })
    }
}

/// Session 构建（decision.rs::build_session 同模式：Level3 优化 + intra
/// threads + CUDA EP feature 分支）
fn build_session(model_file: &Path, device: &DeviceType) -> Result<Session, VecboostError> {
    // 评审 PERF-4：多 pair 时消费启动期分摊的全局预算（防 16×2 Session
    // 全核过订阅）；0 = 未设置回退全核（单 pair 场景）
    let num_threads = match SESSION_INTRA_THREADS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4),
        n => n,
    };
    log::info!(
        "Initializing ONNX Runtime session for translation: {model_file:?} (intra_threads={num_threads})"
    );
    let builder = Session::builder()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_intra_threads(num_threads)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;

    let mut session = if *device == DeviceType::Cuda {
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
                 using CPU execution provider for translation engine"
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

/// 探测命中的模型文件名 → 对外精度标签（quantized 变体报 Int8，
/// decision.rs::precision_for_model_file 同口径，不硬编码 Fp32 失真）
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

/// ort 调用错误 → InferenceError（wire 由上层 to_api_error 统一脱敏）
fn inference_err(step: &str) -> impl Fn(ort::Error) -> VecboostError + '_ {
    move |e: ort::Error| VecboostError::InferenceError(format!("translation {step} failed: {e}"))
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
            Some(p) => TranslationParams::new(p.max_source_tokens, p.max_new_tokens)?,
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
        let encoder = build_session(&encoder_file, &config.device)?;
        let decoder = build_session(&decoder_file, &config.device)?;
        let precision = precision_for_model_file(&decoder_file);

        log::info!(
            "Translation engine initialized: bundle={}, encoder={:?}, decoder={:?}, \
             precision={:?}, max_source_tokens={}, max_new_tokens={}, \
             decoder_start_token_id={}, eos_token_id={}",
            bundle_dir.display(),
            encoder_file,
            decoder_file,
            precision,
            params.max_source_tokens,
            params.max_new_tokens,
            generation.decoder_start_token_id,
            generation.eos_token_id
        );

        Ok(Self {
            sessions: Arc::new(Mutex::new(TranslateSessions { encoder, decoder })),
            tokenizer,
            decoder_start_token_id: generation.decoder_start_token_id,
            eos_token_id: generation.eos_token_id,
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

    /// 翻译本体：encode → 预算二次校验（显性拒绝，禁止静默截断——截断会
    /// 丢源语义致译文错译）→ encoder 单次前向 → 贪心循环（argmax、EOS
    /// 停止、`max_new_tokens` 硬顶；无缓存全量重算，依据见模块文档）→ decode。
    pub(crate) fn translate(
        &self,
        req: &TranslateRequest,
    ) -> Result<TranslateResponse, VecboostError> {
        let started = std::time::Instant::now();
        // add_special_tokens 必须为 true：Marian 源序列需结尾 </s>
        // （T015 实测，缺它解码退化为重复循环）
        let encoding = self
            .tokenizer
            .encode(req.text.as_str(), true)
            .map_err(|e| {
                VecboostError::InferenceError(format!("translation tokenization failed: {e}"))
            })?;
        let source_ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let source_len = source_ids.len();
        if source_len > self.params.max_source_tokens {
            // InvalidInput（to_api_error 映射 400）：design D2 承诺超限冒泡为
            // 400 语义；服务层预判为第一道，此处为引擎兜底（口径一致）
            return Err(VecboostError::InvalidInput(format!(
                "source text exceeds token budget: {source_len} tokens > {} (max_source_tokens); \
                 shorten the text or raise [translation.pairs] max_source_tokens",
                self.params.max_source_tokens
            )));
        }
        let attention_mask: Vec<i64> = vec![1; source_len];

        let generated: Vec<u32> = {
            // 评审 PERF-3：panic 后恢复中毒锁（into_inner 取回内层数据）——
            // 每步解码输入独立构造、Session 无跨请求可变状态，恢复安全；
            // 否则单次 panic 即该 pair 永久 "lock poisoned" 直至重启
            let mut sessions = self
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            // encoder 单次前向：last_hidden_state → decoder 侧
            // encoder_hidden_states（输出名 ≠ 输入名，改名搬运，U2 清单实测）
            let enc_outputs = sessions
                .encoder
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array((vec![1i64, source_len as i64], source_ids.clone()))
                        .map_err(inference_err("encoder input_ids"))?,
                    "attention_mask" => Tensor::from_array(
                        (vec![1i64, source_len as i64], attention_mask.clone())
                    )
                    .map_err(inference_err("encoder attention_mask"))?,
                ])
                .map_err(inference_err("encoder run"))?;
            let (enc_shape, enc_data) = enc_outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(inference_err("encoder last_hidden_state"))?;
            let enc_dims: Vec<i64> = enc_shape.to_vec();
            let enc_data = enc_data.to_vec();
            drop(enc_outputs);
            // 评审 PERF-2：循环不变输入循环外构造一次（review 建议的 Arc 共享
            // 在 ort rc.13 pub API 不可用——Value 无 pub Clone），改借用 view
            //（TensorRef::from_array_view）消除每步 enc_data（源 512 token≈1MB）
            // 整份克隆；view 借用 enc_data/attention_mask，存活至循环结束
            let enc_hidden_view =
                ort::value::TensorRef::from_array_view((enc_dims.as_slice(), enc_data.as_slice()))
                    .map_err(inference_err("encoder hidden view"))?;
            let mask_view = ort::value::TensorRef::from_array_view((
                [1i64, source_len as i64],
                attention_mask.as_slice(),
            ))
            .map_err(inference_err("attention mask view"))?;

            // 贪心循环（无缓存分支，依据见模块文档「缓存分支禁用」）：每步
            // 对 [start + 已生成] 全量重算、取末位 logits argmax。past 输入
            // 以 24 个 [1,8,0,64] 空张量占位（图输入必填，else 分支不消费值）。
            let mut generated: Vec<u32> = Vec::with_capacity(self.params.max_new_tokens);
            let mut decoder_ids: Vec<i64> = vec![self.decoder_start_token_id as i64];
            for _step in 0..self.params.max_new_tokens {
                let dec_len = decoder_ids.len() as i64;
                let mut inputs = ort::inputs![
                    "input_ids" => Tensor::from_array((vec![1i64, dec_len], decoder_ids.clone()))
                        .map_err(inference_err("decoder input_ids"))?,
                    "use_cache_branch" => Tensor::from_array((vec![1usize], vec![false]))
                        .map_err(inference_err("decoder use_cache_branch"))?,
                    "encoder_attention_mask" => mask_view.clone(),
                    "encoder_hidden_states" => enc_hidden_view.clone(),
                ];
                // 静态名表（评审 PERF-2）：消除每步 24 次 format! 堆分配；
                // 空占位张量零数据元素，每步重建成本可忽略
                for (name, _, _) in PAST_TENSOR_NAMES {
                    inputs.push((
                        Cow::Borrowed(name),
                        Tensor::from_array((vec![1i64, 8, 0i64, 64], Vec::<f32>::new()))
                            .map_err(inference_err("decoder past placeholder"))?
                            .into(),
                    ));
                }
                let outputs = sessions
                    .decoder
                    .run(inputs)
                    .map_err(inference_err("decoder step"))?;

                let (logits_shape, logits_data) = outputs["logits"]
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
                // 末位 token 的分布（logits [1, dec_len, V]，取最后一行）
                let vocab = logits_shape[2] as usize;
                let last_row =
                    &logits_data[(dec_len as usize - 1) * vocab..dec_len as usize * vocab];
                // 评审 PERF-3/ARC-6：total_cmp 全序比较（NaN 排序确定不 panic、
                // 不中毒锁；与 candle_decision 的容忍先例同语义强度）
                let (best, _score) = last_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("logits 非空");
                let best = best as u32;

                if best == self.eos_token_id {
                    break;
                }
                generated.push(best);
                decoder_ids.push(best as i64);
            }
            generated
        };

        self.output_tokens_last
            .store(generated.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let translation = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| {
                VecboostError::InferenceError(format!("translation detokenization failed: {e}"))
            })?
            .trim()
            .to_string();
        Ok(TranslateResponse {
            translation,
            source_language: req.source_language.clone(),
            target_language: req.target_language.clone(),
            detected_source_language: None,
            processing_time_ms: started.elapsed().as_millis() as u64,
        })
    }
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
        let encoder = build_session(&encoder_file, &DeviceType::Cpu)?;
        let decoder = build_session(&decoder_file, &DeviceType::Cpu)?;
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
}
