// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! whisper-large-v3-turbo ONNX 推理（encoder + decoder_model_merged 双图，30s 窗口贪心解码）。
//!
//! 与同目录 `asr.rs`（Qwen3-ASR 三图、KV [1,512,8,128]、无 cross-attention）结构不同：
//! whisper 为 4 解码层 / 20 头 / head_dim=64 / 含 encoder cross-attention KV，
//! 且 `decoder_model_merged.onnx` 一张图两用，靠 BOOL `use_cache_branch` 区分首步/后续步。
//!
//! ONNX 契约（本次会话联网实测，来源逐条列于模块尾注）：
//! - encoder: 1 输入 `input_features` FLOAT[batch,128,3000]，1 输出 `last_hidden_state`
//!   FLOAT[batch,1500,1280]（protobuf 解析官方图）
//! - decoder merged: 19 输入 / 17 输出；输入含 input_ids INT64、encoder_hidden_states
//!   FLOAT[batch,1500,1280]、past_key_values.{0..3}.{decoder,encoder}.{key,value} FLOAT、
//!   use_cache_branch BOOL[1]；输出含 logits FLOAT[batch,seq,51866] 与 present.* KV
//! - 首步无 past → use_cache_branch=false、past KV 第三维 0；后续步 → true，
//!   present.* 改名为 past_key_values.* 复用（transformers.js v3.7.6 实测源码）
//! - 30s 窗口：n_samples=480000、chunk_length=30、n_fft=400、hop_length=160、
//!   feature_size=128、nb_max_frames=3000（preprocessor_config.json 实测）
//!
//! 未做 ORT 实跑：I/O 名/形来自官方 protobuf 元数据，加载时逐项 I/O 内省 + 显性校验，
//! 量化变体 I/O 未逐一验证，命中后仍走同一守卫，不符即 `ModelLoadError`，不静默降级。
//!
//! 并发契约（与 `Qwen3Asr` 同口径，见 asr.rs:286-293）：ort rc.13 禁同 Session 并发
//! Run，`Session` 同步且 `&self` 可 Run，但并发 Run 未定义；实例内串行由上层 `Mutex`
//! 负责，本结构**不自带锁**（worker 多实例由调用方承担）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::error::VecboostError;

use super::asr::{
    AsrSessionOpts, greedy_argmax, is_degenerate_repetition, session_from, trim_degenerate_tail,
};
use super::features::{MEL_DIM, WhisperFbank};
use super::whisper_tokenizer::WhisperTokenizer;

pub(crate) type WhisperResult<T> = Result<T, VecboostError>;

fn whisper_load(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::ModelLoadError(crate::i18n::tr_with_args(
        "detail-whisper-asr-load-e-2e8b4",
        crate::i18n::tr_args(&[("e", &e.to_string())]),
    ))
}
fn whisper_infer(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::InferenceError(crate::i18n::tr_with_args(
        "detail-whisper-asr-infer-e-9d1f7",
        crate::i18n::tr_args(&[("e", &e.to_string())]),
    ))
}

/// 30s 窗口（16 kHz × 30s = 480000 样本；480000/160 = 3000 mel 帧）。
pub(crate) const WINDOW_SAMPLES: usize = 480_000;
pub(crate) const WINDOW_FRAMES: usize = 3_000;
/// 贪心解码步数上限（不含 SOT prompt 首步）。whisper 30s 段最多输出 448 个文本 token。
pub(crate) const MAX_NEW_TOKENS: usize = 448;
/// turbo 解码层数（config.json: decoder_layers=4）、每头维度（d_model/heads=1280/20）。
const NUM_DECODER_LAYERS: usize = 4;
const NUM_ATTENTION_HEADS: usize = 20;
const HEAD_DIM: usize = 64;
/// encoder 输出 / decoder cross-attention KV 序列长（3000 mel 帧 / 2 = 1500）。
const ENCODER_SEQ_LEN: usize = 1500;
const ENCODER_HIDDEN: usize = 1280;
/// logits 词表宽（config.json vocab_size=51866）。
const VOCAB_SIZE: usize = 51_866;

// ── 图名候选链（单一事实源，风格同 asr.rs:47-57）──────────────────────────────────
// 契约基准名优先（I/O 内省按 protobuf 实测形态校验），量化变体按 HF 同族后缀兜底。
// 勘察明确：量化变体 I/O 未逐一验证，命中后仍走 I/O 内省守卫，不符即显性拒绝。
// 历史缺陷（asr.rs:49-54）：候选数组与加载处二次拼字面量分叉 → 无损 bundle 无入口；
// 本模块探测与加载共用同一组常量，禁止在别处再写图名字面量。
pub(crate) const ENCODER_CANDIDATES: &[&str] = &[
    "encoder_model.onnx",
    "encoder_model_int8.onnx",
    "encoder_model_quantized.onnx",
    "encoder_model_uint8.onnx",
    "encoder_model_fp16.onnx",
    "encoder_model_q4.onnx",
    "encoder_model_q4f16.onnx",
    "encoder_model_bnb4.onnx",
];
pub(crate) const DECODER_CANDIDATES: &[&str] = &[
    "decoder_model_merged.onnx",
    "decoder_model_merged_int8.onnx",
    "decoder_model_merged_quantized.onnx",
    "decoder_model_merged_uint8.onnx",
    "decoder_model_merged_fp16.onnx",
    "decoder_model_merged_q4.onnx",
    "decoder_model_merged_q4f16.onnx",
    "decoder_model_merged_bnb4.onnx",
];

// 契约 I/O 名（protobuf 实测，逐字固定）。
const IN_INPUT_FEATURES: &str = "input_features";
const OUT_LAST_HIDDEN: &str = "last_hidden_state";
const IN_INPUT_IDS: &str = "input_ids";
const IN_ENCODER_HIDDEN: &str = "encoder_hidden_states";
const IN_USE_CACHE_BRANCH: &str = "use_cache_branch";
const OUT_LOGITS: &str = "logits";

#[derive(Debug, Clone)]
pub(crate) struct WhisperGraphPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
}

/// 按候选链探测双图路径；任一图全 miss → `ModelLoadError`（含全部尝试路径）。
/// 调用方必须先本函数、后 `Session::builder`（缺 ORT 动态库时 builder 会 non-unwinding panic）。
pub(crate) fn resolve_graph_paths(models_dir: &Path) -> WhisperResult<WhisperGraphPaths> {
    Ok(WhisperGraphPaths {
        encoder: crate::engine::session::probe_model_file(
            models_dir,
            ENCODER_CANDIDATES,
            "whisper encoder",
        )?,
        decoder: crate::engine::session::probe_model_file(
            models_dir,
            DECODER_CANDIDATES,
            "whisper decoder (merged)",
        )?,
    })
}

/// f32-only I/O 守卫（同 asr.rs:89-132 口径，本模块不导出该私有函数故同款复制）：
/// 只允许 FLOAT32 / INT64 / BOOL，拒绝 fp16/bf16/f64——KV 与 logits 均按 f32 提取，
/// 半精度图能加载但会在首次 Run 炸掉，这里在构建后显性拒绝（禁"加载成功、请求才失败"）。
fn element_type_supported(ty: &TensorElementType) -> bool {
    matches!(
        ty,
        TensorElementType::Float32 | TensorElementType::Int64 | TensorElementType::Bool
    )
}

fn ensure_f32_io(session: &Session, role: &str) -> WhisperResult<()> {
    for (i, input) in session.inputs().iter().enumerate() {
        if let ValueType::Tensor { ty, .. } = input.dtype()
            && !element_type_supported(ty)
        {
            return Err(whisper_load(format!(
                "{role} input[{i}] '{}' dtype {ty:?} unsupported: f32-only graph required \
                 (KV and logits extracted as f32)",
                input.name()
            )));
        }
    }
    for (i, output) in session.outputs().iter().enumerate() {
        if let ValueType::Tensor { ty, .. } = output.dtype()
            && !element_type_supported(ty)
        {
            return Err(whisper_load(format!(
                "{role} output[{i}] '{}' dtype {ty:?} unsupported: f32-only graph required",
                output.name()
            )));
        }
    }
    Ok(())
}

/// I/O 期望：(名字, dtype, 形状；-1 = 动态维通配)。
type IoSpec = (String, TensorElementType, Vec<i64>);

/// 形状匹配（纯函数，-1 期望维通配动态维）：rank 必须相等，非 -1 维必须逐维相等。
pub(crate) fn shape_matches(actual: &[i64], expected: &[i64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| *e == -1 || *a == *e)
}

/// 逐项 I/O 内省校验：数量、名字、dtype、形状；不符 → ModelLoadError 列实际与期望。
fn check_io(
    session: &Session,
    is_input: bool,
    expected: &[IoSpec],
    role: &str,
) -> WhisperResult<()> {
    let actual_io = if is_input {
        session.inputs()
    } else {
        session.outputs()
    };
    let kind = if is_input { "input" } else { "output" };
    if actual_io.len() != expected.len() {
        return Err(whisper_load(format!(
            "{role} {kind} count mismatch: actual {}, expected {} (actual names: {:?})",
            actual_io.len(),
            expected.len(),
            actual_io.iter().map(|o| o.name()).collect::<Vec<_>>()
        )));
    }
    for (name, want_ty, want_shape) in expected {
        let Some(outlet) = actual_io.iter().find(|o| o.name() == name.as_str()) else {
            return Err(whisper_load(format!(
                "{role} {kind} '{name}' missing: actual names {:?}",
                actual_io.iter().map(|o| o.name()).collect::<Vec<_>>()
            )));
        };
        if let ValueType::Tensor { ty, shape, .. } = outlet.dtype() {
            let actual_shape: Vec<i64> = shape.to_vec();
            if ty != want_ty {
                return Err(whisper_load(format!(
                    "{role} {kind} '{name}' dtype mismatch: actual {ty:?}, expected {want_ty:?}"
                )));
            }
            if !shape_matches(&actual_shape, want_shape) {
                return Err(whisper_load(format!(
                    "{role} {kind} '{name}' shape mismatch: actual {actual_shape:?}, \
                     expected {want_shape:?}"
                )));
            }
        } else {
            return Err(whisper_load(format!(
                "{role} {kind} '{name}' is not a tensor"
            )));
        }
    }
    Ok(())
}

/// encoder/decoder 的契约 I/O 期望（按 protobuf 实测，逐项构造）。
fn expected_io() -> (Vec<IoSpec>, Vec<IoSpec>, Vec<IoSpec>, Vec<IoSpec>) {
    let enc_in = vec![(
        IN_INPUT_FEATURES.to_string(),
        TensorElementType::Float32,
        vec![-1, MEL_DIM as i64, WINDOW_FRAMES as i64],
    )];
    let enc_out = vec![(
        OUT_LAST_HIDDEN.to_string(),
        TensorElementType::Float32,
        vec![-1, ENCODER_SEQ_LEN as i64, ENCODER_HIDDEN as i64],
    )];

    let mut dec_in: Vec<IoSpec> = vec![
        (
            IN_INPUT_IDS.to_string(),
            TensorElementType::Int64,
            vec![-1, -1],
        ),
        (
            IN_ENCODER_HIDDEN.to_string(),
            TensorElementType::Float32,
            vec![-1, ENCODER_SEQ_LEN as i64, ENCODER_HIDDEN as i64],
        ),
    ];
    let mut dec_out: Vec<IoSpec> = vec![(
        OUT_LOGITS.to_string(),
        TensorElementType::Float32,
        vec![-1, -1, VOCAB_SIZE as i64],
    )];
    for layer in 0..NUM_DECODER_LAYERS {
        for side in ["decoder", "encoder"] {
            for component in ["key", "value"] {
                // past 第三维必须通配：decoder 从首步的 0 增长到最多 448；encoder
                // 首步 past_enc 为空（同样传 0）、后续固定 1500。钉死 1500 会让
                // 声明动态维的图在加载期 I/O 守卫处被误拒绝；通配才能覆盖两种图。
                dec_in.push((
                    format!("past_key_values.{layer}.{side}.{component}"),
                    TensorElementType::Float32,
                    vec![-1, NUM_ATTENTION_HEADS as i64, -1, HEAD_DIM as i64],
                ));
                dec_out.push((
                    format!("present.{layer}.{side}.{component}"),
                    TensorElementType::Float32,
                    vec![-1, NUM_ATTENTION_HEADS as i64, -1, HEAD_DIM as i64],
                ));
            }
        }
    }
    dec_in.push((
        IN_USE_CACHE_BRANCH.to_string(),
        TensorElementType::Bool,
        vec![1],
    ));
    (enc_in, enc_out, dec_in, dec_out)
}

/// KV 张量元素数 → 序列长（[1,20,seq,64]），纯函数供离线单测。
pub(crate) fn kv_seq_len(data_len: usize) -> usize {
    data_len / (NUM_ATTENTION_HEADS * HEAD_DIM)
}

/// 切窗半开区间：len=0 → 空；每窗 480000；尾窗不足亦成一窗（调用方零填充）。
pub(crate) fn window_ranges(len: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(len.div_ceil(WINDOW_SAMPLES));
    let mut start = 0;
    while start < len {
        let end = (start + WINDOW_SAMPLES).min(len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// 尾窗零填充到 480000 样本（纯函数，便于核对填充数学）。
pub(crate) fn padded_window(samples: &[f32], start: usize, end: usize) -> Vec<f32> {
    debug_assert!(start <= end && end <= samples.len());
    let mut window = vec![0.0f32; WINDOW_SAMPLES];
    let valid = end - start;
    window[..valid].copy_from_slice(&samples[start..end]);
    window
}

/// mel [frames,128] 行主序 → encoder 输入 [1,128,3000] 通道优先（纯函数）。
pub(crate) fn transpose_mel(mel: &[f32], frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; MEL_DIM * frames];
    for f in 0..frames {
        for m in 0..MEL_DIM {
            out[m * frames + f] = mel[f * MEL_DIM + m];
        }
    }
    out
}

/// whisper 推理会话：encoder + merged decoder 双图，KV 缓存驻留本结构。
pub struct WhisperAsr {
    encoder: Session,
    decoder: Session,
    fbank: WhisperFbank,
    /// decoder self-attention KV，每步由 present.* 更新（首步前为空 → 第三维 0）。
    past_dec: Vec<(Vec<f32>, Vec<f32>)>,
    /// encoder cross-attention KV，首步由 present.encoder.* 取得，后续步沿用。
    past_enc: Vec<(Vec<f32>, Vec<f32>)>,
}

impl WhisperAsr {
    /// 每 worker 独立加载一套 session（并发契约见模块头）。先探测后构建。
    /// 返回实例与探测到的双图路径（上层可用 decoder 实际路径做精度判定）。
    pub fn new(
        models_dir: &Path,
        opts: &AsrSessionOpts,
    ) -> WhisperResult<(Self, WhisperGraphPaths)> {
        let graphs = resolve_graph_paths(models_dir)?;
        let encoder = session_from(&graphs.encoder, opts)?;
        let decoder = session_from(&graphs.decoder, opts)?;
        ensure_f32_io(&encoder, "whisper encoder")?;
        ensure_f32_io(&decoder, "whisper decoder")?;
        let (enc_in, enc_out, dec_in, dec_out) = expected_io();
        check_io(&encoder, true, &enc_in, "whisper encoder")?;
        check_io(&encoder, false, &enc_out, "whisper encoder")?;
        check_io(&decoder, true, &dec_in, "whisper decoder")?;
        check_io(&decoder, false, &dec_out, "whisper decoder")?;

        Ok((
            Self {
                encoder,
                decoder,
                fbank: WhisperFbank::new(),
                past_dec: Vec::with_capacity(NUM_DECODER_LAYERS),
                past_enc: Vec::with_capacity(NUM_DECODER_LAYERS),
            },
            graphs,
        ))
    }

    /// mel 特征 [1,128,3000] → encoder 输出 [1,1500,1280]（owned，供每步 decoder 复用）。
    fn encode(&mut self, feats: &[f32]) -> WhisperResult<Vec<f32>> {
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            IN_INPUT_FEATURES,
            SessionInputValue::from(
                Tensor::from_array((
                    vec![1i64, MEL_DIM as i64, WINDOW_FRAMES as i64],
                    feats.to_vec(),
                ))
                .map_err(whisper_infer)?,
            ),
        );
        let outputs = self.encoder.run(inputs).map_err(whisper_infer)?;
        let (shape, data) = outputs[OUT_LAST_HIDDEN]
            .try_extract_tensor::<f32>()
            .map_err(whisper_infer)?;
        let dims: Vec<i64> = shape.to_vec();
        if dims.len() != 3 || dims[1] != ENCODER_SEQ_LEN as i64 || dims[2] != ENCODER_HIDDEN as i64
        {
            return Err(whisper_load(format!(
                "unexpected encoder output shape {dims:?}, expected [1,{ENCODER_SEQ_LEN},{ENCODER_HIDDEN}]"
            )));
        }
        Ok(data.to_vec())
    }

    /// merged decoder 一步：input_ids + encoder_hidden_states + past KV + use_cache_branch
    /// → logits 末行 argmax，并把 present.* 写回 KV 缓存（首步 false / 后续 true）。
    fn forward(
        &mut self,
        ids: &[i64],
        enc_hidden: &[f32],
        use_cache_branch: bool,
    ) -> WhisperResult<i64> {
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            IN_INPUT_IDS,
            SessionInputValue::from(
                Tensor::from_array((vec![1i64, ids.len() as i64], ids.to_vec()))
                    .map_err(whisper_infer)?,
            ),
        );
        inputs.insert(
            IN_ENCODER_HIDDEN,
            SessionInputValue::from(
                Tensor::from_array((
                    vec![1i64, ENCODER_SEQ_LEN as i64, ENCODER_HIDDEN as i64],
                    enc_hidden.to_vec(),
                ))
                .map_err(whisper_infer)?,
            ),
        );
        inputs.insert(
            IN_USE_CACHE_BRANCH,
            SessionInputValue::from(
                Tensor::from_array((vec![1i64], vec![use_cache_branch])).map_err(whisper_infer)?,
            ),
        );

        // past KV：首步为空（第三维 0），后续步为上一步 present 写回的 owned 数据。
        // 输入名 String 存活至 run 结束（禁止 Box::leak 永久泄漏内存）。
        let mut past_names: Vec<String> = Vec::with_capacity(NUM_DECODER_LAYERS * 4);
        for layer in 0..NUM_DECODER_LAYERS {
            for side in ["decoder", "encoder"] {
                for component in ["key", "value"] {
                    past_names.push(format!("past_key_values.{layer}.{side}.{component}"));
                }
            }
        }
        let mut name_idx = 0usize;
        for layer in 0..NUM_DECODER_LAYERS {
            // 顺序须与 past_names 生成序一致：layer → (decoder,encoder) → (key,value)。
            for cache in [&self.past_dec, &self.past_enc] {
                let (k, v): (&[f32], &[f32]) = if layer < cache.len() {
                    (&cache[layer].0[..], &cache[layer].1[..])
                } else {
                    (&[][..], &[][..])
                };
                for data in [k, v] {
                    let name = past_names[name_idx].as_str();
                    name_idx += 1;
                    let seq_len = kv_seq_len(data.len());
                    let tensor = Tensor::from_array((
                        vec![
                            1i64,
                            NUM_ATTENTION_HEADS as i64,
                            seq_len as i64,
                            HEAD_DIM as i64,
                        ],
                        data.to_vec(),
                    ))
                    .map_err(whisper_infer)?;
                    inputs.insert(name, SessionInputValue::from(tensor));
                }
            }
        }

        let outputs = self.decoder.run(inputs).map_err(whisper_infer)?;

        // logits [1, seq, vocab]：取末行 argmax（复用 asr.rs 贪心采样，跳过非有限值）。
        let (lshape, ldata) = outputs[OUT_LOGITS]
            .try_extract_tensor::<f32>()
            .map_err(whisper_infer)?;
        let ldims: Vec<i64> = lshape.to_vec();
        if ldims.len() != 3 || ldims[1] <= 0 || ldims[2] != VOCAB_SIZE as i64 {
            return Err(whisper_load(format!(
                "unexpected logits shape {ldims:?}, expected [1,seq,{VOCAB_SIZE}]"
            )));
        }
        let seq = ldims[1] as usize;
        let vocab = ldims[2] as usize;
        if ldata.len() != seq * vocab {
            return Err(whisper_load(format!(
                "logits len {} != {seq}*{vocab}",
                ldata.len()
            )));
        }
        let next = greedy_argmax(&ldata[(seq - 1) * vocab..], None);

        // present.* → owned 写回 KV。decoder self-attention KV 每步增长 1，必须逐步
        // 更新；encoder cross-attention KV 由首步产生、之后恒定（后续步仅透传），
        // 无须每步读回重写——4 层×2×1500×1280×4B≈61 MB/步，448 步/窗将浪费数十 GB
        // 内存拷贝。故只在首步（use_cache_branch=false）写回 self.past_enc。
        let mut dec_new = Vec::with_capacity(NUM_DECODER_LAYERS);
        let mut enc_new = Vec::with_capacity(NUM_DECODER_LAYERS);
        for layer in 0..NUM_DECODER_LAYERS {
            let read = |name: String| -> WhisperResult<Vec<f32>> {
                outputs[name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map(|(_, d)| d.to_vec())
                    .map_err(whisper_infer)
            };
            let dk = read(format!("present.{layer}.decoder.key"))?;
            let dv = read(format!("present.{layer}.decoder.value"))?;
            dec_new.push((dk, dv));
            if !use_cache_branch {
                let ek = read(format!("present.{layer}.encoder.key"))?;
                let ev = read(format!("present.{layer}.encoder.value"))?;
                enc_new.push((ek, ev));
            }
        }
        self.past_dec = dec_new;
        if !use_cache_branch {
            self.past_enc = enc_new;
        }
        Ok(next)
    }

    /// 一段样本 → 贪心解码文本。>30s 按 30s 切窗，尾窗零填充，逐窗独立转录后拼接。
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        tok: &WhisperTokenizer,
        language: Option<&str>,
    ) -> WhisperResult<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let prompt = tok.sot_prompt(language)?;
        let ranges = window_ranges(samples.len());
        let mut texts = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            let window = padded_window(samples, start, end);
            let mel = self.fbank.compute(&window);
            if mel.len() < WINDOW_FRAMES * MEL_DIM {
                return Err(whisper_load(format!(
                    "mel len {} < {} (window frames)",
                    mel.len(),
                    WINDOW_FRAMES * MEL_DIM
                )));
            }
            let feats = transpose_mel(&mel, WINDOW_FRAMES);
            let enc_hidden = self.encode(&feats)?;

            // 每窗独立解码：首步必须传空 past（第三维 0），否则上一段的 KV 会被
            // use_cache_branch=false 的图当作输入，导致跨窗污染（契约 D）。
            self.past_dec.clear();
            self.past_enc.clear();
            let mut next = self.forward(&prompt, &enc_hidden, false)?;
            if next == super::whisper_tokenizer::EOT_ID {
                continue;
            }
            let mut generated = vec![next];
            for _ in 1..MAX_NEW_TOKENS {
                // 每段独立的 KV 必须从空开始；首步已写回 present，此处需清空再重跑首步？
                // 否：首步 present 是**后续步**的 past，本步用 use_cache=true 复用它。
                next = self.forward(&[next], &enc_hidden, true)?;
                if next == super::whisper_tokenizer::EOT_ID {
                    break;
                }
                generated.push(next);
                if is_degenerate_repetition(&generated) {
                    trim_degenerate_tail(&mut generated);
                    break;
                }
            }
            let text = tok.decode(&generated)?;
            if !text.trim().is_empty() {
                texts.push(text);
            }
        }
        Ok(texts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_ranges_split_exactly_at_thirty_seconds() {
        assert!(window_ranges(0).is_empty(), "空样本不应产生窗");
        assert_eq!(
            window_ranges(WINDOW_SAMPLES),
            vec![(0, WINDOW_SAMPLES)],
            "恰好 30s 只切一窗"
        );
        assert_eq!(
            window_ranges(WINDOW_SAMPLES + 1),
            vec![(0, WINDOW_SAMPLES), (WINDOW_SAMPLES, WINDOW_SAMPLES + 1)],
            "多 1 样本必须另起一窗（尾窗零填充）"
        );
        assert_eq!(
            window_ranges(2 * WINDOW_SAMPLES),
            vec![(0, WINDOW_SAMPLES), (WINDOW_SAMPLES, 2 * WINDOW_SAMPLES)],
            "整 60s 切两窗，无空尾窗"
        );
    }

    #[test]
    fn padded_window_zero_fills_tail() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let w = padded_window(&samples, 0, 10);
        assert_eq!(w.len(), WINDOW_SAMPLES, "填充后长度必须为 480000");
        assert_eq!(&w[..10], &samples[..], "前 10 样本必须原样保留");
        assert!(
            w[10..].iter().all(|&v| v == 0.0),
            "其余 479990 样本必须为零"
        );
        let full = padded_window(&samples, 0, samples.len());
        assert_eq!(full.len(), WINDOW_SAMPLES);
    }

    #[test]
    fn transpose_mel_maps_frames_to_channels() {
        // 2 帧 × 3 mel 模拟：用 MEL_DIM 实际值验证 [frames,MEL_DIM] → [1,MEL_DIM,frames]。
        let frames = 2;
        let mut mel = vec![0.0f32; frames * MEL_DIM];
        for f in 0..frames {
            for m in 0..MEL_DIM {
                mel[f * MEL_DIM + m] = (f * MEL_DIM + m) as f32;
            }
        }
        let out = transpose_mel(&mel, frames);
        assert_eq!(out.len(), MEL_DIM * frames);
        for f in 0..frames {
            for m in 0..MEL_DIM {
                assert_eq!(
                    out[m * frames + f],
                    mel[f * MEL_DIM + m],
                    "通道优先布局错位 (f={f},m={m})"
                );
            }
        }
    }

    #[test]
    fn shape_matches_treats_minus_one_as_wildcard() {
        assert!(
            shape_matches(&[1, 128, 3000], &[-1, 128, 3000]),
            "batch 维动态应通配"
        );
        assert!(!shape_matches(&[1, 127, 3000], &[-1, 128, 3000]));
        assert!(!shape_matches(&[1, 128], &[-1, 128, 3000]), "rank 必须相等");
        assert!(
            shape_matches(&[1, 20, 0, 64], &[-1, 20, -1, 64]),
            "首步空 past 的 seq=0 必须被通配接受"
        );
    }

    #[test]
    fn kv_seq_len_derives_from_kv_element_count() {
        // [1,20,seq,64] → 每序列长 1280 元素。
        assert_eq!(kv_seq_len(0), 0, "首步空 past 序列长应为 0");
        assert_eq!(kv_seq_len(NUM_ATTENTION_HEADS * HEAD_DIM), 1);
        assert_eq!(kv_seq_len(5 * NUM_ATTENTION_HEADS * HEAD_DIM), 5);
    }

    #[test]
    fn expected_io_counts_match_contract() {
        let (enc_in, enc_out, dec_in, dec_out) = expected_io();
        assert_eq!(enc_in.len(), 1, "encoder 1 输入");
        assert_eq!(enc_out.len(), 1, "encoder 1 输出");
        assert_eq!(dec_in.len(), 19, "merged decoder 19 输入");
        assert_eq!(dec_out.len(), 17, "merged decoder 17 输出");
        assert!(
            dec_in.iter().any(|(n, _, _)| n == IN_USE_CACHE_BRANCH),
            "必须含 use_cache_branch"
        );
        assert!(
            dec_in
                .iter()
                .any(|(n, _, _)| n == "past_key_values.3.encoder.value"),
            "必须含第 3 层 encoder value KV"
        );
        assert!(
            dec_out.iter().any(|(n, _, _)| n == "present.0.decoder.key"),
            "必须含 present decoder KV"
        );
    }

    #[test]
    fn decode_guard_truncates_whisper_repetition() {
        // 复用 asr.rs 守卫：whisper 生成 64 窗口内 ≤3 distinct 即判复读。
        let mut ids = vec![100i64];
        ids.extend([11i64, 22, 33].iter().copied().cycle().take(64 * 2));
        assert!(is_degenerate_repetition(&ids), "3-token 循环必须判定塌缩");
        trim_degenerate_tail(&mut ids);
        assert_eq!(ids.len(), 1 + 64, "裁尾只保留循环之前的 token");
        assert_eq!(ids[0], 100);
    }
}

// 契约来源（本次会话联网实测，逐条列出）：
// A. 文件清单：https://huggingface.co/onnx-community/whisper-large-v3-turbo/tree/main/onnx
//    与 config.json（use_external_data_format 指明 encoder 走外置数据）
// B. encoder 图 I/O：下载 encoder_model.onnx 用 onnx 1.17 解析（opset 14）
// C. decoder merged I/O：HTTP Range 取末尾 2 MB，官方 protobuf 解析
// D. 首步/后续步：@huggingface/transformers@3.7.6 src/models.js:607-629、:2002-2017，
//    dist/transformers.node.mjs:4619-4644、:1943-1972、:10720-10770
// E. tokenizer：HF API sibling 列表 + tokenizer_config.json / generation_config.json
// F. 30s 窗口：preprocessor_config.json（chunk_length=30、n_samples=480000、
//    hop_length=160、feature_size=128、nb_max_frames=3000）
