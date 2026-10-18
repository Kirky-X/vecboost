// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-ASR 三图推理与贪心解码（conv_frontend → encoder → decoder KV-cache）。
//!
//! 全部推理逻辑为独立实现：帧数→音频 token 数换算、音频尾部静音裁剪、
//! 复读守卫（64 窗口内 distinct ≤3 判定塌缩、尾部裁剪）、贪心采样
//! （跳过非有限值、首 token EOS 重采样一次）、prompt 模板构造、
//! KV delta 原地写回（按 cache_position 连续写入）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::ArrayView4;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::config::model::DeviceType;
use crate::engine::session::build_session;
use crate::error::VecboostError;

use super::tokenizer::{AsrTokenizer, EOS_ID};

pub(crate) type AsrResult<T> = Result<T, VecboostError>;

/// 加载/IO 与运行期解码错误映射助手（R-transcribe-engine-004：无 Box<dyn Error>）。
fn asr_load(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::ModelLoadError(format!("transcribe asr load: {e}"))
}
fn asr_infer(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::InferenceError(format!("transcribe asr: {e}"))
}

pub(crate) const MAX_NEW_TOKENS: usize = 128; // 解码步数上限
pub(crate) const MAX_TOTAL_LEN: i64 = 512;
pub(crate) const KV_HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;
pub(crate) const CONV_CHUNK_SIZE: i32 = 100; // kQwen3ChunkSize
pub(crate) const MIN_FRAMES: usize = 2; // 帧数 <2 直接空结果

// ── 三图文件名候选链（**单一事实源**）────────────────────────────────────────────
// 会话构建与上层精度探测共用本组常量，禁止在别处再写图名字面量。
// 历史缺陷（2026-10-18 修复）：候选数组各只有一个却写死 `.int8.onnx`，且加载处
// 又按同一字面量二次拼路径 → 未量化（fp32）/ 半精度 bundle **无入口**。
//
// 优先级：int8 在前（与现有资产 `assets/models/` 和该导出通路的约定形态一致），
// 纯名兜底（无损导出按正名接入）。两图同时存在时取 int8——与修复前行为逐字一致；
// 想用无损图就不要把 int8 文件放进 bundle。
pub(crate) const CONV_CANDIDATES: &[&str] = &["qwen3-conv_frontend.onnx"];
pub(crate) const ENCODER_CANDIDATES: &[&str] = &["qwen3-encoder.int8.onnx", "qwen3-encoder.onnx"];
pub(crate) const DECODER_CANDIDATES: &[&str] = &["qwen3-decoder.int8.onnx", "qwen3-decoder.onnx"];

/// 探测命中的三图实际路径（探测与加载共用，消除“探测用数组、加载用字面量”的分叉）。
#[derive(Debug, Clone)]
pub(crate) struct AsrGraphPaths {
    pub conv: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
}

/// 按候选链探测三图路径；任一图全 miss → `ModelLoadError`（消息含全部尝试路径）。
/// 调用方必须先本函数、后 `Session::builder`（ort/load-dynamic 缺库时 builder panic）。
pub(crate) fn resolve_graph_paths(models_dir: &Path) -> AsrResult<AsrGraphPaths> {
    Ok(AsrGraphPaths {
        conv: crate::engine::session::probe_model_file(
            models_dir,
            CONV_CANDIDATES,
            "conv_frontend",
        )?,
        encoder: crate::engine::session::probe_model_file(
            models_dir,
            ENCODER_CANDIDATES,
            "encoder",
        )?,
        decoder: crate::engine::session::probe_model_file(
            models_dir,
            DECODER_CANDIDATES,
            "decoder",
        )?,
    })
}

/// 允许的 I/O 元素类型：f32（计算与 KV cache）、i64（token id / cache_position / 索引）、
/// bool（encoder attention mask）。其余浮点宽度（f16/bf16/f64）一律拒绝。
fn element_type_supported(ty: &TensorElementType) -> bool {
    matches!(
        ty,
        TensorElementType::Float32 | TensorElementType::Int64 | TensorElementType::Bool
    )
}

/// 图 I/O dtype 守卫：当前实现为 **f32-only**——KV cache 常驻 `Vec<f32>`、张量按 f32
/// 构造与提取（`try_extract_tensor::<f32>`）。fp16/bf16 图能加载但会在首次 Run 以
/// 类型不匹配炸掉，这里在构建后立即显性拒绝并给出可执行提示，避开“加载成功、
/// 请求时才失败”的假成功。
///
/// 为何不在边界做 f32↔f16 转换：1.7B 单步 KV 已是 29.4M 元素（f32 117 MiB /
/// f16 59 MiB，28 层×8 头×128 维×512），每步转换会被内存搬运支配；要支持 fp16
/// 必须把 KV 存储与张量 I/O 一起 dtype 泛型化
/// （见 `docs/benchmarks/transcribe_engine_baseline.md` 的 fp16 书签）。
fn ensure_f32_io(session: &Session, role: &str) -> AsrResult<()> {
    for (i, input) in session.inputs().iter().enumerate() {
        if let ValueType::Tensor { ty, .. } = input.dtype()
            && !element_type_supported(ty)
        {
            return Err(asr_load(format!(
                "{role} input[{i}] '{}' dtype {ty:?} is not supported: this build runs f32 \
                 graphs only (KV cache and tensor I/O are f32); fp16/bf16 graphs require \
                 dtype-generic KV storage",
                input.name()
            )));
        }
    }
    for (i, output) in session.outputs().iter().enumerate() {
        if let ValueType::Tensor { ty, .. } = output.dtype()
            && !element_type_supported(ty)
        {
            return Err(asr_load(format!(
                "{role} output[{i}] '{}' dtype {ty:?} is not supported: this build runs f32 \
                 graphs only (logits/KV are extracted as f32)",
                output.name()
            )));
        }
    }
    Ok(())
}

/// 复读守卫常量
const LOOP_WINDOW: usize = 64;
const LOOP_MAX_DISTINCT: usize = 3;

/// session 线程配置：intra 线程数与自旋开关（→ extra_config allow_spinning），并透
/// 传 CUDA EP（→ device=Cuda）。env_pool（POC 跑分开关）不并入库语义——段级并行
/// 由调用方持多引擎实例承担（design D1/Consequences）。
#[derive(Debug, Clone, Copy)]
pub struct AsrSessionOpts {
    pub intra_threads: usize,
    pub no_spin: bool,
    pub cuda: bool,
}

/// 加载一个 ort session——builder 收编至 [`build_session`]（R-engine-inference-001，
/// 主仓内非 session.rs 不再出现 `Session::builder()`）。
pub(crate) fn session_from(path: &Path, opts: &AsrSessionOpts) -> AsrResult<Session> {
    let extra: &[(&str, &str)] = if opts.no_spin {
        &[("session.intra_op.allow_spinning", "0")]
    } else {
        &[]
    };
    let device = if opts.cuda {
        DeviceType::Cuda
    } else {
        DeviceType::Cpu
    };
    build_session(
        path,
        &crate::engine::session::SessionOpts {
            intra_threads: opts.intra_threads,
            device,
            extra_config: extra,
        },
    )
}

/// FeatToAudioTokensLen
pub fn feat_to_audio_tokens_len(feat_len: i32, chunk_size: i32) -> i32 {
    if feat_len <= 0 || chunk_size <= 0 {
        return 0;
    }
    let conv_out_len_3x_stride2 = |n: i32| -> i32 {
        let mut x = (n + 1) / 2;
        x = (x + 1) / 2;
        (x + 1) / 2
    };
    let aftercnn = |x: i32| -> i32 {
        if x <= 0 {
            return 0;
        }
        let mut y = (x - 1) / 2 + 1;
        y = (y - 1) / 2 + 1;
        (y - 1) / 2 + 1
    };
    let full = feat_len / chunk_size;
    let rem = feat_len % chunk_size;
    let tn = conv_out_len_3x_stride2(chunk_size);
    (full * tn + if rem > 0 { aftercnn(rem) } else { 0 }).max(0)
}

/// 复读守卫：最近 LOOP_WINDOW 个 token 内 distinct ≤3
pub(crate) fn is_degenerate_repetition(ids: &[i64]) -> bool {
    if ids.len() < LOOP_WINDOW {
        return false;
    }
    let mut distinct = [0i64; LOOP_MAX_DISTINCT];
    let mut num_distinct = 0usize;
    for &id in &ids[ids.len() - LOOP_WINDOW..] {
        if distinct[..num_distinct].contains(&id) {
            continue;
        }
        if num_distinct == LOOP_MAX_DISTINCT {
            return false;
        }
        distinct[num_distinct] = id;
        num_distinct += 1;
    }
    true
}

/// 裁掉塌缩循环尾部：只删属于末窗口 token 集的尾 token
pub(crate) fn trim_degenerate_tail(ids: &mut Vec<i64>) {
    let mut distinct = [0i64; LOOP_MAX_DISTINCT];
    let mut num_distinct = 0usize;
    for &id in &ids[ids.len() - LOOP_WINDOW..] {
        if !distinct[..num_distinct].contains(&id) && num_distinct < LOOP_MAX_DISTINCT {
            distinct[num_distinct] = id;
            num_distinct += 1;
        }
    }
    let window_start = ids.len() - LOOP_WINDOW;
    let mut keep = ids.len();
    while keep > window_start && distinct[..num_distinct].contains(&ids[keep - 1]) {
        keep -= 1;
    }
    ids.truncate(keep);
}

/// 贪心 argmax，temperature=1e-6 语义：跳过非有限值
pub(crate) fn greedy_argmax(row: &[f32], avoid: Option<i64>) -> i64 {
    let mut best = 0i64;
    let mut best_val = f32::NEG_INFINITY;
    let mut found = false;
    for (i, &v) in row.iter().enumerate() {
        if avoid.is_some_and(|a| a == i as i64) {
            continue;
        }
        if v.is_finite() && v > best_val {
            best_val = v;
            best = i as i64;
            found = true;
        }
    }
    if found { best } else { 0 }
}

/// 前缀脚手架清理：模型可能自发生成
/// "language Chinese<asr_text>" 类前缀（1.7B int8 导出实测高频出现，且可能带
/// 前导空格/不同语言词），只要前 16 个 token 内出现 <asr_text>，就丢弃其及
pub(crate) fn strip_language_prefix(generated: &[i64], _tok: &AsrTokenizer) -> Vec<i64> {
    if generated.is_empty() {
        return Vec::new();
    }
    let prefix_window = 16.min(generated.len());
    match generated[..prefix_window]
        .iter()
        .position(|&id| id == super::tokenizer::ASR_TEXT_ID)
    {
        Some(pos) if pos > 0 => return generated[pos + 1..].to_vec(),
        _ => {}
    }
    generated.to_vec()
}

pub struct Qwen3Asr {
    conv: Session,
    encoder: Session,
    decoder: Session,
    dec_in: Vec<String>,
    dec_out: Vec<String>,
    num_layers: usize,
    max_total_len: i64,
    kv_heads: i64,
    head_dim: i64,
    // 当前段 encoder 输出（行主序 [audio_len, audio_hidden]，decode 每步借用）
    audio_feats: Vec<f32>,
    audio_len: usize,
    audio_hidden: usize,
    cache: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Qwen3Asr {
    /// 每个 worker 独立加载一套 session（ort rc.13 禁止并发 Run 同一 Session，
    /// session/mod.rs:118-122 官方推荐 per-thread session；权重内存 ×worker 数）。
    /// 三图先探测后构建（decision/translate 同款"探测先于 Session::builder"——
    /// ort/load-dynamic 缺动态库时 builder 触发 non-unwinding panic，缺失必须
    /// 在此之前以 ModelLoadError 显性返回，R-transcribe-engine-004）。
    /// 返回实例与探测到的三图路径：上层（`Qwen3AsrEngine`）需用 decoder 实际路径做
    /// 精度判定，避免按写死文件名二次拼路径（历史缺陷，见 `DECODER_CANDIDATES`）。
    pub fn new(models_dir: &Path, opts: &AsrSessionOpts) -> AsrResult<(Self, AsrGraphPaths)> {
        let graphs = resolve_graph_paths(models_dir)?;
        let conv = session_from(&graphs.conv, opts)?;
        if conv.inputs().len() != 1 || conv.outputs().len() != 1 {
            return Err(asr_load(format!(
                "conv_frontend IO mismatch: {} inputs / {} outputs, expect 1/1",
                conv.inputs().len(),
                conv.outputs().len()
            )));
        }
        let encoder = session_from(&graphs.encoder, opts)?;
        if encoder.inputs().len() != 2 || encoder.outputs().len() != 1 {
            return Err(asr_load(format!(
                "encoder IO mismatch: {} inputs / {} outputs, expect 2/1",
                encoder.inputs().len(),
                encoder.outputs().len()
            )));
        }
        let decoder = session_from(&graphs.decoder, opts)?;
        // I/O dtype 守卫（f32-only，见 ensure_f32_io 文档）
        ensure_f32_io(&conv, "conv_frontend")?;
        ensure_f32_io(&encoder, "encoder")?;
        ensure_f32_io(&decoder, "decoder")?;

        let num_outputs = decoder.outputs().len();
        if num_outputs < 1 || (num_outputs - 1) % 2 != 0 {
            return Err(asr_load(format!(
                "decoder outputs mismatch: expect 1+2*num_layers, got {num_outputs}"
            )));
        }
        let num_layers = (num_outputs - 1) / 2;
        if decoder.inputs().len() != 4 + 2 * num_layers {
            return Err(asr_load(format!(
                "decoder inputs mismatch: expect {}, got {}",
                4 + 2 * num_layers,
                decoder.inputs().len()
            )));
        }

        let dec_in: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let dec_out: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let numel = (MAX_TOTAL_LEN * KV_HEADS * HEAD_DIM) as usize;
        let cache = vec![(vec![0.0f32; numel], vec![0.0f32; numel]); num_layers];

        Ok((
            Self {
                conv,
                encoder,
                decoder,
                dec_in,
                dec_out,
                num_layers,
                max_total_len: MAX_TOTAL_LEN,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                audio_feats: Vec::new(),
                audio_len: 0,
                audio_hidden: 0,
                cache,
            },
            graphs,
        ))
    }

    /// mel [num_frames×128] → conv → mask → encoder；结果缓存在 self.audio_feats，
    /// 返回 valid_audio_token_len
    fn encode_audio(&mut self, mel: &[f32], num_frames: usize) -> AsrResult<usize> {
        let expected = feat_to_audio_tokens_len(num_frames as i32, CONV_CHUNK_SIZE);

        let conv_in = self.conv.inputs()[0].name().to_string();
        let conv_out = self.conv.outputs()[0].name().to_string();
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            conv_in.as_str(),
            SessionInputValue::from(
                Tensor::from_array((
                    vec![1i64, num_frames as i64, super::features::MEL_DIM as i64],
                    mel.to_vec(),
                ))
                .map_err(asr_infer)?,
            ),
        );
        let outputs = self.conv.run(inputs).map_err(asr_infer)?;
        let (shape, conv_data) = {
            let (s, d) = outputs[conv_out.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(asr_infer)?;
            (s[..].to_vec(), d.to_vec())
        };
        if shape.len() != 3 || shape[1] <= 0 {
            return Err(asr_load(format!("unexpected conv output shape {shape:?}")));
        }
        let conv_frames = shape[1] as usize;
        let hidden = shape[2] as usize;

        // encoder mask：前 min(expected, conv_frames) 个 true
        let valid = (expected as usize).min(conv_frames);
        let mut mask = vec![false; conv_frames];
        mask[..valid].fill(true);

        let enc_in0 = self.encoder.inputs()[0].name().to_string();
        let enc_in1 = self.encoder.inputs()[1].name().to_string();
        let enc_out = self.encoder.outputs()[0].name().to_string();
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            enc_in0.as_str(),
            SessionInputValue::from(
                Tensor::from_array((vec![1i64, conv_frames as i64, hidden as i64], conv_data))
                    .map_err(asr_infer)?,
            ),
        );
        inputs.insert(
            enc_in1.as_str(),
            SessionInputValue::from(
                Tensor::from_array((vec![1i64, conv_frames as i64], mask)).map_err(asr_infer)?,
            ),
        );
        let outputs = self.encoder.run(inputs).map_err(asr_infer)?;
        let (shape, data) = {
            let (s, d) = outputs[enc_out.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(asr_infer)?;
            (s[..].to_vec(), d.to_vec())
        };
        if shape.len() != 3 || shape[1] <= 0 {
            return Err(asr_load(format!(
                "unexpected encoder output shape {shape:?}"
            )));
        }
        self.audio_len = shape[1] as usize;
        self.audio_hidden = shape[2] as usize;
        self.audio_feats = data;
        Ok(valid)
    }

    /// 一次 decoder 前向：input_ids/attention_mask/cache_position(iota(pos0..))
    /// owned，KV 与 audio_features 以 View 借用；返回最后一行 logits 的
    /// (argmax, 避开 EOS 的 argmax)，并把 KV delta 原地写回 cache
    /// （等价 ForwardLLM + ApplyKvDeltaInplace + 两次采样）
    fn forward_llm(&mut self, ids: &[i64], pos0: i64) -> AsrResult<(i64, i64)> {
        // 字段分裂借用：decoder 独占可变借用，其余字段不可变借用
        let Self {
            decoder,
            dec_in,
            dec_out,
            cache,
            audio_feats,
            audio_len,
            audio_hidden,
            kv_heads,
            head_dim,
            max_total_len,
            ..
        } = self;

        let kv_shape = [
            1usize,
            *max_total_len as usize,
            *kv_heads as usize,
            *head_dim as usize,
        ];
        let per_pos = (*kv_heads * *head_dim) as usize;
        let mask: Vec<i64> = vec![1; ids.len()];
        let pos: Vec<i64> = (pos0..pos0 + ids.len() as i64).collect();

        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            dec_in[0].as_str(),
            SessionInputValue::from(
                Tensor::from_array((vec![1i64, ids.len() as i64], ids.to_vec()))
                    .map_err(asr_infer)?,
            ),
        );
        inputs.insert(
            dec_in[1].as_str(),
            SessionInputValue::from(
                Tensor::from_array((
                    vec![1i64, *audio_len as i64, *audio_hidden as i64],
                    audio_feats.clone(),
                ))
                .map_err(asr_infer)?,
            ),
        );
        inputs.insert(
            dec_in[2].as_str(),
            SessionInputValue::from(
                Tensor::from_array((vec![1i64, ids.len() as i64], mask)).map_err(asr_infer)?,
            ),
        );
        inputs.insert(
            dec_in[3].as_str(),
            SessionInputValue::from(
                Tensor::from_array((vec![pos.len() as i64], pos.clone())).map_err(asr_infer)?,
            ),
        );
        for (i, (k, v)) in cache.iter().enumerate() {
            let kview = ArrayView4::from_shape(kv_shape, k).map_err(asr_load)?;
            let vview = ArrayView4::from_shape(kv_shape, v).map_err(asr_load)?;
            inputs.insert(
                dec_in[4 + 2 * i].as_str(),
                SessionInputValue::from(
                    ort::value::TensorRef::from_array_view(kview).map_err(asr_infer)?,
                ),
            );
            inputs.insert(
                dec_in[5 + 2 * i].as_str(),
                SessionInputValue::from(
                    ort::value::TensorRef::from_array_view(vview).map_err(asr_infer)?,
                ),
            );
        }

        let outputs = decoder.run(inputs).map_err(asr_infer)?;

        // logits [1, seq, vocab]：取最后一行做两次 argmax
        let (lshape, ldata) = outputs[dec_out[0].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(asr_infer)?;
        let ldims: Vec<i64> = lshape[..].to_vec();
        if ldims.len() != 3 || ldims[1] <= 0 || ldims[2] <= 0 {
            return Err(asr_load(format!("unexpected logits shape {ldims:?}")));
        }
        let time_dim = ldims[1] as usize;
        let vocab = ldims[2] as usize;
        if ldata.len() != time_dim * vocab {
            return Err(asr_load(format!(
                "logits len {} != {time_dim}*{vocab}",
                ldata.len()
            )));
        }
        let last_row = &ldata[(time_dim - 1) * vocab..];
        let argmax = greedy_argmax(last_row, None);
        let argmax_no_eos = greedy_argmax(last_row, Some(super::tokenizer::EOS_ID));

        // KV delta → 原地写回（ApplyKvDeltaInplace，B=1、pos 连续）
        let mut deltas: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(cache.len());
        for i in 0..cache.len() {
            let (_, dk) = outputs[dec_out[1 + 2 * i].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(asr_infer)?;
            let (_, dv) = outputs[dec_out[2 + 2 * i].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(asr_infer)?;
            deltas.push((dk.to_vec(), dv.to_vec()));
        }
        drop(outputs);

        let delta_seq = deltas.first().map(|(k, _)| k.len() / per_pos).unwrap_or(0);
        let copy_s = delta_seq.min(pos.len());
        if copy_s > 0 {
            let start = pos[0] as usize;
            if start + copy_s > *max_total_len as usize {
                return Err(asr_load(format!(
                    "cache_position {start} + {copy_s} exceeds max_total_len {max_total_len}"
                )));
            }
            for (layer, ((ck, cv), (dk, dv))) in cache.iter_mut().zip(deltas.iter()).enumerate() {
                if dk.len() != copy_s * per_pos || dv.len() != copy_s * per_pos {
                    return Err(asr_load(format!("layer {layer} kv delta len mismatch")));
                }
                ck[start * per_pos..(start + copy_s) * per_pos]
                    .copy_from_slice(&dk[..copy_s * per_pos]);
                cv[start * per_pos..(start + copy_s) * per_pos]
                    .copy_from_slice(&dv[..copy_s * per_pos]);
            }
        }

        Ok((argmax, argmax_no_eos))
    }

    /// 一段 mel 特征 → 转录文本
    pub fn transcribe(
        &mut self,
        mel: &[f32],
        num_frames: usize,
        tok: &AsrTokenizer,
    ) -> AsrResult<String> {
        if num_frames < MIN_FRAMES || mel.len() < num_frames * super::features::MEL_DIM {
            return Ok(String::new());
        }

        let valid = self.encode_audio(mel, num_frames)?;

        // TrimAudioFeatures：从尾向前丢 |v|≤1e-6 的全静音帧
        let h = self.audio_hidden;
        let a = self.audio_len;
        let mut a_valid = 0usize;
        for i in (0..a).rev() {
            let row = &self.audio_feats[i * h..(i + 1) * h];
            if row.iter().any(|v| v.abs() > 1e-6) {
                a_valid = i + 1;
                break;
            }
        }
        let all_silent = a_valid == 0;
        let mut audio_token_len = valid.min(a_valid);
        if all_silent {
            audio_token_len = 0;
        }
        if audio_token_len == 0 {
            return Ok(String::new());
        }
        self.audio_feats.truncate(audio_token_len * h);
        self.audio_len = audio_token_len;

        // prompt 构造
        let before =
            tok.encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
        let after = tok.encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?;
        let one_audio_len = 1usize; // <|audio_pad|> 单 token（tokenizer_config.json 实测）
        let mut source_ids = before.clone();
        source_ids.reserve(audio_token_len * one_audio_len + after.len());
        source_ids.extend(std::iter::repeat_n(
            super::tokenizer::AUDIO_PAD_ID,
            audio_token_len * one_audio_len,
        ));
        source_ids.extend_from_slice(&after);
        let before_len = before.len();
        let after_len = after.len();
        let mut context_len = source_ids.len();

        // context 超限时截断 audio 占位
        let max_seq_len = self.max_total_len as usize;
        if context_len > max_seq_len {
            let keep_audio = (max_seq_len - before_len - after_len) / one_audio_len;
            if keep_audio == 0 {
                return Err(asr_load(
                    "no room for audio placeholders within max_total_len",
                ));
            }
            if keep_audio < audio_token_len {
                let mut ids = before.clone();
                ids.extend(std::iter::repeat_n(
                    super::tokenizer::AUDIO_PAD_ID,
                    keep_audio * one_audio_len,
                ));
                ids.extend_from_slice(&after);
                source_ids = ids;
                self.audio_len = keep_audio;
                self.audio_feats.truncate(keep_audio * h);
                context_len = source_ids.len();
            }
        }

        // 重置全零 KV（每段独立，CreateEmptyKVCache）
        let numel = (self.max_total_len * self.kv_heads * self.head_dim) as usize;
        self.cache = vec![(vec![0.0f32; numel], vec![0.0f32; numel]); self.num_layers];

        // prefill + 首 token：首 token 为 EOS 时去掉 EOS
        // 重采样一次，仍 EOS → 空结果
        let (mut next_id, next_no_eos) = self.forward_llm(&source_ids, 0)?;
        if next_id == EOS_ID {
            next_id = next_no_eos;
            if next_id == EOS_ID {
                return Ok(String::new());
            }
        }

        let mut generated: Vec<i64> = vec![next_id];

        for (cur_len, _step) in (context_len as i64..).zip(1..MAX_NEW_TOKENS) {
            if cur_len >= max_seq_len as i64 {
                break;
            }
            let (id, _) = self.forward_llm(&[next_id], cur_len)?;
            next_id = id;
            if next_id == EOS_ID {
                break;
            }
            generated.push(next_id);

            if is_degenerate_repetition(&generated) {
                trim_degenerate_tail(&mut generated);
                break;
            }
        }

        let cleaned = strip_language_prefix(&generated, tok);
        let text = tok.decode(&cleaned);
        Ok(AsrTokenizer::remove_replacement_chars(&text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feat_to_audio_tokens_len_matches_reference() {
        // 整块：100 帧 → conv_out_len_3x_stride2(100)=(51→26→13)=13 token
        assert_eq!(feat_to_audio_tokens_len(100, 100), 13);
        // 多块 + 余数：250 = 2×100 + 50 → 2×13 + aftercnn(50)=(25→13→7)=7
        assert_eq!(feat_to_audio_tokens_len(250, 100), 33);
        assert_eq!(feat_to_audio_tokens_len(0, 100), 0);
        // DESIGN §7 核算：10s 音频 1000 帧 ≈ 130 token 上限，26 token/20s 段量级
        assert_eq!(feat_to_audio_tokens_len(1000, 100), 130);
    }

    #[test]
    fn degenerate_repetition_detection() {
        // 少于窗口：不判定
        let short = vec![10i64, 20, 30];
        assert!(!is_degenerate_repetition(&short));
        // 末 64 个窗口恰为 3-token 循环：判定塌缩
        let mut ids = vec![10i64];
        ids.extend([11i64, 22, 33].iter().copied().cycle().take(64 * 3));
        assert_eq!(ids.len(), 1 + 64 * 3);
        assert!(is_degenerate_repetition(&ids));
        trim_degenerate_tail(&mut ids);
        assert_eq!(ids.len(), 1 + 64 * 3 - LOOP_WINDOW);
        assert_eq!(ids[0], 10);
        // 末窗口含 4 个 distinct：不判定塌缩
        let mut ids = vec![10i64, 20];
        ids.extend([11i64, 22, 33, 44].iter().copied().cycle().take(64 * 3));
        assert!(!is_degenerate_repetition(&ids));
    }

    #[test]
    fn greedy_argmax_skips_non_finite_and_avoids() {
        let row = [1.0f32, f32::NAN, 5.0, f32::NEG_INFINITY, 3.0];
        assert_eq!(greedy_argmax(&row, None), 2);
        assert_eq!(greedy_argmax(&row, Some(2)), 4);
        assert_eq!(greedy_argmax(&[f32::NAN; 4], None), 0);
    }

    /// 回归钉（2026-10-18 修复）：纯名（无损导出）图必须能探测到——
    /// 修复前候选写死 `.int8.onnx`，fp32/fp16 bundle 无入口。
    /// 两图共存时 int8 优先（与修复前行为逐字一致）。
    #[test]
    fn resolve_graph_paths_prefers_int8_then_plain_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("qwen3-conv_frontend.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("qwen3-encoder.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("qwen3-decoder.onnx"), b"x").unwrap();

        let g = resolve_graph_paths(dir.path()).unwrap();
        assert_eq!(
            g.encoder.file_name().unwrap(),
            "qwen3-encoder.onnx",
            "纯名无损图必须可探测（修复前无入口）"
        );
        assert_eq!(g.decoder.file_name().unwrap(), "qwen3-decoder.onnx");

        std::fs::write(dir.path().join("qwen3-encoder.int8.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("qwen3-decoder.int8.onnx"), b"x").unwrap();
        let g = resolve_graph_paths(dir.path()).unwrap();
        assert_eq!(g.encoder.file_name().unwrap(), "qwen3-encoder.int8.onnx");
        assert_eq!(g.decoder.file_name().unwrap(), "qwen3-decoder.int8.onnx");
    }

    /// miss 时报错消息必须列全**该角色**的全部候选（含纯名），且带角色名，
    /// 便于定位“导出后缀不对”。首个 miss 即返回（encoder 先于 decoder），
    /// 故只钉 encoder 候选，不要求 decoder 候选也出现。
    #[test]
    fn resolve_graph_paths_miss_lists_all_candidates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("qwen3-conv_frontend.onnx"), b"x").unwrap();
        let err = resolve_graph_paths(dir.path()).unwrap_err();
        let msg = err.error_detail();
        assert!(msg.contains("encoder"), "错误消息须带角色名，got {msg}");
        for name in ENCODER_CANDIDATES {
            assert!(msg.contains(name), "错误消息须含候选 {name}，got {msg}");
        }
    }

    /// I/O dtype 守卫：只放行 f32 / i64 / bool；fp16、bf16、f64 必须拒绝
    /// （当前 KV cache 与张量 I/O 均写死 f32，半精度图需 dtype 泛型化才能支持）。
    #[test]
    fn element_type_guard_rejects_half_precision() {
        for ok in [
            TensorElementType::Float32,
            TensorElementType::Int64,
            TensorElementType::Bool,
        ] {
            assert!(element_type_supported(&ok), "应当放行 {ok:?}");
        }
        for bad in [
            TensorElementType::Float16,
            TensorElementType::Bfloat16,
            TensorElementType::Float64,
        ] {
            assert!(!element_type_supported(&bad), "必须拒绝 {bad:?}");
        }
    }
}
