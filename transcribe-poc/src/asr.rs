//! Qwen3-ASR 三图推理与贪心解码（conv_frontend → encoder → decoder KV-cache）。
//!
//! 移植参考（Apache-2.0，仅作移植参考、非 Cargo 依赖），k2-fsa/sherpa-onnx
//! v1.13.8（commit 11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf）：
//! - offline-recognizer-qwen3-asr-impl.cc:77-108  FeatToAudioTokensLen
//! - 同文件 :129-207  TrimAudioFeatures（从尾向前丢 |v|≤1e-6 帧）
//! - 同文件 :321-350, 1089-1102  复读守卫（窗口 64、distinct ≤3）与尾部裁剪
//! - 同文件 :391-401  去除 U+FFFD
//! - 同文件 :459-507  BuildSourceIds（prompt 模板，PoC 不传 hotwords/language）
//! - 同文件 :940-1147  prefill + 逐步 greedy（temperature=1e-6 退化为 argmax，
//!   跳过非有限值）、首 token EOS 重采样一次、cur_len 达 max_seq_len 停止
//! - offline-qwen3-asr-model.cc:38-39, 438-462  decoder 输入序
//!   [ids, audio, mask, cache_position, past_kv…]；:500-530 输出序
//!   [logits, kv delta…]；:533-581 全零 KV 预分配 [1, max_total_len, 8, 128]；
//!   :583-749 ApplyKvDeltaInplace（按 pos0 连续 memcpy）
//! - max_total_len：past_key dim1 为符号维度时回退配置默认 512
//!   （offline-qwen3-asr-model.cc:311-324 + offline-qwen3-asr-model-config.h:31；
//!   ort rc.13 不暴露静态 shape，本实现直接取回退值 512；kv_heads/head_dim
//!   同理取 8/128，与 docs-transcribe/DESIGN.md §4 对实际模型的侦察一致）
//!
//! 串行实现（本轮正确性优先）：单线程 session，KV 经 ArrayView 借用输入，
//! 避免每步复制约 117 MiB 的纯浪费流量。

use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayView4;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::tokenizer::{AsrTokenizer, EOS_ID};

pub type AsrResult<T> = Result<T, Box<dyn std::error::Error>>;

pub(crate) const MAX_TOTAL_LEN: i64 = 512;
pub(crate) const KV_HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;
pub(crate) const CONV_CHUNK_SIZE: i32 = 100; // kQwen3ChunkSize（impl.cc:43）
pub(crate) const MAX_NEW_TOKENS: usize = 128; // sherpa CLI 默认（config.h:34）
pub(crate) const MIN_FRAMES: usize = 2; // impl.cc:1177：帧数 <2 直接空结果

/// 复读守卫常量（impl.cc:321-322）
const LOOP_WINDOW: usize = 64;
const LOOP_MAX_DISTINCT: usize = 3;

/// session 线程配置：env_pool=true 时环境已建全局线程池，session 跳过
/// intra 设置（ORT 自动用 env pool）；no_spin 关闭 intra 线程自旋，
/// 多 session 并发时减少核间争抢；cuda=true 注册 CUDA EP（需 dylib 带
/// CUDA EP，由 TRANSCRIBE_EP=cuda 开启，失败显性报错不静默回退）
#[derive(Debug, Clone, Copy)]
pub struct SessionOpts {
    pub intra_threads: usize,
    pub no_spin: bool,
    pub env_pool: bool,
    pub cuda: bool,
}

pub(crate) fn session_from(path: &Path, opts: &SessionOpts) -> AsrResult<Session> {
    let mut builder = Session::builder()?;
    if !opts.env_pool {
        builder = builder.with_intra_threads(opts.intra_threads)?;
    }
    if opts.no_spin {
        builder = builder.with_config_entry("session.intra_op.allow_spinning", "0")?;
    }
    if opts.cuda {
        builder = builder.with_execution_providers([ort::ep::CUDA::default().build()])?;
    }
    Ok(builder
        .commit_from_file(path)
        .map_err(|e| format!("load {} failed: {e}", path.display()))?)
}

/// FeatToAudioTokensLen（impl.cc:77-108）
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

/// 复读守卫（impl.cc:326-350）：最近 LOOP_WINDOW 个 token 内 distinct ≤3
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

/// 裁掉塌缩循环尾部（impl.cc:357-389）：只删属于末窗口 token 集的尾 token
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

/// 贪心 argmax，temperature=1e-6 语义：跳过非有限值（impl.cc:593-627）
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

/// 前缀脚手架清理（impl.cc:1105-1123）：模型可能自发生成
/// "language Chinese<asr_text>" 类前缀（1.7B int8 导出实测高频出现，且可能带
/// 前导空格/不同语言词），只要前 16 个 token 内出现 <asr_text>，就丢弃其及
/// 之前全部内容（sherpa 原实现同样只按 <asr_text> 位置裁剪）
pub(crate) fn strip_language_prefix(generated: &[i64], tok: &AsrTokenizer) -> Vec<i64> {
    if generated.is_empty() {
        return Vec::new();
    }
    let prefix_window = 16.min(generated.len());
    if let Some(pos) = generated[..prefix_window]
        .iter()
        .position(|&id| id == crate::tokenizer::ASR_TEXT_ID)
    {
        if pos > 0 {
            return generated[pos + 1..].to_vec();
        }
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
    /// session/mod.rs:118-122 官方推荐 per-thread session；权重内存 ×worker 数）
    pub fn new(models_dir: &Path, opts: &SessionOpts) -> AsrResult<Self> {
        let conv = session_from(&models_dir.join("qwen3-conv_frontend.onnx"), opts)?;
        if conv.inputs().len() != 1 || conv.outputs().len() != 1 {
            return Err(format!(
                "conv_frontend IO mismatch: {} inputs / {} outputs, expect 1/1",
                conv.inputs().len(),
                conv.outputs().len()
            )
            .into());
        }
        let encoder = session_from(&models_dir.join("qwen3-encoder.int8.onnx"), opts)?;
        if encoder.inputs().len() != 2 || encoder.outputs().len() != 1 {
            return Err(format!(
                "encoder IO mismatch: {} inputs / {} outputs, expect 2/1",
                encoder.inputs().len(),
                encoder.outputs().len()
            )
            .into());
        }
        let decoder = session_from(&models_dir.join("qwen3-decoder.int8.onnx"), opts)?;

        let num_outputs = decoder.outputs().len();
        if num_outputs < 1 || (num_outputs - 1) % 2 != 0 {
            return Err(format!(
                "decoder outputs mismatch: expect 1+2*num_layers, got {num_outputs}"
            )
            .into());
        }
        let num_layers = (num_outputs - 1) / 2;
        if decoder.inputs().len() != 4 + 2 * num_layers {
            return Err(format!(
                "decoder inputs mismatch: expect {}, got {}",
                4 + 2 * num_layers,
                decoder.inputs().len()
            )
            .into());
        }

        let dec_in: Vec<String> =
            decoder.inputs().iter().map(|o| o.name().to_string()).collect();
        let dec_out: Vec<String> =
            decoder.outputs().iter().map(|o| o.name().to_string()).collect();

        let numel = (MAX_TOTAL_LEN * KV_HEADS * HEAD_DIM) as usize;
        let cache = vec![(vec![0.0f32; numel], vec![0.0f32; numel]); num_layers];

        Ok(Self {
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
        })
    }

    /// mel [num_frames×128] → conv → mask → encoder；结果缓存在 self.audio_feats，
    /// 返回 valid_audio_token_len（impl.cc:1196-1231 的模型段）
    fn encode_audio(&mut self, mel: &[f32], num_frames: usize) -> AsrResult<usize> {
        let expected = feat_to_audio_tokens_len(num_frames as i32, CONV_CHUNK_SIZE);

        let conv_in = self.conv.inputs()[0].name().to_string();
        let conv_out = self.conv.outputs()[0].name().to_string();
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            conv_in.as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, num_frames as i64, crate::features::MEL_DIM as i64],
                mel.to_vec(),
            ))?),
        );
        let outputs = self.conv.run(inputs)?;
        let (shape, conv_data) = {
            let (s, d) = outputs[conv_out.as_str()].try_extract_tensor::<f32>()?;
            (s[..].to_vec(), d.to_vec())
        };
        if shape.len() != 3 || shape[1] <= 0 {
            return Err(format!("unexpected conv output shape {shape:?}").into());
        }
        let conv_frames = shape[1] as usize;
        let hidden = shape[2] as usize;

        // encoder mask：前 min(expected, conv_frames) 个 true（impl.cc:1206-1218）
        let valid = (expected as usize).min(conv_frames);
        let mut mask = vec![false; conv_frames];
        mask[..valid].fill(true);

        let enc_in0 = self.encoder.inputs()[0].name().to_string();
        let enc_in1 = self.encoder.inputs()[1].name().to_string();
        let enc_out = self.encoder.outputs()[0].name().to_string();
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            enc_in0.as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, conv_frames as i64, hidden as i64],
                conv_data,
            ))?),
        );
        inputs.insert(
            enc_in1.as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, conv_frames as i64],
                mask,
            ))?),
        );
        let outputs = self.encoder.run(inputs)?;
        let (shape, data) = {
            let (s, d) = outputs[enc_out.as_str()].try_extract_tensor::<f32>()?;
            (s[..].to_vec(), d.to_vec())
        };
        if shape.len() != 3 || shape[1] <= 0 {
            return Err(format!("unexpected encoder output shape {shape:?}").into());
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
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, ids.len() as i64],
                ids.to_vec(),
            ))?),
        );
        inputs.insert(
            dec_in[1].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, *audio_len as i64, *audio_hidden as i64],
                audio_feats.clone(),
            ))?),
        );
        inputs.insert(
            dec_in[2].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![1i64, ids.len() as i64],
                mask,
            ))?),
        );
        inputs.insert(
            dec_in[3].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![pos.len() as i64],
                pos.clone(),
            ))?),
        );
        for (i, (k, v)) in cache.iter().enumerate() {
            let kview = ArrayView4::from_shape(kv_shape, k)
                .map_err(|e| format!("key view: {e}"))?;
            let vview = ArrayView4::from_shape(kv_shape, v)
                .map_err(|e| format!("value view: {e}"))?;
            inputs.insert(
                dec_in[4 + 2 * i].as_str(),
                SessionInputValue::from(ort::value::TensorRef::from_array_view(kview)?),
            );
            inputs.insert(
                dec_in[5 + 2 * i].as_str(),
                SessionInputValue::from(ort::value::TensorRef::from_array_view(vview)?),
            );
        }

        let outputs = decoder.run(inputs)?;

        // logits [1, seq, vocab]：取最后一行做两次 argmax
        let (lshape, ldata) = outputs[dec_out[0].as_str()].try_extract_tensor::<f32>()?;
        let ldims: Vec<i64> = lshape[..].to_vec();
        if ldims.len() != 3 || ldims[1] <= 0 || ldims[2] <= 0 {
            return Err(format!("unexpected logits shape {ldims:?}").into());
        }
        let time_dim = ldims[1] as usize;
        let vocab = ldims[2] as usize;
        if ldata.len() != time_dim * vocab {
            return Err(format!("logits len {} != {time_dim}*{vocab}", ldata.len()).into());
        }
        let last_row = &ldata[(time_dim - 1) * vocab..];
        let argmax = greedy_argmax(last_row, None);
        let argmax_no_eos = greedy_argmax(last_row, Some(crate::tokenizer::EOS_ID));

        // KV delta → 原地写回（ApplyKvDeltaInplace，B=1、pos 连续）
        let mut deltas: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(cache.len());
        for i in 0..cache.len() {
            let (_, dk) = outputs[dec_out[1 + 2 * i].as_str()].try_extract_tensor::<f32>()?;
            let (_, dv) = outputs[dec_out[2 + 2 * i].as_str()].try_extract_tensor::<f32>()?;
            deltas.push((dk.to_vec(), dv.to_vec()));
        }
        drop(outputs);

        let delta_seq = deltas.first().map(|(k, _)| k.len() / per_pos).unwrap_or(0);
        let copy_s = delta_seq.min(pos.len());
        if copy_s > 0 {
            let start = pos[0] as usize;
            if start + copy_s > *max_total_len as usize {
                return Err(format!(
                    "cache_position {start} + {copy_s} exceeds max_total_len {max_total_len}"
                )
                .into());
            }
            for (layer, ((ck, cv), (dk, dv))) in cache.iter_mut().zip(deltas.iter()).enumerate() {
                if dk.len() != copy_s * per_pos || dv.len() != copy_s * per_pos {
                    return Err(format!("layer {layer} kv delta len mismatch").into());
                }
                ck[start * per_pos..(start + copy_s) * per_pos]
                    .copy_from_slice(&dk[..copy_s * per_pos]);
                cv[start * per_pos..(start + copy_s) * per_pos]
                    .copy_from_slice(&dv[..copy_s * per_pos]);
            }
        }

        Ok((argmax, argmax_no_eos))
    }

    /// 一段 mel 特征 → 转录文本（impl.cc Decode + GenerateText 全流程）
    pub fn transcribe(
        &mut self,
        mel: &[f32],
        num_frames: usize,
        tok: &AsrTokenizer,
    ) -> AsrResult<String> {
        if num_frames < MIN_FRAMES || mel.len() < num_frames * crate::features::MEL_DIM {
            return Ok(String::new());
        }

        let valid = self.encode_audio(mel, num_frames)?;

        // TrimAudioFeatures（impl.cc:129-207）：从尾向前丢 |v|≤1e-6 的全静音帧
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

        // BuildSourceIds（impl.cc:459-507，hotwords/language 为空）
        let before =
            tok.encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
        let after = tok.encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?;
        let one_audio_len = 1usize; // <|audio_pad|> 单 token（tokenizer_config.json 实测）
        let mut source_ids = before.clone();
        source_ids.reserve(audio_token_len * one_audio_len + after.len());
        source_ids.extend(std::iter::repeat_n(
            crate::tokenizer::AUDIO_PAD_ID,
            audio_token_len * one_audio_len,
        ));
        source_ids.extend_from_slice(&after);
        let before_len = before.len();
        let after_len = after.len();
        let mut context_len = source_ids.len();

        // context 超限时截断 audio 占位（impl.cc:873-938）
        let max_seq_len = self.max_total_len as usize;
        if context_len > max_seq_len {
            let keep_audio = (max_seq_len - before_len - after_len) / one_audio_len;
            if keep_audio == 0 {
                return Err("no room for audio placeholders within max_total_len".into());
            }
            if keep_audio < audio_token_len {
                let mut ids = before.clone();
                ids.extend(std::iter::repeat_n(
                    crate::tokenizer::AUDIO_PAD_ID,
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

        // prefill + 首 token（impl.cc:956-1025）：首 token 为 EOS 时去掉 EOS
        // 重采样一次，仍 EOS → 空结果
        let (mut next_id, next_no_eos) = self.forward_llm(&source_ids, 0)?;
        if next_id == EOS_ID {
            next_id = next_no_eos;
            if next_id == EOS_ID {
                return Ok(String::new());
            }
        }

        let mut generated: Vec<i64> = vec![next_id];

        // cur_len = 本次前向的 cache_position（与 sherpa :1026-1103 同节奏）
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
        // 裁剪：只删末窗口内的循环 token（sherpa 语义，窗口外不回溯）
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
}
