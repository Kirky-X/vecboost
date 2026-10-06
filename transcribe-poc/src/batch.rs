//! 批量合批推理：多 VAD 段 lockstep prefill + decode。
//!
//! 借鉴官方 qwen_asr `max_inference_batch_size` 的合批思路：解码每步重复
//! 读取全量权重（CPU 上访存是瓶颈，档位 A/B 已证多 worker 已饱和带宽），
//! 合批把权重流量摊薄给 batch 内多段。三张图 batch 维均为动态
//! （docs-transcribe/DESIGN.md §5-7 实测 [batch, ...]）。
//!
//! 填充策略（官方 split_audio_into_chunks 同款）：mel 统一填充到批内最长
//! 帧数（真实静音帧），encoder 全真 mask，全部 item 占位数/prompt 长一致——
//! 不存在 pad KV，无需依赖图内 mask 屏蔽语义（实测图的 mask 不屏蔽 0 填充，
//! 右填充方案 CER 劣化 7.75%→11.28% 已废弃）。解码步 mask 与串行一致传全 1。

use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayView4;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::asr::{
    feat_to_audio_tokens_len, greedy_argmax, is_degenerate_repetition, session_from,
    strip_language_prefix, trim_degenerate_tail, AsrResult, SessionOpts, CONV_CHUNK_SIZE,
    KV_HEADS, MAX_NEW_TOKENS, MAX_TOTAL_LEN, MIN_FRAMES,
};
use crate::tokenizer::{AsrTokenizer, AUDIO_PAD_ID, EOS_ID};

/// 一段待转录的 mel 特征（帧数 × 128，行主序）
pub struct BatchItem {
    pub mel: Vec<f32>,
    pub num_frames: usize,
}

pub(crate) struct Slot {
    pub(crate) prompt_len: usize,
    pub(crate) done: bool,
    pub(crate) generated: Vec<i64>,
}

/// conv/encoder 合批 + 逐 item prompt 构建的结果（batch 与 gpu 两路共用）
pub(crate) struct Prepared {
    pub(crate) slots: Vec<Slot>,
    pub(crate) prompts: Vec<Vec<i64>>,
    pub(crate) audio_buf: Vec<f32>,
    pub(crate) a_max: usize,
    pub(crate) a_hidden: usize,
    pub(crate) p_max: usize,
}

pub struct BatchAsr {
    conv: Session,
    encoder: Session,
    decoder: Session,
    conv_in: String,
    conv_out: String,
    enc_in0: String,
    enc_in1: String,
    enc_out: String,
    dec_in: Vec<String>,
    dec_out: Vec<String>,
    num_layers: usize,
    per_pos: usize,
    cache_numel: usize,
}

impl BatchAsr {
    pub fn new(models_dir: &Path, opts: &SessionOpts) -> AsrResult<Self> {
        let conv = session_from(&models_dir.join("qwen3-conv_frontend.onnx"), opts)?;
        if conv.inputs().len() != 1 || conv.outputs().len() != 1 {
            return Err("conv_frontend IO mismatch: expect 1/1".into());
        }
        let encoder = session_from(&models_dir.join("qwen3-encoder.int8.onnx"), opts)?;
        if encoder.inputs().len() != 2 || encoder.outputs().len() != 1 {
            return Err("encoder IO mismatch: expect 2/1".into());
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
        let conv_in = conv.inputs()[0].name().to_string();
        let conv_out = conv.outputs()[0].name().to_string();
        let enc_in0 = encoder.inputs()[0].name().to_string();
        let enc_in1 = encoder.inputs()[1].name().to_string();
        let enc_out = encoder.outputs()[0].name().to_string();
        let dec_in: Vec<String> =
            decoder.inputs().iter().map(|o| o.name().to_string()).collect();
        let dec_out: Vec<String> =
            decoder.outputs().iter().map(|o| o.name().to_string()).collect();
        Ok(Self {
            conv,
            encoder,
            decoder,
            conv_in,
            conv_out,
            enc_in0,
            enc_in1,
            enc_out,
            dec_in,
            dec_out,
            num_layers,
            per_pos: (KV_HEADS * 128) as usize,
            cache_numel: (MAX_TOTAL_LEN * KV_HEADS * 128) as usize,
        })
    }

    /// 全部段转录；输出顺序与输入一致。
    pub fn transcribe_all(
        &mut self,
        segs: &[BatchItem],
        tok: &AsrTokenizer,
        batch_size: usize,
    ) -> AsrResult<Vec<String>> {
        let n = segs.len();
        let mut out: Vec<String> = vec![String::new(); n];
        // 按帧数升序分批：相邻段长度相近，填充浪费最小
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| segs[i].num_frames);
        for chunk in order.chunks(batch_size.max(1)) {
            let items: Vec<&BatchItem> = chunk.iter().map(|&i| &segs[i]).collect();
            let texts = self.transcribe_batch(&items, tok)?;
            for (&src, text) in chunk.iter().zip(texts) {
                out[src] = text;
            }
        }
        Ok(out)
    }

    /// 一次 decoder 前向（batch 维 B）：返回展平 logits [B,seq,V] 与逐层 KV delta。
    fn decoder_forward(
        &mut self,
        b: usize,
        ids: &[i64],
        mask: &[i64],
        mask_len: i64,
        pos: &[i64],
        audio: &[f32],
        audio_len: i64,
        a_hidden: i64,
        seq: i64,
        kc: &[Vec<f32>],
        vc: &[Vec<f32>],
    ) -> AsrResult<(Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>)> {
        let kv_shape = [b, MAX_TOTAL_LEN as usize, KV_HEADS as usize, 128];
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            self.dec_in[0].as_str(),
            SessionInputValue::from(Tensor::from_array((vec![b as i64, seq], ids.to_vec()))?),
        );
        inputs.insert(
            self.dec_in[1].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![b as i64, audio_len, a_hidden],
                audio.to_vec(),
            ))?),
        );
        inputs.insert(
            self.dec_in[2].as_str(),
            SessionInputValue::from(Tensor::from_array((vec![b as i64, mask_len], mask.to_vec()))?),
        );
        inputs.insert(
            self.dec_in[3].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![pos.len() as i64],
                pos.to_vec(),
            ))?),
        );
        for l in 0..self.num_layers {
            let kview = ArrayView4::from_shape(kv_shape, &kc[l])
                .map_err(|e| format!("key view layer {l}: {e}"))?;
            let vview = ArrayView4::from_shape(kv_shape, &vc[l])
                .map_err(|e| format!("value view layer {l}: {e}"))?;
            inputs.insert(
                self.dec_in[4 + 2 * l].as_str(),
                SessionInputValue::from(ort::value::TensorRef::from_array_view(kview)?),
            );
            inputs.insert(
                self.dec_in[5 + 2 * l].as_str(),
                SessionInputValue::from(ort::value::TensorRef::from_array_view(vview)?),
            );
        }
        let outputs = self.decoder.run(inputs)?;
        let (lshape, ldata) = {
            let (s, d) = outputs[self.dec_out[0].as_str()].try_extract_tensor::<f32>()?;
            (s[..].to_vec(), d.to_vec())
        };
        if lshape.len() != 3 || lshape[0] as usize != b || lshape[2] <= 0 {
            return Err(format!("unexpected logits shape {lshape:?}").into());
        }
        let mut deltas = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let (_, dk) = outputs[self.dec_out[1 + 2 * l].as_str()].try_extract_tensor::<f32>()?;
            let (_, dv) = outputs[self.dec_out[2 + 2 * l].as_str()].try_extract_tensor::<f32>()?;
            deltas.push((dk.to_vec(), dv.to_vec()));
        }
        drop(outputs);
        Ok((ldata, deltas))
    }

    fn transcribe_batch(
        &mut self,
        items: &[&BatchItem],
        tok: &AsrTokenizer,
    ) -> AsrResult<Vec<String>> {
        let b = items.len();
        let prepared = conv_encoder_prepare(
            &mut self.conv, &mut self.encoder,
            &self.conv_in, &self.conv_out, &self.enc_in0, &self.enc_in1, &self.enc_out,
            items, tok,
        )?;
        let Prepared { mut slots, prompts, audio_buf, a_max, a_hidden, p_max } = prepared;

        // ---- 解码输入缓冲（prefill 建、每步复用） ----
        let mut ids_buf = vec![0i64; b * p_max];
        let mut mask_buf = vec![0i64; b * p_max];
        for (bi, p) in prompts.iter().enumerate() {
            ids_buf[bi * p_max..bi * p_max + p.len()].copy_from_slice(p);
            mask_buf[bi * p_max..bi * p_max + p.len()].fill(1);
        }

        // 逐层连续 KV：单 view [B,512,8,128] 覆盖 batch
        let mut kc = vec![vec![0f32; b * self.cache_numel]; self.num_layers];
        let mut vc = vec![vec![0f32; b * self.cache_numel]; self.num_layers];

        // ---- prefill ----
        let pos: Vec<i64> = (0..p_max as i64).collect();
        let (ldata, deltas) = self.decoder_forward(
            b, &ids_buf, &mask_buf, p_max as i64, &pos, &audio_buf, a_max as i64,
            a_hidden as i64, p_max as i64, &kc, &vc,
        )?;
        let vocab = ldata.len() / (b * p_max);
        // 首 token：逐 item 取最后一个真实 token 位置的 logits 行；
        // EOS 重采样一次，仍 EOS → 空结果
        for (bi, slot) in slots.iter_mut().enumerate() {
            if slot.done {
                continue;
            }
            let row = &ldata[(bi * p_max + slot.prompt_len - 1) * vocab..][..vocab];
            let mut next = greedy_argmax(row, None);
            if next == EOS_ID {
                next = greedy_argmax(row, Some(EOS_ID));
                if next == EOS_ID {
                    slot.done = true;
                    continue;
                }
            }
            slot.generated.push(next);
        }
        // KV delta [B, P, per_pos] → 逐 item 写入 0..P
        for (l, (dk, dv)) in deltas.into_iter().enumerate() {
            for bi in 0..b {
                let src = bi * p_max * self.per_pos;
                let dst = bi * self.cache_numel;
                kc[l][dst..dst + p_max * self.per_pos]
                    .copy_from_slice(&dk[src..src + p_max * self.per_pos]);
                vc[l][dst..dst + p_max * self.per_pos]
                    .copy_from_slice(&dv[src..src + p_max * self.per_pos]);
            }
        }

        // ---- lockstep decode（cur = p_max + step - 1；全宽 mask 排除 pad） ----
        for step in 1..MAX_NEW_TOKENS {
            let cur = p_max + step - 1;
            if cur >= MAX_TOTAL_LEN as usize || slots.iter().all(|s| s.done) {
                break;
            }
            let step_ids: Vec<i64> = slots
                .iter()
                .map(|s| if s.done { EOS_ID } else { *s.generated.last().unwrap_or(&EOS_ID) })
                .collect();
            let width = cur + 1;
            let step_mask = vec![1i64; b * width];
            let pos = vec![cur as i64];
            let (ldata, deltas) = self.decoder_forward(
                b, &step_ids, &step_mask, width as i64, &pos, &audio_buf, a_max as i64,
                a_hidden as i64, 1, &kc, &vc,
            )?;
            let vocab = ldata.len() / b;
            for (bi, slot) in slots.iter_mut().enumerate() {
                if slot.done {
                    continue;
                }
                let row = &ldata[bi * vocab..(bi + 1) * vocab];
                let next = greedy_argmax(row, None);
                if next == EOS_ID {
                    slot.done = true;
                    continue;
                }
                slot.generated.push(next);
                if is_degenerate_repetition(&slot.generated) {
                    trim_degenerate_tail(&mut slot.generated);
                    slot.done = true;
                }
            }
            for (l, (dk, dv)) in deltas.into_iter().enumerate() {
                for bi in 0..b {
                    let src = bi * self.per_pos;
                    let dst = bi * self.cache_numel + cur * self.per_pos;
                    kc[l][dst..dst + self.per_pos].copy_from_slice(&dk[src..src + self.per_pos]);
                    vc[l][dst..dst + self.per_pos].copy_from_slice(&dv[src..src + self.per_pos]);
                }
            }
        }

        Ok(finalize_texts(slots, tok))
    }
}

/// conv/encoder 合批 + 逐 item prompt 构建（batch 与 gpu 两路共用）。
/// 填充策略：官方 split_audio_into_chunks 同款——mel 补真实静音帧至批内最长，
/// encoder 全真 mask，批内占位数/prompt 长一致，不存在 pad KV。
pub(crate) fn conv_encoder_prepare(
    conv: &mut Session,
    encoder: &mut Session,
    conv_in: &str,
    conv_out: &str,
    enc_in0: &str,
    enc_in1: &str,
    enc_out: &str,
    items: &[&BatchItem],
    tok: &AsrTokenizer,
) -> AsrResult<Prepared> {
    let b = items.len();

    // ---- conv 合批：帧维右填充（零帧 → 输出被 encoder mask 排除） ----
    let f_max = items.iter().map(|it| it.num_frames).max().unwrap_or(0);
    if f_max < MIN_FRAMES {
        return Ok(Prepared {
            slots: (0..b).map(|_| Slot { prompt_len: 0, done: true, generated: Vec::new() }).collect(),
            prompts: Vec::new(),
            audio_buf: Vec::new(),
            a_max: 0,
            a_hidden: 0,
            p_max: 0,
        });
    }
    let mel_dim = crate::features::MEL_DIM;
    let mut mel = vec![0f32; b * f_max * mel_dim];
    for (bi, it) in items.iter().enumerate() {
        let len = it.num_frames * mel_dim;
        mel[bi * f_max * mel_dim..bi * f_max * mel_dim + len]
            .copy_from_slice(&it.mel[..len]);
    }
    let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
    inputs.insert(
        conv_in,
        SessionInputValue::from(Tensor::from_array((
            vec![b as i64, f_max as i64, mel_dim as i64],
            mel,
        ))?),
    );
    let outputs = conv.run(inputs)?;
    let (cshape, cdata) = {
        let (s, d) = outputs[conv_out].try_extract_tensor::<f32>()?;
        (s[..].to_vec(), d.to_vec())
    };
    drop(outputs);
    if cshape.len() != 3 || cshape[1] <= 0 || cshape[2] <= 0 {
        return Err(format!("unexpected conv output shape {cshape:?}").into());
    }
    let t_conv = cshape[1] as usize;
    let c_dim = cshape[2] as usize;

    // ---- encoder 合批：静音填充帧全部为真实输入（官方 chunk 同款），全真 mask ----
    let expected = feat_to_audio_tokens_len(f_max as i32, CONV_CHUNK_SIZE).max(0) as usize;
    let valid = expected.min(t_conv);
    let enc_mask = vec![true; b * t_conv];
    let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
    inputs.insert(
        enc_in0,
        SessionInputValue::from(Tensor::from_array((
            vec![b as i64, t_conv as i64, c_dim as i64],
            cdata,
        ))?),
    );
    inputs.insert(
        enc_in1,
        SessionInputValue::from(Tensor::from_array((
            vec![b as i64, t_conv as i64],
            enc_mask,
        ))?),
    );
    let outputs = encoder.run(inputs)?;
    let (eshape, edata) = {
        let (s, d) = outputs[enc_out].try_extract_tensor::<f32>()?;
        (s[..].to_vec(), d.to_vec())
    };
    drop(outputs);
    if eshape.len() != 3 || eshape[1] as usize != t_conv || eshape[2] <= 0 {
        return Err(format!("unexpected encoder output shape {eshape:?}").into());
    }
    let a_hidden = eshape[2] as usize;

    // ---- 逐 item：统一占位数（静音填充后批内一致）与 prompt ----
    let before =
        tok.encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>")?;
    let after = tok.encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n")?;
    let mut slots: Vec<Slot> = Vec::with_capacity(b);
    let mut prompts: Vec<Vec<i64>> = Vec::with_capacity(b);
    let mut audio_rows: Vec<Vec<f32>> = Vec::with_capacity(b);
    let audio_tokens = valid;
    for bi in 0..b {
        let base = bi * t_conv * a_hidden;
        let mut ids = before.clone();
        ids.extend(std::iter::repeat_n(AUDIO_PAD_ID, audio_tokens));
        ids.extend_from_slice(&after);
        // context 超限截断音频占位
        if ids.len() > MAX_TOTAL_LEN as usize {
            let keep = MAX_TOTAL_LEN as usize - before.len() - after.len();
            if keep == 0 {
                return Err("no room for audio placeholders within max_total_len".into());
            }
            ids.truncate(before.len() + after.len());
            ids.splice(
                before.len()..before.len(),
                std::iter::repeat_n(AUDIO_PAD_ID, keep),
            );
            audio_rows.push(edata[base..base + keep * a_hidden].to_vec());
        } else {
            audio_rows.push(edata[base..base + audio_tokens * a_hidden].to_vec());
        }
        slots.push(Slot { prompt_len: ids.len(), done: false, generated: Vec::new() });
        prompts.push(ids);
    }

    let p_max = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
    let a_max = audio_rows
        .iter()
        .map(|a| if a_hidden > 0 { a.len() / a_hidden } else { 0 })
        .max()
        .unwrap_or(0);
    if p_max == 0 || a_max == 0 {
        return Ok(Prepared {
            slots: (0..b).map(|_| Slot { prompt_len: 0, done: true, generated: Vec::new() }).collect(),
            prompts: Vec::new(),
            audio_buf: Vec::new(),
            a_max: 0,
            a_hidden: 0,
            p_max: 0,
        });
    }

    let mut audio_buf = vec![0f32; b * a_max * a_hidden];
    for (bi, a) in audio_rows.iter().enumerate() {
        audio_buf[bi * a_max * a_hidden..bi * a_max * a_hidden + a.len()].copy_from_slice(a);
    }

    Ok(Prepared { slots, prompts, audio_buf, a_max, a_hidden, p_max })
}

/// 逐 item 文本还原（两路共用）
pub(crate) fn finalize_texts(slots: Vec<Slot>, tok: &AsrTokenizer) -> Vec<String> {
    slots
        .into_iter()
        .map(|s| {
            let cleaned = strip_language_prefix(&s.generated, tok);
            let text = tok.decode(&cleaned);
            AsrTokenizer::remove_replacement_chars(&text)
        })
        .collect()
}
