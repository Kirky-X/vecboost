// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 流式 ASR 核心（sherpa-onnx streaming zipformer 中英双语 int8 三图）。
//!
//! I/O 契约来自 ort 实图内省（`src/bin/inspect_streaming_io.rs` 实测，非猜测）：
//! - encoder: 36 输入（`x` [1,39,80] + 35 个 `cached_*`）、36 输出
//!   （`encoder_out` [1,frames,512] + 35 个 `new_cached_*`）
//! - decoder: `y` [1,2] → `decoder_out` [1,512]
//! - joiner: `encoder_out` + `decoder_out` → `logit` [1,6254]
//!
//! 与 Qwen3/Whisper 整段解码后端并存、互不干扰：本模块只由 transcribe-stream
//! feature 的 WS 端点消费，不进现有 EngineFactory / TranscribeSession 分派链。

use std::collections::HashMap;
use std::path::Path;

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::config::model::DeviceType;
use crate::engine::session::{SessionOpts, build_session, probe_model_file};
use crate::error::VecboostError;

use super::features::WhisperFbank;

/// 模型采样率与帧参数（sherpa streaming zipformer 与 whisper fbank 同口径）。
pub const SAMPLE_RATE: usize = 16_000;
/// encoder `x` 输入帧数 = metadata T=39（ort 实图固定维度，非 chunk_len=32）。
/// 每 chunk 喂 decode_chunk_len=32 帧新特征，encoder 内部与 left context 拼成 39。
const WINDOW_FRAMES: usize = 39;
/// 每次送 encoder 的新特征帧数（metadata decode_chunk_len）。
const CHUNK_FRAMES: usize = 32;
const MEL_DIM: usize = 80;
const ENCODER_DIM: usize = 512;
const CONTEXT_SIZE: usize = 2;
const VOCAB_SIZE: usize = 6254;
const BLANK_ID: usize = 0;
/// sherpa greedy decoder 初始 context（online-transducer-greedy-search-decoder.cc:61-65）：
/// `tokens.resize(context_size,-1); tokens.back()=blank_id(0)` → [-1, 0]。
const INIT_CONTEXT_0: i64 = -1;
const INIT_CONTEXT_1: i64 = 0; // blank id
/// blank penalty（greedy decoder 从 logits[0] 减去以抑制 blank）
const BLANK_PENALTY: f32 = 0.0;

/// 流式三图文件名（单一事实源，探测与加载共用）。
const ENCODER_CANDIDATES: &[&str] = &[
    "encoder-epoch-99-avg-1.int8.onnx",
    "encoder_model_int8.onnx",
];
const DECODER_CANDIDATES: &[&str] = &["decoder-epoch-99-avg-1.int8.onnx", "decoder.int8.onnx"];
const JOINER_CANDIDATES: &[&str] = &["joiner-epoch-99-avg-1.int8.onnx"];

type StreamResult<T> = Result<T, VecboostError>;

fn load_err(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::ModelLoadError(format!("streaming asr load: {e}"))
}
fn infer_err(e: impl std::fmt::Display) -> VecboostError {
    VecboostError::InferenceError(format!("streaming asr: {e}"))
}

/// 从 tokens.txt 读 BPE 词表（`<blk> 0` / `<sos/eos> 1` / `<unk> 2` ...）。
fn load_tokens(models_dir: &Path) -> StreamResult<Vec<String>> {
    let path = models_dir.join("tokens.txt");
    let text =
        std::fs::read_to_string(&path).map_err(|e| load_err(format!("{}: {e}", path.display())))?;
    let mut entries: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let token = parts.next().unwrap_or("");
        let id: usize = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        entries.push((id, token.to_string()));
    }
    entries.sort_by_key(|(id, _)| *id);
    // tokens.txt 条目数可大于 joiner logits 维度（前 3 个特殊 token
    // <blk>/<sos/eos>/<unk> 不占 logit 位；实测本模型 6257 tokens vs 6254 logits）。
    // 只校验「token id 连续且覆盖 logits 维度」，不强行相等。
    if entries.len() < VOCAB_SIZE || entries.first().map(|(id, _)| *id) != Some(0) {
        return Err(load_err(format!(
            "tokens.txt has {} entries (min {VOCAB_SIZE}, first id {:?})",
            entries.len(),
            entries.first().map(|(id, _)| *id)
        )));
    }
    for (idx, (id, _)) in entries.iter().enumerate() {
        if *id != idx {
            return Err(load_err(format!(
                "tokens.txt id {id} at position {idx} is not contiguous"
            )));
        }
    }
    Ok(entries.into_iter().map(|(_, t)| t).collect())
}

/// 三图Session + fbank + 词表；每实例独占一套 Session（并发流各自持实例）。
pub struct StreamingAsr {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    fbank: WhisperFbank,
    tokens: Vec<String>,
}

/// 单条流的推理状态（与 Session 分离，便于 mock 测试与并发隔离）。
#[derive(Debug, Clone)]
pub struct StreamState {
    /// 尚未凑满一个 chunk 的 PCM 样本（16kHz mono）
    pcm_pending: Vec<f32>,
    /// decoder 上文 token（初始 [sos, sos]），每步右移一位
    context: Vec<i64>,
    /// 已输出的 token id（去 blank / 去重复）
    emitted: Vec<i64>,
    /// 每步的非 blank 次优 token（供纠错层 N-best 候选，空白表示无次优）
    alternatives: Vec<Option<i64>>,
    /// 编码器 cache（35 项，首帧全零、后续由 new_cached_* 回填）
    caches: HashMap<String, Vec<f32>>,
    /// 缓存长度（5 层；INT64 [n,1]），首帧 0
    cached_lens: HashMap<String, Vec<i64>>,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            pcm_pending: Vec::new(),
            context: vec![INIT_CONTEXT_0, INIT_CONTEXT_1],
            emitted: Vec::new(),
            alternatives: Vec::new(),
            caches: HashMap::new(),
            cached_lens: HashMap::new(),
        }
    }
}

/// 状态推进纯逻辑（mock 测试用，不依赖 ort）：给定 joiner logits 更新
/// context/emitted，返回新增文本 token id。blank(0) 与重复 token 丢弃。
pub fn advance_decode(
    state: &mut StreamState,
    logits: &[f32],
) -> Result<Option<i64>, VecboostError> {
    if logits.len() != VOCAB_SIZE {
        return Err(infer_err(format!(
            "logits len {} != {VOCAB_SIZE}",
            logits.len()
        )));
    }
    // blank penalty（sherpa greedy decoder 同款）：logits[0] -= penalty 再 argmax。
    // penalty=0 时直接在原切片 argmax，免 6254 元素 clone（性能审查 H4）。
    let best = if BLANK_PENALTY > 0.0 {
        let mut scored = logits.to_vec();
        scored[BLANK_ID] -= BLANK_PENALTY;
        scored
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i64)
    } else {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i64)
    }
    .unwrap_or(0);
    if best == BLANK_ID as i64 {
        return Ok(None);
    }
    // 次优非 blank token（N-best 候选源）：score 与 top-1 差距小于阈值时保留
    let runner_up = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != BLANK_ID && *i as i64 != best)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i64);
    if state.emitted.last() == Some(&best) {
        return Ok(None); // 重复 token：不 push emitted，也不 push alternatives（保持索引对齐）
    }
    state.emitted.push(best);
    state.alternatives.push(runner_up); // 与 emitted 同步入栈（架构复审 H1-3）
    // 右移：先存旧 context[1]，再写入 context[1]=best，最后 context[0]=旧值。
    // 若先写 context[1] 再读它，context[0] 会拿到新 token（实测 bug，已修）。
    let prev = state.context[1];
    state.context[CONTEXT_SIZE - 1] = best;
    state.context[0] = prev;
    Ok(Some(best))
}

/// 把 token id 序列解码为文本（BPE 词片拼接 + 恢复空格）。
pub fn tokens_to_text(tokens: &[String], ids: &[i64]) -> String {
    let mut out = String::new();
    for &id in ids {
        let id = id as usize;
        if id <= 2 || id >= tokens.len() {
            continue;
        }
        let piece = &tokens[id];
        if let Some(rest) = piece.strip_prefix('▁') {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(rest);
        } else {
            out.push_str(piece);
        }
    }
    out.trim().to_string()
}

impl StreamingAsr {
    /// 加载三图；图缺失/IO 不符显性 ModelLoadError（禁静默降级）。
    pub fn load(models_dir: &Path, intra_threads: usize) -> StreamResult<Self> {
        let enc_path = probe_model_file(models_dir, ENCODER_CANDIDATES, "streaming encoder")?;
        let dec_path = probe_model_file(models_dir, DECODER_CANDIDATES, "streaming decoder")?;
        let join_path = probe_model_file(models_dir, JOINER_CANDIDATES, "streaming joiner")?;
        let opts = SessionOpts {
            intra_threads,
            device: DeviceType::Cpu,
            extra_config: &[],
        };
        let encoder = build_session(&enc_path, &opts)?;
        let decoder = build_session(&dec_path, &opts)?;
        let joiner = build_session(&join_path, &opts)?;
        // dtype 守卫：三图 I/O 必须 f32/i64（int8 权重图的 I/O 仍是 f32）
        for (sess, role) in [
            (&encoder, "encoder"),
            (&decoder, "decoder"),
            (&joiner, "joiner"),
        ] {
            for io in sess.inputs().iter().chain(sess.outputs().iter()) {
                if let ValueType::Tensor { ty, .. } = io.dtype()
                    && !matches!(ty, TensorElementType::Float32 | TensorElementType::Int64)
                {
                    return Err(load_err(format!(
                        "{role} '{}' dtype {ty:?} unsupported (f32/i64 only)",
                        io.name()
                    )));
                }
            }
        }
        let tokens = load_tokens(models_dir)?;
        Ok(Self {
            encoder,
            decoder,
            joiner,
            fbank: WhisperFbank::new_kaldi(MEL_DIM),
            tokens,
        })
    }

    /// 送一帧 PCM（20ms = 320 样本）进编码器，返回 joiner logits。
    /// 状态推进：首帧 cache 全零（由 ort 图默认值/显式零张量提供），
    /// 后续帧的 new_cached_* 回填到下一次输入。
    pub fn encode_frame(&mut self, state: &mut StreamState, pcm: &[f32]) -> StreamResult<Vec<f32>> {
        // 累积 PCM 至一个 decoder chunk（32 帧 × 160 样本 = 5120 样本）再推理。
        // sherpa recognizer 按 decode_chunk_len=32 从特征流取帧，T=39 是 encoder
        // 内部含 left context 的窗口——先前误把 39 当 x 帧数是根因（logits 全 blank）。
        let state = &mut *state;
        state.pcm_pending.extend_from_slice(pcm);
        const CHUNK_SAMPLES: usize = CHUNK_FRAMES * 160; // FRAME_SHIFT=160
        if state.pcm_pending.len() < CHUNK_SAMPLES {
            return Ok(Vec::new()); // 不足一帧，等待下一次喂入
        }
        let chunk: Vec<f32> = state.pcm_pending.drain(..CHUNK_SAMPLES).collect();
        let mel = self.fbank.compute(&chunk);
        let want = WINDOW_FRAMES * MEL_DIM;
        let mut x = vec![0.0f32; want];
        let take = mel.len().min(want);
        x[..take].copy_from_slice(&mel[..take]);

        let mut inputs = std::collections::HashMap::<String, SessionInputValue>::new();
        inputs.insert(
            "x".to_string(),
            SessionInputValue::from(
                Tensor::from_array(([1usize, WINDOW_FRAMES, MEL_DIM], x)).map_err(infer_err)?,
            ),
        );
        // cache 输入：首帧零，后续由 state.caches 回填（维度从实图 new_cached_* 取）
        for (name, io) in self
            .encoder
            .inputs()
            .iter()
            .map(|io| (io.name().to_string(), io))
        {
            if name == "x" {
                continue;
            }
            if let ValueType::Tensor { shape, .. } = io.dtype() {
                // 动态维 -1 首帧填 1（诊断探针 MODE2 实测唯一跑通；填 0 触发
                // MatMul 广播失败，填 left_context 触发 Add 维度不匹配）。
                let dims: Vec<usize> = shape
                    .iter()
                    .map(|d| if *d < 0 { 1 } else { *d as usize })
                    .collect();
                let n = dims.iter().product::<usize>();
                let is_i64 = matches!(
                    io.dtype(),
                    ValueType::Tensor {
                        ty: TensorElementType::Int64,
                        ..
                    }
                );
                let value: SessionInputValue = if let Some(cached) = state.caches.get(&name) {
                    SessionInputValue::from(
                        Tensor::from_array((dims.clone(), cached.clone())).map_err(infer_err)?,
                    )
                } else if let Some(cached) = state.cached_lens.get(&name) {
                    SessionInputValue::from(
                        Tensor::from_array((dims.clone(), cached.clone())).map_err(infer_err)?,
                    )
                } else if is_i64 {
                    SessionInputValue::from(
                        Tensor::from_array((dims.clone(), vec![0i64; n])).map_err(infer_err)?,
                    )
                } else {
                    SessionInputValue::from(
                        Tensor::from_array((dims.clone(), vec![0.0f32; n])).map_err(infer_err)?,
                    )
                };
                inputs.insert(name.clone(), value);
            }
        }
        let outputs = self.encoder.run(inputs).map_err(infer_err)?;
        // 回填 cache：new_cached_* → cached_*（去 new_ 前缀）。
        // 按 dtype 分别处理：cached_len_* 是 INT64（旧实现只取 f32 导致静默跳过，
        // 每帧都当无历史 → encoder 无法累积上下文，logits 恒 blank）。禁静默失败。
        for (name, value) in outputs.iter() {
            if let Some(rest) = name.strip_prefix("new_") {
                if let Ok((_shape, data)) = value.try_extract_tensor::<f32>() {
                    state.caches.insert(rest.to_string(), data.to_vec());
                } else if let Ok((_shape, data)) = value.try_extract_tensor::<i64>() {
                    state.cached_lens.insert(rest.to_string(), data.to_vec());
                } else {
                    return Err(infer_err(format!(
                        "cache output {name} is neither f32 nor i64"
                    )));
                }
            }
        }
        // encoder_out → joiner（取最后一帧）；独立作用域释放 outputs 借用，
        // 否则 &self（outputs 借用）与下方 &mut self（decode_and_join）冲突。
        let last = {
            let (shape, edata) = outputs["encoder_out"]
                .try_extract_tensor::<f32>()
                .map_err(infer_err)?;
            let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
            let frames = dims.get(1).copied().unwrap_or(1);
            edata[(frames - 1) * ENCODER_DIM..frames * ENCODER_DIM].to_vec()
        };
        // outputs 借用 &mut self.encoder，必须在借用 &mut self 调 decode_and_join 前 drop
        drop(outputs);
        self.decode_and_join(state, &last)
    }

    /// decoder + joiner：上下文 token → decoder_out，与 encoder_out 相加后取 logits。
    fn decode_and_join(&mut self, state: &mut StreamState, enc: &[f32]) -> StreamResult<Vec<f32>> {
        let y = state.context.clone();
        let mut dinputs = std::collections::HashMap::<String, SessionInputValue>::new();
        dinputs.insert(
            "y".to_string(),
            SessionInputValue::from(
                Tensor::from_array(([1usize, CONTEXT_SIZE], y)).map_err(infer_err)?,
            ),
        );
        let dout = self.decoder.run(dinputs).map_err(infer_err)?;
        // ddata 转 owned：dout 的借用不能跨到下方 self.joiner.run(&mut self)
        let ddata: Vec<f32> = {
            let (_, d) = dout["decoder_out"]
                .try_extract_tensor::<f32>()
                .map_err(infer_err)?;
            d.to_vec()
        };
        // joiner 两输入 = encoder 输出 + decoder 输出（joiner 内部做融合；
        // 不要在外面预加 merged——那会让 joiner 收到错误的双重编码特征）。
        let mut jinputs = std::collections::HashMap::<String, SessionInputValue>::new();
        jinputs.insert(
            "encoder_out".to_string(),
            SessionInputValue::from(
                Tensor::from_array(([1usize, ENCODER_DIM], enc.to_vec())).map_err(infer_err)?,
            ),
        );
        jinputs.insert(
            "decoder_out".to_string(),
            SessionInputValue::from(
                Tensor::from_array(([1usize, ENCODER_DIM], ddata.clone())).map_err(infer_err)?,
            ),
        );
        let jout = self.joiner.run(jinputs).map_err(infer_err)?;
        let (_, logits) = jout["logit"]
            .try_extract_tensor::<f32>()
            .map_err(infer_err)?;
        Ok(logits.to_vec())
    }

    /// 当前 partial 文本（state.emitted 解码）。
    pub fn partial_text(&self, state: &StreamState) -> String {
        tokens_to_text(&self.tokens, &state.emitted)
    }

    /// 流结束：返回已解码文本。
    pub fn finish(&self, state: &StreamState) -> String {
        tokens_to_text(&self.tokens, &state.emitted)
    }

    /// 构造 N-best 候选文本（top-1 + 每步次优替换），供纠错层重排。
    pub fn finish_candidates(&self, state: &StreamState) -> Vec<String> {
        let primary = tokens_to_text(&self.tokens, &state.emitted);
        let mut out = vec![primary.clone()];
        if !state.alternatives.is_empty() {
            // 次优整体替换视角：把每个 emitted token 用对应次优替换，生成 1 条强候选
            let mut alt_ids = state.emitted.clone();
            let mut changed = false;
            for (idx, alt) in state.alternatives.iter().enumerate() {
                if idx < alt_ids.len()
                    && let Some(t) = alt
                {
                    // 只在次优与当前不同的位置替换（取第一个差异构造候选，避免全替换失真）
                    if !changed && *t != alt_ids[idx] {
                        alt_ids[idx] = *t;
                        changed = true;
                    }
                }
            }
            if changed {
                let alt_text = tokens_to_text(&self.tokens, &alt_ids);
                if alt_text != primary {
                    out.push(alt_text);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_decode_ignores_blank_and_repeats() {
        let mut st = StreamState::default();
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[0] = 1.0; // blank
        assert_eq!(advance_decode(&mut st, &logits).unwrap(), None);
        logits[0] = 0.0;
        logits[10] = 1.0;
        assert_eq!(advance_decode(&mut st, &logits).unwrap(), Some(10));
        assert_eq!(advance_decode(&mut st, &logits).unwrap(), None); // repeat
        logits[10] = 0.0;
        logits[11] = 1.0;
        assert_eq!(advance_decode(&mut st, &logits).unwrap(), Some(11));
        assert_eq!(st.emitted, vec![10, 11]);
    }

    #[test]
    fn context_slides_by_one_token() {
        let mut st = StreamState::default();
        assert_eq!(st.context, vec![-1, 0]);
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        logits[5] = 1.0;
        advance_decode(&mut st, &logits).unwrap();
        assert_eq!(st.context, vec![0, 5]);
        logits[5] = 0.0;
        logits[6] = 1.0;
        advance_decode(&mut st, &logits).unwrap();
        assert_eq!(st.context, vec![5, 6]);
    }

    #[test]
    fn tokens_to_text_restores_word_boundaries() {
        let mut tokens: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("t{i}")).collect();
        tokens[3] = "▁hello".to_string();
        tokens[4] = "▁world".to_string();
        tokens[5] = "!".to_string();
        // BPE 规则：▁ 标记词首空格；不带 ▁ 的词片粘在前者后。
        assert_eq!(tokens_to_text(&tokens, &[3, 4]), "hello world");
        assert_eq!(tokens_to_text(&tokens, &[3, 5]), "hello!");
    }

    /// 真实三图一帧 smoke：验证 I/O 契约喂法正确（需 ORT_DYLIB_PATH + 模型资产）。
    /// 用 #[ignore] 隔离，避免 CI 无资产时失败；手动 `cargo test -- --ignored` 跑。
    #[test]
    #[ignore = "requires models/whisper-stream int8 assets + ORT_DYLIB_PATH"]
    fn real_model_single_frame() {
        let dir = std::path::Path::new("models/whisper-stream");
        if !dir.join("tokens.txt").exists() {
            eprintln!("skip: no model assets");
            return;
        }
        let mut asr = StreamingAsr::load(dir, 1).expect("load");
        let mut state = StreamState::default();
        // 一个 chunk = 32 帧 × 160 样本 = 5120 样本；不足时 encode_frame 返回空
        let mut logits = Vec::new();
        for _ in 0..16 {
            logits = asr
                .encode_frame(&mut state, &vec![0.0f32; 320])
                .expect("frame");
        }
        assert_eq!(logits.len(), VOCAB_SIZE);
        println!("logit[0]={} sum={}", logits[0], logits.iter().sum::<f32>());
    }

    #[test]
    fn alternatives_stay_aligned_with_emitted() {
        let mut st = StreamState::default();
        let mut l = vec![0.0f32; VOCAB_SIZE];
        l[5] = 1.0;
        l[6] = 0.5;
        assert_eq!(advance_decode(&mut st, &l).unwrap(), Some(5));
        assert_eq!(st.emitted.len(), 1);
        assert_eq!(st.alternatives.len(), 1);
        // 重复 token 不应改变两向量长度（索引对齐）
        assert_eq!(advance_decode(&mut st, &l).unwrap(), None);
        assert_eq!(st.emitted.len(), 1);
        assert_eq!(st.alternatives.len(), 1);
    }

    #[test]
    fn wrong_logits_len_rejected() {
        let mut st = StreamState::default();
        assert!(advance_decode(&mut st, &[0.0; 10]).is_err());
    }
}
