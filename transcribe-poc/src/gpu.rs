//! GPU 通路：KV 缓存常驻显存（io-binding 持久绑定）+ ScatterND 显存内合并。
//!
//! GPU 劣化根因（实测）：此前每个解码步都把整份 KV 缓存（56 张量 × B×2MB）
//! 从主机经 PCIe 重传上卡、delta 再回传，PCIe 流量与同步开销支配总耗时
//! （GPU 利用率仅 21%）。本模块把 KV 缓存放显存：
//! - decoder 的 56 个 KV 输入 io-bind 到显存持久张量（乒乓双缓冲，跨步零拷贝）
//! - 56 个 delta 输出 io-bind 到显存 staging
//! - scatter.onnx（28 层 ScatterND，tools/gen_scatter.py 生成）在显存内把
//!   delta 合并进缓存——onnxruntime-genai 的同款模式
//! - logits 走默认输出（每步 ~B×600KB D2H），ids/mask/pos 主机值按步重绑
//! - 未初始化显存安全：图内注意力按 cache_position 因果开窗，只读已写窗口

use std::collections::HashMap;
use std::path::Path;

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynTensor, Tensor, TensorElementType};

use crate::asr::{
    greedy_argmax, is_degenerate_repetition, session_from, trim_degenerate_tail, AsrResult,
    SessionOpts, KV_HEADS, MAX_NEW_TOKENS, MAX_TOTAL_LEN,
};
use crate::batch::{conv_encoder_prepare, finalize_texts, BatchItem};
use crate::tokenizer::{AsrTokenizer, EOS_ID};

pub struct GpuBatchAsr {
    conv: Session,
    encoder: Session,
    decoder: Session,
    scatter: Session,
    conv_in: String,
    conv_out: String,
    enc_in0: String,
    enc_in1: String,
    enc_out: String,
    dec_in: Vec<String>,
    dec_out: Vec<String>,
    /// scatter 的 56 组名称，序 i=2l+k（l=层，k=0 key/1 value），与 decoder
    /// KV 输入/输出序一一对应
    sc_cache: Vec<String>,
    sc_delta: Vec<String>,
    sc_new: Vec<String>,
    num_layers: usize,
    per_pos: usize,
    cache_numel: usize,
}

impl GpuBatchAsr {
    pub fn new(models_dir: &Path, opts: &SessionOpts) -> AsrResult<Self> {
        let conv = session_from(&models_dir.join("qwen3-conv_frontend.onnx"), opts)?;
        let encoder = session_from(&models_dir.join("qwen3-encoder.int8.onnx"), opts)?;
        let decoder = session_from(&models_dir.join("qwen3-decoder.int8.onnx"), opts)?;
        let scatter = session_from(&models_dir.join("scatter.onnx"), opts)?;

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
        let n_kv = num_layers * 2;
        let conv_in = conv.inputs()[0].name().to_string();
        let conv_out = conv.outputs()[0].name().to_string();
        let enc_in0 = encoder.inputs()[0].name().to_string();
        let enc_in1 = encoder.inputs()[1].name().to_string();
        let enc_out = encoder.outputs()[0].name().to_string();
        let dec_in: Vec<String> =
            decoder.inputs().iter().map(|o| o.name().to_string()).collect();
        let dec_out: Vec<String> =
            decoder.outputs().iter().map(|o| o.name().to_string()).collect();
        // 序 i=2l+k → dec KV 输入 [4+2l+k]、delta 输出 [1+2l+k]
        let dec_kv_in: Vec<String> = (0..n_kv)
            .map(|i| dec_in[4 + i].clone())
            .collect();
        let dec_delta_out: Vec<String> = (0..n_kv)
            .map(|i| dec_out[1 + i].clone())
            .collect();
        let sc_cache: Vec<String> = (0..num_layers)
            .flat_map(|l| [format!("cache_k_{l}"), format!("cache_v_{l}")])
            .collect();
        let sc_delta: Vec<String> = (0..num_layers)
            .flat_map(|l| [format!("delta_k_{l}"), format!("delta_v_{l}")])
            .collect();
        let sc_new: Vec<String> = (0..num_layers)
            .flat_map(|l| [format!("new_k_{l}"), format!("new_v_{l}")])
            .collect();
        // scatter IO 与 decoder 侧核对
        {
            let ins: Vec<String> =
                scatter.inputs().iter().map(|o| o.name().to_string()).collect();
            for n in sc_cache.iter().chain(sc_delta.iter()).chain(std::iter::once(&"indices".to_string())) {
                if !ins.contains(n) {
                    return Err(format!("scatter.onnx missing input {n}").into());
                }
            }
            let outs: Vec<String> =
                scatter.outputs().iter().map(|o| o.name().to_string()).collect();
            for n in &sc_new {
                if !outs.contains(n) {
                    return Err(format!("scatter.onnx missing output {n}").into());
                }
            }
        }
        Ok(Self {
            conv,
            encoder,
            decoder,
            scatter,
            conv_in,
            conv_out,
            enc_in0,
            enc_in1,
            enc_out,
            dec_in,
            dec_out,
            sc_cache,
            sc_delta,
            sc_new,
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
        let crate::batch::Prepared { mut slots, prompts, audio_buf, a_max, a_hidden, p_max } =
            prepared;
        if p_max == 0 || a_max == 0 {
            return Ok(finalize_texts(slots, tok));
        }

        // ---- prefill（默认绑定：KV 用主机零视图，delta 回主机） ----
        let mut ids_buf = vec![0i64; b * p_max];
        let mut mask_buf = vec![0i64; b * p_max];
        for (bi, p) in prompts.iter().enumerate() {
            ids_buf[bi * p_max..bi * p_max + p.len()].copy_from_slice(p);
            mask_buf[bi * p_max..bi * p_max + p.len()].fill(1);
        }
        let host_zeros = vec![0f32; b * self.cache_numel];
        let kv_shape = [b, MAX_TOTAL_LEN as usize, KV_HEADS as usize, 128];
        let n_kv = self.num_layers * 2;
        let kv_in_names: Vec<String> =
            (0..n_kv).map(|i| self.dec_in[4 + i].clone()).collect();
        let delta_out_names: Vec<String> =
            (0..n_kv).map(|i| self.dec_out[1 + i].clone()).collect();
        let mut inputs: HashMap<&str, SessionInputValue> = HashMap::new();
        inputs.insert(
            self.dec_in[0].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![b as i64, p_max as i64],
                ids_buf.clone(),
            ))?),
        );
        inputs.insert(
            self.dec_in[1].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![b as i64, a_max as i64, a_hidden as i64],
                audio_buf.clone(),
            ))?),
        );
        inputs.insert(
            self.dec_in[2].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![b as i64, p_max as i64],
                mask_buf.clone(),
            ))?),
        );
        inputs.insert(
            self.dec_in[3].as_str(),
            SessionInputValue::from(Tensor::from_array((
                vec![p_max as i64],
                (0..p_max as i64).collect::<Vec<_>>(),
            ))?),
        );
        for (i, name) in kv_in_names.iter().enumerate() {
            let view = ndarray::ArrayView4::from_shape(kv_shape, &host_zeros)
                .map_err(|e| format!("prefill kv view {i}: {e}"))?;
            inputs.insert(
                name.as_str(),
                SessionInputValue::from(ort::value::TensorRef::from_array_view(view)?),
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
        let vocab = lshape[2] as usize;
        let mut prefill_deltas: Vec<Vec<f32>> = Vec::with_capacity(n_kv);
        for (i, name) in delta_out_names.iter().enumerate() {
            let (_, d) = outputs[name.as_str()].try_extract_tensor::<f32>()?;
            prefill_deltas.push(d.to_vec());
        }
        drop(outputs);

        // 首 token 采样（impl.cc:956-1025：EOS 重采样一次，仍 EOS → 空结果）
        for (bi, slot) in slots.iter_mut().enumerate() {
            let row = &ldata[(bi * p_max + slot.prompt_len - 1) * vocab..][..vocab];
            let mut next = greedy_argmax(row, None);
            if next == EOS_ID {
                next = greedy_argmax(row, Some(EOS_ID));
                if next == EOS_ID {
                    slot.done = true;
                }
            }
            if !slot.done {
                slot.generated.push(next);
            }
            eprintln!("gpu prefill: item {bi} first_token={next} (eos={})", next == EOS_ID);
        }

        // ---- 主机侧应用 prefill delta（与已验证的 CPU 合批路径完全同款） ----
        let n_kv = self.num_layers * 2;
        let mut host_kv: Vec<Vec<f32>> = vec![vec![0f32; b * self.cache_numel]; n_kv];
        for (i, d) in prefill_deltas.iter().enumerate() {
            for bi in 0..b {
                let src = bi * p_max * self.per_pos;
                let dst = bi * self.cache_numel;
                host_kv[i][dst..dst + p_max * self.per_pos]
                    .copy_from_slice(&d[src..src + p_max * self.per_pos]);
            }
        }

        // ---- 显存资源：CUDA 分配器 + 乒乓双缓存 + delta staging ----
        let mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let alloc = Allocator::new(&self.decoder, mem_info)?;
        let shape = [b, MAX_TOTAL_LEN as usize, KV_HEADS as usize, 128];
        let mut sets: Vec<Vec<DynTensor>> = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut set = Vec::with_capacity(self.num_layers * 2);
            for _ in 0..(self.num_layers * 2) {
                set.push(DynTensor::new(&alloc, TensorElementType::Float32, shape)?);
            }
            sets.push(set);
        }
        let mut staging: Vec<DynTensor> = Vec::with_capacity(self.num_layers * 2);
        for _ in 0..(self.num_layers * 2) {
            staging.push(DynTensor::new(&alloc, TensorElementType::Float32, [b, 1, KV_HEADS as usize, 128])?);
        }

        // ---- 整份缓存上传：ScatterND 以全位置 indices 把主机缓存写入显存 ----
        // （每批仅一次 ~112MB 上行；delta 走零输入，indices 覆盖全部位置故零不落地）
        let all_pos_indices: Vec<i64> = (0..b as i64)
            .flat_map(|bi| (0..MAX_TOTAL_LEN).map(move |p| [bi, p]))
            .flat_map(|pair| pair)
            .collect();
        let zero_updates = vec![0f32; b * self.cache_numel];
        {
            let mut binding = self.scatter.create_binding()?;
            let sc_cache_names: Vec<String> = self.sc_cache.clone();
            let sc_delta_names: Vec<String> = self.sc_delta.clone();
            let sc_new_names: Vec<String> = self.sc_new.clone();
            for i in 0..n_kv {
                binding.bind_input(
                    sc_cache_names[i].as_str(),
                    &Tensor::from_array((
                        vec![b as i64, MAX_TOTAL_LEN, KV_HEADS, 128],
                        vec![0f32; b * self.cache_numel],
                    ))?,
                )?;
                binding.bind_input(
                    sc_delta_names[i].as_str(),
                    &Tensor::from_array((vec![b as i64, MAX_TOTAL_LEN, KV_HEADS, 128], host_kv[i].clone()))?,
                )?;
            }
            binding.bind_input("indices", &Tensor::from_array((
                vec![all_pos_indices.len() as i64 / 2, 2],
                all_pos_indices,
            ))?)?;
            for i in 0..n_kv {
                binding.bind_output(sc_new_names[i].as_str(), sets[0][i].clone())?;
            }
            self.scatter.run_binding(&binding)?;
            // 诊断：读回上传后的缓存，确认非全零（临时）
            let (dsz, ddata) = sets[0][n_kv - 1].try_extract_tensor::<f32>()?;
            let nz = ddata.iter().filter(|v| v.abs() > 1e-9).count();
            eprintln!("gpu diag: scatter#1 output set0 last tensor non-zero {nz}/{} shape{:?}", ddata.len(), &dsz[..4]);
        }
        let _ = zero_updates;

        // ---- lockstep decode：decoder 全绑定（KV=显存，logits 默认回主机）， ----
        // ---- scatter 显存内合并 delta（乒乓 sets[src]→sets[dst]） ----
        let mut src = 0usize;
        // io-binding 运行只产出已绑定的输出：logits 绑到主机预分配张量，
        // ORT 每步隐式 D2H 写入（Value clone 共享底层缓冲）
        let mut logits_host = Tensor::from_array((
            vec![b as i64, 1, vocab as i64],
            vec![0f32; b * vocab],
        ))?;
        let n_kv = self.num_layers * 2;
        let kv_in_names: Vec<String> =
            (0..n_kv).map(|i| self.dec_in[4 + i].clone()).collect();
        let delta_out_names: Vec<String> =
            (0..n_kv).map(|i| self.dec_out[1 + i].clone()).collect();
        let sc_cache_names: Vec<String> = self.sc_cache.clone();
        let sc_delta_names: Vec<String> = self.sc_delta.clone();
        let sc_new_names: Vec<String> = self.sc_new.clone();
        for step in 1..MAX_NEW_TOKENS {
            let cur = p_max + step - 1;
            if cur >= MAX_TOTAL_LEN as usize || slots.iter().all(|s| s.done) {
                break;
            }
            let dst = 1 - src;
            let step_ids: Vec<i64> = slots
                .iter()
                .map(|s| if s.done { EOS_ID } else { *s.generated.last().unwrap_or(&EOS_ID) })
                .collect();
            let mut binding = self.decoder.create_binding()?;
            for i in 0..n_kv {
                binding.bind_input(kv_in_names[i].as_str(), &sets[src][i])?;
                binding.bind_output(delta_out_names[i].as_str(), staging[i].clone())?;
            }
            binding.bind_output(self.dec_out[0].as_str(), logits_host.clone())?;
            binding.bind_input(
                self.dec_in[0].as_str(),
                &Tensor::from_array((vec![b as i64, 1], step_ids))?,
            )?;
            binding.bind_input(
                self.dec_in[1].as_str(),
                &Tensor::from_array((
                    vec![b as i64, a_max as i64, a_hidden as i64],
                    audio_buf.clone(),
                ))?,
            )?;
            // 窄 mask [B,1]：GPU 实测唯一正确形态（CUDA EP 的 int8 注意力核对
            // 解码步 mask 宽度>1 处理异常——全宽 mask 在 GPU 上即时塌缩输出 EOS，
            // 同代码同数据在 CPU 上正确）
            binding.bind_input(
                self.dec_in[2].as_str(),
                &Tensor::from_array((vec![b as i64, 1], vec![1i64; b]))?,
            )?;
            binding.bind_input(
                self.dec_in[3].as_str(),
                &Tensor::from_array((vec![1], vec![cur as i64]))?,
            )?;
            self.decoder.run_binding(&binding)?;
            let (lshape, ldata) = {
                let (s, d) = logits_host.try_extract_tensor::<f32>()?;
                (s[..].to_vec(), d.to_vec())
            };
            if lshape.len() != 3 || lshape[0] as usize != b || lshape[2] <= 0 {
                return Err(format!("unexpected logits shape {lshape:?}").into());
            }
            for (bi, slot) in slots.iter_mut().enumerate() {
                if slot.done {
                    continue;
                }
                let row = &ldata[bi * vocab..(bi + 1) * vocab];
                let next = greedy_argmax(row, None);
                if step <= 3 {
                    eprintln!("gpu decode: step {step} item {bi} next={next} (eos={})", next == EOS_ID);
                }
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
            // 显存内合并：cache[src] + delta@cur → cache[dst]
            let indices: Vec<i64> = (0..b as i64).flat_map(|bi| [bi, cur as i64]).collect();
            let mut sbinding = self.scatter.create_binding()?;
            for i in 0..n_kv {
                sbinding.bind_input(sc_cache_names[i].as_str(), &sets[src][i])?;
                sbinding.bind_input(sc_delta_names[i].as_str(), &staging[i])?;
            }
            sbinding.bind_input("indices", &Tensor::from_array((
                vec![b as i64, 2],
                indices,
            ))?)?;
            for i in 0..n_kv {
                sbinding.bind_output(sc_new_names[i].as_str(), sets[dst][i].clone())?;
            }
            self.scatter.run_binding(&sbinding)?;
            src = dst;
        }

        Ok(finalize_texts(slots, tok))
    }
}
