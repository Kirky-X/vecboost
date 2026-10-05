# 自研转录引擎 PoC 设计说明（transcribe-own）

日期：2026-10-05。作者：准备工程师（环境与设计契约阶段）。
约束：**完全自研**——Cargo 依赖只允许 ort/tokenizers/hound/rustfft，禁止引入 sherpa-onnx；
sherpa-onnx（Apache-2.0）与 kaldi-native-fbank（Apache-2.0）源码仅作移植参考，
已存 `docs-transcribe-reference/`，出处逐文件注明（见 §8）。

所有形状/名称均经两路验证：①参考源码行号；②对 worktree 内实际 onnx 文件的侦察
（命令与输出见 §2-§5，侦察用 onnx 1.23.1，装于 /tmp/e2e-poc/qwen-env2 venv）。

## 1. 资产与基线

worktree：`/home/dev/projects/vecboost-transcribe`（分支 `feat/transcribe-own-ort`，基线 main@bdf8b80）。

| 资产 | 字节 | 来源与校验 |
| --- | --- | --- |
| assets/audio_from_m4a.wav | 22392910 | /tmp/e2e-poc 复制；md5 520fabe9b55afa08146677f92d11faf0 与源一致；16kHz/mono/s16，**699.776s**（python wave 实测） |
| assets/smoke10s.wav | 320078 | 同上；10.0s |
| assets/models/qwen3-conv_frontend.onnx | 44148281 | 同上 |
| assets/models/qwen3-encoder.int8.onnx | 182491662 | 同上 |
| assets/models/qwen3-decoder.int8.onnx | 755914231 | 同上；md5 55a36f9f3bc1c36de8153b00972176dd 与源一致 |
| assets/models/silero_vad.onnx | 643854 | 同上 |
| assets/models/tokenizer/{vocab.json,merges.txt,tokenizer_config.json} | 2776833/1671853/12487 | sherpa 导出的 tokenizer 三件套 |
| assets/reference.txt | 12600 | 参照逐字稿（与基线 CER 同源） |
| scripts/cer.py | 1604 | `python3 cer.py <ref> <hyp>` → 输出 CER 与 S/I/D 明细 |
| 3rdparty/libonnxruntime.so.1.28.2 | 24277040 | 自主仓只读复制（chmod 0444），md5 7294310fc3d98bfdc98246c2c53c4cae；git 按主仓先例不入库 |

既有实测基线（同一音频、同一参照稿，来源：主仓 out/transcribe-poc/e2e/summary.md 末段 + 任务说明）：
sherpa-onnx 0.6B-int8 CPU 单线程 wall **256.4s**（RTF 0.366，CER **7.82%**，峰值内存 **2721MB**）；
sherpa 同模型 GPU RTF 0.524（CER 7.57%）；官方 qwen_asr 1.7B bf16 GPU RTF 0.101（CER 2.79%，显存 7.51GB）；
官方 0.6B bf16 GPU RTF 0.080（长音频 CER **17.35%**——无 VAD 内部切块导致质量崩塌）。
**对拍目标**：自研管线 CER 应落在 sherpa 同款 VAD+int8 管线量级（≈7.6-7.8%），wall 显著优于 256.4s。

## 2. conv_frontend.onnx（侦察输出）

```
opset 17
INPUT  input_features : FLOAT [batch, n_frames, 128]
OUTPUT conv_output    : FLOAT [batch, n_audio_tokens, C]   # C 动态
```

## 3. encoder.int8.onnx（侦察输出）

```
opset 17
INPUT  input_features        : FLOAT [batch, n_audio_tokens, 896]
INPUT  feature_attention_mask: BOOL  [batch, n_audio_tokens]
OUTPUT audio_features        : FLOAT [batch, n_audio_tokens, 1024]
```

## 4. decoder.int8.onnx（侦察输出，60 入 / 57 出）

```
opset 17
IN  input_ids     : INT64 [batch, seq]
IN  audio_features: FLOAT [batch, n_audio_tokens, 1024]
IN  attention_mask: INT64 [batch, seq]
IN  cache_position: INT64 [seq]
IN  cache_key_i / cache_value_i : FLOAT [batch, max_total_len, 8, 128]   # i=0..27，共 28 层
OUT logits        : FLOAT [batch, seq, 151936]
OUT key_delta_i / value_delta_i : FLOAT [batch, seq, 8, 128]             # i=0..27
```

- 与 sherpa 解析一致：decoder 输入序 [0]=ids,[1]=audio,[2]=mask,[3]=cache_position,
  [4..4+2L]=past_kv（docs-transcribe-reference/offline-qwen3-asr-model.cc:38-39）；
  输出 [0]=logits，[1..1+2L]=present kv，L=(outputs-1)/2（同文件 :293-300）。
- **max_total_len 是符号维度（dim_param）**，运行期无静态值；sherpa 取不到时回退配置默认
  **512**（offline-qwen3-asr-model.cc:311-324 + offline-qwen3-asr-model-config.h:31
  `max_total_len = 512`）。PoC 采用 512。
- KV 缓存为**预分配全零静态缓存**，每步把 delta 按 `cache_position[0]` 原地写入
  （CreateEmptyKVCache / ApplyKvDeltaInplace，offline-qwen3-asr-model.cc:533-581, 583-749）。
  KV 元素类型 FLOAT（本 int8 导出 KV 与 logits 均为 f32）。
- 每段内存量级（命令产出）：单段 KV = 28×2×[1,512,8,128]×4B = **117,440,512 B ≈ 112.0 MiB**；
  4/6/8 段并发 ≈ 448/672/896 MiB；prefill logits L=250 时 ≈ 145 MiB（临时）。
- 词表 151936 = base 151643（vocab.json 实测条数，id 0..151642）+ added tokens（至 id 151935）。

## 5. silero_vad.onnx（k2-fsa 导出 v4 型，侦察输出）

```
opset 13
IN  x    : FLOAT [1, 512]        # 每窗 512 样本（32ms @16kHz），窗移=窗长（无重叠）
IN  h    : FLOAT [2, 1, 64]      # LSTM 状态，零初始化
IN  c    : FLOAT [2, 1, 64]
OUT prob : FLOAT [N, 1]
OUT new_h: FLOAT [2, 1, 64]
OUT new_c: FLOAT [2, 1, 64]
```

- 判型逻辑（3 入 2 出 → v5；4 入 3 出或 k2fsa 命名 x/h/c→prob/new_h/new_c → v4，
  docs-transcribe-reference/silero-vad-model.cc:208-231, 277-287）。本模型走 k2fsa 分支。
- 状态机默认值（silero-vad-model-config.h:20-41）：threshold=0.5，min_silence=0.5s，
  min_speech=0.25s，window_size=512，max_speech_duration=20s，neg_threshold=-1
  （运行时取 max(threshold-0.15, 0.01)，silero-vad-model.cc:138-142）。
- 状态机逻辑照搬 silero-vad-model.cc:97-173（含 min_speech/min_silence 判定与
  temp_start/temp_end 处理）。
- 分段（voice-activity-detector.cc:46-135, 203-231）：语音开始回溯 2×窗长+min_speech 样本；
  语音结束截去 min_silence 样本；缓存超过 max_speech_duration(=320000 样本) 时收紧为
  min_silence=0.1s、threshold=0.90 强制切分；流结束 Flush() 收尾段。

## 6. 特征链（Whisper 风格 log-mel，128 维）

参考：features.cc:332-343 / offline-stream.cc:72-95（WhisperTag+对齐中心）、
knf-whisper-feature.cc:126-145（帧参+mel 参数）、knf-feature-window.cc:34-38（周期 hann）、
offline-recognizer-qwen3-asr-impl.cc:39-47（kQwen3ChunkSize=100, kQwen3MelDim=128）、
math.cc:115-132（归一化）。官方 preprocessor_config.json 佐证：n_fft=400, hop_length=160,
feature_size=128, dither=0。

- 采样率 16000；样本取 float [-1,1]（normalize_samples=true，i16/32768）
- 帧移 10ms（160 样本），帧长 25ms（400 样本），n_fft=400（round_to_power_of_two=false）
- **周期 hann 窗**：w[i] = 0.5 - 0.5·cos(2πi/400)
- dither=0，preemph=0，remove_dc_offset=false，snip_edges=false
- 功率谱：|FFT|²，取 201 个 bin
- mel：128 bins，low=0，high=8kHz（Nyquist），librosa 式三角（bin 数 201+1、中心/左右斜坡，
  knf-mel-computations.cc:299-375）。mel 尺度与归一化：knf-whisper-feature.cc:139-144 只设
  num_bins/low_freq/is_librosa=true，`norm="slaney"` 与 `use_slaney_mel_scale=true` 保持
  mel-computations.h:58-70 默认值 → 实际为 **Slaney 尺度 + Slaney 归一化**（与 librosa.filters.mel
  默认一致；本文档此前写「HTK 公式、无 slaney 归一化」与源码不符，2026-10-05 实现阶段核对
  后修正），输出**线性**能量（whisper 通路不取 log，knf-feature-fbank.cc:98-102 仅 fbank 通路取 log；whisper-feature.cc:147-167 无 log）
- **居中对齐**：AcceptWaveform 前先补 80 样本（16000×0.01/2）静音，帧数取 floor(N/160)，
  丢掉尾部多出的 1 帧（offline-stream.cc:84-94 注释；sherpa#3535：不做会对齐会让 1.7B 塌缩成复读）
- **归一化**（math.cc:115-132，全局一次）：
  `x = log10(max(x, 1e-10)); x = max(x, 全局max-8.0); x = (x+4.0)/4.0`
- conv 输入 [1, n_frames, 128]；audio_token_len = FeatToAudioTokensLen(n_frames, 100)
  （offline-recognizer-qwen3-asr-impl.cc:77-108）：每 100 帧整块 → 13 token；
  余数 r → ceil3(ceil3(ceil3(r/2)))（(x-1)/2+1 三连）。feature_attention_mask 前
  min(expected, conv_frames) 个为 true。
- TrimAudioFeatures（同文件 :129-207）：从尾向前丢 |值|≤1e-6 的全静音帧；全静音则直接空结果。

## 7. Prompt 模板与解码循环

- 特殊 token id（tokenizer_config.json added_tokens_decoder 实测）：
  `<|endoftext|>`=151643，`<|im_start|>`=151644，`<|im_end|>`=151645（=EOS），
  `<|audio_start|>`=151669，`<|audio_end|>`=151670，`<|audio_pad|>`=151676（**单 token**），
  `<asr_text>`=151704。
- prompt 构造（offline-recognizer-qwen3-asr-impl.cc:439-507，音频占位注入方式）：

```
Encode("<|im_start|>system\n" + hotwords(空格连接) + "<|im_end|>\n<|im_start|>user\n<|audio_start|>")   # before
+ audio_token_len × Encode("<|audio_pad|>")                                                            # 占位（151676 × N）
+ Encode("<|audio_end|><|im_end|>\n" + "<|im_start|>assistant\n")                                       # after
# 若指定 language：after 再加 Encode("language " + lang) + id(<asr_text>)；PoC 不传 language
```

  Encode 对特殊 token 整体查表（qwen-asr-tokenizer.cc:1202-1244）；Decode 跳过
  `<|im_start|>`/`<|im_end|>`（:780-782），其余特殊 token 原样输出文本。
- 解码（同文件 :765-1147）：
  1. prefill：input_ids=[1,L]，attention_mask 全 1，cache_position=iota(0..L)，
     audio_features 全量传入（图内按占位拼接）；取 logits 最后一位采样；
  2. 温度 1e-6（config.h:37）→ 退化为 **greedy argmax**（跳过非有限值）；top_p=0.8、seed=42 不参与；
  3. 逐步：单 token 输入，cache_position=[cur_len]，每步 ApplyKvDeltaInplace；
  4. 首 token 为 EOS 则去掉 EOS 重采样一次，仍 EOS → 空结果；其后 EOS → 停止；
  5. max_new_tokens=128（config.h:34，sherpa CLI 默认）；cur_len 达 max_seq_len(=512) 停止；
     需要时按 keep_audio 截断占位帧（:873-938）；
  6. 复读守卫：生成序列最近 64 token 内不同 id ≤3 → 判定塌缩，去除尾部循环 token 并停止
     （:321-350, 1089-1102）；
  7. 去除 UTF-8 U+FFFD 替换符（:391-401）。
- 解码上下文上限核算：before(~20-30) + audio_pad(13 token/10s 音频) + after(~8) + ≤128 生成
  ⇒ 512 上限对 ≤20s 段（≈26 audio token）余量充足。

## 8. 参考源码出处（docs-transcribe-reference/，仅移植参考）

均自 GitHub raw（tag 直取），Apache-2.0：

| 文件 | 出处 |
| --- | --- |
| offline-qwen3-asr-model.{cc,h} | k2-fsa/sherpa-onnx v1.13.8（commit 11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf）sherpa-onnx/csrc/ |
| offline-recognizer-qwen3-asr-impl.{cc,h} | 同上 |
| offline-qwen3-asr-model-config.{cc,h} | 同上 |
| qwen-asr-tokenizer.{cc,h} | 同上 |
| silero-vad-model.cc、silero-vad-model-config.h、voice-activity-detector.cc、vad-model-config.h | 同上 |
| offline-recognizer-impl.{cc,h}、features.{cc,h}、offline-stream.cc、math.{cc,h}、offline-recognizer-whisper-tpl-impl.h | 同上 |
| knf-{whisper-feature,mel-computations,feature-window,rfft,feature-fbank,online-feature}.cc | csukuangfj/kaldi-native-fbank v1.22.3（sherpa-onnx v1.13.8 cmake/kaldi-native-fbank.cmake pin，SHA256 9176cc66…93cf61）kaldi-native-fbank/csrc/ |
| preprocessor_config.json | HF Qwen/Qwen3-ASR-0.6B（/tmp/e2e-poc/dl/qwen3-0.6B-official/ 复制） |

注：v1.13.8 起仓库为 monorepo 布局，路径前缀 `sherpa-onnx/csrc/`（csrc 顶层路径 404）。

## 9. PoC 实现顺序建议（给实现阶段）

1. hound 读 wav → f32 [-1,1]；补 80 样本零；rustfft 自写 fbank（§6 全参数）→ NormalizeWhisperFeatures
2. conv_frontend → FeatToAudioTokensLen → mask → encoder → TrimAudioFeatures
3. silero VAD（512 窗，v4 状态机）→ 段列表（起/止样本）
4. 每段：BuildSourceIds → 静态 KV(512) → prefill/逐步 greedy（§7）
5. CLI 契约按 transcribe-poc/CONTRACT.md（stdout 仅 WALL_S/RTF/SEGMENTS/MODE 四行）
6. 对拍：`python3 scripts/cer.py assets/reference.txt <outDir>/transcript.txt`

## 10. 并行策略（nproc=12，命令 `nproc` 实测）

- 基线反事实：sherpa 单线程 RTF 0.366 ⇒ 700s 音频 wall 256.4s；解码（自回归逐步）占大头。
- 段间并行优先：VAD 段相互独立，conv/encoder 天然可并发；ORT 同一 Session 允许并发 Run，
  权重只加载一份，各 worker 仅自持 KV 缓存（112 MiB/段）。
- 推荐：**4 个段 worker × decoder/encoder intra=2**（≈8 活跃线程 + 主线程，留 3-4 核给
  fbank/IO/系统，避免 WSL2 上超订阅）；内存增量 ≈ 4×112 MiB KV + 权重 ≈ 与 sherpa 峰值
  2721MB 同量级。激进档 6 workers × intra=1（内存 +672 MiB KV）。
- MODE 行建议格式：`seg-parallel:<W>workers,ort-intra:<T>`。
- 以上为假设，需以 SEGMENTS 实测与门禁数据验证（预期收益 2-4×，未实测前不作为结论）。

## 11. 已验证/未验证声明

已验证：资产 md5、音频时长、三模型+VAD 的 IO（onnx 直读）、dylib 经 ort rc.13 load-dynamic
加载并建 session（`cargo run --example ort_dylib_smoke` → `DYLIB_OK … inputs=3 outputs=3`）、
cargo check/build 通过、stub 退出码与 stdout 空。
未验证（实现阶段完成前无法验证）：端到端 CER、RTF、SEGMENTS 数值、并行收益。
