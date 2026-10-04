# transcribe-own CLI 契约（自动门禁逐字依赖，不得偏离）

创建：准备工程师，2026-10-05。后续自动门禁按本契约构建与调用本 crate。

## 调用形式

```
transcribe-own <wav> <outDir> [--models <dir>]
```

- `<wav>`：16kHz 单声道 16-bit PCM WAV（PoC 不做重采样；输入即 `assets/audio_from_m4a.wav` 或 `assets/smoke10s.wav`）
- `<outDir>`：输出目录，不存在则创建
- `[--models <dir>]`：模型目录，缺省为 `assets/models`（相对进程 CWD）；门禁调用时显式传入

## 模型目录布局（--models 指向）

```
<models>/qwen3-conv_frontend.onnx
<models>/qwen3-encoder.int8.onnx
<models>/qwen3-decoder.int8.onnx
<models>/silero_vad.onnx
<models>/tokenizer/vocab.json
<models>/tokenizer/merges.txt
<models>/tokenizer/tokenizer_config.json
```

## stdout 关键行（运行中打印且仅打印这些行）

```
WALL_S=<浮点秒>      # 端到端 wall 时钟（秒，浮点）
RTF=<浮点>           # wall_s / 音频时长_s
SEGMENTS=<整数>      # VAD 切出的段数
MODE=<并行方式描述>   # 如 "seg-parallel:4x{intra:3}"，自由文本但单行无换行符
```

进度/日志一律走 stderr；stdout 除以上四行外不得有任何其他输出。

## 输出文件

- `<outDir>/transcript.txt`：纯文本全文（各段文本拼接）
- `<outDir>/segments.txt`：每行 `起 -- 止: 文本`，起止为秒（浮点），例如 `0.00 -- 5.12: 你好`

## 运行时依赖

- onnxruntime 动态库：环境变量 `ORT_DYLIB_PATH` 指向 worktree 内 `3rdparty/libonnxruntime.so.1.28.2`
  （自主仓 `/home/dev/projects/vecboost/3rdparty/onnxruntime/libonnxruntime.so.1.28.2` 只读复制，
  md5=7294310fc3d98bfdc98246c2c53c4cae，24277040 字节；ort crate 以 `load-dynamic` 加载）
- Cargo 依赖（见 Cargo.toml）：ort 2.0.0-rc.13（load-dynamic+ndarray+std）、tokenizers 0.23、
  hound 3.5、rustfft 6.4；全部 default-features=false。**禁止引入 sherpa-onnx**（其源码仅作移植参考，
  见 docs-transcribe-reference/ 与 docs-transcribe/DESIGN.md 出处注记）

## 退出码

- 0 成功；2 用法错误；1 运行失败（stderr 说明原因）
