# DEPRECATED — transcribe-poc 实验通路

> 日期：2026-10-10 · 关联变更：`specmark/archive/2026-10-10-transcribe-poc-shell/`（裁定记录见其 `design.md` D1）

## 状态

`transcribe-poc/`（独立 crate，二进制 `transcribe-own`）**不再是产品化转录交付物**。它保留为**独立实验与基准通路**，与主仓 `vecboost` 的发布解耦：不在主 workspace 构建、不随主 crate 发布、无兼容性承诺。

## 产品化实现位置

Qwen3-ASR 转录引擎已由 `onnx-engine-unify` 忠实并入主仓：

- 实现：`src/engine/transcribe/`（`asr`/`vad`/`features`/`tokenizer`/`wav`/`engine`，feature `transcribe` 门控）
- 入口：`ModelTask::Transcribe` + `InferenceEngine::transcribe`（经 `EngineFactory::create` 按 task 分派）
- 配置：`config` 段 `task = "transcribe"` 与 `transcription_params`（`models_dir` / `vad_threshold`）

**新消费方一律走主仓 API，不要再依赖本目录的本地推理实现。**

## 为何降级而非薄壳化

薄壳化（POC 路径依赖主仓、删除本地推理实现）需要主仓公开面提供：VAD-only 分段、段级转录、wav 读取、会话旋钮（intra/no_spin/env-pool/cuda）——现有公开入口 `InferenceEngine::transcribe` 只有整文件串行语义，无法逐字保持 `CONTRACT.md` 契约（`--vad-only`、默认段级并行 4 workers、`TRANSCRIBE_ENV_POOL`/`TRANSCRIBE_BATCH`/`TRANSCRIBE_GPU` 旋钮）。裁定为保留原实现 + 本指针（完全可逆）。重新薄壳化的前置论证见归档变更的 `design.md` D1/D2。

## 保留内容与用途

- `pytorch_backend.rs`：pyo3 基线对拍通路（GPU 1.7B bf16），供主仓 int8 引擎做精度/RTF 对照
- `batch.rs`（lockstep 合批）/ `gpu.rs`（KV io-binding）：实验结论已记录于 `out/transcribe-poc/own-engine/summary.md`，代码随本目录保留备查
- `CONTRACT.md`：自动门禁调用契约，**不是迁移对象**，保持原样

## 构建（仅实验用途）

```bash
ORT_DYLIB_PATH=<onnxruntime-1.28.2.so> cargo run --release --manifest-path transcribe-poc/Cargo.toml -- <wav> <outDir> [--models <dir>] [--vad-only]
```
