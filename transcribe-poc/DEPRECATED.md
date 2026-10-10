# DEPRECATED — 实验通路（见根目录 DEPRECATED.md）

本 crate（二进制 `transcribe-own`）**不再是产品化转录交付物**，保留为独立实验与基准通路：lockstep 合批（`batch.rs`）、KV io-binding（`gpu.rs`）、py-torch 基线对拍（`pytorch_backend.rs`）等主 crate 未承接的旋钮，不随主 crate 发布、无兼容性承诺。

产品化 Qwen3-ASR 转录实现已并入主仓 `src/engine/transcribe/`（feature `transcribe`，`InferenceEngine::transcribe` 入口）。完整降级说明、理由与重新薄壳化的前置论证：

- 根目录 [`../DEPRECATED.md`](../DEPRECATED.md)
- 裁定记录：specmark 归档变更 `transcribe-poc-shell`（2026-10-10 归档，从 `onnx-engine-unify` Phase P 拆出独立立项；`design.md` D1/D2）

`CONTRACT.md` 调用契约保持原样，不是迁移对象。
