# DEPRECATED — 实验通路（薄壳化已完成，见根目录 DEPRECATED.md）

本 crate（二进制 `transcribe-own`）**已收敛为薄壳 CLI**：本地推理实现已删除，推理经主仓公开门面 `vecboost::transcribe`（`transcribe` feature）执行；本 crate 只保留 CLI 解析、路径归一化、stdout 契约（五行：`WALL_S`/`RTF`/`SEGMENTS`/`TRANSCRIPT_CHARS`/`MODE`）、退出码 0/1/2 与 py-torch 基线对拍通路（`run_pytorch_baseline`）。

- 产品化实现：主仓 `src/engine/transcribe/` + 门面 `vecboost::transcribe` + 服务端点 `POST /api/1/transcribe`（`[transcribe]` 段启用）。
- 完整现状/沿革/实验旋钮退役说明：根目录 [`../DEPRECATED.md`](../DEPRECATED.md)（2026-10-10 勘误版）。
- 裁定记录：归档变更 `transcribe-poc-shell`（B 初裁）与 `transcribe-poc-shell-a`（A 执行，2026-10-10）。
- `CONTRACT.md` 调用契约保持原样，不是迁移对象。
- 三个实验旋钮（ENV_POOL/BATCH/GPU）已退役，结论见 `out/transcribe-poc/own-engine/summary.md`；设置时本 CLI stderr 显性告警。
