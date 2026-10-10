# DEPRECATED — transcribe-poc 实验通路（薄壳化已完成）

> 日期：2026-10-10（勘误更新：薄壳化已执行）· 关联归档：`transcribe-poc-shell`（B 裁定，2026-10-10）、`transcribe-poc-shell-a`（A 执行，2026-10-10）

## 现状

`transcribe-poc/`（独立 crate，二进制 `transcribe-own`）**已收敛为薄壳 CLI**：本地推理实现（asr/vad/features/tokenizer/wav + batch/gpu 实验通路，~1700 行）已全部删除，推理改调主仓公开门面 `vecboost::transcribe`（`transcribe` feature，实现位于 `src/engine/transcribe/`）。CLI 解析 / 路径归一化 / stdout 契约 / 退出码 / py-torch 基线对拍通路保留在本 crate。

**语义级双实现已消除**（曾以 `repo-semantic-dedup` 同类口径记录接受，2026-10-10 零接受项指令下彻底解决）。

## 历史沿革（为何本文档曾主张降级）

1. `onnx-engine-unify` 把转录引擎并入主 crate 时，POC 收敛方式被拆分为独立变更（最小 pub 入口是不可逆单向门）。
2. `transcribe-poc-shell` 归档件初版裁定 B（D5 降级）：彼时主仓无 VAD-only/段级/wav/旋钮公开面，忠实薄壳需 3+ 公开项、且会丢失 ENV_POOL/BATCH/GPU 实验旋钮，故保留本地实现+本指针（可逆）。
3. `transcribe-poc-shell-a`（2026-10-10）落地 A 路径：主仓门面建立后执行薄壳化。**本文档"为何降级而非薄壳化"一节的判据已作废**——公开面现已存在（`vecboost::transcribe::{open, read_wav_16k_mono, detect_segments, transcribe_segment, TranscribeOpts}`），契约逐字保持（`CONTRACT.md`：五行 stdout、退出码 0/1/2、输出文件行为含 `--vad-only` 空 segments.txt 复刻）。

## 实验通路退役说明

`TRANSCRIBE_ENV_POOL` / `TRANSCRIBE_BATCH` / `TRANSCRIBE_GPU` 三个实验旋钮随实验文件删除而退役——**结论已成文**于 `out/transcribe-poc/own-engine/summary.md`（§11：io-binding 非确定性读回；合批与 env-pool 实测档位），代码冗余正是薄壳化消除对象，非缺陷遗留。门禁若设置这些变量，薄壳 stderr 显性告警并忽略（禁静默）。

## 消费指引

- **新消费方**：服务面 `POST /api/1/transcribe`（`[transcribe] enabled=true`，需 `transcribe` feature）或库 API `vecboost::transcribe`；`transcribe-poc` 仅作 CLI/基准 harness 保留。
- **重新薄壳化的前置论证**（如未来再评估）：见归档 `specmark/archive/2026-10-10-transcribe-poc-shell/design.md` D1/D2 与 `specmark/archive/2026-10-10-transcribe-poc-shell-a/design.md`。

## 勘误记录（2026-10-10）

- `specmark/archive/2026-10-10-transcribe-poc-shell/tasks.md` T004 稿称 stdout"四行契约"为旧稿；`CONTRACT.md` 与 delta spec R-transcribe-poc-cli-001（2026-10-09 勘误）均为**五行**（含 `TRANSCRIPT_CHARS`）。归档只读不改写，以 spec/CONTRACT 为准，特此勘误。
