# 🧹 VecBoost 维护规范

本文沉淀**跨变更长期有效**的维护约束，来源为已退役的根目录废弃报告 `DEPRECATED.md`（2026-10-10 勘误版，随薄壳化完成已从仓库移除）。沿革叙述与裁定的单一事实源是 `specmark/archive/**`（归档只读，且 `specmark/` 不入库）；本文只保留仍然有效的规则。根目录 `AGENTS.md`（agent 入口指针，按 `.gitignore` 约定不入库）只指向本文，不重复正文。

## 📋 目录

- [转录能力消费面](#-转录能力消费面)
- [transcribe-own CLI 契约](#-cli-契约自动门禁逐字依赖不得偏离)
- [退役配置项处理纪律](#-退役配置项处理纪律显性告警禁静默)
- [废弃文档与勘误治理](#-废弃文档与勘误治理)
- [i18n 约束](#-i18n-约束默认英文禁硬编码中文)
- [门禁命令](#-门禁命令)

---

## 🚚 转录能力消费面

| 消费方式 | 入口 | 前置条件 |
|---|---|---|
| 服务端 | `POST /api/v1/transcribe` | `[transcribe] enabled=true`，需 `transcribe` feature |
| 库门面 | `vecboost::transcribe::{open, read_wav_16k_mono}`；`TranscribeSession::{detect_segments, transcribe_segment}`；`TranscribeOpts` | 需 `transcribe` feature |

- 推理实现只允许存在于 `src/engine/transcribe/`。`transcribe-poc/`（独立 crate，二进制 `transcribe-own`，非 workspace 成员）仅作 CLI 与基准 harness 保留；**该目录为本机目录不入库**（2026-10-15 owner 裁定：不发布、与主仓功能面无关，经 `.gitignore` 排除），仓库内以本节与下方 CLI 契约要点为其规范面。
- **禁止语义级双实现**：不得在 `transcribe-poc` 内重写 asr / vad / features / tokenizer / wav。历史上"为保留实验旋钮而降级保留本地实现"的口径已彻底作废。
- 段级并行由调用方持有多个 `TranscribeSession` 实现（ort rc.13 禁止同一 Session 并发 `Run`，每会话独占一套），门面内不做线程池/会话池。
- 公开门面即 POC 的消费面：**不得**为主仓内部实现细节在门面开口子。
- 输入格式与特征链的**入库事实源为代码本身**：`src/engine/transcribe/features.rs`（帧移 160 / 帧长 400 / n_fft=400 / mel 128 bins / Slaney 尺度与归一化 / 前导 80 样本居中对齐 / 全局归一化）与 `src/engine/transcribe/wav.rs`。wav 必须是 16kHz / 单声道 / 16-bit PCM，不符即显性报错，**不做静默重采样**。PoC 期设计稿 `docs-transcribe/DESIGN.md`（原 §6 与本节同义）**已于 2026-10-18 改为本机不入库**，不得作为门禁或实现依据引用。

## 📜 CLI 契约（自动门禁逐字依赖，不得偏离）

权威文本：`transcribe-poc/CONTRACT.md`（门禁依据本身，不是被迁移对象、不得改写；**本机文件，不入库**）与主 spec `specmark/specs/transcribe-poc-cli/spec.md`（`R-transcribe-poc-cli-001`，`specmark/` 仅本机存在）。要点：

- **调用形式**：`transcribe-own <wav> <outDir> [--models <dir>] [--vad-only]`；`--models` 缺省为相对进程 CWD 的 `assets/models`，门禁调用时显式传入。
- **stdout 五行**：全管线运行中打印且**仅**打印 `WALL_S=<浮点秒>`、`RTF=<浮点>`、`SEGMENTS=<整数>`、`TRANSCRIPT_CHARS=<整数>`、`MODE=<单行自由文本>`。
- **行数以 spec / CONTRACT 为准（五行）**：归档稿中"四行契约"（缺 `TRANSCRIPT_CHARS`）一律视为旧稿，不得据其实现或验收。
- **`--vad-only`**：stdout 仅 `SEGMENTS=<整数>` 一行，且仍须写出 `<outDir>/segments.txt`（内容为空文件）——复刻旧二进制行为，不得省略该文件。
- **输出文件**：`<outDir>/transcript.txt`（各段文本拼接的全文）、`<outDir>/segments.txt`（每行 `起 -- 止: 文本`，起止为秒的浮点）。
- **日志分流**：进度与诊断一律走 stderr；stdout 除契约行外不得有任何其他输出。`MODE=` 为自由文本但须单行无换行，且**不得回填已退役旋钮字段**。
- **退出码**：`0` 成功 / `2` 用法错误 / `1` 运行失败（原因写 stderr）。
- **运行时依赖**：`ORT_DYLIB_PATH` 指向 `3rdparty/libonnxruntime.so.1.28.2`（ort 以 `load-dynamic` 加载）；POC 依赖白名单仅 `vecboost`（`transcribe` feature）+ `pyo3`（py-torch 基线对拍 `run_pytorch_baseline`）。

## 🔇 退役配置项处理纪律：显性告警，禁静默

- `TRANSCRIBE_ENV_POOL` / `TRANSCRIBE_BATCH` / `TRANSCRIBE_GPU` 随 POC 实验文件退役，不再提供任何实现；`TranscribeOpts` 只保留 `intra_threads` / `no_spin` / `cuda` / `vad_threshold`。
- 已退役的配置项仍被设置时，必须 stderr 显性告警后忽略（实现见 `transcribe-poc/src/main.rs` 的 `warn_retired_knobs`），**禁止静默吞掉**。同一纪律适用于非契约输入格式与 CUDA EP 不可用：一律显性报错。
- 实验结论记录于 `out/transcribe-poc/own-engine/summary.md` §11（io-binding 非确定性读回；合批与 env-pool 实测档位）。注意 `out/` 与 `temp/` 均在 `.gitignore` 中（未入库，仅本机可查）：任何以它们为验收证据的门禁都必须准备文件缺失的分支，不得在 CI 中假设其存在。

## 🧹 废弃文档与勘误治理

- **归档只读**：`specmark/archive/**` 一经归档不再改写。发现归档稿与实现/契约不符，勘误落在活文档（主 spec `specmark/specs/**` 或 `CONTRACT.md`），归档正文保留原样。
- **废弃报告不落根目录**：一次性退役说明随变更进归档；只有跨变更长期有效的约束才提炼进本文。根目录不保留 `DEPRECATED.md` 这类历史叙述型文档。
- **判据失效即作废**：历史降级/保留理由所依赖的前提一旦消失（例如"当时主仓无公开门面，故保留本地实现"），该判据不得再作为后续决策依据；重新论证须引用归档 `design.md` 的原始决策。
- **删除前查入链**：移除文档或文件前先 `git grep` 入链并同步修正（`README.md` / `README_EN.md` / `docs/**` / `src/**` 文档注释 / `.gitignore` 注释），避免死链；只删除确认无消费方的产物。
- **设计文档禁止入库**（2026-10-18 owner 裁定）：PoC 期设计说明、可行性评估这类描述性文档不得出现在 git 仓库（与 `docs/TRAIT-KIT-SDK-ASSESSMENT.md` 等本机专有文档同一口径）；经 `.gitignore` 排除并 `git rm --cached` 取消跟踪，本地保留。其仍有效的结论须提炼进本文或落到代码文档注释，不得以“设计稿已入库”充当事实源。
- **勿误删的关联件**：`config-e2e-transcribe.toml`（WAV 链路验收配置）为在用入库件。`transcribe-poc/` 整目录（契约原文、pytorch 基线对拍）与 `docs-transcribe/`（PoC 期设计稿，含外机路径与未入库参考源码行号）**均为本机不入库目录**——属本地资产而非可废弃件，勿从磁盘删除，也勿重新加入 git。

## ✅ 门禁命令

```bash
# transcribe-poc 为本机不入库目录；torch 后端编译需 PYO3_PYTHON 指向 cpython-3.11
cargo check --manifest-path transcribe-poc/Cargo.toml --all-targets
cargo clippy --manifest-path transcribe-poc/Cargo.toml --all-targets -- -D warnings
# doc_consistency_check.py 共五类核验：路由 / gRPC / 环境变量 / CLI 子命令 / **文档锚点**
# （锚点项按实测校准的 GitHub slug 规则校验 `](#frag)` 与 `](path#frag)` 可达性，
# 并禁止标题残留 U+FE0F 变体选择符）
python3 scripts/doc_consistency_check.py
python3 scripts/i18n_consistency_check.py
```

## 🌐 i18n 约束（默认英文，禁硬编码中文）

- **默认显示语言严格为英文**（`en`）：`detect_locale()` 的回退链
  `VECBOOST_LANG → LC_ALL → LC_MESSAGES → LANG → sys-locale → "en"` 不得改为恒返 zh；
  `locales/en/*.ftl` 必须零中文（含注释）。中文只能由显式环境变量或 `Accept-Language` 触发。
- **协议面单一事实源**：`i18n::init()` 末尾调 `sync_sdforge_i18n()`，把本仓 FTL 的
  `forge-*` 键注册进 sdforge 注册表并对齐 locale；`SDFORGE_LANG` 不得反向决定应用面语言。
  新增 `#[forge(cli = true, …)]` 必须带 `i18n_key = "forge-cli-<name>"`（无键则 `--help` 不跟随 locale）。
- **不得被 i18n 改写的机器输出**：`transcribe-own` stdout 五行契约、`PASS/WARN/FAIL` 与
  `CheckResult.name` 等可 grep token、HTTP JSON 字段名、`error_code`、Prometheus 文本格式。
- **回归拦截**：`scripts/i18n_consistency_check.py`（R1 默认英文、R2 en/zh 键齐性、
  R3 `tr()`/`i18n_key` 引用有效、R4 生产码禁硬编码中文、R5 未国际化文案棘轮）；
  豁免记 `scripts/i18n_allowlist.json`（必带 reason），封顶记 `scripts/i18n_baseline.json`（只降不升）。
