# 🤝 VecBoost 贡献指南

感谢关注 **VecBoost**！本文档描述本地开发的工作流约定与提交规范。无论您是在修复缺陷、添加新特性、改进文档还是帮助他人，您的贡献都弥足珍贵。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [环境要求](#-环境要求)
- [构建](#-构建)
- [测试](#-测试)
- [质量门禁](#-质量门禁提交前必须全绿)
- [变更管理（specmark）](#-变更管理specmark)
- [提交规范](#-提交规范)
- [文档约定](#-文档约定)
- [安全](#-安全)

</details>

---

## 🧰 环境要求

- Rust ≥ 1.91（edition 2024；仓库 `Cargo.toml` 的 `rust-version` 字段为权威值）
- Python ≥ 3.10 + pytest（场景测试）
- 可选：protobuf-compiler（gRPC E2E）、docker

## 🧱 构建

```bash
# 库 + 二进制（默认 http feature）
cargo check -p vecboost -p vecboost-examples --all-targets

# 常用特性组合
cargo check -p vecboost --features http,grpc,cli,auth,db,onnx
cargo check -p vecboost --features mcp
```

> **路径依赖提示**：`../base/*` 7 个生态库是路径依赖的活仓库，会独立演进。
> fmt/clippy/测试命令必须以 `-p vecboost -p vecboost-examples` 限定，
> 否则会把外部仓库的中间态卷进门禁。

## 🧪 测试

| 层 | 命令 | 说明 |
|---|---|---|
| 单元 + 集成(Rust) | `cargo test -p vecboost -p vecboost-examples` | 2237 内联测试 + tests/（63 个集成/专项测试函数） |
| gRPC E2E | `cargo test -p vecboost --features http,grpc --test grpc_e2e` | 拉起真实二进制 |
| 场景(Python) | `pytest tests/scenario -q` | conftest 拉起真实服务器；models/ 缺席时推理用例自动 SKIP |
| 微基准 | `cargo bench -p vecboost` | criterion |

三层性能测试栈的职责划分与场景穷举见 [docs/TEST_SCENARIOS.md](TEST_SCENARIOS.md)。

### worktree 开发：测试资产准备

大体积资产（`models/`、`3rdparty/`、`assets/`）均不入库（gitignore），
相关测试（真机推理、场景套件、ORT 动态库）在资产缺席时按守卫 SKIP。
在 git worktree 中需要真跑这些测试时：

```bash
bash scripts/link-assets.sh   # 自动探测主工作区并符号链接三类资产
# 清除: rm models 3rdparty assets
```

注意：符号链接不被 `.gitignore` 的 `models/`（尾斜杠仅匹配目录）模式
匹配——`git status` 会显示为 untracked，不要 `git add`。

## ✅ 质量门禁(提交前必须全绿)

```bash
cargo fmt --check -p vecboost -p vecboost-examples
cargo clippy -p vecboost -p vecboost-examples --all-targets -- -D warnings -W clippy::unwrap_used
cargo test -p vecboost -p vecboost-examples
python3 scripts/doc_consistency_check.py   # 修改 README/API 后
python3 scripts/i18n_consistency_check.py # 修改用户可见文案/语言包后
```

- `clippy.toml` 开启了 `allow-unwrap-in-tests`：生产代码禁止 `unwrap/expect` panic 面，测试豁免。
- `deny.toml` 为 cargo-deny 配置(advisories/licenses/bans)；新豁免须写明理由。
- i18n：**默认显示语言严格为英文**，用户可见文案（错误/日志/CLI 帮助）一律走
  `src/i18n/locales/{en,zh}/*.ftl` 键位，禁硬编码中文；约束正文见 [MAINTENANCE](MAINTENANCE.md)。
  豁免与基线分别记在 `scripts/i18n_allowlist.json` / `scripts/i18n_baseline.json`（后者只降不升）。
- pre-commit 钩子（`.pre-commit-config.yaml` → `scripts/pre-commit.sh`）：fmt / clippy / check / build + Apache-2.0 版权头检查。

## 🔄 变更管理(specmark)

多步骤变更走 `specmark/` 工作流：propose(proposal/design/tasks/specs)→ analyze → apply → converge → archive。`specmark/` 目录不入库(gitignore)，归档目录只读——发现归档稿与实现不符，勘误落在活文档（主 spec 或契约文件），不改写归档。

## 📝 提交规范

- 格式：`type(scope): 摘要`，type ∈ feat/fix/test/docs/chore/refactor/perf
- 每个逻辑变更一个 commit；行为变更(默认值、API 契约)必须在 CHANGELOG 的
  `Unreleased` 段登记，并同步更新 README 双语版

## 📚 文档约定

- `docs/` 平铺 UPPER_SNAKE 命名：`USER_GUIDE.md`、`API_REFERENCE.md`、`ARCHITECTURE.md`、`CHANGELOG.md`、`CONTRIBUTING.md`、`FAQ.md`、`PERFORMANCE.md`、`SECURITY.md`、`TEST_SCENARIOS.md`、`MAINTENANCE.md`。
- 每篇文档内部结构：`# <emoji> <标题>` → `## 📋 目录` → emoji 章节。
- 修改 README/API 后运行 `python3 scripts/doc_consistency_check.py` 保持文档与实现一致。
- 性能数字必须来自 `docs/benchmarks/` 实测归档或标注测量方法，禁止编造；实验结论回写 [docs/PERFORMANCE.md](PERFORMANCE.md)。
- 跨变更的长期约束（转录消费面、`transcribe-own` CLI 契约、退役配置项纪律、废弃文档治理）统一沉在 [🧹 维护规范](MAINTENANCE.md)；根目录不保留 `DEPRECATED.md` 这类历史叙述型废弃报告，一次性退役说明随变更进归档。

## 🔐 安全

发现安全漏洞请勿公开 issue，优先走 GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy)（备选联系 Kirky.X：<kirky-x@outlook.com>），流程见 [docs/SECURITY.md](SECURITY.md)。
依赖 advisory 由 CI `cargo-audit` 门禁；豁免配置在 `.cargo/audit.toml`(须附理由)。
