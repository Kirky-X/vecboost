# 🤝 VecBoost 贡献指南

感谢关注 **VecBoost**！本文档描述本地开发的工作流约定与提交规范。无论您是在修复缺陷、添加新特性、改进文档还是帮助他人，您的贡献都弥足珍贵。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [环境要求](#环境要求)
- [构建](#构建)
- [测试](#测试)
- [质量门禁](#质量门禁提交前必须全绿)
- [变更管理（specmark）](#变更管理specmark)
- [提交规范](#提交规范)
- [文档约定](#文档约定)
- [安全](#安全)

</details>

---

## 🧰 环境要求

- Rust ≥ 1.91（edition 2024；仓库 `Cargo.toml` 的 `rust-version` 字段为权威值）
- Python ≥ 3.10 + pytest（场景测试）
- 可选：protobuf-compiler（gRPC E2E）、docker

## 🏗️ 构建

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
| 单元 + 集成(Rust) | `cargo test -p vecboost -p vecboost-examples` | 1700+ 内联测试 + tests/ |
| gRPC E2E | `cargo test -p vecboost --features http,grpc --test grpc_e2e` | 拉起真实二进制 |
| 场景(Python) | `pytest tests/scenario -q` | conftest 拉起真实服务器；models/ 缺席时推理用例自动 SKIP |
| 微基准 | `cargo bench -p vecboost` | criterion |

三层性能测试栈的职责划分与场景穷举见 [docs/TEST_SCENARIOS.md](TEST_SCENARIOS.md)。

## ✅ 质量门禁(提交前必须全绿)

```bash
cargo fmt --check -p vecboost -p vecboost-examples
cargo clippy -p vecboost -p vecboost-examples --all-targets -- -D warnings -W clippy::unwrap_used
cargo test -p vecboost -p vecboost-examples
python3 scripts/doc_consistency_check.py   # 修改 README/API 后
```

- `clippy.toml` 开启了 `allow-unwrap-in-tests`：生产代码禁止 `unwrap/expect` panic 面，测试豁免。
- `deny.toml` 为 cargo-deny 配置(advisories/licenses/bans)；新豁免须写明理由。
- pre-commit 钩子（`.pre-commit-config.yaml` → `scripts/pre-commit.sh`）：fmt / clippy / check / build + MIT 版权头检查。

## 🔄 变更管理(specmark)

多步骤变更走 `specmark/` 工作流：propose(proposal/design/tasks/specs)→ analyze → apply → converge → archive。`specmark/` 目录不入库(gitignore)，归档目录只读。

## 📝 提交规范

- 格式：`type(scope): 摘要`，type ∈ feat/fix/test/docs/chore/refactor/perf
- 每个逻辑变更一个 commit；行为变更(默认值、API 契约)必须在 CHANGELOG 的
  `Unreleased` 段登记，并同步更新 README 双语版

## 📚 文档约定

- `docs/` 平铺 UPPER_SNAKE 命名：`USER_GUIDE.md`、`API_REFERENCE.md`、`ARCHITECTURE.md`、`CHANGELOG.md`、`CONTRIBUTING.md`、`FAQ.md`、`PERFORMANCE.md`、`SECURITY.md`、`TEST_SCENARIOS.md`。
- 每篇文档内部结构：`# <emoji> <标题>` → `## 📋 目录` → emoji 章节。
- 修改 README/API 后运行 `python3 scripts/doc_consistency_check.py` 保持文档与实现一致。
- 性能数字必须来自 `docs/benchmarks/` 实测归档或标注测量方法，禁止编造；实验结论回写 [docs/PERFORMANCE.md](PERFORMANCE.md)。

## 🔐 安全

发现安全漏洞请勿公开 issue，联系 maintainer(kirky-x@outlook.com)，流程见 [docs/SECURITY.md](SECURITY.md)。
依赖 advisory 由 CI `cargo-audit` 门禁；豁免配置在 `.cargo/audit.toml`(须附理由)。
