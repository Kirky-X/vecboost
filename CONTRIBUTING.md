# CONTRIBUTING

感谢关注 VecBoost!本文档描述本地开发的工作流约定。

## 环境要求

- Rust ≥ 1.85(edition 2024;仓库 `rust-version` 字段为权威值)
- Python ≥ 3.10 + pytest(场景测试)
- 可选:protobuf-compiler(gRPC E2E)、docker

## 构建

```bash
# 库 + 二进制(默认 http feature)
cargo check -p vecboost -p vecboost-examples --all-targets

# 常用特性组合
cargo check -p vecboost --features http,grpc,cli,auth,db,onnx
cargo check -p vecboost --features mcp
```

> **路径依赖提示**:`../base/*` 7 个生态库是路径依赖的活仓库,会独立演进。
> fmt/clippy/测试命令必须以 `-p vecboost -p vecboost-examples` 限定,
> 否则会把外部仓库的中间态卷进门禁。

## 测试

| 层 | 命令 | 说明 |
|---|---|---|
| 单元 + 集成(Rust) | `cargo test -p vecboost -p vecboost-examples` | 1700+ 内联测试 + tests/ |
| gRPC E2E | `cargo test -p vecboost --features http,grpc --test grpc_e2e` | 拉起真实二进制 |
| 场景(Python) | `pytest tests/scenario -q` | conftest 拉起真实服务器;models/ 缺席时推理用例自动 SKIP |
| 微基准 | `cargo bench -p vecboost` | criterion |

三层性能测试栈的职责划分见 [docs/TESTING.md](docs/TESTING.md)。

## 质量门禁(提交前必须全绿)

```bash
cargo fmt --check -p vecboost -p vecboost-examples
cargo clippy -p vecboost -p vecboost-examples --all-targets -- -D warnings -W clippy::unwrap_used
cargo test -p vecboost -p vecboost-examples
python3 scripts/doc_consistency_check.py   # 修改 README/API 后
```

- `clippy.toml` 开启了 `allow-unwrap-in-tests`:生产代码禁止 `unwrap/expect` panic 面,测试豁免。
- `deny.toml` 为 cargo-deny 配置(advisories/licenses/bans);新豁免须写明理由。

## 变更管理(specmark)

多步骤变更走 `specmark/` 工作流:propose(proposal/design/tasks/specs)→ analyze → apply → converge → archive。`specmark/` 目录不入库(gitignore),归档目录只读。

## 提交规范

- 格式:`type(scope): 摘要`,type ∈ feat/fix/test/docs/chore/refactor/perf
- 每个逻辑变更一个 commit;行为变更(默认值、API 契约)必须在 CHANGELOG 的
  `Unreleased` 段登记,并同步更新 README 双语版

## 安全

发现安全漏洞请勿公开 issue,联系 maintainer(kirky-x@outlook.com)。
依赖 advisory 由 CI `cargo-audit` 门禁;豁免配置在 `.cargo/audit.toml`(须附理由)。
