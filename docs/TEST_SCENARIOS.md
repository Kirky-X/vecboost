# 🧪 VecBoost 测试场景矩阵

> 适用版本：VecBoost **0.2.1**（workspace，Rust edition 2024 / `rust-version = 1.91`）。
> 用途：穷举 vecboost 的测试栈职责划分与全部验收场景，回答"该把测试写在哪里"。
> 编写依据（只读核对）：`tests/scenario/*.py`（15 个套件及 docstring 中的 R-* 场景 ID）、`tests/*.rs`、`tests/perf/`、`benches/`、`.github/workflows/`（health-check / feature-matrix / scenario-tests）、`scripts/run-scenario-tests.sh`、`tests/scenario/conftest.py`。
> 所有套件名与场景 ID 均经 `grep` 核实存在。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [三层测试栈总览](#-三层测试栈总览)
- [场景套件矩阵（tests/scenario）](#-场景套件矩阵testsscenario)
- [Rust 集成与专项测试](#-rust-集成与专项测试)
- [性能与基准测试](#-性能与基准测试)
- [单元测试](#-单元测试)
- [运行命令（与 CI 一致）](#️-运行命令与-ci-一致)
- [测试模式与夹具](#-测试模式与夹具)
- [模型矩阵](#-模型矩阵)
- [约定](#-约定)

</details>

---

## 🧭 三层测试栈总览

vecboost 的性能/质量验证分三层，职责边界如下：

| 层 | 位置 | 引擎 | 职责 | 运行方式 |
|---|---|---|---|---|
| **微基准** | `benches/`（criterion） | Mock/真实均可 | 度量原子操作的绝对性能并防回归（相似度 SIMD、批调度、语义缓存索引、吞吐基线） | `cargo bench -p vecboost` |
| **回归阈值** | `tests/perf/performance_test.rs` | MockEngine | 断言吞吐/延迟不劣化（相对阈值），CI 可跑、不依赖真实模型 | `cargo test -p vecboost --test perf` |
| **真实服务 E2E** | `tests/scenario/*.py`（pytest） | 真实二进制 + `models/` 本地模型 | 端到端行为契约：HTTP 矩阵/CLI/MCP/生命周期/安全；P99 延迟与真实吞吐 | `pytest tests/scenario -q`（conftest 自动拉起服务器，端口 9101-9143） |

---

## 🗂️ 场景套件矩阵（tests/scenario）

每个"配置档 profile"是一个独立服务器进程：配置写入 `tests/scenario/run/<name>/config/config.toml`，以该目录为 CWD 启动编译产物二进制（应用从 CWD 读 `config/config.toml`）。全部断言走 HTTP/进程行为的黑盒观测，不依赖内部状态。用例名以 `R-<域>-NNN` 开头对账（场景 ID 见 specmark 变更 specs）。

| 套件 | 场景 ID | 用例数 | 覆盖内容 |
|------|---------|--------|----------|
| `test_embed_normal.py` | R-embed-001 ~ R-embed-008 | 8 | 嵌入服务正常场景（M1 = BAAI/bge-small-en-v1.5，384 维） |
| `test_embed_abnormal.py` | R-embed-009 ~ R-embed-010 | 5 | 嵌入服务异常场景（超长文本、非法输入等错误路径） |
| `test_auth.py` | R-auth-001 ~ R-auth-008 | 8 | 认证场景（auth 配置档，端口 9103） |
| `test_security.py` | R-auth-009 ~ R-auth-011 | 6 | 限流、白名单、路径遍历、错误脱敏、审计日志 |
| `test_server_modes.py` | R-server-001 ~ R-server-008 | 9 | 服务器模式（默认/自定义配置/MCP 模式等启动路径） |
| `test_lifecycle.py` | R-server-002 / R-auth-002 | 3 | 生命周期补缺（优雅关闭、信号处理） |
| `test_cli_mcp.py` | R-server-003（design.md M4/M5） | 8 | CLI / MCP 模式探针 |
| `test_config_device.py` | R-config-001 ~ R-config-006 | 6 | 配置与设备场景（`--config` fail-fast、环境变量覆盖、GPU 回退） |
| `test_http_matrix.py` | R-api-001 / R-embed-002/003 / R-rerank-002 | 21 | HTTP 协议矩阵（补齐 design.md M2 缺口） |
| `test_rerank.py` | R-rerank-001 ~ R-rerank-006 | 6 | 重排服务（M1 9101 英文 + M2 9102 中文双模型） |
| `test_model_matrix.py` | R-model-001（design.md M0） | 3 | 模型矩阵：4 模型 × 3 厂商 × 2 架构经 HTTP 验证 |
| `test_model_zh.py` | R-model-002 ~ R-model-007 | 6 | 模型管理（zh 配置档 = M2 服务端 HF 镜像下载，512 维） |
| `test_api_enhancements.py` | AE-*（api-config-enhancements） | 10 | API/配置增强（语义检索、模型卸载、similarity metric 等） |

合计 15 个套件、99 个场景用例（另有 `tests/perf` 的 27 个 pytest 用例，见下节）。CI 中该层为每夜定时任务（`scenario-tests.yml`，UTC 03:00），不阻塞 PR；本地用 `scripts/run-scenario-tests.sh` 一键运行。

---

## 🔩 Rust 集成与专项测试

| 文件 | 形态 | 覆盖内容 |
|------|------|----------|
| `tests/integration.rs` → `tests/integration/` | Rust 集成 | `api_test.rs`（API 面）与 `real_engine.rs`（RealTestEngine + TestMode mock/light/full） |
| `tests/grpc_e2e.rs` | 专项 E2E | gRPC 全方法矩阵：拉起真实二进制（`http,grpc` feature），tonic 客户端经 `vecboost::sdforge` 再导出访问 |
| `tests/scenario_sdk.rs` | SDK 矩阵 | Library SDK（非 HTTP）模式下的引擎/模型矩阵，按 `models/<目录>` 存在性自动 SKIP |
| `tests/doctor.rs` | 专项 | `vecboost doctor` 只读诊断（config/tokenizer/缓存持久层/线程/GPU/模型完整性，FAIL 退出码 1） |
| `tests/quantized_parity.rs` | 质量门 | GGUF 量化对齐：Q8_0 与 fp32 余弦中位数 ≥ 0.98、Q4_K ≥ 0.95（模型经 `VECBOOST_GGUF_MODEL`，语料 `tests/fixtures/golden_corpus.txt` ≥32 条中英混合） |
| `tests/model_snapshot_regression.rs` | 回归 | 模型快照回归（向量输出与快照比对） |
| `tests/common/mod.rs` | 共享夹具 | `MockEngine`（FNV-1a + LCG 确定性向量）、`create_test_engine()` |

> 已知偏离（D8）：`tests/integration.rs` 与 `tests/integration/` 子目录并存、`tests/perf.rs` 与 `tests/perf/` 子目录并存，会产生 Rust 模块系统警告；合并修复推迟到 v0.3.0。

---

## ⚡ 性能与基准测试

### tests/perf 的 Python 部分

`tests/perf/` 中同时存在两类 pytest 用例，**用标记区分**：

- `@pytest.mark.sim`：打 Python `api_simulator.py` 的语义模拟，用于快速回归 API 形状/错误路径，**不代表 Rust 服务行为**（`test_api.py` 23 例）；
- 无标记 / `@pytest.mark.real`：经 `real_service.py` 打真实 `localhost:9002`（`test_server_integration.py` 4 例）。

```bash
pytest tests/perf -m "not sim"   # 只跑真实服务用例
pytest tests/perf -m sim          # 只跑模拟器用例
```

> 修改 Rust API 语义时，`sim` 用例必须同步更新；若发现 sim 与真实行为冲突，以真实服务为准并修 sim。

### Rust 回归阈值

`tests/perf/performance_test.rs`（8 个用例）以 MockEngine 断言吞吐/延迟相对阈值，确定性、CI 可跑。

### 微基准（benches/，criterion）

| 基准 | 度量对象 |
|------|----------|
| `similarity_bench` | 单向量相似度（cosine/euclidean/dot/manhattan × 128/384/768/1024 维） |
| `batch_scheduling_bench` | 批调度（稳定/突发负载 × 等待窗口，`assemble_batch` 时间窗路径） |
| `semantic_cache_bench` | 语义缓存（精确命中/未命中、trigram 搜索规模扩展） |
| `embed_throughput_bench` | 吞吐基线（单文本 embed 与 32 文本 embed_batch；模型经 `VECBOOST_BENCH_MODEL` 指定，未设置时 skip） |

基线数字记录在 `docs/benchmarks/`（标注环境与噪声区间），调优开关注册表见 [⚡ 性能指南](PERFORMANCE.md)。

---

## 🧱 单元测试

`src/` 各模块内联 `#[cfg(test)]` 模块（grep 统计 `#[test]` / `#[tokio::test]` 约 1700+ 个），随 `cargo test --lib` 执行。测试代码允许 `unwrap`（`clippy.toml` 开启 `allow-unwrap-in-tests`），生产代码禁止。共享环境变量竞态经 `ENV_LOCK` 串行化（安全默认值测试组）。

---

## ▶️ 运行命令（与 CI 一致）

```bash
# 单元 + 集成（CI：health-check.yml test job 分 --lib 与 --tests 两步）
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --lib
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --tests

# 本地门禁（../base/* 为路径依赖活仓库，须以 -p 限定）
cargo test -p vecboost -p vecboost-examples

# gRPC E2E
cargo test -p vecboost --features http,grpc --test grpc_e2e

# 场景测试（CI：scenario-tests.yml 每夜）
cargo build -p vecboost --features http
pytest tests/scenario -q --junitxml=scenario-results.xml

# 性能回归阈值（Rust）
cargo test -p vecboost --test perf

# 微基准
cargo bench -p vecboost

# 覆盖率（CI 硬门禁：行覆盖率 ≥ 80%）
cargo tarpaulin --features "grpc,cli,auth,onnx,db,openapi,mcp" --all-targets --out lcov --out xml --output-dir coverage/
```

CI 工作流与测试的对应关系：

| 工作流 | 触发 | 测试内容 |
|--------|------|----------|
| `health-check.yml`（CI） | push/PR（main、develop） | fmt、clippy（含 unwrap_used 门禁）、构建矩阵（default/onnx/grpc）、`--lib` + `--tests`、tarpaulin 覆盖率 ≥80% 硬门禁、Windows 测试（continue-on-error）、`cargo doc` 死链、`cargo audit`、main 分支 push 跑 benchmark |
| `feature-matrix.yml` | push/PR | default/grpc/ecosystem（auth,redis,db,onnx）三档 fmt + check + clippy + build + `--lib` 测试 |
| `scenario-tests.yml` | 每夜 UTC 03:00 + 手动 | 构建 http 二进制后 `pytest tests/scenario`（无模型环境自动 SKIP 推理类用例） |
| `codeql.yml` / `docker.yml` | push/PR/每周 | CodeQL 静态分析；Docker 构建 + Trivy 扫描（CRITICAL 即失败） |

---

## 🎛️ 测试模式与夹具

`TEST_MODE` 环境变量控制测试引擎行为：

| 模式 | 行为 |
|------|------|
| `mock`（默认） | `MockEngine`（FNV-1a 哈希 + LCG）确定性向量，无需模型 |
| `light` | 尝试真实推理，失败回退 mock |
| `full` | 强制真实推理引擎（需要模型下载） |

> 断言语义相似性的测试（如"相似文本 > 不同文本"）标注 `#[ignore]`（mock 向量语义随机），需在 `TEST_MODE=light` 或 `full` 下以 `cargo test --test integration -- --ignored` 运行。

---

## 🗺️ 模型矩阵

`tests/scenario/test_model_matrix.py` 与 `tests/scenario_sdk.rs` 按 `models/<目录名>` 存在性自动 SKIP。本机 4 模型（BGE-en 384 / MiniLM 384 / BGE-zh 512 / multilingual-e5-small 384）齐全时覆盖 XlmRoberta + Bert 双架构、三种 pooling。

---

## 📌 约定

1. **新性能断言优先写回归阈值层**（MockEngine、确定性），微基准只用于优化工作的前后对比；
2. **新端点行为契约写 scenario 层**（真实服务器 + HTTP），这同时是 API 文档的活规格；
3. **基线数字记录到 `docs/benchmarks/`**，标注环境与噪声区间；
4. 测试代码允许 `unwrap`（`clippy.toml` 已豁免），生产代码禁止。

---

## 📚 相关文档

| 文档 | 说明 |
|:-----|:-----|
| [📖 用户指南](USER_GUIDE.md) | 安装、配置和使用的完整说明 |
| [🏗️ 架构文档](ARCHITECTURE.md) | 模块划分与测试架构 |
| [⚡ 性能指南](PERFORMANCE.md) | 基准数据与调优开关 |
| [🤝 贡献指南](CONTRIBUTING.md) | 质量门禁与提交规范 |
| [📋 更新日志](CHANGELOG.md) | 每个版本的变更记录 |
