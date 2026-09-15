<div align="center">

<img src="docs/image/vecboost.png" alt="VecBoost Logo" width="200"/>

[![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT) [![GitHub release](https://img.shields.io/github/v/release/Kirky-X/vecboost?style=for-the-badge)](https://github.com/Kirky-X/vecboost/releases) [![Rustc 1.85+](https://img.shields.io/badge/Rustc-1.85+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/)

**中文** | [English](README_EN.md)

**高性能、生产级嵌入向量服务，使用 Rust 编写。VecBoost 提供高效的文本向量化服务，支持多种推理引擎、GPU 加速和企业级功能。**

[✨ 功能特性](#-功能特性) • [🚀 快速开始](#-快速开始) • [📚 文档](#-文档) • [💻 示例](#-示例) • [🤝 参与贡献](#-参与贡献)

</div>

---

<div align="center">

### 🎯 写一份接口，四种协议即刻可用

接口处理函数只写一份，`sdforge` 宏在编译期生成四协议绑定，剩下交给编译器。

<table style="width:100%; border-collapse: collapse">
<tr>
<td align="center" width="25%">🌐<br><b>REST</b><br><span style="color:#64748B">Web 接入 · 默认启用</span></td>
<td align="center" width="25%">📡<br><b>gRPC</b><br><span style="color:#64748B">微服务 · 强类型调用</span></td>
<td align="center" width="25%">🤖<br><b>MCP</b><br><span style="color:#64748B">LLM 工具 · 标准输入输出</span></td>
<td align="center" width="25%">💻<br><b>CLI</b><br><span style="color:#64748B">脚本调用 · 快速验证</span></td>
</tr>
</table>

</div>

---

## 📋 目录

- [✨ 功能特性](#-功能特性)
- [🚀 快速开始](#-快速开始)
- [🔌 API 使用](#-api-使用)
- [⚙️ 配置](#️-配置)
- [📚 文档](#-文档)
- [💻 示例](#-示例)
- [🏗️ 架构](#️-架构)
- [🧪 测试](#-测试)
- [📊 性能](#-性能)
- [🔒 安全](#-安全)
- [🗺️ 开发路线图](#️-开发路线图)
- [🤝 参与贡献](#-参与贡献)
- [📋 更新日志](#-更新日志)
- [📄 许可证](#-许可证)
- [🙏 致谢](#-致谢)
- [📞 联系与支持](#-联系与支持)
- [⭐ Star 历史](#-star-历史)

---

## ✨ 功能特性

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🚀 <b>高性能</b><br><span style="color:#64748B">优化的 Rust 代码库，支持批处理与并发请求处理；Linux 下默认启用 jemalloc 全局分配器</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🔧 <b>多引擎支持</b><br><span style="color:#64748B"><code>Candle</code>（原生 Rust）和 <code>ONNX Runtime</code> 推理引擎，经 <code>EngineFactory</code> 工厂切换</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔁 <b>Rerank 重排序</b><br><span style="color:#64748B">基于 bi-encoder 的文档重排序，HTTP/gRPC/CLI 三协议支持</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌍 <b>国际化（i18n）</b><br><span style="color:#64748B">ICU+Fluent 中英双语错误响应，<code>Accept-Language</code> 请求级语言协商</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🎮 <b>GPU 加速</b><br><span style="color:#64748B">NVIDIA CUDA、Apple Metal 原生支持；<code>mkl</code>/<code>accelerate</code> CPU 加速后端 opt-in</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌐 <b>多协议接口</b><br><span style="color:#64748B">HTTP/REST、gRPC、MCP、CLI 四种接口由 <code>sdforge</code> 从单一源生成</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🧩 <b>7 库生态</b><br><span style="color:#64748B"><code>trait-kit</code>/<code>confers</code>/<code>inklog</code>/<code>oxcache</code>/<code>limiteron</code>/<code>dbnexus</code>/<code>sdforge</code> 模块化生态</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">📊 <b>智能缓存</b><br><span style="color:#64748B">基于 <code>oxcache</code> 的高性能缓存（LRU/LFU/FIFO + TTL）与语义缓存三级查询</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔐 <b>企业级安全</b><br><span style="color:#64748B">JWT 认证、CSRF 保护、基于角色的访问控制、TOTP、账号锁定与审计日志</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">⚡ <b>速率限制</b><br><span style="color:#64748B">基于 <code>limiteron</code> 的令牌桶限流（全局/IP/用户/API 密钥多维独立计数）</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">📈 <b>优先级队列</b><br><span style="color:#64748B">可配置优先级的请求队列、加权公平调度与时间窗动态拼批</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🧊 <b>Matryoshka 支持</b><br><span style="color:#64748B">动态维度约简（截断后自动重归一化），支持更小更快的嵌入向量（OpenAI 兼容）</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔍 <b>可观测性</b><br><span style="color:#64748B">Prometheus 指标、健康检查、结构化日志（inklog 控制台 + 文件轮转）</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">📦 <b>云原生部署</b><br><span style="color:#64748B">多架构 Docker 镜像（linux/amd64 + arm64）；Kubernetes 提供部署指引（清单需自备）</span></td>
</tr>
</table>

除上述核心能力外，OpenAI 兼容端点（`POST /v1/embeddings`，支持 `encoding_format=base64`）、BF16 推理与 SIMD 向量相似度、GPU 内存分页、`vecboost doctor` 只读诊断、Library SDK 集成（Library 模式）与 `config_full.toml` / `config_minimal.toml` 配置预设也已可用；端点与参数明细见 [🔌 API 使用](#-api-使用) 一节，配置项说明见 [⚙️ 配置](#️-配置) 一节。

---

## 🚀 快速开始

### 📦 安装

前置条件：

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| **Rust** | 1.91+ | edition 2024（以 `Cargo.toml` 的 `rust-version` 字段为权威值） |
| **Cargo** | 1.91+ | 随 Rust 附带 |
| **CUDA Toolkit** | 12.x | 可选，NVIDIA GPU 支持（`cuda` feature） |
| **Metal SDK** | 最新版 | 可选，Apple Silicon GPU 支持（`metal` feature） |
| **protobuf-compiler** | 最新版 | 可选，gRPC E2E 测试需要 |

> **💡 提示**: 运行 `rustc --version` 验证 Rust 安装。

```bash
# 1. 克隆仓库
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost

# 2. 默认构建（http feature，含 OpenAPI 文档）
cargo build --release

# 3. 构建 GPU 支持
#    Linux (CUDA):
cargo build --release --features cuda
#    macOS (Metal):
cargo build --release --features metal

# 4. 构建多协议接口（HTTP + gRPC + CLI）
cargo build --release --features grpc,cli

# 5. 构建 MCP 接口（stdio 模式，--mcp 启动）
cargo build --release --features mcp

# 6. 构建 CI 全特性组合（数据库 + 认证 + ONNX + OpenAPI + 全协议）
cargo build --release --features grpc,cli,auth,onnx,db,openapi,mcp
```

最小构建：`cargo build --no-default-features --features http`。

配置并运行：

```bash
# 复制并自定义配置（默认从 config/config.toml 读取）
cp config/config.toml config/config_custom.toml
# 编辑 config/config_custom.toml

# 使用默认配置运行
./target/release/vecboost

# 使用自定义配置（--config，CLI 子命令模式下须写在子命令之前）
./target/release/vecboost --config config/config_custom.toml
```

> **✅ 成功**: 服务默认在 `http://127.0.0.1:9002` 启动（安全默认仅监听回环地址）。

> **🐳 Docker**：`docker build -t vecboost:latest .` 后挂载 `config/` 与 `models/` 运行即可；Docker Compose 与 Kubernetes 部署见 [📖 用户指南 · Docker 部署](docs/USER_GUIDE.md#-docker-部署)。

### 💡 最小示例

以下示例改编自 [`examples/http/embed_api.rs`](examples/http/embed_api.rs)，通过 HTTP 生成嵌入向量（完整端点见 [📘 API 参考](docs/API_REFERENCE.md)）：

```bash
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

响应：

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

也可以直接使用 CLI（`cli` feature）或 library SDK（`library` 模式）：

```bash
# 单文本嵌入
cargo run --features cli -- embed --text "Hello, world!"
```

### 🧭 核心概念

- **模型与引擎**：`ModelConfig` 声明 HuggingFace 模型（默认 `BAAI/bge-small-en-v1.5`），`EngineFactory::create(engine_type, config)` 创建 `Candle`（默认）或 `ONNX`（`onnx` feature）引擎；支持 Bert / XlmRoberta 双架构与 mean/cls/max 池化。
- **四协议单一源**：`src/api/embedding.rs` 中的处理函数经 `#[forge(...)]` 宏标注，由 `sdforge` 生成 HTTP/gRPC/MCP/CLI 绑定，禁止手写协议代码。
- **7 库生态**：`trait-kit` 以 typestate 模块注册中心（`Kit<Unbuilt> → Kit<Ready>`）装配全部模块；`confers` 接管配置、`inklog` 日志、`oxcache` 缓存、`limiteron` 限流、`dbnexus` 持久化（`db` feature）、`sdforge` 接口生成。
- **配置优先级**：TOML 文件 + `VECBOOST_` 前缀环境变量覆盖（敏感项 `VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD` 必须走环境变量）；配置文件变更校验并打日志，重启后生效。
- **特性门控**：全部可选能力均为独立 feature（见 [🏷️ Feature 标志](#️-feature-标志)），最小构建只含 HTTP 服务。

---

## 🔌 API 使用

VecBoost 由 `sdforge` 从 `src/api/embedding.rs` 单一源生成四种协议接口。全部端点、参数、请求/响应示例、gRPC 方法表与消息类型见 [📘 API 参考](docs/API_REFERENCE.md)，概要如下：

- **HTTP/REST**：`/api/1/*` 提供嵌入（单文本/批量/文件）、相似度、语义检索、重排序、模型管理与健康检查端点；
- **OpenAI 兼容**：`POST /v1/embeddings`，响应遵循 OpenAI 格式（`object` / `data` / `usage`），支持 `encoding_format=base64`；
- **Matryoshka 维度约简**：`/v1/embeddings` 传 `dimensions`（256/512/1024 等）换取更小更快的向量，截断后自动 L2 重归一化保证余弦相似度正确；
- **gRPC**：`grpc` feature 在 50051 端口（可配置）暴露 13 个 `vecboost.*` 方法（sdforge 统一 Call 协议，无需手写 proto），JWT 认证、限流、最大连接数与超时均可配置；
- **MCP**：`mcp` feature 以 stdio 模式（`vecboost --mcp`）向 LLM 暴露 `embed` / `embed_batch` / `similarity` / `list_models` 工具；
- **CLI**：`cli` feature 提供 embed / batch / similarity / rerank 子命令（见 [💡 最小示例](#-最小示例)）；
- **推理引擎**：Candle（原生 Rust，默认）与 ONNX Runtime（`onnx` feature），经 `EngineFactory::create` 工厂切换；
- **可观测性与运维**：`/metrics`（Prometheus 指标）、`/health`（存活探针）与 `/health?depth=full`（真实就绪探测）、`/api-docs`（Swagger UI）；只读诊断 `vecboost doctor`（config / tokenizer / 缓存 / 线程 / GPU / 模型完整性，FAIL 退出码 1）。

交互式 OpenAPI 文档：`http://localhost:9002/api-docs`（Swagger UI）与 `/api-docs/openapi.json`（规范 JSON，需 `openapi` feature；ReDoc 推迟到 v0.3.0）。分阶段指标（拼批/去重/分段延迟等）见 [⚡ 性能指南 · 新增指标](docs/PERFORMANCE.md#-新增指标)。

### 🏷️ Feature 标志

下表逐项对应 `Cargo.toml` 的 `[features]` 定义，`default = ["http"]`。

| Feature | 默认 | 说明 |
|---------|------|------|
| `http` | ✅ | HTTP/REST API + OpenAPI 文档 + Prometheus 指标 |
| `grpc` | - | gRPC 服务器（sdforge `#[forge(grpc_method)]` 生成） |
| `cli` | - | CLI 命令行工具 |
| `mcp` | - | MCP 协议接口（LLM 工具集成，stdio 模式） |
| `openapi` | - | OpenAPI/Swagger UI 文档（独立于 `http` 启用） |
| `schema` | - | OpenAPI Schema 派生（`http`/`openapi` 自动启用；支持 library 模式类型导出） |
| `db` | - | dbnexus 数据库持久化（SQLite） |
| `postgres` | - | PostgreSQL 支持（含 `db`） |
| `auth` | - | JWT 认证 + CSRF + RBAC + AES-256-GCM 加密 |
| `cuda` | - | NVIDIA CUDA GPU 加速 |
| `metal` | - | Apple Silicon Metal GPU |
| `onnx` | - | ONNX Runtime 引擎 |
| `mkl` | - | x86_64 CPU MKL 加速后端（opt-in，需链接正常的工具链） |
| `accelerate` | - | aarch64 macOS Accelerate 加速后端（opt-in） |
| `quantized-gguf` | - | GGUF 量化引擎开关（推理后端待 candle 上游落地） |

> **📦 内置依赖说明**: `confers`（配置）、`inklog`（日志）、`oxcache`（缓存）、`limiteron`（限流）、`trait-kit`（模块注册）为必选依赖，始终启用，无需通过 feature 开启。`sdforge` 在 `http`/`grpc`/`cli`/`mcp` 任一协议 feature 下启用。

---

## ⚙️ 配置

默认从 `config/config.toml` 读取（`--config <path>` 指定其他路径，路径不存在时 fail-fast 报错退出；预置 `config_full.toml` / `config_minimal.toml` 示例）。环境变量以 `VECBOOST_` 前缀覆盖配置文件，敏感项（`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`）必须走环境变量；配置文件变更会校验并打日志，重启后生效。

全部配置段（server / model / embedding / rerank / monitoring / auth / rate_limit / audit / database / logging / pipeline.worker / semantic_cache / device）的逐项键位、默认值、环境变量全表与完整示例配置见 [📖 用户指南 · 配置](docs/USER_GUIDE.md#️-配置)，也可直接查看 [`config/config.toml`](config/config.toml)。

> **⚠️ 注意**：`[flow_control]` 与 `[cache]` 两个 TOML 段当前版本不解析（历史遗留段名）；限流走 `[rate_limit]`，缓存走 `[embedding]` 与 `[semantic_cache]`。见 [❓ FAQ](docs/FAQ.md#️-配置与部署)。

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| [📖 用户指南](docs/USER_GUIDE.md) | 从安装到进阶的完整使用教程（含部署选项） |
| [📘 API 参考](docs/API_REFERENCE.md) | REST / gRPC / OpenAI 兼容接口的完整说明 |
| [🏗️ 架构文档](docs/ARCHITECTURE.md) | 设计原则、模块划分与数据流 |
| [⚡ 性能指南](docs/PERFORMANCE.md) | 基准数据、调优开关注册表与实验纪律 |
| [🔒 安全文档](docs/SECURITY.md) | 安全设计、支持版本与漏洞报告流程 |
| [❓ FAQ](docs/FAQ.md) | 常见问题解答 |
| [🧪 测试场景矩阵](docs/TEST_SCENARIOS.md) | 测试栈职责划分与场景穷举矩阵 |
| [📋 更新日志](docs/CHANGELOG.md) | 每个版本的变更记录 |
| [🤝 贡献指南](docs/CONTRIBUTING.md) | 如何参与项目开发 |
| [📈 基准数据归档](docs/benchmarks/) | 历史基准数据（相似度/批调度/语义缓存/GPU 管线） |
| [🌍 I18N 缺失审计](docs/I18N_MISSING_AUDIT.md) | 国际化翻译键审计记录 |

---

## 💻 示例

全部示例位于 [`examples/`](examples/) 目录，作为独立 workspace member crate `vecboost-examples`（13 个分类、30 个可执行二进制），覆盖基础嵌入、HTTP/CLI 调用、引擎切换、认证、缓存、限流、监控、审计、语义缓存与 Library SDK 集成；逐分类清单见 [`examples/README.md`](examples/README.md)。

```bash
# 运行单个示例
cargo run -p vecboost-examples --bin embed
cargo run -p vecboost-examples --bin library_usage
cargo run -p vecboost-examples --bin matryoshka

# ONNX 引擎示例（需要 ONNX Runtime）
cargo run -p vecboost-examples --bin onnx --features onnx
```

---

## 🏗️ 架构

VecBoost 采用模块化生态架构：`trait-kit` 以 typestate 模块注册中心（`Kit<Unbuilt> → Kit<Ready>`）装配 17 个模块，`sdforge` 从 `src/api/embedding.rs` 单一源生成四协议绑定，推理经 `EngineFactory` 抽象到 Candle / ONNX 引擎，请求经优先级队列与时间窗拼批进入推理管线。

7 库生态（trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge）的版本与分工、模块依赖图、数据流、缓存/安全/部署架构与扩展点说明见 [🏗️ 架构文档](docs/ARCHITECTURE.md)。

---

## 🧪 测试

### 🎯 测试策略

测试栈分六层：`src/` 内联单元测试、`tests/integration/` 集成测试、专项集成（doctor / gRPC E2E / 模型快照回归 / 量化质量门 / SDK 矩阵）、`tests/scenario/*.py` 真实服务场景测试（15 个 pytest 套件）、`tests/perf/` 性能回归阈值与 `benches/` 的 4 组 Criterion 微基准。`TEST_MODE` 环境变量控制测试引擎（`mock` 默认 / `light` / `full`）。各层职责、场景穷举矩阵与 CI 工作流对应关系见 [🧪 测试场景矩阵](docs/TEST_SCENARIOS.md)。

### ▶️ 运行命令（与 CI 一致）

以下命令提取自 `.github/workflows/health-check.yml`（CI）、`feature-matrix.yml`、`scenario-tests.yml` 与 `docs/CONTRIBUTING.md`。路径依赖提示：`../base/*` 生态库是路径依赖的活仓库，本地门禁命令须以 `-p vecboost -p vecboost-examples` 限定。

```bash
# 格式与 Lint 门禁（CI：clippy unwrap_used 为生产代码 panic 面门禁）
cargo fmt --all -- --check
cargo clippy --features "grpc,cli,auth,onnx,db,openapi,mcp" --all-targets -- -D warnings -W clippy::unwrap_used

# 全特性编译检查（feature-matrix）
cargo check --features "grpc,cli,auth,onnx,db,openapi,mcp"

# 单元 + 集成测试（CI 分 --lib 与 --tests 两步）
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --lib
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --tests

# gRPC E2E（拉起真实二进制）
cargo test -p vecboost --features http,grpc --test grpc_e2e

# 场景测试（pytest，conftest 自动拉起真实服务器；models/ 缺席时推理用例自动 SKIP）
cargo build -p vecboost --features http
pytest tests/scenario -q --junitxml=scenario-results.xml

# Python 性能测试（sim 标记区分模拟器用例）
pytest tests/perf -m "not sim"   # 只跑真实服务用例
pytest tests/perf -m sim          # 只跑模拟器用例

# 覆盖率（CI 硬门禁：行覆盖率 ≥ 80%，tarpaulin）
cargo tarpaulin --features "grpc,cli,auth,onnx,db,openapi,mcp" --all-targets --out lcov --out xml --output-dir coverage/

# 基准测试（CI benchmark job）
cargo bench --features "grpc,cli,auth,onnx,db,openapi,mcp"

# 文档构建与死链检查
cargo doc --workspace --no-deps

# 依赖安全审计
cargo audit
```

### 📊 测试规模

截至 v0.2.1 工作区：单元测试（`src/` 内联）约 1700+、Rust 集成/专项测试（`tests/*.rs`）61 个、Python 场景/性能用例 126 个（15 个场景套件）、Criterion 微基准 4 组；CI 硬门禁为行覆盖率不低于 80%（tarpaulin），Python 场景测试为每夜定时任务（UTC 03:00）不阻塞 PR。逐项统计与场景矩阵见 [🧪 测试场景矩阵](docs/TEST_SCENARIOS.md)。

---

## 📊 性能

基准数据来自 `docs/benchmarks/` 实测归档（criterion，2026-08 采集，Linux x86_64，噪声约 ±5-10%）：SIMD 向量相似度较标量最高 **3.06x** 加速（1024 维 cosine 约 341.7 ns），`ContinuousBatchLoop` 连续批调度较固定等待 **5.6x**（100 请求稳定负载 1.110s vs 6.194s），语义缓存精确命中约 10 ns；吞吐基线（`embed_throughput_bench`）需本地模型，**待实测**。完整基准表、性能设计要点（时间窗拼批/批内去重/SIMD/线程调优/jemalloc）、GGUF 量化与调优开关注册表见 [⚡ 性能指南](docs/PERFORMANCE.md)，微基准可用 `cargo bench` 复现（命令见上文[测试](#-测试)一节）。

---

## 🔒 安全

### 🛡️ 安全设计

VecBoost 默认安全：出厂仅回环绑定，`auth.enabled=false` 时绑定非回环地址拒绝启动（逃生阀 `VECBOOST_ALLOW_INSECURE=1` 打 ERROR 告警）；认证授权基于 garrison（JWT + CSRF + RBAC admin 角色 + TOTP + 账号锁定），并覆盖 XFF 信任反转、文件路径白名单、输入长度校验、AES-256-GCM 配置加密、审计日志与 i18n 双语错误脱敏。逐项机制的代码级细节见 [🔒 安全文档](docs/SECURITY.md)。

### ⛓️ 供应链与门禁

`cargo audit`、`cargo deny check`、CodeQL、Trivy/Checkov 镜像扫描、gitleaks 私密信息扫描与 pre-commit 钩子在 CI 与本地双重执行，完整清单与处置策略见 [🔒 安全文档 · 供应链与安全门禁](docs/SECURITY.md#️-供应链与安全门禁)。

### 🚨 报告安全漏洞

请勿通过公开 issue 报告安全漏洞，请联系 maintainer：<kirky-x@outlook.com>。依赖 advisory 由 CI `cargo-audit` 门禁。完整政策与支持版本见 [SECURITY.md](docs/SECURITY.md)。

---

## 🗺️ 开发路线图

<table style="width:100%; border-collapse: collapse">
<tr><th style="text-align:center">状态</th><th style="text-align:left">方向</th><th style="text-align:left">条目</th></tr>
<tr><td align="center">✅</td><td>核心服务</td><td>四协议单一源生成、Candle/ONNX 双引擎、Bert/XlmRoberta 双架构、优先级队列与时间窗拼批</td></tr>
<tr><td align="center">✅</td><td>生态集成</td><td>7 库生态接线（trait-kit 注册中心、confers 配置、inklog 日志、oxcache 缓存、limiteron 限流、dbnexus 持久化、sdforge 接口）</td></tr>
<tr><td align="center">✅</td><td>安全与 i18n</td><td>JWT/CSRF/RBAC/TOTP、安全默认值收敛、审计日志、ICU+Fluent 双语错误</td></tr>
<tr><td align="center">✅</td><td>性能基座</td><td>SIMD 相似度、连续批处理、语义缓存、GPU 内存分页、BF16 推理、Matryoshka 降维</td></tr>
<tr><td align="center">🚧</td><td>审计修复（Unreleased）</td><td>安全默认值/登录收敛/XFF 信任反转/RBAC 接线/缓存键模型命名空间等破坏性行为变更（见 [更新日志](#-更新日志)）</td></tr>
<tr><td align="center">🚧</td><td>调优开关（Unreleased）</td><td>GGUF 量化、向量输出量化、多模型 LFRU 驻留、缓存 WAL、硬件感知规划、doctor 诊断、启动预热</td></tr>
<tr><td align="center">📋</td><td>量化推理后端</td><td>GGUF 推理后端待 candle-transformers 上游 quantized BERT 落地（路由/魔数校验/质量门脚手架已就绪）</td></tr>
<tr><td align="center">📋</td><td>多副本会话外置</td><td>auth 会话外置需 garrison db 后端补齐（pool-backed DAO）</td></tr>
<tr><td align="center">📋</td><td>性能基线补全</td><td><code>embed_throughput_bench</code> 吞吐基线待实测；MKL/Accelerate 在链接正常工具链上的对比基线</td></tr>
<tr><td align="center">📋</td><td>文档与可观测性</td><td>ReDoc 文档（v0.3.0）、Grafana 预配置仪表板</td></tr>
</table>

---

## 🤝 参与贡献

详细的贡献流程与代码规范请参阅 [🤝 贡献指南](docs/CONTRIBUTING.md)。

### 🛠️ 开发环境

工具链为 Rust 1.91+（`Cargo.toml` `rust-version` 为权威值）与 Python ≥ 3.10 + pytest（可选 protobuf-compiler、docker）；提交前须通过 fmt / clippy（`unwrap_used` panic 面门禁）/ 测试 / `scripts/doc_consistency_check.py` 四道质量门禁，Git 钩子经 [pre-commit](https://pre-commit.com/)（`.pre-commit-config.yaml` → `scripts/pre-commit.sh`）自动执行；提交信息遵循 Conventional Commits，行为变更须在 CHANGELOG `Unreleased` 段登记并同步双语 README。环境搭建、构建组合与质量门禁命令见 [🤝 贡献指南](docs/CONTRIBUTING.md)。

### 💖 贡献方式

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="33%" align="center" style="padding: 16px">

### 🐛 报告 Bug

发现问题？<br>
<a href="https://github.com/Kirky-X/vecboost/issues/new">创建 Issue</a>

</td>
<td width="33%" align="center" style="padding: 16px">

### 💡 功能建议

有好想法？<br>
<a href="https://github.com/Kirky-X/vecboost/issues/new">发起讨论</a>

</td>
<td width="33%" align="center" style="padding: 16px">

### 🔧 提交 PR

想贡献代码？<br>
<a href="https://github.com/Kirky-X/vecboost/pulls">Fork 并提交 PR</a>

</td>
</tr>
</table>

---

## 📋 更新日志

完整版本历史见 [📋 更新日志](docs/CHANGELOG.md)（遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，语义化版本）。

| 版本 | 日期 | 要点 |
|------|------|------|
| Unreleased | - | 审计修复与调优开关：安全默认值收敛、HF tokenizers 全平台统一、时间窗拼批/批内去重、GGUF 量化路径、语义缓存比较模式、多模型 LFRU、缓存 WAL、doctor 诊断、启动预热 |
| 0.2.1 | 2026-09-06 | i18n 国际化（114 个翻译键）、Rerank 重排序三协议、语义缓存三级查询、BF16 精度、SIMD 相似度、连续批处理调度、GPU 内存分页、Library 模式 |
| 0.2.0 | 2026-07-24 | sdforge 四协议统一生成、7 库生态接线、Matryoshka 截断重归一化、vuln-0009 repo_id 校验 |
| 0.1.0 | 2025-12-15 | VecBoost 初始发布 |

Unreleased 含多项破坏性行为变更（安全默认值收敛、登录收敛、XFF 信任反转、RBAC 接线、缓存键/分词器变更等），**升级必读**：逐项「旧行为 → 新行为 → 迁移动作」对照表见 [📋 更新日志 · Unreleased](docs/CHANGELOG.md#unreleased)。多副本边界（auth 会话存进程内存，仅限单副本）与热重载语义（配置变更重启后生效）见 [❓ FAQ](docs/FAQ.md#️-配置与部署)。

---

## 📄 许可证

本项目采用 **MIT 许可证** - 查看 [LICENSE](LICENSE) 文件了解更多。

---

## 🙏 致谢

### 🌟 核心依赖

VecBoost 站在以下优秀开源项目的肩膀上：

| 依赖 | 用途 |
|------|------|
| [candle](https://github.com/huggingface/candle) | 原生 Rust ML 推理框架（默认引擎） |
| [tokenizers](https://github.com/huggingface/tokenizers) | HuggingFace 分词器（全平台统一） |
| [hf-hub](https://crates.io/crates/hf-hub) | HuggingFace Hub 模型下载 |
| [trait-kit](https://crates.io/crates/trait-kit) | 模块注册中心与 typestate 依赖管理 |
| [confers](https://crates.io/crates/confers) | 配置加载（TOML + 环境变量 + 校验） |
| [inklog](https://crates.io/crates/inklog) | 结构化日志基础设施 |
| [oxcache](https://crates.io/crates/oxcache) | 高性能缓存后端 |
| [limiteron](https://crates.io/crates/limiteron) | 令牌桶限流器 |
| [dbnexus](https://crates.io/crates/dbnexus) | 数据库持久化（`db` feature） |
| [sdforge](https://crates.io/crates/sdforge) | 多协议接口生成 |
| [garrison](https://crates.io/crates/garrison) | 认证与安全加固（`auth` feature） |
| [axum](https://github.com/tokio-rs/axum) | HTTP 框架（由 sdforge 生成） |
| [tokio](https://github.com/tokio-rs/tokio) | 异步运行时 |
| [utoipa](https://github.com/juhaku/utoipa) | OpenAPI 文档 |
| [prometheus](https://github.com/tikv/rust-prometheus) | 指标导出 |
| [criterion](https://github.com/bheisler/criterion.rs) | 基准测试 |
| [tikv-jemallocator](https://github.com/tikv/jemallocator) | jemalloc 全局分配器（Linux glibc） |

### 💝 特别感谢

感谢 Rust 社区、Hugging Face（模型与分词器生态）与所有[贡献者](https://github.com/Kirky-X/vecboost/graphs/contributors)。

---

## 📞 联系与支持

<table style="width:100%; max-width: 600px">
<tr>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost/issues"><b style="color:#991B1B">Issues</b></a><br>
<span style="color:#64748B">报告问题和 Bug</span>
</td>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost/issues"><b style="color:#1E40AF">讨论区</b></a><br>
<span style="color:#64748B">提问和分享想法</span>
</td>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost"><b style="color:#1E293B">GitHub</b></a><br>
<span style="color:#64748B">查看源代码</span>
</td>
</tr>
</table>

---

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Kirky-X/vecboost&type=Date)](https://star-history.com/#Kirky-X/vecboost&Date)

如果这个项目对您有帮助，请考虑给它一个 ⭐️！

**由 Kirky.X 构建**

---

<sub>© 2026 Kirky.X. 保留所有权利。</sub>
