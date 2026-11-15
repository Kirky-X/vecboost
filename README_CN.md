<div align="center">

<img src="docs/assets/vecboost.png" alt="VecBoost Logo" width="240"/>

[![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![Rustc 1.91+](https://img.shields.io/badge/Rustc-1.91+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/) [![release](https://img.shields.io/github/v/release/Kirky-X/vecboost?style=for-the-badge)](https://github.com/Kirky-X/vecboost/releases) [![CI](https://img.shields.io/github/actions/workflow/status/Kirky-X/vecboost/feature-matrix.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/Kirky-X/vecboost/actions/workflows/feature-matrix.yml) [![security](https://img.shields.io/github/actions/workflow/status/Kirky-X/vecboost/codeql.yml?branch=main&label=security&style=for-the-badge)](https://github.com/Kirky-X/vecboost/actions/workflows/codeql.yml) [![coverage](https://img.shields.io/codecov/c/github/Kirky-X/vecboost?style=for-the-badge)](https://codecov.io/gh/Kirky-X/vecboost) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

[English](README.md) | 中文

**高性能、生产级多功能推理服务，使用 Rust 编写。VecBoost 提供文本嵌入、重排序、决策问答、翻译、语言检测与语音转录六类推理能力，支持多种推理引擎、GPU 加速和企业级功能。**

[✨ 功能特性](#-功能特性) • [🚀 快速开始](#-快速开始) • [📚 文档](#-文档) • [💻 示例](#-示例) • [🤝 参与贡献](#-参与贡献)

</div>

---

<div align="center">

### 🎯 六类推理能力，一套服务交付

文本嵌入、重排序、决策问答、翻译、语言检测与语音转录，统一经 `EngineFactory` 装载模型，共享配置、缓存与限流设施。

<table style="width:100%; border-collapse: collapse">
<tr>
<td align="center" width="33%">🧠<br><b>文本嵌入</b><br><span style="color:#64748B">批量向量 · Matryoshka 约简 · OpenAI 兼容</span></td>
<td align="center" width="33%">🔁<br><b>重排序</b><br><span style="color:#64748B">bi-encoder 文档相关性排序</span></td>
<td align="center" width="33%">🎲<br><b>决策问答</b><br><span style="color:#64748B">choice / score / noul 三题型</span></td>
</tr>
<tr>
<td align="center" width="33%">🌐<br><b>翻译</b><br><span style="color:#64748B">Marian 多语言对 · auto 源语检测</span></td>
<td align="center" width="33%">🔤<br><b>语言检测</b><br><span style="color:#64748B">置信度与可靠性判定</span></td>
<td align="center" width="33%">🎙️<br><b>语音转录</b><br><span style="color:#64748B">Qwen3-ASR · VAD 切分</span></td>
</tr>
</table>

</div>

---

## 📋 目录

<details open>
<summary>📑 目录</summary>

- [✨ 功能特性](#-功能特性)
- [🚀 快速开始](#-快速开始)
- [🔌 API 使用](#-api-使用)
- [🔧 配置](#-配置)
- [📚 文档](#-文档)
- [💻 示例](#-示例)
- [🧱 架构](#-架构)
- [🧪 测试](#-测试)
- [📊 性能](#-性能)
- [🔒 安全](#-安全)
- [🧭 开发路线图](#-开发路线图)
- [🤝 参与贡献](#-参与贡献)
- [📋 更新日志](#-更新日志)
- [📄 许可证](#-许可证)
- [🙏 致谢](#-致谢)
- [📞 联系与支持](#-联系与支持)

</details>

---

## ✨ 功能特性

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🚀 <b>高性能</b><br><span style="color:#64748B">优化的 Rust 代码库，支持批处理与并发请求处理；Linux 下默认启用 jemalloc 全局分配器</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🔧 <b>多引擎支持</b><br><span style="color:#64748B"><code>ONNX Runtime</code>（默认）与 <code>Candle</code>（opt-in 回退）推理引擎，经 <code>EngineFactory</code> 工厂切换</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔁 <b>Rerank 重排序</b><br><span style="color:#64748B">基于 bi-encoder 的文档重排序，HTTP/gRPC/CLI 三协议支持</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌍 <b>国际化（i18n）</b><br><span style="color:#64748B">ICU+Fluent 中英双语错误响应，<code>Accept-Language</code> 请求级语言协商</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🌐 <b>翻译与语言检测</b><br><span style="color:#64748B">Marian/opus-mt 多语言对翻译（(src,tgt) 路由 + auto 检测）与 whatlang 语言检测，<code>onnx</code> feature</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🎮 <b>决策问答</b><br><span style="color:#64748B">Laya 决策管线：choice / score / noul 三题型，带温度校准的概率输出</span></td>
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
<td width="50%" style="vertical-align:top; padding: 12px">📦 <b>云原生部署</b><br><span style="color:#64748B">Docker 镜像（`linux/amd64`，默认单架构构建）；Kubernetes 提供部署指引（清单需自备）</span></td>
</tr>
</table>

除上述核心能力外，OpenAI 兼容端点（`POST /v1/embeddings`，支持 `encoding_format=base64`）、BF16 推理与 SIMD 向量相似度、GPU 内存分页、`vecboost doctor` 只读诊断、Library SDK 集成（Library 模式）与 `config_full.toml` / `config_minimal.toml` 配置预设也已可用；完整 Feature 标志矩阵见 [🏷️ Feature 标志](docs/cn/USER_GUIDE.md#-feature-标志)，端点与参数明细见 [🔌 API 使用](#-api-使用) 一节。

---

## 🚀 快速开始

### 📦 安装

前置条件：

<details>
<summary>📋 表格</summary>

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| **Rust** | 1.91+ | edition 2024（以 `Cargo.toml` 的 `rust-version` 字段为权威值） |
| **Cargo** | 1.91+ | 随 Rust 附带 |
| **CUDA Toolkit** | 12.x | 可选，NVIDIA GPU 支持（`cuda` feature） |
| **Metal SDK** | 最新版 | 可选，Apple Silicon GPU 支持（`metal` feature） |
| **protobuf-compiler** | 最新版 | 可选，gRPC E2E 测试需要 |

</details>

> **💡 提示**: 运行 `rustc --version` 验证 Rust 安装。

<details>
<summary>📄 代码 · bash</summary>

```bash
# 1. 克隆仓库
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost

# 2. 默认构建（http + onnx feature）
cargo build --release

# 3. 运行（默认从 config/config.toml 读取）
./target/release/vecboost
```

</details>

最小构建：`cargo build --no-default-features --features http`。更多构建组合（GPU / 多协议 / MCP / CI 全特性）见 [📖 用户指南 · 安装](docs/cn/USER_GUIDE.md#-安装)。

配置并运行：

<details>
<summary>📄 代码 · bash</summary>

```bash
# 复制并自定义配置（默认从 config/config.toml 读取）
cp config/config.toml config/config_custom.toml
# 编辑 config/config_custom.toml

# 使用自定义配置（--config；参数在 clap 解析前被剥离，写在子命令前后均可）
./target/release/vecboost --config config/config_custom.toml
```

</details>

> **✅ 成功**: 按出厂 `config/config.toml`（`host = "127.0.0.1"`，端口 `9002`）启动时，服务监听 `http://127.0.0.1:9002`（仅本机可访问）；当配置未指定 `host` 时，代码默认 `0.0.0.0`（监听所有接口，供容器/局域网部署——改配置 `[server] host` 即可覆盖）。

> **🐳 Docker**：`docker build -t vecboost:latest .` 后挂载 `config/` 与 `models/` 运行即可；Docker Compose 与 Kubernetes 部署见 [📖 用户指南 · Docker 部署](docs/cn/USER_GUIDE.md#-docker-部署)。

### 💡 最小示例

以下示例改编自 [`examples/http/embed_api.rs`](examples/http/embed_api.rs)，通过 HTTP 生成嵌入向量（完整端点见 [📘 API 参考](docs/cn/API_REFERENCE.md)）：

<details>
<summary>📄 代码 · bash</summary>

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

</details>

响应：

<details>
<summary>📄 代码 · json</summary>

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

</details>

决策问答、翻译、语言检测、转录等其余端点的请求/响应示例见 [📘 API 参考](docs/cn/API_REFERENCE.md)；也可以直接使用 CLI（`cli` feature）或 library SDK（library 模式），子命令与工具清单见 [💻 CLI 工具](docs/cn/API_REFERENCE.md#-cli-工具)。

---

## 🔌 API 使用

VecBoost 由 `sdforge` 从处理函数单一源生成四种协议接口（HTTP/gRPC/MCP/CLI，`src/api/` 各域一份单一源）；全部端点、参数、请求/响应示例、gRPC 方法表、MCP 工具清单与错误码见 [📘 API 参考](docs/cn/API_REFERENCE.md)，交互式 OpenAPI 文档见 [📚 OpenAPI 文档](docs/cn/API_REFERENCE.md#-openapi-文档)（`/swagger-ui/` 与 `/redoc`，需 `swagger-ui` feature）。

---

## 🔧 配置

默认从 `config/config.toml` 读取（`--config <path>` 指定其他路径，路径不存在时 fail-fast 报错退出；预置 `config_full.toml` / `config_minimal.toml` 示例）。环境变量以 `VECBOOST_` 前缀覆盖配置文件，敏感项（`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`）必须走环境变量；配置文件变更会校验并打日志，重启后生效。

全部配置段（server / model / embedding / rerank / monitoring / auth / rate_limit / audit / database / logging / pipeline.worker / semantic_cache / device / translation / transcribe）的逐项键位、默认值、环境变量全表与完整示例配置见 [📖 用户指南 · 配置](docs/cn/USER_GUIDE.md#-配置)，也可直接查看 [`config/config.toml`](config/config.toml)。

> **⚠️ 注意**：`[flow_control]` 与 `[cache]` 两个 TOML 段当前版本不解析（历史遗留段名）；限流走 `[rate_limit]`，缓存走 `[embedding]` 与 `[semantic_cache]`。见 [❓ FAQ](docs/cn/FAQ.md#-配置与部署)。

---

## 📚 文档

<details>
<summary>📋 表格</summary>

| 文档 | 说明 |
|------|------|
| [📖 用户指南](docs/cn/USER_GUIDE.md) | 从安装到进阶的完整使用教程（含部署选项） |
| [📘 API 参考](docs/cn/API_REFERENCE.md) | REST / gRPC / OpenAI 兼容接口的完整说明 |
| [🏗️ 架构文档](docs/cn/ARCHITECTURE.md) | 设计原则、模块划分与数据流 |
| [⚡ 性能指南](docs/cn/PERFORMANCE.md) | 基准数据、调优开关注册表与实验纪律 |
| [🔒 安全文档](docs/cn/SECURITY.md) | 安全设计、支持版本与漏洞报告流程 |
| [❓ FAQ](docs/cn/FAQ.md) | 常见问题解答 |
| [🧪 测试场景矩阵](docs/cn/TEST_SCENARIOS.md) | 测试栈职责划分与场景穷举矩阵 |
| [📋 更新日志](docs/cn/CHANGELOG.md) | 每个版本的变更记录 |
| [🤝 贡献指南](docs/cn/CONTRIBUTING.md) | 如何参与项目开发 |
| [🧹 维护规范](docs/cn/MAINTENANCE.md) | 跨变更的长期约束：转录消费面、CLI 契约、退役配置项纪律、废弃文档治理 |

</details>

> **ℹ️ 说明**：`docs/benchmarks/`（基准归档）、`docs/reviews/`（评审台账）与 `docs-transcribe/` 均为**仅本机保留**目录（见 `.gitignore`），不入 git，干净 clone 中不存在。基准关键数字已内联于 [⚡ 性能指南](docs/cn/PERFORMANCE.md)。

---

## 💻 示例

全部示例位于 [`examples/`](examples/) 目录，作为独立 workspace member crate `vecboost-examples`（12 个分类、31 个可执行二进制），覆盖基础嵌入、HTTP/CLI 调用、引擎切换、认证、缓存、限流、监控、审计、语义缓存与 Library SDK 集成；逐分类清单见 [`examples/README.md`](examples/README.md)。

---

## 🧱 架构

VecBoost 采用模块化生态架构：`trait-kit` 以 typestate 模块注册中心（`Kit<Unbuilt> → Kit<Ready>`）装配模块（`src/registry/` 共 20 个生产 `ModuleMeta` 实现，另有 3 个测试夹具；部分随 `auth` 等 feature 门控），`sdforge` 从 `src/api/embedding.rs` 单一源生成四协议绑定，推理经 `EngineFactory` 抽象到 Candle / ONNX 引擎，请求经优先级队列与时间窗拼批进入推理管线。

7 库生态（trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge）的版本与分工、模块依赖图、数据流、缓存/安全/部署架构与扩展点说明见 [🏗️ 架构文档](docs/cn/ARCHITECTURE.md)。

---

## 🧪 测试

测试栈分六层：`src/` 内联单元测试、`tests/integration/` 集成测试、专项集成（doctor / gRPC E2E / 模型快照回归 / 量化质量门 / SDK 矩阵）、`tests/scenario/*.py` 真实服务场景测试（16 个 pytest 套件）、`tests/perf/` 性能回归阈值与 `benches/` 的 6 组 Criterion 微基准。分层职责、场景穷举矩阵与运行命令（与 CI 一致）见 [🧪 测试场景矩阵](docs/cn/TEST_SCENARIOS.md)，环境搭建与质量门禁见 [🤝 贡献指南](docs/cn/CONTRIBUTING.md)。

---

## 📊 性能

基准数据来自 `docs/benchmarks/` 实测归档（criterion；该目录为**仅本机保留**，见 `.gitignore`，干净 clone 中不存在）：SIMD 向量相似度较标量最高 **3.06x**（1024 维 cosine 约 341.7 ns），语义缓存精确命中约 10 ns，candle 单文本在 `--features mkl` 下 **4.0×** 加速。完整基准表、引擎选型实测证据、性能设计要点（时间窗拼批/批内去重/SIMD/线程调优/jemalloc）、GGUF 量化与调优开关注册表见 [⚡ 性能指南](docs/cn/PERFORMANCE.md)。

---

## 🔒 安全

### 🔒 安全设计

VecBoost 默认安全：出厂仅回环绑定，`auth.enabled=false` 时绑定非回环地址拒绝启动（逃生阀 `VECBOOST_ALLOW_INSECURE=1` 打 ERROR 告警）；认证授权基于 garrison（JWT + CSRF + RBAC admin 角色 + TOTP + 账号锁定），并覆盖 XFF 信任反转、文件路径白名单、输入长度校验、AES-256-GCM 配置加密、审计日志与 i18n 双语错误脱敏。逐项机制的代码级细节见 [🔒 安全文档](docs/cn/SECURITY.md)。

### 🔗 供应链与门禁

`cargo audit`、`cargo deny check`、CodeQL、Trivy/Checkov 镜像扫描、gitleaks 私密信息扫描与 pre-commit 钩子在 CI 与本地双重执行，完整清单与处置策略见 [🔒 安全文档 · 供应链与安全门禁](docs/cn/SECURITY.md#-供应链与安全门禁)。

### 🚨 报告安全漏洞

请勿通过公开 issue 报告安全漏洞。请优先使用 GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy) 私密披露通道提交报告（不可用时再联系 Kirky.X：<kirky-x@outlook.com>）。依赖 advisory 由 CI `cargo-audit` 门禁。完整政策与支持版本见 [SECURITY.md](docs/cn/SECURITY.md)。

---

## 🧭 开发路线图

路线图条目（核心服务 / 生态集成 / 安全与 i18n / 性能基座 / 调优开关 / 多副本共享状态等，✅ 已完成 / 📋 计划中）随版本演进维护，见 [📋 更新日志 · 开发路线图](docs/cn/CHANGELOG.md#-开发路线图)。

---

## 🤝 参与贡献

详细的贡献流程与代码规范请参阅 [🤝 贡献指南](docs/cn/CONTRIBUTING.md)。发现问题请[创建 Issue](https://github.com/Kirky-X/vecboost/issues/new)，有好想法欢迎[发起讨论](https://github.com/Kirky-X/vecboost/issues/new)，欢迎 Fork 并提交 [PR](https://github.com/Kirky-X/vecboost/pulls)。

---

## 📋 更新日志

完整版本历史见 [📋 更新日志](docs/cn/CHANGELOG.md)（遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，语义化版本）。

0.3.0-rc.1 含多项破坏性行为变更（安全默认值收敛、登录收敛、XFF 信任反转、RBAC 接线、缓存键/分词器变更等），**升级必读**：逐项「旧行为 → 新行为 → 迁移动作」对照表见 [📋 更新日志 · 0.3.0-rc.1](docs/cn/CHANGELOG.md#030-rc1---2026-09-22)。多副本边界（auth 会话默认进程内存，可经 `session_backend = "database"` 外置；限流/缓存现可经 `distributed-redis` 外置 Redis）与热重载语义（配置变更重启后生效）见 [❓ FAQ](docs/cn/FAQ.md#-配置与部署)。

---

## 📄 许可证

本项目基于 **Apache License 2.0** 发布 - 查看 [LICENSE](LICENSE) 文件了解更多。Copyright © 2025-2026 Kirky.X🌠。

---

## 🙏 致谢

核心依赖清单（onnxruntime / candle / tokenizers / hf-hub / 7 库生态 / garrison / axum / tokio / utoipa / prometheus / criterion / jemallocator）见 [🏗️ 架构文档 · 核心依赖致谢](docs/cn/ARCHITECTURE.md#-核心依赖致谢)。

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

<sub>© 2026 Kirky.X. 保留所有权利。</sub>
