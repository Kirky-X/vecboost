# ❓ VecBoost FAQ

本页汇总 VecBoost 的常见问题与解答，按主题分组。没有找到答案？欢迎前往 [GitHub Issues](https://github.com/Kirky-X/vecboost/issues) 提问。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [通用问题](#-通用问题)
- [安装与构建](#-安装与构建)
- [模型与推理](#-模型与推理)
- [配置与部署](#️-配置与部署)
- [认证与安全](#-认证与安全)
- [API 使用](#-api-使用)
- [测试与性能](#-测试与性能)

</details>

---

## 🧭 通用问题

### ❓ 什么是 VecBoost？

VecBoost 是一个使用 Rust 构建的**高性能嵌入向量服务**：把文本向量化为嵌入向量，通过 HTTP/REST、gRPC、MCP、CLI 四种协议对外提供服务，支持 Candle（原生 Rust）与 ONNX Runtime 双推理引擎、CUDA/Metal GPU 加速、缓存、限流、认证等企业级能力。它由 7 个 Rust 库（trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge）组成模块化生态，经 `trait-kit` 模块注册中心装配。

### ❓ VecBoost 可以用于生产环境吗？

可以。项目具备：默认安全（回环绑定、无认证拒绝非回环启动）、JWT/CSRF/RBAC 认证授权、审计日志、Prometheus 指标与健康检查、优雅关闭、多架构 Docker 镜像（linux/amd64 + arm64）。升级到 Unreleased 版本前请阅读 [CHANGELOG](CHANGELOG.md) 的破坏性变更清单（安全默认值收敛等 13 项行为变更）。

### ❓ 支持哪些平台？

Linux（x86_64/aarch64，CI 覆盖 Ubuntu + Windows 测试）、macOS（Apple Silicon 经 `metal` feature）。CUDA 12.x 为可选（`cuda` feature）；Linux glibc 下默认启用 jemalloc 分配器。

### ❓ 在哪里可以获得帮助？

- 使用问题：[GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
- 文档：[📖 用户指南](USER_GUIDE.md)、[📘 API 参考](API_REFERENCE.md)、[❓ 本页](FAQ.md)
- 安全漏洞：**勿用公开 issue**，邮件 <kirky-x@outlook.com>（见 [🔒 安全文档](SECURITY.md)）

### ❓ 项目采用什么许可证？

MIT。详见 [LICENSE](../LICENSE)。

---

## 📦 安装与构建

### ❓ 如何安装构建？

```bash
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost
cargo build --release          # 默认 http feature
./target/release/vecboost      # 服务默认 http://127.0.0.1:9002
```

要求 Rust 1.91+（`Cargo.toml` 的 `rust-version` 为权威值）。更多组合见 [用户指南](USER_GUIDE.md)。

### ❓ 如何选择 feature 组合？

| 需求 | 构建 |
|------|------|
| 仅 HTTP 服务（默认） | `cargo build --release` |
| GPU（NVIDIA / Apple） | `--features cuda` / `--features metal` |
| gRPC / CLI / MCP | `--features grpc` / `--features cli` / `--features mcp` |
| 数据库持久化 | `--features db`（SQLite）/ `--features postgres` |
| 认证 | `--features auth` |
| CI 全特性 | `--features grpc,cli,auth,onnx,db,openapi,mcp` |

`confers`/`inklog`/`oxcache`/`limiteron`/`trait-kit` 为必选依赖始终启用，无需 feature。完整矩阵见 README [🏷️ Feature 标志](../README.md#️-feature-标志)。

### ❓ 构建失败提示链接错误（hgemm_）？

MKL 加速后端（`--features mkl`）依赖 intel-mkl-src 2020.1 静态库，该库缺少 `hgemm_` 半精度符号（candle 0.11 `mkl.rs:166` 引用），部分工具链会链接失败。这是已知的 opt-in 限制：请在链接正常的工具链/CI 矩阵中启用 `mkl`/`accelerate`（`cargo check` 不链接，两种组合均通过）。详见 [⚡ 性能指南](PERFORMANCE.md) 偏差记录。

### ❓ 为什么没有 `redis` feature？

`redis` 不是当前 `Cargo.toml` 中的 feature（缓存后端为进程内 oxcache；`db` feature 提供数据库持久化）。若旧文档/脚本引用了 `redis` feature，请以 `cargo build --features ...` 的报错与 `Cargo.toml [features]` 为准。

---

## 🤖 模型与推理

### ❓ 支持哪些模型？

默认 `BAAI/bge-small-en-v1.5`（384 维）。支持 HuggingFace 上 Bert / XlmRoberta 双架构嵌入模型（如 BGE 系列、MiniLM、multilingual-e5），mean/cls/max 池化，`expected_dimension` 按模型设置（BGE-M3 为 1024）。4 模型矩阵（BGE-en 384 / MiniLM 384 / BGE-zh 512 / multilingual-e5-small 384）由场景测试覆盖。

### ❓ 模型存放在哪里？必须联网下载吗？

模型经 HuggingFace Hub 下载（`hf-hub`），默认缓存于本地；也可 `model_path` 指向本地目录（优先使用）。国内网络可设 `HF_ENDPOINT` 指向镜像端点。首次启动会自动下载，可用 `examples/download_model.rs` 预下载。

### ❓ 如何切换模型 / 同时驻留多个模型？

- 运行时切换：`POST /api/1/model/switch`（admin 角色）或 gRPC `vecboost.model_switch`；
- 多模型驻留：`[model] max_resident_models` / `resident_memory_budget_mb`（LFRU 驱逐 + 热度持久化 `data/model_heat.json`）；
- 可用模型列表：`GET /api/1/models`。

### ❓ GGUF 量化模型能用了吗？

部分可用。`[model] quantized = true` + `--features quantized-gguf` 的路由、魔数校验与质量门（Q8_0 余弦中位数 ≥0.98 / Q4_K ≥0.95，`tests/quantized_parity.rs`）已就绪，但 **candle-transformers 0.11 尚无 quantized BERT 模型，推理后端待上游落地**。详见 [⚡ 性能指南](PERFORMANCE.md)。

### ❓ 为什么升级后需要重建向量索引？

Unreleased 版本将分词器统一为全平台 HuggingFace `tokenizers`（更正确），与旧自研 WordPiece 分词结果存在差异；缓存键也从 `text:{原文}` 改为 `emb:{model}:{xxh3_128}`。**存量向量与重建向量不兼容**，需重建索引；升级后首轮缓存全 miss 属一次性现象。

### ❓ Matryoshka 维度约简是什么？

对支持 Matryoshka 表示的模型，可请求更小维度（如 256/512/1024）以换取更小存储与更快检索；服务在截断后自动 L2 重归一化，保证余弦相似度正确。通过 `/v1/embeddings` 的 `dimensions` 参数使用。

---

## ⚙️ 配置与部署

### ❓ 配置文件在哪里？修改后要重启吗？

默认 `config/config.toml`（`--config <path>` 指定其他路径）。预置 `config_full.toml` / `config_minimal.toml`。**热重载语义**：配置变更会校验并打日志，重启后生效（非运行时热切换）；Kubernetes 用 ConfigMap 滚动更新。

### ❓ 哪些配置项必须用环境变量？

敏感项必须走环境变量（前缀 `VECBOOST_`）：`VECBOOST_JWT_SECRET`（启用认证时必填，≥32 字符）、`VECBOOST_ADMIN_PASSWORD`（启用认证时必填，≥12 位）、`VECBOOST_ENCRYPTION_KEY` 等。完整清单见 [📖 用户指南 · 环境变量](USER_GUIDE.md#-环境变量) 一节。

### ❓ `[flow_control]` / `[cache]` 配置段为什么不生效？

这两个段是历史遗留段名，**当前版本不解析**：限流走 `[rate_limit]`，缓存走 `[embedding]` 与 `[semantic_cache]`（persist 落盘、comparison_mode 也在后两者）。详见 [⚡ 性能指南](PERFORMANCE.md) 开关注册表。

### ❓ 服务绑定失败 / 生产如何暴露端口？

出厂 `host = "127.0.0.1"` 仅回环；`auth.enabled=false` 时绑定非回环地址会拒绝启动。受信网络容器可设 `VECBOOST_ALLOW_INSECURE=1`（打 ERROR 告警），生产环境应启用认证后绑定 `0.0.0.0`，并经反代（HTTPS）暴露，显式配置 `trusted_proxies`。

### ❓ Docker / Kubernetes 怎么部署？

Docker：`docker build -t vecboost:latest .` 后挂载 `config/` 与 `models/` 运行（CI 构建多架构镜像 linux/amd64 + arm64）。Kubernetes：需自备部署清单，用 ConfigMap 滚动更新配置；`/health` 为存活探针、`/health?depth=full` 为真实就绪探测。容器架构与扩展策略见 [🏗️ 架构文档](ARCHITECTURE.md) 部署架构一节。

### ❓ auth 开启时能多副本部署吗？

不能。推理路径无状态可水平扩展，但 `auth.enabled=true` 时认证会话存进程内存（oxcache DAO），**仅限单副本**；会话外置需 garrison db 后端补齐（规划中）。

---

## 🔐 认证与安全

### ❓ 如何启用认证？

```toml
[auth]
enabled = true
```

```bash
export VECBOOST_JWT_SECRET="至少 32 字符的密钥"
export VECBOOST_ADMIN_PASSWORD="至少 8 位的密码"
```

启用后登录端点获取 JWT，业务请求带 `Authorization: Bearer <token>`。完整流程见 [用户指南](USER_GUIDE.md) 认证一节与 [API 参考](API_REFERENCE.md)。

### ❓ 哪些接口需要 admin 角色？

`/api/1/model/*`（模型管理：switch/unload 等）与 `/embed/file`。业务嵌入调用使用普通账号即可。

### ❓ Token 有效期多久？

Unreleased 起 `token_expiration_hours` 缺省 **1 小时**（原 garrison 默认 30 天）。长会话场景请显式配置。

### ❓ 发现安全漏洞怎么办？

请勿公开 issue，邮件 <kirky-x@outlook.com>。流程与支持版本见 [🔒 安全文档](SECURITY.md)。

---

## 🌐 API 使用

### ❓ 有哪些协议和入口？

| 协议 | Feature | 入口 |
|------|---------|------|
| HTTP/REST | `http`（默认） | `http://127.0.0.1:9002/api/1/*` |
| OpenAI 兼容 | `http` | `POST /v1/embeddings`（支持 `dimensions` 与 base64） |
| gRPC | `grpc` | `:50051`（11 个 `vecboost.*` 方法） |
| MCP | `mcp` | `vecboost --mcp`（stdio，工具 embed/embed_batch/similarity/list_models） |
| CLI | `cli` | `vecboost embed --text "..."` 等 |

交互式文档：`http://localhost:9002/api-docs`（Swagger UI）。

### ❓ 为什么 embeddings 接口的 usage token 数与估算不符？

`/v1/embeddings` 的 usage 为**真实 token 计数**（Unreleased 起由 HF tokenizer 统计），与按字符估算的工具可能不同，以 tokenizer 结果为准。

### ❓ `/embed/file` 报错"未配置允许根"？

Unreleased 起 `/embed/file` 必须显式配置 `[server] grpc_allowed_roots`（不再回退进程 cwd），单文件 ≤10 MiB，`text_preview` 仅 admin。请为业务目录配置允许根。

---

## 🧪 测试与性能

### ❓ 如何在本地跑全部测试？

```bash
cargo test -p vecboost -p vecboost-examples
pytest tests/scenario -q    # 需先 cargo build -p vecboost --features http
cargo bench -p vecboost
```

场景测试在无 `models/` 时自动 SKIP 推理类用例。完整命令（与 CI 一致）见 [🧪 测试场景矩阵](TEST_SCENARIOS.md)。

### ❓ 性能数据在哪里？为什么 README 不写具体 QPS？

基准数据一律来自 `docs/benchmarks/` 实测归档（相似度 SIMD 3.06x、ContinuousBatchLoop 5.6x、语义缓存 ~10ns 命中等），吞吐基线（`embed_throughput_bench`）依赖本地模型，**标注待实测**——项目禁止编造基准数字。测量方法与复现命令见 [⚡ 性能指南](PERFORMANCE.md)。

### ❓ 如何诊断服务状态？

`vecboost doctor`：只读检查 config / tokenizer / 缓存持久层 / 线程 / GPU / 模型完整性（safetensors 头、gguf 魔数、hidden_size 配对），有 FAIL 项退出码 1。运行时指标：`/metrics`（Prometheus），含 `vecboost_stage_seconds{stage}` 分段延迟。

### ❓ 如何参与贡献？

阅读 [🤝 贡献指南](CONTRIBUTING.md)：提交前须通过 fmt / clippy（unwrap_used 门禁）/ 测试 / `scripts/doc_consistency_check.py` 四道门禁；行为变更须登记 CHANGELOG 并同步双语 README。
