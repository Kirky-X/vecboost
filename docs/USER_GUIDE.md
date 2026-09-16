# 📖 VecBoost 用户指南

**安装、配置和使用的完整说明**

[![Version 0.2.1](https://img.shields.io/badge/Version-0.2.1-green.svg?style=for-the-badge)](https://github.com/Kirky-X/vecboost) [![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![REST API](https://img.shields.io/badge/REST-API-9002-blue.svg?style=for-the-badge)](http://localhost:9002)

*安装、配置和使用 VecBoost 的完整说明。*

---

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [快速开始](#-快速开始)
- [安装](#-安装)
- [配置](#-配置)
- [运行服务](#-运行服务)
- [使用 API](#-使用-api)
- [认证](#-认证)
- [Docker 部署](#-docker-部署)
- [Kubernetes 部署](#-kubernetes-部署)
- [监控](#-监控)
- [国际化（i18n）](#-国际化i18n)
- [故障排除](#-故障排除)
- [最佳实践](#-最佳实践)
- [常见问题](#-常见问题)
- [下一步](#-下一步)
- [相关文档](#-相关文档)

</details>

---

## 🚀 快速开始

对于有经验的用户，这是最快的入门方式：

```bash
# 1. 克隆并构建
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost
cargo build --release

# 2. 使用默认设置运行
./target/release/vecboost

# 3. 测试 API（在新终端中）
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, VecBoost!"}'
```

**预期输出:**

```json
{
  "embedding": [0.123, 0.456, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

> **⏱️ 预计时间**: 2-5 分钟（取决于网络和硬件）

---

## 📦 安装

### 📋 前置条件

| 依赖 | 最低版本 | 说明 | 可选 |
|------|----------|------|------|
| **Rust** | 1.75+ | 编程语言（需要 2024 版） | ❌ |
| **Cargo** | 1.75+ | 构建工具（随 Rust 附带） | ❌ |
| **CUDA Toolkit** | 12.0 | NVIDIA GPU 支持 | ✅ |
| **Metal SDK** | - | Apple Silicon GPU 支持 | ✅ |

---

### ✅ 验证前置条件

```bash
# 检查 Rust 版本
rustc --version  # 应输出: rustc 1.75+

# 检查 Cargo 版本
cargo --version

# 检查 CUDA (Linux)
nvidia-smi  # 如果可用，应显示 GPU 信息

# 检查 Metal (macOS)
system_profiler SPDisplaysDataType
```

---

### 🔨 从源码构建

#### 选项 1: 仅 CPU（所有平台）

```bash
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost
cargo build --release
```

#### 选项 2: CUDA 支持（Linux）

```bash
# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 构建
cargo build --release --features cuda
```

#### 选项 3: Metal 支持（macOS）

```bash
cargo build --release --features metal
```

#### 选项 4: 全部功能

```bash
# 全协议（HTTP/gRPC/MCP/CLI）
cargo build --release --features http,grpc,mcp,cli

# 全协议 + 全可选功能（GPU/ONNX/认证/Redis）
cargo build --release --features http,grpc,mcp,cli,cuda,onnx,auth,redis
```

> **💡 提示**: `default = ["http"]`，默认构建仅启用 HTTP 协议。gRPC/MCP/CLI 协议需通过 `--features` 显式开启。confers（配置）、inklog（日志）、oxcache（缓存）、limiteron（限流）、trait-kit（模块注册）为必选依赖，无需 feature 开启。

---

### 🔍 验证构建

```bash
# 检查二进制文件
ls -lh target/release/vecboost

# 查看帮助信息
./target/release/vecboost --help
```

> **💡 提示**: 首次构建可能需要下载依赖和模型，请耐心等待。

---

## ⚙️ 配置

### 📄 配置文件

复制示例配置：

```bash
cp config/config.toml config/config_custom.toml
```

通过全局参数 `--config <path>`（或 `--config=<path>`）指定配置文件，服务器与 CLI 模式均生效。CLI 子命令模式下该参数须写在子命令之前，例如 `vecboost --config config_custom.toml embed --text "Hello"`。显式路径不存在时 fail-fast 报错退出（码 2）。预置三份配置：`config/config.toml`（默认，安全默认：回环绑定、GPU 关闭）、`config/config_full.toml`（完整示例）、`config/config_minimal.toml`（最小示例）。

---

### 🗂️ 配置段总览

下表对应 `AppConfig`（`src/config/app_config.rs`）实际解析的配置段：

| 区块 | 说明 | 依赖库 / Feature |
|------|------|------------------|
| `[server]` | 绑定地址、端口、超时、CORS、gRPC 设置 | - |
| `[model]` | 模型仓库、设备、精度、量化、多模型驻留 | - |
| `[embedding]` | 聚合模式、相似度度量、缓存、文本长度上限、WAL 落盘 | oxcache |
| `[rerank]` | 重排序服务配置 | - |
| `[monitoring]` | 内存限制、指标收集 | prometheus |
| `[auth]` | JWT、CSRF、管理员账号、token 有效期 | garrison（`auth`） |
| `[rate_limit]` | 多维令牌桶限流 | limiteron |
| `[audit]` | 审计日志 | inklog |
| `[database]` | 数据库连接（`[database] url = "sqlite:vecboost.db"`） | dbnexus（`db`） |
| `[logging]` | 日志级别、控制台、文件轮转 | inklog |
| `[pipeline.worker]` | 时间窗拼批 `batch_wait_ms` / `max_batch_size` | - |
| `[semantic_cache]` | 语义缓存 `comparison_mode` | oxcache |
| `[device]` | 硬件感知规划 `auto_plan` | - |

> **⚠️ 注意**：`[flow_control]` 与 `[cache]` 两个 TOML 段**当前版本不解析，编辑不生效**（历史遗留段名）；限流走 `[rate_limit]`，缓存走 `[embedding]` 与 `[semantic_cache]`。调优开关注册表见 [⚡ 性能指南](PERFORMANCE.md)。

---

### 🔧 主要配置选项

#### 服务器设置

```toml
[server]
host = "0.0.0.0"    # 绑定地址
port = 9002         # HTTP 端口
timeout = 30        # 请求超时（秒）
cors_enabled = false          # 是否启用 CORS（默认关闭）
cors_allow_origins = []       # 允许的跨域来源；含 "*" 或留空 = 任意来源
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `0.0.0.0` | 绑定地址 |
| `port` | `9002` | HTTP 端口 |
| `timeout` | `30` | 请求超时（秒） |
| `cors_enabled` | `false` | 是否启用 CORS 跨域支持 |
| `cors_allow_origins` | `[]` | 允许的跨域来源列表；包含 `"*"` 或留空表示允许任意来源（仅 `cors_enabled = true` 时生效） |

> **💡 提示**: HTTP 响应的 gzip 压缩始终启用，无需额外配置。

#### gRPC 设置

gRPC 协议通过 `grpc` feature 开启（`cargo build --release --features grpc`），所有协议处理函数均由 sdforge 生成。gRPC 配置项位于 `[server]` 段：

```toml
[server]
grpc_enabled = false              # 是否启用 gRPC 服务
grpc_host = "0.0.0.0"             # gRPC 绑定地址（可选，缺省同 host）
grpc_port = 50051                 # gRPC 端口
grpc_max_connections = 1000       # 最大并发连接数
grpc_timeout_seconds = 30         # gRPC 请求超时（秒）
grpc_require_auth = true          # 是否强制 JWT 认证
grpc_allowed_roots = ["/data"]    # 路径校验根目录列表（可选）
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grpc_enabled` | `false` | 是否启用 gRPC 服务 |
| `grpc_host` | 同 `host` | gRPC 绑定地址（可选） |
| `grpc_port` | `50051` | gRPC 端口 |
| `grpc_max_connections` | `1000` | 最大并发连接数 |
| `grpc_timeout_seconds` | `30` | gRPC 请求超时（秒） |
| `grpc_require_auth` | `true` | 是否强制 JWT 认证（默认强制） |
| `grpc_allowed_roots` | - | 路径校验根目录列表（可选，用于限定可访问路径） |

> **⚠️ 安全提示**: `grpc_require_auth` 默认为 `true`，调用方需通过配置 `grpc_require_auth = false` 显式关闭认证。生产环境建议保持开启，并配置 `grpc_allowed_roots` 限定可访问路径。

---

#### 模型设置

```toml
[model]
model_repo = "BAAI/bge-m3"  # HuggingFace 模型 ID
use_gpu = false             # 启用 GPU（需要相应功能）
batch_size = 32             # 批处理大小
expected_dimension = 1024   # 嵌入维度
max_sequence_length = 8192  # 每请求最大令牌数
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_repo` | `BAAI/bge-m3` | HuggingFace 模型 ID |
| `use_gpu` | `false` | 是否使用 GPU |
| `batch_size` | `32` | 批处理大小 |
| `expected_dimension` | `1024` | 嵌入向量维度 |
| `max_sequence_length` | `8192` | 最大序列长度 |

---

#### 缓存与文本长度设置

```toml
[embedding]
cache_enabled = true    # 启用缓存
cache_size = 1024       # 最大缓存条目数
max_text_length = 8192  # 单个文本最大字节长度
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_enabled` | `true` | 是否启用缓存 |
| `cache_size` | `1024` | 最大缓存条目数 |
| `max_text_length` | `8192` | 单个文本最大字节长度（防止资源耗尽攻击） |

---

#### 认证设置

```toml
[auth]
enabled = true
# ⚠️ 生产环境必须通过环境变量设置:
# export VECBOOST_JWT_SECRET="your-32-char-min-secret"
# export VECBOOST_ADMIN_PASSWORD="your-secure-password"
token_expiration_hours = 1
default_admin_username = "admin"
trusted_proxies = []  # 受信任代理 CIDR 列表
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 是否启用认证 |
| `jwt_secret` | - | JWT 密钥（至少 32 字符，**生产环境通过环境变量 `VECBOOST_JWT_SECRET` 设置**） |
| `token_expiration_hours` | `1` | 服务端令牌超时（小时），实际 `expires_in` 由 garrison 管理 |
| `default_admin_username` | `admin` | 默认管理员用户名 |
| `default_admin_password` | - | 默认管理员密码（**生产环境通过环境变量 `VECBOOST_ADMIN_PASSWORD` 设置**） |
| `trusted_proxies` | `[]` | 受信任代理 CIDR 列表（XFF 信任边界，空列表=无条件信任） |

> **⚠️ 安全提示**: 生产环境中请通过环境变量设置敏感信息！`jwt_secret` 通过 `VECBOOST_JWT_SECRET`，`default_admin_password` 通过 `VECBOOST_ADMIN_PASSWORD` 注入，禁止在配置文件中明文存储。生产部署建议配置 `trusted_proxies` 为实际反代 CIDR，防止客户端伪造 X-Forwarded-For。

---

#### 日志设置

```toml
[logging]
level = "info"                  # 日志级别
console = true                  # 是否输出到控制台
file_path = "logs/vecboost.log" # 日志文件路径（空字符串 = 不写文件）
rotation_size_mb = 100          # 单个日志文件轮转大小（MB）
max_files = 10                  # 保留的日志文件数量
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level` | `info` | 日志级别：`trace` / `debug` / `info` / `warn`（别名 `warning`）/ `error`；可被环境变量 `VECBOOST_LOG_LEVEL` 覆盖 |
| `console` | `true` | 是否输出到控制台（⚠️ 受上游 inklog 缺陷影响，运行期关闭暂不生效；CLI/MCP 模式下日志自动改道 stderr，stdout 仅承载结果/协议消息） |
| `file_path` | `logs/vecboost.log` | 日志文件路径，空字符串表示不写入文件 |
| `rotation_size_mb` | `100` | 单个日志文件轮转大小（MB），轮转后的历史文件自动压缩 |
| `max_files` | `10` | 保留的日志文件数量 |

---

### 🔄 环境变量

使用环境变量覆盖配置文件：

| 环境变量 | 对应配置 | 示例值 |
|----------|----------|--------|
| `VECBOOST_SERVER_PORT` | `server.port` | `9002` |
| `VECBOOST_MODEL_REPO` | `model.model_repo` | `BAAI/bge-m3` |
| `VECBOOST_JWT_SECRET` | `auth.jwt_secret` | `your-secret-key`（≥32 字符） |
| `VECBOOST_ADMIN_PASSWORD` | `auth.default_admin_password` | `your-admin-password`（≥12 字符） |
| `VECBOOST_ALLOW_INSECURE` | - | `1` = 允许非回环绑定 + 无认证（打 ERROR 告警，仅供受信网络容器） |
| `VECBOOST_ENCRYPTION_KEY` | - | 32 字节 hex 密钥（用于敏感配置加密） |
| `VECBOOST_REQUIRE_ENCRYPTION` | - | 强制加密配置 |
| `VECBOOST_KEY_STORAGE_TYPE` / `VECBOOST_KEY_FILE_PATH` | - | 密钥存储后端选择与路径 |
| `VECBOOST_LOG_LEVEL` | `logging.level`（优先级高于配置文件） | `trace`, `debug`, `info`, `warn`/`warning`, `error` |
| `VECBOOST_LANG` | - | `zh`, `en`（全局默认语言） |
| `VECBOOST_DATABASE_PASSWORD` | - | 数据库密码（`db` feature） |
| `VECBOOST_MODEL_API_KEY` | - | 模型仓库 API Key |
| `VECBOOST_NO_THREAD_TUNE` | - | `1` = 关闭物理核线程调优 |
| `HF_ENDPOINT` | - | HuggingFace 镜像端点 |

---

### 📋 完整示例配置

```toml
# config_custom.toml
[server]
host = "0.0.0.0"
port = 9002
# gRPC 配置（需以 --features grpc 构建）
grpc_enabled = true
grpc_port = 50051
grpc_max_connections = 1000
grpc_timeout_seconds = 30
grpc_require_auth = true
grpc_allowed_roots = ["/data"]

[model]
model_repo = "BAAI/bge-m3"
use_gpu = true
batch_size = 64
expected_dimension = 1024

[embedding]
cache_enabled = true
cache_size = 2048

[auth]
enabled = true
# ⚠️ 敏感信息通过环境变量注入:
# VECBOOST_JWT_SECRET / VECBOOST_ADMIN_PASSWORD
token_expiration_hours = 24

[rate_limit]
enabled = true
global_requests_per_minute = 2000

[pipeline.worker]     # 时间窗动态拼批
batch_wait_ms = 5
max_batch_size = 8

[semantic_cache]      # 向量输出量化比较
comparison_mode = "exact"   # exact | i8 | binary
```

---

## 🏃 运行服务

### 🚀 开发模式

```bash
# 使用默认配置运行
cargo run

# 使用自定义配置运行
cargo run -- --config config_custom.toml

# 使用调试日志运行
RUST_LOG=debug cargo run
```

---

### 🏢 生产模式

```bash
# 1. 先构建
cargo build --release

# 2. 运行二进制文件
./target/release/vecboost --config config_custom.toml

# 3. 在后台运行
nohup ./target/release/vecboost --config config_custom.toml > vecboost.log 2>&1 &

# 4. 检查状态
ps aux | grep vecboost
```

---

### 🐳 Docker 部署

```bash
# 构建镜像
docker build -t vecboost:latest .

# 运行容器
docker run -d \
  -p 9002:9002 \
  -v $(pwd)/config/config_custom.toml:/app/config/config.toml \
  -v $(pwd)/models:/app/models \
  --name vecboost \
  vecboost:latest

# 检查日志
docker logs -f vecboost

# 停止容器
docker stop vecboost
```

---

### ✅ 验证服务

```bash
# 健康检查
curl http://localhost:9002/health

# 预期响应:
# {"status":"OK"}
```

---

## 🌐 使用 API

### 📝 生成嵌入向量

#### 单个文本

```bash
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

**响应:**

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

---

#### 批量嵌入

```bash
curl -X POST http://localhost:9002/api/1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "第一个文档",
      "第二个文档",
      "第三个文档"
    ],
    "normalize": true
  }'
```

---

#### 归一化选项

`normalize` 选项返回单位长度嵌入向量（用于余弦相似度）：

```bash
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "要嵌入的文本", "normalize": true}'
```

---

### 📊 计算相似度

```bash
curl -X POST http://localhost:9002/api/1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "source": "机器学习是人工智能的一个分支",
    "target": "深度学习使用神经网络"
  }'
```

**响应:**

```json
{
  "score": 0.85
}
```

> **💡 说明**: 相似度 API 接受两段文本（`source` 和 `target`），自动向量化后计算余弦相似度。

---

### 🔍 重排序（Rerank）

```bash
curl -X POST http://localhost:9002/api/1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支",
      "今天天气很好",
      "深度学习使用神经网络"
    ],
    "top_k": 2
  }'
```

---

### 🛠️ 管理模型

#### 获取当前模型

```bash
curl http://localhost:9002/api/1/model/current
```

#### 获取模型详细信息

```bash
curl http://localhost:9002/api/1/model/info
```

#### 列出可用模型

```bash
curl http://localhost:9002/api/1/models
```

---

## 🔐 认证

### 启用认证

1. 在 `[auth]` 部分设置 `enabled = true`
2. 配置 `jwt_secret`（至少 32 个字符）
3. 设置管理员凭据

```toml
[auth]
enabled = true
# jwt_secret 通过环境变量 VECBOOST_JWT_SECRET 注入
default_admin_username = "admin"
# default_admin_password 通过环境变量 VECBOOST_ADMIN_PASSWORD 注入
```

---

### 获取令牌

```bash
curl -X POST http://localhost:9002/api/1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "Secure@Passw0rd!2026"
  }'
```

**响应:**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 0
}
```

> **💡 说明**: `expires_in` 为 `0` 表示令牌过期时间由 garrison 服务端统一管理（通过 `GarrisonConfig.timeout` 控制），客户端无需自行计算过期。

---

### 使用令牌

在 API 请求中包含令牌：

```bash
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{"text": "Hello, world!"}'
```

---

### 令牌过期

令牌过期时间由 garrison 服务端统一管理（`expires_in` 返回 `0`）。可在 `config/config.toml` 中配置服务端超时：

```toml
[auth]
token_expiration_hours = 24
```

---

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t vecboost:latest .
```

---

### 使用 Docker Compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  vecboost:
    image: vecboost:latest
    ports:
      - "9002:9002"
      - "50051:50051"
    volumes:
      - ./config/config_custom.toml:/app/config/config.toml
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - VECBOOST_JWT_SECRET=${JWT_SECRET}
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

启动服务:

```bash
docker-compose up -d
```

---

### Docker 环境变量

| 变量 | 描述 | 必需 |
|------|------|------|
| `VECBOOST_JWT_SECRET` | JWT 密钥（认证时必需，≥32 字符） | ✅ |
| `VECBOOST_ADMIN_PASSWORD` | 管理员密码（认证时必需，≥12 字符） | ✅ |
| `VECBOOST_ENCRYPTION_KEY` | 敏感配置加密密钥（32 字节 hex） | 推荐 |
| `VECBOOST_LANG` | 默认语言（`zh` 或 `en`） | ❌ |
| `VECBOOST_LOG_LEVEL` | 日志级别 (`trace`, `debug`, `info`, `warn`/`warning`, `error`)，覆盖 `[logging].level` | ❌ |

---

## ☸️ Kubernetes 部署

### 前置条件

- Kubernetes 集群（1.20+）
- kubectl 已配置
- Helm（可选）

---

### 使用 kubectl 部署

```bash
# 创建命名空间
kubectl create namespace vecboost

# 应用配置（请替换为你的实际清单路径）
kubectl apply -f <your-k8s-manifests>/ -n vecboost

# 检查部署状态
kubectl get pods -n vecboost

# 查看日志
kubectl logs -f deployment/vecboost -n vecboost
```

---

### GPU 部署

对于 GPU 工作负载:

```bash
# 应用 GPU 部署清单（请替换为你的实际清单路径）
kubectl apply -f <your-gpu-deployment>.yaml -n vecboost
```

---

### 扩缩容

```bash
# 手动扩缩容
kubectl scale deployment vecboost --replicas=3 -n vecboost

# 或使用 HPA（请替换为你的实际 HPA 清单路径）
kubectl apply -f <your-hpa>.yaml -n vecboost
```

---

### 访问服务

```bash
# 端口转发以进行本地访问
kubectl port-forward -n vecboost svc/vecboost 9002:9002

# 或使用 ingress（请替换为你的实际 ingress 清单路径）
kubectl apply -f <your-ingress>.yaml
```

---

## 🌍 国际化（i18n）

VecBoost 支持中英双语错误响应，通过 `Accept-Language` 请求头自动协商语言。

### 语言优先级

1. **请求级**：`Accept-Language` 请求头（如 `Accept-Language: zh-CN,zh;q=0.9`）
2. **全局默认**：`VECBOOST_LANG` 环境变量（如 `VECBOOST_LANG=zh`）
3. **系统 locale**：`LC_ALL`/`LANG` 环境变量
4. **兜底**：英文 (`en`)

### 使用示例

```bash
# 中文错误响应
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -H "Accept-Language: zh-CN" \
  -d '{"text": ""}'

# 响应（中文）
{"error": {"code": "INVALID_INPUT", "message": "文本不能为空"}}

# 英文错误响应（默认）
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'

# 响应（英文）
{"error": {"code": "INVALID_INPUT", "message": "Text cannot be empty"}}
```

> **💡 说明**: i18n 覆盖所有用户可见消息：HTTP JSON 错误响应、CLI 帮助文本、gRPC 错误详情、启动错误日志。翻译键定义在 `src/i18n/locales/{en,zh}/messages.ftl` 和 `errors.ftl`（共 114 个键）。

---

## 📊 监控

### 健康端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 服务健康状态（返回 `{"status": "OK"}` 或 503） |
| `/metrics` | GET | Prometheus 指标 |

---

### Prometheus 指标

在 `/metrics` 访问指标:

```
# HELP vecboost_requests_total 总请求数
# TYPE vecboost_requests_total counter
vecboost_requests_total{method="POST",endpoint="/api/1/embed"} 1234

# HELP vecboost_embedding_latency_seconds 嵌入生成延迟
# TYPE vecboost_embedding_latency_seconds histogram
vecboost_embedding_latency_seconds_bucket{le="0.005"} 100
vecboost_embedding_latency_seconds_bucket{le="0.01"} 500
```

---

### Grafana 仪表板

导入 VecBoost 仪表板（Grafana 仪表板配置计划中）：

1. 打开 Grafana
2. 导航到仪表板 → 导入
3. 上传 JSON 文件（或配置 Prometheus 数据源后手动创建）

---

### 告警配置

配置 Prometheus 告警规则（示例）：

```yaml
alerts:
  - alert: VecBoostDown
    expr: up{job="vecboost"} == 0
    for: 5m
    annotations:
      summary: "VecBoost 服务已关闭"
```

---

## 🔧 故障排除

### 常见问题

#### 服务无法启动

**问题**: 服务因配置错误启动失败。

```bash
# 检查日志
./target/release/vecboost 2>&1 | head -50
```

**常见原因和解决方案:**

| 问题 | 解决方案 |
|------|----------|
| JWT 密钥太短 | 确保 `jwt_secret` 至少 32 个字符 |
| 端口已被占用 | 检查端口: `lsof -i :9002` |
| 模型下载失败 | 验证网络连接 |

---

#### GPU 未检测到

**问题**: GPU 加速不工作。

```bash
# 检查 GPU 可用性
nvidia-smi

# 验证 CUDA 安装
nvcc --version

# 检查应用程序日志
grep -i cuda target/release/vecboost.log
```

**解决方案:**

1. 安装 CUDA toolkit
2. 使用 `--features cuda` 重新构建
3. 验证 GPU 驱动是最新版本

---

#### 内存不足

**问题**: 服务因 OOM 崩溃。

**解决方案:**

1. 减小配置中的 `batch_size`
2. 限制 `cache_size`
3. 启用 CPU 回退: `gpu_oom_fallback_enabled = true`
4. 增加容器内存限制

---

#### 认证失败

**问题**: 401 未授权错误。

```bash
# 检查令牌是否有效（通过 /auth/me 端点验证当前用户信息）
curl http://localhost:9002/api/1/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

**解决方案:**

1. 通过登录端点刷新令牌
2. 检查系统时钟同步
3. 验证 JWT 密钥是否更改

---

#### 速率限制

**问题**: 429 请求过多。

**解决方案:**

1. 实现指数退避重试
2. 在配置中增加速率限制
3. 将 IP 添加到白名单

```toml
[rate_limit]
ip_whitelist = ["127.0.0.1", "10.0.0.0/8"]
```

---

### 收集调试信息

```bash
# 启用调试日志
export RUST_LOG=debug

# 使用详细输出运行
./target/release/vecboost --config config_custom.toml 2>&1 | tee debug.log

# 收集系统信息
uname -a
nvidia-smi  # 如果有 GPU
free -h     # 内存
```

---

### 获取帮助

- 查看现有[问题](https://github.com/Kirky-X/vecboost/issues)
- 查看 [API 参考](API_REFERENCE.md)
- 查看 [架构设计](ARCHITECTURE.md)

---

## ✅ 最佳实践

### 🔒 安全性

| 检查项 | 建议 |
|--------|------|
| JWT 密钥 | 使用强密钥（32+ 字符） |
| HTTPS | 生产环境启用 HTTPS |
| 速率限制 | 配置适当的速率限制 |
| 审计日志 | 启用审计日志 |
| 密钥轮换 | 定期轮换密钥 |

---

### 🚀 性能

| 检查项 | 建议 |
|--------|------|
| GPU 加速 | 高吞吐量场景使用 GPU |
| 批处理大小 | 根据硬件调整 `batch_size` |
| 缓存配置 | 配置适当的缓存大小 |
| 内存监控 | 监控内存使用情况 |
| 连接池 | 使用连接池 |

---

### 🛡️ 可靠性

| 检查项 | 建议 |
|--------|------|
| 健康检查 | 配置健康检查端点 |
| 熔断器 | 配置熔断器 |
| 重试机制 | 实现重试逻辑 |
| 多副本 | 使用多个副本 |
| 备份 | 定期备份配置 |

---

## ❓ 常见问题

**问: 可以使用自己的模型吗？**

答: 可以，将 `model_repo` 设置为 HuggingFace 模型 ID 或本地路径。

---

**问: 如何在运行时更改模型？**

答: 使用 `POST /api/1/model/switch` 端点。

---

**问: 最大批处理大小是多少？**

答: 可通过 `batch_size` 配置，默认为 32。更大的值会增加吞吐量但使用更多内存。

---

**问: VecBoost 支持流式传输吗？**

答: 当前不支持，但计划在将来版本中添加。

---

**问: 如何更新到新版本？**

答: 停止服务、构建/拉取新版本、必要时更新配置、重新启动。

---

**问: 可以运行多个实例吗？**

答: 可以，配置带会话亲和性的负载均衡器以处理认证请求。

---

## 🎯 下一步

- [📚 API 参考](API_REFERENCE.md) - 详细 API 文档
- [🏗️ 架构设计](ARCHITECTURE.md) - 系统设计详情
- [💻 示例代码](../examples/) - 代码示例

---

## 📚 相关文档

| 文档 | 说明 |
|:-----|:-----|
| [📘 API 参考](API_REFERENCE.md) | 完整的 REST API 和 gRPC 文档 |
| [🏗️ 架构设计](ARCHITECTURE.md) | 内部架构、组件与设计决策 |
| [📋 更新日志](CHANGELOG.md) | 每个版本的变更记录 |

---

> **📝 最后更新**: 2026-09-06 | **版本**: 0.2.1 | **问题反馈**: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
