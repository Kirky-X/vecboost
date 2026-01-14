<div align="center">

# 📖 VecBoost 用户指南

**安装、配置和使用的完整说明**

[![Version 0.1.0](https://img.shields.io/badge/Version-0.1.0-green.svg?style=for-the-badge)](https://github.com/Kirky-X/vecboost) [![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![REST API](https://img.shields.io/badge/REST-API-9002-blue.svg?style=for-the-badge)](http://localhost:9002)

*安装、配置和使用 VecBoost 的完整说明。*

</div>

---

## 📋 目录

| 章节 | 说明 |
|------|------|
| [快速开始](#-快速开始) | 快速上手指南 |
| [安装](#-安装) | 系统要求和安装步骤 |
| [配置](#-配置) | 配置文件详解 |
| [运行服务](#-运行服务) | 启动和管理服务 |
| [使用 API](#-使用-api) | API 调用示例 |
| [认证](#-认证) | JWT 认证配置 |
| [Docker 部署](#-docker-部署) | Docker 容器化部署 |
| [Kubernetes 部署](#-kubernetes-部署) | K8s 集群部署 |
| [监控](#-监控) | 可观测性配置 |
| [故障排除](#-故障排除) | 常见问题解决 |
| [最佳实践](#-最佳实践) | 安全、性能和可靠性建议 |
| [常见问题](#-常见问题) | FAQ |
| [下一步](#-下一步) | 相关资源链接 |

---

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
curl -X POST http://localhost:9002/api/v1/embed \
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
cargo build --release --features cuda,onnx,grpc,auth,redis
```

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
cp config.toml config_custom.toml
```

---

### 🔧 主要配置选项

#### 服务器设置

```toml
[server]
host = "0.0.0.0"    # 绑定地址
port = 9002         # HTTP 端口
timeout = 30        # 请求超时（秒）
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `0.0.0.0` | 绑定地址 |
| `port` | `9002` | HTTP 端口 |
| `timeout` | `30` | 请求超时（秒） |

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

#### 缓存设置

```toml
[embedding]
cache_enabled = true    # 启用缓存
cache_size = 1024       # 最大缓存条目数
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_enabled` | `true` | 是否启用缓存 |
| `cache_size` | `1024` | 最大缓存条目数 |

---

#### 认证设置

```toml
[auth]
enabled = true
jwt_secret = "your-secure-secret-key-at-least-32-chars"
token_expiration_hours = 1
default_admin_username = "admin"
default_admin_password = "Secure@Passw0rd!2026"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | 是否启用认证 |
| `jwt_secret` | - | JWT 密钥（至少 32 字符） |
| `token_expiration_hours` | `1` | 令牌过期时间（小时） |
| `default_admin_username` | `admin` | 默认管理员用户名 |
| `default_admin_password` | - | 默认管理员密码 |

> **⚠️ 安全提示**: 生产环境中请修改默认管理员密码！

---

### 🔄 环境变量

使用环境变量覆盖配置文件：

| 环境变量 | 对应配置 | 示例值 |
|----------|----------|--------|
| `VECBOOST_SERVER_PORT` | `server.port` | `9002` |
| `VECBOOST_MODEL_REPO` | `model.model_repo` | `BAAI/bge-m3` |
| `VECBOOST_JWT_SECRET` | `auth.jwt_secret` | `your-secret-key` |
| `VECBOOST_CACHE_SIZE` | `embedding.cache_size` | `1024` |
| `VECBOOST_LOG_LEVEL` | - | `debug`, `info`, `warn`, `error` |

---

### 📋 完整示例配置

```toml
# config_custom.toml
[server]
host = "0.0.0.0"
port = 9002

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
jwt_secret = "your-very-long-secret-key-min-32-chars"

[rate_limit]
enabled = true
global_requests_per_minute = 2000
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
  -v $(pwd)/config_custom.toml:/app/config.toml \
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
# {"status":"healthy","version":"0.1.0",...}
```

---

## 🌐 使用 API

### 📝 生成嵌入向量

#### 单个文本

```bash
curl -X POST http://localhost:9002/api/v1/embed \
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
curl -X POST http://localhost:9002/api/v1/embed/batch \
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
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "要嵌入的文本", "normalize": true}'
```

---

### 📊 计算相似度

```bash
curl -X POST http://localhost:9002/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "vector1": [0.1, 0.2, 0.3, ...],
    "vector2": [0.1, 0.2, 0.3, ...],
    "metric": "cosine"
  }'
```

---

### 🔍 搜索文档

```bash
curl -X POST http://localhost:9002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "搜索查询",
    "documents": [
      "关于 AI 的文档",
      "关于 Rust 的文档",
      "关于 ML 的文档"
    ],
    "top_k": 2
  }'
```

---

### 🛠️ 管理模型

#### 获取当前模型

```bash
curl http://localhost:9002/api/v1/model
```

#### 列出可用模型

```bash
curl http://localhost:9002/api/v1/models
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
jwt_secret = "your-very-long-secret-key-min-32-chars"
default_admin_username = "admin"
default_admin_password = "Secure@Passw0rd!2026"
```

---

### 获取令牌

```bash
curl -X POST http://localhost:9002/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "Secure@Passw0rd!2026"
  }'
```

**响应:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

### 使用令牌

在 API 请求中包含令牌：

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{"text": "Hello, world!"}'
```

---

### 令牌过期

默认令牌过期时间为 1 小时。在 `config.toml` 中配置：

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
      - ./config_custom.toml:/app/config.toml
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
| `VECBOOST_JWT_SECRET` | JWT 密钥（认证时必需） | ✅ |
| `VECBOOST_LOG_LEVEL` | 日志级别 (`debug`, `info`, `warn`, `error`) | ❌ |
| `VECBOOST_CACHE_SIZE` | 缓存大小覆盖 | ❌ |

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

# 应用配置
kubectl apply -f deployments/kubernetes/ -n vecboost

# 检查部署状态
kubectl get pods -n vecboost

# 查看日志
kubectl logs -f deployment/vecboost -n vecboost
```

---

### GPU 部署

对于 GPU 工作负载:

```bash
# 应用特定 GPU 部署
kubectl apply -f deployments/kubernetes/gpu-deployment.yaml -n vecboost
```

---

### 扩缩容

```bash
# 手动扩缩容
kubectl scale deployment vecboost --replicas=3 -n vecboost

# 或使用 HPA
kubectl apply -f deployments/kubernetes/hpa.yaml -n vecboost
```

---

### 访问服务

```bash
# 端口转发以进行本地访问
kubectl port-forward -n vecboost svc/vecboost 9002:9002

# 或使用 ingress
kubectl apply -f deployments/kubernetes/ingress.yaml
```

---

## 📊 监控

### 健康端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 服务健康状态 |
| `/ready` | GET | 就绪探针 |
| `/metrics` | GET | Prometheus 指标 |

---

### Prometheus 指标

在 `/metrics` 访问指标:

```
# HELP vecboost_requests_total 总请求数
# TYPE vecboost_requests_total counter
vecboost_requests_total{method="POST",endpoint="/api/v1/embed"} 1234

# HELP vecboost_embedding_latency_seconds 嵌入生成延迟
# TYPE vecboost_embedding_latency_seconds histogram
vecboost_embedding_latency_seconds_bucket{le="0.005"} 100
vecboost_embedding_latency_seconds_bucket{le="0.01"} 500
```

---

### Grafana 仪表板

从 `deployments/grafana-dashboard.json` 导入仪表板:

1. 打开 Grafana
2. 导航到仪表板 → 导入
3. 上传 JSON 文件

---

### 告警配置

在 `deployments/alerts.yml` 中配置告警:

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
# 检查令牌是否有效
curl http://localhost:9002/api/v1/auth/verify
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
- 查看 [API 参考](API_REFERENCE_zh.md)
- 查看 [架构设计](ARCHITECTURE_zh.md)

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

答: 使用 `POST /api/v1/model/switch` 端点。

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

- [📚 API 参考](API_REFERENCE_zh.md) - 详细 API 文档
- [🏗️ 架构设计](ARCHITECTURE_zh.md) - 系统设计详情
- [🤝 贡献指南](../CONTRIBUTING.md) - 如何贡献代码
- [💻 示例代码](../examples/) - 代码示例

---

> **📝 最后更新**: 2026-01-14 | **版本**: 0.1.0 | **问题反馈**: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
