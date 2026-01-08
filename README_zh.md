<div align="center">

# 🚀 VecBoost

<p>
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build">
</p>

<p align="center">
  <strong>专为生产环境优化的高性能 Rust 向量嵌入服务</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#架构设计">架构设计</a> •
  <a href="#配置说明">配置说明</a> •
  <a href="#api-参考">API 参考</a>
</p>

</div>

---

## ✨ 功能特性

| 核心功能 | 高级功能 |
|---------|---------|
| ✅ **高性能** - Rust + Tokio 实现最大吞吐量 | 🚀 **批处理** - 高效的批量请求处理 |
| ✅ **多引擎支持** - Candle (默认) 和 ONNX Runtime | 🔐 **认证授权** - JWT、CSRF、API Key |
| ✅ **GPU 加速** - CUDA、Metal、ROCm 支持 | 📊 **监控指标** - Prometheus 指标导出 |
| ✅ **双协议支持** - HTTP REST 和 gRPC API | 📦 **流量限制** - 多维度限流 |
| ✅ **自动扩缩容** - 带优先级的请求管道队列 | 🔍 **审计日志** - 完整操作追踪 |
| ✅ **多级缓存** - ARC、LFU、LRU 分层缓存 | 🛡️ **安全机制** - Argon2、AES-GCM 加密 |

---

## 🚀 快速开始

### 安装构建

```bash
# CPU 版本构建
cargo build --release

# GPU 支持构建
cargo build --release --features cuda      # NVIDIA CUDA
cargo build --release --features metal     # Apple Silicon
cargo build --release --features onnx      # ONNX Runtime
cargo build --release --features grpc      # gRPC 服务器

# 所有功能
cargo build --release --features cuda,metal,onnx,grpc
```

### 运行服务

```bash
# 默认配置运行
cargo run --release

# 自定义配置文件
cargo run --release -- --config config.toml
```

### Docker 部署

```bash
docker build -t vecboost .
docker run -p 9002:9002 -p 50051:50051 -p 9090:9090 vecboost
```

---

## 📚 文档导航

- [📖 用户指南](docs/USER_GUIDE.md)
- [🏗️ 架构设计](docs/ARCHITECTURE.md)
- [📘 API 参考](docs/API_REFERENCE.md)
- [🤝 贡献指南](docs/CONTRIBUTING.md)

---

## 🏗️ 架构设计

```mermaid
graph TB
    Client --> HTTP[HTTP API :9002]
    Client --> gRPC[gRPC API :50051]
    
    HTTP --> Auth[认证中间件]
    gRPC --> Auth
    
    Auth --> RateLimit[流量限制]
    RateLimit --> Router[请求路由]
    
    Router --> Embedding[/embed]
    Router --> Similarity[/similarity]
    Router --> Search[/search]
    Router --> Health[/health]
    
    Embedding --> Service[嵌入服务]
    Similarity --> Service
    Search --> Service
    
    Service --> Engine[推理引擎]
    Engine --> Candle[Candle 引擎]
    Engine --> ONNX[ONNX 引擎]
    
    Engine --> Device[设备管理]
    Device --> GPU[GPU/CUDA]
    Device --> CPU[CPU]
    
    Service --> Cache[KV 缓存]
    Service --> Pipeline[优先级管道]
    
    Pipeline --> Queue[请求队列]
    Queue --> Scheduler[批处理调度器]
    
    Service --> Metrics[指标收集]
    Metrics --> Prometheus[:9090]
```

---

## ⚙️ 配置说明

### 默认端口

| 服务 | 端口 |
|------|------|
| HTTP API | 9002 |
| gRPC API | 50051 |
| Prometheus 指标 | 9090 |

### 配置文件示例 (config.toml)

```toml
[server]
host = "0.0.0.0"
port = 9002

[model]
model_repo = "BAAI/bge-m3"
use_gpu = false
batch_size = 32

[auth]
enabled = true
jwt_secret = "您的密钥至少32个字符"

[rate_limit]
enabled = true
global_requests_per_minute = 1000
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| VECBOOST_HOST | 服务绑定地址 | 0.0.0.0 |
| VECBOOST_PORT | HTTP 端口 | 9002 |
| VECBOOST_GRPC_PORT | gRPC 端口 | 50051 |
| VECBOOST_METRICS_PORT | Prometheus 端口 | 9090 |
| VECBOOST_MODEL_REPO | 模型仓库 | BAAI/bge-m3 |
| VECBOOST_USE_GPU | 是否使用 GPU | false |

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 吞吐量 | 10,000+ 请求/秒 |
| P95 延迟 | < 50ms |
| P99 延迟 | < 100ms |
| 错误率 | < 0.1% |

---

## 📁 项目结构

```
vecboost/
├── src/
│   ├── audit/          # 审计日志模块
│   ├── auth/           # 认证授权 (JWT、CSRF、用户存储)
│   ├── cache/          # 多级缓存实现
│   ├── config/         # 配置管理
│   ├── device/         # GPU/CPU 设备管理
│   ├── domain/         # 领域类型定义
│   ├── engine/         # 推理引擎 (Candle、ONNX)
│   ├── grpc/           # gRPC 服务器
│   ├── metrics/        # Prometheus 指标
│   ├── pipeline/       # 请求队列与调度
│   ├── rate_limit/     # 流量限制
│   ├── routes/         # HTTP 路由处理
│   ├── security/       # 加密、密钥存储
│   ├── service/        # 业务逻辑
│   └── text/           # 文本处理
├── tests/              # 集成测试与性能测试
├── examples/           # 示例代码
├── deployments/        # Docker、Kubernetes 配置
└── docs/               # 项目文档
```

---

## 🧪 测试

```bash
# 运行所有测试（包含所有功能）
cargo test --all-features

# 单元测试
cargo test --lib

# 集成测试
cargo test --tests

# 性能基准测试
cargo test --features cuda,grpc --test performance_test
```

---

## 🤝 贡献指南

请参阅 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解如何参与项目贡献。

### 开发流程

1. Fork 本仓库
2. 创建功能分支: `git checkout -b feature/xxx`
3. 提交更改: `git commit -m 'feat: xxx'`
4. 推送分支: `git push origin feature/xxx`
5. 创建 Pull Request

### 代码规范

- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 检查代码质量
- 所有公开 API 必须添加文档注释
- 新功能需要添加对应的测试用例

---

## 📄 开源许可

MIT License - 详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">

**由 VecBoost 团队用 ❤️ 构建**

</div>
