# VecBoost 示例

本目录是独立的 workspace member crate `vecboost-examples`(`publish = false`),包含 VecBoost 向量嵌入服务的全功能示例代码,按功能分类组织为 10 个分类、27 个可执行二进制。

## 运行示例

`vecboost-examples` 作为 workspace member,通过 `-p vecboost-examples --bin <name>` 运行:

```bash
# 基础示例(默认 features)
cargo run -p vecboost-examples --bin embed

# GPU 示例(需要 CUDA)
cargo run -p vecboost-examples --bin gpu_candle_engine_test --features cuda

# ONNX 示例(需要 ONNX Runtime)
cargo run -p vecboost-examples --bin onnx --features onnx

# GPU + ONNX 性能对比
cargo run -p vecboost-examples --bin gpu_performance_comparison --features cuda,onnx
```

## 注册方式

每个示例在 `examples/Cargo.toml` 的 `[[bin]]` 段显式注册 `name` 与 `path`,引擎类示例通过 `required-features` 门控:

```toml
[[bin]]
name = "embed"
path = "basic/embed.rs"

[[bin]]
name = "gpu_candle_engine_test"
path = "gpu/candle_engine_test.rs"
required-features = ["cuda"]
```

## Features 透传

`vecboost-examples` 通过自身 features 透传引擎特性到主 crate:

| Feature | 启用 | 说明 |
|---------|------|------|
| `cuda` | `vecboost/cuda` + `candle-core`/`candle-nn`/`candle-transformers` | NVIDIA CUDA GPU 加速 |
| `onnx` | `vecboost/onnx` | ONNX Runtime 引擎 |
| `metal` | `vecboost/metal` | Apple Silicon GPU(macOS) |

默认 features 为空(`default = []`),基础示例无需额外 feature 即可运行。

## 示例分类

| 分类 | 目录 | 示例数 | 说明 |
|------|------|--------|------|
| basic | `basic/` | 3 | 基础用法:单文本嵌入、批量嵌入、相似度计算 |
| engine | `engine/` | 3 | 引擎抽象:Candle/ONNX 引擎初始化与运行时切换 |
| gpu | `gpu/` | 4 | GPU 引擎:设备检测、Candle/ONNX GPU 推理、性能对比 |
| http | `http/` | 3 | HTTP API:REST 接口调用(reqwest) |
| cli | `cli/` | 2 | 命令行工具:CLI 子命令调用 |
| auth | `auth/` | 3 | 认证授权:JWT 令牌、CSRF 防护、token 刷新 |
| cache | `cache/` | 2 | 缓存:oxcache 后端配置与 TTL 验证 |
| rate-limiting | `rate-limiting/` | 2 | 限流:limiteron 多维度限流配置 |
| pipeline | `pipeline/` | 2 | 请求管道:优先级队列与工作线程 |
| grpc | `grpc/` | 1 | gRPC 接口:客户端调用 embedding_service |
| monitoring | `monitoring/` | 2 | 监控:Prometheus 指标与性能采集 |

另外 `download_model.rs` 位于根目录,是模型下载工具。

## 示例清单

### basic — 基础用法
- `embed` — 单文本嵌入,演示 `EmbeddingService` + `api::embed`
- `batch` — 批量嵌入,演示 `api::embed_batch` 处理多条文本
- `similarity` — 余弦相似度,演示 `api::compute_similarity`

### engine — 引擎抽象
- `candle` — Candle 引擎初始化与推理
- `onnx` — ONNX 引擎初始化(需要 `onnx` feature)
- `switch` — 运行时 `EmbeddingService::switch_model`

### gpu — GPU 引擎(需要 `cuda` feature)
- `gpu_basic_device_test` — GPU 设备检测与基础测试
- `gpu_candle_engine_test` — Candle 引擎 GPU 推理测试
- `gpu_onnx_engine_test` — ONNX Runtime 引擎 GPU 推理测试(需要 `cuda,onnx`)
- `gpu_performance_comparison` — 引擎性能对比(需要 `cuda,onnx`)

### http — HTTP API
- `embed_api` — HTTP POST `/embed`
- `batch_api` — HTTP POST `/embed/batch`
- `similarity_api` — HTTP POST `/similarity`

### cli — 命令行工具
- `embed_cli` — 调用 `vecboost embed --text "hello"`
- `batch_cli` — 调用 `vecboost batch --input texts.txt`

### auth — 认证授权
- `jwt_auth` — JWT token 生成与验证
- `csrf` — CSRF token 获取与使用
- `refresh` — token 刷新流程

### cache — 缓存
- `cache_config` — oxcache 配置
- `ttl` — TTL 过期验证

### rate-limiting — 限流
- `rate_limit` — limiteron 限流配置
- `multi_dimension` — 全局+IP+用户多维度限流

### pipeline — 请求管道
- `priority_queue` — 优先级队列入队与出队
- `workers` — WorkerManager 启动与扩缩容

### grpc — gRPC 接口
- `grpc_client` — gRPC 客户端调用 embedding_service

### monitoring — 监控
- `metrics` — Prometheus 指标暴露
- `performance` — 推理性能监控

### 工具
- `download_model` — 下载预训练模型
