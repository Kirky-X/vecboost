# VecBoost - Rust Text Vectorization Module

基于 Rust 和 Candle 实现的高性能文本向量化服务，支持 BGE-M3 等 HuggingFace 模型。

## 功能特性

- 🚀 **纯 Rust 实现**：基于 Candle 推理引擎，无 Python 依赖。
- ⚡ **高性能**：支持 CPU/GPU 推理，自动降级。
- 📦 **开箱即用**：自动从 ModelScope/HuggingFace 下载模型。
- 🛠 **API 服务**：提供 RESTful API 进行向量化和相似度计算。
- 📊 **多种相似度度量**：支持余弦相似度、欧几里得距离、点积、曼哈顿距离。
- 📈 **性能监控**：内置性能指标收集器，实时监控推理延迟、吞吐量和资源利用率。

## 快速开始

### 1. 环境要求

- Rust 1.75+
- (可选) CUDA Toolkit 11.8+ 用于 GPU 加速

### 2. 安装与运行

```bash
# 克隆项目
git clone https://github.com/your-repo/vecboost.git
cd vecboost

# 配置环境变量
cp .env.example .env

# 运行服务 (首次运行会自动下载模型，约 500MB-1GB)
cargo run --release
```

## API 使用

### 文本向量化

```bash
curl -X POST http://localhost:8080/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "rust is fast"]}'
```

### 相似度计算

```bash
curl -X POST http://localhost:8080/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["hello world", "rust is great"],
    "metric": "cosine"
  }'
```

支持以下相似度度量方式：
- `cosine` - 余弦相似度（默认）
- `euclidean` - 欧几里得距离
- `dot` - 点积
- `manhattan` - 曼哈顿距离

## 性能监控

服务内置性能指标收集功能，可通过 API 获取：

```bash
curl http://localhost:8080/api/v1/metrics
```

监控指标包括：
- 推理延迟（P50/P95/P99）
- 吞吐量（请求/秒）
- 错误率
- 模型使用统计
- 资源利用率（CPU/内存）

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│                  (axum web framework)                        │
├─────────────────────────────────────────────────────────────┤
│                      Service Layer                           │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│    │ EmbedService │  │SimilaritySvc │  │MetricsCollector│   │
│    └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                             │
│                  (Candle ONNX Runtime)                       │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│    │   BGE-M3     │  │   TextChunker│  │  Aggregator  │     │
│    └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                     Utils Layer                              │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│    │InputValidator│  │SimilarityFunc│  │  Config      │     │
│    └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```
