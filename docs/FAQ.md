# 常见问题解答（FAQ）

**项目名称**: VecBoost - Rust 文本向量化模块  
**版本**: v1.0.0  
**最后更新**: 2025-12-26

---

## 1. 通用问题

### 1.1 VecBoost 是什么？

VecBoost 是一个基于 Rust 开发的高性能文本向量化库，支持本地部署的 BGE-M3 等嵌入模型进行推理。它提供了文本语义编码、相似度计算和大文件处理等核心功能，适用于需要文本向量化的各类应用场景。

**核心特性**：

- **纯 Rust 实现**：无 Python 依赖，部署简单，性能卓越
- **双引擎支持**：Candle + ONNX Runtime 双引擎架构，支持故障自动降级
- **流式处理**：支持 GB 级文件内存可控处理，处理时间可预测
- **开箱即用**：批处理、内存优化、并发优化默认开启
- **中文优化**：针对中英文混合场景专项优化

**典型应用场景**包括语义搜索、文本聚类、相似度匹配、推荐系统等。

### 1.2 为什么选择 Rust 而不是 Python？

选择 Rust 作为开发语言主要基于以下考量：

**性能优势**：Rust 的零成本抽象和内存安全特性使其在计算密集型任务中表现出色。根据项目测试，Rust 实现的文本向量化性能可达到 Python 的 3-5 倍，同时内存占用仅为 Python 的三分之一。

**部署简化**：传统 Python 方案需要安装 Python 运行时、依赖库和 CUDA 驱动，部署包体积大且环境兼容性问题多。Rust 编译为单一可执行文件，无需运行时环境，大幅简化部署流程。

**并发支持**：Rust 的 async/await 语法和 tokio 异步运行时天然支持高并发场景，单实例可处理数千并发请求，适合构建高性能服务。

**类型安全**：Rust 的强类型系统和借用检查机制能够在编译期捕获大量潜在错误，提高代码可靠性和可维护性。

### 1.3 支持哪些模型？

当前版本支持以下模型：

| 模型名称 | 引擎 | 维度 | 语言 | 说明 |
|---------|------|------|------|------|
| BGE-M3 | Candle/ONNX | 1024 | 多语言 | 默认模型，支持多语言文本 |
| BGE-Large-zh-v1.5 | ONNX | 1024 | 中文 | 专为中文优化的模型 |
| BGE-Small-zh-v1.5 | ONNX | 512 | 中文 | 轻量级中文模型 |

模型文件需要从 Hugging Face 或 ModelScope 下载，放置在配置的模型目录中。详细模型配置方法请参考[配置说明](#32-如何配置模型路径)。

### 1.4 与 sentence-transformers 的输出是否一致？

VecBoost 追求与 Python sentence-transformers 库输出一致性，目标一致率大于 99.5%。但由于以下因素，可能存在微小差异：

**池化策略差异**：当前默认使用 CLS pooling（取 Transformer 最后输出序列的第一个 token），而部分 Python 实现使用 Mean Pooling（取所有 token 的平均值）。CLS pooling 在大多数场景下表现相当，但特定任务可能需要调整。

**模型精度**：ONNX Runtime 引擎默认使用 float32 精度，与 PyTorch 实现一致。Candle 引擎使用相同的模型格式，理论上输出一致。

**分词器差异**：Rust 版 tokenizers 库与 Python 版略有差异，可能导致 token 划分不同，进而影响最终向量。建议在生产环境中进行输出一致性验证。

---

## 2. 安装与配置

### 2.1 如何安装 VecBoost？

**从源码编译**：

```bash
# 克隆项目
git clone https://github.com/your-org/vecboost.git
cd vecboost

# 编译项目
cargo build --release

# 运行测试
cargo test --lib
```

**依赖环境要求**：

- Rust 1.70 或更高版本
- CMake 3.18 或更高版本（用于编译 Candle）
- 对于 GPU 支持：CUDA 12.x + cuDNN 8.x，或 ROCm 6.0（AMD GPU）

**预编译二进制**：项目稳定后将提供预编译的二进制文件，可直接下载使用。

### 2.2 如何配置模型路径？

VecBoost 支持多种模型配置方式，推荐使用配置文件进行统一管理。

**目录结构要求**：

```
models/
├── bge-m3/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── vocab.txt
└── bge-large-zh/
    └── ...
```

**配置文件示例**（config/app.toml）：

```toml
[model]
name = "bge-m3"
engine = "auto"  # auto/candle/onnx
device = "auto"  # auto/cuda/cpu

[model.paths]
local = "./models/bge-m3"
cache = "~/.cache/vecboost"
hf_mirror = "https://hf-mirror.com"
```

**环境变量配置**：

```bash
# 模型路径
export VECBOOST_MODEL_PATH="./models"

# 设备选择
export VECBOOST_DEVICE="cuda"  # cuda/cpu/auto

# 日志级别
export RUST_LOG="info"
```

### 2.3 如何启用 GPU 支持？

**NVIDIA GPU（CUDA）**：

确保已安装 CUDA 12.x 和 cuDNN 8.x，然后使用 cuda feature 编译：

```bash
cargo build --release --features cuda
```

运行时配置：

```toml
[model]
engine = "candle"
device = "cuda"
```

**AMD GPU（ROCm）**：

ROCm 支持正在开发中，预计下一版本发布。启用后编译方式：

```bash
cargo build --release --features rocm
```

**验证 GPU 识别**：

启动服务后查看日志，确认 GPU 正常识别：

```
INFO vecboost::device::manager: CUDA device detected: NVIDIA GeForce RTX 4090
INFO vecboost::device::manager: GPU memory: 24576 MB total, 2048 MB available
```

### 2.4 配置文件有哪些选项？

VecBoost 支持丰富的配置选项，满足不同部署场景需求。

**应用配置**（app 部分）：

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| host | String | "127.0.0.1" | 服务绑定地址 |
| port | u16 | 8080 | 服务监听端口 |
| workers | usize | cpu_count | HTTP 服务工作线程数 |
| log_level | String | "info" | 日志级别：debug/info/warn/error |

**模型配置**（model 部分）：

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| name | String | "bge-m3" | 模型名称 |
| engine | String | "auto" | 推理引擎：auto/candle/onnx |
| device | String | "auto" | 计算设备：auto/cuda/cpu |
| dimension | usize | 1024 | 输出向量维度 |
| max_batch_size | usize | 32 | 批处理最大批次 |

**资源限制**（resource 部分）：

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| max_memory_mb | usize | 4096 | 最大内存使用（MB） |
| gpu_memory_limit_mb | usize | 8192 | GPU 显存限制（MB） |
| batch_timeout_ms | u64 | 1000 | 批处理超时（毫秒） |

---

## 3. 使用指南

### 3.1 如何调用文本向量化 API？

VecBoost 提供 RESTful API 接口，支持单文本和批量向量化。

**单文本向量化**：

```bash
curl -X POST http://localhost:8080/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "normalize": true}'
```

**响应格式**：

```json
{
  "text_preview": "Hello, world!",
  "embedding": [0.0123, -0.0456, ...],
  "dimension": 1024,
  "processing_time_ms": 15
}
```

**批量向量化**：

```bash
curl -X POST http://localhost:8080/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["文本1", "文本2", "文本3"],
    "normalize": true,
    "aggregation": "average"
  }'
```

**响应格式**：

```json
{
  "embeddings": [
    {"text_preview": "文本1", "embedding": [...]},
    {"text_preview": "文本2", "embedding": [...]},
    {"text_preview": "文本3", "embedding": [...]}
  ],
  "dimension": 1024,
  "processing_time_ms": 42
}
```

### 3.2 如何计算文本相似度？

VecBoost 提供两种相似度计算方式：直接计算和检索模式。

**1对1 相似度计算**：

```bash
curl -X POST http://localhost:8080/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["我喜欢苹果", "我爱吃水果"],
    "metric": "cosine"
  }'
```

**响应**：

```json
{
  "results": [
    {"from": "我喜欢苹果", "to": "我爱吃水果", "score": 0.8234}
  ],
  "processing_time_ms": 32
}
```

**1对N 检索模式**：

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能技术",
    "candidates": [
      {"id": "1", "text": "机器学习是AI的核心"},
      {"id": "2", "text": "今天天气很好"},
      {"id": "3", "text": "深度学习推动了计算机视觉发展"}
    ],
    "metric": "cosine",
    "top_k": 2,
    "threshold": 0.5
  }'
```

**响应**：

```json
{
  "results": [
    {"id": "3", "text": "深度学习推动了计算机视觉发展", "score": 0.9123},
    {"id": "1", "text": "机器学习是AI的核心", "score": 0.7567}
  ],
  "processing_time_ms": 28
}
```

**相似度度量方式**：

| 方式 | 描述 | 取值范围 | 适用场景 |
|-----|------|---------|---------|
| cosine | 余弦相似度 | [0, 1] | 通用，推荐默认 |
| euclidean | 欧氏距离 | [0, +∞) | 距离敏感场景 |
| dot | 点积 | (-∞, +∞) | 需配合归一化使用 |
| manhattan | 曼哈顿距离 | [0, +∞) | 高维稀疏向量 |

### 3.3 如何处理大文件？

对于 GB 级大文件处理，VecBoost 提供流式 API 和分块策略。

**文件向量化请求**：

```bash
curl -X POST http://localhost:8080/api/v1/embed/file \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/large_file.txt",
    "chunk_size": 512,
    "overlap": 50,
    "aggregation": "average",
    "output_mode": "chunk"  # chunk / document
  }'
```

**分块策略配置**：

| 策略 | 参数 | 说明 |
|-----|------|------|
| 滑动窗口 | chunk_size, overlap | 固定窗口滑动，支持重叠 |
| 段落 | separator | 按段落分隔，自动切分 |
| 固定大小 | chunk_size | 按字符数切分 |

**聚合方式**：

| 方式 | 说明 |
|-----|------|
| average | 所有 chunk 向量取平均 |
| max_pooling | 所有 chunk 向量取最大值 |
| min_pooling | 所有 chunk 向量取最小值 |
| weighted | 加权平均，边缘块权重降低 |

### 3.4 如何动态切换模型？

VecBoost 支持运行时动态切换模型，无需重启服务。

**获取可用模型列表**：

```bash
curl http://localhost:8080/api/v1/models
```

**响应**：

```json
{
  "current_model": "bge-m3",
  "available_models": [
    {"name": "bge-m3", "dimension": 1024, "engine": "candle"},
    {"name": "bge-large-zh", "dimension": 1024, "engine": "onnx"}
  ]
}
```

**切换模型**：

```bash
curl -X POST http://localhost:8080/api/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-large-zh"}'
```

**响应**：

```json
{
  "success": true,
  "from_model": "bge-m3",
  "to_model": "bge-large-zh",
  "switch_time_ms": 1500
}
```

### 3.5 如何使用 SDK？

除了 REST API，VecBoost 还提供 Rust SDK 直接集成到项目中。

**添加依赖**：

```toml
[dependencies]
vecboost = { version = "0.1", features = ["onnx"] }
```

**使用示例**：

```rust
use vecboost::{EmbeddingService, EmbeddingRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化服务
    let service = EmbeddingService::new("./config/app.toml").await?;
    
    // 单文本向量化
    let request = EmbeddingRequest {
        text: "Hello, world!".to_string(),
        normalize: true,
        ..Default::default()
    };
    let response = service.process_text(request).await?;
    println!("Vector dimension: {}", response.embedding.len());
    
    // 批量向量化
    let batch_request = vec![
        EmbeddingRequest { text: "文本1".to_string(), ..Default::default() },
        EmbeddingRequest { text: "文本2".to_string(), ..Default::default() },
    ];
    let batch_response = service.process_batch(batch_request).await?;
    
    // 相似度计算
    let similarity = service.process_similarity(
        vec!["我喜欢苹果".to_string(), "我爱吃水果".to_string()],
        "cosine".to_string(),
    ).await?;
    
    Ok(())
}
```

---

## 4. 性能优化

### 4.1 如何提升批量处理性能？

批量处理是提升吞吐量的关键手段。以下是优化建议：

**增大批次大小**：

在配置文件中调整 max_batch_size：

```toml
[model]
max_batch_size = 64  # 根据 GPU 显存调整
```

**使用并行处理**：

VecBoost 默认启用 tokio 异步并发处理多批次任务，确保配置足够的工作线程：

```toml
[app]
workers = 8  # 与 CPU 核心数匹配
```

**批处理 API 调用**：

```bash
# 单次批量调用（推荐）
curl -X POST http://localhost:8080/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["文本1", "文本2", ...]}'
```

避免在客户端循环调用单条请求，这会产生大量 HTTP 开销。

### 4.2 如何优化 GPU 利用率？

**启用 GPU 加速**：

确保使用 cuda feature 编译，并在配置中指定 GPU 设备：

```bash
cargo build --release --features cuda
```

```toml
[model]
engine = "candle"
device = "cuda"
```

**调整批处理大小**：

根据 GPU 显存调整批次大小，避免显存溢出：

```toml
[model]
max_batch_size = 32  # 约占用 4GB 显存
```

**监控 GPU 状态**：

查看服务日志获取 GPU 利用率信息：

```
INFO vecboost::device::cuda: GPU utilization: 78%
INFO vecboost::device::memory: GPU memory: 8192 MB / 24576 MB (33%)
```

### 4.3 内存使用如何优化？

**设置内存限制**：

在配置中设置合理的内存上限：

```toml
[resource]
max_memory_mb = 4096
```

**监控内存状态**：

```bash
curl http://localhost:8080/api/v1/health/memory
```

**响应**：

```json
{
  "status": "ok",
  "used_mb": 2048,
  "limit_mb": 4096,
  "percentage": 50.0
}
```

**大文件处理建议**：

- 使用流式 API 分块处理
- 设置合理的 chunk_size（512-1024 tokens）
- 启用 overlap 策略确保跨块语义连贯

### 4.4 如何调整并发连接数？

HTTP 服务默认配置适合中小规模场景，可根据需要调整：

```toml
[app]
workers = 4  # HTTP 工作线程数

[server]
max_connections = 10000  # 最大并发连接数
request_timeout_ms = 30000  # 请求超时时间
```

对于高并发场景，建议：

- 使用反向代理（Nginx/HAProxy）分担负载
- 启用连接复用（HTTP keep-alive）
- 考虑部署多个服务实例

---

## 5. 故障排查

### 5.1 服务无法启动怎么办？

**常见原因及解决方案**：

**端口被占用**：

```bash
# 检查端口占用
lsof -i :8080

# 更换端口或kill占用进程
kill $(lsof -t -i:8080)
```

**模型文件缺失**：

```
ERROR vecboost::model: Model file not found: ./models/bge-m3/model.safetensors
```

确认模型文件路径配置正确，且文件存在。

**CUDA 环境问题**：

```
ERROR vecboost::device: CUDA driver not found
```

检查 CUDA 安装：

```bash
nvidia-smi
nvcc --version
```

确认环境变量 LD_LIBRARY_PATH 包含 CUDA 库路径。

**内存不足**：

```
ERROR vecboost::resource: Out of memory
```

降低批处理大小或增加系统内存。

### 5.2 向量化结果异常如何排查？

**输出全为零**：

可能原因：模型文件损坏或分词器配置错误。

排查步骤：

1. 检查模型文件完整性（SHA256 校验）
2. 验证分词器配置
3. 查看服务日志详细输出

**向量维度不匹配**：

```bash
# 确认模型配置维度
curl http://localhost:8080/api/v1/models
```

**相似度计算结果异常**：

确认输入向量已归一化（normalize=true），尤其在使用点积（dot）度量时。

### 5.3 GPU 相关问题

**GPU 内存不足**：

```
ERROR vecboost::device: Out of GPU memory
```

解决方案：

- 减小批处理大小
- 使用 CPU 回退模式
- 启用内存限制检查

**CUDA 版本不兼容**：

```
ERROR vecboost::device: CUDA version mismatch
```

确保 CUDA 版本与编译时使用的版本一致。推荐使用 CUDA 12.x。

**NVIDIA 驱动过旧**：

```
ERROR vecboost::device: NVIDIA driver version too old
```

升级 NVIDIA 驱动至 525.x 或更新版本。

### 5.4 日志如何查看？

**启用详细日志**：

```bash
export RUST_LOG=debug
./vecboost
```

**日志文件配置**：

```toml
[logging]
level = "info"
format = "json"
output = "file"
path = "./logs/vecboost.log"
```

**查看最近日志**：

```bash
tail -f ./logs/vecboost.log
```

### 5.5 如何报告问题？

遇到问题时，请提供以下信息以便快速定位：

**环境信息**：

```bash
# 系统信息
uname -a
cat /etc/os-release

# Rust 版本
rustc --version
cargo --version

# GPU 信息
nvidia-smi

# VecBoost 版本
./vecboost --version
```

**问题描述**：

- 复现步骤
- 期望行为
- 实际行为
- 错误日志

**提交方式**：

在 GitHub Issues 页面提交问题，标题格式：`[Bug] 简要描述`

---

## 6. 最佳实践

### 6.1 生产环境部署建议

**服务配置**：

```toml
[app]
host = "0.0.0.0"
port = 8080
workers = 8  # 与 CPU 核心数一致

[model]
engine = "auto"
device = "cuda"
max_batch_size = 32

[resource]
max_memory_mb = 8192
gpu_memory_limit_mb = 6144

[logging]
level = "info"
format = "json"
```

**进程管理**：使用 systemd 或 supervisord 管理服务进程。

**健康检查**：配置负载均衡器健康检查端点：

```bash
curl http://localhost:8080/api/v1/health
```

**响应**：

```json
{
  "status": "healthy",
  "model": "bge-m3",
  "device": "cuda",
  "uptime_seconds": 86400
}
```

### 6.2 安全建议

**网络隔离**：

- 生产环境不应暴露管理 API
- 使用防火墙限制访问来源
- 启用 HTTPS（通过反向代理）

**输入验证**：

- 对输入文本长度进行限制
- 启用请求体验证中间件
- 防止恶意请求耗尽资源

**资源限制**：

```toml
[resource]
max_memory_mb = 4096
max_request_size_mb = 10
max_batch_size = 100
```

### 6.3 监控告警

**关键指标**：

| 指标 | 阈值 | 告警级别 |
|-----|------|---------|
| 内存使用率 | > 80% | warning |
| 内存使用率 | > 95% | critical |
| GPU 显存使用率 | > 90% | warning |
| 请求延迟 P99 | > 500ms | warning |
| 错误率 | > 1% | warning |

**监控端点**：

```bash
# 获取指标
curl http://localhost:8080/api/v1/metrics
```

### 6.4 备份与恢复

**模型文件备份**：

```bash
# 备份模型目录
tar -czvf models-backup-$(date +%Y%m%d).tar.gz models/

# 备份配置文件
cp config/app.toml config/app.toml.backup
```

**恢复操作**：

```bash
# 恢复模型
tar -xzvf models-backup-20241226.tar.gz -C /path/to/vecboost/

# 恢复配置
cp config/app.toml.backup config/app.toml
```

---

## 7. 高级功能

### 7.1 自定义聚合策略

VecBoost 支持自定义 embedding 聚合策略，适用于特殊场景需求。

**内置聚合器**：

| 聚合器 | 说明 | 适用场景 |
|-------|------|---------|
| average | 平均池化 | 通用场景 |
| max_pooling | 最大池化 | 特征选择 |
| min_pooling | 最小池化 | 特征过滤 |
| cls | CLS token | BERT 类模型 |

**自定义聚合器实现**（示例）：

```rust
use vecboost::aggregator::{EmbeddingAggregator, AggregatorConfig};

struct WeightedAggregator {
    weights: Vec<f32>,
}

#[async_trait::async_trait]
impl EmbeddingAggregator for WeightedAggregator {
    async fn aggregate(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<f32>, AppError> {
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        
        for (i, emb) in embeddings.iter().enumerate() {
            let weight = self.weights.get(i).copied().unwrap_or(1.0);
            for (j, val) in emb.iter().enumerate() {
                result[j] += val * weight;
            }
        }
        
        // 归一化
        normalize_l2(&mut result);
        
        Ok(result)
    }
}
```

### 7.2 模型微调支持

当前版本不直接支持模型微调，但可通过以下方式扩展：

**导出 ONNX 模型**：

将微调后的 PyTorch 模型导出为 ONNX 格式：

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("your-finetuned-model")
model.eval()

dummy_input = torch.randint(0, 1000, (1, 512))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"}
    }
)
```

**加载自定义模型**：

```toml
[model]
name = "custom-model"
engine = "onnx"
dimension = 1024

[model.paths]
local = "./models/custom-model"
```

### 7.3 扩展新引擎

VecBoost 架构支持扩展新的推理引擎。

**实现 InferenceEngine trait**：

```rust
use async_trait::async_trait;
use crate::{AppError, Embedding};

#[async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn load(&mut self, model_path: &Path) -> Result<(), AppError>;
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError>;
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;
    fn get_dimension(&self) -> usize;
    fn health_check(&self) -> bool;
}
```

**注册新引擎**：

在 `src/engine/mod.rs` 中注册：

```rust
pub enum AnyEngine {
    Candle(CandleEngine),
    Onnx(OnnxEngine),
    Custom(CustomEngine),  // 新增
}
```

### 7.4 分布式部署

**水平扩展**：

VecBoost 支持无状态部署，可通过负载均衡器扩展多个实例。

**配置建议**：

```toml
[app]
host = "0.0.0.0"
port = 8080

# 禁用本地缓存，使用共享缓存
[cache]
type = "redis"  # 即将支持
```

**部署架构**：

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
    │  VecBoost │      │  VecBoost │      │  VecBoost │
    │  Node 1   │      │  Node 2   │      │  Node 3   │
    └───────────┘      └───────────┘      └───────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared Storage │
                    │ (Model Files)   │
                    └─────────────────┘
```

---

## 8. 常见术语表

| 术语 | 英文 | 说明 |
|-----|------|------|
| 向量化 | Vectorization | 将文本转换为高维向量表示的过程 |
| 嵌入 | Embedding | 文本的向量表示形式 |
| 池化 | Pooling | 将序列向量聚合为单一向量的操作 |
| 批处理 | Batch Processing | 一次性处理多个输入以提高效率 |
| 推理 | Inference | 使用模型进行预测的过程 |
| 分词 | Tokenization | 将文本切分为模型可处理的 token 序列 |
| 归一化 | Normalization | 将向量长度调整为 1 的操作 |
| 余弦相似度 | Cosine Similarity | 衡量两个向量方向相似程度的指标 |
| 欧氏距离 | Euclidean Distance | 衡量两个向量在空间中距离的指标 |
