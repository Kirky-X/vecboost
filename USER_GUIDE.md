<div align="center">

# 📖 User Guide

### 完整的 vecboost 使用指南

[🏠 首页](../README.md) • [📚 文档](README.md) • [🎯 示例](../examples/) • [❓ 常见问题](FAQ.md)

---

</div>

## 📋 目录

- [简介](#简介)
- [快速入门](#快速入门)
  - [先决条件](#先决条件)
  - [安装](#安装)
  - [第一步](#第一步)
- [核心概念](#核心概念)
- [基础用法](#基础用法)
  - [单文本嵌入](#单文本嵌入)
  - [批量文本嵌入](#批量文本嵌入)
  - [相似度计算](#相似度计算)
- [高级用法](#高级用法)
  - [文件嵌入](#文件嵌入)
  - [模型切换](#模型切换)
  - [性能优化](#性能优化)
  - [监控与日志](#监控与日志)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)
- [后续步骤](#后续步骤)

---

## 简介

<div align="center">

### 🎯 你将学到什么

</div>

<table>
<tr>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/rocket.png" width="64"><br>
<b>快速入门</b><br>
10 分钟内完成向量嵌入服务搭建
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/settings.png" width="64"><br>
<b>多引擎支持</b><br>
Candle 和 ONNX Runtime 双引擎
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/code.png" width="64"><br>
<b>最佳实践</b><br>
高性能向量嵌入服务的使用指南
</td>
<td width="25%" align="center">
<img src="https://img.icons8.com/fluency/96/000000/rocket-take-off.png" width="64"><br>
<b>高级特性</b><br>
GPU 加速、文件嵌入、批量处理
</td>
</tr>
</table>

**vecboost** 是一个高性能的 Rust 向量嵌入服务，旨在为各种 AI 应用提供快速、可靠的向量嵌入生成和相似度计算能力。它支持多引擎（Candle、ONNX Runtime）和 GPU 加速（CUDA、Metal），并提供简洁的 gRPC API 接口。

> 💡 **提示**: 本指南假设你具备基本的 Rust 知识和向量嵌入的概念。如果你是 Rust 新手，建议先阅读 [Rust 官方教程](https://doc.rust-lang.org/book/)。

---

## 快速入门

### 先决条件

在开始之前，请确保你已安装以下工具：

<table>
<tr>
<td width="50%">

**必选**
- ✅ Rust 1.75+ (stable)
- ✅ Cargo (随 Rust 一起安装)
- ✅ Git

</td>
<td width="50%">

**可选**
- 🔧 支持 Rust 的 IDE (如 VS Code + rust-analyzer)
- 🔧 NVIDIA GPU (用于 CUDA 加速)
- 🔧 Docker (用于容器化部署)

</td>
</tr>
</table>

<details>
<summary><b>🔍 验证安装</b></summary>

```bash
# 检查 Rust 版本
rustc --version
# 预期: rustc 1.75.0 (或更高)

# 检查 Cargo 版本
cargo --version
# 预期: cargo 1.75.0 (或更高)
```

</details>

### 安装

克隆 vecboost 仓库：

```bash
git clone https://github.com/your-org/vecboost.git
cd vecboost
```

安装依赖并构建项目：

```bash
# 基本构建（CPU 版本）
cargo build --release

# 带 CUDA 加速的构建
cargo build --release --features cuda

# 带 ONNX Runtime 支持的构建
cargo build --release --features onnx

# 带所有功能的构建
cargo build --release --features cuda,onnx
```

### 第一步

让我们启动 vecboost 服务并测试基本功能。首先，启动服务：

```bash
# 启动基本服务
cargo run --release

# 或使用预构建的二进制文件
./target/release/vecboost
```

服务将在默认端口 50051 上启动 gRPC 接口。

现在，让我们使用 gRPC 客户端测试文本嵌入功能：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::{EmbedRequest, Empty};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建 gRPC 客户端
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    // 获取当前模型信息
    let model_info = client.get_current_model(tonic::Request::new(Empty {})).await?.into_inner();
    println!("当前模型: {:?}", model_info);
    
    // 测试单文本嵌入
    let request = tonic::Request::new(EmbedRequest {
        text: "Hello, vecboost!",
        normalize: Some(true),
    });
    
    let response = client.embed(request).await?;
    let embedding = response.into_inner();
    println!("嵌入向量维度: {}", embedding.dimension);
    println!("处理时间: {:.2}ms", embedding.processing_time_ms);
    println!("嵌入向量前 5 个元素: {:?}", &embedding.embedding[0..5]);
    
    Ok(())
}
```

---

## 核心概念

理解这些核心概念将帮助你更有效地使用 `vecboost`。

### 1️⃣ 向量嵌入

向量嵌入是将文本、图像等非结构化数据转换为低维实数向量的过程。这些向量捕获了原始数据的语义信息，使得可以通过数学方法（如相似度计算）比较不同数据点之间的关系。

### 2️⃣ 嵌入引擎

`vecboost` 支持两种嵌入引擎：
- **Candle**: Rust 原生的机器学习框架，轻量高效
- **ONNX Runtime**: 支持各种预训练模型的通用推理引擎

### 3️⃣ 批量处理

批量处理允许一次性处理多个文本，显著提高处理效率。`vecboost` 提供了 `EmbedBatch` API 来支持批量文本嵌入。

### 4️⃣ 相似度计算

相似度计算用于衡量两个向量之间的语义相似性。`vecboost` 支持多种相似度指标，包括余弦相似度、欧氏距离等。

---

## 基础用法

### 单文本嵌入

单文本嵌入用于将单个文本转换为向量表示：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::EmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    let request = tonic::Request::new(EmbedRequest {
        text: "这是一个测试文本",
        normalize: Some(true),
    });
    
    let response = client.embed(request).await?;
    let embedding = response.into_inner();
    
    println!("嵌入向量: {:?}", embedding);
    Ok(())
}
```

### 批量文本嵌入

批量文本嵌入用于同时处理多个文本，提高效率：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::BatchEmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    let request = tonic::Request::new(BatchEmbedRequest {
        texts: vec![
            "这是第一个测试文本",
            "这是第二个测试文本",
            "这是第三个测试文本",
        ],
        normalize: Some(true),
    });
    
    let response = client.embed_batch(request).await?;
    let batch_response = response.into_inner();
    
    println!("总处理文本数: {}", batch_response.total_count);
    println!("总处理时间: {:.2}ms", batch_response.processing_time_ms);
    
    for (i, embedding) in batch_response.embeddings.iter().enumerate() {
        println!("文本 {} 嵌入维度: {}", i+1, embedding.dimension);
    }
    
    Ok(())
}
```

### 相似度计算

相似度计算用于比较两个向量的语义相似性：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::{EmbedRequest, SimilarityRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    // 获取两个文本的嵌入向量
    let text1 = "猫是一种哺乳动物";
    let text2 = "狗也是一种哺乳动物";
    let text3 = "太阳是太阳系的中心";
    
    let embed_req1 = EmbedRequest { text: text1.to_string(), normalize: Some(true) };
    let embed_req2 = EmbedRequest { text: text2.to_string(), normalize: Some(true) };
    let embed_req3 = EmbedRequest { text: text3.to_string(), normalize: Some(true) };
    
    let vec1 = client.embed(tonic::Request::new(embed_req1)).await?.into_inner().embedding;
    let vec2 = client.embed(tonic::Request::new(embed_req2)).await?.into_inner().embedding;
    let vec3 = client.embed(tonic::Request::new(embed_req3)).await?.into_inner().embedding;
    
    // 计算相似性
    let sim_req1 = SimilarityRequest { vector1: vec1.clone(), vector2: vec2.clone(), metric: "cosine".to_string() };
    let sim_req2 = SimilarityRequest { vector1: vec1, vector2: vec3, metric: "cosine".to_string() };
    
    let sim1 = client.compute_similarity(tonic::Request::new(sim_req1)).await?.into_inner();
    let sim2 = client.compute_similarity(tonic::Request::new(sim_req2)).await?.into_inner();
    
    println!("\"{}\" 和 \"{}\" 的相似度: {:.4}", text1, text2, sim1.score);
    println!("\"{}\" 和 \"{}\" 的相似度: {:.4}", text1, text3, sim2.score);
    
    Ok(())
}
```

---

## 高级用法

### 文件嵌入

文件嵌入用于将整个文件转换为向量表示，支持文本分块和聚合：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::FileEmbedRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    let request = tonic::Request::new(FileEmbedRequest {
        path: "/path/to/your/document.txt".to_string(),
        mode: Some("document".to_string()), // "document" 或 "paragraph"
        chunk_size: Some(1000),
        overlap: Some(100),
    });
    
    let response = client.embed_file(request).await?;
    let file_response = response.into_inner();
    
    println!("文件嵌入模式: {}", file_response.mode);
    println!("文件统计信息: {:?}", file_response.stats);
    println!("聚合嵌入维度: {}", file_response.embedding.len());
    
    if let Some(paragraphs) = file_response.paragraphs.get(0) {
        println!("段落嵌入示例: 段落 {} - 维度 {}", paragraphs.index, paragraphs.embedding.len());
    }
    
    Ok(())
}
```

### 模型切换

`vecboost` 支持动态切换不同的嵌入模型：

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::{ModelSwitchRequest, Empty};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    // 列出所有可用模型
    let models_response = client.list_models(tonic::Request::new(Empty {})).await?.into_inner();
    println!("可用模型:");
    for model in models_response.models {
        println!("- {}", model.model_name);
    }
    
    // 切换到指定模型
    let switch_request = tonic::Request::new(ModelSwitchRequest {
        model_name: "bge-m3".to_string(),
        engine_type: None, // 自动选择
        device_type: None, // 自动选择
    });
    
    let switch_response = client.model_switch(switch_request).await?;
    println!("模型切换结果: {:?}", switch_response.into_inner());
    
    Ok(())
}
```

### 性能优化

#### GPU 加速

启用 GPU 加速可以显著提高嵌入性能：

```bash
# 启用 CUDA 加速
cargo run --release --features cuda

# 启用 Metal 加速 (macOS)
cargo run --release --features metal
```

#### 批量大小优化

调整批量大小可以平衡吞吐量和延迟：

```rust
// 根据硬件和需求调整批量大小
let batch_size = 32;
let texts = vec!["文本 1", "文本 2", ...]; // 32 个文本

let request = BatchEmbedRequest {
    texts,
    normalize: Some(true),
};
```

### 监控与日志

`vecboost` 提供详细的日志和监控指标：

```bash
# 设置日志级别
export RUST_LOG=vecboost=info

# 启动服务
cargo run --release
```

服务会输出请求处理时间、错误率等关键指标，方便监控和调优。

---

## 最佳实践

<div align="center">

### 🌟 推荐的设计模式

</div>

### ✅ 推荐做法

- **批量处理**: 尽量使用 `EmbedBatch` API 处理多个文本，减少网络开销
- **模型选择**: 根据任务需求选择合适的模型（如语义搜索选择 bge-m3）
- **向量归一化**: 启用 `normalize` 参数，确保向量具有相同的尺度
- **错误处理**: 妥善处理 gRPC 请求错误，实现重试机制
- **监控与告警**: 监控服务的响应时间和错误率，设置合理的告警阈值

### ❌ 避免做法

- **频繁小请求**: 避免大量的单文本嵌入请求，优先使用批量 API
- **不设置超时**: 始终为 gRPC 请求设置合理的超时时间
- **忽略错误**: 生产环境应严格检查错误并实现优雅降级
- **硬编码配置**: 使用环境变量或配置文件管理服务配置

---

## 故障排除

<details>
<summary><b>❓ 问题：服务无法启动</b></summary>

**解决方案：**
1. 检查端口 50051 是否已被占用
2. 检查依赖是否正确安装
3. 查看日志输出获取详细错误信息
4. 确保 GPU 驱动（如果使用）已正确安装

```bash
# 检查端口占用
lsof -i :50051

# 查看详细日志
export RUST_LOG=vecboost=debug
cargo run
```

</details>

<details>
<summary><b>❓ 问题：GPU 加速不工作</b></summary>

**解决方案：**
1. 确保使用 `--features cuda` 或 `--features metal` 构建
2. 检查 GPU 驱动版本是否符合要求
3. 查看日志中的设备初始化信息
4. 验证 CUDA 或 Metal 环境是否正确配置

</details>

<details>
<summary><b>❓ 问题：嵌入质量差</b></summary>

**解决方案：**
1. 尝试切换到更适合的模型（如 bge-m3）
2. 确保输入文本质量，避免过短或无意义的文本
3. 启用向量归一化
4. 对于长文本，尝试使用文件嵌入 API 进行分块处理

</details>

<details>
<summary><b>❓ 问题：性能不佳</b></summary>

**解决方案：**
1. 启用 GPU 加速
2. 增加批量大小
3. 减少网络延迟（如果客户端和服务不在同一机器）
4. 检查系统资源使用情况（CPU、内存、GPU）

</details>

<div align="center">

**💬 仍然需要帮助？** [提交 Issue](../../issues) 或 [访问文档中心](https://github.com/project/vecboost)

</div>

---

## 后续步骤

<div align="center">

### 🎯 继续探索

</div>

<table>
<tr>
<td width="33%" align="center">
<a href="API_REFERENCE.md">
<img src="https://img.icons8.com/fluency/96/000000/graduation-cap.png" width="64"><br>
<b>📚 API 参考</b>
</a><br>
详细的接口说明
</td>
<td width="33%" align="center">
<a href="ARCHITECTURE.md">
<img src="https://img.icons8.com/fluency/96/000000/settings.png" width="64"><br>
<b>🔧 架构设计</b>
</a><br>
深入了解内部机制
</td>
<td width="33%" align="center">
<a href="../examples/">
<img src="https://img.icons8.com/fluency/96/000000/code.png" width="64"><br>
<b>💻 示例代码</b>
</a><br>
真实场景的代码样例
</td>
</tr>
</table>

---

<div align="center">

**[📖 API 文档](https://docs.rs/vecboost)** • **[❓ 常见问题](FAQ.md)** • **[🐛 报告问题](../../issues)**

由 Architecture Team 用 ❤️ 制作

[⬆ 回到顶部](#-用户指南)

</div>