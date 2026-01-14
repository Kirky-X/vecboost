<div align="center">

# 📚 VecBoost API 参考文档

**完整的 REST HTTP 端点和 gRPC 服务方法文档**

[![Version 0.1.0](https://img.shields.io/badge/Version-0.1.0-green.svg?style=for-the-badge)](https://github.com/Kirky-X/vecboost) [![REST API](https://img.shields.io/badge/REST-API-9002-blue.svg?style=for-the-badge)](http://localhost:9002) [![gRPC](https://img.shields.io/badge/gRPC-50051-green.svg?style=for-the-badge)](localhost:50051)

*VecBoost API 的完整文档，包括 REST HTTP 端点和 gRPC 服务方法。*

</div>

---

## 📋 目录

| 章节 | 说明 |
|------|------|
| [基础 URL](#基础-url) | API 端点基础地址 |
| [认证](#认证) | JWT 认证和令牌管理 |
| [REST API](#rest-api) | HTTP REST 接口文档 |
| [gRPC API](#grpc-api) | gRPC 服务定义和消息类型 |
| [错误处理](#错误处理) | 错误码和响应格式 |
| [速率限制](#速率限制) | 速率限制策略和响应头 |

---

## 🌍 基础 URL

| 环境 | 协议 | URL | 端口 |
|------|------|-----|------|
| **REST API** | HTTP | `http://localhost:9002` | `9002` |
| **gRPC API** | HTTP/2 | `localhost:50051` | `50051` |
| **Prometheus** | HTTP | `http://localhost:9090` | `9090` |

> **💡 提示**: 所有 REST API 端点都以 `/api/v1/` 为前缀。

---

## 🔐 认证

启用认证时，请在 `Authorization` 头中包含 Bearer 令牌：

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -d '{"text": "Hello, world!"}'
```

### 获取令牌

**端点:** `POST /api/v1/auth/login`

**请求体:**

```json
{
  "username": "admin",
  "password": "Secure@Passw0rd!"
}
```

**响应:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `access_token` | string | JWT 访问令牌 |
| `token_type` | string | 令牌类型（始终为 `bearer`） |
| `expires_in` | integer | 令牌过期时间（秒） |

> **⚠️ 注意**: 令牌默认 1 小时后过期，可在配置中调整。

---

## 🌐 REST API

### 嵌入向量

#### 生成嵌入向量

为单个文本生成向量嵌入。

**端点:** `POST /api/v1/embed`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 要嵌入的文本 |
| `normalize` | boolean | ❌ | 是否归一化向量（默认: false） |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "normalize": true
  }'
```

**响应:**

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `embedding` | array | 嵌入向量数组 |
| `dimension` | integer | 向量维度 |
| `processing_time_ms` | number | 处理时间（毫秒） |

---

#### 批量嵌入

在单个请求中为多个文本生成嵌入向量。

**端点:** `POST /api/v1/embed/batch`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `texts` | array | ✅ | 文本数组 |
| `normalize` | boolean | ❌ | 是否归一化向量 |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["第一个文档", "第二个文档", "第三个文档"],
    "normalize": true
  }'
```

**响应:**

```json
{
  "embeddings": [
    {
      "embedding": [...],
      "dimension": 1024,
      "processing_time_ms": 12.3
    },
    {
      "embedding": [...],
      "dimension": 1024,
      "processing_time_ms": 11.8
    }
  ],
  "total_count": 2,
  "processing_time_ms": 25.5
}
```

---

#### 文件嵌入

为文件生成嵌入向量。

**端点:** `POST /api/v1/embed/file`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | ✅ | 文件路径 |
| `mode` | string | ❌ | 嵌入模式 (`paragraph` 或 `chunk`) |
| `chunk_size` | integer | ❌ | 分块大小（默认: 512） |
| `overlap` | integer | ❌ | 重叠大小（默认: 50） |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/embed/file \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/document.txt",
    "mode": "paragraph"
  }'
```

**响应:**

```json
{
  "mode": "paragraph",
  "stats": {
    "total_lines": 150,
    "total_chars": 5000,
    "total_paragraphs": 25,
    "processed_chunks": 25,
    "processing_time_ms": 150.5
  },
  "embedding": [0.123, ...],
  "paragraphs": [
    {
      "index": 0,
      "text": "First paragraph...",
      "embedding": [0.456, ...]
    }
  ]
}
```

---

### 相似度计算

#### 计算相似度

计算两个向量之间的相似度。

**端点:** `POST /api/v1/similarity`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `vector1` | array | ✅ | 第一个向量 |
| `vector2` | array | ✅ | 第二个向量 |
| `metric` | string | ❌ | 相似度度量 (`cosine`, `euclidean`, `dot_product`, `manhattan`) |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "vector1": [0.1, 0.2, 0.3, ...],
    "vector2": [0.1, 0.2, 0.3, ...],
    "metric": "cosine"
  }'
```

**响应:**

```json
{
  "score": 0.9876,
  "metric": "cosine"
}
```

---

#### 相似文档搜索

从文档集合中找到最相似的向量。

**端点:** `POST /api/v1/search`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 搜索查询文本 |
| `documents` | array | ✅ | 文档数组 |
| `top_k` | integer | ❌ | 返回结果数量（默认: 5） |
| `metric` | string | ❌ | 相似度度量 |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI 技术发展",
    "documents": [
      "关于人工智能的文档",
      "关于机器学习的文档",
      "关于深度学习的文档"
    ],
    "top_k": 2,
    "metric": "cosine"
  }'
```

**响应:**

```json
{
  "results": [
    {
      "index": 0,
      "text": "关于人工智能的文档",
      "score": 0.95
    },
    {
      "index": 1,
      "text": "关于机器学习的文档",
      "score": 0.87
    }
  ],
  "query_embedding": [0.123, ...]
}
```

---

### 模型管理

#### 获取当前模型

获取当前加载模型的信息。

**端点:** `GET /api/v1/model`

**响应:**

```json
{
  "name": "BAAI/bge-m3",
  "engine_type": "candle",
  "device_type": "cuda",
  "dimension": 1024,
  "precision": "fp32",
  "max_batch_size": 32,
  "cache_enabled": true,
  "cache_size": 1024
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 模型名称（HuggingFace ID） |
| `engine_type` | string | 引擎类型 (`candle` 或 `onnx`) |
| `device_type` | string | 设备类型 (`cpu`, `cuda`, `metal`) |
| `dimension` | integer | 嵌入向量维度 |
| `precision` | string | 模型精度 (`fp16`, `fp32`) |
| `max_batch_size` | integer | 最大批处理大小 |

---

#### 列出可用模型

列出所有可用模型。

**端点:** `GET /api/v1/models`

**响应:**

```json
{
  "models": [
    {
      "name": "BAAI/bge-m3",
      "version": "main",
      "dimension": 1024,
      "supported_devices": ["cpu", "cuda", "metal"]
    },
    {
      "name": "BAAI/bge-small-en-v1.5",
      "version": "main",
      "dimension": 384,
      "supported_devices": ["cpu", "cuda", "metal"]
    }
  ],
  "current_model": "BAAI/bge-m3"
}
```

---

#### 切换模型

切换到不同的模型。

**端点:** `POST /api/v1/model/switch`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | ✅ | 模型名称 |
| `engine_type` | string | ❌ | 引擎类型 (`candle`, `onnx`) |
| `device_type` | string | ❌ | 设备类型 (`auto`, `cpu`, `cuda`, `metal`) |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/model/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model_name": "BAAI/bge-small-en-v1.5",
    "engine_type": "candle",
    "device_type": "auto"
  }'
```

**响应:**

```json
{
  "success": true,
  "message": "Model switched successfully",
  "model_info": {
    "name": "BAAI/bge-small-en-v1.5",
    "dimension": 384
  }
}
```

---

### 健康检查

#### 健康检查

检查服务健康状态。

**端点:** `GET /health`

**响应:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime": "2h30m45s",
  "model_loaded": "BAAI/bge-m3"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 健康状态 (`healthy`, `degraded`, `unhealthy`) |
| `version` | string | 服务版本 |
| `uptime` | string | 运行时间 |
| `model_loaded` | string | 当前加载的模型名称 |

---

#### 就绪检查

检查服务是否准备好接收请求。

**端点:** `GET /ready`

**响应:**

```json
{
  "ready": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `ready` | boolean | 是否准备好接收请求 |

---

#### Prometheus 指标

Prometheus 指标端点。

**端点:** `GET /metrics`

| 指标名称 | 类型 | 说明 |
|----------|------|------|
| `vecboost_requests_total` | counter | 总请求数 |
| `vecboost_embedding_latency_seconds` | histogram | 嵌入延迟分布 |
| `vecboost_cache_hit_ratio` | gauge | 缓存命中率 |
| `vecboost_batch_size` | histogram | 批处理大小分布 |
| `vecboost_model_load_duration_seconds` | histogram | 模型加载时间 |
| `vecboost_active_requests` | gauge | 活跃请求数 |

**示例查询:**

```promql
# 请求率
rate(vecboost_requests_total[5m])

# p99 延迟
histogram_quantile(0.99, rate(vecboost_embedding_latency_seconds_bucket[5m]))

# 缓存命中率
vecboost_cache_hit_ratio
```

---

## 🔌 gRPC API

### 服务定义

```protobuf
syntax = "proto3";

package vecboost;

service EmbeddingService {
  // 嵌入相关
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(BatchEmbedRequest) returns (BatchEmbedResponse);
  rpc EmbedFile(FileEmbedRequest) returns (FileEmbedResponse);
  
  // 相似度计算
  rpc ComputeSimilarity(SimilarityRequest) returns (SimilarityResponse);
  
  // 模型管理
  rpc ModelSwitch(ModelSwitchRequest) returns (ModelSwitchResponse);
  rpc GetCurrentModel(Empty) returns (ModelInfo);
  rpc GetModelInfo(Empty) returns (ModelMetadata);
  rpc ListModels(Empty) returns (ModelListResponse);
  
  // 健康检查
  rpc HealthCheck(Empty) returns (HealthResponse);
}
```

> **💡 提示**: 使用 `proto/embedding.proto` 文件生成客户端存根。

---

### 服务方法概览

| 方法 | 输入类型 | 输出类型 | 说明 |
|------|----------|----------|------|
| `Embed` | `EmbedRequest` | `EmbedResponse` | 生成单个嵌入向量 |
| `EmbedBatch` | `BatchEmbedRequest` | `BatchEmbedResponse` | 批量生成嵌入向量 |
| `EmbedFile` | `FileEmbedRequest` | `FileEmbedResponse` | 文件嵌入 |
| `ComputeSimilarity` | `SimilarityRequest` | `SimilarityResponse` | 计算相似度 |
| `ModelSwitch` | `ModelSwitchRequest` | `ModelSwitchResponse` | 切换模型 |
| `GetCurrentModel` | `Empty` | `ModelInfo` | 获取当前模型信息 |
| `GetModelInfo` | `Empty` | `ModelMetadata` | 获取模型元数据 |
| `ListModels` | `Empty` | `ModelListResponse` | 列出可用模型 |
| `HealthCheck` | `Empty` | `HealthResponse` | 健康检查 |

---

### 消息类型定义

#### 嵌入请求/响应

```protobuf
message EmbedRequest {
  string text = 1;
  bool normalize = 2;
}

message EmbedResponse {
  repeated float embedding = 1;
  int64 dimension = 2;
  double processing_time_ms = 3;
}

message BatchEmbedRequest {
  repeated string texts = 1;
  bool normalize = 2;
}

message BatchEmbedResponse {
  repeated EmbedResponse embeddings = 1;
  int64 total_count = 2;
  double processing_time_ms = 3;
}
```

#### 相似度请求/响应

```protobuf
message SimilarityRequest {
  repeated float vector1 = 1;
  repeated float vector2 = 2;
  string metric = 3;  // cosine, euclidean, dot_product, manhattan
}

message SimilarityResponse {
  double score = 1;
  string metric = 2;
}
```

#### 文件嵌入

```protobuf
message FileEmbedRequest {
  string path = 1;
  string mode = 2;       // paragraph | chunk
  int32 chunk_size = 3;
  int32 overlap = 4;
}

message FileEmbedResponse {
  string mode = 1;
  FileStats stats = 2;
  repeated float embedding = 3;
  repeated ParagraphEmbedding paragraphs = 4;
}

message FileStats {
  int64 total_lines = 1;
  int64 total_chars = 2;
  int64 total_paragraphs = 3;
  int64 processed_chunks = 4;
  double processing_time_ms = 5;
}

message ParagraphEmbedding {
  int32 index = 1;
  string text = 2;
  repeated float embedding = 3;
}
```

#### 模型管理

```protobuf
message ModelSwitchRequest {
  string model_name = 1;
  string engine_type = 2;   // candle | onnx
  string device_type = 3;   // auto | cpu | cuda | metal
}

message ModelSwitchResponse {
  bool success = 1;
  string message = 2;
  ModelInfo model_info = 3;
}

message ModelInfo {
  string name = 1;
  string engine_type = 2;
  string device_type = 3;
  int64 dimension = 4;
  string precision = 5;
  int64 max_batch_size = 6;
  bool cache_enabled = 7;
  int64 cache_size = 8;
}

message ModelMetadata {
  string model_name = 1;
  string version = 2;
  string architecture = 3;
  int64 max_position_embeddings = 4;
  int64 vocab_size = 5;
  int64 hidden_size = 6;
  int64 num_hidden_layers = 7;
  int64 num_attention_heads = 8;
  int64 intermediate_size = 9;
  repeated string supported_devices = 10;
  repeated string supported_precisions = 11;
}

message ModelListResponse {
  repeated ModelMetadata models = 1;
  string current_model = 2;
}
```

#### 健康检查

```protobuf
message HealthResponse {
  string status = 1;
  string version = 2;
  string uptime = 3;
  string model_loaded = 4;
}

message Empty {}
```

---

### SDK 使用示例

#### Python

```python
import grpc
import embedding_pb2
import embedding_pb2_grpc

# 连接 gRPC 服务
channel = grpc.insecure_channel('localhost:50051')
stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)

# 单个嵌入请求
request = embedding_pb2.EmbedRequest(
    text="Hello, world!",
    normalize=True
)
response = stub.Embed(request)
print(f"Embedding dimension: {response.dimension}")
print(f"Processing time: {response.processing_time_ms:.2f}ms")

# 批量嵌入请求
batch_request = embedding_pb2.BatchEmbedRequest(
    texts=["文档1", "文档2", "文档3"],
    normalize=True
)
batch_response = stub.EmbedBatch(batch_request)
print(f"Processed {batch_response.total_count} embeddings")
```

#### Go

```go
import (
    "context"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    pb "vecboost/proto"
)

func main() {
    // 连接 gRPC 服务
    conn, err := grpc.Dial("localhost:50051", 
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    client := pb.NewEmbeddingServiceClient(conn)
    
    // 单个嵌入请求
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    resp, err := client.Embed(ctx, &pb.EmbedRequest{
        Text: "Hello, world!",
        Normalize: true,
    })
    if err != nil {
        log.Fatalf("Embed failed: %v", err)
    }
    
    fmt.Printf("Dimension: %d, Time: %.2fms\n", 
        resp.Dimension, resp.ProcessingTimeMs)
}
```

---

## ⚠️ 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| `200` | ✅ 成功 |
| `400` | ❌ 请求参数错误 |
| `401` | 🔒 未授权（缺少或无效令牌） |
| `403` | 🚫 禁止访问（权限不足） |
| `429` | ⚡ 请求过于频繁（速率限制） |
| `500` | 💥 服务器内部错误 |
| `503` | ⏸️ 服务不可用 |

---

### 错误响应格式

所有错误响应遵循统一格式：

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text input cannot be empty",
    "details": null
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | string | 错误码 |
| `message` | string | 错误描述 |
| `details` | object | 错误详情（可选） |

---

### 常见错误码

| 错误码 | HTTP 状态码 | 说明 | 解决方案 |
|--------|-------------|------|----------|
| `INVALID_INPUT` | 400 | 请求参数无效 | 检查请求体格式 |
| `UNAUTHORIZED` | 401 | 认证失败 | 获取并使用有效令牌 |
| `FORBIDDEN` | 403 | 权限不足 | 联系管理员提升权限 |
| `RATE_LIMITED` | 429 | 超出速率限制 | 使用指数退避重试 |
| `MODEL_NOT_FOUND` | 404 | 模型不存在 | 检查模型名称 |
| `INFERENCE_ERROR` | 500 | 推理失败 | 检查模型状态 |
| `GPU_OOM` | 500 | GPU 内存不足 | 减小批处理大小或使用 CPU |
| `FILE_NOT_FOUND` | 404 | 文件不存在 | 检查文件路径 |
| `CONFIG_ERROR` | 500 | 配置错误 | 检查配置文件 |

> **💡 提示**: 启用认证时，401 错误也可能表示令牌已过期。

---

## ⚡ 速率限制

### 默认限制策略

| 限制类型 | 请求数 | 时间窗口 | 适用场景 |
|----------|--------|----------|----------|
| **全局** | 1,000 | 每分钟 | 保护整体服务 |
| **每 IP** | 100 | 每分钟 | 防止单 IP 攻击 |
| **每用户** | 200 | 每分钟 | 用户级别限制 |
| **每 API Key** | 500 | 每分钟 | API 密钥级别 |

---

### 速率限制响应头

所有响应包含速率限制信息：

| 头信息 | 说明 |
|--------|------|
| `X-RateLimit-Limit` | 当前限制的最大请求数 |
| `X-RateLimit-Remaining` | 剩余请求数 |
| `X-RateLimit-Reset` | 限制重置时间戳（Unix） |

**示例:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

### 超出速率限制

当超出速率限制时，返回 `429 Too Many Requests` 错误：

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Try again in 30 seconds.",
    "details": {
      "retry_after": 30,
      "limit": 100,
      "current": 100
    }
  }
}
```

---

### 自定义速率限制

在配置文件中调整速率限制：

```toml
[rate_limit]
enabled = true

[rate_limit.global]
requests = 2000
window_seconds = 60

[rate_limit.ip]
requests = 200
window_seconds = 60

[rate_limit.user]
requests = 500
window_seconds = 60

[rate_limit.api_key]
requests = 1000
window_seconds = 60
```

---

## 📊 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| `0.1.0` | 2026-01-10 | ✨ 初始发布，支持 REST 和 gRPC API |

---

> **📝 最后更新**: 2026-01-14 | **问题反馈**: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
