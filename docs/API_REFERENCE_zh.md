# API 参考文档

本文档提供 VecBoost API 的完整文档，包括 REST HTTP 端点和 gRPC 服务方法。

## 目录

- [基础 URL](#基础-url)
- [认证](#认证)
- [REST API](#rest-api)
  - [嵌入向量](#嵌入向量)
  - [相似度计算](#相似度计算)
  - [模型管理](#模型管理)
  - [健康检查](#健康检查)
- [gRPC API](#grpc-api)
  - [服务方法](#服务方法)
  - [消息类型](#消息类型)
- [错误处理](#错误处理)
- [速率限制](#速率限制)

---

## 基础 URL

| 环境 | URL |
|------|-----|
| 生产环境 | `http://localhost:9002` |
| gRPC | `localhost:50051` |

---

## 认证

启用认证时，请在 Authorization 头中包含 Bearer 令牌：

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -d '{"text": "Hello, world!"}'
```

### 获取令牌

```bash
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "Secure@Passw0rd!"
}
```

响应：

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## REST API

### 嵌入向量

#### 生成嵌入向量

为单个文本生成向量嵌入。

**端点:** `POST /api/v1/embed`

**请求体:**

```json
{
  "text": "string",
  "normalize": "boolean (可选)"
}
```

**响应:**

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

**示例:**

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog"}'
```

#### 批量嵌入

在单个请求中为多个文本生成嵌入向量。

**端点:** `POST /api/v1/embed/batch`

**请求体:**

```json
{
  "texts": ["string", "string", ...],
  "normalize": "boolean (可选)"
}
```

**响应:**

```json
{
  "embeddings": [
    {"embedding": [...], "dimension": 1024, "processing_time_ms": 12.3},
    {"embedding": [...], "dimension": 1024, "processing_time_ms": 11.8}
  ],
  "total_count": 2,
  "processing_time_ms": 25.5
}
```

#### 文件嵌入

为文件生成嵌入向量。

**端点:** `POST /api/v1/embed/file`

**请求体:**

```json
{
  "path": "/path/to/file.txt",
  "mode": "paragraph | chunk",
  "chunk_size": 512,
  "overlap": 50
}
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

**请求体:**

```json
{
  "vector1": [0.1, 0.2, 0.3, ...],
  "vector2": [0.1, 0.2, 0.3, ...],
  "metric": "cosine | euclidean | dot_product | manhattan"
}
```

**响应:**

```json
{
  "score": 0.9876,
  "metric": "cosine"
}
```

#### 相似文档搜索

从集合中找到最相似的向量。

**端点:** `POST /api/v1/search`

**请求体:**

```json
{
  "query": "search text",
  "documents": ["doc1", "doc2", "doc3"],
  "top_k": 5,
  "metric": "cosine"
}
```

**响应:**

```json
{
  "results": [
    {
      "index": 0,
      "text": "doc1",
      "score": 0.95
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
    }
  ],
  "current_model": "BAAI/bge-m3"
}
```

#### 切换模型

切换到不同的模型。

**端点:** `POST /api/v1/model/switch`

**请求体:**

```json
{
  "model_name": "BAAI/bge-small-en-v1.5",
  "engine_type": "candle",
  "device_type": "auto"
}
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

#### 就绪检查

检查服务是否准备好接收请求。

**端点:** `GET /ready`

**响应:**

```json
{
  "ready": true
}
```

#### 指标

Prometheus 指标端点。

**端点:** `GET /metrics`

返回 Prometheus 格式的指标，包括：
- `vecboost_requests_total` - 总请求数
- `vecboost_embedding_latency_seconds` - 嵌入延迟
- `vecboost_cache_hit_ratio` - 缓存命中率
- `vecboost_batch_size` - 批处理大小

---

## gRPC API

### 服务定义

```protobuf
service EmbeddingService {
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EmbedBatch(BatchEmbedRequest) returns (BatchEmbedResponse);
  rpc ComputeSimilarity(SimilarityRequest) returns (SimilarityResponse);
  rpc EmbedFile(FileEmbedRequest) returns (FileEmbedResponse);
  rpc ModelSwitch(ModelSwitchRequest) returns (ModelSwitchResponse);
  rpc GetCurrentModel(Empty) returns (ModelInfo);
  rpc GetModelInfo(Empty) returns (ModelMetadata);
  rpc ListModels(Empty) returns (ModelListResponse);
  rpc HealthCheck(Empty) returns (HealthResponse);
}
```

---

### 服务方法

#### Embed

为单个文本生成嵌入向量。

```protobuf
message EmbedRequest {
  string text = 1;
  optional bool normalize = 2;
}

message EmbedResponse {
  repeated float embedding = 1;
  int64 dimension = 2;
  double processing_time_ms = 3;
}
```

**示例 (Go):**

```go
client := embedding.NewEmbeddingServiceClient(conn)
resp, err := client.Embed(ctx, &embedding.EmbedRequest{
    Text: "Hello, world!",
    Normalize: proto.Bool(true),
})
```

#### EmbedBatch

为多个文本生成嵌入向量。

```protobuf
message BatchEmbedRequest {
  repeated string texts = 1;
  optional bool normalize = 2;
}

message BatchEmbedResponse {
  repeated EmbedResponse embeddings = 1;
  int64 total_count = 2;
  double processing_time_ms = 3;
}
```

#### ComputeSimilarity

计算两个向量之间的相似度。

```protobuf
message SimilarityRequest {
  repeated float vector1 = 1;
  repeated float vector2 = 2;
  string metric = 3;
}

message SimilarityResponse {
  double score = 1;
  string metric = 2;
}
```

#### EmbedFile

为文件生成嵌入向量。

```protobuf
message FileEmbedRequest {
  string path = 1;
  optional string mode = 2;
  optional int32 chunk_size = 3;
  optional int32 overlap = 4;
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

#### ModelSwitch

切换活动模型。

```protobuf
message ModelSwitchRequest {
  string model_name = 1;
  optional string engine_type = 2;
  optional string device_type = 3;
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
```

#### GetCurrentModel

获取当前加载模型的信息。

```protobuf
rpc GetCurrentModel(Empty) returns (ModelInfo);
```

#### GetModelInfo

获取当前模型的详细元数据。

```protobuf
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

rpc GetModelInfo(Empty) returns (ModelMetadata);
```

#### ListModels

列出所有可用模型。

```protobuf
message ModelListResponse {
  repeated ModelMetadata models = 1;
  string current_model = 2;
}

rpc ListModels(Empty) returns (ModelListResponse);
```

#### HealthCheck

检查服务健康状态。

```protobuf
message HealthResponse {
  string status = 1;
  string version = 2;
  string uptime = 3;
  optional string model_loaded = 4;
}

rpc HealthCheck(Empty) returns (HealthResponse);
```

---

## 错误处理

### HTTP 状态码

| 码 | 含义 |
|----|------|
| 200 | 成功 |
| 400 | 请求错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 429 | 请求过多 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

### 错误响应格式

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Text input cannot be empty",
    "details": null
  }
}
```

### 常见错误码

| 码 | 消息 |
|----|------|
| `INVALID_INPUT` | 无效的请求体 |
| `UNAUTHORIZED` | 缺少或无效的认证令牌 |
| `FORBIDDEN` | 权限不足 |
| `RATE_LIMITED` | 超出请求速率 |
| `MODEL_NOT_FOUND` | 未找到模型 |
| `INFERENCE_ERROR` | 模型推理失败 |
| `GPU_OOM` | GPU 内存不足 |

---

## 速率限制

### 默认限制

| 范围 | 请求数 | 时间窗口 |
|------|--------|----------|
| 全局 | 1000 | 每分钟 |
| 每 IP | 100 | 每分钟 |
| 每用户 | 200 | 每分钟 |
| 每 API 密钥 | 500 | 每分钟 |

### 速率限制头

响应包含速率限制信息：

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

---

## SDK

### Python

```python
import grpc
import embedding_pb2
import embedding_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)

request = embedding_pb2.EmbedRequest(
    text="Hello, world!",
    normalize=True
)
response = stub.Embed(request)
print(response.embedding)
```

### Go

```go
import (
    "context"
    "google.golang.org/grpc"
    pb "vecboost/proto"
)

conn, _ := grpc.Dial("localhost:50051")
client := pb.NewEmbeddingServiceClient(conn)

resp, err := client.Embed(context.Background(), &pb.EmbedRequest{
    Text: "Hello, world!",
})
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.0 | 2026-01-10 | 初始发布 |
