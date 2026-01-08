<div align="center">

# 📘 VecBoost API 参考文档

本文档提供 VecBoost 所有 API 的详细说明，包括 HTTP REST API 和 gRPC API。

</div>

---

## 📋 目录

- [🔗 API 概览](#api-概览)
- [🌐 HTTP REST API](#http-rest-api)
- [🔧 gRPC API](#grpc-api)
- [📊 公共数据类型](#公共数据类型)
- [⚠️ 错误处理](#错误处理)
- [📝 请求示例](#请求示例)

---

## 🔗 API 概览

### 服务端点

| 协议 | 地址 | 描述 |
|------|------|------|
| HTTP REST | `http://localhost:9002` | REST API 服务 |
| gRPC | `grpc://localhost:50051` | gRPC API 服务 |
| Prometheus | `http://localhost:9090` | 指标监控端口 |

### API 列表

| 端点 | 方法 | 描述 |
|------|------|------|
| `/embed` | POST | 生成文本嵌入向量 |
| `/embed/batch` | POST | 批量生成嵌入向量 |
| `/similarity` | POST | 计算两文本相似度 |
| `/search` | POST | 语义搜索 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |

---

## 🌐 HTTP REST API

### 1. 生成文本嵌入

生成单个文本的向量嵌入。

**端点**: `POST /embed`

**请求体**:

```json
{
  "text": "要向量化的文本内容",
  "normalize": true
}
```

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `text` | string | 是 | 输入文本，最大长度 8192 tokens |
| `normalize` | boolean | 否 | 是否归一化向量，默认为 true |

**响应体**:

```json
{
  "embedding": [0.123, 0.456, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

| 字段 | 类型 | 描述 |
|------|------|------|
| `embedding` | number[] | 生成的向量数组 |
| `dimension` | integer | 向量维度 |
| `processing_time_ms` | number | 处理时间（毫秒） |

**示例请求**:

```bash
curl -X POST http://localhost:9002/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"text": "人工智能是未来的发展方向", "normalize": true}'
```

### 2. 批量生成嵌入

批量生成多个文本的向量嵌入。

**端点**: `POST /embed/batch`

**请求体**:

```json
{
  "texts": ["文本1", "文本2", "文本3"],
  "normalize": true
}
```

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `texts` | string[] | 是 | 文本数组，最大 64 条 |
| `normalize` | boolean | 否 | 是否归一化向量 |

**响应体**:

```json
{
  "embeddings": [
    [0.123, 0.456, ...],
    [0.789, 0.012, ...]
  ],
  "total_count": 2,
  "processing_time_ms": 25.0
}
```

**示例请求**:

```bash
curl -X POST http://localhost:9002/embed/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"texts": ["机器学习", "深度学习", "神经网络"]}'
```

### 3. 计算相似度

计算两个文本之间的相似度。

**端点**: `POST /similarity`

**请求体**:

```json
{
  "source": "文本A",
  "target": "文本B",
  "metric": "cosine"
}
```

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `source` | string | 是 | 源文本 |
| `target` | string | 是 | 目标文本 |
| `metric` | string | 否 | 相似度算法: `cosine`, `euclidean`, `dot_product`, `manhattan` |

**响应体**:

```json
{
  "score": 0.8567,
  "metric": "cosine"
}
```

**示例请求**:

```bash
curl -X POST http://localhost:9002/similarity \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"source": "人工智能", "target": "机器学习", "metric": "cosine"}'
```

### 4. 语义搜索

在文本集合中搜索与查询最相似的文本。

**端点**: `POST /search`

**请求体**:

```json
{
  "query": "搜索查询文本",
  "texts": ["文本1", "文本2", "文本3", "文本4"],
  "top_k": 5,
  "metric": "cosine"
}
```

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `query` | string | 是 | 查询文本 |
| `texts` | string[] | 是 | 待搜索的文本列表 |
| `top_k` | integer | 否 | 返回结果数量，默认为 10 |
| `metric` | string | 否 | 相似度算法 |

**响应体**:

```json
{
  "results": [
    {
      "text": "匹配的文本",
      "score": 0.9231,
      "index": 1
    }
  ],
  "query_embedding": [0.123, ...]
}
```

**示例请求**:

```bash
curl -X POST http://localhost:9002/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "关于编程语言的选择",
    "texts": ["Python是一门易学的语言", "Java是企业级首选", "Rust注重安全"],
    "top_k": 2
  }'
```

### 5. 健康检查

检查服务健康状态。

**端点**: `GET /health`

**响应体**:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model": "BAAI/bge-m3",
  "device": "cpu",
  "uptime_seconds": 3600
}
```

**示例请求**:

```bash
curl http://localhost:9002/health
```

### 6. Prometheus 指标

获取 Prometheus 格式的监控指标。

**端点**: `GET /metrics`

**示例请求**:

```bash
curl http://localhost:9002/metrics
```

**常用指标**:

```
# 帮助信息
vecboost_requests_total{endpoint="embed"} 1234
vecboost_request_duration_seconds_bucket{endpoint="embed",le="0.005"} 1000
vecboost_embedding_duration_seconds 0.015
vecboost_cache_hits_total 567
vecboost_cache_misses_total 123
```

---

## 🔧 gRPC API

### 服务定义

```protobuf
service EmbeddingService {
  // 单文本嵌入
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  
  // 批量嵌入
  rpc EmbedBatch(BatchEmbedRequest) returns (BatchEmbedResponse);
  
  // 计算相似度
  rpc ComputeSimilarity(SimilarityRequest) returns (SimilarityResponse);
  
  // 语义搜索
  rpc Search(SearchRequest) returns (SearchResponse);
  
  // 健康检查
  rpc HealthCheck(Empty) returns (HealthResponse);
}
```

### 消息类型

#### EmbedRequest

```protobuf
message EmbedRequest {
  string text = 1;
  bool normalize = 2;
}
```

#### EmbedResponse

```protobuf
message EmbedResponse {
  repeated float embedding = 1;
  int64 dimension = 2;
  double processing_time_ms = 3;
}
```

#### BatchEmbedRequest

```protobuf
message BatchEmbedRequest {
  repeated string texts = 1;
  bool normalize = 2;
}
```

#### BatchEmbedResponse

```protobuf
message BatchEmbedResponse {
  repeated EmbedResponse embeddings = 1;
  int64 total_count = 2;
  double processing_time_ms = 3;
}
```

#### SimilarityRequest

```protobuf
message SimilarityRequest {
  string source = 1;
  string target = 2;
  string metric = 3;  // cosine, euclidean, dot_product, manhattan
}
```

#### SimilarityResponse

```protobuf
message SimilarityResponse {
  double score = 1;
  string metric = 2;
}
```

#### SearchRequest

```protobuf
message SearchRequest {
  string query = 1;
  repeated string texts = 2;
  int32 top_k = 3;
  string metric = 4;
}
```

#### SearchResponse

```protobuf
message SearchResponse {
  repeated SearchResult results = 1;
  int64 query_dimension = 2;
}

message SearchResult {
  string text = 1;
  double score = 2;
  int32 index = 3;
}
```

#### HealthResponse

```protobuf
message HealthResponse {
  string status = 1;  // healthy, degraded, unhealthy
  string version = 2;
  string model = 3;
  string device = 4;
  int64 uptime_seconds = 5;
}
```

### gRPC 客户端示例

```rust
use vecboost::grpc::embedding_service_client::EmbeddingServiceClient;
use vecboost::grpc::{EmbedRequest, BatchEmbedRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbeddingServiceClient::connect("http://localhost:50051").await?;
    
    // 单文本嵌入
    let request = tonic::Request::new(EmbedRequest {
        text: "人工智能是未来的发展方向".to_string(),
        normalize: true,
    });
    
    let response = client.embed(request).await?;
    println!("Embedding: {:?}", response.into_inner().embedding);
    
    // 批量嵌入
    let batch_request = tonic::Request::new(BatchEmbedRequest {
        texts: vec!["机器学习".to_string(), "深度学习".to_string()],
        normalize: true,
    });
    
    let batch_response = client.embed_batch(batch_request).await?;
    println!("Batch embeddings: {:?}", batch_response.into_inner().embeddings);
    
    Ok(())
}
```

---

## 📊 公共数据类型

### 相似度算法

| 算法 | 描述 | 值范围 |
|------|------|--------|
| `cosine` | 余弦相似度 | [-1, 1] |
| `euclidean` | 欧氏距离 | [0, ∞) |
| `dot_product` | 点积 | (-∞, ∞) |
| `manhattan` | 曼哈顿距离 | [0, ∞) |

### 优先级

| 值 | 描述 |
|------|------|
| `low` | 低优先级 |
| `normal` | 普通优先级 |
| `high` | 高优先级 |
| `critical` | 最高优先级 |

### 设备类型

| 值 | 描述 |
|------|------|
| `cpu` | CPU 计算 |
| `cuda` | NVIDIA GPU |
| `metal` | Apple Silicon GPU |

### 模型精度

| 值 | 描述 |
|------|------|
| `fp32` | 32位浮点 |
| `fp16` | 16位浮点 |
| `int8` | 8位整数 |

---

## ⚠️ 错误处理

### 错误响应格式

```json
{
  "error": {
    "code": "INVALID_TEXT",
    "message": "文本内容不能为空",
    "details": {...}
  }
}
```

### 错误码

| 错误码 | HTTP 状态码 | 描述 |
|--------|-------------|------|
| `SUCCESS` | 200 | 成功 |
| `INVALID_TEXT` | 400 | 无效的文本输入 |
| `TEXT_TOO_LONG` | 400 | 文本超出长度限制 |
| `BATCH_TOO_LARGE` | 400 | 批量请求超出限制 |
| `INVALID_METRIC` | 400 | 无效的相似度算法 |
| `UNAUTHORIZED` | 401 | 未授权 |
| `FORBIDDEN` | 403 | 禁止访问 |
| `RATE_LIMITED` | 429 | 请求过于频繁 |
| `MODEL_NOT_LOADED` | 503 | 模型未加载 |
| `INFERENCE_ERROR` | 500 | 推理错误 |
| `GPU_OUT_OF_MEMORY` | 507 | GPU 内存不足 |
| `INTERNAL_ERROR` | 500 | 内部错误 |

### gRPC 状态码

| 状态码 | 描述 |
|--------|------|
| `OK` | 成功 |
| `INVALID_ARGUMENT` | 无效参数 |
| `UNAUTHENTICATED` | 未认证 |
| `PERMISSION_DENIED` | 权限不足 |
| `RESOURCE_EXHAUSTED` | 资源耗尽（限流）|
| `UNAVAILABLE` | 服务不可用 |
| `INTERNAL` | 内部错误 |

---

## 📝 请求示例

### Python 请求示例

```python
import requests

API_BASE = "http://localhost:9002"
HEADERS = {"Authorization": "Bearer your-token-here"}

def embed(text, normalize=True):
    response = requests.post(
        f"{API_BASE}/embed",
        json={"text": text, "normalize": normalize},
        headers=HEADERS
    )
    return response.json()

def batch_embed(texts, normalize=True):
    response = requests.post(
        f"{API_BASE}/embed/batch",
        json={"texts": texts, "normalize": normalize},
        headers=HEADERS
    )
    return response.json()

def similarity(source, target, metric="cosine"):
    response = requests.post(
        f"{API_BASE}/similarity",
        json={"source": source, "target": target, "metric": metric},
        headers=HEADERS
    )
    return response.json()

def search(query, texts, top_k=5):
    response = requests.post(
        f"{API_BASE}/search",
        json={"query": query, "texts": texts, "top_k": top_k},
        headers=HEADERS
    )
    return response.json()
```

### JavaScript/Node.js 请求示例

```javascript
const API_BASE = 'http://localhost:9002';
const HEADERS = { 'Authorization': 'Bearer your-token-here' };

async function embed(text, normalize = true) {
    const response = await fetch(`${API_BASE}/embed`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, normalize })
    });
    return response.json();
}

async function batchEmbed(texts, normalize = true) {
    const response = await fetch(`${API_BASE}/embed/batch`, {
        method: 'POST',
        headers: { ...HEADERS, 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts, normalize })
    });
    return response.json();
}
```

---

## 🔐 认证

### Bearer Token 认证

所有 API 端点（除 `/health` 和 `/metrics` 外）都需要认证：

```bash
curl -H "Authorization: Bearer <your-jwt-token>" http://localhost:9002/embed
```

### 获取 Token

```bash
# 登录获取 Token
curl -X POST http://localhost:9002/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# 响应
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## 📖 相关文档

- [🏗️ 架构设计](ARCHITECTURE.md)
- [📝 用户指南](USER_GUIDE.md)
- [🤝 贡献指南](CONTRIBUTING.md)

---

<div align="center">

**文档版本**: 1.0.0  
**最后更新**: 2026-01-08

</div>
