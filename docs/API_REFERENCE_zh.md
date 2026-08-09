<div align="center">

# 📚 VecBoost API 参考文档

**完整的 REST HTTP 端点和 gRPC 服务方法文档**

[![Version 0.2.1](https://img.shields.io/badge/Version-0.2.1-green.svg?style=for-the-badge)](https://github.com/Kirky-X/vecboost) [![REST API](https://img.shields.io/badge/REST-API-9002-blue.svg?style=for-the-badge)](http://localhost:9002) [![gRPC](https://img.shields.io/badge/gRPC-50051-green.svg?style=for-the-badge)](localhost:50051)

*VecBoost API 的完整文档，包括 REST HTTP 端点和 gRPC 服务方法。*

</div>

---

## 📋 目录

| 章节 | 说明 |
|------|------|
| [基础 URL](#基础-url) | API 端点基础地址 |
| [认证](#认证) | JWT 认证和令牌管理 |
| [REST API](#rest-api) | HTTP REST 接口文档 |
| [OpenAPI 文档](#-openapi-文档) | Swagger UI 与 OpenAPI 规范端点 |
| [gRPC API](#grpc-api) | gRPC 服务定义、配置与消息类型 |
| [错误处理](#错误处理) | 错误码和响应格式 |
| [速率限制](#速率限制) | 速率限制策略和响应头 |

---

## 🌍 基础 URL

| 环境 | 协议 | URL | 端口 |
|------|------|-----|------|
| **REST API** | HTTP | `http://localhost:9002` | `9002` |
| **gRPC API** | HTTP/2 | `localhost:50051` | `50051` |
| **Prometheus** | HTTP | `http://localhost:9002/metrics` | `9002` |

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
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 0
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `token` | string | JWT 访问令牌 |
| `token_type` | string | 令牌类型（始终为 `Bearer`） |
| `expires_in` | integer | 令牌过期时间（由 garrison 管理，固定为 0） |

> **ℹ️ 注意**: 令牌过期时间由 garrison 认证框架的 `GarrisonConfig.timeout` 统一管理，响应中 `expires_in` 固定为 0。

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
      "text_preview": "第一个文档",
      "embedding": [0.123, 0.456, ...]
    },
    {
      "text_preview": "第二个文档",
      "embedding": [0.789, 0.012, ...]
    }
  ],
  "dimension": 1024,
  "processing_time_ms": 25
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `embeddings` | array | 嵌入结果数组（每项含 `text_preview` 和 `embedding`） |
| `dimension` | integer | 向量维度 |
| `processing_time_ms` | number | 总处理时间（毫秒） |

> **⚠️ 批量大小校验**: 批量请求数量受 `EmbeddingConfig.max_batch_size` 限制（默认 64）。超限时返回 `400 INVALID_INPUT`。

---

#### 文件嵌入

为文件生成嵌入向量。

**端点:** `POST /api/v1/embed/file`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | ✅ | 文件路径（受 `grpc_allowed_roots` 路径校验约束） |
| `mode` | string | ❌ | 聚合模式 (`document` 或 `paragraph`，默认: `document`) |

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
    "total_chunks": 25,
    "successful_chunks": 25,
    "failed_chunks": 0,
    "processing_time_ms": 150
  },
  "embedding": null,
  "paragraphs": [
    {
      "embedding": [0.456, ...],
      "position": 0,
      "text_preview": "First paragraph..."
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `mode` | string | 使用的聚合模式 |
| `stats` | object | 文件处理统计（`total_chunks`/`successful_chunks`/`failed_chunks`/`processing_time_ms`） |
| `embedding` | array \| null | 聚合后的嵌入向量（`document` 模式时有值） |
| `paragraphs` | array \| null | 分段嵌入结果（`paragraph` 模式时有值） |

---

### 🌐 OpenAI 兼容 API

VecBoost 提供 OpenAI 兼容的 embeddings API 端点，支持 `dimensions` 参数进行 Matryoshka 维度约简。

#### 生成嵌入向量（OpenAI 兼容）

**端点:** `POST /v1/embeddings`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | string \| array | ✅ | 要嵌入的文本或文本数组（最多 2048 项） |
| `model` | string | ✅ | 模型 ID（支持 `text-embedding-ada-002` 映射） |
| `encoding_format` | string | ❌ | 编码格式 (`float`, `base64`) |
| `dimensions` | integer | ❌ | 输出维度（支持 1-1024，BGE-M3 最大 1024） |
| `user` | string | ❌ | 用户标识符 |

**请求示例（单文本）：**

```bash
curl -X POST http://localhost:9002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "text-embedding-ada-002",
    "dimensions": 256
  }'
```

**请求示例（批量）：**

```bash
curl -X POST http://localhost:9002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["文本1", "文本2", "文本3"],
    "model": "text-embedding-ada-002",
    "dimensions": 512
  }'
```

**响应：**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, 0.456, 0.789, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.111, 0.222, 0.333, ...],
      "index": 1
    }
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

#### 🧊 Matryoshka 维度约简

通过 `dimensions` 参数控制输出向量维度，实现更小更快的嵌入：

| 请求 `dimensions` | 实际返回维度 | 使用场景 |
|-------------------|-------------|----------|
| `256` | 256 | 最大速度，最小存储 |
| `512` | 512 | 平衡性能 |
| `1024` | 1024 | 最大质量（默认值） |

**维度限制：**
- 最小值: 1
- 最大值: 模型最大维度（BGE-M3 为 1024）
- 无 `dimensions` 参数时返回完整维度

**截断后自动重归一化**:

VecBoost 在执行 Matryoshka 截断（`truncate_vector`）后会立即调用 `normalize_l2` 对截断后的向量重新执行 L2 归一化。这是必要的——截断破坏了原向量的单位长度，若不重新归一化会导致：
- 余弦相似度计算偏离 `[-1, 1]` 区间
- 点积与余弦相似度不再等价
- 与原始 1024 维向量的相似度比较失真

> **💡 提示**: 该归一化对所有支持 Matryoshka 的入口生效（HTTP `/v1/embeddings`、`/api/v1/embed*`、gRPC `vecboost.embed*`、MCP、CLI）。若请求中显式指定 `normalize: false` 但同时传 `dimensions`，截断后仍会执行重归一化以保证语义正确。

**错误响应（维度超限）：**

```json
{
  "error": {
    "message": "dimensions 2048 exceeds model maximum 1024",
    "type": "invalid_request_error"
  }
}
```

---

### 相似度计算

#### 计算相似度

计算两段文本之间的余弦相似度。服务端自动对文本进行嵌入后计算相似度。

**端点:** `POST /api/v1/similarity`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source` | string | ✅ | 源文本 |
| `target` | string | ✅ | 目标文本 |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "source": "机器学习是人工智能的子领域",
    "target": "深度学习基于神经网络"
  }'
```

**响应:**

```json
{
  "score": 0.8765
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | number | 余弦相似度分数（范围 [-1, 1]） |

---

### 重排序（Rerank）

#### 文档重排序

根据与查询文本的相关性对文档列表进行重排序。

**端点:** `POST /api/v1/rerank`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 查询文本 |
| `documents` | array | ✅ | 待排序文档数组 |
| `top_k` | integer | ❌ | 返回前 K 个结果 |
| `return_documents` | boolean | ❌ | 是否在结果中返回文档原文 |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习",
    "documents": ["机器学习是人工智能的子领域", "今天天气很好", "深度学习基于神经网络"],
    "top_k": 2,
    "return_documents": true
  }'
```

**响应:**

```json
{
  "results": [
    {
      "index": 0,
      "score": 0.95,
      "document": "机器学习是人工智能的子领域"
    },
    {
      "index": 2,
      "score": 0.82,
      "document": "深度学习基于神经网络"
    }
  ],
  "processing_time_ms": 8
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `results` | array | 排序结果（按相关性降序） |
| `results[].index` | integer | 原文档在输入数组中的索引 |
| `results[].score` | number | 相关性分数 |
| `results[].document` | string | 文档原文（仅当 `return_documents=true` 时返回） |
| `processing_time_ms` | number | 处理时间（毫秒） |

#### 批量重排序

**端点:** `POST /api/v1/rerank/batch`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `queries` | array | ✅ | RerankRequest 数组（每项含 `query`、`documents`、`top_k`、`return_documents`） |

**响应:**

```json
{
  "responses": [
    {
      "results": [...],
      "processing_time_ms": 8
    }
  ]
}
```

---

### 模型管理

#### 获取当前模型

获取当前加载模型的信息。

**端点:** `GET /api/v1/model/current`

**响应:**

```json
{
  "name": "BAAI/bge-m3",
  "engine_type": "candle",
  "dimension": 1024,
  "is_loaded": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 模型名称（HuggingFace ID） |
| `engine_type` | string | 引擎类型 (`candle` 或 `onnx`) |
| `dimension` | integer | 嵌入向量维度（可选） |
| `is_loaded` | boolean | 模型是否已加载 |

#### 获取模型元数据

获取当前加载模型的详细元数据。

**端点:** `GET /api/v1/model/info`

**响应:**

```json
{
  "name": "BAAI/bge-m3",
  "version": "main",
  "engine_type": "candle",
  "dimension": 1024,
  "max_input_length": 8192,
  "is_loaded": true,
  "loaded_at": "2026-08-10T10:30:00Z"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 模型名称 |
| `version` | string | 模型版本 |
| `engine_type` | string | 引擎类型 |
| `dimension` | integer | 向量维度（可选） |
| `max_input_length` | integer | 最大输入长度 |
| `is_loaded` | boolean | 是否已加载 |
| `loaded_at` | string | 加载时间（可选） |

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
      "engine_type": "candle",
      "dimension": 1024,
      "is_loaded": true
    }
  ],
  "total_count": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `models` | array | 模型列表（每项为 `ModelInfo`） |
| `total_count` | integer | 可用模型总数 |

---

#### 切换模型

切换到不同的模型。

**端点:** `POST /api/v1/model/switch`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | ✅ | 模型名称 |
| `model_path` | string | ❌ | 本地模型路径 |
| `tokenizer_path` | string | ❌ | 分词器路径 |
| `device` | string | ❌ | 设备类型 (`cpu`, `cuda`, `metal`) |
| `max_batch_size` | integer | ❌ | 最大批处理大小 |
| `pooling_mode` | string | ❌ | 池化模式 |
| `expected_dimension` | integer | ❌ | 期望维度 |
| `memory_limit_bytes` | integer | ❌ | 内存限制（字节） |
| `oom_fallback_enabled` | boolean | ❌ | OOM 自动降级 |

**请求示例:**

```bash
curl -X POST http://localhost:9002/api/v1/model/switch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model_name": "BAAI/bge-small-en-v1.5",
    "device": "cpu",
    "expected_dimension": 384
  }'
```

**响应:**

```json
{
  "previous_model": "BAAI/bge-m3",
  "current_model": "BAAI/bge-small-en-v1.5",
  "success": true,
  "message": "Model switched successfully"
}
```

---

### 健康检查

#### 健康检查

检查服务健康状态。聚合各模块（EmbeddingModule、RateLimitModule、CacheModule 等）的 `health_status()` 状态。

**端点:** `GET /health`

**HTTP 状态码:**
- `200 OK` — 所有模块健康
- `503 Service Unavailable` — 任一模块异常

**响应示例 (200):**

```json
{
  "status": "OK"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 聚合健康状态（`OK` 表示所有模块健康） |

> **ℹ️ 注意**: 健康检查通过 trait-kit `health` feature 查询各注册模块（EmbeddingModule、RerankModule、RateLimitModule、CacheModule）状态。任一模块不健康时返回 `503 Service Unavailable`。配合 sdforge `graceful-shutdown` feature，服务在收到 SIGTERM/SIGINT 后会等待在途请求完成再关闭（超时 30s）。

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
| `rate_limit_allowed_total` | counter | 限流放行的请求数（按维度标签） |
| `rate_limit_denied_total` | counter | 限流拒绝的请求数（按维度标签） |

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

### 📚 OpenAPI 文档

VecBoost 内置 Swagger UI 与 OpenAPI 规范端点（由 `utoipa` + `utoipa-swagger-ui` 生成，需启用 `openapi` feature）：

| 端点 | 说明 |
|------|------|
| `GET /api-docs` | Swagger UI 交互式文档（可在浏览器中直接发起请求） |
| `GET /api-docs/openapi.json` | OpenAPI 3.0 规范 JSON |

**访问示例:**

```
http://localhost:9002/api-docs           # 浏览器打开 Swagger UI
http://localhost:9002/api-docs/openapi.json  # 拉取 OpenAPI 规范
```

> **ℹ️ 说明**: v0.2.0 的 Swagger UI 路径为 `/api-docs`（基于 `utoipa-swagger-ui` 默认配置）。ReDoc 端点推迟到 v0.3.0 提供。所有 REST API（含 OpenAI 兼容 `/v1/embeddings`）的请求/响应 schema 均在 OpenAPI 规范中描述，可直接用于生成客户端 SDK。

---

## 🔌 gRPC API

### 服务定义

v0.2.0 起，VecBoost 的 gRPC 接口不再依赖手写 `.proto` 文件，而是由 `sdforge` 框架通过 `#[forge(grpc_method = "...")]` 宏从 `src/api/embedding.rs` 中的单一源定义自动生成。客户端通过 sdforge 统一的 `SdForgeService/Call` RPC 调用对应方法：

- 请求载荷为 JSON 序列化的领域类型，通过 `CallRequest.data` 传递
- 响应载荷为 JSON 序列化的领域类型，通过 `CallResponse.data` 返回
- 服务名固定为 `SdForgeService`，方法名为下表中的 `vecboost.*` 标识符

```rust
// src/api/embedding.rs
#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed",
    version = "v1",
    grpc_method = "vecboost.embed",
    description = "Generate embedding vector for input text"
)]
pub async fn grpc_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}
```

> **💡 提示**: 无需 `proto/` 目录或 `tonic-build` 生成客户端存根。所有 gRPC 方法都在 `src/api/embedding.rs` 中通过 `#[forge(grpc_method = "...")]` 宏注册。

### gRPC 服务配置

gRPC 服务通过 `sdforge::grpc::build_server_with_config` 启动，配置项定义在 `[server]` 段：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `grpc_max_connections` | `usize` | `1000` | 最大并发连接数 |
| `grpc_timeout_seconds` | `u64` | `30` | 请求超时（秒） |
| `grpc_require_auth` | `bool` | `true` | 是否强制 JWT 认证（BearerAuth） |
| `grpc_allowed_roots` | `Vec<PathBuf>` | 工作目录 | gRPC 文件嵌入路径校验根目录（拒绝 `/`、`/etc`、`/root` 等敏感目录作为 fallback） |

启用 `grpc_require_auth = true` 时，必须同时满足：
- `[auth] enabled = true`
- 编译启用 `auth` feature
- `VECBOOST_JWT_SECRET` 环境变量已设置（≥32 字符）或 `auth.jwt_secret` 配置项非空

否则服务启动失败。此外，gRPC 默认启用 `LimiteronAdapter` 速率限制（默认 100 burst / 10 req/s）。

---

### 服务方法概览

| gRPC 方法 | 处理函数 | 输入类型 | 输出类型 | 说明 |
|-----------|----------|----------|----------|------|
| `vecboost.embed` | `grpc_embed` | `EmbedRequest` | `EmbedResponse` | 生成单个嵌入向量 |
| `vecboost.embed_batch` | `grpc_embed_batch` | `BatchEmbedRequest` | `BatchEmbedResponse` | 批量生成嵌入向量 |
| `vecboost.compute_similarity` | `grpc_compute_similarity` | `SimilarityRequest` | `SimilarityResponse` | 计算向量相似度 |
| `vecboost.embed_file` | `grpc_embed_file` | `FileEmbedRequest` | `FileEmbedResponse` | 文件嵌入（路径校验） |
| `vecboost.rerank` | `grpc_rerank` | `RerankRequest` | `RerankResponse` | 按相关性重排序文档 |
| `vecboost.rerank_batch` | `grpc_rerank_batch` | `BatchRerankRequest` | `BatchRerankResponse` | 批量重排序 |
| `vecboost.model_switch` | `grpc_model_switch` | `ModelSwitchRequest` | `ModelSwitchResponse` | 切换模型 |
| `vecboost.get_current_model` | `grpc_get_current_model` | （空） | `ModelInfo` | 获取当前模型信息 |
| `vecboost.get_model_info` | `grpc_get_model_info` | （空） | `ModelMetadata` | 获取模型元数据 |
| `vecboost.list_models` | `grpc_list_models` | （空） | `ModelListResponse` | 列出可用模型 |
| `vecboost.health_check` | `grpc_health_check` | （空） | `serde_json::Value` | 健康检查 |

---

### 消息类型定义

> **ℹ️ 说明**: 以下结构为 `src/domain/` 中定义的 Rust 领域类型（serde 序列化）。gRPC 请求/响应以 JSON 字节流形式通过 sdforge `CallRequest.data` / `CallResponse.data` 传递；HTTP 接口同样以 JSON body 提交。字段名采用 `snake_case`。

#### 嵌入请求/响应

```rust
// 请求
struct EmbedRequest {
    text: String,
    normalize: Option<bool>,
}

// 响应
struct EmbedResponse {
    embedding: Vec<f32>,
    dimension: usize,
    processing_time_ms: u128,
    information_retention_rate: Option<f32>,  // Matryoshka 截断时填充
}

// 批量请求
struct BatchEmbedRequest {
    texts: Vec<String>,
    mode: Option<AggregationMode>,
    normalize: Option<bool>,
}

// 批量响应
struct BatchEmbedResponse {
    embeddings: Vec<BatchEmbeddingResult>,  // 每项含 text_preview + embedding
    dimension: usize,
    processing_time_ms: u128,
}
```

> **⚠️ 批量大小校验**: 批量请求数量受 `validate_batch_size` 限制，上限取自 `EmbeddingConfig.max_batch_size`（默认 64）。超限时返回 `400 INVALID_INPUT`，错误信息形如 `batch size N exceeds max M (config embedding.max_batch_size)`。HTTP `/api/v1/embed/batch` 与 OpenAI 兼容 `/v1/embeddings` 批量端点均执行此校验。

#### 相似度请求/响应

```rust
struct SimilarityRequest {
    source: String,  // 源文本（服务端自动嵌入后计算）
    target: String,  // 目标文本
}

struct SimilarityResponse {
    score: f32,
}
```

#### 文件嵌入

```rust
struct FileEmbedRequest {
    path: String,
    mode: Option<AggregationMode>,  // "document" | "paragraph"
}

struct FileEmbedResponse {
    mode: AggregationMode,
    stats: FileProcessingStats,
    embedding: Option<Vec<f32>>,     // document 模式时有值
    paragraphs: Option<Vec<ParagraphEmbedding>>,  // paragraph 模式时有值
}

struct FileProcessingStats {
    total_chunks: usize,
    successful_chunks: usize,
    failed_chunks: usize,
    processing_time_ms: u128,
}

struct ParagraphEmbedding {
    embedding: Vec<f32>,
    position: usize,
    text_preview: String,
}
```

> **🔒 路径校验**: gRPC `vecboost.embed_file` 受 `grpc_allowed_roots` 配置约束；未配置时回退到当前工作目录，并拒绝 `/`、`/etc`、`/root`、`/var`、`/usr` 等敏感根目录作为 fallback，防止意外暴露整个文件系统。

#### 模型管理

```rust
struct ModelSwitchRequest {
    model_name: String,
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    device: Option<DeviceType>,       // "cpu" | "cuda" | "metal"
    max_batch_size: Option<usize>,
    pooling_mode: Option<PoolingMode>,
    expected_dimension: Option<usize>,
    memory_limit_bytes: Option<u64>,
    oom_fallback_enabled: Option<bool>,
}

struct ModelSwitchResponse {
    previous_model: Option<String>,
    current_model: String,
    success: bool,
    message: String,
}

struct ModelInfo {
    name: String,
    engine_type: String,
    dimension: Option<usize>,
    is_loaded: bool,
}

struct ModelMetadata {
    name: String,
    version: String,
    engine_type: String,
    dimension: Option<usize>,
    max_input_length: usize,
    is_loaded: bool,
    loaded_at: Option<String>,
}

struct ModelListResponse {
    models: Vec<ModelInfo>,
    total_count: usize,
}
```

#### 重排序

```rust
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    top_k: Option<usize>,
    return_documents: Option<bool>,
}

struct RerankResult {
    index: usize,
    score: f32,
    document: Option<String>,
}

struct RerankResponse {
    results: Vec<RerankResult>,
    processing_time_ms: u128,
}

struct BatchRerankRequest {
    queries: Vec<RerankRequest>,
}

struct BatchRerankResponse {
    responses: Vec<RerankResponse>,
}
```

#### 健康检查

```rust
// 响应（serde_json::Value）
{
    "status": "OK"
}
```

> **ℹ️ 注意**: v0.2.0 中 `vecboost.health_check`、`vecboost.get_current_model`、`vecboost.get_model_info`、`vecboost.list_models` 不需要请求载荷（即原 `Empty` 已移除），客户端调用 `SdForgeService/Call` 时传空 JSON 即可。健康检查返回 `{"status": "OK"}` 或 `503 Service Unavailable`。

---

### SDK 使用示例

> **ℹ️ 说明**: v0.2.0 起，VecBoost 不再发布自己的 `.proto` 与生成客户端。客户端使用 sdforge 框架的统一 proto（`sdforge/proto/sdforge.v1.proto`，包名 `sdforge.v1`）生成的 `SdForgeService` stub，通过 `Call` RPC 传入方法名（如 `vecboost.embed`）与 JSON 载荷调用具体方法。下方示例展示这种调用模式。

#### grpcurl（命令行调试）

```bash
# 1. 查询服务可用方法（含 vecboost.*）
grpcurl -plaintext localhost:50051 sdforge.v1.SdForgeService/GetInfo

# 2. 调用 vecboost.embed（method 字段传方法名，data 字段传 JSON）
grpcurl -plaintext -d '{
  "method": "vecboost.embed",
  "data": "{\"text\":\"Hello, world!\",\"normalize\":true}"
}' localhost:50051 sdforge.v1.SdForgeService/Call
```

#### Python

```python
import json
import grpc
import sdforge_v1_pb2
import sdforge_v1_pb2_grpc

# 连接 gRPC 服务（SdForgeService 由 sdforge.v1 proto 生成）
channel = grpc.insecure_channel('localhost:50051')
stub = sdforge_v1_pb2_grpc.SdForgeServiceStub(channel)

# 单个嵌入请求：method=vecboost.embed，data=JSON 序列化的 EmbedRequest
call_req = sdforge_v1_pb2.CallRequest(
    method="vecboost.embed",
    data=json.dumps({"text": "Hello, world!", "normalize": True}),
)
call_resp = stub.Call(call_req)
if not call_resp.success:
    raise RuntimeError(f"vecboost.embed failed: {call_resp.error}")

embed_resp = json.loads(call_resp.data)
print(f"Embedding dimension: {embed_resp['dimension']}")
print(f"Processing time: {embed_resp['processing_time_ms']:.2f}ms")

# 批量嵌入请求：method=vecboost.embed_batch
batch_req = sdforge_v1_pb2.CallRequest(
    method="vecboost.embed_batch",
    data=json.dumps({"texts": ["文档1", "文档2", "文档3"], "normalize": True}),
)
batch_resp = stub.Call(batch_req)
batch_data = json.loads(batch_resp.data)
print(f"Processed {batch_data['total_count']} embeddings")
```

#### Go

```go
import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    pb "sdforge/v1" // 由 sdforge/proto/sdforge.v1.proto 生成
)

func main() {
    // 连接 sdforge gRPC 服务（SdForgeService/Call）
    conn, err := grpc.Dial("localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    client := pb.NewSdForgeServiceClient(conn)

    // 构造 EmbedRequest 并 JSON 序列化作为 CallRequest.data
    payload, _ := json.Marshal(map[string]any{
        "text":      "Hello, world!",
        "normalize": true,
    })

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := client.Call(ctx, &pb.CallRequest{
        Method: "vecboost.embed",
        Data:   string(payload),
    })
    if err != nil {
        log.Fatalf("Call failed: %v", err)
    }
    if !resp.Success {
        log.Fatalf("vecboost.embed failed: %s", resp.Error)
    }

    var embed struct {
        Dimension          int64   `json:"dimension"`
        ProcessingTimeMs   float64 `json:"processing_time_ms"`
    }
    if err := json.Unmarshal([]byte(resp.Data), &embed); err != nil {
        log.Fatalf("decode response: %v", err)
    }
    fmt.Printf("Dimension: %d, Time: %.2fms\n",
        embed.Dimension, embed.ProcessingTimeMs)
}
```

> **📝 生成客户端存根**: Python 用 `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sdforge/proto/sdforge.v1.proto`；Go 用 `protoc --go_out=. --go-grpc_out=. sdforge/proto/sdforge.v1.proto`。生成后即可调用任何 `vecboost.*` 方法，无需 VecBoost 自身发布 proto。

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

限流基于 limiteron 原生 `TokenBucketLimiter`，支持 Global/Ip/User/ApiKey 四维度独立限流：

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
global_requests_per_minute = 2000
ip_requests_per_minute = 200
user_requests_per_minute = 500
api_key_requests_per_minute = 1000
window_secs = 60
ip_whitelist = ["127.0.0.1"]
```

---

## 📊 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| `0.2.1` | 2026-08-09 | 🐛 文档同步更新、代码质量改进 |
| `0.2.0` | 2026-02-01 | ✨ 生态重构：7 库架构、trait-kit 模块注册、多协议接口；gRPC 由 sdforge `#[forge(grpc_method)]` 宏生成，移除 `proto/`、`src/grpc/`、`src/routes/`、`src/cli/` |
| `0.1.2` | 2026-01-16 | ✨ 添加 Matryoshka 维度约简支持、OpenAI 兼容 API |
| `0.1.0` | 2026-01-10 | ✨ 初始发布，支持 REST 和 gRPC API |

---

> **📝 最后更新**: 2026-08-09 | **问题反馈**: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
