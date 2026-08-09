# HTTP API 示例

本目录包含 VecBoost HTTP API 的调用示例,使用 `reqwest` 客户端访问运行中的 VecBoost 服务。

## 前置条件

先启动 VecBoost HTTP 服务(默认端口 9002):

```bash
cargo run --release --features http
```

## 示例列表

| 示例 | 端点 | 说明 |
|------|------|------|
| `embed_api.rs` | `POST /api/v1/embed` | 单文本嵌入 |
| `batch_api.rs` | `POST /api/v1/embed/batch` | 批量文本嵌入 |
| `similarity_api.rs` | `POST /api/v1/similarity` | 文本相似度计算 |
| `rerank_api.rs` | `POST /api/v1/rerank` | 文档重排序 |

## 运行命令

```bash
cargo run -p vecboost-examples --bin embed_api
cargo run -p vecboost-examples --bin batch_api
cargo run -p vecboost-examples --bin similarity_api
cargo run -p vecboost-examples --bin rerank_api
```

## 请求/响应格式

- `POST /api/v1/embed`:`{"text": "...", "normalize": true}` → `{"embedding": [...], "dimension": N, "processing_time_ms": M}`
- `POST /api/v1/embed/batch`:`{"texts": ["...", "..."], "normalize": true}` → `{"embeddings": [{"text_preview": "...", "embedding": [...]}], "dimension": N, "processing_time_ms": M}`
- `POST /api/v1/similarity`:`{"source": "...", "target": "..."}` → `{"score": 0.87}`
- `POST /api/v1/rerank`:`{"query": "...", "documents": ["..."], "top_k": 3, "return_documents": true}` → `{"results": [{"index": N, "score": 0.95, "document": "..."}], "processing_time_ms": M}`
