# Basic 示例

基础用法示例，使用 `MockEngine`（实现 `InferenceEngine` trait）避免依赖真实模型文件，可独立运行。

## 示例列表

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `embed.rs` | 单文本嵌入 | `cargo run -p vecboost-examples --bin embed` |
| `batch.rs` | 批量嵌入 | `cargo run -p vecboost-examples --bin batch` |
| `similarity.rs` | 余弦相似度 | `cargo run -p vecboost-examples --bin similarity` |
| `rerank.rs` | 文档重排序 | `cargo run -p vecboost-examples --bin rerank` |

## MockEngine 说明

四个示例均使用自定义 `MockEngine`（实现 `InferenceEngine` trait）返回固定维度向量，避免依赖真实模型文件。这使得示例可以独立运行，专注于演示 `EmbeddingService`、`RerankService` 与 `api` 层的用法。
