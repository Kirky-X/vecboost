# Basic 示例

基础用法示例，使用 `MockEngine`（实现 `InferenceEngine` trait）避免依赖真实模型文件，可独立运行。

## 示例列表

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `embed.rs` | 单文本嵌入 | `cargo run -p vecboost-examples --bin embed` |
| `batch.rs` | 批量嵌入 | `cargo run -p vecboost-examples --bin batch` |
| `similarity.rs` | 余弦相似度 | `cargo run -p vecboost-examples --bin similarity` |
| `rerank.rs` | 文档重排序 | `cargo run -p vecboost-examples --bin rerank` |
| `matryoshka.rs` | Matryoshka 维度约简 | `cargo run -p vecboost-examples --bin matryoshka` |
| `validation.rs` | 输入验证 | `cargo run -p vecboost-examples --bin validation` |

## MockEngine 说明

`embed`/`batch`/`similarity`/`rerank` 四个示例使用自定义 `MockEngine`（实现 `InferenceEngine` trait）返回固定维度向量，避免依赖真实模型文件。这使得示例可以独立运行，专注于演示 `EmbeddingService`、`RerankService` 与 `api` 层的用法。

`matryoshka` 和 `validation` 示例直接操作公共工具函数（`truncate_vector`、`normalize_l2`、`InputValidator` 等），无需推理引擎。
