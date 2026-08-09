# Library 示例

演示 `VecBoostLibrary` SDK — 将 VecBoost 作为嵌入式向量化/重排序模块直接集成到 Rust 应用。

## 示例

| 名称 | 说明 |
|------|------|
| `library_usage` | 异步+同步 API：`embed`、`embed_batch`、`rerank`、`embed_sync`、`embed_batch_sync`、`rerank_sync` |

## 运行

```bash
# 需要先下载模型
cargo run -p vecboost-examples --bin download_model -- --small

# 运行 library 示例
cargo run -p vecboost-examples --bin library_usage
```

## 说明

`VecBoostLibrary` 是 library 模式的唯一入口，无需启动 HTTP/gRPC 服务器即可使用完整的嵌入和重排序功能。

- **异步 API**：`embed()`、`embed_batch()`、`rerank()` — 适用于 tokio 运行时
- **同步 API**：`embed_sync()`、`embed_batch_sync()`、`rerank_sync()` — 适用于非异步上下文

注意：需要真实模型文件才能成功初始化。若模型不存在，示例会优雅降级并输出 API 用法参考。
