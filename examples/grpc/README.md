# gRPC 示例

VecBoost 服务客户端调用示例。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|-------------|
| `grpc_client` | 通过 HTTP 端点调用 embedding_service | `http` |

## 运行命令

```bash
# 1. 先启动 VecBoost 服务 (HTTP 端口 9002)
cargo run --release --features http

# 2. 在另一个终端运行客户端示例
cargo run -p vecboost-examples --bin grpc_client --features http
```

## 说明

- 示例使用 `reqwest` 通过 HTTP 端点 (`http://localhost:9002/embed`) 调用 VecBoost 服务
- 请求体: `{"text": "hello", "normalize": true}`
- 返回体包含 `embedding`(向量)、`dimension`(维度)、`processing_time_ms`(耗时)
- VecBoost 同时支持 HTTP (端口 9002) 和 gRPC (端口 50051) 协议;此示例用 HTTP 简化依赖
