# Monitoring 示例

VecBoost 监控模块示例,演示 Prometheus 指标收集与推理性能监控。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|-------------|
| `metrics` | InferenceCollector 收集推理指标 + PrometheusCollector 导出 Prometheus 格式文本 | 无（http 为主 crate 默认 feature） |
| `performance` | 使用 MockEngine 调用 process_text,统计平均/最大/最小推理耗时 | 无 |

## 运行命令

```bash
# 注意：vecboost-examples 不定义 http feature，加 --features http 会被 cargo 拒绝

# Prometheus 指标示例
cargo run -p vecboost-examples --bin metrics

# 推理性能监控示例
cargo run -p vecboost-examples --bin performance
```

## 依赖说明

- `InferenceCollector`: 收集推理记录、性能样本,支持摘要统计 (平均延迟/吞吐量/成功率)
- `PrometheusCollector`: 注册 HTTP 请求计数器、延迟直方图、活跃连接数、缓存命中/未命中等 Prometheus 指标（真实名 `http_requests_total` / `http_request_duration_seconds` / `active_connections` / `cache_hits_total` / `cache_misses_total`，见 `src/metrics/prometheus_exporter.rs`）
- `performance` 示例使用 `MockEngine` 避免依赖真实模型,通过 `EmbedResponse.processing_time_ms` 获取每次推理耗时
- 生产环境中 HTTP `/metrics` 端点自动暴露 Prometheus 格式指标供 Prometheus server 抓取
