# 批量调度示例

演示 `DynamicBatchScheduler` — 基于 P99 延迟和吞吐量自动调整批量大小的动态调度器。

## 示例

| 名称 | 说明 |
|------|------|
| `batch_scheduler` | 自适应批量调整、优先级请求、并发控制、性能统计 |

## 运行

```bash
cargo run -p vecboost-examples --bin batch_scheduler
```

## 说明

`DynamicBatchScheduler` 是高吞吐场景的核心组件：

- **自适应调整**：基于 P99 延迟自动增减批量大小（目标 P99 < 80ms，吞吐 > 150 req/s）
- **优先级支持**：High / Normal / Low 三级优先级
- **并发控制**：信号量限制最大并发批次数
- **性能跟踪**：记录延迟/吞吐历史，支持统计查询

核心 API：
- `DynamicBatchScheduler::new(config)` — 创建调度器
- `scheduler.submit_request(request)` — 提交请求
- `scheduler.try_get_batch()` — 尝试获取批次
- `scheduler.record_batch_completion(size, latency)` — 记录完成并触发调整
- `scheduler.get_performance_stats()` — 查询性能统计
