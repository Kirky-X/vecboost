# Batch Scheduling Baseline Results

## 测试环境
- **日期**: 2026-08-09
- **CPU**: Host system
- **Rust**: stable (bench profile, opt-level=3)
- **Criterion**: --quick mode

## 基线数据：当前 DynamicBatchScheduler

### 稳定负载（100 请求，~100 req/s 均匀到达）

| max_wait_time_ms | 总耗时 | 每请求平均延迟 | 吞吐量 |
|------------------|--------|---------------|--------|
| 50ms (默认) | 6.194s | ~61.9ms | ~16 req/s |
| 20ms | 3.219s | ~32.2ms | ~31 req/s |
| 5ms | 1.742s | ~17.4ms | ~57 req/s |

### 突发负载（3 轮 × 50 请求，瞬间到达）

| max_wait_time_ms | 总耗时 | 每请求平均延迟 | 吞吐量 |
|------------------|--------|---------------|--------|
| 50ms (默认) | 454ms | ~3.0ms | ~330 req/s |
| 20ms | 366ms | ~2.4ms | ~410 req/s |
| 5ms | 323ms | ~2.2ms | ~465 req/s |

## 分析

### 延迟瓶颈
1. **稳定负载下**：每个请求的等待时间 ≈ max_wait_time_ms（50ms 默认值），因为请求逐个到达，每个都需要等待凑批超时
2. **突发负载下**：50 个请求同时到达，第一批 32 个（max_batch_size）在 50ms 后被收集，剩余 18 个再等 50ms，整体延迟被摊薄

### 核心问题
- 当前调度器是**被动凑批**：`try_get_batch()` 被外部轮询调用，非主动调度
- 固定 `max_wait_time_ms=50ms` 导致低负载时每个请求额外等待 0-50ms
- 高负载时批量大小阈值（32）和等待时间（50ms）的交互导致不可预测的延迟

### ContinuousBatchLoop 预期改进
- 消除固定等待：1ms tick 粒度 + 条件刷新
- 稳定负载 P50 延迟预期从 ~50ms 降至 ~5ms
- 突发负载延迟变化不大（已受 batch_size 阈值主导）

---

## ContinuousBatchLoop 对比数据

### 稳定负载对比（100 请求，~100 req/s）

| 调度器 | 总耗时 | 对比基线(wait_50ms) |
|----------|--------|---------------------|
| DynamicBatchScheduler (wait_50ms) | 6.194s | 基线 |
| DynamicBatchScheduler (wait_20ms) | 3.219s | 1.9x |
| DynamicBatchScheduler (wait_5ms) | 1.742s | 3.6x |
| **ContinuousBatchLoop** | **1.110s** | **5.6x** |

### 突发负载对比（3 轮 × 50 请求）

| 调度器 | 总耗时 | 对比基线(wait_50ms) |
|----------|--------|---------------------|
| DynamicBatchScheduler (wait_50ms) | 454ms | 基线 |
| DynamicBatchScheduler (wait_20ms) | 366ms | 1.2x |
| DynamicBatchScheduler (wait_5ms) | 323ms | 1.4x |
| **ContinuousBatchLoop** | **157ms** | **2.9x** |

### 分析

ContinuousBatchLoop 相比默认 DynamicBatchScheduler (wait_50ms)：
- **稳定负载：5.6x 更快** — 消除 0-50ms 固定等待，1ms tick 粒度实现即时响应
- **突发负载：2.9x 更快** — 优先级感知出队 + 条件刷新减少无效等待
- 甚至优于 wait_5ms 配置（稳定 1.57x，突发 2.06x），因为连续调度无需任何固定等待
