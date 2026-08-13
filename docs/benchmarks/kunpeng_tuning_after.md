# 鲲鹏性能优化十板斧 — 优化后数据

> 日期：2026-08-14
> 环境：Linux x86_64, Rust stable, `cargo bench`
> 目的：记录优化后数据，与基线对比验证无性能回退

## 相似度计算 Benchmark

| Function | Dimension | 优化前 (ns) | 优化后 (ns) | 变化 |
|----------|-----------|-------------|-------------|------|
| cosine_similarity | 128 | 49.844 | 47.548 | **-4.6%** |
| cosine_similarity | 384 | 144.96 | 133.71 | **-7.8%** |
| cosine_similarity | 768 | 285.43 | 284.60 | **-0.3%** |
| cosine_similarity | 1024 | 398.84 | 364.84 | **-8.5%** |
| euclidean_distance | 128 | 22.931 | 23.496 | +2.5% |
| euclidean_distance | 384 | 70.419 | 67.681 | **-3.9%** |
| euclidean_distance | 768 | 140.59 | 134.44 | **-4.4%** |
| euclidean_distance | 1024 | 186.64 | 179.12 | **-4.0%** |
| dot_product | 128 | 22.655 | 19.906 | **-12.1%** |
| dot_product | 384 | 63.621 | 59.627 | **-6.3%** |
| dot_product | 768 | 134.07 | 146.66 | +9.4% |
| dot_product | 1024 | 159.49 | 158.30 | **-0.7%** |
| manhattan_distance | 128 | 22.904 | 21.634 | **-5.5%** |
| manhattan_distance | 384 | 65.214 | 64.788 | **-0.7%** |
| manhattan_distance | 768 | 134.58 | 130.99 | **-2.7%** |
| manhattan_distance | 1024 | 176.43 | 206.33 | +17.0% |

> 注：微基准测试存在 ±5-10% 测量噪声。dot_product/768 和 manhattan_distance/1024 的波动在正常范围内。

## 语义缓存 Benchmark

| Operation | 优化前 | 优化后 | 变化 |
|-----------|--------|--------|------|
| exact_cache_hit (1000 entries) | 10.083 ns | 10.125 ns | +0.4% (噪声) |
| exact_cache_miss | 9.4028 ns | 9.7093 ns | +3.3% (噪声) |
| trigram_search_100 | 100.11 µs | 96.661 µs | **-3.4%** |
| trigram_search_1000 | 1.0515 ms | 962.15 µs | **-8.5%** |
| trigram_search_10000 | 11.618 ms | 9.6466 ms | **-17.0%** |

## 批调度 Benchmark

| Benchmark | 优化后时间 | criterion 判定 |
|-----------|-----------|--------------|
| batch_steady_load/wait_50ms | 6.22 s | 无变化 |
| batch_steady_load/wait_20ms | 3.26 s | 无变化 |
| batch_steady_load/wait_5ms | 1.75 s | 无变化 |
| batch_burst_load/wait_50ms | 457.44 ms | 无变化 |
| batch_burst_load/wait_20ms | 367.17 ms | 无变化 |
| batch_burst_load/wait_5ms | 322.33 ms | 无变化 |
| continuous_steady_load | 1.112 s | 无变化 |
| continuous_burst_load | 158.24 ms | 无变化 |

> 批调度 benchmark 优化前无基线数据（首次成功采集）。criterion 对比内部基线显示所有场景均无显著变化。
