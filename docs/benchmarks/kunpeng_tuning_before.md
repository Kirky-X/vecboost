# 鲲鹏性能优化十板斧 — 优化前基线数据

> 日期：2026-08-13
> 环境：Linux x86_64, Rust stable, `cargo bench --quick`
> 目的：记录优化前基线，用于对比防止恶性优化

## 相似度计算 Benchmark

| Function | Dimension | Time (ns/iter) |
|----------|-----------|----------------|
| cosine_similarity | 128 | 49.844 |
| cosine_similarity | 384 | 144.96 |
| cosine_similarity | 768 | 285.43 |
| cosine_similarity | 1024 | 398.84 |
| euclidean_distance | 128 | 22.931 |
| euclidean_distance | 384 | 70.419 |
| euclidean_distance | 768 | 140.59 |
| euclidean_distance | 1024 | 186.64 |
| dot_product | 128 | 22.655 |
| dot_product | 384 | 63.621 |
| dot_product | 768 | 134.07 |
| dot_product | 1024 | 159.49 |
| manhattan_distance | 128 | 22.904 |
| manhattan_distance | 384 | 65.214 |
| manhattan_distance | 768 | 134.58 |
| manhattan_distance | 1024 | 176.43 |

## 语义缓存 Benchmark

| Operation | Time |
|-----------|------|
| exact_cache_hit (1000 entries) | 10.083 ns |
| exact_cache_miss | 9.4028 ns |
| trigram_search_100 | 100.11 µs |
| trigram_search_1000 | 1.0515 ms |
| trigram_search_10000 | 11.618 ms |

## 批调度 Benchmark

> batch_scheduling_bench 使用 mock 引擎，测量调度延迟而非推理性能。
> 数据待补充（benchmark 编译时间较长）。
