# GPU Pipeline 优化前基线数据

> **日期**: 2026-08-14
> **环境**: Linux x86_64, Rust stable, `cargo bench --release`

## 向量相似度 Benchmark（优化前快照）

> 此数据作为 gpu-pipeline-opt 变更的对比基线。

| Function | Dimension | Time (ns/iter) |
|----------|-----------|----------------|
| cosine_similarity | 128 | 77.52 |
| cosine_similarity | 384 | 231.48 |
| cosine_similarity | 768 | 518.33 |
| cosine_similarity | 1024 | 675.14 |
| euclidean_distance | 128 | 26.03 |
| euclidean_distance | 384 | 70.55 |
| euclidean_distance | 768 | 140.33 |
| euclidean_distance | 1024 | 187.40 |
| dot_product | 128 | 20.40 |
| dot_product | 384 | 58.50 |
| dot_product | 768 | 172.76 |
| dot_product | 1024 | 240.00 |
| manhattan_distance | 128 | 23.48 |
| manhattan_distance | 384 | 69.61 |
| manhattan_distance | 768 | 142.47 |
| manhattan_distance | 1024 | 241.56 |

## 参考：历史基线数据

详见：
- `docs/benchmarks/baseline_results.md` — 标量实现基线 + chunk-based SIMD 优化后数据
- `docs/benchmarks/batch_baseline_results.md` — 批调度基线 + ContinuousBatchLoop 对比
- `docs/benchmarks/semantic_cache_baseline_results.md` — 语义缓存基线
