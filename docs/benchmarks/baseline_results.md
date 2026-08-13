# Similarity Benchmark — Baseline Results

> 标量实现基线（优化前）。运行环境：Linux x86_64, Rust edition 2024, `cargo bench --release`。

## Baseline (Scalar Implementation)

| Function | Dimension | Time (ns/iter) | Throughput (M ops/s) |
|----------|-----------|----------------|----------------------|
| cosine_similarity | 128 | 89.04 | 11.23 |
| cosine_similarity | 384 | 356.51 | 2.81 |
| cosine_similarity | 768 | 758.81 | 1.32 |
| cosine_similarity | 1024 | 1047.20 | 0.95 |
| euclidean_distance | 128 | 32.01 | 31.24 |
| euclidean_distance | 384 | 114.93 | 8.70 |
| euclidean_distance | 768 | 244.92 | 4.08 |
| euclidean_distance | 1024 | 335.00 | 2.99 |
| dot_product | 128 | 27.09 | 36.92 |
| dot_product | 384 | 114.53 | 8.73 |
| dot_product | 768 | 257.94 | 3.88 |
| dot_product | 1024 | 332.81 | 3.00 |
| manhattan_distance | 128 | 29.88 | 33.47 |
| manhattan_distance | 384 | 117.39 | 8.52 |
| manhattan_distance | 768 | 265.04 | 3.77 |
| manhattan_distance | 1024 | 341.60 | 2.93 |

## Analysis

- **cosine_similarity** 最慢（3 次完整遍历：dot + 2x norm），1024 维约 1µs
- **dot_product / euclidean / manhattan** 均为单次遍历，1024 维约 330-345ns
- 所有函数随维度线性增长，无 SIMD 加速迹象
- 预期 SIMD chunk-based 展开可带来 2-4x 加速（尤其 dot_product 和 cosine_similarity）

---

## After Optimization (Chunk-based SIMD)

> chunk-based 4-wide 手动展开实现。编译器自动向量化为 SIMD 指令。

| Function | Dimension | Baseline (ns) | Optimized (ns) | Speedup |
|----------|-----------|---------------|----------------|----------|
| cosine_similarity | 128 | 89.04 | 45.19 | **1.97x** |
| cosine_similarity | 384 | 356.51 | 134.78 | **2.64x** |
| cosine_similarity | 768 | 758.81 | 286.02 | **2.65x** |
| cosine_similarity | 1024 | 1047.20 | 341.71 | **3.06x** |
| euclidean_distance | 128 | 32.01 | 22.83 | **1.40x** |
| euclidean_distance | 384 | 114.93 | 67.23 | **1.71x** |
| euclidean_distance | 768 | 244.92 | 145.92 | **1.68x** |
| euclidean_distance | 1024 | 335.00 | 194.53 | **1.72x** |
| dot_product | 128 | 27.09 | 18.47 | **1.47x** |
| dot_product | 384 | 114.53 | 57.47 | **1.99x** |
| dot_product | 768 | 257.94 | 141.68 | **1.82x** |
| dot_product | 1024 | 332.81 | 153.42 | **2.17x** |
| manhattan_distance | 128 | 29.88 | 20.54 | **1.45x** |
| manhattan_distance | 384 | 117.39 | 59.08 | **1.99x** |
| manhattan_distance | 768 | 265.04 | 125.01 | **2.12x** |
| manhattan_distance | 1024 | 341.60 | 168.16 | **2.03x** |

## Summary

- **cosine_similarity** 提升最显著：1024 维 **3.06x** 加速（3 次遍历 + 维度越大收益越大）
- **dot_product** 1024 维 **2.17x** 加速
- **euclidean_distance** 1024 维 **1.72x** 加速
- **manhattan_distance** 1024 维 **2.03x** 加速
- 所有函数在所有维度下均超过 1.4x 加速，满足 spec R-similarity-001 的 1.5x 最低要求（384+ 维度）
- 零 unsafe 代码，零新 crate 依赖，全部 42 个测试通过
