# 基准回归扫描 — 2026-09-16（晚）

> 以 `docs/benchmarks/` 既有归档为基准的全量回归扫描，同时刷新吞吐与质量门数据。
> **结论：四个基准全部无回退，GGUF 质量门逐位复现，未做优化。**

## 测试环境

- **日期**: 2026-09-16 晚（约 21:00-21:50 CST）
- **代码**: `dev` 分支 HEAD `3198ff6`（含 f70016e colibri 吸收、c8d60f4 依赖升级）
- **主机**: AMD Ryzen 9 9950X（WSL2 可见 6C/12T），70GB RAM，Linux(WSL2) x86_64
- **工具链**: rustc 1.97.1，criterion 0.8.2
- **方法**: 归档数据用 criterion 默认配置（100 samples / 3s warm-up / 5s measure）；
  附注的 quick 口径为 `--sample-size 10 --warm-up-time 1 --measurement-time 2`。
  扫描期间机器有中等背景负载（load ~2-4 / 12 逻辑核）。

## 1. 向量相似度（similarity_bench）

对比基线：`kunpeng_tuning_after.md`（2026-08-14，两者间相似度核心实现未变）。

| Function | Dim | 基线 (ns) | 本次 (ns) | 变化 |
|----------|-----|-----------|-----------|------|
| cosine_similarity | 128 | 47.55 | 45.80 | **-3.7%** |
| cosine_similarity | 384 | 133.71 | 140.34 | +5.0% (噪声) |
| cosine_similarity | 768 | 284.60 | 271.92 | **-4.5%** |
| cosine_similarity | 1024 | 364.84 | 356.84 | **-2.2%** |
| euclidean_distance | 128 | 23.50 | 22.10 | **-6.0%** |
| euclidean_distance | 384 | 67.68 | 70.68 | +4.4% (噪声) |
| euclidean_distance | 768 | 134.44 | 135.53 | +0.8% |
| euclidean_distance | 1024 | 179.12 | 189.46 | +5.8% (噪声) |
| dot_product | 128 | 19.91 | 18.27 | **-8.2%** |
| dot_product | 384 | 59.63 | 58.50 | **-1.9%** |
| dot_product | 768 | 146.66 | 114.88 | **-21.7%** |
| dot_product | 1024 | 158.30 | 147.04 | **-7.1%** |
| manhattan_distance | 128 | 21.63 | 21.84 | +1.0% |
| manhattan_distance | 384 | 64.79 | 63.67 | **-1.7%** |
| manhattan_distance | 768 | 130.99 | 132.78 | +1.4% |
| manhattan_distance | 1024 | 206.33 | 172.75 | **-16.3%** |

- 16 项中 10 项快于基线，最大回退 +5.8%（euclidean/1024），在文档既定 ±5-10% 噪声带内
- dot/768 与 manhattan/1024 的基线当日即标注为偏高（见 `kunpeng_tuning_after.md` 注），本次回落属预期
- 对照标量基线（`baseline_results.md`），cosine/1024 仍保持 **2.9x**，spec R-similarity-001 的 1.5x 门槛满足

## 2. 语义缓存（semantic_cache_bench）

对比基线：`kunpeng_tuning_after.md`。

| Operation | 基线 | 本次 | 变化 |
|-----------|------|------|------|
| exact_cache_hit (1000 entries) | 10.125 ns | 9.485 ns | **-6.3%** |
| exact_cache_miss | 9.7093 ns | 9.653 ns | -0.6% |
| trigram_search_100 | 96.661 µs | 90.61 µs | **-6.3%** |
| trigram_search_1000 | 962.15 µs | 912.34 µs | **-5.2%** |
| trigram_search_10000 | 9.6466 ms | 9.1361 ms | **-5.3%** |

5 项全部持平或更快，无回退。

## 3. 批调度（batch_scheduling_bench）— 新机制，口径已更换

> **重要**：f70016e（2026-09-16）退役 DynamicBatchScheduler 与 ContinuousBatchLoop，
> 本基准整体重写为预填队列 + `assemble_batch` 时间窗拼批（`batch_wait_ms`，0=kill-switch）。
> 预填口径下耗时 ≈ 窗口时长 + ~1ms 轮询开销，**与旧归档中 6.194s/1.110s 等数字
> （请求匀速到达 + 旧调度器轮询口径）不可直接对比**。旧批调度数据自本日起仅存历史价值。

| Scenario | wait_50ms | wait_20ms | wait_5ms |
|----------|-----------|-----------|----------|
| steady（预填 100 请求，凑批排空） | 50.90 ms | 21.02 ms | 6.13 ms |
| burst（3 轮 × 50 请求） | 152.67 ms | 62.67 ms | 18.34 ms |

与机制设计完全吻合（各档 ≈ 窗口时长 + ~1ms），无异常。

## 4. 吞吐基线（embed_throughput_bench，bge-small-en-v1.5）

| 构建 | 单文本 | 32 文本 batch | 口径 |
|------|--------|---------------|------|
| 默认 | 中位 **81.8 ms** | 中位 **193.7 ms** | 100 samples（本次归档） |
| 默认 | 中位 69.7 ms | 中位 169.7 ms | quick（同晚 21:00 测） |
| `--features mkl` | 中位 **20.5 ms** | 中位 **157.6 ms** | 100 samples（本次归档，run1） |
| `--features mkl` | 中位 24.2 ms | 中位 185.8 ms | 100 samples（同晚 run2） |

mkl 命令：`RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc" VECBOOST_BENCH_MODEL=models/BAAI-bge-small-en-v1.5 cargo bench --bench embed_throughput_bench --features mkl`

### 环境漂移归因（非代码回退）

- 本次扫描中重负载多线程基准（embed）的绝对值随机器持续负载状态上浮 ~15-17%：
  同一份 HEAD 代码，默认档 quick（21:00）69.7 ms → 100-sample（21:45）81.8 ms；
  mkl 档两次 100-sample 间亦波动 20.5 → 24.2 ms（±9%）。
- **但加速比完全不变**：默认/mkl = 81.8/20.5 ≈ **4.0×**（文档记录 4.0×），
  batch 193.7/157.6 ≈ **1.23×**（文档记录 1.17×）。比值不变、绝对值整体平移 =
  环境状态漂移（持续负载降频），非代码性能回退。
- 单线程纳秒级微基准（相似度/缓存）同期无漂移，佐证上述归因。

## 5. GGUF 量化质量门（tests/quantized_parity）

`cargo test --test quantized_parity --features quantized-gguf --release`：4/4 通过，数值逐位复现：

| 格式 | 余弦中位数 | min | 门值 |
|------|-----------|-----|------|
| Q8_0 | **0.9999** | 0.9998 | ≥ 0.98 ✓ |
| Q4_K | **0.9987** | 0.9955 | ≥ 0.95 ✓ |

与 `PERFORMANCE.md` 记录（2026-09-16 凌晨）完全一致——确定性流程（本地 bge-small 自产 GGUF + 固定 32 条语料）。

## 6. 未复测项说明

| 归档数据 | 状态 |
|----------|------|
| 标量实现基线（`baseline_results.md` 上半） | 历史记录；SIMD chunked 已成唯一实现，不可复测 |
| DynamicBatchScheduler / ContinuousBatchLoop 批调度数据 | 组件已于 f70016e 退役，不可复测 |
| GPU 管线数据（`gpu_pipeline_*.md`） | 本机 WSL2 无 CUDA，仅编译级验证 |

## 复现

```bash
cargo bench --bench similarity_bench
cargo bench --bench semantic_cache_bench
cargo bench --bench batch_scheduling_bench
VECBOOST_BENCH_MODEL=models/BAAI-bge-small-en-v1.5 cargo bench --bench embed_throughput_bench
cargo test --test quantized_parity --features quantized-gguf --release
```
