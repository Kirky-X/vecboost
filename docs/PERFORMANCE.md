# ⚡ VecBoost 性能指南

VecBoost 为高吞吐、低延迟的嵌入向量服务而生。本指南汇总实测基准数据、性能设计要点、全部调优开关注册表与 A/B 实验纪律，帮助在生产负载下优化性能。

> 实证纪律（port 自 colibri）：默认保守、opt-in 开启、每个优化一个 kill-switch、每个性能数字标注主机配置与基准。本文件由 `docs/tuning.md` 与 `docs/benchmarks/` 归档数据合并而来。

## 📋 目录

<details open>
<summary>📑 目录（点击展开）</summary>

- [基准数据（实测）](#-基准数据实测)
- [吞吐基线（待实测）](#-吞吐基线待实测)
- [性能设计要点](#-性能设计要点)
- [GGUF 量化](#-gguf-量化)
- [开关与环境变量注册表](#-开关与环境变量注册表)
- [新增指标](#-新增指标)
- [调优建议](#-调优建议)
- [实验纪律](#-实验纪律)

</details>

---

## 📊 基准数据（实测）

> 口径：以下数据来自 `docs/benchmarks/` 归档，criterion 测量（`cargo bench --release` / `--quick`），采集日期 2026-08（Linux x86_64，Rust stable）。微基准测量噪声约 ±5-10%；`kunpeng_tuning_after.md` 中 dot_product/768 与 manhattan_distance/1024 的波动在正常范围内。环境差异（CPU 频率、系统负载、编译器版本）可导致 ±50% 级波动（见 `gpu_pipeline_after.md` 归因分析）。

### 向量相似度（similarity_bench）

标量基线 → chunk-based 4-wide 手动展开（编译器自动向量化为 SIMD）：

| Function | Dimension | 标量基线 (ns) | SIMD 优化后 (ns) | 加速比 |
|----------|-----------|---------------|------------------|--------|
| cosine_similarity | 128 | 89.04 | 45.19 | 1.97x |
| cosine_similarity | 384 | 356.51 | 134.78 | 2.64x |
| cosine_similarity | 768 | 758.81 | 286.02 | 2.65x |
| cosine_similarity | 1024 | 1047.20 | 341.71 | **3.06x** |
| euclidean_distance | 1024 | 335.00 | 194.53 | 1.72x |
| dot_product | 1024 | 332.81 | 153.42 | 2.17x |
| manhattan_distance | 1024 | 341.60 | 168.16 | 2.03x |

全部函数在全部维度下加速超过 1.4x；零 unsafe 代码、零新依赖。完整数据见 [baseline_results.md](benchmarks/baseline_results.md)。

### 批调度（batch_scheduling_bench）

DynamicBatchScheduler（固定等待窗）与 ContinuousBatchLoop（1ms tick + 条件刷新）对比（Mock 引擎，100 请求稳定负载 / 3 轮 × 50 请求突发负载）：

| 调度器 | 稳定负载总耗时 | 突发负载总耗时 |
|--------|----------------|----------------|
| DynamicBatchScheduler（wait_50ms，基线） | 6.194 s | 454 ms |
| DynamicBatchScheduler（wait_5ms） | 1.742 s（3.6x） | 323 ms（1.4x） |
| **ContinuousBatchLoop** | **1.110 s（5.6x）** | **157 ms（2.9x）** |

完整数据见 [batch_baseline_results.md](benchmarks/batch_baseline_results.md)。

### 语义缓存（semantic_cache_bench）

| Operation | 耗时 |
|-----------|------|
| 精确缓存命中（1000 条目） | ~10 ns |
| 精确缓存未命中 | ~9.7 ns |
| trigram 搜索（100 条目） | ~97 µs |
| trigram 搜索（1,000 条目） | ~962 µs |
| trigram 搜索（10,000 条目） | ~9.6 ms |

精确匹配为纳秒级（O(1)）；trigram 暴力搜索 O(n) 线性增长，10K 条目 < 10 ms。完整数据见 [semantic_cache_baseline_results.md](benchmarks/semantic_cache_baseline_results.md) 与 [kunpeng_tuning_after.md](benchmarks/kunpeng_tuning_after.md)。

---

## ⏱️ 吞吐基线

`embed_throughput_bench`（T012）度量单文本 `embed` 与 32 文本 `embed_batch`（bge-small 级模型，CPU），criterion 100 samples 取中位：

```bash
VECBOOST_BENCH_MODEL=models/BAAI-bge-small-en-v1.5 cargo bench --bench embed_throughput_bench
```

| 主机 | 配置 | 单文本 | 32 文本 batch | 备注 |
|------|------|--------|---------------|------|
| AMD Ryzen 9 9950X（WSL2 可见 6C/12T），70GB RAM，Linux(WSL2) x86_64 | 默认构建 | 中位 70.3 ms | 中位 173.6 ms（≈184 texts/s） | 2026-09-15 |
| 同上 | `--features mkl`（hgemm_ 垫片） | 中位 17.5 ms（**4.0×**） | 中位 148.4 ms（**1.17×**） | 2026-09-16；`RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"` 绕开 lld 链接 MKL 的已知问题 |

> **加速后端说明（T011 → T035 收敛闭环）**：mkl 已可用。上游版本错配——candle 0.11 调用 fp16 GEMM `hgemm_`，而 intel-mkl-src 0.8.1 Linux 静态路径锁死 MKL 2020.1（ghcr.io/rust-math，无该符号）——由 `src/engine/mkl_shim.rs` 的 `hgemm_` 垫片解决（f16 入 → f32 累加 `sgemm_` → f16 出，与硬件 hgemm 数值语义一致）。两个注意点：① 本机 rust-lld 链接 MKL 存在额外问题，需 `RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"`；② `--features mkl` 保持 opt-in，默认构建不引入 MKL。accelerate（macOS）保持 opt-in，本机无法验证。单文本 4.0× 主要来自 fp32 GEMM；32 批仅 1.17×，说明批路径瓶颈已不在 matmul。

---

## 🚀 性能设计要点

- **时间窗动态拼批**：`[pipeline.worker] batch_wait_ms`（默认 5ms，`0` 还原排空式），窗口内聚合请求凑满 `max_batch_size` 提前发出；
- **批内去重**：`/embed/batch` 相同文本批内只推理一次，结果按索引回填（字节等同）；
- **SIMD 相似度**：`src/utils/vector.rs` chunk-based 展开向量化；批量相似度经 rayon `par_iter()` 并行（多候选场景约 3-5x，8 核）；
- **批量 CLS 提取**：`narrow(1,0,1).squeeze(1).to_vec2()` 单次批量张量操作替代逐样本索引（批量 32 样本 ~30x 张量索引开销减少）；
- **物理核线程调优**：tokio/rayon 按物理核生效（SMT/E-core 检测），`VECBOOST_NO_THREAD_TUNE=1` 关闭；多 socket 启动输出 numactl 建议；
- **jemalloc**：Linux glibc 下 tikv-jemallocator 全局分配器（background_threads）；
- **GPU 管线**：CUDA 真实设备信息查询（nvidia-smi）、GPU 调优检测（持久模式/透明大页/时钟/ECC/计算模式，见 `scripts/gpu-tuning.sh`）、权重分页（WeightPagingManager LRU-K）；
- **release profile**：`opt-level = 3` + `lto = "fat"` + `codegen-units = 1` + `panic = "abort"` + `strip = true`。

---

## 🧊 GGUF 量化

- **开关**：`[model] quantized = true` + `model_path` 指向 `.gguf` 文件，且构建启用 `--features quantized-gguf`。默认 `false`（safetensors fp32 路径不变）。
- **回退**：`quantized = false` 或路径非 `.gguf` 即回原路径；未启用 feature 的构建遇到 GGUF 配置时启动报错（不静默回退）。
- **GGUF 获取**：外部文件用 `llama.cpp` 的 `convert_hf_to_gguf.py` + `llama-quantize` 转换（放 `models/` 下记录 sha256），经 `VECBOOST_GGUF_MODEL` 供质量门；也可用内置 `write_gguf_from_safetensors` 从本地 safetensors 自产（llama.cpp 命名约定，质量门缺省路径，零网络依赖）。
- **质量门**（`tests/quantized_parity.rs`，golden 语料 32 条中英混合）：Q8_0 与 fp32 输出余弦中位数 ≥ **0.98**，Q4_K ≥ **0.95**；未达门值不得标注为推荐配置。——**2026-09-16 实测通过**：
  - Q8_0：余弦中位数 **0.9999**（min 0.9998）✓
  - Q4_K：余弦中位数 **0.9987**（min 0.9955）✓
- **压缩实测**（bge-small-en-v1.5，fp32 133.5 MB）：Q8_0 **35.5 MB（3.76×）**，75 张量零兜底；Q4_K 56.4 MB（2.37×），63/75 张量因 384 维对 Q4_K 256 块不整除回退 F16（candle 0.11 量化无 padding）。**bge-small 级模型推荐 Q8_0**；Q4_K 真正生效需上游 padding 能力。
- **实现方式（T035 收敛落地）**：加载期反量化桥。candle-transformers 0.11 无 `quantized_bert`，故 `write_gguf_from_safetensors`（写出）+ `QuantizedCandleEngine::load`（逐张量反量化 f32 → 复用 `BertModel` 前向，路由经 `EngineFactory`）。收益为**存储与加载体积**，运行期计算与 fp32 等价，不宣称算力加速；算力加速走 `--features mkl`。

---

## 🎛️ 开关与环境变量注册表

> 命名以代码实现为准。注：变更提案原写 `[cache]` 段，实现时落位于实际解析的 `[embedding]`（persist 两项）与 `[semantic_cache]`（comparison_mode）——**`[cache]` 段当前版本不生效，勿在该段配置**。

| 开关 | 类型 | 默认 | 关闭/回退行为 | 验证入口 |
|------|------|------|---------------|----------|
| `VECBOOST_NO_THREAD_TUNE=1` | env | 未设（物理核调优启用） | 跳过检测，tokio/rayon 用 num_cpus | `device::thread_tune` 单测 |
| `[pipeline.worker] batch_wait_ms` | 配置 | 5 | `0` = 立即返回首请求（严格还原排空式） | `pipeline::worker` assemble_batch 测试 |
| `[pipeline.worker] max_batch_size` | 配置 | 8 | — | 同上 |
| `[model] quantized` | 配置 | false | false = safetensors fp32 原路径；feature 未编译遇 GGUF 启动报错 | `tests/quantized_parity.rs`（模型经 `VECBOOST_GGUF_MODEL`） |
| `--features quantized-gguf` | 构建 | 关 | 未编译不加载 GGUF | 同上 |
| `--features mkl` / `--features accelerate` | 构建 | 关（opt-in） | 不启用 = candle 默认内核。mkl 经 `hgemm_` 垫片已可链接（见吞吐基线节），本机需 `RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"`；accelerate 仅 macOS 未验证 | `benches/embed_throughput_bench.rs`（模型经 `VECBOOST_BENCH_MODEL`） |
| `[embedding] persist_path` | 配置 | None | 未设 = 纯内存，零写盘 | `cache::oxcache_backend` WAL 测试 |
| `[embedding] persist_max_bytes` | 配置 | None → 1 GiB | 超限触发启动紧凑化 | 同上 |
| `[semantic_cache] comparison_mode` | 配置 | exact | `exact` = 原始向量比较（现状）；`i8`/`binary` = vquant 粗筛+精确复验 | `cache::semantic_cache` 测试 |
| `[model] max_resident_models` | 配置 | None | 未设 = 不限制驻留（现状） | `model::manager` LFRU 测试 |
| `[model] resident_memory_budget_mb` | 配置 | None | 未设 = 不按内存预算驱逐 | 同上 |
| `[device] auto_plan` | 配置 | false | false = 零计划行为；true 时计划仅填充未显式配置的字段 | `device::planner` 测试 |
| `vecboost doctor` | CLI | — | 只读诊断：config/tokenizer/cache-persist/threads/gpu/models；有 FAIL 退出码 1 | `tests/doctor.rs` |
| `vecboost --warmup N` | CLI | 0 | 0 = 不预热 | 启动预热（T031） |
| `scripts/autotune.py` | 脚本 | — | 手动运行；安全门（漂移>1e-6 取消/增益<3% 不采纳/胜者反序复测） | `--help` |
| `scripts/validate_manifest.py` | 脚本 | — | 实验协议校验（字段/runs≥3/中位数/sha256） | [docs/experiments/README.md](experiments/README.md) |

---

## 📈 新增指标

- `vecboost_batch_size`（histogram）— 时间窗拼批后批次大小
- `vecboost_batch_wait_seconds`（histogram）— 窗口等待时长（0 窗口观测 0）
- `vecboost_inbatch_dedup_ratio`（gauge）— 批内去重率，全重复 n 条 = (n-1)/n
- `vecboost_stage_seconds{stage=tokenize|inference|pool}`（histogram）— 引擎分阶段每次调用均值；批次路径 drain 汇出，单文本累计延迟到下次批次或抓取

---

## 🔧 调优建议

| 目标 | 建议 |
|------|------|
| 降低 P99 延迟 | 缩短 `[pipeline.worker] batch_wait_ms`（或 `0` 关闭时间窗）；确认物理核线程调优生效（不要设置 `VECBOOST_NO_THREAD_TUNE=1`） |
| 提升吞吐 | 增大 `[pipeline.worker] max_batch_size` 与 `[model] batch_size`；启用批内去重；评估 `[semantic_cache] comparison_mode = i8` |
| 降低内存/体积 | GGUF 量化（过质量门后）；`[model] max_resident_models` / `resident_memory_budget_mb` 限制驻留 |
| 长期驻留缓存 | `[embedding] persist_path` 开启 WAL 落盘，配合 `persist_max_bytes` 控制紧凑化 |
| 硬件不匹配 | `[device] auto_plan = true` 生成保守计划；GPU 部署先跑 `scripts/gpu-tuning.sh`；极端性能场景评估 `scripts/pgo-build.sh`（PGO 构建） |
| 定位瓶颈 | 观察 `vecboost_stage_seconds{stage}` 分段指标；`vecboost doctor` 检查配置/线程/GPU 状态 |
| 单变量验证 | `scripts/autotune.py` 坐标下降实测调优（内置安全门），结果按实验协议归档 |

---

## 🔬 实验纪律

任何开关的采纳与否按 [docs/experiments/README.md](experiments/README.md) 协议执行：

1. **单变量**：一次实验只改一个变量；
2. **次数与中位数**：baseline 与 trial 各 ≥3 次，报中位数而非单次最佳，实验间预热；
3. **ABBA 顺序**：按 `baseline→trial→trial→baseline` 交替执行，抵抗机器热态漂移；
4. **吞吐不得搬移时间**：吞吐提升不得以 startup/首请求延迟为代价；无损优化必须证明输出与基线**字节等同**；
5. 产物为 `docs/experiments/<日期>-<slug>.manifest.json`，经 `python3 scripts/validate_manifest.py` 校验后连同 evidence 提交；结论（正反皆记）回写本文件对应条目。

---

## 📚 相关文档

| 文档 | 说明 |
|:-----|:-----|
| [🧪 测试场景矩阵](TEST_SCENARIOS.md) | 测试栈职责与场景穷举 |
| [📈 基准数据归档](benchmarks/) | 历史基准数据明细 |
| [🔬 实验协议](experiments/README.md) | A/B manifest 协议与校验器 |
| [🏗️ 架构文档](ARCHITECTURE.md) | 性能优化设计（批处理/内存/GPU/并发） |
| [📋 更新日志](CHANGELOG.md) | 调优开关的版本记录 |
