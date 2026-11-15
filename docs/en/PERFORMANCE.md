# ⚡ VecBoost Performance Guide

VecBoost is built for high-throughput, low-latency embedding service. This guide consolidates measured benchmark data, performance design highlights, the full tuning-switch registry, and A/B experiment discipline to help you optimize under production load.

> Evidence discipline (ported from colibri): conservative defaults, opt-in switches, one kill-switch per optimization, and every performance number annotated with host configuration and baseline. Benchmark archives live in `docs/benchmarks/`.

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents (click to expand)</summary>

- [Benchmark data (measured)](#-benchmark-data-measured)
- [Throughput baseline](#-throughput-baseline)
- [Engine selection evidence (onnx vs candle vs embedded Python)](#-engine-selection-evidence-onnx-vs-candle-vs-embedded-python)
- [Performance design highlights](#-performance-design-highlights)
- [GGUF quantization](#-gguf-quantization)
- [Switch and environment-variable registry](#-switch-and-environment-variable-registry)
- [New metrics](#-new-metrics)
- [Tuning recommendations](#-tuning-recommendations)
- [Experiment discipline](#-experiment-discipline)

</details>

---

## 📊 Benchmark data (measured)

> **Link availability**: `docs/benchmarks/` is a **local-only** directory (`.gitignore`); the links below pointing to `benchmarks/*.md` do not exist in a clean clone; the key numbers are all inlined in the body.
>
> Basis: the data below comes from the `docs/benchmarks/` archive, criterion measurements (`cargo bench --release` / `--quick`), collected 2026-08 (Linux x86_64, Rust stable). Microbenchmark noise is roughly ±5-10%; the variance of dot_product/768 and manhattan_distance/1024 in `kunpeng_tuning_after.md` is within the normal range. Environment differences (CPU frequency, system load, compiler version) can cause ±50%-scale swings (see the attribution analysis in `gpu_pipeline_after.md`).
>
> **2026-09-16 regression sweep**: all benchmarks above were re-measured in full with no regressions (worst +5.8%, inside the noise band); see [regression_sweep_2026-09-16.md](../benchmarks/regression_sweep_2026-09-16.md).

### Vector similarity (similarity_bench)

Scalar baseline → chunk-based 4-wide manual unrolling (compiler auto-vectorizes to SIMD):

| Function | Dimension | Scalar baseline (ns) | After SIMD optimization (ns) | Speedup |
|----------|-----------|---------------|------------------|--------|
| cosine_similarity | 128 | 89.04 | 45.19 | 1.97x |
| cosine_similarity | 384 | 356.51 | 134.78 | 2.64x |
| cosine_similarity | 768 | 758.81 | 286.02 | 2.65x |
| cosine_similarity | 1024 | 1047.20 | 341.71 | **3.06x** |
| euclidean_distance | 1024 | 335.00 | 194.53 | 1.72x |
| dot_product | 1024 | 332.81 | 153.42 | 2.17x |
| manhattan_distance | 1024 | 341.60 | 168.16 | 2.03x |

Every function exceeds 1.4x speedup at every dimension; zero unsafe code, zero new dependencies. Full data in [baseline_results.md](../benchmarks/baseline_results.md).

2026-09-16 re-measurement: 10 of 16 items faster than the `kunpeng_tuning_after` baseline (cosine/1024 at 356.8 ns, scalar comparison 2.9x); worst +5.8% inside the noise band — no regression.

### Batch scheduling (batch_scheduling_bench)

> **⚠️ Historical data**: DynamicBatchScheduler and ContinuousBatchLoop were retired in f70016e (2026-09-16); the current mechanism is time-window batching `assemble_batch` (`[pipeline.worker] batch_wait_ms`, default 5ms, `0` = drain-style kill-switch). The table below is an archived record of the retired components and **is not comparable with the current mechanism**.

DynamicBatchScheduler (fixed wait window) vs ContinuousBatchLoop (1ms tick + conditional flush) comparison (Mock engine, 100-request steady load / 3 rounds × 50-request burst load):

| Scheduler | Steady-load total time | Burst-load total time |
|--------|----------------|----------------|
| DynamicBatchScheduler (wait_50ms, baseline) | 6.194 s | 454 ms |
| DynamicBatchScheduler (wait_5ms) | 1.742 s (3.6x) | 323 ms (1.4x) |
| **ContinuousBatchLoop** | **1.110 s (5.6x)** | **157 ms (2.9x)** |

Full data in [batch_baseline_results.md](../benchmarks/batch_baseline_results.md).

Current time-window batching benchmark (2026-09-16, pre-filled-queue basis — steady-state time ≈ window duration + ~1ms polling overhead):

| Scenario | wait_50ms | wait_20ms | wait_5ms |
|------|-----------|-----------|----------|
| steady (pre-filled 100 requests drained) | 50.9 ms | 21.0 ms | 6.1 ms |
| burst (3 rounds × 50 requests) | 152.7 ms | 62.7 ms | 18.3 ms |

### Semantic cache (semantic_cache_bench)

| Operation | Time |
|-----------|------|
| Exact cache hit (1000 entries) | ~10 ns |
| Exact cache miss | ~9.7 ns |
| trigram search (100 entries) | ~97 µs |
| trigram search (1,000 entries) | ~962 µs |
| trigram search (10,000 entries) | ~9.6 ms |

Exact matching is nanosecond-scale (O(1)); trigram brute-force search grows linearly O(n), 10K entries < 10 ms. Full data in [semantic_cache_baseline_results.md](../benchmarks/semantic_cache_baseline_results.md) and [kunpeng_tuning_after.md](../benchmarks/kunpeng_tuning_after.md).

2026-09-16 re-measurement: hit 9.5 ns / miss 9.7 ns / trigram 90.6 µs / 912 µs / 9.14 ms — all equal or faster, no regression.

---

## 🕐 Throughput baseline

> **Basis note (onnx-primary)**: the tables in this section are **historical baselines for candle (now demoted to opt-in fallback backend)**, used for longitudinal comparison of mkl/load-induced downclocking; **the primary-engine ort comparison data is in the next section, "Engine selection evidence"**.

`embed_throughput_bench` measures single-text `embed` and 32-text `embed_batch` (bge-small-class model, CPU), criterion 100 samples median:

```bash
VECBOOST_BENCH_MODEL=models/BAAI-bge-small-en-v1.5 cargo bench --bench embed_throughput_bench
```

| Host | Configuration | Single text | 32-text batch | Notes |
|------|------|--------|---------------|------|
| AMD Ryzen 9 9950X (WSL2-visible 6C/12T), 70GB RAM, Linux(WSL2) x86_64 | default build | median 70.3 ms | median 173.6 ms (≈184 texts/s) | 2026-09-15 |
| Same host | `--features mkl` (hgemm_ shim) | median 17.5 ms (**4.0×**) | median 148.4 ms (**1.17×**) | 2026-09-16; `RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"` to bypass the known lld-linking-MKL issue |
| Same host | default build + `--features mkl` (consecutive re-run same evening) | default 81.8 ms / mkl 20.5 ms (**4.0×**) | default 193.7 ms / mkl 157.6 ms (**1.23×**) | 2026-09-16 evening regression sweep ([regression_sweep_2026-09-16.md](../benchmarks/regression_sweep_2026-09-16.md)); the absolute values across the last two rows drift up ~15-17% under sustained-load downclocking (the same code measured 69.7 ms with the quick mode, matching the 2026-09-15 row); **the speedup ratio is unchanged — not a code regression** |

> **Acceleration backend note**: mkl is usable. The upstream version mismatch — candle 0.11 calls the fp16 GEMM `hgemm_`, while intel-mkl-src 0.8.1's Linux static path is locked to MKL 2020.1 (ghcr.io/rust-math, lacking the symbol) — is solved by the `hgemm_` shim in `src/engine/mkl_shim.rs` (f16 in → f32-accumulate `sgemm_` → f16 out, numerically consistent with hardware hgemm). Two caveats: ① rust-lld linking MKL has additional problems on this machine, requiring `RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"`; ② `--features mkl` stays opt-in — the default build does not pull in MKL. accelerate (macOS) stays opt-in and cannot be verified on this machine (verification protocol below). The single-text 4.0× comes mainly from fp32 GEMM; the 32-batch case only reaches 1.17×, showing the batch-path bottleneck is no longer matmul.

### Accelerate (macOS) pending-verification protocol

`accelerate` is candle's macOS acceleration backend, same opt-in class as the local MKL profile. **No data has been collected** (no Apple hardware/OS); the baseline table must not contain inferred values or cross-platform transplants. When conditions allow, measure per the following protocol:

- **Prerequisites**: macOS 13+ (Apple Silicon or Intel), Rust stable, `--features accelerate` (implies candle).
- **Command**: `VECBOOST_BENCH_MODEL=<local bge-small directory> cargo bench --bench embed_throughput_bench`.
- **Basis**: criterion 100 samples median; idle windows (no concurrent load, fixed power mode, background indexing off); 2–3 runs each for single-text and 32-text batch, take the median.
- **Recording**: add a `macOS + --features accelerate` row to the "Throughput baseline" table, together with a same-machine `default build` comparison row (otherwise the speedup is not attributable).
- **Judgment**: report only measured speedups, never a preset multiplier; if the batch path is <1.1×, record it under the existing "batch-path bottleneck is no longer matmul" conclusion.
- **Forbidden**: filling the macOS row with MKL/Linux data; keep the "pending verification" status when no measurement exists.

---

## 🔗 Redis shared backend (multi-replica)

The Redis path for rate limiting and the exact cache introduces 1–2 network round trips per request (Lua eval). Redis is not deployed on this machine and **no latency data has been collected**; once an environment exists, measure per the following basis:

- **Prerequisites**: Redis 7+, `--features distributed-redis`;
- **Command**: `VECBOOST_TEST_REDIS_URL=redis://127.0.0.1:6379 cargo bench --bench embed_throughput_bench` (hot cache path) + a dedicated rate-limit Lua latency stress test;
- **Basis**: criterion 100 samples median, idle windows; compare the per-request P50/P95 delta against the `memory` backend;
- **Recording**: add a `Redis (same host)` row to this section with the network RTT noted; estimated values must not substitute for measurements.

> Functional semantics: global and per-key rate limiting are both Lua-atomic fixed windows (per-key fixed 60s; `global_window_ms` applies only to the global dimension); a fixed window allows up to 2× quota bursts at window boundaries, and rejected requests are counted too. The exact cache, the semantic-cache fuzzy index, and the ban list can all be shared (via `[embedding] cache_backend`, `[semantic_cache] backend`, and `distributed-redis` respectively); the bloom filter stays per-replica.
>
> Degradation semantics: Redis rate-limit checks are fail-closed — a request is rejected on any error or when a single Lua round trip exceeds 500ms without response (bounded timeout; a half-dead Redis never hangs request threads); if Redis is unreachable at startup, startup is refused loudly, never silently falling back to memory. Non-TLS `redis://` connections are rejected by the oxcache guard by default; use `rediss://` or the development escape hatch `OXCACHE_ALLOW_INSECURE_REDIS=I_UNDERSTAND_THE_RISKS`.

---

## 🧭 Engine selection evidence (onnx vs candle vs embedded Python)

> **Measurement conditions**: 2026-10-18, AMD Ryzen 9 9950X (16 cores) / 66 GiB / RTX 5080 16 GiB (WSL2);
> ORT ① `3rdparty/libonnxruntime.so.1.28.2` (CPU) ② `onnxruntime-node 1.24.3` (with CUDA EP);
> candle is official 0.11; embeddings use a dual export of the same model (bge-small-en-v1.5, single / batch×32);
> transcription uses Qwen3-ASR (0.6B/1.7B, `assets/audio_from_m4a.wav` 699.777 s); each cell is the stable value across 2–3 runs.

### Embedding and rerank

| Engine × device | Embed single | Embed batch×32 | Rerank single pair | Rerank batch×32 | VRAM |
|---|---|---|---|---|---|
| **ort × CPU (product default)** | **2.56 ms** | **26.35 ms** | 5.55 ms | 47.78 ms | — |
| candle × CPU | 78.0 ms | 186.5 ms | 109.3 ms | 278.0 ms | — |
| **ort × CUDA** | 2.70 ms | **3.06 ms** | **2.79 ms** | **3.47 ms** | ~1.96 GB / sm 37.4% |
| candle × CUDA (`candle-cuda`) | 5.05 ms | 6.70 ms | 5.78 ms | 7.83 ms | ~1.95 GB / sm 44.1% |

- **The direct reason for the primary-engine switch**: candle CPU is **30.5×** slower single and **7.1×** slower batched; `--features mkl`'s 4× only brings single-text down to ~17.5 ms, still ~6.8× slower than ort.
- **GPU head-to-head**: ort is **1.9~2.3×** faster than `candle-cuda` at the same VRAM level; but candle GPU is **15.4~35.5×** faster than its own CPU — what lags is the engine implementation, not the device.
- **Embedding quantization**: `int8` gives −22% single / −16% batch / −55% memory (204→92 MiB) with median cosine **0.9971**; `q4` is faster but cosine drops to 0.9849.
  Gate: `tests/engine_backend_parity.rs::onnx_quantized_vs_fp32_parity`.
- **Memory shape**: embedding/rerank single process 92~269 MiB; **translation loading both en-zh + en-es bundles in the same process needs 1 927 MiB**.

### Transcription (no candle implementation; only ort vs embedded Python)

| Path | 700 s audio | CER | Resources |
|---|---|---|---|
| **ort 0.6B int8 × CPU** | **70.4 s (4 parallel) / 137.2 s (service serial)** | 7.75% | no VRAM |
| ort 0.6B int8 × CUDA | 199.4 s | 7.36% | 14.4 GiB (**regression 2.6×**) |
| embedded Python 0.6B bf16 × CUDA | 51.2 s | 22.81% (collapses on long audio) | 11.7 GiB + Python stack |
| ort 1.7B int8 × CPU | 135.7~142.1 s | 15.78~18.99% | no VRAM |
| ort 1.7B int8 × CUDA | 216.4 s | 12.66% | 15.6 GiB (regression 1.6×) |
| **ort 1.7B fp32 (lossless) × CPU** | 698.8 s | **7.24%** | no VRAM |
| ort 1.7B fp32 × CUDA | 1 576.3 s | 7.24% | 15.9 GiB (regression 2.3×) |
| **embedded Python 1.7B bf16 × CUDA** | **55.9 s** | **2.49%** | 14.5 GiB + Python stack |

- **GPU is a regression across the board for transcription**: no exception across the three precision paths — SM utilization ~15% while VRAM fills up; the per-segment decoding kernels are too small and dominated by launch/synchronization overhead.
- **Quantization attribution** (mixed bundles): dual fp32 7.24% → **int8 encoder + fp32 decoder 7.27%** (near-lossless, saves 6% time)
  → fp32 encoder + int8 decoder **15.13%** → dual int8 15.78%. **All the loss comes from the decoder's dynamic quantization**.
- **Service basis**: `TRANSCRIBE_WORKERS=1` (isomorphic to the server's `permits=1`) is 1.8~2.0× slower than 4-parallel;
  the measured HTTP end-to-end (scenario suite) server RTF is 0.194, 0.9% apart from the CLI serial mode — HTTP+JSON overhead is only 3~5%.
- **fp16**: the engine is f32-only (KV cache and tensor I/O hardwired to f32), with **load-time explicit rejection** already added;
  full fp16 support requires dtype generalization (one KV step is 29.4M elements: 117 MiB f32 / 59 MiB f16 — boundary conversions would be memory-bandwidth bound).

> The full matrices and per-cell evidence (md5 fingerprints, mixed-bundle decomposition, fp16 guard) live in local-only directories:
> `docs/benchmarks/transcribe_engine_baseline.md` and `docs/benchmarks/embed_rerank_translate_baseline.md` (not committed per `.gitignore`).

---

## 🚀 Performance design highlights

- **Time-window dynamic batching**: `[pipeline.worker] batch_wait_ms` (default 5ms, `0` restores drain-style); requests within the window are aggregated and dispatched early once `max_batch_size` is reached;
- **In-batch dedup**: identical texts in `/embed/batch` are inferred once per batch, results back-filled by index (bitwise equal);
- **SIMD similarity**: `src/utils/vector.rs` chunk-based unrolled vectorization; batch similarity `calculate_similarity_batch` is a serial iteration (deliberate trade-off: with `collect::<Result>` under rayon parallelism the "first error" is not reproducible; this function is currently test-only — the service hot path has no batch-similarity call);
- **Batched CLS extraction**: `narrow(1,0,1).squeeze(1).to_vec2()` — one batched tensor operation replaces per-sample indexing (~30x less tensor-indexing overhead for a 32-sample batch);
- **Physical-core thread tuning**: tokio/rayon sized to physical cores (SMT/E-core detection), disabled by `VECBOOST_NO_THREAD_TUNE=1`; multi-socket startups print numactl advice;
- **jemalloc**: tikv-jemallocator global allocator on Linux glibc (background_threads);
- **GPU pipeline**: real CUDA device info via nvidia-smi, GPU tuning detection (persistence mode / transparent huge pages / clocks / ECC / compute mode — see `scripts/gpu-tuning.sh`), weight paging (WeightPagingManager LRU-K);
- **release profile**: `opt-level = 3` + `lto = "fat"` + `codegen-units = 1` + `panic = "abort"` + `strip = true`.

---

## 🧊 GGUF quantization

- **Switch**: `[model] quantized = true` + `model_path` pointing to a `.gguf` file, and a build with `--features quantized-gguf`. Default `false` (the safetensors fp32 path is unchanged).
- **Fallback**: `quantized = false` or a non-`.gguf` path returns to the original path; a build without the feature errors at startup when it encounters a GGUF configuration (no silent fallback).
- **Obtaining GGUF**: convert external files with llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` (place under `models/` and record sha256), fed to the quality gate via `VECBOOST_GGUF_MODEL`; or self-produce from local safetensors with the built-in `write_gguf_from_safetensors` (llama.cpp naming convention, the quality gate's default path, zero network dependency).
- **Quality gate** (`tests/quantized_parity.rs`, golden corpus of 32 mixed Chinese/English lines): median cosine of Q8_0 vs fp32 ≥ **0.98**, Q4_K ≥ **0.95**; below-gate configurations must not be labeled as recommended. — **Measured pass on 2026-09-16 (bitwise identical on that evening's regression sweep re-run)**:
  - Q8_0: median cosine **0.9999** (min 0.9998) ✓
  - Q4_K: median cosine **0.9987** (min 0.9955) ✓
- **Compression measured** (bge-small-en-v1.5, fp32 133.5 MB): Q8_0 **35.5 MB (3.76×)** with 75 tensors and zero fallback; Q4_K 56.4 MB (2.37×) with 63/75 tensors falling back to F16 because 384 dims do not divide evenly into Q4_K 256 blocks (candle 0.11 quantization has no padding). **Q8_0 is recommended for bge-small-class models**; Q4_K needs upstream padding support to be truly effective.
- **Implementation**: a load-time dequantization bridge. candle-transformers 0.11 has no `quantized_bert`, so `write_gguf_from_safetensors` (writing) + `QuantizedCandleEngine::load` (per-tensor dequantization to f32 → reuse of the `BertModel` forward, routed via `EngineFactory`). The benefit is **storage and load size**; runtime computation is fp32-equivalent — no compute speedup is claimed; compute acceleration goes through `--features mkl`.

---

## 🔀 Switch and environment-variable registry

> Naming follows the code implementation. Note: the change proposal originally wrote the `[cache]` section, but the implementation landed in the actually-parsed `[embedding]` (the two persist keys) and `[semantic_cache]` (comparison_mode) — **the `[cache]` section does not take effect in the current version; do not configure it**.

| Switch | Type | Default | Off/fallback behavior | Verification entry |
|------|------|------|---------------|----------|
| `VECBOOST_NO_THREAD_TUNE=1` | env | unset (physical-core tuning enabled) | Skips detection; tokio/rayon use num_cpus | `device::thread_tune` unit test |
| `[pipeline.worker] batch_wait_ms` | config | 5 | `0` = return the first request immediately (strictly restores drain-style) | `pipeline::worker` assemble_batch test |
| `[pipeline.worker] max_batch_size` | config | 8 | — | Same as above |
| `[model] quantized` | config | false | false = safetensors fp32 original path; a build without the feature errors at startup on GGUF | `tests/quantized_parity.rs` (model via `VECBOOST_GGUF_MODEL`) |
| `--features quantized-gguf` | build | off | GGUF is not loaded without compilation | Same as above |
| `--features mkl` / `--features accelerate` | build | off (opt-in) | Without them = candle default kernels. mkl now links via the `hgemm_` shim (see the throughput baseline section); this machine needs `RUSTFLAGS="-C linker=x86_64-linux-gnu-gcc"`; accelerate is macOS-only and unverified | `benches/embed_throughput_bench.rs` (model via `VECBOOST_BENCH_MODEL`) |
| `[embedding] persist_path` | config | None | Unset = pure memory, zero disk writes | `cache::oxcache_backend` WAL test |
| `[embedding] persist_max_bytes` | config | None → 1 GiB | Over-limit triggers compaction at startup | Same as above |
| `[semantic_cache] comparison_mode` | config (deprecated) | exact | **Deprecated and ineffective**: parsing is kept for compatibility (invalid values error at startup), always exact | `cache::semantic_cache` test |
| `[model] max_resident_models` | config | None | Unset = no residency limit (current behavior) | `model::manager` LFRU test |
| `[model] resident_memory_budget_mb` | config | None | Unset = no memory-budget eviction | Same as above |
| `[device] auto_plan` | config | false | false = zero planning behavior; when true the plan only fills fields not explicitly configured | `device::planner` test |
| `vecboost doctor` | CLI | — | Read-only diagnostics: config/tokenizer/cache-persist/threads/gpu/models; exit code 1 on FAIL | `tests/doctor.rs` |
| `vecboost --warmup N` | CLI | 0 | 0 = no warmup | Startup warmup |
| `scripts/autotune.py` | script | — | Manual run; safety gates (cancel on drift >1e-6 / reject gains <3% / winner re-tested in reverse order) | `--help` |

---

## 📈 New metrics

- `vecboost_batch_size` (histogram) — batch size after time-window batching
- `vecboost_batch_wait_seconds` (histogram) — window wait duration (0 observed with a zero window)
- `vecboost_inbatch_dedup_ratio` (gauge) — in-batch dedup ratio; n all-duplicate items = (n-1)/n
- `vecboost_stage_seconds{stage=tokenize|inference|pool}` (histogram) — per-call engine stage averages; batch paths export at drain, single-text latency accumulates until the next batch or scrape

---

## 🧮 SIMD vector reduction and compilation targets

The four hot functions in `src/utils/vector_simd.rs` (dot/sum_of_squares/euclidean/manhattan)
use 4 independent accumulators to break the serial dependency chain, letting the compiler
auto-vectorize them into SIMD reductions.
**The actual code generation depends on the compilation target features**:

| Scenario | RUSTFLAGS | Expectation |
|:-----|:----------|:-----|
| Default build | (none) | x86-64 SSE2 baseline / aarch64 NEON baseline |
| Production recommendation | `-C target-cpu=native` | All locally available ISAs (AVX2/FMA/AVX-512) |
| Platinum distribution | `RUSTFLAGS="-C target-cpu=skylake-avx512"` etc. | Targeted at specific machine models |

Verification: `cargo bench --bench similarity_bench` comparing cosine_similarity throughput per dimension;
or `cargo rustc -- --emit asm` to check `vmulps/vaddps` (vectorized) vs `vmulss` (scalar).
Note: the floating-point reduction order is identical across target-cpu choices (the 4-accumulator merge order is fixed),
but the SIMD intra-lane merge may differ from the scalar baseline by ~1e-7 — semantic-cache threshold decisions are unaffected.

**⚠️ fat-LTO codegen drift observation (2026-09-24)**: under this repo's `[profile.release] lto = "fat"` +
`codegen-units = 1`, whether vector reductions get vectorized depends on the **overall shape of the whole-program IR** —
crate growth can make the same function degrade from vectorized to scalar in cross-fat-LTO assembly (a measured 2.1× gap under identical conditions).
Always do A/B benchmark comparisons **within the same commit tree**; cross-tree comparisons: see the `docs/benchmarks/` archive notes.

---

## 🔧 Tuning recommendations

| Goal | Recommendation |
|------|------|
| Lower P99 latency | Shorten `[pipeline.worker] batch_wait_ms` (or `0` to disable the time window); confirm physical-core thread tuning is active (do not set `VECBOOST_NO_THREAD_TUNE=1`) |
| Raise throughput | Increase `[pipeline.worker] max_batch_size` and `[model] batch_size`; enable in-batch dedup |
| Lower memory/size | GGUF quantization (after passing the quality gate); `[model] max_resident_models` / `resident_memory_budget_mb` to limit residency |
| Long-residency cache | `[embedding] persist_path` enables WAL persistence, with `persist_max_bytes` controlling compaction |
| Hardware mismatch | `[device] auto_plan = true` generates a conservative plan; run `scripts/gpu-tuning.sh` before GPU deployment; evaluate `scripts/pgo-build.sh` (PGO build) for extreme performance scenarios |
| Locate bottlenecks | Watch the `vecboost_stage_seconds{stage}` per-stage metrics; `vecboost doctor` checks config/thread/GPU state |
| Single-variable verification | `scripts/autotune.py` coordinate-descent measured tuning (with built-in safety gates); archive results to `docs/benchmarks/` |

---

## 🔬 Experiment discipline

Whether to adopt any switch follows this discipline:

1. **Single variable**: one experiment changes one variable at a time;
2. **Runs and medians**: ≥3 runs each for baseline and trial, report the median rather than the single best, warm up between experiments;
3. **ABBA order**: alternate as `baseline→trial→trial→baseline` to resist machine thermal drift;
4. **Throughput must not borrow time**: throughput gains must not come at the cost of startup/first-request latency; lossless optimizations must prove byte-equality of output with the baseline.

Conclusions (positive or negative) are written back to the corresponding entries in this file.

---

## 📚 Related Documentation

| Document | Description |
|:-----|:-----|
| [🧪 Test Scenarios](TEST_SCENARIOS.md) | Test-stack responsibilities and the exhaustive scenario matrix |
| [📈 Benchmark archive](../benchmarks/) | Historical benchmark details (**local-only**, excluded by `.gitignore`, absent in a clean clone) |
| [🏗️ Architecture](ARCHITECTURE.md) | Performance optimization design (batching/memory/GPU/concurrency) |
| [📋 Changelog](CHANGELOG.md) | Version history of tuning switches |
