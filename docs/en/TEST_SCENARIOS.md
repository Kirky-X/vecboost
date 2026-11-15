# 🧪 VecBoost Test Scenario Matrix

> Applies to VecBoost **0.3.0** (workspace, Rust edition 2024 / `rust-version = 1.91`).
> Purpose: enumerate the test-stack responsibility split and every acceptance scenario, answering "where should this test live".
> Authoring basis (read-only cross-check): `tests/scenario/*.py` (16 suites and the R-* scenario IDs in their docstrings), `tests/*.rs`, `tests/perf/`, `benches/`, `.github/workflows/` (health-check / feature-matrix / scenario-tests), `scripts/run-scenario-tests.sh`, `tests/scenario/conftest.py`.
> All suite names and scenario IDs were verified to exist via `grep`.

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents (click to expand)</summary>

- [Three-layer test stack overview](#-three-layer-test-stack-overview)
- [Scenario suite matrix (tests/scenario)](#-scenario-suite-matrix-testsscenario)
- [Rust integration and specialized tests](#-rust-integration-and-specialized-tests)
- [Performance and benchmark tests](#-performance-and-benchmark-tests)
- [Unit tests](#-unit-tests)
- [Run commands (identical to CI)](#-run-commands-identical-to-ci)
- [Test modes and fixtures](#-test-modes-and-fixtures)
- [Model matrix](#-model-matrix)
- [Conventions](#-conventions)

</details>

---

## 🧭 Three-layer test stack overview

VecBoost's performance/quality verification has three layers with these responsibilities:

| Layer | Location | Engine | Responsibility | How to run |
|---|---|---|---|---|
| **Microbenchmarks** | `benches/` (criterion) | Mock or real | Measure absolute performance of atomic operations and guard against regressions (similarity SIMD, batch scheduling, semantic-cache indexing, embed/rerank/translate throughput baselines) — 6 suites | `cargo bench -p vecboost` |
| **Regression thresholds** | `tests/perf/performance_test.rs` | MockEngine | Assert throughput/latency does not degrade (relative thresholds); CI-runnable, no real models needed | `cargo test -p vecboost --test perf` |
| **Real-service E2E** | `tests/scenario/*.py` (pytest) | Real binary + local `models/` | End-to-end behavior contracts: HTTP matrix/CLI/MCP/lifecycle/security; P99 latency and real throughput | `pytest tests/scenario -q` (conftest spawns servers on ports 9101-9143) |

---

## 📁 Scenario suite matrix (tests/scenario)

Each "configuration profile" is an independent server process: the config is written to `tests/scenario/run/<name>/config/config.toml` and the compiled binary is launched with that directory as CWD (the app reads `config/config.toml` from CWD). All assertions are black-box observations over HTTP/process behavior with no reliance on internal state. Case names start with `R-<domain>-NNN` for reconciliation (scenario IDs live in the specmark change specs). The pytest scenario server port ranges: session fixtures 9101-9107, model matrix 9111-9114, bad models 9120-9122, probes 9130-9132, lifecycle 9141-9143, gRPC 9151, config probes 9155-9161, fault-tolerance contract 9165, semantic cache 9166-9167, auth expiry 9133, API audit 9171-9174.

| Suite | Scenario IDs | Cases | Coverage |
|------|---------|--------|----------|
| `test_embed_normal.py` | R-embed-001 ~ R-embed-008 | 8 | Embedding service normal scenarios (M1 = BAAI/bge-small-en-v1.5, 384 dims) |
| `test_embed_abnormal.py` | R-embed-009 ~ R-embed-010 | 5 | Embedding service abnormal scenarios (overlong text, invalid inputs, other error paths) |
| `test_auth.py` | R-auth-001 ~ R-auth-011 | 11 | Authentication scenarios (auth profile, ports 9103/9133; invalid refresh_token, malformed Authorization, second-level expiry full chain) |
| `test_security.py` | R-auth-009 ~ R-auth-011 | 6 | Rate limiting, allowlist, path traversal, error masking, audit log |
| `test_server_modes.py` | R-server-001 ~ R-server-008 | 9 | Server modes (default / custom config / MCP mode startup paths) |
| `test_lifecycle.py` | R-server-002 / R-auth-002 | 3 | Lifecycle gap-filling (graceful shutdown, signal handling) |
| `test_cli_mcp.py` | R-server-003 (design.md M4/M5) | 8 | CLI / MCP mode probes |
| `test_config_device.py` | R-config-001 ~ R-config-006 | 6 | Config and device scenarios (`--config` fail-fast, env overrides, GPU fallback) |
| `test_http_matrix.py` | R-api-001 / R-embed-002/003 / R-rerank-002 | 21 | HTTP protocol matrix (fills the design.md M2 gap) |
| `test_rerank.py` | R-rerank-001 ~ R-rerank-006 | 6 | Rerank service (M1 9101 English + M2 9102 Chinese dual models) |
| `test_model_matrix.py` | R-model-001 (design.md M0) | 3 | Model matrix: 4 models × 3 vendors × 2 architectures over HTTP |
| `test_model_zh.py` | R-model-002 ~ R-model-007 | 6 | Model management (zh profile = M2 server-side HF mirror download, 512 dims; hot switch back goes through the local path) |
| `test_api_enhancements.py` | AE-* (api-config-enhancements) | 10 | API/config enhancements (semantic search, model unload, similarity metric, etc.) |
| `test_fault_tolerance.py` | FT-* (fault-tolerance-contract, port 9165) | 12 | Fault tolerance and API logic contracts: batch cap 64 / dedup scatter / concurrent batching consistency / rerank strict contract with partial-failure tolerance / search contract / metrics format / dimension self-consistency / Matryoshka prefix / IETF rate-limit headers and 429 |
| `test_semantic_cache.py` | SC-* (semantic-cache-e2e, ports 9166-9167) | 5 | Semantic cache: exact hit / trigram semantic hit (black-box signal = bitwise-equal cached vectors) / control group (vectors for the same text must differ without a cache server) / invalid comparison_mode rejects startup |
| `test_transcribe.py` | R-transcribe domain TR-01 ~ TR-04 | 4 | Transcription endpoint (`POST /api/v1/transcribe`): real-device ASR online contract, empty sample 400, non-finite sample 400, 48s full-length sample validates that `body_limit_mb` really applies; SKIPs when the `transcribe` feature or transcription assets are missing |

In total: 16 suites and 123 scenario cases (plus 27 pytest cases in `tests/perf`, next section). The pytest config sets `filterwarnings = ["error"]` (root pyproject.toml) — a **zero-warnings gate**: any runtime warning fails the run. In CI this layer is a nightly scheduled job (`scenario-tests.yml`, UTC 03:00) that does not block PRs; locally, `scripts/run-scenario-tests.sh` runs it in one command.

---

## 🔩 Rust integration and specialized tests

| File | Shape | Coverage |
|------|------|----------|
| `tests/integration.rs` → `tests/integration/` | Rust integration | `api_test.rs` (API surface) and `detect_test.rs` / `translate_test.rs` (candle-feature gated); RealTestEngine + TestMode mock/light/full were merged into `tests/common/mod.rs` |
| `tests/grpc_e2e.rs` | Specialized E2E | Full gRPC method matrix: spawns the real binary (`http,grpc` features); the tonic client goes through the `vecboost::sdforge` re-export |
| `tests/scenario_sdk.rs` | SDK matrix | Engine/model matrix in Library SDK (non-HTTP) mode, auto-SKIP based on `models/<dir>` existence |
| `tests/doctor.rs` | Specialized | `vecboost doctor` read-only diagnostics (config/tokenizer/cache persistence/threads/GPU/model integrity, exit code 1 on FAIL) |
| `tests/quantized_parity.rs` | Quality gate | GGUF quantization alignment: Q8_0 vs fp32 median cosine ≥ 0.98, Q4_K ≥ 0.95 (model via `VECBOOST_GGUF_MODEL`, corpus `tests/fixtures/golden_corpus.txt` ≥32 mixed Chinese/English lines) |
| `tests/model_snapshot_regression.rs` | Regression | Model snapshot regression (vector output compared against snapshots) |
| `tests/common/mod.rs` | Shared fixtures | `MockEngine` (FNV-1a + LCG deterministic vectors), `create_test_engine()` |

> Known deviation (D8): `tests/integration.rs` / `tests/perf.rs` coexist with same-named subdirectories — the entry files declare the modules explicitly via `#[path = "..."] mod ...;`, `cargo test --tests` builds with zero module-system warnings, and the current shape is kept (the "merge" originally planned for v0.2 is no longer necessary).

---

## ⚡ Performance and benchmark tests

### The Python part of tests/perf

`tests/perf/` contains two kinds of pytest cases, **distinguished by marker**:

- `@pytest.mark.sim`: hits the Python `api_simulator.py` semantic simulator for fast regression of API shapes/error paths and **does not represent Rust service behavior** (`test_api.py`, 23 cases);
- No marker / `@pytest.mark.real`: hits the real `localhost:9002` via `real_service.py` (`test_server_integration.py`, 4 cases).

```bash
pytest tests/perf -m "not sim"   # real-service cases only
pytest tests/perf -m sim          # simulator cases only
```

> When changing Rust API semantics, the `sim` cases must be updated in sync; if sim contradicts real behavior, the real service wins and sim gets fixed.

### Rust regression thresholds

`tests/perf/performance_test.rs` (8 cases) asserts relative throughput/latency thresholds with MockEngine — deterministic and CI-runnable.

### Microbenchmarks (benches/, criterion)

| Benchmark | What it measures |
|------|----------|
| `similarity_bench` | Single-vector similarity (cosine/euclidean/dot/manhattan × 128/384/768/1024 dims) |
| `batch_scheduling_bench` | Batch scheduling (steady/burst load × wait window, the `assemble_batch` time-window path) |
| `semantic_cache_bench` | Semantic cache (exact hit/miss, trigram search scaling) |
| `embed_throughput_bench` | Throughput baseline (single-text embed and 32-text embed_batch; model via `VECBOOST_BENCH_MODEL`, skipped when unset) |
| `rerank_bench` | Rerank throughput baseline (single-pair `rerank` and 1 query × 32 documents `rerank_batch`; model via `VECBOOST_BENCH_MODEL`, engine/device via `VECBOOST_BENCH_ENGINE`/`VECBOOST_BENCH_DEVICE`) |
| `translate_bench` | Translation throughput/latency baseline (seq2seq language pairs; baseline in `docs/benchmarks/translate_baseline_results.md`) |

Baseline numbers are recorded in `docs/benchmarks/` (annotated with environment and noise bands; that directory is **local-only**, excluded by `.gitignore`); the tuning-switch registry is in [⚡ Performance Guide](PERFORMANCE.md).

---

## 🧱 Unit tests

Inline `#[cfg(test)]` modules in each `src/` module (grep counts 2370 `#[test]` / `#[tokio::test]` attributes), run by `cargo test --lib`. Test code may `unwrap` (`clippy.toml` enables `allow-unwrap-in-tests`); production code may not. Shared environment-variable races are serialized via `ENV_LOCK` (secure-defaults test group).

---

## ⚡ Run commands (identical to CI)

```bash
# Unit + integration (CI: health-check.yml test job splits --lib and --tests)
cargo test --features "grpc,cli,auth,onnx,db,sqlite,openapi,mcp" --lib
cargo test --features "grpc,cli,auth,onnx,db,sqlite,openapi,mcp" --tests

# Local gates (scope with -p to workspace members to avoid pulling in other members' intermediate states; the 7 libraries come from the crates.io registry)
cargo test -p vecboost -p vecboost-examples

# gRPC E2E
cargo test -p vecboost --features http,grpc --test grpc_e2e

# Scenario tests (CI: scenario-tests.yml nightly)
# Note: the suites cover auth/CLI/MCP/gRPC capabilities — the binary must carry all protocol features;
# building with only --features http makes the auth suite fail as a group (login 404) and CLI/MCP skip.
cargo build -p vecboost --features http,auth,cli,mcp,grpc
pytest tests/scenario -q --junitxml=scenario-results.xml

# Performance regression thresholds (Rust)
cargo test -p vecboost --test perf

# Microbenchmarks
cargo bench -p vecboost

# Coverage (CI hard gate: line coverage >= 90%, llvm-cov; the denominator includes lib + integration tests + the real-boot leg; narrow --lib runs are forbidden)
cargo llvm-cov --features "grpc,cli,auth,onnx,db,sqlite,openapi,mcp,distributed-redis" --tests --lcov --output-path coverage/lcov.info
```

How CI workflows map to tests:

| Workflow | Trigger | Test content |
|--------|------|----------|
| `ci.yml` (CI) | push/PR (main) | fmt, clippy (incl. the unwrap_used gate), check in four tiers (empty/grpc/auth,db,sqlite,onnx/postgres), test in four tiers (default/grpc/ecosystem/shared-state), llvm-cov coverage ≥90% hard gate (lib + integration tests share the same denominator; narrow `--lib` runs forbidden; 90% is the minimum floor authorized by the owner on 2026-10-21 — the original 95% proved unreachable), `cargo deny` dependency audit |
| `health-check.yml` (Health Check) | push/PR (main) | Build matrix (default/onnx/grpc), `--lib` + `--tests`, llvm-cov coverage ≥90% hard gate (since 2026-10-15, replacing tarpaulin whose expression was incompatible with the new stable toolchain; aligned with ci.yml at the 90% floor since 2026-10-21), Windows tests (continue-on-error), `cargo doc` dead links, `cargo audit`, benchmark on main-branch push, dependency update check |
| `feature-matrix.yml` | push/PR | default/grpc/ecosystem (auth,db,sqlite,onnx)/postgres-driver in four tiers: fmt + check + clippy + build + `--lib` tests (the postgres-driver leg skips `--lib`) |
| `scenario-tests.yml` | nightly UTC 03:00 + manual | `cargo build -p vecboost --features http,auth,cli,mcp,grpc` (the suites cover auth/CLI/MCP/gRPC), then `pytest tests/scenario` (inference cases auto-SKIP without models) |
| `codeql.yml` / `docker.yml` | push/PR/weekly | CodeQL static analysis; Docker build + Trivy scan (fails on CRITICAL) |
| `security.yml` (Security) | push/PR (main) + Mondays 03:17 + manual | actionlint (YAML syntax/expressions hard gate), zizmor (GitHub Actions security audit, first-round non-blocking baseline), gitleaks incremental scan (push/PR commit range) + weekly full history scan (`--log-opts=--all`) |

---

## 🔀 Test modes and fixtures

The `TEST_MODE` environment variable controls test-engine behavior:

| Mode | Behavior |
|------|------|
| `mock` (default) | `MockEngine` (FNV-1a hash + LCG) deterministic vectors, no models needed |
| `light` | Try real inference, fall back to mock on failure |
| `full` | Force the real inference engine (models must be downloaded) |

> Tests that assert semantic similarity (e.g. "similar text > different text") are annotated `#[ignore]` (mock vectors are semantically random); run them under `TEST_MODE=light` or `full` via `cargo test --test integration -- --ignored`.

---

## 🧭 Model matrix

`tests/scenario/test_model_matrix.py` and `tests/scenario_sdk.rs` auto-SKIP based on `models/<dir-name>` existence. When all four local models (BGE-en 384 / MiniLM 384 / BGE-zh 512 / multilingual-e5-small 384) are present, coverage spans both XlmRoberta + Bert architectures and three pooling modes.

---

## 📌 Conventions

1. **New performance assertions go to the regression-threshold layer first** (MockEngine, deterministic); microbenchmarks are only for before/after comparisons of optimization work;
2. **New endpoint behavior contracts go to the scenario layer** (real server + HTTP) — this is also the living spec of the API documentation;
3. **Baseline numbers are recorded to `docs/benchmarks/`**, annotated with environment and noise bands;
4. Test code may `unwrap` (exempted in `clippy.toml`); production code may not.

---

## 📚 Related Documentation

| Document | Description |
|:-----|:-----|
| [📖 User Guide](USER_GUIDE.md) | Complete installation, configuration, and usage instructions |
| [🏗️ Architecture](ARCHITECTURE.md) | Module layout and test architecture |
| [⚡ Performance Guide](PERFORMANCE.md) | Benchmark data and tuning switches |
| [🤝 Contributing](CONTRIBUTING.md) | Quality gates and commit conventions |
| [📋 Changelog](CHANGELOG.md) | Change history per version |
