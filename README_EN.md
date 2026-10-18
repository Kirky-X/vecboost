<div align="center">

<img src="docs/assets/vecboost.png" alt="VecBoost Logo" width="200"/>

[![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0) [![GitHub release](https://img.shields.io/github/v/release/Kirky-X/vecboost?style=for-the-badge)](https://github.com/Kirky-X/vecboost/releases) [![Rustc 1.91+](https://img.shields.io/badge/Rustc-1.91+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/)

[Chinese](README.md) | **English**

**A high-performance, production-grade multi-task inference service written in Rust. VecBoost provides six inference capabilities — text embedding, reranking, decision answering, translation, language detection, and speech transcription — with multiple inference engines, GPU acceleration, and enterprise-grade features.**

[✨ Features](#-features) • [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [💻 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

---

<div align="center">

### 🎯 Write the API Once, Get Four Protocols

Interface handlers are written just once; `sdforge` macros generate all four protocol bindings at compile time — the compiler does the rest.

<table style="width:100%; border-collapse: collapse">
<tr>
<td align="center" width="25%">🌐<br><b>REST</b><br><span style="color:#64748B">Web clients · On by default</span></td>
<td align="center" width="25%">📡<br><b>gRPC</b><br><span style="color:#64748B">Microservices · Type-safe</span></td>
<td align="center" width="25%">🤖<br><b>MCP</b><br><span style="color:#64748B">LLM tools · stdio transport</span></td>
<td align="center" width="25%">💻<br><b>CLI</b><br><span style="color:#64748B">Scripting · Quick checks</span></td>
</tr>
</table>

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🔌 API Usage](#-api-usage)
- [⚙️ Configuration](#-configuration)
- [📚 Documentation](#-documentation)
- [💻 Examples](#-examples)
- [🏗️ Architecture](#-architecture)
- [🧪 Testing](#-testing)
- [📊 Performance](#-performance)
- [🔒 Security](#-security)
- [🗺️ Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📋 Changelog](#-changelog)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [📞 Contact & Support](#-contact--support)
- [⭐ Star History](#-star-history)

---

## ✨ Features

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🚀 <b>High Performance</b><br><span style="color:#64748B">Optimized Rust codebase with batching and concurrent request processing; jemalloc global allocator enabled by default on Linux</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🔧 <b>Multiple Engines</b><br><span style="color:#64748B"><code>ONNX Runtime</code> (default) and <code>Candle</code> (opt-in fallback) inference engines, switchable via <code>EngineFactory</code></span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔁 <b>Rerank</b><br><span style="color:#64748B">Bi-encoder based document reranking over HTTP/gRPC/CLI</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌍 <b>Internationalization (i18n)</b><br><span style="color:#64748B">ICU+Fluent bilingual (English/Chinese) error responses with <code>Accept-Language</code> per-request negotiation</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🌐 <b>Translation & Language Detection</b><br><span style="color:#64748B">Marian/opus-mt multi-pair translation ((src,tgt) routing + auto detection) and whatlang language detection, <code>onnx</code> feature</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🎮 <b>Decision Answering</b><br><span style="color:#64748B">Laya decision pipeline: choice / score / noul with temperature-calibrated probabilities</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🎮 <b>GPU Acceleration</b><br><span style="color:#64748B">Native NVIDIA CUDA and Apple Metal support; opt-in <code>mkl</code>/<code>accelerate</code> CPU backends</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌐 <b>Multi-Protocol Interface</b><br><span style="color:#64748B">HTTP/REST, gRPC, MCP, and CLI generated from a single source by <code>sdforge</code></span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🧩 <b>7-Library Ecosystem</b><br><span style="color:#64748B">Modular ecosystem: <code>trait-kit</code>/<code>confers</code>/<code>inklog</code>/<code>oxcache</code>/<code>limiteron</code>/<code>dbnexus</code>/<code>sdforge</code></span></td>
<td width="50%" style="vertical-align:top; padding: 12px">📊 <b>Smart Caching</b><br><span style="color:#64748B">High-performance caching via <code>oxcache</code> (LRU/LFU/FIFO + TTL) and three-stage semantic cache lookups</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔐 <b>Enterprise Security</b><br><span style="color:#64748B">JWT authentication, CSRF protection, role-based access control, TOTP, account lockout, and audit logging</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">⚡ <b>Rate Limiting</b><br><span style="color:#64748B">Token-bucket rate limiting via <code>limiteron</code> (global/IP/user/API-key dimensions)</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">📈 <b>Priority Queue</b><br><span style="color:#64748B">Configurable request priorities, weighted fair scheduling, and time-window dynamic batching</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🧊 <b>Matryoshka Support</b><br><span style="color:#64748B">Dynamic dimensionality reduction (auto re-normalization after truncation) for smaller, faster embeddings (OpenAI-compatible)</span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔍 <b>Observability</b><br><span style="color:#64748B">Prometheus metrics, health checks, structured logging (inklog console + file rotation)</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">📦 <b>Cloud-Native Deployment</b><br><span style="color:#64748B">Docker image (linux/amd64, single-arch by default build); Kubernetes deployment guidance (manifests bring-your-own)</span></td>
</tr>
</table>

In addition to the core capabilities above, an OpenAI-compatible endpoint (`POST /v1/embeddings`, with `encoding_format=base64` support), BF16 inference and SIMD vector similarity, GPU memory paging, read-only `vecboost doctor` diagnostics, Library SDK integration (library mode), and the `config_full.toml` / `config_minimal.toml` config presets are also available; see [🔌 API Usage](#-api-usage) for endpoint details and [⚙️ Configuration](#-configuration) for configuration options.

---

## 🚀 Quick Start

### 📦 Installation

Prerequisites:

| Dependency | Version | Notes |
|--------|------|------|
| **Rust** | 1.91+ | edition 2024 (`rust-version` in `Cargo.toml` is authoritative) |
| **Cargo** | 1.91+ | ships with Rust |
| **CUDA Toolkit** | 12.x | optional, NVIDIA GPU support (`cuda` feature) |
| **Metal SDK** | latest | optional, Apple Silicon GPU support (`metal` feature) |
| **protobuf-compiler** | latest | optional, required for gRPC E2E tests |

> **💡 Tip**: run `rustc --version` to verify your Rust installation.

```bash
# 1. Clone the repository
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost

# 2. Default build (http feature, includes OpenAPI docs)
cargo build --release

# 3. Build with GPU support
#    Linux (CUDA):
cargo build --release --features cuda
#    macOS (Metal):
cargo build --release --features metal

# 4. Build multi-protocol interfaces (HTTP + gRPC + CLI)
cargo build --release --features grpc,cli

# 5. Build the MCP interface (stdio mode, start with --mcp)
cargo build --release --features mcp

# 6. Build the CI full-feature combination (database + auth + ONNX + OpenAPI + all protocols)
cargo build --release --features grpc,cli,auth,onnx,db,openapi,mcp
```

Minimal build: `cargo build --no-default-features --features http`.

Configure and run:

```bash
# Copy and customize the config (defaults to config/config.toml)
cp config/config.toml config/config_custom.toml
# Edit config/config_custom.toml

# Run with the default config
./target/release/vecboost

# Run with a custom config (--config; in CLI subcommand mode it must precede the subcommand)
./target/release/vecboost --config config/config_custom.toml
```

> **✅ Success**: the service starts at `http://127.0.0.1:9002` by default (secure default: loopback only).

> **🐳 Docker**: `docker build -t vecboost:latest .`, then run with `config/` and `models/` mounted; Docker Compose and Kubernetes deployment are covered in the [📖 User Guide · Docker deployment](docs/USER_GUIDE.md#-docker-部署).

### 💡 Minimal Example

The following example is adapted from [`examples/http/embed_api.rs`](examples/http/embed_api.rs) and generates embeddings over HTTP (full endpoints in the [📘 API Reference](docs/API_REFERENCE.md)):

```bash
curl -X POST http://localhost:9002/api/1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

Response:

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

You can also use the CLI (`cli` feature) or the library SDK (library mode):

```bash
# Single-text embedding
cargo run --features cli -- embed --req '{"text":"Hello, world!"}'
```

### 🧭 Core Concepts

- **Models & engines**: `ModelConfig` declares a HuggingFace model (code default `BAAI/bge-m3`, 1024 dims; the shipped `config/config.toml` explicitly sets `BAAI/bge-small-en-v1.5`); `EngineFactory::create(engine_type, config)` creates an `ONNX` (default, `onnx` feature enabled via default features) or `Candle` (`candle` feature opt-in fallback) engine; Bert / XlmRoberta architectures and mean/cls/max pooling are supported.
- **Four protocols, one source**: handlers in `src/api/embedding.rs` are annotated with `#[forge(...)]` macros; `sdforge` generates the HTTP/gRPC/MCP/CLI bindings. Hand-written protocol code is forbidden.
- **7-library ecosystem**: `trait-kit` wires all modules through a typestate module registry (`Kit<Unbuilt> → Kit<Ready>`); `confers` owns configuration, `inklog` logging, `oxcache` caching, `limiteron` rate limiting, `dbnexus` persistence (`db` feature), and `sdforge` interface generation.
- **Configuration precedence**: TOML file + `VECBOOST_`-prefixed environment variable overrides (secrets such as `VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD` must be provided via env vars); config file changes are validated and logged, taking effect after restart.
- **Feature gating**: every optional capability is an independent feature (see [🏷️ Feature Flags](#-feature-flags)); the minimal build contains only the HTTP server.

---

## 🔌 API Usage

VecBoost generates its four protocol interfaces from single-source handlers via `sdforge` (one source per domain under `src/api/`: embedding / rerank / decisions / translate / detect). All endpoints, parameters, request/response examples, the gRPC method table, and message types are documented in the [📘 API Reference](docs/API_REFERENCE.md); summary:

- **HTTP/REST**: `/api/1/*` provides embedding (single / batch / file), similarity, semantic search, rerank, decision answering (choice/score/noul), translation (multi-pair + auto detection), transcription (16kHz speech samples to text, `transcribe` feature), language detection, model management, and health endpoints;
- **OpenAI-compatible**: `POST /v1/embeddings`, responding in the OpenAI format (`object` / `data` / `usage`) with `encoding_format=base64` support;
- **Matryoshka dimensionality reduction**: pass `dimensions` (256/512/1024, etc.) to `/v1/embeddings` for smaller, faster vectors; truncated vectors are automatically L2 re-normalized to keep cosine similarity correct;
- **gRPC**: with the `grpc` feature, 17 `vecboost.*` methods are exposed on port 50051 (configurable) over the sdforge unified Call protocol — no hand-written proto — including `vecboost.translate` / `vecboost.detect`, with JWT auth, rate limiting, max connections, and timeouts all configurable;
- **MCP**: the `mcp` feature exposes tools registered by `#[forge(tool_name=…)]` to LLMs over stdio (`vecboost --mcp`) — tool names are `embed_text` / `embed_batch` / `compute_similarity` / `search` / `file_embed` / `rerank` / `rerank_batch` / `decisions` / `translate` / `detect` / `transcribe` / `list_models`, etc. (MCP `tool_name` is not the HTTP endpoint name; the full list is in the [📘 API Reference](docs/API_REFERENCE.md));
- **CLI**: the `cli` feature provides the embed / embed_batch / compute_similarity / search / rerank / decisions / translate / detect / transcribe subcommands (see [💡 Minimal Example](#-minimal-example); the `transcribe` subcommand takes a WAV file path and reads 16kHz mono samples);
- **Inference engines**: ONNX Runtime (default, `onnx` feature enabled via default features) and Candle (`candle` feature opt-in fallback), switched via the `EngineFactory::create` factory; the `[model]` `engine_type` / `task` configuration drives engine and task-dimension selection — `task = "embedding"` (default) / `task = "decision"` (Laya decision pipeline) / `task = "translate"` (seq2seq translation, onnx only) / `task = "transcribe"` (Qwen3-ASR transcription, `transcribe` feature); endpoints of non-current tasks respond with 400 Unsupported task (bundle layouts in [`config/config.toml`](config/config.toml) `[model]` comments, translation pairs in the `[translation]` section);
- **Observability & operations**: `/metrics` (Prometheus), `/health` (liveness) and `/health?depth=full` (real readiness probe), `/swagger-ui/` (interactive OpenAPI docs, requires the `swagger-ui` feature; spec JSON at `/api-docs/openapi.json`); read-only diagnostics with `vecboost doctor` (config / tokenizer / cache / threads / GPU / model integrity; exit code 1 on FAIL).

Translation and language detection request examples (fields match the [API Reference](docs/API_REFERENCE.md)):

```bash
curl -X POST http://localhost:9002/api/1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, my dog is cute", "source_language": "en", "target_language": "zh"}'

curl -X POST http://localhost:9002/api/1/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog near the riverbank every morning."}'
```

Detection response (`language: null` + `is_reliable: false` means "undetectable" — digits-only / too-short text; explicitly reported, never a guessed language; full error semantics in the [API Reference](docs/API_REFERENCE.md#翻译translate)):

```json
{
  "language": "en",
  "script": "Latin",
  "confidence": 1.0,
  "is_reliable": true,
  "processing_time_ms": 0
}
```

> **⚠️ Prerequisites**: translation requires an `--features onnx` build and `[[translation.pairs]]` language pairs configured in the `[translation]` section (unconfigured requests get 400 `translate-not-configured`); language detection has no model dependency and works with any build.

> **🔧 Output post-processing & timeout**: translation applies deterministic post-processing at the engine exit (shared by the service layer and library users) — Traditional→Simplified normalization for `target=zh`, repair of URL protocol separators and LaTeX escape symbols; model-level mistranslations are fixed by the literal replacement table `[[translation.pairs.output_fixes]]` (in declaration order), see the `[translation]` section of [`config/config.toml`](config/config.toml). The HTTP global request timeout is configured via `[server] request_timeout_seconds` (CPU seq2seq translation of paper-length abstracts takes 15-130+s; `config/config.toml` sets it to 300s explicitly).

Interactive OpenAPI docs (Swagger UI): build with `cargo build --features swagger-ui` and open `http://localhost:9002/swagger-ui/`; spec JSON: `/api-docs/openapi.json` (single binary, one build; the root path `/api-docs` serves no UI; ReDoc is deferred to v0.3.0). Stage-level metrics (batching / dedup / per-stage latency) are in the [⚡ Performance Guide · New metrics](docs/PERFORMANCE.md#-新增指标).

### 🔖 Feature Flags

The table below maps one-to-one to the `[features]` section of `Cargo.toml`; `default = ["http"]`.

| Feature | Default | Description |
|---------|------|------|
| `http` | ✅ | HTTP/REST API + OpenAPI docs + Prometheus metrics |
| `grpc` | - | gRPC server (generated by sdforge `#[forge(grpc_method)]`) |
| `cli` | - | CLI tool |
| `mcp` | - | MCP protocol interface (LLM tool integration, stdio mode) |
| `openapi` | - | OpenAPI spec programmatic export (sdforge/openapi, without the swagger-ui chain) |
| `swagger-ui` | - | Swagger UI interactive page + `/api-docs/openapi.json` (served on the main service port; not compiled in default builds; single binary, one build) |
| `schema` | - | OpenAPI schema derive (auto-enabled by `http`/`openapi`; for library-mode type export) |
| `db` | - | dbnexus database persistence infrastructure (driver-agnostic; requires picking one driver feature `sqlite` or `postgres`) |
| `sqlite` | - | SQLite database driver (implies `db` + `dbnexus/sqlite` + `inklog/sqlite`) |
| `postgres` | - | PostgreSQL database driver (implies `db` + `dbnexus/postgres` + `inklog/postgres`) |
| `auth` | - | JWT auth + CSRF + RBAC + AES-256-GCM encryption |
| `cuda` | - | CUDA device support (cudarc driver + nvml detection, backend-agnostic; candle's CUDA compute backend goes through `candle-cuda`) |
| `candle-cuda` | - | CUDA compute acceleration for the candle backend (implies `candle`) |
| `metal` | - | Apple Silicon Metal GPU (candle backend, implies `candle`) |
| `onnx` | - | ONNX Runtime engine (**enabled by default**, part of default features) |
| `candle` | - | candle fallback backend (opt-in: CandleEngine embedding / candle decision head / GGUF quantization bridge; `engine_type="candle"` is explicitly rejected at startup when not enabled) |
| `transcribe` | - | Qwen3-ASR transcription engine (three-graph + silero VAD + fbank), library-level `InferenceEngine::transcribe` entry; HTTP/gRPC endpoints wired (`POST /api/1/transcribe`, configured under `[transcribe]`) |
| `mkl` | - | x86_64 CPU MKL acceleration backend (candle-path opt-in, implies `candle`; requires a toolchain that links cleanly) |
| `accelerate` | - | aarch64 macOS Accelerate backend (candle-path opt-in, implies `candle`) |
| `quantized-gguf` | - | GGUF quantization engine switch (inference backend pending upstream candle support) |
| `character_converter` | - | Enable the `character_converter` character-conversion dependency standalone (already implied by `onnx`) |

> **📦 Built-in dependencies**: `confers` (config), `inklog` (logging), `oxcache` (cache), `limiteron` (rate limiting), and `trait-kit` (module registry) are mandatory dependencies that are always enabled — no feature flag needed. `sdforge` is enabled by any protocol feature (`http`/`grpc`/`cli`/`mcp`).

> **🧪 Experiment path**: The local transcription experiment harness (standalone crate, binary `transcribe-own`) has been converged into a thin-shell CLI — inference goes through the main-crate facade `vecboost::transcribe`, with the py-torch baseline path retained; **this directory is local-only and not tracked in git**. The productized implementation lives in `src/engine/transcribe/` — see [🧹 Maintenance Norms](docs/MAINTENANCE.md) for the CLI contract summary and consumption norms.

---

## 🔧 Configuration

The default config path is `config/config.toml` (`--config <path>` selects another; a missing explicit path fails fast with exit code 2; the repo ships `config_full.toml` / `config_minimal.toml` presets). Environment variables with the `VECBOOST_` prefix override the config file, and secrets (`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`) must be provided via env vars; config file changes are validated and logged, taking effect after restart.

Per-key options and defaults for every config section (server / model / embedding / rerank / monitoring / auth / rate_limit / audit / database / logging / pipeline.worker / semantic_cache / device / translation / transcribe), the full environment-variable table, and a complete example config are documented in the [📖 User Guide · Configuration](docs/USER_GUIDE.md#-配置); you can also read [`config/config.toml`](config/config.toml) directly.

> **⚠️ Note**: the `[flow_control]` and `[cache]` TOML sections are not parsed in the current version (legacy section names); rate limiting uses `[rate_limit]`, caching uses `[embedding]` and `[semantic_cache]`. See the [❓ FAQ](docs/FAQ.md#-配置与部署).

---

## 📚 Documentation

| Document | Description |
|------|------|
| [📖 User Guide](docs/USER_GUIDE.md) | Complete tutorial from installation to advanced usage (incl. deployment) |
| [📘 API Reference](docs/API_REFERENCE.md) | Full REST / gRPC / OpenAI-compatible interface documentation |
| [🏗️ Architecture](docs/ARCHITECTURE.md) | Design principles, module layout, and data flow |
| [⚡ Performance Guide](docs/PERFORMANCE.md) | Benchmark data, tuning-switch registry, and experiment discipline |
| [🔒 Security](docs/SECURITY.md) | Security design, supported versions, and vulnerability reporting |
| [❓ FAQ](docs/FAQ.md) | Frequently asked questions |
| [🧪 Test Scenarios](docs/TEST_SCENARIOS.md) | Test-stack responsibilities and the scenario matrix |
| [📋 Changelog](docs/CHANGELOG.md) | Release-by-release change log |
| [🤝 Contributing](docs/CONTRIBUTING.md) | How to contribute |
| [🧹 Maintenance Norms](docs/MAINTENANCE.md) | Cross-change standing constraints: transcribe consumption surface, CLI contract, retired-switch discipline, deprecation-doc governance |
| [📈 Benchmark archive](docs/benchmarks/) | Historical benchmark data (similarity / batch scheduling / semantic cache / GPU pipeline) |
| [🌍 I18N missing-key audit](docs/I18N_MISSING_AUDIT.md) | Internationalization audit record |

---

## 💻 Examples

All examples live in [`examples/`](examples/), an independent workspace-member crate `vecboost-examples` (12 categories, 31 runnable binaries) covering basic embedding, HTTP/CLI calls, engine switching, auth, caching, rate limiting, monitoring, audit, semantic cache, and Library SDK integration; the per-category listing is in [`examples/README.md`](examples/README.md).

```bash
# Run a single example
cargo run -p vecboost-examples --bin embed
cargo run -p vecboost-examples --bin library_usage
cargo run -p vecboost-examples --bin matryoshka

# ONNX engine example (requires ONNX Runtime)
cargo run -p vecboost-examples --bin onnx --features onnx
```

---

## 🧱 Architecture

VecBoost uses a modular ecosystem architecture: `trait-kit` wires modules through a typestate module registry (`Kit<Unbuilt> → Kit<Ready>`); `src/registry/` has 20 `ModuleMeta` implementations (several are feature-gated, e.g. under `auth`), `sdforge` generates the four protocol bindings from the single source `src/api/embedding.rs`, inference is abstracted behind `EngineFactory` to the Candle / ONNX engines, and requests flow through a priority queue and time-window batching into the inference pipeline.

The 7-library ecosystem (trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge) versions and responsibilities, the module dependency graph, data flow, and the cache / security / deployment architecture and extension points are documented in the [🏗️ Architecture document](docs/ARCHITECTURE.md).

---

## 🧪 Testing

### 🎯 Testing Strategy

The test stack has six layers: inline unit tests in `src/`, integration tests in `tests/integration/`, specialized integration tests (doctor / gRPC E2E / model snapshot regression / quantization quality gate / SDK matrix), real-service scenario tests in `tests/scenario/*.py` (15 pytest suites), performance regression thresholds in `tests/perf/`, and 5 Criterion microbenchmark suites in `benches/`. The `TEST_MODE` environment variable controls the test engine (`mock` default / `light` / `full`). Layer responsibilities, the exhaustive scenario matrix, and the CI workflow mapping are in [🧪 Test Scenarios](docs/TEST_SCENARIOS.md).

### ⚡ Commands (identical to CI)

The commands below are extracted from `.github/workflows/health-check.yml` (CI), `feature-matrix.yml`, `scenario-tests.yml`, and `docs/CONTRIBUTING.md`. Path-dependency note: the `../base/*` ecosystem libraries are living path dependencies, so local gate commands must be scoped with `-p vecboost -p vecboost-examples`.

```bash
# Format and lint gates (CI treats clippy unwrap_used as the production panic-surface gate)
cargo fmt --all -- --check
cargo clippy --features "grpc,cli,auth,onnx,db,openapi,mcp" --all-targets -- -D warnings -W clippy::unwrap_used

# Full-feature compile check (feature-matrix)
cargo check --features "grpc,cli,auth,onnx,db,openapi,mcp"

# Unit + integration tests (CI runs --lib and --tests separately)
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --lib
cargo test --features "grpc,cli,auth,onnx,db,openapi,mcp" --tests

# gRPC E2E (spawns the real binary)
cargo test -p vecboost --features http,grpc --test grpc_e2e

# Scenario tests (pytest; conftest spawns real servers; inference cases auto-SKIP without models/)
cargo build -p vecboost --features http
pytest tests/scenario -q --junitxml=scenario-results.xml

# Python performance tests (sim marker separates simulator cases)
pytest tests/perf -m "not sim"   # real-service cases only
pytest tests/perf -m sim          # simulator cases only

# Coverage (CI hard gate: line coverage >= 80%, llvm-cov; measured 90.89% locally)
cargo llvm-cov --features "grpc,cli,auth,onnx,db,sqlite,openapi,mcp" --lib --lcov --output-path coverage/lcov.info

# Benchmarks (CI benchmark job)
cargo bench --features "grpc,cli,auth,onnx,db,openapi,mcp"

# Documentation build and broken-link check
cargo doc --workspace --no-deps

# Dependency security audit
cargo audit
```

### 📊 Test Scale

As of the v0.3.0-rc.2 workspace: 2277 inline unit tests in `src/`, 63 Rust integration/specialized test functions (`tests/*.rs`), 146 Python scenario/performance cases (119 scenario + 27 performance, 15 scenario suites), and 5 Criterion microbenchmark suites; the CI hard gate is line coverage no lower than 80% (llvm-cov), and Python scenario tests run as a nightly scheduled job (UTC 03:00) that does not block PRs. Per-item statistics and the scenario matrix are in [🧪 Test Scenarios](docs/TEST_SCENARIOS.md).

---

## 📊 Performance

Benchmark data comes from the measured `docs/benchmarks/` archive (criterion; that directory is **local-only** and git-ignored, so it does not exist in a clean clone): SIMD vector similarity is up to **3.06x** faster than the scalar baseline (~341.7 ns for 1024-dim cosine), exact semantic-cache hits take ~10 ns, and candle single-text embedding is **4.0x** faster with `--features mkl`. Full benchmark tables, performance design highlights (time-window batching / in-batch dedup / SIMD / thread tuning / jemalloc), GGUF quantization and the tuning-switch registry are in the [⚡ Performance Guide](docs/PERFORMANCE.md).

### 🧭 Engine selection: why onnxruntime is the primary engine (measured evidence)

> **Measurement setup**: 2026-10-18, AMD Ryzen 9 9950X (16 cores) / 66 GiB / RTX 5080 16 GiB (WSL2);
> ORT ① 1.28.2 (CPU) / ② 1.24.3 (CUDA EP); embeddings use bge-small-en-v1.5 (two exports of the same model),
> transcription uses Qwen3-ASR (0.6B / 1.7B, 700 s Chinese audio, CER against the same reference); each cell is 2–3 runs.

**Embedding throughput (single / batch×32, lower is better)**

| Engine × device | Single | Batch×32 | Verdict |
|---|---|---|---|
| **onnxruntime × CPU** | **2.56 ms** | **26.35 ms** | Product default, best on CPU |
| candle × CPU | 78.0 ms | 186.5 ms | **30.5× / 7.1×** slower |
| onnxruntime × CUDA | 2.70 ms | **3.06 ms** | Best for batches |
| candle × CUDA (`candle-cuda`) | 5.05 ms | 6.70 ms | 15.4×/27.8× faster than candle CPU, still **1.9×/2.2×** slower than ort GPU |

- **The direct reason for the switch**: candle's CPU forward pass is a native implementation that is **not optimized** — 30× slower single, 7× slower batched; even with `--features mkl` (4× on single) it only reaches ~17.5 ms, still **~6.8× slower than ort**.
- **GPU is not a substitute**: on the same card and model, ort's CUDA EP is still ~2× faster, with the same VRAM footprint (~1.95 GB). candle's GPU is nevertheless a 15–36× win over its own CPU — what lags is the engine implementation, not the device.
- **Embedding quantization**: `int8` vs fp32 gives **−22%** single / **−16%** batch / **−55%** memory, with median cosine **0.9971** (effectively lossless); `q4` is faster but cosine drops to 0.9849. Gate: `tests/engine_backend_parity.rs::onnx_quantized_vs_fp32_parity`.

**Transcription throughput (Qwen3-ASR, 700 s audio; 4-worker / service-serial)**

| Path | Time | CER | Resources |
|---|---|---|---|
| **ort 0.6B int8 × CPU** | **70.4 s / 137.2 s** | 7.75% | no VRAM |
| ort 0.6B int8 × CUDA | 199.4 s | 7.36% | 14.4 GiB VRAM (**GPU is a regression**) |
| embedded Python 0.6B bf16 × CUDA | 51.2 s | 22.81% (collapses on long audio) | 11.7 GiB VRAM + Python stack |
| ort 1.7B int8 × CPU | 135.7–142.1 s | 15.78–18.99% | no VRAM |
| **ort 1.7B fp32 (lossless) × CPU** | 698.8 s | **7.24%** | no VRAM |
| **embedded Python 1.7B bf16 × CUDA** | **55.9 s** | **2.49%** | 14.5 GiB VRAM + Python stack |

- **For ort, GPU is a regression across the board** (0.6B 2.6× slower, 1.7B int8 1.6× slower, 1.7B fp32 2.3× slower): SM utilization stays ~15% while VRAM fills up — per-segment decoding kernels are too small and are dominated by launch/sync overhead.
- **The quality bottleneck is quantization, not the engine**: mixed-bundle decomposition shows **dynamic int8 on the decoder** accounts for all 7.9 CER points, while quantizing the encoder costs almost nothing → the pure-Rust high-quality configuration is **int8 encoder + fp32 decoder (7.27% / 656 s / no VRAM)**.
- Embedded Python is faster and (at 1.7B) the most accurate, but it **skips VAD segmentation**, so long audio degrades to 22.8% CER; it also pulls a Python runtime into the process and needs 11.7–14.5 GiB VRAM.

> Full matrices (service-serial, HTTP end-to-end, mixed-quantization attribution, fp16 guard) live in the local
> `docs/benchmarks/transcribe_engine_baseline.md` and `docs/benchmarks/embed_rerank_translate_baseline.md`.

---

## 🔒 Security

### 🔒 Security Design

VecBoost is secure by default: the factory config binds to loopback only, and non-loopback binding with `auth.enabled=false` refuses to start (the `VECBOOST_ALLOW_INSECURE=1` escape hatch logs an ERROR); authentication and authorization build on garrison (JWT + CSRF + RBAC admin role + TOTP + account lockout), plus XFF trust inversion, file-path allowed roots, input length limits, AES-256-GCM config encryption, audit logging, and i18n bilingual error masking. Mechanism-level details are in the [🔒 Security document](docs/SECURITY.md).

### 🔗 Supply Chain & Gates

`cargo audit`, `cargo deny check`, CodeQL, Trivy/Checkov image scanning, gitleaks secret scanning, and pre-commit hooks run both in CI and locally; the full list and triage policy are in the [🔒 Security document · Supply chain & gates](docs/SECURITY.md#-供应链与安全门禁).

### 🚨 Reporting a Vulnerability

Please do not report security vulnerabilities through public issues; contact the maintainer at <kirky-x@outlook.com>. Dependency advisories are gated by CI `cargo-audit`. For the full policy and supported versions, see [SECURITY.md](docs/SECURITY.md).

---

## 🧭 Roadmap

<table style="width:100%; border-collapse: collapse">
<tr><th style="text-align:center">Status</th><th style="text-align:left">Area</th><th style="text-align:left">Items</th></tr>
<tr><td align="center">✅</td><td>Core service</td><td>Four-protocol single-source generation, Candle/ONNX engines, Bert/XlmRoberta architectures, priority queue and time-window batching</td></tr>
<tr><td align="center">✅</td><td>Ecosystem integration</td><td>7-library ecosystem wiring (trait-kit registry, confers config, inklog logging, oxcache cache, limiteron rate limiting, dbnexus persistence, sdforge interfaces)</td></tr>
<tr><td align="center">✅</td><td>Security & i18n</td><td>JWT/CSRF/RBAC/TOTP, secure defaults hardening, audit logging, ICU+Fluent bilingual errors</td></tr>
<tr><td align="center">✅</td><td>Performance foundation</td><td>SIMD similarity, continuous batching, semantic cache, GPU memory paging, BF16 inference, Matryoshka reduction</td></tr>
<tr><td align="center">✅</td><td>Audit remediation (0.3.0-rc.1)</td><td>Breaking behavioral changes: secure defaults / login convergence / XFF trust inversion / RBAC wiring / model-scoped cache keys (see [Changelog](#-changelog))</td></tr>
<tr><td align="center">✅</td><td>Tuning switches (0.3.0-rc.1)</td><td>GGUF quantization, quantized vector comparison, multi-model LFRU residency, cache WAL, hardware-aware planning, doctor diagnostics, startup warmup</td></tr>
<tr><td align="center">✅</td><td>Quantized inference backend</td><td>GGUF engine delivered via a load-time dequantization bridge (Q8_0/Q4_K; routing, magic validation and quality gate in place); benefit is storage/load size, runtime is fp32-equivalent (no compute speedup); the quality gate needs model assets to run</td></tr>
<tr><td align="center">✅</td><td>Multi-replica session offload</td><td><code>[auth] session_backend = "database"</code> (deployment DB <code>garrison_kv</code>, garrison DbnexusKvDao); rate limiting / semantic cache remain per-process and lockout counters are best-effort across replicas</td></tr>
<tr><td align="center">✅</td><td>Performance baselines (throughput + MKL)</td><td>Throughput and MKL comparison baselines measured (2026-09-16, see <a href="docs/PERFORMANCE.md">performance guide</a>)</td></tr>
<tr><td align="center">📋</td><td>Performance baseline (Accelerate)</td><td>Accelerate (macOS) pending verification: no Apple hardware available; the verification protocol is documented in the performance guide</td></tr>
<tr><td align="center">✅</td><td>Docs & observability</td><td>ReDoc (<code>/redoc</code>, <code>swagger-ui</code> feature) and pre-configured Grafana dashboard (<code>deploy/grafana/</code>)</td></tr>
</table>

---

## 🤝 Contributing

For the detailed workflow and code standards, see the [🤝 Contributing guide](docs/CONTRIBUTING.md).

### 🔨 Development Environment

The toolchain is Rust 1.91+ (`rust-version` in `Cargo.toml` is authoritative) and Python ≥ 3.10 + pytest (optional: protobuf-compiler, docker); before submitting you must pass the four quality gates — fmt / clippy (the `unwrap_used` panic-surface gate) / tests / `scripts/doc_consistency_check.py` — and Git hooks run automatically via [pre-commit](https://pre-commit.com/) (`.pre-commit-config.yaml` → `scripts/pre-commit.sh`); commit messages follow Conventional Commits, and behavioral changes must be recorded in the CHANGELOG `Unreleased` section and mirrored in both READMEs. Environment setup, feature combinations, and the quality-gate commands are in the [🤝 Contributing guide](docs/CONTRIBUTING.md).

### 💖 Ways to Contribute

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="33%" align="center" style="padding: 16px">

### 🐛 Report a Bug

Found a problem?<br>
<a href="https://github.com/Kirky-X/vecboost/issues/new">Open an issue</a>

</td>
<td width="33%" align="center" style="padding: 16px">

### 💡 Suggest a Feature

Have an idea?<br>
<a href="https://github.com/Kirky-X/vecboost/issues/new">Start a discussion</a>

</td>
<td width="33%" align="center" style="padding: 16px">

### 🔧 Submit a PR

Want to contribute code?<br>
<a href="https://github.com/Kirky-X/vecboost/pulls">Fork and open a PR</a>

</td>
</tr>
</table>

---

## 📋 Changelog

The full release history is in the [📋 Changelog](docs/CHANGELOG.md) (following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), semantic versioning).

| Version | Date | Highlights |
|------|------|------|
| Unreleased | - | onnx-primary engine switch (ort default, candle opt-in), audit-log DB/file dual write, in-flight dedup for identical requests, body-limit fix, swagger-ui/ReDoc docs surface, externalized database session backend, Grafana dashboard |
| 0.3.0-rc.2 | 2026-10-15 | Transcribe (Qwen3-ASR) service surface, translation/language-detection across all protocols, decision-task priority queue, self-maintained dependency upgrades |
| 0.3.0-rc.1 | 2026-09-22 | Audit remediation and tuning switches (secure defaults / login convergence / XFF / RBAC / cache keys, GGUF/LFRU/WAL/hardware planning/doctor/warmup), PARQ storage replacing oxcache, unified "generate" interface |
| 0.2.1 | 2026-07-30 | i18n, three-protocol rerank, three-stage semantic cache, BF16 precision, SIMD similarity, continuous batch scheduling, GPU memory paging, library mode |
| 0.2.0 | 2026-07-24 | sdforge four-protocol generation, 7-library ecosystem wiring, Matryoshka truncation re-normalization, vuln-0009 repo_id validation |
| 0.1.2 | 2026-01-16 | Patch fixes |
| 0.1.1 | 2026-01-15 | Patch fixes |
| 0.1.0 | 2026-01-10 | Initial VecBoost release |

0.3.0-rc.1 contains multiple breaking behavioral changes (secure defaults hardening, login convergence, XFF trust inversion, RBAC wiring, cache-key/tokenizer changes, etc.) — **read before upgrading**: the per-item "old behavior → new behavior → migration" table is in the [📋 Changelog · 0.3.0-rc.1](docs/CHANGELOG.md#030-rc1---2026-09-22). The multi-replica boundary (auth sessions default to process memory and can be externalized with `session_backend = "database"`; rate limiting and caches remain per-process) and hot-reload semantics (config changes take effect after restart) are covered in the [❓ FAQ](docs/FAQ.md#-配置与部署).

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details. Copyright © 2025-2026 Kirky.X🌠.

---

## 🙏 Acknowledgements

### 🌟 Core Dependencies

VecBoost stands on the shoulders of these excellent open-source projects:

| Dependency | Purpose |
|------|------|
| [onnxruntime](https://github.com/microsoft/onnxruntime) | Cross-platform inference runtime (default engine, via the [`ort`](https://crates.io/crates/ort) binding with `load-dynamic` loading the `3rdparty/` shared library) |
| [candle](https://github.com/huggingface/candle) | Native Rust ML inference framework (opt-in fallback backend, `candle` feature) |
| [tokenizers](https://github.com/huggingface/tokenizers) | HuggingFace tokenizer (all platforms) |
| [hf-hub](https://crates.io/crates/hf-hub) | HuggingFace Hub model downloads |
| [trait-kit](https://crates.io/crates/trait-kit) | Module registry & typestate dependency management |
| [confers](https://crates.io/crates/confers) | Configuration loading (TOML + env + validation) |
| [inklog](https://crates.io/crates/inklog) | Structured logging infrastructure |
| [oxcache](https://crates.io/crates/oxcache) | High-performance cache backend |
| [limiteron](https://crates.io/crates/limiteron) | Token-bucket rate limiter |
| [dbnexus](https://crates.io/crates/dbnexus) | Database persistence (`db` feature) |
| [sdforge](https://crates.io/crates/sdforge) | Multi-protocol interface generation |
| [garrison](https://crates.io/crates/garrison) | Authentication & security hardening (`auth` feature) |
| [axum](https://github.com/tokio-rs/axum) | HTTP framework (generated by sdforge) |
| [tokio](https://github.com/tokio-rs/tokio) | Async runtime |
| [utoipa](https://github.com/juhaku/utoipa) | OpenAPI documentation |
| [prometheus](https://github.com/tikv/rust-prometheus) | Metrics export |
| [criterion](https://github.com/bheisler/criterion.rs) | Benchmarking |
| [tikv-jemallocator](https://github.com/tikv/jemallocator) | jemalloc global allocator (Linux glibc) |

### 💝 Special Thanks

Thanks to the Rust community, Hugging Face (model & tokenizer ecosystem), and all [contributors](https://github.com/Kirky-X/vecboost/graphs/contributors).

---

## 📞 Contact & Support

<table style="width:100%; max-width: 600px">
<tr>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost/issues"><b style="color:#991B1B">Issues</b></a><br>
<span style="color:#64748B">Report problems and bugs</span>
</td>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost/issues"><b style="color:#1E40AF">Discussions</b></a><br>
<span style="color:#64748B">Ask questions and share ideas</span>
</td>
<td align="center" width="33%">
<a href="https://github.com/Kirky-X/vecboost"><b style="color:#1E293B">GitHub</b></a><br>
<span style="color:#64748B">Browse the source</span>
</td>
</tr>
</table>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kirky-X/vecboost&type=Date)](https://star-history.com/#Kirky-X/vecboost&Date)

If this project helps you, please consider giving it a ⭐️!

**Built by Kirky.X**

---

<sub>© 2026 Kirky.X. All rights reserved.</sub>
