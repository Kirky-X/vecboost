<div align="center">

<img src="docs/image/vecboost.png" alt="VecBoost Logo" width="200"/>

[![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0) [![GitHub release](https://img.shields.io/github/v/release/Kirky-X/vecboost?style=for-the-badge)](https://github.com/Kirky-X/vecboost/releases) [![Rustc 1.91+](https://img.shields.io/badge/Rustc-1.91+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/)

[中文](README.md) | **English**

**A high-performance, production-grade embedding vector service written in Rust. VecBoost provides efficient text vectorization with multiple inference engines, GPU acceleration, and enterprise-grade features.**

[✨ Features](#features) • [🚀 Quick Start](#quick-start) • [📚 Documentation](#documentation) • [💻 Examples](#examples) • [🤝 Contributing](#contributing)

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

- [✨ Features](#features)
- [🚀 Quick Start](#quick-start)
- [🔌 API Usage](#api-usage)
- [⚙️ Configuration](#configuration)
- [📚 Documentation](#documentation)
- [💻 Examples](#examples)
- [🏗️ Architecture](#architecture)
- [🧪 Testing](#testing)
- [📊 Performance](#performance)
- [🔒 Security](#security)
- [🗺️ Roadmap](#roadmap)
- [🤝 Contributing](#contributing)
- [📋 Changelog](#changelog)
- [📄 License](#license)
- [🙏 Acknowledgements](#acknowledgements)
- [📞 Contact & Support](#contact--support)
- [⭐ Star History](#star-history)

---

## ✨ Features

<table style="width:100%; border-collapse: collapse">
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🚀 <b>High Performance</b><br><span style="color:#64748B">Optimized Rust codebase with batching and concurrent request processing; jemalloc global allocator enabled by default on Linux</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🔧 <b>Multiple Engines</b><br><span style="color:#64748B"><code>Candle</code> (native Rust) and <code>ONNX Runtime</code> inference engines, switchable via <code>EngineFactory</code></span></td>
</tr>
<tr>
<td width="50%" style="vertical-align:top; padding: 12px">🔁 <b>Rerank</b><br><span style="color:#64748B">Bi-encoder based document reranking over HTTP/gRPC/CLI</span></td>
<td width="50%" style="vertical-align:top; padding: 12px">🌍 <b>Internationalization (i18n)</b><br><span style="color:#64748B">ICU+Fluent bilingual (English/Chinese) error responses with <code>Accept-Language</code> per-request negotiation</span></td>
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
<td width="50%" style="vertical-align:top; padding: 12px">📦 <b>Cloud-Native Deployment</b><br><span style="color:#64748B">Multi-arch Docker images (linux/amd64 + arm64); Kubernetes deployment guidance (manifests bring-your-own)</span></td>
</tr>
</table>

In addition to the core capabilities above, an OpenAI-compatible endpoint (`POST /v1/embeddings`, with `encoding_format=base64` support), BF16 inference and SIMD vector similarity, GPU memory paging, read-only `vecboost doctor` diagnostics, Library SDK integration (library mode), and the `config_full.toml` / `config_minimal.toml` config presets are also available; see [🔌 API Usage](#api-usage) for endpoint details and [⚙️ Configuration](#configuration) for configuration options.

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

> **🐳 Docker**: `docker build -t vecboost:latest .`, then run with `config/` and `models/` mounted; Docker Compose and Kubernetes deployment are covered in the [📖 User Guide · Docker deployment](docs/USER_GUIDE.md#docker-部署).

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
cargo run --features cli -- embed --text "Hello, world!"
```

### 🧭 Core Concepts

- **Models & engines**: `ModelConfig` declares a HuggingFace model (default `BAAI/bge-small-en-v1.5`); `EngineFactory::create(engine_type, config)` creates a `Candle` (default) or `ONNX` (`onnx` feature) engine; Bert / XlmRoberta architectures and mean/cls/max pooling are supported.
- **Four protocols, one source**: handlers in `src/api/embedding.rs` are annotated with `#[forge(...)]` macros; `sdforge` generates the HTTP/gRPC/MCP/CLI bindings. Hand-written protocol code is forbidden.
- **7-library ecosystem**: `trait-kit` wires all modules through a typestate module registry (`Kit<Unbuilt> → Kit<Ready>`); `confers` owns configuration, `inklog` logging, `oxcache` caching, `limiteron` rate limiting, `dbnexus` persistence (`db` feature), and `sdforge` interface generation.
- **Configuration precedence**: TOML file + `VECBOOST_`-prefixed environment variable overrides (secrets such as `VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD` must be provided via env vars); config file changes are validated and logged, taking effect after restart.
- **Feature gating**: every optional capability is an independent feature (see [🏷️ Feature Flags](#feature-flags)); the minimal build contains only the HTTP server.

---

## 🔌 API Usage

VecBoost generates its four protocol interfaces from the single source `src/api/embedding.rs` via `sdforge`. All endpoints, parameters, request/response examples, the gRPC method table, and message types are documented in the [📘 API Reference](docs/API_REFERENCE.md); summary:

- **HTTP/REST**: `/api/1/*` provides embedding (single / batch / file), similarity, semantic search, rerank, model management, and health endpoints;
- **OpenAI-compatible**: `POST /v1/embeddings`, responding in the OpenAI format (`object` / `data` / `usage`) with `encoding_format=base64` support;
- **Matryoshka dimensionality reduction**: pass `dimensions` (256/512/1024, etc.) to `/v1/embeddings` for smaller, faster vectors; truncated vectors are automatically L2 re-normalized to keep cosine similarity correct;
- **gRPC**: with the `grpc` feature, 13 `vecboost.*` methods are exposed on port 50051 (configurable) over the sdforge unified Call protocol — no hand-written proto — with JWT auth, rate limiting, max connections, and timeouts all configurable;
- **MCP**: the `mcp` feature exposes the `embed` / `embed_batch` / `similarity` / `list_models` tools to LLMs over stdio (`vecboost --mcp`);
- **CLI**: the `cli` feature provides the embed / embed_batch / compute_similarity / search / rerank subcommands (see [💡 Minimal Example](#minimal-example));
- **Inference engines**: Candle (native Rust, default) and ONNX Runtime (`onnx` feature), switched via the `EngineFactory::create` factory;
- **Observability & operations**: `/metrics` (Prometheus), `/health` (liveness) and `/health?depth=full` (real readiness probe), `/api-docs` (Swagger UI); read-only diagnostics with `vecboost doctor` (config / tokenizer / cache / threads / GPU / model integrity; exit code 1 on FAIL).

Interactive OpenAPI docs: `http://localhost:9002/api-docs` (Swagger UI) and `/api-docs/openapi.json` (spec JSON, requires the `openapi` feature; ReDoc is deferred to v0.3.0). Stage-level metrics (batching / dedup / per-stage latency) are in the [⚡ Performance Guide · New metrics](docs/PERFORMANCE.md#新增指标).

### 🏷️ Feature Flags

The table below maps one-to-one to the `[features]` section of `Cargo.toml`; `default = ["http"]`.

| Feature | Default | Description |
|---------|------|------|
| `http` | ✅ | HTTP/REST API + OpenAPI docs + Prometheus metrics |
| `grpc` | - | gRPC server (generated by sdforge `#[forge(grpc_method)]`) |
| `cli` | - | CLI tool |
| `mcp` | - | MCP protocol interface (LLM tool integration, stdio mode) |
| `openapi` | - | OpenAPI/Swagger UI docs (independent of `http`) |
| `schema` | - | OpenAPI schema derive (auto-enabled by `http`/`openapi`; for library-mode type export) |
| `db` | - | dbnexus database persistence (SQLite) |
| `postgres` | - | PostgreSQL support (implies `db`) |
| `auth` | - | JWT auth + CSRF + RBAC + AES-256-GCM encryption |
| `cuda` | - | NVIDIA CUDA GPU acceleration |
| `metal` | - | Apple Silicon Metal GPU |
| `onnx` | - | ONNX Runtime engine |
| `mkl` | - | x86_64 CPU MKL acceleration backend (opt-in; requires a toolchain that links cleanly) |
| `accelerate` | - | aarch64 macOS Accelerate backend (opt-in) |
| `quantized-gguf` | - | GGUF quantization engine switch (inference backend pending upstream candle support) |

> **📦 Built-in dependencies**: `confers` (config), `inklog` (logging), `oxcache` (cache), `limiteron` (rate limiting), and `trait-kit` (module registry) are mandatory dependencies that are always enabled — no feature flag needed. `sdforge` is enabled by any protocol feature (`http`/`grpc`/`cli`/`mcp`).

---

## ⚙️ Configuration

The default config path is `config/config.toml` (`--config <path>` selects another; a missing explicit path fails fast with exit code 2; the repo ships `config_full.toml` / `config_minimal.toml` presets). Environment variables with the `VECBOOST_` prefix override the config file, and secrets (`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`) must be provided via env vars; config file changes are validated and logged, taking effect after restart.

Per-key options and defaults for every config section (server / model / embedding / rerank / monitoring / auth / rate_limit / audit / database / logging / pipeline.worker / semantic_cache / device), the full environment-variable table, and a complete example config are documented in the [📖 User Guide · Configuration](docs/USER_GUIDE.md#配置); you can also read [`config/config.toml`](config/config.toml) directly.

> **⚠️ Note**: the `[flow_control]` and `[cache]` TOML sections are not parsed in the current version (legacy section names); rate limiting uses `[rate_limit]`, caching uses `[embedding]` and `[semantic_cache]`. See the [❓ FAQ](docs/FAQ.md#配置与部署).

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
| [📈 Benchmark archive](docs/benchmarks/) | Historical benchmark data (similarity / batch scheduling / semantic cache / GPU pipeline) |
| [🌍 I18N missing-key audit](docs/I18N_MISSING_AUDIT.md) | Internationalization audit record |

---

## 💻 Examples

All examples live in [`examples/`](examples/), an independent workspace-member crate `vecboost-examples` (13 categories, 30 runnable binaries) covering basic embedding, HTTP/CLI calls, engine switching, auth, caching, rate limiting, monitoring, audit, semantic cache, and Library SDK integration; the per-category listing is in [`examples/README.md`](examples/README.md).

```bash
# Run a single example
cargo run -p vecboost-examples --bin embed
cargo run -p vecboost-examples --bin library_usage
cargo run -p vecboost-examples --bin matryoshka

# ONNX engine example (requires ONNX Runtime)
cargo run -p vecboost-examples --bin onnx --features onnx
```

---

## 🏗️ Architecture

VecBoost uses a modular ecosystem architecture: `trait-kit` wires 17 modules through a typestate module registry (`Kit<Unbuilt> → Kit<Ready>`), `sdforge` generates the four protocol bindings from the single source `src/api/embedding.rs`, inference is abstracted behind `EngineFactory` to the Candle / ONNX engines, and requests flow through a priority queue and time-window batching into the inference pipeline.

The 7-library ecosystem (trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge) versions and responsibilities, the module dependency graph, data flow, and the cache / security / deployment architecture and extension points are documented in the [🏗️ Architecture document](docs/ARCHITECTURE.md).

---

## 🧪 Testing

### 🎯 Testing Strategy

The test stack has six layers: inline unit tests in `src/`, integration tests in `tests/integration/`, specialized integration tests (doctor / gRPC E2E / model snapshot regression / quantization quality gate / SDK matrix), real-service scenario tests in `tests/scenario/*.py` (15 pytest suites), performance regression thresholds in `tests/perf/`, and 4 Criterion microbenchmark suites in `benches/`. The `TEST_MODE` environment variable controls the test engine (`mock` default / `light` / `full`). Layer responsibilities, the exhaustive scenario matrix, and the CI workflow mapping are in [🧪 Test Scenarios](docs/TEST_SCENARIOS.md).

### ▶️ Commands (identical to CI)

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

# Coverage (CI hard gate: line coverage >= 80%, tarpaulin)
cargo tarpaulin --features "grpc,cli,auth,onnx,db,openapi,mcp" --all-targets --out lcov --out xml --output-dir coverage/

# Benchmarks (CI benchmark job)
cargo bench --features "grpc,cli,auth,onnx,db,openapi,mcp"

# Documentation build and broken-link check
cargo doc --workspace --no-deps

# Dependency security audit
cargo audit
```

### 📊 Test Scale

As of the v0.2.1 workspace: ~1700+ inline unit tests in `src/`, 61 Rust integration/specialized tests (`tests/*.rs`), 126 Python scenario/performance cases (15 scenario suites), and 4 Criterion microbenchmark suites; the CI hard gate is line coverage no lower than 80% (tarpaulin), and Python scenario tests run as a nightly scheduled job (UTC 03:00) that does not block PRs. Per-item statistics and the scenario matrix are in [🧪 Test Scenarios](docs/TEST_SCENARIOS.md).

---

## 📊 Performance

Benchmark data comes from the measured `docs/benchmarks/` archive (criterion, collected 2026-08 on Linux x86_64, noise roughly ±5-10%): SIMD vector similarity is up to **3.06x** faster than the scalar baseline (~341.7 ns for 1024-dim cosine), `ContinuousBatchLoop` continuous batch scheduling is **5.6x** faster than fixed waiting (1.110 s vs 6.194 s for a 100-request steady load), and exact semantic-cache hits take ~10 ns; the throughput baseline (`embed_throughput_bench`) requires a local model and is **to be measured**. The full benchmark tables, performance design highlights (time-window batching / in-batch dedup / SIMD / thread tuning / jemalloc), GGUF quantization, and the tuning-switch registry are in the [⚡ Performance Guide](docs/PERFORMANCE.md); microbenchmarks can be reproduced with `cargo bench` (see the [Testing](#testing) section above).

---

## 🔒 Security

### 🛡️ Security Design

VecBoost is secure by default: the factory config binds to loopback only, and non-loopback binding with `auth.enabled=false` refuses to start (the `VECBOOST_ALLOW_INSECURE=1` escape hatch logs an ERROR); authentication and authorization build on garrison (JWT + CSRF + RBAC admin role + TOTP + account lockout), plus XFF trust inversion, file-path allowed roots, input length limits, AES-256-GCM config encryption, audit logging, and i18n bilingual error masking. Mechanism-level details are in the [🔒 Security document](docs/SECURITY.md).

### ⛓️ Supply Chain & Gates

`cargo audit`, `cargo deny check`, CodeQL, Trivy/Checkov image scanning, gitleaks secret scanning, and pre-commit hooks run both in CI and locally; the full list and triage policy are in the [🔒 Security document · Supply chain & gates](docs/SECURITY.md#供应链与安全门禁).

### 🚨 Reporting a Vulnerability

Please do not report security vulnerabilities through public issues; contact the maintainer at <kirky-x@outlook.com>. Dependency advisories are gated by CI `cargo-audit`. For the full policy and supported versions, see [SECURITY.md](docs/SECURITY.md).

---

## 🗺️ Roadmap

<table style="width:100%; border-collapse: collapse">
<tr><th style="text-align:center">Status</th><th style="text-align:left">Area</th><th style="text-align:left">Items</th></tr>
<tr><td align="center">✅</td><td>Core service</td><td>Four-protocol single-source generation, Candle/ONNX engines, Bert/XlmRoberta architectures, priority queue and time-window batching</td></tr>
<tr><td align="center">✅</td><td>Ecosystem integration</td><td>7-library ecosystem wiring (trait-kit registry, confers config, inklog logging, oxcache cache, limiteron rate limiting, dbnexus persistence, sdforge interfaces)</td></tr>
<tr><td align="center">✅</td><td>Security & i18n</td><td>JWT/CSRF/RBAC/TOTP, secure defaults hardening, audit logging, ICU+Fluent bilingual errors</td></tr>
<tr><td align="center">✅</td><td>Performance foundation</td><td>SIMD similarity, continuous batching, semantic cache, GPU memory paging, BF16 inference, Matryoshka reduction</td></tr>
<tr><td align="center">🚧</td><td>Audit remediation (Unreleased)</td><td>Breaking behavioral changes: secure defaults / login convergence / XFF trust inversion / RBAC wiring / model-scoped cache keys (see [Changelog](#changelog))</td></tr>
<tr><td align="center">🚧</td><td>Tuning switches (Unreleased)</td><td>GGUF quantization, quantized vector comparison, multi-model LFRU residency, cache WAL, hardware-aware planning, doctor diagnostics, startup warmup</td></tr>
<tr><td align="center">📋</td><td>Quantized inference backend</td><td>GGUF inference backend awaits upstream quantized BERT in candle-transformers (routing/magic-number checks/quality gate scaffolding ready)</td></tr>
<tr><td align="center">📋</td><td>Multi-replica session offload</td><td>Externalized auth sessions require the garrison db backend (pool-backed DAO)</td></tr>
<tr><td align="center">📋</td><td>Baseline completion</td><td><code>embed_throughput_bench</code> throughput baseline to be measured; MKL/Accelerate comparison baselines on toolchains that link cleanly</td></tr>
<tr><td align="center">📋</td><td>Docs & observability</td><td>ReDoc (v0.3.0), pre-configured Grafana dashboards</td></tr>
</table>

---

## 🤝 Contributing

For the detailed workflow and code standards, see the [🤝 Contributing guide](docs/CONTRIBUTING.md).

### 🛠️ Development Environment

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
| Unreleased | - | Audit remediation and tuning switches: secure defaults hardening, HF tokenizers on all platforms, time-window batching / in-batch dedup, GGUF quantization path, semantic-cache comparison modes, multi-model LFRU, cache WAL, doctor diagnostics, startup warmup |
| 0.2.1 | 2026-09-06 | i18n (114 translation keys), three-protocol rerank, three-stage semantic cache, BF16 precision, SIMD similarity, continuous batch scheduling, GPU memory paging, library mode |
| 0.2.0 | 2026-07-24 | sdforge four-protocol generation, 7-library ecosystem wiring, Matryoshka truncation re-normalization, vuln-0009 repo_id validation |
| 0.1.0 | 2025-12-15 | Initial VecBoost release |

Unreleased contains multiple breaking behavioral changes (secure defaults hardening, login convergence, XFF trust inversion, RBAC wiring, cache-key/tokenizer changes, etc.) — **read before upgrading**: the per-item "old behavior → new behavior → migration" table is in the [📋 Changelog · Unreleased](docs/CHANGELOG.md#unreleased). The multi-replica boundary (auth sessions live in process memory; single replica only) and hot-reload semantics (config changes take effect after restart) are covered in the [❓ FAQ](docs/FAQ.md#配置与部署).

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details. Copyright © 2025-2026 Kirky.X🌠.

---

## 🙏 Acknowledgements

### 🌟 Core Dependencies

VecBoost stands on the shoulders of these excellent open-source projects:

| Dependency | Purpose |
|------|------|
| [candle](https://github.com/huggingface/candle) | Native Rust ML inference framework (default engine) |
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
