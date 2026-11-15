<div align="center">

<img src="docs/assets/vecboost.png" alt="VecBoost Logo" width="240"/>

[![Rust 2024](https://img.shields.io/badge/Rust-2024-edded?logo=rust&style=for-the-badge)](https://www.rust-lang.org/) [![Rustc 1.91+](https://img.shields.io/badge/Rustc-1.91+-orange.svg?style=for-the-badge)](https://www.rust-lang.org/) [![release](https://img.shields.io/github/v/release/Kirky-X/vecboost?style=for-the-badge)](https://github.com/Kirky-X/vecboost/releases) [![CI](https://img.shields.io/github/actions/workflow/status/Kirky-X/vecboost/feature-matrix.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/Kirky-X/vecboost/actions/workflows/feature-matrix.yml) [![security](https://img.shields.io/github/actions/workflow/status/Kirky-X/vecboost/codeql.yml?branch=main&label=security&style=for-the-badge)](https://github.com/Kirky-X/vecboost/actions/workflows/codeql.yml) [![coverage](https://img.shields.io/codecov/c/github/Kirky-X/vecboost?style=for-the-badge)](https://codecov.io/gh/Kirky-X/vecboost) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

English | [中文](README_CN.md)

**A high-performance, production-grade multi-task inference service written in Rust. VecBoost provides six inference capabilities — text embedding, reranking, decision answering, translation, language detection, and speech transcription — with multiple inference engines, GPU acceleration, and enterprise-grade features.**

[✨ Features](#-features) • [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [💻 Examples](#-examples) • [🤝 Contributing](#-contributing)

</div>

---

<div align="center">

### 🎯 Six Inference Capabilities, One Service

Text embedding, reranking, decision answering, translation, language detection, and speech transcription all load models through a single `EngineFactory`, sharing configuration, caching, and rate-limiting infrastructure.

<table style="width:100%; border-collapse: collapse">
<tr>
<td align="center" width="33%">🧠<br><b>Text Embedding</b><br><span style="color:#64748B">Batch vectors · Matryoshka · OpenAI-compatible</span></td>
<td align="center" width="33%">🔁<br><b>Reranking</b><br><span style="color:#64748B">Bi-encoder document relevance</span></td>
<td align="center" width="33%">🎲<br><b>Decision Answering</b><br><span style="color:#64748B">choice / score / noul tasks</span></td>
</tr>
<tr>
<td align="center" width="33%">🌐<br><b>Translation</b><br><span style="color:#64748B">Marian language pairs · auto source detection</span></td>
<td align="center" width="33%">🔤<br><b>Language Detection</b><br><span style="color:#64748B">Confidence and reliability flags</span></td>
<td align="center" width="33%">🎙️<br><b>Speech Transcription</b><br><span style="color:#64748B">Qwen3-ASR with VAD segmentation</span></td>
</tr>
</table>

</div>

---

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents</summary>

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🔌 API Usage](#-api-usage)
- [🔧 Configuration](#-configuration)
- [📚 Documentation](#-documentation)
- [💻 Examples](#-examples)
- [🧱 Architecture](#-architecture)
- [🧪 Testing](#-testing)
- [📊 Performance](#-performance)
- [🔒 Security](#-security)
- [🧭 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📋 Changelog](#-changelog)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [📞 Contact & Support](#-contact--support)

</details>

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

In addition to the core capabilities above, an OpenAI-compatible endpoint (`POST /v1/embeddings`, with `encoding_format=base64` support), BF16 inference and SIMD vector similarity, GPU memory paging, read-only `vecboost doctor` diagnostics, Library SDK integration (library mode), and the `config_full.toml` / `config_minimal.toml` config presets are also available; the full feature-flag matrix is in [🔖 Feature Flags](docs/en/USER_GUIDE.md#-feature-flags), and endpoint details are in [🔌 API Usage](#-api-usage).

---

## 🚀 Quick Start

### 📦 Installation

Prerequisites:

<details>
<summary>📋 Table</summary>

| Dependency | Version | Notes |
|--------|------|------|
| **Rust** | 1.91+ | edition 2024 (`rust-version` in `Cargo.toml` is authoritative) |
| **Cargo** | 1.91+ | ships with Rust |
| **CUDA Toolkit** | 12.x | optional, NVIDIA GPU support (`cuda` feature) |
| **Metal SDK** | latest | optional, Apple Silicon GPU support (`metal` feature) |
| **protobuf-compiler** | latest | optional, required for gRPC E2E tests |

</details>

> **💡 Tip**: run `rustc --version` to verify your Rust installation.

<details>
<summary>📄 Code · bash</summary>

```bash
# 1. Clone the repository
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost

# 2. Default build (http + onnx features)
cargo build --release

# 3. Run (reads config/config.toml by default)
./target/release/vecboost
```

</details>

Minimal build: `cargo build --no-default-features --features http`. For more build combinations (GPU / multi-protocol / MCP / the CI full-feature set), see the [📖 User Guide · Installation](docs/en/USER_GUIDE.md#-installation).

Configure and run:

<details>
<summary>📄 Code · bash</summary>

```bash
# Copy and customize the config (defaults to config/config.toml)
cp config/config.toml config/config_custom.toml
# Edit config/config_custom.toml

# Run with a custom config (--config; it is stripped before clap parsing, so position before or after the subcommand both work)
./target/release/vecboost --config config/config_custom.toml
```

</details>

> **✅ Success**: the service starts at `http://127.0.0.1:9002` by default (secure default: loopback only).

> **🐳 Docker**: `docker build -t vecboost:latest .`, then run with `config/` and `models/` mounted; Docker Compose and Kubernetes deployment are covered in the [📖 User Guide · Docker deployment](docs/en/USER_GUIDE.md#-docker-deployment).

### 💡 Minimal Example

The following example is adapted from [`examples/http/embed_api.rs`](examples/http/embed_api.rs) and generates embeddings over HTTP (full endpoints in the [📘 API Reference](docs/en/API_REFERENCE.md)):

<details>
<summary>📄 Code · bash</summary>

```bash
curl -X POST http://localhost:9002/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

</details>

Response:

<details>
<summary>📄 Code · json</summary>

```json
{
  "embedding": [0.123, 0.456, 0.789, ...],
  "dimension": 1024,
  "processing_time_ms": 15.5
}
```

</details>

Request/response examples for the remaining endpoints — decision answering, translation, language detection, transcription — are in the [📘 API Reference](docs/en/API_REFERENCE.md); you can also use the CLI (`cli` feature) or the library SDK (library mode), with subcommands and tool lists in [💻 CLI tools](docs/en/API_REFERENCE.md#-cli-tools).

---

## 🔌 API Usage

VecBoost generates its four protocol interfaces (HTTP/gRPC/MCP/CLI) from single-source handlers via `sdforge` (one source per domain under `src/api/`); all endpoints, parameters, request/response examples, the gRPC method table, MCP tool lists, and error codes are documented in the [📘 API Reference](docs/en/API_REFERENCE.md), and the interactive OpenAPI docs in [📚 OpenAPI documentation](docs/en/API_REFERENCE.md#-openapi-documentation) (`/swagger-ui/` and `/redoc`, requiring the `swagger-ui` feature).

---

## 🔧 Configuration

The default config path is `config/config.toml` (`--config <path>` selects another; a missing explicit path fails fast with an error; the repo ships `config_full.toml` / `config_minimal.toml` presets). Environment variables with the `VECBOOST_` prefix override the config file, and secrets (`VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD`) must be provided via env vars; config file changes are validated and logged, taking effect after restart.

Per-key options and defaults for every config section (server / model / embedding / rerank / monitoring / auth / rate_limit / audit / database / logging / pipeline.worker / semantic_cache / device / translation / transcribe), the full environment-variable table, and a complete example config are documented in the [📖 User Guide · Configuration](docs/en/USER_GUIDE.md#-configuration); you can also read [`config/config.toml`](config/config.toml) directly.

> **⚠️ Note**: the `[flow_control]` and `[cache]` TOML sections are not parsed in the current version (legacy section names); rate limiting uses `[rate_limit]`, caching uses `[embedding]` and `[semantic_cache]`. See the [❓ FAQ](docs/en/FAQ.md#-configuration-and-deployment).

---

## 📚 Documentation

<details>
<summary>📋 Table</summary>

| Document | Description |
|------|------|
| [📖 User Guide](docs/en/USER_GUIDE.md) | Complete tutorial from installation to advanced usage (incl. deployment) |
| [📘 API Reference](docs/en/API_REFERENCE.md) | Full REST / gRPC / OpenAI-compatible interface documentation |
| [🏗️ Architecture](docs/en/ARCHITECTURE.md) | Design principles, module layout, and data flow |
| [⚡ Performance Guide](docs/en/PERFORMANCE.md) | Benchmark data, tuning-switch registry, and experiment discipline |
| [🔒 Security](docs/en/SECURITY.md) | Security design, supported versions, and vulnerability reporting |
| [❓ FAQ](docs/en/FAQ.md) | Frequently asked questions |
| [🧪 Test Scenarios](docs/en/TEST_SCENARIOS.md) | Test-stack responsibilities and the scenario matrix |
| [📋 Changelog](docs/en/CHANGELOG.md) | Release-by-release change log |
| [🤝 Contributing](docs/en/CONTRIBUTING.md) | How to contribute |
| [🧹 Maintenance Norms](docs/en/MAINTENANCE.md) | Cross-change standing constraints: transcribe consumption surface, CLI contract, retired-switch discipline, deprecation-doc governance |

</details>

> **ℹ️ Note**: `docs/benchmarks/` (benchmark archive), `docs/reviews/` (review ledgers) and `docs-transcribe/` are **local-only** directories (see `.gitignore`); they are not committed and do not exist in a clean clone. Key benchmark numbers are inlined in the [⚡ Performance Guide](docs/en/PERFORMANCE.md).

---

## 💻 Examples

All examples live in [`examples/`](examples/), an independent workspace-member crate `vecboost-examples` (12 categories, 31 runnable binaries) covering basic embedding, HTTP/CLI calls, engine switching, auth, caching, rate limiting, monitoring, audit, semantic cache, and Library SDK integration; the per-category listing is in [`examples/README.md`](examples/README.md).

---

## 🧱 Architecture

VecBoost uses a modular ecosystem architecture: `trait-kit` wires modules through a typestate module registry (`Kit<Unbuilt> → Kit<Ready>`); `src/registry/` has 20 production `ModuleMeta` implementations plus 3 test fixtures (several are feature-gated, e.g. under `auth`), `sdforge` generates the four protocol bindings from the single source `src/api/embedding.rs`, inference is abstracted behind `EngineFactory` to the Candle / ONNX engines, and requests flow through a priority queue and time-window batching into the inference pipeline.

The 7-library ecosystem (trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge) versions and responsibilities, the module dependency graph, data flow, and the cache / security / deployment architecture and extension points are documented in the [🏗️ Architecture document](docs/en/ARCHITECTURE.md).

---

## 🧪 Testing

The test stack has six layers: inline unit tests in `src/`, integration tests in `tests/integration/`, specialized integration tests (doctor / gRPC E2E / model snapshot regression / quantization quality gate / SDK matrix), real-service scenario tests in `tests/scenario/*.py` (16 pytest suites), performance regression thresholds in `tests/perf/`, and 6 Criterion microbenchmark suites in `benches/`. Layer responsibilities, the exhaustive scenario matrix, and the run commands (identical to CI) are in [🧪 Test Scenarios](docs/en/TEST_SCENARIOS.md); environment setup and quality gates are in [🤝 Contributing](docs/en/CONTRIBUTING.md).

---

## 📊 Performance

Benchmark data comes from the measured `docs/benchmarks/` archive (criterion; that directory is **local-only** and git-ignored, so it does not exist in a clean clone): SIMD vector similarity is up to **3.06x** faster than the scalar baseline (~341.7 ns for 1024-dim cosine), exact semantic-cache hits take ~10 ns, and candle single-text embedding is **4.0x** faster with `--features mkl`. Full benchmark tables, measured engine-selection evidence, performance design highlights (time-window batching / in-batch dedup / SIMD / thread tuning / jemalloc), GGUF quantization and the tuning-switch registry are in the [⚡ Performance Guide](docs/en/PERFORMANCE.md).

---

## 🔒 Security

### 🔒 Security Design

VecBoost is secure by default: the factory config binds to loopback only, and non-loopback binding with `auth.enabled=false` refuses to start (the `VECBOOST_ALLOW_INSECURE=1` escape hatch logs an ERROR); authentication and authorization build on garrison (JWT + CSRF + RBAC admin role + TOTP + account lockout), plus XFF trust inversion, file-path allowed roots, input length limits, AES-256-GCM config encryption, audit logging, and i18n bilingual error masking. Mechanism-level details are in the [🔒 Security document](docs/en/SECURITY.md).

### 🔗 Supply Chain & Gates

`cargo audit`, `cargo deny check`, CodeQL, Trivy/Checkov image scanning, gitleaks secret scanning, and pre-commit hooks run both in CI and locally; the full list and triage policy are in the [🔒 Security document · Supply chain & gates](docs/en/SECURITY.md#-supply-chain-and-security-gates).

### 🚨 Reporting a Vulnerability

Please do not report security vulnerabilities through public issues. Use the GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy) private disclosure channel as the primary route (fall back to Kirky.X at <kirky-x@outlook.com> only if that channel is unavailable). Dependency advisories are gated by CI `cargo-audit`. For the full policy and supported versions, see [SECURITY.md](docs/en/SECURITY.md).

---

## 🧭 Roadmap

Roadmap items (core service / ecosystem integration / security & i18n / performance foundation / tuning switches / multi-replica shared state, etc.; ✅ done / 📋 planned) are maintained with version evolution in the [📋 Changelog · Roadmap](docs/en/CHANGELOG.md#-roadmap).

---

## 🤝 Contributing

For the detailed workflow and code standards, see the [🤝 Contributing guide](docs/en/CONTRIBUTING.md). Found a problem? [Open an issue](https://github.com/Kirky-X/vecboost/issues/new); have an idea? [Start a discussion](https://github.com/Kirky-X/vecboost/issues/new); fork and open a [PR](https://github.com/Kirky-X/vecboost/pulls).

---

## 📋 Changelog

The full release history is in the [📋 Changelog](docs/en/CHANGELOG.md) (following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), semantic versioning).

0.3.0-rc.1 contains multiple breaking behavioral changes (secure defaults hardening, login convergence, XFF trust inversion, RBAC wiring, cache-key/tokenizer changes, etc.) — **read before upgrading**: the per-item "old behavior → new behavior → migration" table is in the [📋 Changelog · 0.3.0-rc.1](docs/en/CHANGELOG.md#030-rc1---2026-09-22). The multi-replica boundary (auth sessions default to process memory and can be externalized with `session_backend = "database"`; rate limiting and caches can now be externalized to Redis via `distributed-redis`) and hot-reload semantics (config changes take effect after restart) are covered in the [❓ FAQ](docs/en/FAQ.md#-configuration-and-deployment).

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details. Copyright © 2025-2026 Kirky.X🌠.

---

## 🙏 Acknowledgements

The core-dependency list (onnxruntime / candle / tokenizers / hf-hub / the 7-library ecosystem / garrison / axum / tokio / utoipa / prometheus / criterion / jemallocator) is in [🏗️ Architecture · Core dependency acknowledgements](docs/en/ARCHITECTURE.md#-core-dependency-acknowledgements).

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

<sub>© 2026 Kirky.X. All rights reserved.</sub>
