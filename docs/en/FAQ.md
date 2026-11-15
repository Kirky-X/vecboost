# ❓ VecBoost FAQ

This page collects frequently asked questions about VecBoost, grouped by topic. No answer here? Ask on [GitHub Issues](https://github.com/Kirky-X/vecboost/issues).

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents (click to expand)</summary>

- [General](#-general)
- [Installation and Building](#-installation-and-building)
- [Models and Inference](#-models-and-inference)
- [Configuration and Deployment](#-configuration-and-deployment)
- [Authentication and Security](#-authentication-and-security)
- [API Usage](#-api-usage)
- [Testing and Performance](#-testing-and-performance)

</details>

---

## 🧭 General

### ❓ What is VecBoost?

VecBoost is a **high-performance embedding service** built in Rust: it turns text into embedding vectors and serves them over four protocols — HTTP/REST, gRPC, MCP, and CLI — with dual inference engines (Candle, native Rust, and ONNX Runtime), CUDA/Metal GPU acceleration, caching, rate limiting, authentication, and other enterprise-grade capabilities. It is a modular ecosystem of 7 Rust libraries (trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge) assembled through the `trait-kit` module registry.

### ❓ Is VecBoost production-ready?

Yes. The project ships with: secure defaults (loopback binding; non-loopback binding without auth refuses to start), JWT/CSRF/RBAC authentication and authorization, audit logging, Prometheus metrics and health checks, graceful shutdown, and a Docker image (linux/amd64; currently a single-architecture build, no arm64 yet). The current release line is 0.3.1; read the breaking-change list in the [CHANGELOG](CHANGELOG.md) before upgrading (13 behavioral changes since 0.3.0-rc.1, including hardened secure defaults).

### ❓ Which platforms are supported?

Linux (x86_64/aarch64; CI covers Ubuntu plus Windows tests), macOS (Apple Silicon via the `metal` feature). CUDA 12.x is optional (the `cuda` feature); the jemalloc allocator is enabled by default on Linux glibc.

### ❓ Where can I get help?

- Usage questions: [GitHub Issues](https://github.com/Kirky-X/vecboost/issues)
- Documentation: [📖 User Guide](USER_GUIDE.md), [📘 API Reference](API_REFERENCE.md), [❓ this page](FAQ.md)
- Security vulnerabilities: **never via public issues** — prefer GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy), or email <kirky-x@outlook.com> as a fallback (recipient Kirky.X) (see [🔒 Security](SECURITY.md))

### ❓ What license does the project use?

Apache-2.0. See [LICENSE](../../LICENSE) for details.

---

## 📦 Installation and Building

### ❓ How do I build and install?

```bash
git clone https://github.com/Kirky-X/vecboost.git
cd vecboost
cargo build --release          # default features: http + onnx
./target/release/vecboost      # the service defaults to http://127.0.0.1:9002
```

Requires Rust 1.91+ (the `rust-version` in `Cargo.toml` is authoritative). For more combinations see the [User Guide](USER_GUIDE.md).

### ❓ How do I pick a feature combination?

| Need | Build |
|------|------|
| HTTP service only (default) | `cargo build --release` |
| GPU (NVIDIA / Apple) | `--features cuda` / `--features metal` |
| gRPC / CLI / MCP | `--features grpc` / `--features cli` / `--features mcp` |
| Database persistence | `--features sqlite` (SQLite driver) / `--features postgres` (a driver must be picked explicitly; `db` is infrastructure only) |
| Authentication | `--features auth` |
| CI full-feature set | `--features grpc,cli,auth,onnx,db,sqlite,openapi,mcp` (identical to the `FEATURES` in `.github/workflows/ci.yml:24`; `db` requires the `sqlite` or `postgres` driver) |

`confers`/`inklog`/`oxcache`/`limiteron`/`trait-kit` are mandatory dependencies that are always enabled — no feature needed. The full feature-flag matrix is in [🔖 Feature Flags](USER_GUIDE.md#-feature-flags).

### ❓ The build fails with link errors (hgemm_)?

The MKL acceleration backend (`--features mkl`) depends on intel-mkl-src 2020.1 static libraries, which lack the `hgemm_` half-precision symbol (referenced by candle 0.11 `mkl.rs:166`); some toolchains fail to link. This is a known opt-in limitation: enable `mkl`/`accelerate` only on toolchains/CI matrices that link cleanly (`cargo check` does not link; both combinations pass it). See the deviation log in the [⚡ Performance Guide](PERFORMANCE.md).

### ❓ Why is there no `redis` feature?

`redis` itself is not a directly enableable feature — it is an implementation detail of **`distributed-redis`**. `Cargo.toml` defines `distributed-redis = ["limiteron/cache-redis", "limiteron/distributed", "limiteron/lua-script", "limiteron/ban-sync", "oxcache/redis", "oxcache/lua", "dep:redis"]` (`Cargo.toml:185`) to move rate-limit counters and cache/semantic-cache trigram indexes to Redis for multi-replica setups; the default build has zero Redis dependencies and the cache backend is in-process oxcache. On the code side this corresponds to `CacheBackendKind::Redis` (`src/config/app.rs`), startup validation (`startup-semantic-redis-feature-missing` / `-url-missing` at `src/config/app.rs:624-640`), and the assembly branches (`src/main.rs:455-480`, `src/main.rs:611-618`) — without the feature the configuration is rejected loudly, never silently degraded. Pure single-replica deployments need no Redis feature at all. If old documents/scripts reference a `redis` feature, defer to the `cargo build --features ...` error and `Cargo.toml [features]`.

---

## 🤖 Models and Inference

### ❓ Which models are supported?

The code default is `BAAI/bge-m3` (1024 dims); the shipped `config/config.toml` explicitly sets `BAAI/bge-small-en-v1.5` (384 dims), so an out-of-the-box run (with the shipped config) actually uses 384. Any HuggingFace Bert / XlmRoberta embedding model is supported (BGE family, MiniLM, multilingual-e5, etc.), with mean/cls/max pooling and `expected_dimension` set per model (BGE-M3 is 1024). A 4-model matrix (BGE-en 384 / MiniLM 384 / BGE-zh 512 / multilingual-e5-small 384) is covered by scenario tests.

### ❓ Where are models stored? Do I need internet access?

Models are downloaded via the HuggingFace Hub (`hf-hub`) and cached locally by default; `model_path` may also point to a local directory (preferred when set). On restricted networks, set `HF_ENDPOINT` to a mirror endpoint. The first start downloads automatically; `examples/download_model.rs` can pre-download.

### ❓ How do I switch models / keep multiple models resident?

- Runtime switch: `POST /api/v1/model/switch` (admin role) or gRPC `vecboost.model_switch`;
- Multi-model residency: `[model] max_resident_models` / `resident_memory_budget_mb` (LFRU eviction + heat persistence in `data/model_heat.json`);
- Available models: `GET /api/v1/models`.

### ❓ Do GGUF quantized models work yet?

Partially. Routing, magic validation, and the quality gate for `[model] quantized = true` + `--features quantized-gguf` (Q8_0 median cosine ≥0.98 / Q4_K ≥0.95, `tests/quantized_parity.rs`) are in place, but **candle-transformers 0.11 has no quantized BERT model yet — the inference backend awaits upstream support**. See the [⚡ Performance Guide](PERFORMANCE.md).

### ❓ Why must I rebuild the vector index after upgrading?

Since 0.3.0-rc.1 the tokenizer is the cross-platform HuggingFace `tokenizers` (more correct), which differs from the old in-house WordPiece results; cache keys also changed from `text:{raw text}` to `emb:{model}:{xxh3_128}`. **Existing vectors are incompatible with newly built ones** — rebuild the index; the first-run cache misses after upgrading are a one-time effect.

### ❓ What is Matryoshka dimensionality reduction?

For models with Matryoshka representations you can request smaller dimensions (256/512/1024, etc.) in exchange for smaller storage and faster retrieval; the service L2 re-normalizes after truncation to keep cosine similarity correct. Use the `dimensions` parameter on `/v1/embeddings`.

---

## 🔧 Configuration and Deployment

### ❓ Where is the config file? Do changes require a restart?

The default is `config/config.toml` (`--config <path>` selects another). Presets: `config_full.toml` / `config_minimal.toml`. **Hot-reload semantics**: config changes are validated and logged, taking effect after restart (no runtime hot swap); on Kubernetes use ConfigMap rolling updates.

### ❓ Which settings must use environment variables?

Secrets must go through environment variables (prefix `VECBOOST_`): `VECBOOST_JWT_SECRET` (required when auth is enabled, ≥32 chars), `VECBOOST_ADMIN_PASSWORD` (required when auth is enabled, ≥12 chars), `VECBOOST_ENCRYPTION_KEY`, etc. The full list is in the [📖 User Guide · Environment variables](USER_GUIDE.md#-environment-variables) section.

### ❓ Why don't the `[flow_control]` / `[cache]` config sections take effect?

These two sections are legacy section names that **the current version does not parse**: rate limiting uses `[rate_limit]`, caching uses `[embedding]` and `[semantic_cache]` (persist-to-disk and comparison_mode live in those two). See the tuning-switch registry in the [⚡ Performance Guide](PERFORMANCE.md).

### ❓ The service fails to bind / how do I expose ports in production?

The shipped config sets `host = "127.0.0.1"` (loopback; the built-in default for `[server] host` is `"0.0.0.0"`); binding a non-loopback address with `auth.enabled=false` refuses to start. Trusted-network containers may set `VECBOOST_ALLOW_INSECURE=1` (logs an ERROR); production should enable auth, bind `0.0.0.0`, expose via an HTTPS reverse proxy, and configure `trusted_proxies` explicitly.

### ❓ How do I deploy with Docker / Kubernetes?

Docker: `docker build -t vecboost:latest .`, then run with `config/` and `models/` mounted (CI builds a single-architecture linux/amd64 image). Kubernetes: bring your own manifests and roll config updates via ConfigMap; `/health` is the liveness probe and `/health?depth=full` the real readiness probe. The container architecture and scaling strategy are in the deployment architecture section of the [🏗️ Architecture document](ARCHITECTURE.md).

### ❓ Can I run multiple replicas with auth enabled?

Yes, but sessions must be externalized explicitly. The inference path is stateless; with `auth.enabled=true`, sessions live in process memory by default (`session_backend = "memory"`, single replica only); multi-replica requires `session_backend = "database"` (deployment table `garrison_kv`) plus a build with `--features "db,sqlite"` or `"db,postgres"` and a configured `[database]`. Account lockout counters use `AtomicKvDao` atomic increments with no loss across replicas; for cross-replica rate limiting / exact caching additionally set `[rate_limit] backend = "redis"` and `[embedding] cache_backend = "redis"` (requires a `distributed-redis` build). Semantic-cache **fuzzy hits** use `[semantic_cache] backend = "redis"`; the **ban list** is shared via limiteron `ban-sync` as soon as `distributed-redis` is built — both no longer apply per replica.

---

## 🔐 Authentication and Security

### ❓ How do I enable authentication?

```toml
[auth]
enabled = true
```

```bash
export VECBOOST_JWT_SECRET="a secret of at least 32 characters"
export VECBOOST_ADMIN_PASSWORD="a password of at least 12 characters"
```

Once enabled, log in via the login endpoint to get a JWT and send `Authorization: Bearer <token>` on business requests. The full flow is in the authentication section of the [User Guide](USER_GUIDE.md) and the [API Reference](API_REFERENCE.md).

### ❓ Which endpoints require the admin role?

`/api/v1/model/*` (model management: switch/unload, etc.) and `/embed/file`. Business embedding calls work with a regular account.

### ❓ How long is a token valid?

Since 0.3.0-rc.1, `token_expiration_hours` defaults to **24 hours** (previously the garrison default was 30 days). Configure it explicitly for long-lived sessions.

### ❓ I found a security vulnerability — what now?

Do not open a public issue; prefer GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy) (fallback email <kirky-x@outlook.com>, recipient Kirky.X). The process and supported versions are in [🔒 Security](SECURITY.md).

---

## 🌐 API Usage

### ❓ Which protocols and entry points exist?

| Protocol | Feature | Entry point |
|------|---------|------|
| HTTP/REST | `http` (default) | `http://127.0.0.1:9002/api/v1/*` |
| OpenAI-compatible | `http` | `POST /v1/embeddings` (supports `dimensions` and base64) |
| gRPC | `grpc` | `:50051` (17 `vecboost.*` methods) |
| MCP | `mcp` | `vecboost --mcp` (stdio; tools `embed_text`/`embed_batch`/`compute_similarity`/`list_models` etc.) |
| CLI | `cli` | `vecboost embed --req "..."` etc. |

Interactive documentation (Swagger UI): `http://localhost:9002/swagger-ui/` (spec JSON: `/api-docs/openapi.json`; the root path `/api-docs` has no UI registered — direct access returns 404).

### ❓ Why don't the embeddings usage token counts match my estimates?

`/v1/embeddings` usage is a **real token count** (counted by the HF tokenizer since 0.3.0-rc.1) and may differ from character-based estimation tools — the tokenizer result is authoritative.

### ❓ `/embed/file` errors with "no allowed roots configured"?

Since 0.3.0-rc.1, `/embed/file` requires an explicitly configured `[server] grpc_allowed_roots` (no cwd fallback; requests return 400 when unconfigured), single file ≤10 MiB, `text_preview` admin-only. Configure allowed roots for your business directories.

---

## 🧪 Testing and Performance

### ❓ How do I run all tests locally?

```bash
cargo test -p vecboost -p vecboost-examples
pytest tests/scenario -q    # requires cargo build -p vecboost --features http first
cargo bench -p vecboost
```

Scenario tests auto-SKIP inference cases when `models/` is absent. The full command set (identical to CI) is in [🧪 Test Scenarios](TEST_SCENARIOS.md).

### ❓ Where is the performance data? Why doesn't the README quote specific QPS?

Benchmark data always comes from the measured `docs/benchmarks/` archive (SIMD similarity 3.06x, ~10ns semantic-cache hits, etc.); the throughput baseline (`embed_throughput_bench`) was measured with a local bge-small: default build 70.3 ms single-text / 173.6 ms for a 32-batch, 4.0× faster with `--features mkl` — fabricating benchmark numbers is forbidden. The 2026-09-16 regression sweep re-ran everything with no regressions; see the [regression sweep archive](../benchmarks/regression_sweep_2026-09-16.md); measurement methods and reproduction commands are in the [⚡ Performance Guide](PERFORMANCE.md). Note: the `ContinuousBatchLoop 5.6x` batch-scheduling numbers in the old archive correspond to a retired component and are not comparable with the current time-window batching.

### ❓ How do I diagnose the service state?

`vecboost doctor`: read-only checks of config / tokenizer / cache persistence / threads / GPU / model integrity (safetensors headers, gguf magic, hidden_size pairing); exit code 1 on any FAIL. Runtime metrics: `/metrics` (Prometheus), including per-stage `vecboost_stage_seconds{stage}` latency.

### ❓ How do I contribute?

Read the [🤝 Contributing guide](CONTRIBUTING.md): before submitting you must pass the four gates — fmt / clippy (unwrap_used gate) / tests / `scripts/doc_consistency_check.py`; behavioral changes must be registered in the CHANGELOG and mirrored in both README versions.
