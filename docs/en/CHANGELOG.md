# 📋 VecBoost Changelog

This file records all notable changes to VecBoost.

The format is based on [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/lang/zh-CN/spec/v2.0.0.html).

## 📋 Table of Contents

- [0.3.1 - 2026-11-15](#031---2026-11-15)
- [0.3.0 - 2026-11-14](#030---2026-11-14)
- [0.3.0-rc.3 - 2026-10-20](#030-rc3---2026-10-20)
- [0.3.0-rc.2 - 2026-10-15](#030-rc2---2026-10-15)
- [0.3.0-rc.1 - 2026-09-22](#030-rc1---2026-09-22)
- [0.2.1 - 2026-07-30](#021---2026-07-30)
- [0.2.0 - 2026-07-24](#020---2026-07-24)
- [0.1.2 - 2026-01-16](#012---2026-01-16)
- [0.1.1 - 2026-01-15](#011---2026-01-15)
- [0.1.0 - 2026-01-10](#010---2026-01-10)

---

<a id="031---2026-11-15"></a>

## [0.3.1] - 2026-11-15

### Security

- **Removed the transitive dependency on bincode (RUSTSEC-2025-0141, unmaintained across all versions)**: `character_converter` 2.1.5 (unmaintained since 2023-11) pulled in bincode 1.3.3 with no upgrade path. Traditional→Simplified normalization now uses the in-repo `engine::t2s` module — the dictionary was migrated entry-by-entry from upstream `data/t2s.profile` (115,571 entries; source and MIT license noted in the `t2s_table.txt` header/footer), with longest-prefix-replacement semantics aligned to upstream `convert_script`; equivalence was verified on a golden sample of 264,176 samples (all dictionary keys + concatenated strings + mixed examples) with **0 differences**. The RUSTSEC-2025-0141 exemption in `.cargo/audit.toml` was removed in sync.

### Compatibility

- The `character_converter` feature remains as an empty shell (enabling it no longer pulls any dependency) to keep downstream `--features character_converter` invocations from breaking; the `onnx` feature no longer implies `dep:character_converter`. No public API changes (patch release).

<a id="030---2026-11-14"></a>

## [0.3.0] - 2026-11-14

### Changed
- **i18n engine retirement (refactor)**: the in-house FluentBundle engine was retired (net −242 lines), replaced by lightweight FTL parsing (`src/i18n/catalog.rs`) + a unified `sdforge::i18n` registration surface. **Ruling on record (R-07)**: this repo has no ICU plural/gender formatting needs, so i18n stays a pure text-translation surface — the trait-kit `i18n` formatting layer is not enabled (`I18nFormatter` has zero call sites in this repo; "formatting has no use case" declared).
- **Dependency upgrades**: `icu` 2.3 → 2.4, `redis` 1.7 → 1.8 (`distributed-redis` feature), `uuid` 1.26 → 1.29 (http feature), `tonic` 0.14.6 → 0.14.8 (lock refresh), `ort` 2.0.0-rc.14 → rc.15 (latest rc on crates.io for the same 2.0.0 baseline; version requirement synced), `character_converter` version requirement normalized 2.1.5 → 2.1 (resolved version unchanged at 2.1.5).
- **tokio 1.53 → 1.54 with features trimmed from `full` to an explicit 8** (`rt-multi-thread`/`macros`/`time`/`sync`/`signal`/`fs`/`net`/`io-util`): a repo-wide grep confirmed `process`/`io-std`/`parking` have zero usage, so compiled artifacts no longer include the unrelated modules.
- **examples' reqwest switched from `default-tls` to `rustls`**: unified with the main crate's single TLS stack; the workspace build graph no longer pulls in native-tls.
- **Explicit feature de-duplication**: removed items implied by other features in the same set — confers `toml` (⊂`modules`), trait-kit `confers` (⊂`reload`), dbnexus `cache` (⊂`permission`) and `pool-health-check` (⊂`failover`), garrison `web-axum`/`secure-ct-eq`/`cache-memory`/`account-credential`/`listener`/`firewall`/`firewall-ddos` (all transitively enabled by retained features; each implied definition was verified against garrison/confers/trait-kit/dbnexus sources).
- **CI actions fully upgraded and unified to SHA pinning**: checkout unified at v7.0.1, codeql-action v4.39.1, cache v6.1.0, upload-artifact v7.0.2, setup-python v7.1.0, codecov v7.1.1, login-action v4.7.0, gitleaks v3.0.0, zizmor v0.8.0, install-action v2.88.0, crates-io-auth v1.0.6; the two floating `master` references in trivy-action and checkov-action were pinned to release tags. Follow-ups before finalization: install-action 2.89.1 (#18), zizmor-action 0.8.1 (#17), checkov-action 12.3139.0 (#16).
- **Scenario-suite CI fixes**: the scenario suite build adds the `candle,db,sqlite` features (conftest unconditionally pins `engine_type=candle` and profiles contain `sqlite::memory:`; without the features the server refuses to start and 90 cases go red in setup); the scenario suite installs grpcio; the coverage gate lands at the 90% floor per the 95/90 policy.
- **Release finalization**: all self-maintained dependencies converged from rc pins to minor-precision stable versions — confers `0.6`, trait-kit `0.5`, oxcache `0.5`, dbnexus `0.6`, inklog `0.3`, sdforge `0.5`, limiteron `0.3`, garrison `0.9`.

### Fixed
- **Security advisory triage (exemptions on record)**: `RUSTSEC-2026-0450` (ocipkg OCI unpacking path traversal — only candle optional → intel-mkl-src **build-deps** build-time path; no CI job enables candle; upstream intel-mkl-src is pinned `^0.2.8` and ocipkg is unmaintained, so the fix chain is broken) and `RUSTSEC-2026-0459` (libsqlite3-sys bundled SQLite CVE-2026-119375 — sqlx-sqlite 0.9.0 already pins the latest `<0.38`; the fix requires `>=0.39` with a broken upstream chain; the victim condition requires opening a DB+wal file pair from a less-trusted party, while all DB paths in this repo come from operator configuration and the test surface uses `sqlite::memory:` without WAL, so the attack surface does not apply) were recorded as dual-file exemptions in deny.toml + `.cargo/audit.toml` per the repo's existing criteria (unreachable path / inapplicable attack surface may be exempted), each with an exit condition.
- **ort load-dynamic default dylib path now selected per target platform** (new `ort_dylib_default` in `src/engine/session.rs`: Linux `.so` / macOS `.dylib` / Windows `.dll`): previously 8 test-side spots hardcoded `libonnxruntime.so`, which pointed at an ELF file on macOS and failed dlopen. `3rdparty/onnxruntime/README.md` gained macOS arm64 dylib placement instructions.
- **`uuid_like_id` gains an in-process monotonic sequence** (3-digit suffix, the `err-<u128>` format contract unchanged): consecutive calls within the same tick no longer produce duplicate error ids under macOS's μs-granularity real-time clock.
- **clippy::collapsible_if ×2** (`src/device/thread_tune.rs`) collapsed into let-chains per clippy 1.99; the symlink-escape test now builds a target outside a self-created root (the original `/etc/hostname` is a Linux-specific path — canonicalize hit ENOENT first on macOS).

---

<a id="030-rc3---2026-10-20"></a>

## [0.3.0-rc.3] - 2026-10-20

### Added
- **Multi-replica shared semantic cache and ban list (phase 2)**: `[semantic_cache] backend = "redis"` (requires `distributed-redis`) writes the trigram Jaccard fuzzy index into a Redis Hash + LRU ZSET, backfilling the local hot cache after a cross-replica hit; `limiteron/ban-sync` + `CacheBanStorage` share the ban list across replicas; startup fail-closed validation of feature/URL; new `startup-semantic-redis-*` messages. The index is cleared per model (`SemanticCache::clear_model`).

### Changed
- **`[database] max_connections` now actually takes effect + in-memory SQLite forced to a single connection**: previously this setting was parsed but never passed to the pool, so the pool size was always dbnexus's default of 20, making the shipped `config/config.toml` (10), `config_full.toml` (20) and `config_minimal.toml` (5) values all ineffective; the configured value is now used (default 10). The `sqlite::memory:` exception — every connection to a SQLite in-memory database is an independent database, so when the pool dispatches the "login write session" and the "auth read session" to different connections, the read side cannot see the just-written session row, manifesting as valid tokens being rejected with 401 (the `tests/http_auth_surface.rs` auth-surface case failed intermittently, ~1/8 locally reproduced) — is now forced to a single connection. File-based DBs and Postgres are unaffected. After this fix the effective concurrency cap is the configured value (default 10; shipped
  configs are 10 / 20 / 5). The structurally identical `[database] connect_timeout_secs` was also a dead knob
  (parsed but never passed down); it now maps to dbnexus's `pool_config.acquire_timeout`
  (converted from seconds to milliseconds): the wait cap for acquiring a connection permit changed from a fixed 5000ms to configurable
  (shipped configs 5s / 5s / 10s). This cap directly decides whether a failed permit acquisition queues or returns 500 immediately, so the configured value should not be
  below the actual DB concurrency.
- **Build configuration migration `.cargo/config` → `.cargo/config.toml`**: the old name made cargo 1.97 print a `deprecated in favor of config.toml` warning on every invocation; nothing in the repo (CI / Dockerfile / scripts) references the old name, and local overrides also go through the user-level `~/.cargo/config.toml`, so the `.gitignore` blank rule of "ignore config.toml, commit under the old name" is void — replaced by the reverse rule ignoring the old name to prevent regressions. The `[env] JEMALLOC_SYS_WITH_MALLOC_CONF` semantics are unchanged — a build.rs temporary probe verified the value is still injected into the build environment after the migration (`Ok("dirty_decay_ms:1000,muzzy_decay_ms:1000")`), so the jemalloc decay defense has no regression

- **`utoipa` 5.5 → 6.0 major-version jump**: the version constraint in `Cargo.toml` changed from `5.5` to `6.0`; the feature set (`macros` / `axum_extras` / `chrono` / `uuid`) and `default-features = false` are unchanged. Before the upgrade the lockfile simultaneously contained utoipa 5.5.0 (direct dependency) and 6.0.0 (transitive via sdforge); after the upgrade it converges on a single 6.0.0. `ToSchema` derive and `openapi!` macro usage required no source changes in this repo; the check/clippy/test gates all pass.

- **Fixed two workflows grabbing the same concurrency group and cancelling each other**: `ci.yml` and `health-check.yml` were both named `CI`, expanding their concurrency groups to `CI-<ref>` and `ci-<ref>` — GitHub group names are case-insensitive, so both landed in the same group, and together with `cancel-in-progress: true` the two workflows cancelled each other on every push, guaranteeing one never finished: Windows tests, coverage, `cargo doc` dead links, `cargo audit`, and the benchmark stayed in "cancelled" rather than "passed" state for a long time. `health-check.yml` was renamed `Health Check` with group name `health-check-<ref>`, so both gates now complete.

- **New security-check workflow and pre-commit secret scanning**: new `.github/workflows/security.yml` — actionlint (YAML syntax/expression hard gate), zizmor (GitHub Actions security audit, first-round non-blocking baseline), gitleaks incremental scan (the commit range of the push/PR) + a weekly full-history scan (`--log-opts=--all`); `.pre-commit-config.yaml` wires in gitleaks pre-commit interception. Actions are pinned to commit SHAs per repo convention, container images to digests. The two regex exemptions in the shipped `.gitleaks.toml` were simultaneously narrowed to `docs/` / `tests/` / `examples/` (the JWT rule also includes `src/security/`), preventing real credentials in `src/` from being swallowed by a global exemption.

- **Branches converged to a single `main`**: the repository only ever had `main` as a long-lived branch; `develop` in five workflow triggers never existed on the remote — dead configuration, now removed with the trigger description in `docs/TEST_SCENARIOS.md` synced.


- **Authentication sessions can be externalized (multi-replica)**: `[auth] session_backend = "memory" | "database"` (default memory, behavior unchanged); `database` writes to the deployment table `garrison_kv` via garrison `DbnexusKvDao` (`ensure_table` creates idempotently), requires a `db,sqlite|postgres` build and a configured `[database]`; `database` + `auth.enabled=false` or an uncompiled `db` is rejected loudly at startup. ~~Rate limiting and the semantic cache remain in-process semantics; lockout counters use read-modify-write `incr`, best-effort under multiple replicas~~ (superseded within the same release by the "multi-replica shared state" and "lockout counter atomization" entries below; strikethrough kept for the record). Convergence hardening: with a build lacking the `auth` feature, `session_backend="database"` is also rejected loudly at startup (`startup-auth-session-backend-auth-feature-missing`, mirroring the discipline of rejecting `engine_type="candle"` without candle); session-backend validation moved ahead of database-pool initialization (contradictory configs no longer cause connection/DDL side effects on the deployment database); the two-step non-atomicity of `set_if_absent`'s expired-key cleanup path and "database backups are equivalent to the session credential store" boundaries are documented in `docs/SECURITY.md`

- **Primary inference engine switched to onnxruntime (ort); candle demoted to opt-in fallback backend (onnx-primary)**:
  - `default = ["http", "onnx"]` — the default build is ort-primary and its dependency tree no longer contains candle (the zip-parsing risk leaves with it; the candle trio became optional; the full yank of zip was lifted upstream on 2026-10-16, candle points back at official 0.11, and the emergency fork was deleted)
  - The `EngineType` enum and all task-routing semantics are unchanged (wire/API contract untouched); configuring `engine_type="candle"` without the `candle` feature is rejected loudly at startup (`startup-engine-candle-not-compiled`, silent downgrade forbidden)
  - **Migration note**: legacy configs whose `model_path` points at a safetensors directory (no `engine_type` field) will error at startup after the upgrade — either set `engine_type = "candle"` explicitly and build with `--features candle`, or migrate to an onnx bundle
  - Feature semantics adjustments: `cuda` no longer implies the candle GPU backend (add `candle-cuda`); `metal`/`mkl`/`accelerate`/`quantized-gguf` all imply `candle`; `transcribe` still goes through `ort/cuda` (unaffected)
- **The effective default HTTP body limit is 2MB→5MB**: after removing the hidden hard cap, `body_limit_mb` (default 5) genuinely takes effect (previously the extractor layer's 2MB rejected before the config applied)
- **Audit log DB mode gains file dual-write**: with a db build, `logs/audit.log` (rotated) and the `audit_logs` table coexist (previously table-only, leaving the file-side operations surface missing)
- **In-flight dedup for concurrent identical requests**: the pipeline entry infers identical text with identical normalize only once, multicasting result copies — bitwise-identical results for concurrent identical text are structurally guaranteed by "compute once"
- **Swagger UI converged into the `swagger-ui` feature (not compiled by default, served on the main service port, single binary one build)**: `http`/`openapi` no longer carry `dep:utoipa-swagger-ui` and `sdforge/docs`; the interactive page and `/api-docs/openapi.json` are mounted only in `--features swagger-ui` builds (same port as the main service; one `cargo build` yields the complete binary); the `openapi` feature remains programmatic spec export (sdforge/openapi). Note: the Cargo lockfile resolves as the union of all features — optional dependencies (at the resolution layer) always exist, and feature isolation only trims compilation; on 2026-10-17 zip 3.0.0/8.6.0 were verified to resolve fine (the full yank had been lifted the day before).
- **Removed the legacy direct `utoipa-swagger-ui` dependency**: since sdforge 0.5.0-rc.9 the Swagger UI page and static assets are provided by its built-in docgen (`sdforge::docs`); the repo has zero references to `utoipa_swagger_ui`, so the declaration and the `dep:utoipa-swagger-ui` in the `swagger-ui` feature were deleted (the feature now composes as `openapi` + `sdforge/docs`; build commands and endpoints unchanged); the lockfile dropped `utoipa-swagger-ui`/`rust-embed`/zip 3.0.0 and 9 other packages. Afterwards zip 8.6.0 is held only by optional `candle-core` (no CI job enables candle); the three zip-unpacking advisories (RUSTSEC-2026-0367/0368/0369, victim API `extract()`, while candle uses `ZipArchive::new` for reading) are recorded as exemptions in `.cargo/audit.toml` as unreachable paths, to be lifted when candle is upgraded or removed

### Added

- **Multi-replica shared state (`distributed-redis`)**: `[rate_limit] backend = "redis"` (global + IP/User/ApiKey Lua-atomic fixed windows; per-key fixed 60s window, `global_window_ms` applies only to the global dimension; single Lua round trip bounded at 500ms, fail-closed), `[embedding] cache_backend = "redis"` (exact vector cache shared; bloom disabled, WAL mutually exclusive, cache keys bound to the weight fingerprint to prevent stale vectors across weight changes from hitting after restart); non-TLS `redis://` connections are rejected by the oxcache guard by default (`rediss://` or the dev escape hatch `OXCACHE_ALLOW_INSECURE_REDIS`); startup validates feature/URL/WAL conflicts/non-zero window (before any connection side effects); startup logs output only the masked endpoint
- **Lockout counter atomization**: with `session_backend="database"`, `AtomicKvDao` implements `incr` with a single-statement `INSERT ... ON CONFLICT ... RETURNING` (SQLite/PostgreSQL) — no loss across replicas (`tests/atomic_lockout_counter.rs` dual-instance 50×2 concurrent + single-instance 50 concurrent); SQLite concurrent writes enable WAL + bounded busy retry (dbnexus has no busy_timeout configuration surface; the pool-level fix belongs upstream)
- **ReDoc endpoint**: under the `swagger-ui` feature, `GET /redoc` is mounted, sharing `/api-docs/openapi.json` with Swagger UI; the page HTML is inlined, the rendering script pins `redoc@2.5.4` exactly and carries an `integrity` SRI check + `crossorigin="anonymous"` (against CDN/npm supply-chain tampering; the browser needs internet — Swagger UI remains vendored and offline-capable)
- **Preconfigured Grafana dashboard**: `deploy/grafana/vecboost-dashboard.json` (14 panels: HTTP/batching/queue/stage/cache/decisions/detect/translate) with import instructions in `deploy/grafana/README.md`

### Fixed

- **REST route version-prefix error (`/api/1/*` → `/api/v1/*`)**: all `#[forge]` endpoint declarations used `version = 1` (numeric literal), which the sdforge macro injected as `/api/1/*` based on the token text — a long-standing divergence from the spec and every document's `/api/v1/*` (the auth allowlist had been reverse-adapted, masking the root cause). Changed to `version = "v1"` (48 places), restoring routes to `/api/v1/*`; the auth allowlist/RBAC prefixes, metrics/audit unit tests, Rust/Python integration tests, README/docs/examples/config, and `scripts/doc_consistency_check.py` were updated in sync, plus a new regression gate for the old prefix. **BREAKING**: the old `/api/1/*` is no longer served (it never appeared in documents); `/health` and the OpenAI-compatible `/v1/embeddings` are unchanged.
- **candle `device="cuda"` silent downgrade**: when `cuda_is_available()` is false (build lacks `candle-cuda` or no usable GPU), the old code logged INFO and fell to CPU — effectively recording CPU numbers under the GPU label → now a `ConfigError` at startup (new FTL key `engine-candle-cuda-unavailable`, en/zh at parity) + regression test `resolve_compute_device_rejects_cuda_without_candle_cuda`
- **Lossless (fp32/fp16) transcription bundles had no entry point + precision misreported by filename**: `asr.rs` hardcoded encoder/decoder candidate names as `.int8.onnx` and the loader re-assembled paths by literal → candidate chain (int8 first, bare-name fallback) + `resolve_graph_paths()` as a single source of truth; `session::detect_onnx_weight_precision` now judges precision by **graph content** (QuantizeLinear/DequantizeLinear operator markers) without falling back to filenames; new startup INFO log `log-engine-transcribe-graphs`; 5 regression tests (including crossing 1 MiB read-block boundaries and "renaming must not misreport")
- **Three compatibility defects in the real-device onnx embedding path** (previously untested with real assets; exposed by the first benchmark run): single input rank-1→`[1, seq]` 2-D (standard BERT ONNX graph spec); three-input graphs (optimum/transformers.js exports) probe `session.inputs()` and feed all-zero `token_type_ids`; rank-2 input ↔ rank-3 output index alignment
- **onnx embedding pooling hardcoded to mean** (silently changing semantics for BGE-family models and making the `ModelConfig.pooling_mode` config ineffective) → inferred from the model name at construction (same `infer_pooling_mode` single source as candle) + Cls/Mean/Max support; after the fix, per-text cosine of same-source dual exports = 1.000000
- **onnx HF fetch incompatible with `onnx/` subdirectory repos** (Xenova-style direct references always failed) → candidate chain `model.onnx → onnx/model.onnx → model_quantized.onnx → onnx/model_quantized.onnx`
- **axum extractor-layer `DefaultBodyLimit` 2MB hard cap** masked the `[server].body_limit_mb` config (silently ineffective above 2MiB; a 48s transcription audio got 413) → both layers unified to the same config basis
- **FT-03 bitwise inconsistency for concurrent identical text**: with min_workers>1 identical text landed in multiple batches, and GEMM batch-shape differences caused bitwise drift → structurally fixed by in-flight dedup
- **Missing R-auth-009c rate-limit audit file surface**: db builds wrote DB-only while the scenario anchored `logs/audit.log`, which stayed empty → fixed by DB dual-write
- **Batch OOM fallback did not consume the real config (fallback semantics distorted)**: `embed_chunks_concurrent`'s single-chunk path carried an inline fallback loop — calling `try_fallback_to_cpu` with `ModelConfig::default()` and bypassing the `ModelManager` gate, with a local retry cap of 1 (inconsistent with the single-text path's 2). Now uniformly delegated to `common::handle_oom_fallback` (real `ModelConfig`+`ModelManager`, retry cap 2); library mode without a manager fails loudly with `oom-no-fallback-available` instead of pretending to fall back. The stale `log-service-oom-error-detected-batch` / `log-service-successfully-fell-back-cpu-2` FTL keys were removed along the way
- **Audit flush falsely reported success**: `AuditLogger::flush()` previously did `let _ = rx.await` — after the writer circuit-broke and exited on consecutive write failures (`ERROR_THRESHOLD=10`), in-flight flushes got no ack, yet flush still returned `Ok(())`, so the shutdown path misjudged events as persisted. Now three explicit failure modes: send failure (writer dead) / ack timeout (`FLUSH_ACK_TIMEOUT=5s`, writer alive but unresponsive) / oneshot Err (handler exited mid-flight); new FTL keys `audit-flush-writer-unavailable` and `audit-flush-timeout` (the shutdown path in `registry/impl_.rs` already consumes that Err — no change needed)
- Engine/decision factory test-isolation defects (2 places): `test_create_decision_task_dispatches_to_candle_engine` and `test_process_decision_engine_panic_maps_to_inference_error` depended on i18n initialization order → each now calls `i18n::init()` explicitly

### Quality and testing

- **CI build surface completed**: feature-matrix adds `swagger-ui` and `quantized-gguf --lib` compile checks plus the `quantized_parity` quality-gate target compilation (a real run still needs model assets; missing assets follow the existing SKIP semantics)
- **Session externalization regression**: `tests/auth_session_backend.rs` (cross-connection write visibility, delete visibility, TTL expiry, idempotent table creation)
- **Grafana asset regression**: `tests/grafana_dashboard.rs` (JSON parseable, datasource template variables, required panels and PromQL fragments)
- **Quantized-tier precision gate**: `engine_backend_parity::onnx_quantized_vs_fp32_parity` (int8 vs fp32 median cosine **0.997085**, q4 **0.984944**; thresholds reuse the repo's GGUF basis 0.98/0.95; auto-SKIP without assets)
- **Rerank throughput benchmark**: new `benches/rerank_bench.rs` (single pair + 1×32 batch), and `benches/common/mod.rs` gains **per-engine device knobs** (`VECBOOST_BENCH_DEVICE`; ort needs `transcribe`, candle needs `candle-cuda`; missing features exit=1 loudly — CPU numbers must never be recorded as GPU)
- **Engine selection measurements (2026-10-18 idle window, 2–3 runs per cell)**: embedding single onnx 2.56 ms vs candle 78.0 ms (**30.5×**), batch 26.35 vs 186.5 ms (**7.1×**); `candle-cuda` is **15.4~35.5×** faster than candle CPU, yet still **1.9~2.3×** slower than ort's CUDA EP on the same card (same model, same VRAM-class need ~1.95 GB); details in `docs/PERFORMANCE.md` "Engine selection evidence"
- **Cross-engine reconciliation baselines**: embedding accuracy (per-text cosine of same-source dual exports, threshold 0.999), decision accuracy (|Δlogit| ≤ 1e-4, existing parity gate), decision latency observation (candle 455ms vs onnx 118ms/question, informational)
- **Performance benchmarks** (same machine, CPU, criterion): embedding single onnx 2.89ms vs candle 71.99ms (~25×), batch×32 27.65ms vs 176.32ms (~6.4×); `embed_throughput_bench` supports `VECBOOST_BENCH_ENGINE` engine selection + engine name in the ID + `expect` guarding against "fast-fail fake performance"
- **New tests**: `onnx_embed_e2e` (real-device embedding contract and single-vs-batch consistency), `engine_backend_parity` (accuracy reconciliation + performance observation), scenario `test_transcribe.py` (HTTP transcription e2e: real-device ASR/size/defense, 4 cases), gRPC `vecboost.decide`/`vecboost.transcribe` method tests
- **Test infrastructure fixes**: scenario conftest derives the binary path dynamically (fixes worktrees testing the wrong target), integration/scenario config templates set `engine_type` explicitly, `scripts/link-assets.sh` (worktree asset linking), `download_model --onnx/--translate` (embedding/translation bundle asset acquisition)
- **Batch OOM fallback contract test**: the `RecordingOomEngine` mock pins that the fallback really consumes the real `ModelConfig` (asserts `name != "default"`, `model_path` is the injected value) and that no fallback is attempted without a manager (zero calls to `try_fallback_to_cpu`)
- **Audit flush contract test**: fast Err when the writer is unavailable / events persisted after ack / fast-flush failure after the circuit-breaker exit with pre-break events retained in the rotated file

### Known limitations

- The transcription engine is currently **f32-only** (KV cache and tensor I/O hardwired to f32): fp16/bf16 graphs are explicitly rejected at load time (`conv_frontend input[0] dtype Float16 is not supported`); full fp16 support needs dtype generalization (one KV step is 29.4M elements: 117 MiB f32 / 59 MiB f16)
- candle's `metal` (Apple Silicon) tier has no local hardware to verify; the `candle-cuda` build needs `nvcc` on `PATH`, and running under WSL2 needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib`

- The translation golden gate (`translate_e2e`) needs both en-zh + en-es bundles (the latter obtainable via `--translate Xenova/opus-mt-en-es`); en-zh real-device assertions are covered by integration tests
- gRPC transcription requests are bound by sdforge's Call-layer 1MB application cap (large audio goes over HTTP); the `samples` array protocol is unfriendly to large audio (48s ≈ 14MB JSON); streaming/chunked transfer awaits protocol evolution
- candle's single forward pass is inherently slower on its CPU implementation (~85ms@bge-small before batch amortization); demoted to a fallback backend with no optimization investment

---

<a id="030-rc2---2026-10-15"></a>

## [0.3.0-rc.2] - 2026-10-15

### Added

- **Transcription service (transcribe engine and service surface merged into the main crate)**: in-house transcription engine (hand-written fbank + conv/encoder + 28-layer KV-cache greedy decoding + BPE restoration; hand-written silero VAD and a `--vad-only` path; 4 parallel workers across VAD segments; global thread-pool/EP knobs fixed, full-audio RTF 0.1535); HTTP/gRPC service surface wired (the `/api/1/transcribe` endpoint always registered + assembly gate 400 + backpressure 503, the `[transcribe]` config section); CLI subcommand (WAV path input via `read_wav_16k_mono` into the domain request); `transcribe-poc` converged into a thin-shell CLI (inference calls the `vecboost::transcribe` public facade, stdout five-line contract, PyO3 torch backend cross-check retained)
- **Translation inference hardening**: pre-translation token protection + post-translation backfill (DTI abbreviations/arXiv identifiers no longer lost at the model level, placeholders ⟦Zn⟧ backfilled in two tiers); sentence-level parallelism for 600+ character inputs (4 workers; worst paper 155.7s→22.0s, 7.1×; parallelism failure fails the whole request loudly, no Partial sentence-dropping); M2M100/NLLB-200 multi-vendor families; two-pass decoding (whole-segment direct + per-sentence rescue on degeneration); EOS suppression window `min_new_tokens` (fixes greedy early truncation); exit post-processing chain (traditional/simplified normalization, symbol repair, output replacement table); the `translate_ab` A/B selection harness; translation/language-detection Criterion benchmarks and baseline data
- **Decision tasks join the priority queue**: Overloaded backpressure and the `vecboost_decision_seconds` metric completed

### Fixed

- **onnx path crashed at startup (since v0.2.1)**: the ort feature changed from `alternative-backend` to `load-dynamic` — the former requires the consumer to supply OrtApi and the repo had zero call sites, so building the first Session panic-aborted; at runtime `ORT_DYLIB_PATH` points to libonnxruntime ≥1.23 (pinned 1.28.2), see `3rdparty/onnxruntime/README.md`
- **ort rc.14 process-level Environment singleton**: rc.14 allows only one `Environment` per process; `build_session` moved to a `OnceLock` process-level shared environment (concurrent losers re-probe instead of misreporting) — building a second engine in the same process used to crash, exposed by the real translation bundle tests unlocked by the fix
- **warm_up_cache key-scheme drift**: warmup wrote entries keyed by bare text while the read path queried `emb:{model_id}:{xxh3}`-format keys, so warmed entries were never readable — extracted `cache_key_for` as the unified constructor with a regression test pinning it
- **Token default dual-track drift**: `DEFAULT_TOKEN_EXPIRATION_SECS=3600` ("default 1 hour") was actually unreachable; now derived from the single source `DEFAULT_TOKEN_EXPIRATION_HOURS` (86400), with docs/tests/examples unified at 24 hours
- **process_search_batch multi-chunk global indices**: the second chunk's vectors never landed and its errors overwrote the first chunk
- **RateLimitExceeded direct-return path unified to 503**: 429 belongs to the auth rate-limit middleware; the same error is no longer double-mapped

### Changed

- **Self-maintained dependencies upgraded to the latest crates.io rc**: trait-kit 0.5.0-rc.9 / confers 0.6.0-rc.10 / inklog 0.3.0-rc.9 / oxcache 0.5.0-rc.8 / limiteron 0.3.0-rc.7 / dbnexus 0.6.0-rc.9 / garrison 0.9.0-rc.3 / sdforge 0.5.0-rc.8; upgrade adaptations for confers Zeroizing, garrison's release error variants, sdforge UnifiedError/gRPC Status contracts, and utoipa 5.5/6.0 explicit disambiguation
- **ort 2.0.0-rc.13 → rc.14**: `Session::builder(&env)` explicit environment, CUDA EP moved to `ort::ep::CUDA`, CUDA EP gating via cuda→transcribe (ort/cuda actually arrives through the transcribe feature)
- **Refactoring and convergence**: overlong functions split in 10 batches to zero (startup orchestration / worker state machine / inference hot path, etc.); the shared ONNX Session build layer absorbed three drifts (probe/precision single source); single-flight extracted into a shared module (reused by SemanticCache and OxCacheBackend); dead-code cleanup and test-asset dedup
- **Dependency feature trimming**: fluent-bundle/writeable `default-features = false`; utoipa-swagger-ui vendored (compilable offline)
- **GitHub actions upgraded**: codeql-action v4.38.3, setup-buildx v4.4.1, metadata-action v6.2.0, etc.

### Build and CI

- Docker workflow fixes: Dockerfile aligned to the repo-root context (previously red on every push), HF_HOME pointed to a writable volume fixing the container startup crash, Trivy SARIF upload given security-events:write, the deploy leg split GHCR/Docker Hub publishing by credentials; scenario CI adds protoc

### Quality and testing

- Line coverage reached **90.89%** (llvm-cov, CI-identical feature set); inline unit tests 2237 → 2277 (+124 this round: all branches of the four auth handlers, translation real-bundle end-to-end, pipeline SLA harvest/cancellation propagation, doctor/device/utils boundaries and error branches, etc.)
- Documentation consistency checker generalized: CLI whitelist 5 → all forge(cli=true) names; the env consumption surface now includes scripts/ ops scripts (removing 8 false positives for the memcap knobs)
- Pre-release SAST remediation: transcribe-poc lockfile rustls aligned to 0.23.46 (4 MEDIUM) and ort rc.14 synced; `scripts/autotune.py`/`warmup_corpus.py` gained http(s) scheme guards (bandit B310); pyo3 0.22.6 known vulnerabilities (RUSTSEC-2026-0176 etc.) assessed and **risk accepted** — they exist only in the non-published transcribe-poc local baseline tooling; upgrading requires a pyo3 0.22→0.29 major migration, deferred until its /home/dev interpreter pin is lifted

### Evaluation archive

- trait-kit unified SDK wrapper assessment: technically feasible but not implemented this round (defer_with_proposal; P1 metadata honesty → P2 unified health probing → P3 assembly convergence → P4 resource contracts; prerequisites and risks in [TRAIT-KIT-SDK-ASSESSMENT](../reviews/TRAIT_KIT_SDK_ASSESSMENT.md))

<a id="030-rc1---2026-09-22"></a>

## [0.3.0-rc.1] - 2026-09-22

### Fixed (fix-audit-defects-r1: six-theme audit, all 35 findings closed)

- **Scheduling correctness**: priority aging no longer strands the queue head (a head exceeding its wait threshold is served as a fallback instead of being skipped forever at its level); new `dequeue_expired` harvests queued requests past their SLA, eliminating "head skipped forever + queue/response-channel leaks"; requests whose own timeout arrives inside the batch-assembly window complete with 504 when leaving the batch instead of masquerading as rate-limited
- **Batched inference correctness**: candle fp32 batch pooling dispatches by `pooling_mode` (previously hardcoded CLS, so Mean/Max models got wrong batch results with single/batch cosine ~0.5); the ONNX mean-pooling divisor no longer scales with hidden_size; `max_seq_len==0` errors loudly instead of silently returning zero vectors
- **Engine exit contract**: unified as "L2-normalized vectors" — the quantized engine re-normalizes, the candle single path re-normalizes, trait docs corrected (`normalize_l2` idempotent; service-layer behavior unchanged)
- **NaN/Inf full-chain defense**: `normalize_l2`/`cosine_similarity` error loudly on non-finite input; engine output → cache write → response exit validated layer-by-layer with `is_finite`, so bad vectors neither enter the cache nor reach clients
- **Semantic cache**: internal keys include model_id and the semantic index is isolated per model (no cross-model vector pollution); the fuzzy-hit threshold is an independent `fuzzy_threshold` (default 0.85; `similarity_threshold` no longer mixes two scales); fuzzy hits emit `vecboost_semantic_cache_fuzzy_hits_total`; `comparison_mode` marked deprecated; `unload_model`/`switch_model` both clear the semantic cache; model switching propagates to rerank (`replace_engine`); the rerank cache key is hashed to eliminate concatenation ambiguity and includes the model name
- **Cache robustness**: production TTL defaults to 3600s (`OxCacheBackend::with_ttl`); the semantic-index read path touches `last_access` (eviction corrected from FIFO to true LRU); WAL replay aligns committed by file sequence number (corrupt records inside the prefix no longer resurrect uncommitted tail records); per-key single-flight (concurrent identical text inferred once)
- **Vector computation**: same-condition benchmarks disproved the "pseudo-vectorization" hypothesis — the original single-accumulator implementation was already vectorized by LLVM under the baseline build (1024-dim 353ns); the 4-accumulator rewrite was 2.2× worse, so the loop body keeps the original implementation with a one-time warning for the unrotated `rotate` tail; PERFORMANCE.md gained the RUSTFLAGS vectorization note and the fat-LTO codegen drift observation
- **GPU memory**: three CUDA pool accounting fixes (Drop returns counts / failed frees do not return + error log / clear no longer fakes zeroing); memory caps genuinely enforced (`[model] memory_limit_bytes` checked before model load when configured, over-limit rejected loudly, accounted by directory size after load); VRAM query failures report 0 available + warn (no more pretending plenty); CUDA version now really probed (NVML `sys_cuda_driver_version` / nvidia-smi banner parsing, replacing the fabricated `Some(11)`); the unwired `tensor_pool.rs` orphan file deleted; the paging prefetch completion path gained a budget check + rollback, and swap-in eviction moved to a multi-victim loop
- **Pipeline resilience**: worker inference panics are isolated (in-batch requests get explicit errors; the worker survives); client-disconnect cancellation propagates (`CancellationRegistry`, cancelled requests dropped at dequeue to free slots); server-side timeouts get a dedicated `RequestTimeout` error code (HTTP main path 503+Retry-After=1, direct `VecboostError` path 504 — no more masquerading as 400/429); graceful shutdown drains the queue (queued requests immediately get shutting-down 503) + configurable `drain_timeout_secs`; the scale monitor starts in production (`WorkerManager::start`) + new metric `vecboost_queue_wait_seconds`; the consumer-less `WorkerTask::ProcessRequest` dead branch removed
- **Test assets**: `model_snapshot_regression` SKIP decision aligned with weight existence (same basis as lib tests); the pytorch bin loading test gained a prerequisite check; bge-small-en fp32 weights added

### Fixed (G1 review residual closure)

- **Memory-cap enforcement wiring**: `AnyEngine`'s `InferenceEngine` impl now forwards `attach_memory_limit_controller` (previously missed, so the service-layer switch_model and startup-wired controllers all landed on the trait's default no-op, silently disabling the Candle/ONNX engines' Exceeded/Critical CPU-fallback chain)
- **Input surface tightened**: `/v1/embeddings` and decision requests (name/options/state string leaves) changed from accepting U+2028/U+2029 (LINE/PARAGRAPH SEPARATOR) to 400 — a rendering-level log-obfuscation surface on par with `\n`; error echo also literalizes these two characters (echo single-lining does not depend on the upfront validation, defense in depth). Clean existing data before ingestion (see USER_GUIDE "Input character defense")
- **Decision request validation**: choice-type options reject empty/whitespace-only strings (same basis as name, removing ambiguity for string-matching consumers)
- **Quantized routing basis**: `switch_model`'s `.gguf` path detection now reuses `should_use_quantized_engine` (case-insensitive, consistent with startup; `"MODEL.GGUF"` previously forked to the safetensors path and errored)
- **Config fails loudly**: engine-level `ModelConfig` deserialization adds `deny_unknown_fields` (misspelled keys like `tsak` error loudly instead of being silently ignored and falling back to defaults); unknown keys in the TOML `[model]` section remain silently ignored, now annotated in the USER_GUIDE configuration section

### Added

- **Translation interface (add-translation-and-langdetect)**: the `[translation]` section with `[[translation.pairs]]` multi-pair configuration (built eagerly at startup, pair cap 16, any missing bundle fails fast at startup); the `/api/1/translate` endpoint routes precisely by `(source_language, target_language)` (unconfigured pairs get 400 with the full configured pair set echoed); `source_language` defaults to whatlang auto-detection (unreliable → 400 `translate-detect-failed`, detected language backfilled into `detected_source_language`); Marian/opus-mt seq2seq engine (`src/engine/onnx_translate.rs`, encoder+decoder_model_merged dual-Session greedy decoding, cache branch disabled — the Xenova merged-graph export defect measured, uncached full recompute O(L²) bounded); two-tier token budget validation (service-layer pre-check + engine re-check after encode, over-limit rejected loudly); the E2E golden gate `tests/translate_e2e.rs` (20-sentence golden exact equality + greedy determinism + quality output `out/translate-quality.json`); the four-protocol surface (HTTP/gRPC `vecboost.translate`/MCP `translate`/CLI), residency accounting Translate arm (disk × 2.0 conservative factor, int8 has no calibration statement), and USER_GUIDE/API_REFERENCE doc sync
- **Language detection interface (add-translation-and-langdetect)**: the `/api/1/detect` endpoint and the detection core in `src/text/langdetect.rs` (whatlang 0.18, exhaustive ISO 639-3→639-1 mapping of 70 variants, non-equivalent 639-3 codes fall back); `DetectService` is µs-level pure CPU without spawn_blocking; undetectable returns 200 + `language: null` + `is_reliable: false` (explicit); the `allow` allowlist narrows candidates (invalid codes get an explicit 400); translation auto-detection shares the same function with this endpoint (single source of truth)
- **Project positioning change**: from embedding service to multi-task inference service (embedding/rerank/decision/translation/language detection) — Cargo `description`/`keywords`, bilingual README positioning taglines and feature lists, API_REFERENCE (translation/detection sections + gRPC method table), and USER_GUIDE (translation config/bundle contracts/limitations) rewritten in sync
- **Translation coverage closure (items not covered by the feasibility study)**: CJK half-width punctuation normalization for zh/ja/ko code-family targets (zh/zho/cmn/yue, ja/jpn, ko/kor) in the engine layer (`src/engine/onnx_translate.rs`, TemperatureCalibration precedent at the same layer; `,;:?!` → full-width; digits flanked by digits (thousands separators/time context) and periods excluded; golden-diff verified exactly 4 zh sentences changed, punctuation only, with zero change for en-es); the `benches/translate_bench.rs` benchmark (langdetect short/10k cap, en→zh short/medium/long, en→es short, count_tokens; baseline `docs/benchmarks/translate_baseline_results.md`); the `cuda` feature compile-chain verified on real hardware (RTX 5080 + CUDA 13.0 nvcc, candle-kernels actually compiled) with the cuda-gating unit tests actually run; the three-review 30-item disposition ledger committed as `docs/reviews/TRANSLATION_REVIEW_LEDGER.md` (29 fixed, 1 explicitly kept, with per-item evidence locations; `docs/reviews/` is a local-only directory, not in git)

- **Multi-checkpoint configuration surface (laya-p2p3-coverage P2-2)**: the `[model.checkpoints.<name>]` preset table — `switch_model` applies the matched checkpoint's decision sequence budget (`head_max_len`/`max_len`, parameterizing `DecisionParams` instead of passing compile-time constants) plus model_path/task/engine_type/tokenizer_path (request-explicit values take precedence, the preset is the default layer, and engine_type inherits the currently loaded model when unconfigured); the built-in `laya-multilingual` preset (head_max_len 256 / max_len 256, the documentation §2.1 multilingual basis) switches with zero configuration; a same-name explicit entry overrides the built-in field-by-field (unwritten optional fields inherit built-in values, so customizing only model_path keeps the multilingual budget); the preset table is validated fail-fast at startup (unknown fields/missing required/zero-or-over-hard-cap parameters — a single field over 8192, or head+state+4 fixed special tokens exceeding the window, or table keys drifting from name all refuse startup, rejecting silent clamping) and injected into EmbeddingService for switch lookups; without configuration, behavior is exactly as before
- **Candle-native decision head (laya-p2p3-coverage P3)**: `task = "decision" + engine_type = "candle"` runs a pure-Rust forward (`src/engine/candle_decision.rs`, no onnxruntime dependency) — a ModernBERT-large backbone (candle-transformers, `encoder.`→`model.` key remapping; transformers 5.0's lack of bias/layer0-fused attn_norm weights measured compatible) + a decision head (type_emb broadcast per question → 2 norm_first TransformerEncoderLayers (fused `in_proj_weight`, FFN ReLU) → marker gather → option scorer → act/escalate heads); preprocessing/post-processing/temperature calibration share the onnx pipeline's source (protocol split into `decision_protocol.rs`, single implementation of sequence construction/collate/softmax/three post-processings); the official `convaiinnovations/laya` checkpoint key mapping is grounded in a safetensors header measurement of 206 tensors (missing keys error loudly with a full list); the alignment gate `tests/candle_decision_parity.rs` cross-checks the same questions at `|Δlogit| ≤ 1e-4` (one question each for choice/score/noul; measured pass with real checkpoint + real bundle; the file SKIPs without assets); the `EngineFactory` decision arm dispatches backends by engine_type (candle/onnx), and `AnyEngine` gained a `supports_rerank` variant forward (previously the decision engine's false was unreachable through AnyEngine); checkpoint acquisition steps and limitations are in USER_GUIDE "Candle-native decision path". **Migration note (a behavior change for existing task=decision deployments)**: the old decision task ignored engine_type and always consumed the ONNX bundle; after the upgrade, `[model].engine_type` unset (candle) requires a PyTorch checkpoint laid out as `models/laya-pytorch/` (a loud load failure at startup indicates this change) — place the checkpoint, or set `engine_type = "onnx"` explicitly to keep the existing ONNX bundle. `memory_limit` accounting for decision + candle scales by 2× disk (F16 checkpoint → F32 materialized-residency basis; switch-projection enforcement and startup accounting share the source), so deployments no longer need to reserve 2× disk headroom
- **Decision pipeline access layer (G3)**: the `/api/1/decisions` four-protocol endpoint (HTTP/MCP/CLI/gRPC `vecboost.decide`) — `DecisionService` (state/questions validation chain + domain-layer validate reuse + supports_task gate-before-call + spawn_blocking isolation of blocking inference), `DecisionModule` registered through the server/MCP/CLI kit assemblies; `[model] engine_type` / `task` configuration drives it (eliminating the hardcoded Candle on the startup path; unknown values error loudly at startup); the switch contract gains the `task` field (request value first → inherit current → fall back to embedding) and propagates engine replacement to decisions; either-or semantics for switching (`task=decision` makes embed/rerank endpoints 400, and vice versa decisions gets 400)
- **Laya decision pipeline (G2)**: `src/engine/decision.rs` — `[CLS] head [SEP] [MASK] option… [SEP] state [SEP]` preprocessing (marker_pos/qtype encoding, head+options 192-token budget pre-check, state 256-token truncation), 5-tensor ONNX inference (input_ids/attention_mask/marker_pos/marker_mask bool/qtype), per-cardinality temperature calibration (`laya_config.json`, clamped to [0.5,5] at load, missing-bucket fallback 1.2, missing file warns without silence), choice/score/noul post-processing (argmax + full probability table / expected grade Σi·p_i + distribution / P(true))
- **Factory task dispatch arm**: `EngineFactory::create` routes all `task=decision` to the decision pipeline (the `AnyEngine::Decision` production construction site; embed/rerank endpoints reject decision models loudly with UnsupportedTask 400)
- **Decision metrics**: `vecboost_decision_seconds` (per-request end-to-end decision latency histogram; `_count` is the call count); the decision pipeline's take_stage_snapshot is always None (Stage three-value exemption; observation goes through the dedicated metric)
- **laya_poc P0 numerical-parity example**: `cargo run -p vecboost-examples --bin laya_poc --features onnx` — full EngineFactory→AnyEngine→trait chain call; when `LAYA_GOLDEN_JSON` points to a Python onnxruntime export, per-value asserts of probability deltas <1e-4; SKIPs with download guidance when the bundle is missing
- **Inference correctness**: full implementation of `PoolingMode::{Cls,Mean,Max,Auto}` (Auto inferred from the model name), mean as attention-mask weighted average; cross-platform unified HuggingFace `tokenizers` replacing the in-house WordPiece with its silent 250-word fallback; vocab_size derived from the model config.json; `/v1/embeddings` usage is a real token count (audit-remediation)
- **Swagger UI**: `/api-docs/openapi.json` + `/swagger-ui/` (OpenAPI generated by sdforge) (audit-remediation-gaps)
- **Real readiness probe**: `/health?depth=full` (DB SELECT 1 + engine dummy inference + limiter health) (audit-remediation-gaps)
- **Time-window dynamic batching**: `[pipeline.worker] batch_wait_ms` (default 5ms, `0` restores drain-style); requests within the window are aggregated and dispatched early once `max_batch_size` is reached (absorb-colibri-optimizations)
- **In-batch dedup**: identical texts in `/embed/batch` inferred once per batch, results back-filled by index (bitwise equal); metric `vecboost_inbatch_dedup_ratio` (absorb-colibri-optimizations)
- **Physical-core thread tuning**: tokio/rayon sized to physical cores with SMT/E-core detection, disabled by `VECBOOST_NO_THREAD_TUNE=1`; multi-socket startups print numactl advice (absorb-colibri-optimizations)
- **GGUF quantization path**: `[model] quantized` + `--features quantized-gguf` (Q8_0/Q4_K); quality gate median cosine ≥0.98/≥0.95 (`tests/quantized_parity.rs`, model via `VECBOOST_GGUF_MODEL`) (absorb-colibri-optimizations)
- **Vector-output quantization**: `[semantic_cache] comparison_mode = i8|binary`, Hadamard rotation (kv_tq.h-isomorphic) prefilter + exact re-verification on the original vectors (absorb-colibri-optimizations)
- **Multi-model LFRU residency**: `[model] max_resident_models` / `resident_memory_budget_mb`; heat<<8|recent scoring, 25%+4 hysteresis, heat persisted to `data/model_heat.json` (absorb-colibri-optimizations)
- **Cache WAL persistence**: `[embedding] persist_path` / `persist_max_bytes`, crash-safe two-phase append, startup replay rebuild (absorb-colibri-optimizations)
- **Hardware-aware planning**: `[device] auto_plan = true` probes RAM/physical cores/GPU/models at startup to generate a conservative plan (explicit config wins); `scripts/autotune.py` coordinate-descent measured tuning (absorb-colibri-optimizations)
- **Read-only diagnostics**: `vecboost doctor` — config/tokenizer/cache persistence/threads/GPU/model integrity (safetensors headers, `__metadata__`, gguf magic, hidden_size pairing; matched by content role), exit code 1 on FAIL (absorb-colibri-optimizations)
- **Startup warmup**: `--warmup N` synthetic-text warmup; `scripts/warmup_corpus.py` warm-corpus replay (dedup, order-preserving) (absorb-colibri-optimizations)
- **New metrics**: `vecboost_batch_size`, `vecboost_batch_wait_seconds`, `vecboost_inbatch_dedup_ratio`, `vecboost_stage_seconds{stage=tokenize|inference|pool}` (absorb-colibri-optimizations)

### Changed (breaking, must-read before upgrading)

- **Secure defaults**: the factory default of `host` changed `0.0.0.0` → `127.0.0.1`; binding a non-loopback address while `auth.enabled=false` refuses startup (the `VECBOOST_ALLOW_INSECURE=1` escape hatch logs an ERROR); the factory default of `use_gpu` changed `true` → `false`
- **Login convergence**: only `default_admin_username` (default admin) may log in; enabling auth requires `VECBOOST_ADMIN_PASSWORD` or startup is refused; arbitrary-username + shared-password login is rejected
- **XFF trust inversion**: when `trusted_proxies` is empty, `X-Forwarded-For` is ignored (previously trusted unconditionally); reverse-proxy deployments must configure `trusted_proxies` explicitly
- **RBAC wiring**: `/api/1/model/*` and `/embed/file` require the admin role
- **`/embed/file` convergence**: requires explicitly configured `[server] grpc_allowed_roots` (no cwd fallback); single-file limit 10 MiB; `text_preview` admin-only
- **Token lifecycle**: `token_expiration_hours` defaults to 24 hours (previously the garrison default of 30 days)
- **CSRF default**: follows `auth.enabled` (`AuthConfig` default csrf.enabled=false)
- **`--config` fail-fast**: an explicitly specified missing path → error and exit (previously silent fallback to defaults)
- **Cache and tokenizer behavior**: cache keys bound to the model identity (first-run cache misses after upgrade); HF tokenizers produce different results than the old in-house implementation (more correct) — existing vectors must be rebuilt
- **CLI**: unknown subcommands error and exit (code 2) instead of silently starting the HTTP server; `--help` works

#### Upgrade migration guide (old behavior → new behavior → migration action)

| Change | Old behavior | New behavior | Migration action |
|------|--------|--------|----------|
| Bind security | `auth.enabled=false` could bind `0.0.0.0` | Non-loopback binding + no auth → **refuses to start** | Keep `127.0.0.1` for local development; set `VECBOOST_ALLOW_INSECURE=1` in trusted-network containers (ERROR logged); enable auth in production |
| Admin password | `VECBOOST_ADMIN_PASSWORD` optional; any credentials logged in when missing | `auth.enabled=true` and missing → **refuses to start** | Set `VECBOOST_ADMIN_PASSWORD` (≥12 chars) |
| Login username | Any username + admin password could log in | Only `default_admin_username` (default `admin`) may log in | Clients must use the admin username |
| XFF trust | `X-Forwarded-For` trusted unconditionally when `trusted_proxies` empty | **XFF ignored** when empty; direct address used | Configure `trusted_proxies = ["10.0.0.0/8"]` etc. for reverse proxies |
| RBAC | `/model/*`, `/embed/file` needed only login | Requires the **admin role** | Business users call embed with a regular account; model management uses admin |
| `/embed/file` | Default allowed root = process cwd, text preview returned | Requires explicit `grpc_allowed_roots`; single file ≤10 MiB; `text_preview` admin-only | Configure allowed roots; clients stop relying on preview |
| Token lifetime | Default 30 days (garrison default) | Default **24 hours** | Configure `token_expiration_hours` explicitly for long sessions |
| CSRF | Off by default | Follows `auth.enabled` (explicit `csrf.enabled=false` still honored) | No impact for pure-Bearer APIs |
| `use_gpu` | Factory `true` (warn-and-fallback without a GPU build) | Factory `false`; requesting GPU without the feature → WARN + CPU fallback | GPU-build users enable it explicitly |
| `--config` | Missing path → silent fallback to defaults | **Errors and exits** (code 2) | Check path spelling |
| Cache keys | `text:{raw}` (no model) | `emb:{model}:{xxh3_128}`; switching models clears the cache | First-run cache misses after upgrade (one-time) |
| Tokenizer | Linux in-house WordPiece (250-word fallback on anomaly) | Cross-platform HuggingFace `tokenizers`; load failures **error** | Existing vectors are incompatible with rebuilt ones — rebuild the index |
| CLI | Unknown subcommands silently started the HTTP server | stderr usage hint + exit code 2 | Adjust scripts relying on the old behavior; `--help` works |

**Multi-replica boundary**: the inference path is stateless, so horizontal scaling works with auth disabled (rate limiting/caching are in-process semantics). With `auth.enabled=true`, auth sessions live in process memory (oxcache DAO), **single replica only**; session externalization needs the garrison db backend (see [FAQ](FAQ.md) and [SECURITY](SECURITY.md)). (Status note: the preceding sentence quotes the original text at the roadmap-revision point; session externalization and rate-limit/cache Redis sharing landed within this release via the "authentication session externalization" and "multi-replica shared state" entries, so that wording no longer describes the current state.)

**Hot-reload semantics**: config-file changes are validated and logged, taking effect after restart (no runtime hot swap). On Kubernetes, roll updates via ConfigMap.

### Fixed

- Security: any credentials could become admin when the admin password was missing (P0); `/model/switch` path allowlist; request-path unwrap removal; `SecretKey` zeroization + read-only env keystore; test env-var races (shared ENV_LOCK)
- Performance: queue-enqueue wakeup replacing 5s backoff polling; worker drain batching; cache keys xxh3 + model namespace; `/search` batched inference
- Documentation: full correction of README/API_REFERENCE inconsistencies with the implementation; doc_consistency_check added to the gates
- Delivery: Dockerfile compatible with edition 2024; LICENSE completed; CI coverage hard gate + Python scenario tests nightly; clippy unwrap_used gate

### Added (translation-int8-optin: translation weight int8 quantization opt-in)

- **Quantization script**: `scripts/quantize-translate-bundles.py` (onnxruntime dynamic int8; graph I/O contract validated at generation); 3 config bundles measured at −34% per pair on the main set (encoder −75%, decoder limited by fp32 vocab residue), disk 6.4G→4.2G
- **Selection knob**: enable with `TRANSLATION_PREFER_INT8=1` (default fp32-first, zero contract change); `candidate_order` pure function with four-state unit test
- **Quality gates**: `test_int8_quality_gate_m2m100` (strict byte-for-byte equality; two sentences measured fully equal) / `test_int8_quality_gate_opus` (tolerance tier: prefix similarity ≥0.85 + tail extension ≤1.5×; both quantization-noise classes measured contained) — opus does not switch by default, pending the paper-translate-quality evaluation

### Added (wsl2-oom-hardening-r2: second hardening round)

- **`.wslconfig` kernel thresholds**: added `vm.min_free_kbytes=262144` (44MB → 256MB, moving the OOM decision earlier; effective after the next `wsl --shutdown`; original file backed up in temp/)
- **Generic process memory cap**: `scripts/run-memcap.sh` (systemd --user scope; the service self-destructs on abnormal growth to protect the host; MemoryMax/MemoryHigh/MemorySwapMax knobs, cgroup attributes and over-limit kills both measured)
- **jemalloc decay baked at compile time**: `.cargo/config (old name committed; config.toml gitignored) [env] JEMALLOC_SYS_WITH_MALLOC_CONF` → decay 1000ms (RSS falls back within seconds after large requests); the three injection-path exclusion evidence chain (mallctl absent at runtime / `je_malloc_conf` vetoed by hidden visibility) in specmark design D3; observed at startup via `effective_boot_conf()`

### Added (wsl2-oom-hardening: six defenses from the WSL2 OOM incident)

- **Session memory gate**: `build_session` checks MemAvailable before creating any ORT allocation (demand estimated as model file ×3/2), failing fast with an explicit error when insufficient (message includes available/need/floor and remediation advice), effective uniformly at all 6 engine call sites; knobs `SESSION_MEMORY_GUARD` (default on) / `SESSION_MEMORY_FLOOR_MB` (default 2048)
- **Translation parallel-worker lazy loading**: the extra Session set for sentence-level parallelism changed from startup preloading to on-demand build on the first long-input parallel request — idle residency dropped from ~30G+ (4 pairs × 4 workers all-FP32) to one set per pair; the P7 latency contract unchanged (`TRANSLATION_PARALLEL_WORKERS` semantics unchanged). Root cause and measured evidence in [MEMORY_SAFETY](../MEMORY_SAFETY.md)
- **Ops defenses (root-free)**: `scripts/build-memcap.sh` (systemd --user scope build memory cap, cgroup `memory.max` verified effective at the kernel level); `scripts/mem-guard.sh` + `mem-guard-install.sh` (user-level earlyoom killing target processes at a MemAvailable threshold, both directions measured, resident after install)

### Changed (deprecation-report retirement and maintenance-norm consolidation)

- **Deprecation reports removed from tracking**: root `DEPRECATED.md` and `transcribe-poc/DEPRECATED.md` deleted via `git rm` (historical commits keep them; no longer tracked); the source of truth for history and rulings is `specmark/archive/**` (read-only)
- **Added [🧹 Maintenance Norms](MAINTENANCE.md)**: consolidates the still-valid long-term constraints — the transcription consumption surface and the no-semantic-duplication rule, the `transcribe-own` CLI contract (stdout five lines including `TRANSCRIPT_CHARS`, `--vad-only` still writes an empty segments.txt, exit codes 0/1/2, logs always to stderr), the "explicit warning, no silence" discipline for retired switches, and the deprecation-document/errata governance meta-rules
- **Inbound links re-pointed**: both READMEs' "see DEPRECATED.md" now point to `transcribe-poc/CONTRACT.md` and [MAINTENANCE](MAINTENANCE.md); the root `AGENTS.md` (not committed per `.gitignore` convention) converged into a pointer to repository documents without duplicating normative text (avoiding a second source of truth)

### Changed (i18n-audit: full internationalization closure and default-English hardening)

- **Default-language constraint made verifiable**: `detect_locale()` extracted into the pure function `detect_from(get_env, sys_locale)` (priority chain and early-return semantics unchanged) with 5 new guard unit tests: a pristine environment must land on `en`; `C`/`POSIX`/unsupported values must never produce `zh` at any tier; `VECBOOST_LANG` overrides the POSIX chain and the system locale; the `en` pack has zero Chinese (comments included) and the `zh` side must contain Chinese
- **Behavior change (protocol surface follows locale)**: the end of `i18n::init()` adds `sync_sdforge_i18n()` — aligning sdforge's locale with this repo's `current_locale()` and registering the FTL `forge-*` keys into the sdforge registry (single source of truth). Previously `VECBOOST_LANG` could not reach CLI `--help`/MCP descriptions, which read `SDFORGE_LANG`/system locale themselves (dual-track); now `VECBOOST_LANG` is authoritative. All 9 `#[forge(cli = true)]` gained `i18n_key = "forge-cli-<name>"`
- **Hardcoded Chinese zeroed**: the `factory.rs` task=decision-without-onnx error now goes through `factory-decision-no-onnx-feature`; `transcribe-poc` gained `i18n::init()` (previously uninitialized, so any `tr()` fell back to bare key names) and all its stderr diagnostics are FTL-wired; **the stdout five-line contract is untouched verbatim** (`CONTRACT.md` unchanged)
- **User-visible copy fully internationalized**: 300+ hardcoded English literals across startup/assembly/engine/device/cache/audit/service/doctor/CLI extracted into FTL keys (en/zh in the same batch), growing the packs from 240 to 577 keys (`messages.ftl` 557 + `errors.ftl` 20); machine-grep-able `Tag: message` prefixes (`[thread-tune]`, `EmbeddingModule:`, `persist:`, `transcribe-own:`) and `PASS/WARN/FAIL`, `CheckResult.name`, JSON field names, and `error_code` stay English per the red line
- **New regression gate**: `scripts/i18n_consistency_check.py` (R1 default English, R2 en/zh key parity, R3 valid `tr()`/`i18n_key` references, R4 no hardcoded Chinese in production code, R5 non-internationalized-copy ratchet), wired into `.pre-commit-config.yaml` / `scripts/pre-commit.sh i18n` / CI `Check`; exemptions in `scripts/i18n_allowlist.json` (9 rules covering 10 sites, all with reasons: developer panic surfaces, test fixtures, benchmark instrumentation), hot-path caps in `scripts/i18n_baseline.json` (63 sites, only goes down)
- **Normative text**: [MAINTENANCE](MAINTENANCE.md) gained the "🌐 i18n constraints" section (default English, protocol-surface single source of truth, machine-output red lines, gate commands); [CONTRIBUTING](CONTRIBUTING.md) quality gates gained the i18n line
- **Drive-by fixes (non-i18n; new local-toolchain clippy interceptions)**: `api/transcribe.rs`'s `use serde::Deserialize` gated by `cli + transcribe` (otherwise feature combos without transcribe report an unused import); `engine/session.rs` removed the redundant `trim()` before `split_whitespace`

---

## [0.2.1] - 2026-07-30

### Added

- **i18n internationalization system**: full ICU+Fluent-compatible bilingual support (English/Chinese), including:
  - A lightweight FTL parser (`src/i18n/bundle.rs`), `include_str!` compile-time embedding, 114 translation keys (errors.ftl 17 + messages.ftl 97)
  - Three-tier locale priority: explicit locale > request level (`Accept-Language` middleware + tokio task_local) > global default
  - `VecboostError::IntoResponse` fully translated via `error_code()` + `tr_with_args()`
  - A manual `Display` impl using `tr_with_args()`, removing thiserror English prefixes
  - P0 (HTTP JSON hardcoding) + P1 (startup/OOM/metrics) + P2 (Display/engine) fully cleared
- **Rerank**: full support over HTTP/gRPC/CLI:
  - The `InferenceEngine` trait gained `rerank()` / `rerank_batch()` / `supports_rerank()` default implementations
  - `RerankService` (`src/service/rerank.rs`) + `RerankConfig`
  - `src/api/rerank.rs` forge/cli/grpc endpoints, growing 11 gRPC methods to include rerank/rerank_batch
- **Semantic cache**: `SemanticCache` (`src/cache/semantic_cache.rs`) with three-tier lookup: exact match → trigram Jaccard semantic search → model-inference backfill
- **BF16 precision inference**: `Precision::Bf16` support with the Candle engine BF16 path
- **SIMD vectorized similarity**: vectorized cosine similarity in `src/utils/vector.rs`
- **Continuous batching scheduler**: `ContinuousBatchLoop` (`src/device/continuous_batch.rs`) continuously collecting and processing batches
- **GPU memory paging**: `WeightPagingManager` (`src/device/memory_paging.rs`) with an LRU-K policy
- **Matryoshka adaptive dimensions**: `information_retention_rate()` + `recommended_dimension()` public APIs
- **Encrypted configuration module**: AES-256-GCM key management in `src/config/encryption.rs`
- **Auth subsystem refactor**: split into a multi-file interface-based architecture under `src/auth/` (middleware/types/csrf)
- **Scenario test suites**: 10 Python scenario tests in `tests/scenario/` + `scripts/run-scenario-tests.sh`
- **Library mode**: independent `schema` feature, separated HTTP feature gate, supporting non-HTTP library usage

### Changed

- **Dependency upgrades**: h2 0.4.15→0.4.19 (RUSTSEC-2026-0258 DoS fix), self-maintained libraries migrated to local paths
- **Rust 2024 migration**: let-chain syntax, `#[allow(dead_code)]` with reasons added
- **Code quality**: magic values extracted to constants, collection preallocation, JoinSet managed uniformly, block_on refactoring

### Fixed

- **Tokenizer Chinese segmentation**: loading the real model vocabulary on non-macOS platforms (30K+ words), WordPiece per-character UNK fallback, encode preserving actual characters
- **Error status mapping**: `to_api_error` now maps ValidationError→400, ModelLoadError→424, NotFound→404
- **Auth allowlist paths**: PUBLIC_PATHS aligned with the sdforge route prefix `/api/1/`
- **Rate-limit middleware**: injected RateLimitEnabled, mounted unconditionally, decoupled from auth
- **Prometheus metrics**: HTTP metrics middleware added + path normalization to prevent label cardinality bloat
- **gRPC startup**: SdforgeLimiteronAdapter 10s timeout protection
- **Graceful shutdown**: the config watcher handle saved and aborted on shutdown
- **HF Hub mirrors**: HF_ENDPOINT mirror detection + ETag incompatibility warning
- **schema feature compilation**: feature-gate fix; `schema` alone now compiles
- **Misc**: multi-byte safe slicing, fallback-path repo_id validation, CLI example parameter format

### Security

- **h2 DoS vulnerability**: RUSTSEC-2026-0258 fixed
- **vuln-0009**: HF Hub repo_id format validation
- **Security hardening**: cascade of auth/rate-limit/audit fixes, Python test-script security-scan fixes

---

## [0.2.0] - 2026-07-24

### Added

- **sdforge unified interface generation**: HTTP/gRPC/MCP/CLI all generated from the single source in `src/api/embedding.rs` via the `#[forge(...)]` macro, eliminating hand-written protocol code.
- **gRPC migrated to sdforge**: the hand-written tonic gRPC implementation replaced by `#[forge(grpc_method = "vecboost.*")]` macro wrappers, supporting 9 gRPC methods (embed/embed_batch/compute_similarity/embed_file/model_switch/get_current_model/get_model_info/list_models/health_check).
- **gRPC configuration keys**: new `grpc_max_connections`, `grpc_timeout_seconds`, `grpc_require_auth`, `grpc_allowed_roots` with JWT auth and rate limiting.
- **Bert/XlmRoberta model architecture support**: auto-detected via the `ModelArchitecture` enum.
- **Matryoshka truncation re-normalization**: `normalize_l2` called automatically after dimension truncation, keeping cosine similarity correct.
- **vuln-0009 security hardening**: HF Hub `repo_id` format validation centralized in `src/utils/hf_hub.rs` (`is_valid_hf_repo_id` + `build_hf_repo`), covering all remote download entry points.
- **Protocol-agnostic handlers**: `*_handler` functions extracted, eliminating code duplication across gRPC/HTTP/CLI.
- **Batch-size validation**: `validate_batch_size` bounded by `EmbeddingConfig.max_batch_size`.

### Changed

- **Dependency upgrades**: hf-hub 0.4→1.0, prometheus 0.13→0.14, tokio 1.52→1.53, aes-gcm 0.10→0.11.
- **Default feature simplification**: `default = ["http"]` (previously `["http", "oxcache", "limiteron"]`).
- **Mandatory dependencies**: `confers`, `inklog`, `oxcache`, `limiteron`, `trait-kit` always enabled, no feature needed.
- **sdforge local dependency**: temporarily `path = "../sdforge"` (v0.4.7, with tokio 1.53 support), switching back once sdforge 0.4.7 published to crates.io.
- **utoipa features explicit**: the `utoipa` dependency explicitly adds the `macros` feature (required by the `ToSchema` derive macro), per rule 25 "dependencies must be used through explicit features".

### Removed

- **Hand-written gRPC implementation**: deleted the `src/grpc/` and `proto/` directories.
- **Hand-written HTTP routing**: deleted the `src/routes/` directory.
- **Hand-written CLI**: deleted the `src/cli/` directory.
- **CandleEngine.tensor_pool**: removed the tensor-pool logic with the all-zero-mask bug.
- **MemoryPoolManager.tensor_pool dead code**: removed the consumer-less tensor-pool fields and methods.
- **tensorrt/openvino stub features**: removed unimplemented engine stubs.

### Fixed

- **schema feature failing to compile standalone**: utoipa 5.5.0's internal `schema.rs` references `crate::utoipa::Number` (gated by `cfg(feature = "macros")`), so enabling `schema` without `macros` failed. Explicitly adding the `macros` feature to `utoipa` fixed it.
- **http-dependent code not gated in schema-only mode**: 17 `test_into_response_*` tests, 3 `sanitize_error_message` boundary tests, the `use regex::Regex` import, the `src/lib.rs` tests module's http-only imports, and the `PrometheusCollectorModule` tests/helpers in `src/registry/tests.rs` lacked `#[cfg(feature = "http")]` gates, breaking `schema`-only builds. All gates added.
- **Config test env-var races**: `test_confers_load_from_minimal_toml` and `test_confers_toml_overrides_defaults` lacked `ENV_LOCK`, racing with `test_apply_security_env_overrides_*` over `VECBOOST_JWT_SECRET`. `ENV_LOCK` and env cleanup added.
- **Worker test flakiness**: `test_worker_loop_exits_immediately_when_not_running` checked `is_alive` immediately after `current_workers() == 0`, but `worker_loop` decrements the worker count before setting `is_alive = false` — a TOCTOU race. Now waits for `is_alive == false && current_workers() == 0` simultaneously.
- **bin target failing in schema-only mode**: `src/main.rs` is the HTTP server entry (`axum::serve`) and depends on the `http` feature, but `Cargo.toml` had no `required-features` for the `[[bin]]` target, so `cargo check --no-default-features --features schema` (with bin) failed. `[[bin]] required-features = ["http"]` added, matching the existing pattern in `examples/Cargo.toml`.
- **Matryoshka truncation without re-normalization**: truncated vectors were non-unit, breaking cosine similarity.
- **attention_mask tensor-pool bug**: the tensor-pool path had a TODO that never backfilled data, so batch inference masks were all zero.
- **Multi-byte slicing panics**: `sanitize_secret`, `sanitize_jwt_secret`, `mask_value`, and `text_preview` use `floor_char_boundary`/`ceil_char_boundary` for UTF-8-safe slicing.
- **Missing repo_id validation on fallback/onnx/recovery paths**: unified through `build_hf_repo`.

### Security

- **vuln-0009**: HF Hub `repo_id` format validation against malicious config injection and path traversal.
- **UTF-8 safe slicing**: all secret-masking and text-preview functions use character-boundary-safe slicing.
- **gRPC path validation**: `PathValidator` reads allowed roots from configuration, rejecting `/`, `/etc`, and other sensitive directories by default.

## [0.1.2] - 2026-01-16

- Added Matryoshka dimensionality reduction
- Added the OpenAI-compatible `/v1/embeddings` endpoint
- Cleaned up unused deployment configuration files

## [0.1.1] - 2026-01-15

- Security hardening phase 2 completed
- Improved error handling and enabled audit logging
- Tests support real inference (mock fallback)
- Documentation enhancements (Mermaid visualization) and cargo publish metadata

## [0.1.0] - 2026-01-10

Initial VecBoost release.

---

## 🧭 Roadmap

<table style="width:100%; border-collapse: collapse">
<tr><th style="text-align:center">Status</th><th style="text-align:left">Area</th><th style="text-align:left">Items</th></tr>
<tr><td align="center">✅</td><td>Core service</td><td>Four-protocol single-source generation, Candle/ONNX engines, Bert/XlmRoberta architectures, priority queue and time-window batching</td></tr>
<tr><td align="center">✅</td><td>Ecosystem integration</td><td>7-library ecosystem wiring (trait-kit registry, confers config, inklog logging, oxcache cache, limiteron rate limiting, dbnexus persistence, sdforge interfaces)</td></tr>
<tr><td align="center">✅</td><td>Security &amp; i18n</td><td>JWT/CSRF/RBAC/TOTP, hardened secure defaults, audit logging, ICU+Fluent bilingual errors</td></tr>
<tr><td align="center">✅</td><td>Performance foundation</td><td>SIMD similarity, continuous batching, semantic cache, GPU memory paging, BF16 inference, Matryoshka reduction</td></tr>
<tr><td align="center">✅</td><td>Audit remediation (0.3.0-rc.1)</td><td>Breaking behavioral changes: secure defaults / login convergence / XFF trust inversion / RBAC wiring / model-scoped cache keys (see [0.3.0-rc.1](#030-rc1---2026-09-22))</td></tr>
<tr><td align="center">✅</td><td>Tuning switches (0.3.0-rc.1)</td><td>GGUF quantization, quantized vector comparison, multi-model LFRU residency, cache WAL, hardware-aware planning, doctor diagnostics, startup warmup</td></tr>
<tr><td align="center">✅</td><td>Quantized inference backend</td><td>GGUF engine delivered via a load-time dequantization bridge (Q8_0/Q4_K; routing, magic validation and quality gate in place); the benefit is storage/load size, runtime is fp32-equivalent (no compute speedup); the quality gate needs model assets to run</td></tr>
<tr><td align="center">✅</td><td>Multi-replica session offload</td><td><code>[auth] session_backend = "database"</code> (deployment DB <code>garrison_kv</code>, garrison DbnexusKvDao); lockout counters are atomic via <code>AtomicKvDao</code> (next row)</td></tr>
<tr><td align="center">✅</td><td>Multi-replica shared state</td><td>Redis backends for rate limiting and the exact cache (<code>distributed-redis</code>, Lua-atomic fixed window), the semantic-cache fuzzy index (Redis Hash + LRU ZSET), cross-replica ban-list sharing (<code>ban-sync</code>), and atomic lockout counters (<code>AtomicKvDao</code>)</td></tr>
<tr><td align="center">✅</td><td>Performance baselines (throughput + MKL)</td><td>Throughput and MKL comparison baselines measured (2026-09-16, see <a href="PERFORMANCE.md">performance guide</a>)</td></tr>
<tr><td align="center">📋</td><td>Performance baseline (Accelerate)</td><td>Accelerate (macOS) pending verification: no Apple hardware available; the verification protocol is registered in the performance guide</td></tr>
<tr><td align="center">✅</td><td>Docs &amp; observability</td><td>ReDoc (<code>/redoc</code>, <code>swagger-ui</code> feature) and the preconfigured Grafana dashboard (<code>deploy/grafana/</code>)</td></tr>
</table>
