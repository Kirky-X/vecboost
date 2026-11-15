# 🔒 VecBoost Security Document

VecBoost treats security as a core design goal. This document describes VecBoost's security policy, vulnerability reporting process, built-in security mechanisms, and security best practices.

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents (click to expand)</summary>

- [Supported Versions](#-supported-versions)
- [Vulnerability Reporting Process](#-vulnerability-reporting-process)
- [Security Design Overview](#-security-design-overview)
- [Supply Chain and Security Gates](#-supply-chain-and-security-gates)
- [Known and Addressed Risks](#-known-and-addressed-risks)
- [Security Best Practices](#-security-best-practices)

</details>

---

## 📌 Supported Versions

| Version | Status | Notes |
|------|------|------|
| 0.3.0 / 0.3.1 (current) | ✅ Supported | i18n engine retirement (lightweight FTL parsing), self-maintained dependencies stabilized from rc; includes all rc-series capabilities (multi-replica shared semantic cache/ban list, effective `[database] max_connections`, externalized sessions, hardened secure defaults, RBAC wiring, XFF trust inversion — see [CHANGELOG](CHANGELOG.md)) |
| 0.3.0-rc.3 and earlier rc | ⛔ Superseded | rc.1 is the breaking-change release; read [CHANGELOG](CHANGELOG.md#030-rc1---2026-09-22) before upgrading |
| 0.2.1 | ✅ Supported | i18n, Rerank, semantic cache, AES-256-GCM config encryption |
| 0.2.0 | ⚠️ Critical fixes only | sdforge four protocols, 7-library ecosystem wiring |
| 0.1.0 | ❌ Not supported | Initial release |

### Minimum Supported Rust Version (MSRV)

VecBoost requires **Rust 1.91+** (edition 2024; the `rust-version` field in `Cargo.toml` is authoritative). Use `rustup update stable` to keep the toolchain current for security patches.

### Dependency Security

```bash
# Run the security audit (same commands as the CI Security Audit job)
cargo install cargo-audit --locked
cargo audit

# Dependency supply-chain gate (advisories/licenses/bans)
cargo deny check
```

- `cargo audit`: RustSec advisory scanning; exemption configuration is in `.cargo/audit.toml` (reason required, consistent with the `deny.toml` triage policy).
- `cargo deny check`: vulnerabilities must be fixed; unmaintained/yanked/unsound are informational warnings that do not block CI — review manually before release and record the triage conclusion.
- `gitleaks` rule set: `.gitleaks.toml` loads the gitleaks default detection rules via `[extend] useDefault = true` — **without this line the rule set is empty and the scan silently no-ops** (the singular `[[allowlist.rules]]` form was deprecated after 8.19; the current form is `[[allowlists]]` + `regexes` + `condition`).
- `gitleaks` exemption scope: both regex exemptions are **narrowed by path**; no globally effective rules remain — placeholder JWT shapes are limited to `docs/` / `tests/` / `examples/` / `src/security/`; `test_`-prefixed test values are limited to `tests/` / `examples/` / `docs/` / `benches/` (`.gitleaks.toml:21-25`). Real credentials of the same shapes inside `src/` are therefore still detected (e.g. `api_key = "testsecret"` in production source fails the scan).
- `gitleaks` history baseline: `.gitleaks-baseline.json` records one located historical test credential (`src/module_registry/tests.rs`, a file deleted from the repository that only exists in historical commits) so the weekly full scan is not blocked by it; new leaks still fail.
- New dependencies require approval: prefer actively maintained crates and enable feature gates on demand (every dependency in Cargo.toml uses `default-features = false` with an explicit feature list).

---

## 🐛 Vulnerability Reporting Process

### How to Report

**Please do not report security vulnerabilities through public issues.**

**Preferred channel**: submit reports through the GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy) private disclosure channel (repository page **Security → Report a vulnerability**).

**Fallback channel**: only if the preferred channel is unavailable, email <kirky-x@outlook.com> (recipient: Kirky.X) with `[security]` in the subject. Kirky.X will respond as soon as possible after receiving a report.

### What to Include

Please include where possible:

- Affected versions (`vecboost --version` or the git commit)
- Vulnerability type and impact scope (auth bypass / path traversal / injection / DoS / information disclosure, etc.)
- Reproduction steps or a PoC (config snippet, sample request)
- Mitigation suggestions (if any)

### Triage Principles

- Vulnerabilities must be fixed; advisories on unreachable paths (transitive / disabled features) may be recorded as exemptions with a reason.
- Fixes ship with semantic-version releases and are recorded in the `### Security` section of the [CHANGELOG](CHANGELOG.md) (precedents: RUSTSEC-2026-0258 h2 DoS, vuln-0009 HF Hub repo_id validation).

---

## 🔒 Security Design Overview

### Secure by Default

- The factory config sets `host = "127.0.0.1"` (loopback-only binding); binding a non-loopback address while `auth.enabled=false` **refuses to start**, and the `VECBOOST_ALLOW_INSECURE=1` escape hatch logs an ERROR (intended for trusted-network containers only).
- Factory `use_gpu = false`; requesting GPU without the feature logs a WARN and falls back to CPU, never silently.
- A missing explicit `--config` path fails fast with an error and exit code 2, never silently falling back to defaults.
- Unknown CLI subcommands error out (exit code 2) instead of silently starting the service.

### Authentication and Authorization (auth feature, garrison)

- JWT authentication (`protocol-jwt`) + CSRF protection (follows `auth.enabled`) + the web CORS/firewall family (bruteforce / ratelimit / ddos / anomalous).
- RBAC: `/api/v1/model/*` and `/embed/file` require the **admin role**.
- Login convergence: only `default_admin_username` (default `admin`) may log in; when auth is enabled, `VECBOOST_ADMIN_PASSWORD` (≥12 chars) must be configured or startup is refused.
- Token lifecycle: `token_expiration_hours` defaults to 24 hours; TOTP (`secure-totp`), account lockout (`account-lockout`), credential zeroization (`account-credential-zeroize`), anomaly detection (`anomalous-detector-dual`), and security alerts are supported.
- Session storage: `[auth] session_backend` selects `memory` (default, in-process oxcache, **single replica only**) or `database` (deployment table `garrison_kv` via garrison `DbnexusKvDao`, shareable across replicas when auth is enabled); `database` requires building with `--features "db,sqlite"` or `"db,postgres"` and configuring `[database]`; startup explicitly refuses when db is not compiled or auth is disabled.
- Multi-replica semantics boundary: with sessions externalized, authentication scales horizontally; **rate limiting, the exact vector cache, the semantic-cache fuzzy layer, and the ban list** can all be shared across replicas via Redis with `distributed-redis` (default `memory` stays in-process). The semantic-cache fuzzy layer uses the same trigram Jaccard semantics as the single-node mode, with the index stored in a Redis Hash + LRU ZSET (no RediSearch dependency); the ban list is shared via limiteron `ban-sync` + `CacheBanStorage`; account lockout counters increment atomically in a single statement via `AtomicKvDao` (`INSERT ... ON CONFLICT ... RETURNING`, SQLite/PostgreSQL; SQLite concurrent writes use bounded busy retry + WAL) with no loss across replicas; `set_if_absent` does not guarantee "exactly one winner" in the window where its expired-key cleanup path is the non-atomic DELETE→set two-step.
- `database` backend runtime prerequisites and session-data hygiene: TTL is based on each replica's UTC wall clock, so multi-replica deployments need NTP synchronization (clock skew stretches session and lockout durations equally); session tokens are stored as plaintext key-value rows — **backing up the deployment database is equivalent to backing up the session credential store and requires the same level of encrypted custody**; expired session rows are deleted lazily only when the same key is accessed again — schedule operational cleanup of expired `garrison_kv` rows (e.g. `DELETE FROM garrison_kv WHERE expire_at IS NOT NULL AND expire_at <= <unix seconds>`).

### Request Boundary Protection

- XFF trust inversion: when `trusted_proxies` is empty, `X-Forwarded-For` is ignored and the direct peer address is used (prevents header-forgery rate-limit bypass).
- Rate limiting: limiteron token bucket with independent global / per-IP / per-user / per-API-key counters, ban management (ban-manager) and GCRA support.
- Input limits: per-text maximum byte length (`max_text_length`, against resource exhaustion) and batch-size validation (`validate_batch_size`).

### File and Path Security

- `/embed/file` requires an explicitly configured `[server] grpc_allowed_roots` allowlist (no cwd fallback); single-file limit 10 MiB; `text_preview` is admin-only.
- `PathValidator` rejects sensitive directories such as `/` and `/etc` by default.
- HF Hub `repo_id` format validation (`is_valid_hf_repo_id` + `build_hf_repo` in `src/utils/hf_hub.rs`) covers all remote download entry points uniformly, guarding against malicious config injection and path traversal (vuln-0009).

### Data and Secret Protection

- AES-256-GCM configuration encryption (`src/config/encryption.rs`); `VECBOOST_ENCRYPTION_KEY` / `VECBOOST_REQUIRE_ENCRYPTION` / `VECBOOST_KEY_STORAGE_TYPE` / `VECBOOST_KEY_FILE_PATH` control key storage.
- `SecretKey` zeroization; the environment-variable keystore is read-only.
- All secret masking and text-preview functions use UTF-8 character-boundary-safe slicing (guards against multi-byte slicing panics).

### Logging and Observability

- Audit logging (audit section + inklog): user, action, resource, IP, timestamp.
- Error-response masking: `sanitize_error_message` normalizes cleanup; error codes are translated bilingually via i18n without leaking internal details.

---

## 🔗 Supply Chain and Security Gates

| Gate | Tool | Configuration |
|------|------|------|
| RustSec advisory scanning | `cargo audit` (CI + pre-release verification) | `.cargo/audit.toml` |
| Dependency license/ban/vulnerability gate | `cargo deny check` | `deny.toml` (license allowlist MIT/Apache-2.0/BSD/ISC etc.) |
| Static security analysis | CodeQL (Rust, push/PR/weekly) | `.github/workflows/codeql.yml` |
| Image vulnerability scanning | Trivy (fails on CRITICAL) | `.github/workflows/docker.yml` |
| Dockerfile scanning | Checkov | `.github/workflows/health-check.yml` |
| Secret scanning (incremental) | gitleaks (push/PR commit range) | `.github/workflows/security.yml` → `gitleaks/gitleaks-action`; version aligned with the pre-commit hook via `GITLEAKS_VERSION`. Runs only on push/PR (schedule/workflow_dispatch is covered by the full-scan job) |
| Secret scanning (history) | gitleaks (weekly full `--log-opts=--all`) | `.github/workflows/security.yml` → `gitleaks-history` job (Mondays 03:17 + manual) |
| Actions static audit | zizmor (first-round non-blocking baseline, becomes blocking after backlog is cleared) | `.github/workflows/security.yml` → `zizmor-action` (annotation mode) |
| Workflow syntax gate | actionlint (shellcheck at error level) | `.github/workflows/security.yml` → `rhysd/actionlint` (image pinned by digest) |
| Pre-commit checks | pre-commit (fmt/clippy/check/build + Apache-2.0 copyright header + gitleaks secret interception) | `.pre-commit-config.yaml` → `scripts/pre-commit.sh` |
| Pre-commit full secret scan | gitleaks (full history + baseline, local repo hook) | `.pre-commit-config.yaml` → `gitleaks-full-history`; the upstream hook only scans the staged area (`--staged`), and a manual `--all-files` run is effectively a no-op — the local hook adds the full scan |
| Panic-surface gate | `clippy::unwrap_used` + `RUSTFLAGS="-D warnings"` | CI env; tests exempt via `clippy.toml` |
| Python script scanning | bandit (production code src/ and scripts/ in full; idiomatic asserts in tests/ excluded as false positives) | `pyproject.toml` |

---

## ✅ Known and Addressed Risks

| Risk | Triage |
|------|------|
| RUSTSEC-2023-0071 (Marvin Attack, RSA) | Exempted and tracked: rsa reaches us transitively via garrison → jsonwebtoken for JWT signing/verification only; no RSA decryption path. No patched upstream version exists — see `deny.toml` / `.cargo/audit.toml` entries; lift the exemption and upgrade once an upstream constant-time implementation ships |
| RUSTSEC-2026-0258 (h2 DoS) | Fixed: h2 0.4.15 → 0.4.19 (v0.2.1) |
| vuln-0009 (HF Hub repo_id injection) | Fixed: `is_valid_hf_repo_id` validates all remote download entry points uniformly |
| Missing admin password allowing any credentials to become admin (P0) | Fixed (since 0.3.0-rc.1): startup refused when auth is enabled and `VECBOOST_ADMIN_PASSWORD` is missing |

---

## 🔒 Security Best Practices

**Production deployment**

- Always expose the service over HTTPS; configure `trusted_proxies` explicitly for reverse-proxy deployments.
- Enable `auth.enabled = true`; inject `VECBOOST_JWT_SECRET` / `VECBOOST_ADMIN_PASSWORD` via environment variables or a secret manager — never into the repo or the image.
- Evaluate `token_expiration_hours` as needed (default 24 hours); pure-Bearer APIs may disable CSRF explicitly.
- Do not use wildcard `"*"` for CORS (equivalent to a public API); list trusted origins explicitly.
- Horizontal scaling: with auth enabled, the default is a single replica (sessions in process memory); multiple replicas require `session_backend = "database"` and a `db,sqlite|postgres` build (sessions shared in the deployment database — backups are equivalent to the session credential store and need the same level of encrypted custody); rate limiting, the semantic cache, and the ban list are shared across replicas via Redis when `distributed-redis` is enabled, while the default `memory` stays per-replica.
- Trusted-network containers that need non-loopback binding must set `VECBOOST_ALLOW_INSECURE=1` explicitly and understand the meaning of the ERROR warning.

**Secrets and configuration**

- Rotate JWT secrets periodically; keep the `/embed/file` allowlist minimal (business directories only).
- Keep dependencies current with `cargo audit` / `cargo deny check`; every new exemption must state a reason.
- Clients must not rely on `/embed/file`'s `text_preview` (now admin-only).

---

## 📚 Related Documentation

| Document | Description |
|:-----|:-----|
| [📖 User Guide](USER_GUIDE.md) | Authentication configuration and deployment guidance |
| [📘 API Reference](API_REFERENCE.md) | Auth endpoints and error response format |
| [🏗️ Architecture](ARCHITECTURE.md) | Security architecture and error-handling design |
| [🧪 Test Scenarios](TEST_SCENARIOS.md) | Security scenario (R-auth-001~011) coverage |
| [📋 Changelog](CHANGELOG.md) | Version history of security fixes |
