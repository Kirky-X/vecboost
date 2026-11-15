# 🤝 VecBoost Contributing Guide

Thanks for your interest in **VecBoost**! This document describes the local development workflow conventions and commit standards. Whether you are fixing defects, adding new features, improving documentation, or helping others, your contribution is valuable.

## 📋 Table of Contents

<details open>
<summary>📑 Table of Contents (click to expand)</summary>

- [Prerequisites](#-prerequisites)
- [Building](#-building)
- [Testing](#-testing)
- [Quality Gates (all must pass before committing)](#-quality-gates-all-must-pass-before-committing)
- [Change Management (specmark)](#-change-management-specmark)
- [Commit Conventions](#-commit-conventions)
- [Documentation Conventions](#-documentation-conventions)
- [Security](#-security)

</details>

---

## 🧰 Prerequisites

- Rust ≥ 1.91 (edition 2024; the `rust-version` field in the repo's `Cargo.toml` is authoritative)
- Python ≥ 3.10 + pytest (scenario tests)
- Optional: protobuf-compiler (gRPC E2E), docker

## 🧱 Building

```bash
# Library + binary (default http + onnx features)
cargo check -p vecboost -p vecboost-examples --all-targets

# Common feature combinations
cargo check -p vecboost --features http,grpc,cli,auth,db,sqlite,onnx
cargo check -p vecboost --features mcp
```

> **Dependency-source note**: the 7 ecosystem libraries (trait-kit / confers / inklog / oxcache / limiteron / dbnexus / sdforge) are currently sourced entirely from the crates.io registry (`Cargo.toml` has no `path =` dependencies and no `[patch]`; `Cargo.lock` records `source = "registry+https://github.com/rust-lang/crates.io-index"`).
> fmt/clippy/test commands should still be scoped with `-p vecboost -p vecboost-examples`
> to avoid pulling other workspace members' intermediate states into the gates.

## 🧪 Testing

| Layer | Command | Notes |
|---|---|---|
| Unit + integration (Rust) | `cargo test -p vecboost -p vecboost-examples` | 2363 inline test attributes + tests/ (130 test attributes, same counting method) |
| gRPC E2E | `cargo test -p vecboost --features http,grpc --test grpc_e2e` | Spawns the real binary |
| Scenario (Python) | `pytest tests/scenario -q` | conftest spawns real servers; inference cases auto-SKIP when `models/` is absent |
| Microbenchmarks | `cargo bench -p vecboost` | criterion |

The responsibility split of the three-layer performance test stack and the exhaustive scenario matrix are in [docs/TEST_SCENARIOS.md](TEST_SCENARIOS.md).

### Worktree development: preparing test assets

Large assets (`models/`, `3rdparty/`, `assets/`) are not committed (gitignore);
related tests (real-device inference, scenario suites, ORT dynamic library) SKIP via guards when assets are absent.
To actually run these tests in a git worktree:

```bash
bash scripts/link-assets.sh   # auto-detects the main workspace and symlinks the three asset types
# Remove: rm models 3rdparty assets
```

Note: the `models/` pattern in `.gitignore` (trailing slash matches directories only) does not
match symbolic links — `git status` will show them as untracked; do not `git add` them.

## ✅ Quality Gates (all must pass before committing)

```bash
cargo fmt --check -p vecboost -p vecboost-examples
cargo clippy -p vecboost -p vecboost-examples --all-targets -- -D warnings -W clippy::unwrap_used
cargo test -p vecboost -p vecboost-examples
python3 scripts/doc_consistency_check.py   # after modifying README/API
python3 scripts/i18n_consistency_check.py  # after modifying user-visible copy/language packs
```

- `clippy.toml` enables `allow-unwrap-in-tests`: production code forbids `unwrap/expect` panic surfaces, tests are exempt.
- `deny.toml` is the cargo-deny configuration (advisories/licenses/bans); new exemptions must state a reason.
- i18n: **the default display language is strictly English**; user-visible copy (errors/logs/CLI help) must go through
  the `src/i18n/locales/{en,zh}/*.ftl` keys; hardcoded Chinese is forbidden; the constraint text is in [MAINTENANCE](MAINTENANCE.md).
  Exemptions and baselines are recorded in `scripts/i18n_allowlist.json` / `scripts/i18n_baseline.json` (the latter only goes down).
- pre-commit hooks (`.pre-commit-config.yaml` → `scripts/pre-commit.sh`): fmt / clippy / check / build + Apache-2.0 copyright header check.

## 🔄 Change Management (specmark)

Multi-step changes go through the `specmark/` workflow: propose(proposal/design/tasks/specs)→ analyze → apply → converge → archive. The `specmark/` directory is not committed (gitignore); archive directories are read-only — if an archived draft contradicts the implementation, corrections land in living documents (main spec or contract files); archives are never rewritten.

## 📝 Commit Conventions

- Format: `type(scope): summary`, type ∈ feat/fix/test/docs/chore/refactor/perf
- One commit per logical change; behavioral changes (defaults, API contracts) must be registered in the CHANGELOG
  `Unreleased` section and mirrored in both README versions

## 📚 Documentation Conventions

- `docs/cn/` (Chinese) and `docs/en/` (English) use flat UPPER_SNAKE naming; files correspond one-to-one across the two directories: `USER_GUIDE.md`, `API_REFERENCE.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `FAQ.md`, `PERFORMANCE.md`, `SECURITY.md`, `TEST_SCENARIOS.md`, `MAINTENANCE.md`.
- Internal structure of each document: `# <emoji> <title>` → `## 📋 Table of Contents` → emoji sections.
- After modifying README/API, run `python3 scripts/doc_consistency_check.py` to keep documentation and implementation consistent.
- Performance numbers must come from the measured `docs/benchmarks/` archive or state the measurement method; fabricating numbers is forbidden; experimental conclusions are written back to [docs/PERFORMANCE.md](PERFORMANCE.md).
- Cross-change standing constraints (transcription consumption surface, `transcribe-own` CLI contract, retired-switch discipline, deprecation-doc governance) are consolidated in [🧹 Maintenance Norms](MAINTENANCE.md); the repository root keeps no `DEPRECATED.md`-style historical narrative deprecation reports — one-shot retirement notes go into the archive with their change.

## 🔐 Security

Do not report security vulnerabilities through public issues; prefer GitHub [Security Advisories](https://github.com/Kirky-X/vecboost/security/policy) (fallback: contact Kirky.X at <kirky-x@outlook.com>); the process is in [docs/SECURITY.md](SECURITY.md).
Dependency advisories are gated by CI `cargo-audit`; exemption configuration is in `.cargo/audit.toml` (reason required).
