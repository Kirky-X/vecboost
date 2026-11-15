# 🧹 VecBoost Maintenance Norms

This document consolidates maintenance constraints that remain valid **across changes**. It originates from the retired root-level deprecation report `DEPRECATED.md` (2026-10-10 errata version, removed from the repository once the thin-shell conversion was completed). The single source of truth for historical narrative and rulings is `specmark/archive/**` (archives are read-only, and `specmark/` is not committed); this document keeps only the rules that are still in force. The root `AGENTS.md` (agent entry pointer, not committed per `.gitignore` convention) only points to this document and does not duplicate its content.

## 📋 Table of Contents

- [Transcription capability consumption surface](#-transcription-capability-consumption-surface)
- [transcribe-own CLI contract](#-cli-contract-verbatim-gate-dependency-must-not-deviate)
- [Retired-switch handling discipline](#-retired-switch-handling-discipline-explicit-warning-no-silence)
- [Deprecation documents and errata governance](#-deprecation-documents-and-errata-governance)
- [i18n constraints](#-i18n-constraints-default-english-no-hardcoded-chinese)
- [Gate commands](#-gate-commands)

---

## 🚚 Transcription capability consumption surface

| Consumption mode | Entry point | Prerequisites |
|---|---|---|
| Server | `POST /api/v1/transcribe` | `[transcribe] enabled=true`, requires the `transcribe` feature |
| Library facade | `vecboost::transcribe::{open, read_wav_16k_mono}`; `TranscribeSession::{detect_segments, transcribe_segment}`; `TranscribeOpts` | Requires the `transcribe` feature |

- Inference implementations are only allowed inside `src/engine/transcribe/`. `transcribe-poc/` (standalone crate, binary `transcribe-own`, not a workspace member) is kept only as a CLI and benchmarking harness; **that directory is local-only and not committed** (2026-10-15 owner ruling: not published, unrelated to the main repository's feature surface, excluded via `.gitignore`); within the repository, this section and the CLI contract summary below constitute its normative surface.
- **No semantically duplicated implementations**: asr / vad / features / tokenizer / wav must not be re-implemented inside `transcribe-poc`. The historical stance of "keeping a degraded local implementation to preserve experimental knobs" is fully void.
- Segment-level parallelism is achieved by the caller holding multiple `TranscribeSession` instances (ort rc.13 forbids concurrent `Run` on one Session; each session owns one set); the facade does not build a thread pool / session pool.
- The public facade is the POC's consumption surface: **no extra openings** may be added to the facade for main-repository internal implementation details.
- The **committed source of truth for input format and the feature chain is the code itself**: `src/engine/transcribe/features.rs` (frame shift 160 / frame length 400 / n_fft=400 / mel 128 bins / Slaney scale and normalization / leading 80-sample centered alignment / global normalization) and `src/engine/transcribe/wav.rs`. WAV must be 16kHz / mono / 16-bit PCM; violations fail loudly, **no silent resampling**. The PoC-era design draft `docs-transcribe/DESIGN.md` (its former §6 was equivalent to this section) **became local-only and uncommitted on 2026-10-18** and must not be cited as a gate or implementation basis.

## 📜 CLI Contract (verbatim gate dependency, must not deviate)

Authoritative text: `transcribe-poc/CONTRACT.md` (the gate's own basis, not a migration target, must not be rewritten; **local file, not committed**) and the main spec `specmark/specs/transcribe-poc-cli/spec.md` (`R-transcribe-poc-cli-001`; `specmark/` exists only locally). Key points:

- **Invocation form**: `transcribe-own <wav> <outDir> [--models <dir>] [--vad-only]`; `--models` defaults to `assets/models` relative to the process CWD, passed explicitly at gate invocations.
- **stdout five lines**: during a full-pipeline run, print and **only** print `WALL_S=<float seconds>`, `RTF=<float>`, `SEGMENTS=<integer>`, `TRANSCRIPT_CHARS=<integer>`, `MODE=<single-line free text>`.
- **Line count follows spec / CONTRACT (five lines)**: archived drafts claiming a "four-line contract" (missing `TRANSCRIPT_CHARS`) are stale drafts; do not implement or accept against them.
- **`--vad-only`**: stdout carries only the single line `SEGMENTS=<integer>`, and `<outDir>/segments.txt` must still be written (as an empty file) — replicating the old binary's behavior; do not omit the file.
- **Output files**: `<outDir>/transcript.txt` (full text concatenating each segment) and `<outDir>/segments.txt` (one line per segment: `start -- end: text`, start/end as float seconds).
- **Log split**: progress and diagnostics go to stderr only; stdout must carry no output besides the contract lines. `MODE=` is free text but must be a single line without newlines, and **must not backfill retired knob fields**.
- **Exit codes**: `0` success / `2` usage error / `1` run failure (reason on stderr).
- **Runtime dependency**: `ORT_DYLIB_PATH` points to `3rdparty/libonnxruntime.so.1.28.2` (ort loads via `load-dynamic`); the POC dependency whitelist is only `vecboost` (`transcribe` feature) + `pyo3` (py-torch baseline cross-check `run_pytorch_baseline`).

## 🔇 Retired-switch handling discipline: explicit warning, no silence

- `TRANSCRIBE_ENV_POOL` / `TRANSCRIBE_BATCH` / `TRANSCRIBE_GPU` retired with the POC experiment files and have no implementation anymore; `TranscribeOpts` keeps only `intra_threads` / `no_spin` / `cuda` / `vad_threshold`.
- When a retired switch is still set, it must be ignored after a loud stderr warning (implementation: `warn_retired_knobs` in `transcribe-poc/src/main.rs`); **silently swallowing it is forbidden**. The same discipline applies to non-contract input formats and unavailable CUDA EP: always fail loudly.
- Experimental conclusions are recorded in `out/transcribe-poc/own-engine/summary.md` §11 (io-binding non-deterministic read-back; measured batching and env-pool settings). Note that `out/` and `temp/` are both in `.gitignore` (not committed, locally visible only): any gate that uses them as acceptance evidence must prepare a branch for missing files and must not assume their existence in CI.

## 🧹 Deprecation documents and errata governance

- **Archives are read-only**: once written, `specmark/archive/**` is never rewritten. When an archived draft contradicts the implementation/contract, corrections land in living documents (main spec `specmark/specs/**` or `CONTRACT.md`); the archived text stays as-is.
- **No deprecation reports at the repository root**: one-shot retirement notes go into the archive with their change; only constraints valid across changes are distilled into this document. The root keeps no `DEPRECATED.md`-style historical narrative documents.
- **Invalid criteria are void**: once the premise behind a historical downgrade/retention rationale disappears (e.g. "the main crate had no public facade back then, so a local implementation was kept"), that criterion must not be used for later decisions; re-argument must cite the original decision in the archived `design.md`.
- **Check inbound links before deleting**: before removing a document or file, `git grep` the inbound links and fix them in sync (`README.md` / `README_CN.md` / `docs/**` / `src/**` doc comments / `.gitignore` comments) to avoid dead links; only delete artifacts with no confirmed consumers.
- **Design documents must not be committed** (2026-10-18 owner ruling): descriptive documents such as PoC-era design notes and feasibility assessments must not appear in the git repository (same stance as local-only documents like `docs/TRAIT-KIT-SDK-ASSESSMENT.md`); they are excluded via `.gitignore` and untracked with `git rm --cached`, kept locally. Their still-valid conclusions must be distilled into this document or into code doc comments; "the design draft is committed" must never serve as a source of truth.
- **Related artifacts you must not delete by mistake**: `config-e2e-transcribe.toml` (WAV-pipeline acceptance configuration) is an in-use committed file. The `transcribe-poc/` directory as a whole (contract text, pytorch baseline cross-check) and `docs-transcribe/` (PoC-era design drafts, containing external-machine paths and line numbers of uncommitted reference sources) **are both local-only uncommitted directories** — they are local assets, not discardable items; do not delete them from disk and do not re-add them to git.

## ✅ Gate commands

```bash
# transcribe-poc is a local uncommitted directory; the torch backend build needs PYO3_PYTHON pointing to cpython-3.11
cargo check --manifest-path transcribe-poc/Cargo.toml --all-targets
cargo clippy --manifest-path transcribe-poc/Cargo.toml --all-targets -- -D warnings
# doc_consistency_check.py runs six classes of verification: routes / gRPC / env vars / CLI subcommands / **document anchors** / stale API-prefix regression
# (the anchor item validates `](#frag)` and `](path#frag)` reachability against the measured-calibrated
# GitHub slug rules, and forbids headings retaining U+FE0F variation selectors)
python3 scripts/doc_consistency_check.py
python3 scripts/i18n_consistency_check.py
```

## 🌐 i18n constraints (default English, no hardcoded Chinese)

- **The default display language is strictly English** (`en`): the fallback chain of `detect_locale()`
  `VECBOOST_LANG → LC_ALL → LC_MESSAGES → LANG → sys-locale → "en"` must not be changed to always return zh;
  `locales/en/*.ftl` must contain zero Chinese (comments included). Chinese may only be triggered by an explicit
  environment variable or `Accept-Language`.
- **Single source of truth for the protocol surface**: the end of `i18n::init()` calls `sync_sdforge_i18n()`,
  registering this repository's FTL `forge-*` keys into the sdforge registry and aligning the locale;
  `SDFORGE_LANG` must not inversely decide the application-side language. New `#[forge(cli = true, …)]` must carry
  `i18n_key = "forge-cli-<name>"` (without the key, `--help` does not follow the locale).
- **Machine output that i18n must not rewrite**: the `transcribe-own` stdout five-line contract, grep-able tokens such as
  `PASS/WARN/FAIL` and `CheckResult.name`, HTTP JSON field names, `error_code`, and the Prometheus text format.
- **Regression interception**: `scripts/i18n_consistency_check.py` (R1 default English, R2 en/zh key parity,
  R3 valid `tr()`/`i18n_key` references, R4 no hardcoded Chinese in production code, R5 ratchet for non-internationalized copy);
  exemptions are recorded in `scripts/i18n_allowlist.json` (reason required), hot-path caps in `scripts/i18n_baseline.json` (only goes down).
