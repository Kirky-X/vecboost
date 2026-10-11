// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 公共 ONNX Runtime Session 构建层（onnx-engine-unify D1）。
//!
//! 收编 `onnx_engine` / `decision` / `onnx_translate` 三处曾各自漂移的
//! `Session::builder()` 链，以及成对复制的 `probe_model_file` /
//! `precision_for_model_file`，使 EP/优化档位/线程策略单点可调。主仓（非 POC）
//! 内除本文件外不得再出现 `Session::builder()`（R-engine-inference-001）。
//!
//! 各调用点的**引擎专属**前置校验（如 `onnx_engine` 的 SHA256 完整性校验、
//! AMD/ROCm warn 分支）不属 builder 链，保留在调用侧——本层只做公共核。

use std::path::{Path, PathBuf};

use crate::config::model::{DeviceType, Precision};
use crate::error::VecboostError;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Session 构建选项：三处调用点的差异全部显式化于此，杜绝隐藏分叉。
pub(crate) struct SessionOpts<'a> {
    /// intra 线程数；`0` = 未设置 → 回退 [`resolve_intra_threads`]（全核）；
    /// `> 0` 透传。翻译侧经 `SESSION_INTRA_THREADS` 预算读入此字段。
    pub intra_threads: usize,
    /// 执行设备；仅 [`DeviceType::Cuda`] 且启用 `cuda` feature 时注册 CUDA EP，
    /// 否则 warn 降 CPU（不报错），与收编前三处现状一致。
    pub device: DeviceType,
    /// ort session config 逐条透传 `with_config_entry`（如转录 POC 的
    /// `("session.intra_op.allow_spinning", "0")`）；主仓三引擎现状传空切片。
    pub extra_config: &'a [(&'a str, &'a str)],
}

/// intra 线程数解析（纯函数，离线可测）：`0` → `available_parallelism`（兜底 4），
/// 非 `0` → 原值透传。与 `onnx_engine.rs`/`decision.rs` 收编前语义逐字一致。
fn resolve_intra_threads(intra_threads: usize) -> usize {
    if intra_threads > 0 {
        intra_threads
    } else {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

/// 构建并 `commit_from_file` 一个 Level3 优化、指定 intra 线程与 EP 策略的
/// Session。所有 ort 构建错误统一映射 [`VecboostError::ModelLoadError`]。
pub(crate) fn build_session(
    model_file: &Path,
    opts: &SessionOpts<'_>,
) -> Result<Session, VecboostError> {
    // 内存可用量门禁（wsl2-oom-hardening D1）：置于 builder 构造之前，
    // 拒绝时不产生任何 ORT 分配
    ensure_memory_available(model_file)?;

    let num_threads = resolve_intra_threads(opts.intra_threads);
    let mut builder = Session::builder()
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        .with_intra_threads(num_threads)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    for (key, value) in opts.extra_config {
        builder = builder
            .with_config_entry(*key, *value)
            .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    }

    let mut session = if opts.device == DeviceType::Cuda {
        log::info!("Attempting to configure CUDA execution provider");
        #[cfg(feature = "cuda")]
        {
            builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| VecboostError::ModelLoadError(e.to_string()))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!(
                "CUDA execution provider requires the cuda feature flag; \
                 using CPU execution provider"
            );
            builder
        }
    } else {
        builder
    };

    session
        .commit_from_file(model_file)
        .map_err(|e| VecboostError::ModelLoadError(e.to_string()))
}

/// bundle 内模型文件探测：`candidates` 按序首中返回其绝对路径；全 miss 返回
/// `ModelLoadError`，消息含 `role` 与全部尝试路径（决策/翻译两处成对复制的
/// 收编，R-engine-inference-002）。
pub(crate) fn probe_model_file(
    bundle_dir: &Path,
    candidates: &[&str],
    role: &str,
) -> Result<PathBuf, VecboostError> {
    let mut tried = Vec::with_capacity(candidates.len());
    for name in candidates {
        let candidate = bundle_dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
        tried.push(candidate.display().to_string());
    }
    Err(VecboostError::ModelLoadError(format!(
        "No {role} model found in {}; tried: {}",
        bundle_dir.display(),
        tried.join(", ")
    )))
}

/// 探测命中的模型文件名 → 对外精度标签：文件名含 `quantized` 或 `int8` 判
/// [`Precision::Int8`]，否则 [`Precision::Fp32`]（避免 quantized/int8 bundle
/// 对外失真报 Fp32）。
pub(crate) fn precision_for_model_file(model_file: &Path) -> Precision {
    let name = model_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if name.contains("quantized") || name.contains("int8") {
        Precision::Int8
    } else {
        Precision::Fp32
    }
}

/// 内存地板缺省值（字节）：`SESSION_MEMORY_FLOOR_MB` 未设/非法时的可用量下限。
const DEFAULT_MEMORY_FLOOR_MB: u64 = 2048;

/// 门禁开关（纯函数）：`SESSION_MEMORY_GUARD` 读出值等于 `off` 才关停，其余（含未设）恒开。
fn guard_switch(raw: Option<&str>) -> bool {
    raw != Some("off")
}

/// 地板解析（纯函数）：未设 → 默认；非数字 → 默认（warn）；合法 → MB 转字节。
fn parse_floor_mb(raw: Option<&str>) -> u64 {
    let mb = match raw {
        Some(v) => v.trim().parse::<u64>().unwrap_or_else(|_| {
            log::warn!(
                "SESSION_MEMORY_FLOOR_MB={v:?} is not a valid integer; \
                 falling back to default {DEFAULT_MEMORY_FLOOR_MB}MB"
            );
            DEFAULT_MEMORY_FLOOR_MB
        }),
        None => DEFAULT_MEMORY_FLOOR_MB,
    };
    mb.saturating_mul(1024 * 1024)
}

/// /proc/meminfo 文本 → MemAvailable 字节数；行缺失或数值非法返回 None。
fn mem_available_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// 门禁判定（纯函数）：`available < need + floor` 拒绝，消息含三个数字与处置建议。
fn memory_gate_decision(available: u64, need: u64, floor: u64) -> Result<(), String> {
    let required = need.saturating_add(floor);
    if available >= required {
        return Ok(());
    }
    Err(format!(
        "memory gate refused model load: available={available} bytes < need={need} + floor={floor} \
         = {required} bytes; free memory / reduce translation pairs or workers \
         (TRANSLATION_PARALLEL_WORKERS), or set SESSION_MEMORY_GUARD=off to bypass"
    ))
}

/// 加载前内存可用量门禁（wsl2-oom-hardening D1）：需求按模型文件 ×3/2 估算
/// （ORT 加载峰值≈2×文件、驻留≈1.1×文件，取中并留余量）。Linux 读
/// /proc/meminfo 的 MemAvailable；其他平台 no-op（尽力而为语义）。
/// 拒绝即 ModelLoadError fail-fast，杜绝静默加载到系统级 OOM。
pub(crate) fn ensure_memory_available(model_file: &Path) -> Result<(), VecboostError> {
    if !guard_switch(std::env::var("SESSION_MEMORY_GUARD").ok().as_deref()) {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
            VecboostError::ModelLoadError(format!("memory gate: cannot read /proc/meminfo: {e}"))
        })?;
        let available = mem_available_bytes(&text).ok_or_else(|| {
            VecboostError::ModelLoadError(
                "memory gate: MemAvailable missing in /proc/meminfo".to_string(),
            )
        })?;
        let need = std::fs::metadata(model_file)
            .map(|m| m.len())
            .unwrap_or(0)
            .saturating_mul(3)
            / 2;
        let floor = parse_floor_mb(std::env::var("SESSION_MEMORY_FLOOR_MB").ok().as_deref());
        match memory_gate_decision(available, need, floor) {
            Ok(()) => {
                log::info!(
                    "memory gate passed: model={} need={need} available={available} floor={floor}",
                    model_file.display()
                );
                Ok(())
            }
            Err(msg) => {
                log::error!("{msg} (model={})", model_file.display());
                Err(VecboostError::ModelLoadError(msg))
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        log::debug!("memory gate skipped: MemAvailable probe is Linux-only");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_switch_only_off_disables() {
        assert!(!guard_switch(Some("off")));
        assert!(guard_switch(Some("on")));
        assert!(guard_switch(Some("OFF"))); // 大小写敏感：非精确 `off` 即开
        assert!(guard_switch(None));
    }

    #[test]
    fn parse_floor_mb_default_on_missing_or_invalid() {
        assert_eq!(parse_floor_mb(None), DEFAULT_MEMORY_FLOOR_MB * 1024 * 1024);
        assert_eq!(
            parse_floor_mb(Some("not-a-number")),
            DEFAULT_MEMORY_FLOOR_MB * 1024 * 1024
        );
    }

    #[test]
    fn parse_floor_mb_valid_value_converts_to_bytes() {
        assert_eq!(parse_floor_mb(Some("1")), 1024 * 1024);
        assert_eq!(parse_floor_mb(Some(" 4096 ")), 4096 * 1024 * 1024);
    }

    #[test]
    fn parse_floor_mb_huge_value_saturates_without_panic() {
        assert_eq!(parse_floor_mb(Some("18446744073709551615")), u64::MAX);
    }

    #[test]
    fn mem_available_parses_among_other_lines() {
        let sample = "MemTotal:       69975212 kB\nMemFree:         1234567 kB\n\
                      MemAvailable:   62368464 kB\nSwapTotal:      17825792 kB\n";
        assert_eq!(mem_available_bytes(sample), Some(62368464 * 1024));
    }

    #[test]
    fn mem_available_missing_line_returns_none() {
        assert_eq!(mem_available_bytes("MemTotal: 100 kB\n"), None);
        assert_eq!(mem_available_bytes(""), None);
    }

    #[test]
    fn mem_available_invalid_number_returns_none() {
        assert_eq!(mem_available_bytes("MemAvailable: abc kB\n"), None);
    }

    #[test]
    fn gate_decision_refuses_below_need_plus_floor() {
        let err = memory_gate_decision(1000, 800, 300).unwrap_err();
        assert!(err.contains("available=1000"));
        assert!(err.contains("need=800"));
        assert!(err.contains("floor=300"));
    }

    #[test]
    fn gate_decision_allows_at_exact_boundary() {
        assert!(memory_gate_decision(1100, 800, 300).is_ok());
        assert!(memory_gate_decision(1200, 800, 300).is_ok());
    }

    #[test]
    fn resolve_intra_threads_zero_falls_back_to_parallelism() {
        // 0 = 未设置 → available_parallelism（≥1），非硬编码
        let n = resolve_intra_threads(0);
        assert!(n >= 1, "回退线程数应 ≥1，got {n}");
    }

    #[test]
    fn resolve_intra_threads_non_zero_passthrough() {
        assert_eq!(resolve_intra_threads(3), 3);
        assert_eq!(resolve_intra_threads(128), 128);
    }

    #[test]
    fn probe_model_file_returns_first_hit_in_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"x").unwrap();
        std::fs::write(dir.path().join("laya.onnx"), b"y").unwrap();
        // 按候选序首中：model.onnx 在 laya.onnx 之前 → 命中 model.onnx
        let hit = probe_model_file(
            dir.path(),
            &["model.onnx", "laya.onnx", "missing.onnx"],
            "test",
        )
        .unwrap();
        assert_eq!(hit.file_name().unwrap(), "model.onnx");
    }

    #[test]
    fn probe_model_file_miss_lists_role_and_all_tried_paths() {
        let dir = tempfile::tempdir().unwrap();
        let err = probe_model_file(dir.path(), &["a.onnx", "b.onnx"], "Laya ONNX").unwrap_err();
        let msg = err.error_detail();
        assert!(msg.contains("Laya ONNX"), "消息须含 role，got {msg}");
        assert!(
            msg.contains("a.onnx") && msg.contains("b.onnx"),
            "须含全部尝试路径，got {msg}"
        );
    }

    #[test]
    fn precision_detects_quantized_and_int8() {
        assert_eq!(
            precision_for_model_file(Path::new("m/model_quantized.onnx")),
            Precision::Int8
        );
        assert_eq!(
            precision_for_model_file(Path::new("m/laya_int8.onnx")),
            Precision::Int8
        );
        assert_eq!(
            precision_for_model_file(Path::new("m/model.onnx")),
            Precision::Fp32
        );
    }
}
