// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 本地 bundle tokenizer 探测的单一事实源。
//!
//! onnx embedding 引擎（`onnx_engine::resolve_local_bundle`）与决策管线
//! （`decision` 的 `DecisionPipeline::load`）共用同一契约：
//! `[model].tokenizer_path` 显式路径 → bundle 根 `tokenizer.json` →
//! `tokenizer/tokenizer.json` 子目录。显式路径已配置但不存在时显性报错，
//! 禁止静默回落——静默换用 bundle 内其他 tokenizer 会在用户无感知下产生
//! 分词漂移（配置错误的信号被吞掉）。

use crate::error::VecboostError;
use std::path::{Path, PathBuf};

/// 按三级顺序解析 bundle tokenizer 路径。
///
/// - `explicit`（`ModelConfig.tokenizer_path`）已配置：命中即用；路径不存在
///   显性报错（不回落）。
/// - 未配置：bundle 根 `tokenizer.json` → `tokenizer/tokenizer.json`；
///   皆缺时报错并列出检查过的路径。
pub(crate) fn resolve_tokenizer_path(
    bundle_dir: &Path,
    explicit: Option<&Path>,
) -> Result<PathBuf, VecboostError> {
    if let Some(explicit_path) = explicit {
        if explicit_path.is_file() {
            return Ok(explicit_path.to_path_buf());
        }
        return Err(VecboostError::ModelLoadError(format!(
            "[model].tokenizer_path is configured but not found: {explicit_path:?}"
        )));
    }

    let root = bundle_dir.join("tokenizer.json");
    let subdir = bundle_dir.join("tokenizer").join("tokenizer.json");
    for candidate in [&root, &subdir] {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(VecboostError::ModelLoadError(format!(
        "Tokenizer not found: no [model].tokenizer_path configured; checked \
         {root:?} and {subdir:?}"
    )))
}
