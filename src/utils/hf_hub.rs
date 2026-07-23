// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! HuggingFace repo ID 校验与仓库句柄构造（vuln-0009 修复）。
//!
//! 本模块统一所有远程下载入口的 repo_id 格式校验，防止路径遍历与恶意 repo ID 注入。
//! `engine` 与 `model` 两层均依赖此共享工具层，避免校验逻辑遗漏到 fallback/onnx/recovery 路径。

use crate::error::VecboostError;
use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeModel, split_id};

/// 验证 HuggingFace repo ID 格式(vuln-0009 修复)
///
/// 合法格式:`organization/model-name` 或单段 `model-name`,每段只允许
/// 字母、数字、`-`、`_`、`.`,不允许 `..`、`//`、开头/结尾的 `/`。
///
/// # 示例
/// - `BAAI/bge-m3` ✓
/// - `bert-base-uncased` ✓
/// - `../etc/passwd` ✗(包含 `..`)
/// - `/etc/passwd` ✗(以 `/` 开头)
/// - `org//model` ✗(包含 `//`)
pub fn is_valid_hf_repo_id(repo_id: &str) -> bool {
    if repo_id.is_empty() {
        return false;
    }

    // 不允许以 / 开头或结尾
    if repo_id.starts_with('/') || repo_id.ends_with('/') {
        return false;
    }

    // 不允许 .. 或 //
    if repo_id.contains("..") || repo_id.contains("//") {
        return false;
    }

    // 最多两段(organization/model)
    let segments: Vec<&str> = repo_id.split('/').collect();
    if segments.len() > 2 {
        return false;
    }

    // 每段只允许字母、数字、-、_、.,且不为空,且不为纯 "."
    segments.iter().all(|seg| {
        !seg.is_empty()
            && *seg != "."
            && seg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    })
}

/// 构建已校验的 HuggingFace model 仓库句柄（blocking）。
///
/// 统一 vuln-0009 的 repo_id 格式校验与 HFClientSync 构造，供所有远程下载入口复用，
/// 避免校验逻辑遗漏到 fallback/onnx/recovery 路径。
pub(crate) fn build_hf_repo(
    repo_id: &str,
) -> Result<HFRepositorySync<RepoTypeModel>, VecboostError> {
    if !is_valid_hf_repo_id(repo_id) {
        return Err(VecboostError::ModelLoadError(format!(
            "Invalid HuggingFace repo ID '{}': must match 'organization/model-name' \
             pattern with alphanumeric, dash, underscore, dot characters only",
            repo_id
        )));
    }
    let api = HFClientSync::new().map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
    let (owner, name) = split_id(repo_id);
    Ok(api.model(owner, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // is_valid_hf_repo_id 单元测试(vuln-0009 修复)
    // =========================================================================

    #[test]
    fn test_is_valid_hf_repo_id_valid_two_segments() {
        assert!(is_valid_hf_repo_id("BAAI/bge-m3"));
        assert!(is_valid_hf_repo_id(
            "sentence-transformers/all-MiniLM-L6-v2"
        ));
        assert!(is_valid_hf_repo_id("org/model_name"));
        assert!(is_valid_hf_repo_id("org/model.v2"));
    }

    #[test]
    fn test_is_valid_hf_repo_id_valid_single_segment() {
        assert!(is_valid_hf_repo_id("bert-base-uncased"));
        assert!(is_valid_hf_repo_id("gpt2"));
        assert!(is_valid_hf_repo_id("model_v1.2"));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_empty() {
        assert!(!is_valid_hf_repo_id(""));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_path_traversal() {
        // vuln-0009 核心:拒绝路径遍历尝试
        assert!(!is_valid_hf_repo_id("../etc/passwd"));
        assert!(!is_valid_hf_repo_id("org/../../etc/passwd"));
        assert!(!is_valid_hf_repo_id("./model"));
        assert!(!is_valid_hf_repo_id("org/.."));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_leading_trailing_slash() {
        assert!(!is_valid_hf_repo_id("/etc/passwd"));
        assert!(!is_valid_hf_repo_id("org/model/"));
        assert!(!is_valid_hf_repo_id("/"));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_double_slash() {
        assert!(!is_valid_hf_repo_id("org//model"));
        assert!(!is_valid_hf_repo_id("//model"));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_more_than_two_segments() {
        assert!(!is_valid_hf_repo_id("org/sub/model"));
        assert!(!is_valid_hf_repo_id("a/b/c/d"));
    }

    #[test]
    fn test_is_valid_hf_repo_id_rejects_special_chars() {
        assert!(!is_valid_hf_repo_id("org/model:name"));
        assert!(!is_valid_hf_repo_id("org/model@v1"));
        assert!(!is_valid_hf_repo_id("org/model name"));
        assert!(!is_valid_hf_repo_id("org/model$evil"));
    }
}
