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

/// 检测是否使用了非官方 HuggingFace 端点（如国内镜像）。
///
/// hf-hub 1.0.0 强制要求服务端返回 ETag 响应头，部分镜像站（如 hf-mirror.com）
/// 可能不提供该头部，导致下载失败（DEFECT-HUB-001）。
fn detect_mirror_risk() -> Option<String> {
    let endpoint = std::env::var("HF_ENDPOINT").ok()?;
    if endpoint.is_empty()
        || endpoint.contains("huggingface.co")
        || endpoint.contains("hf.co")
    {
        return None;
    }
    Some(endpoint)
}

/// 构建已校验的 HuggingFace model 仓库句柄（blocking）。
///
/// 统一 vuln-0009 的 repo_id 格式校验与 HFClientSync 构造，供所有远程下载入口复用，
/// 避免校验逻辑遗漏到 fallback/onnx/recovery 路径。
///
/// DEFECT-HUB-001：当 `HF_ENDPOINT` 指向非官方镜像时，hf-hub 1.0.0 可能因
/// 缺少 ETag 头部而下载失败。此函数提前检测并输出警告，建议用户使用本地模型路径。
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

    // DEFECT-HUB-001: 提前检测镜像端点并警告
    if let Some(endpoint) = detect_mirror_risk() {
        log::warn!(
            "HF_ENDPOINT={} detected — hf-hub 1.0.0 requires ETag headers which \
             some mirrors (e.g. hf-mirror.com) may not provide. If model download \
             fails with 'missing ETag header', pre-download the model manually and \
             set `model_path` in config to use local loading instead.",
            endpoint
        );
    }

    let api = HFClientSync::new().map_err(|e| {
        let msg = e.to_string();
        // 提供更具针对性的错误信息
        if msg.contains("ETag") || msg.contains("missing") {
            VecboostError::ModelLoadError(format!(
                "HuggingFace hub initialization failed: {}. \
                 If using HF_ENDPOINT mirror, it may be incompatible with hf-hub 1.0.0 \
                 (DEFECT-HUB-001). Workaround: pre-download the model and set \
                 `model_path` in config.",
                msg
            ))
        } else {
            VecboostError::ModelLoadError(msg)
        }
    })?;
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
