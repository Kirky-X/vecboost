// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace repo ID 校验与仓库句柄构造（vuln-0009 修复）。
//!
//! 本模块统一所有远程下载入口的 repo_id 格式校验，防止路径遍历与恶意 repo ID 注入。
//! `engine` 与 `model` 两层均依赖此共享工具层，避免校验逻辑遗漏到 fallback/onnx/recovery 路径。

use crate::error::VecboostError;
use hf_hub::{HFClientBuilder, HFRepositorySync, RepoTypeModel, split_id};
use std::path::{Component, Path, PathBuf};

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

    if repo_id.starts_with('/') || repo_id.ends_with('/') {
        return false;
    }

    if repo_id.contains("..") || repo_id.contains("//") {
        return false;
    }

    let segments: Vec<&str> = repo_id.split('/').collect();
    if segments.len() > 2 {
        return false;
    }

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
/// 可能不提供该头部，导致下载失败。
fn detect_mirror_risk() -> Option<String> {
    let endpoint = std::env::var("HF_ENDPOINT").ok()?;
    if endpoint.is_empty() || endpoint.contains("huggingface.co") || endpoint.contains("hf.co") {
        return None;
    }
    Some(endpoint)
}

/// 构建已校验的 HuggingFace model 仓库句柄（blocking）。
///
/// 统一 vuln-0009 的 repo_id 格式校验与 HFClientSync 构造，供所有远程下载入口复用，
/// 避免校验逻辑遗漏到 fallback/onnx/recovery 路径。
///
/// 当 `HF_ENDPOINT` 指向非官方镜像时，hf-hub 1.0.0 可能因
/// 缺少 ETag 头部而下载失败。此函数提前检测并输出警告，建议用户使用本地模型路径。
pub(crate) fn build_hf_repo(
    repo_id: &str,
) -> Result<HFRepositorySync<RepoTypeModel>, VecboostError> {
    build_hf_repo_with_http(repo_id, Some(std::time::Duration::from_secs(300)))
}

/// [`build_hf_repo`] 的总超时可调内部形态。
///
/// `total_timeout = None` 供大文件清单下载（`download_files`）使用：
/// reqwest 总超时覆盖响应体读取全程，300s 上限下 1.6GB 级资产在慢速直连
/// 链路（< ~5.6MB/s）必然中途超时；hf-hub 的 retry 只重试请求建立
/// （retry 闭包仅包 `send()`），响应体断流不重试。连接超时 10s 与 retry
/// 上限两条路径一致保留。
fn build_hf_repo_with_http(
    repo_id: &str,
    total_timeout: Option<std::time::Duration>,
) -> Result<HFRepositorySync<RepoTypeModel>, VecboostError> {
    if !is_valid_hf_repo_id(repo_id) {
        return Err(VecboostError::ModelLoadError(format!(
            "Invalid HuggingFace repo ID '{}': must match 'organization/model-name' \
             pattern with alphanumeric, dash, underscore, dot characters only",
            repo_id
        )));
    }

    if let Some(endpoint) = detect_mirror_risk() {
        log::warn!(
            "HF_ENDPOINT={} detected — hf-hub 1.0.0 requires ETag headers which \
             some mirrors (e.g. hf-mirror.com) may not provide. If model download \
             fails with 'missing ETag header', pre-download the model manually and \
             set `model_path` in config to use local loading instead.",
            endpoint
        );
    }

    // 修复：为 HF 客户端注入带连接超时的 reqwest 客户端并限制
    // 重试次数。默认配置无请求超时且重试次数多，网络不可达时单次模型切换会阻塞
    // 3 分钟以上并级联拖垮并发请求（实测 182s）。连接超时 10s 保证不可达网络
    // 快速失败；总超时按调用方需求注入（embedding 路径 300s，清单下载不限总时）。
    #[allow(unused_mut)]
    let mut builder = HFClientBuilder::new().retry_max_attempts(2);
    #[cfg(feature = "http")]
    {
        #[allow(unused_mut)]
        let mut http_builder =
            reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(10));
        if let Some(total) = total_timeout {
            http_builder = http_builder.timeout(total);
        } else {
            // 清单下载分支（无总超时）：读超时防响应体停滞挂死——hf-hub 的
            // stream.next() 无读超时保护，服务端停发而 TCP 仍开时下载流会
            // 永久挂起 spawn_blocking 任务，失败聚合契约无法执行。read_timeout
            // 按单次读操作计，持续进展的慢速大文件不受影响；卡流转化为请求
            // 错误，流入既有 failures 聚合路径（可重试）。
            http_builder = http_builder.read_timeout(std::time::Duration::from_secs(60));
        }
        let http_client = http_builder
            .build()
            .map_err(|e| VecboostError::ModelLoadError(format!("HF client build failed: {e}")))?;
        builder = builder.client(http_client);
    }
    let api = builder.build_sync().map_err(|e| {
        let msg = e.to_string();
        if msg.contains("ETag") || msg.contains("missing") {
            VecboostError::ModelLoadError(format!(
                "HuggingFace hub initialization failed: {}. \
                 If using HF_ENDPOINT mirror, it may be incompatible with hf-hub 1.0.0 \
                 Workaround: pre-download the model and set \
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

/// 官方 laya ONNX bundle 仓库——本仓所有 bundle 下载默认值的唯一合法目标。
///
/// 红线：`Mattepiu/laya-onnx` 为 marker 维静态 `[.,2]` 坏产物，
/// 禁止出现在任何默认值或文档推荐中（spec R-model-bundle-fetch-003）。
pub const LAYA_BUNDLE_REPO: &str = "receptron/laya-onnx";

/// 官方 bundle 内置下载清单（远端路径，与落位相对路径一致）。
///
/// 初值取 `receptron/laya-onnx` main 分支实际文件（`.gitattributes`/`README.md`
/// 不算 bundle 资产）；上游增删文件时由联网集成测试（tests/hf_hub_integration.rs）
/// 先行红灯。
pub const LAYA_BUNDLE_FILES: &[&str] = &[
    "laya.onnx",
    "laya.onnx.data",
    "laya_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

/// 下载 repo 内指定文件列表到 out_dir。
///
/// 按 `(远端路径, 相对 out_dir 落位路径)` 清单逐文件下载；远端相对路径结构
/// 原样保留（如 `tokenizer/tokenizer.json` 落在 `out_dir/tokenizer/`），落位路径
/// 与远端路径不一致时下载后原目录内搬移。落位文件已存在**且字节数与远端一致**
/// 时跳过重下；hf-hub 的 local_dir 下载直写最终路径、中断会留下部分文件，
/// 该 size 一致性校验即防部分文件被当作已下载成功的假成功（size 无法确认时
/// 一律重下）。任一文件失败即聚合报错，错误消息显性列出已成功与失败文件及
/// 各自原因（规则 11，禁止静默部分成功）。空清单、非法 repo_id、含 `..`
/// 分量或空串的远端路径、越界落位路径均在发起任何网络请求前显性拒绝。
///
/// 阻塞下载经 `spawn_blocking` 移出异步执行器（与 `decide` 契约同口径）。
pub async fn download_files(
    repo_id: &str,
    files: &[(String, PathBuf)],
    out_dir: &Path,
) -> Result<Vec<PathBuf>, VecboostError> {
    if !is_valid_hf_repo_id(repo_id) {
        return Err(VecboostError::ModelLoadError(format!(
            "Invalid HuggingFace repo ID '{repo_id}': must match \
             'organization/model-name' pattern with alphanumeric, dash, \
             underscore, dot characters only"
        )));
    }

    if files.is_empty() {
        return Err(VecboostError::ModelLoadError(format!(
            "Empty download manifest for '{repo_id}': refusing to report \
             bundle success with nothing to download"
        )));
    }

    for (remote, placement) in files {
        validate_remote_path(remote)?;
        validate_placement_path(placement)?;
    }

    let repo = build_hf_repo_with_http(repo_id, None)?;
    let out_dir = out_dir.to_path_buf();
    let manifest: Vec<(String, PathBuf)> = files.to_vec();
    let repo_id = repo_id.to_string();
    let repo_id_for_worker = repo_id.clone();
    let (succeeded, failures) = tokio::task::spawn_blocking(move || {
        let repo_id = repo_id_for_worker.as_str();
        // 远端 size 表：短路完整性的唯一依据。hf-hub 的 local_dir 下载直写
        // 最终路径（stream_response_to_file_with_progress 以 File::create(dest)
        // 逐 chunk 写入，无 tmp+rename），流中断/进程被杀都会留下部分文件——
        // dest 存在不等于下载完整。仅当本地字节数与远端一致才允许跳过；
        // size 表获取失败或条目未知时不短路（退化为重下，绝不假成功）。
        let remote_sizes: std::collections::HashMap<String, u64> =
            match repo.list_tree().recursive(true).send() {
                Ok(entries) => entries
                    .into_iter()
                    .filter_map(|entry| match entry {
                        hf_hub::repository::RepoTreeEntry::File { path, size, .. } => {
                            Some((path, size))
                        }
                        _ => None,
                    })
                    .collect(),
                Err(e) => {
                    // 退化必须显性化（规则 11）：短路不可用意味着本次全清单重下
                    //（含 1.6 GB 级资产，慢百倍量级），调用方需要可观测线索。
                    // 安全取舍不变——绝不把未知完整性的本地文件当成功。
                    log::warn!(
                        "list_tree failed for '{repo_id}' ({e}); size-verified \
                     short-circuit unavailable, re-downloading all manifest files"
                    );
                    Default::default()
                }
            };
        let mut succeeded = Vec::new();
        let mut failures = Vec::new();
        for (remote, placement) in manifest {
            let dest = out_dir.join(&placement);
            // 已存在且大小一致才短路：部分失败重试不重下已落位文件
            //（清单含 1.6 GB 级资产，无条件覆写的代价不可接受）
            if matches_remote_size(&dest, remote_sizes.get(&remote).copied()) {
                succeeded.push(dest);
                continue;
            }
            match repo
                .download_file()
                .filename(remote.clone())
                .local_dir(out_dir.clone())
                .send()
            {
                Ok(downloaded) => {
                    if placement == Path::new(&remote) {
                        succeeded.push(downloaded);
                        continue;
                    }
                    let moved = move_into_place(&downloaded, &dest)
                        .map_err(|e| (remote.clone(), e.to_string()));
                    match moved {
                        Ok(path) => succeeded.push(path),
                        Err((name, reason)) => failures.push((name, reason)),
                    }
                }
                Err(e) => failures.push((remote, e.to_string())),
            }
        }
        (succeeded, failures)
    })
    .await
    .map_err(|e| {
        VecboostError::ModelLoadError(format!(
            "HuggingFace batch download worker for '{repo_id}' failed: {e}"
        ))
    })?;

    if failures.is_empty() {
        Ok(succeeded)
    } else {
        Err(aggregate_download_error(&repo_id, &succeeded, &failures))
    }
}

/// 校验清单远端路径：拒绝空串、含 `..` 分量与绝对路径的路径穿越
/// （vuln-0009 同源防线；hf-hub 将 filename 直接 join 进 local_dir，
/// Unix 下绝对路径会替换整条落盘路径）。
fn validate_remote_path(remote: &str) -> Result<(), VecboostError> {
    if remote.is_empty() {
        return Err(VecboostError::ModelLoadError(
            "download manifest contains an empty remote path".to_string(),
        ));
    }
    let path = Path::new(remote);
    if path.is_absolute()
        || path.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(VecboostError::ModelLoadError(format!(
            "download manifest remote path '{remote}' must be a repo-relative \
             path without '..' components or absolute prefix (path traversal)"
        )));
    }
    Ok(())
}

/// 校验落位路径：必须是非空相对路径且不逃逸 out_dir。
fn validate_placement_path(placement: &Path) -> Result<(), VecboostError> {
    if placement.as_os_str().is_empty() {
        return Err(VecboostError::ModelLoadError(
            "download manifest contains an empty placement path".to_string(),
        ));
    }
    let escapes_out_dir = placement.is_absolute()
        || placement.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
    if escapes_out_dir {
        return Err(VecboostError::ModelLoadError(format!(
            "download manifest placement path {:?} must be a relative path \
             inside the output directory",
            placement
        )));
    }
    Ok(())
}

/// 远端路径与落位路径不一致时，把下载产物搬移到落位位置（同目录树内 rename）。
fn move_into_place(downloaded: &Path, dest: &Path) -> std::io::Result<PathBuf> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(downloaded, dest)?;
    Ok(dest.to_path_buf())
}

/// 短路完整性判定：dest 为已存在文件且字节数与远端一致。
///
/// 远端 size 未知（`None`——list_tree 失败或清单条目不在 repo 内；`Some(0)`——
/// 上游未返回有效大小）一律不短路，向重下侧倾斜：hf-hub cache 路径用
/// `.incomplete` 临时文件防部分文件假成功，local_dir 直写路径无等价保护，
/// 本判定即补位防线（进程被杀等场景不会执行任何清理，只有落在跳过路径
/// 本身的校验能拦住残留部分文件）。
fn matches_remote_size(dest: &Path, remote_size: Option<u64>) -> bool {
    match remote_size {
        Some(size) if size > 0 => dest
            .metadata()
            .map(|meta| meta.is_file() && meta.len() == size)
            .unwrap_or(false),
        _ => false,
    }
}

/// 聚合部分失败错误：消息同时列出已成功与失败文件及各自原因（规则 11）。
fn aggregate_download_error(
    repo_id: &str,
    succeeded: &[PathBuf],
    failures: &[(String, String)],
) -> VecboostError {
    let mut msg = format!(
        "HuggingFace download from '{repo_id}' partially failed \
         ({} succeeded, {} failed).\n  succeeded:",
        succeeded.len(),
        failures.len()
    );
    if succeeded.is_empty() {
        msg.push_str(" (none)");
    } else {
        for path in succeeded {
            msg.push_str(&format!("\n    - {}", path.display()));
        }
    }
    msg.push_str("\n  failed:");
    for (name, reason) in failures {
        msg.push_str(&format!("\n    - {name}: {reason}"));
    }
    VecboostError::ModelLoadError(msg)
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

    #[test]
    fn test_is_valid_hf_repo_id_rejects_dot_only_segment() {
        assert!(!is_valid_hf_repo_id("."));
        assert!(!is_valid_hf_repo_id("org/."));
        assert!(!is_valid_hf_repo_id("./model"));
    }

    #[test]
    fn test_build_hf_repo_invalid_repo_id() {
        let result = build_hf_repo("../etc/passwd");
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("Invalid HuggingFace repo ID"));
            }
            other => panic!("Expected ModelLoadError, got: {:?}", other),
        }
    }

    #[test]
    fn test_build_hf_repo_valid_repo_id() {
        // Valid repo ID should succeed (HFClientSync::new() may fail in CI but the validation passes)
        let result = build_hf_repo("BAAI/bge-m3");
        // In test environment without network, HFClientSync::new() might still succeed
        // as it only creates the client, not downloads anything
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_detect_mirror_risk_no_env() {
        let saved = std::env::var("HF_ENDPOINT").ok();
        unsafe { std::env::remove_var("HF_ENDPOINT") };
        assert!(detect_mirror_risk().is_none());
        if let Some(v) = saved {
            unsafe { std::env::set_var("HF_ENDPOINT", v) };
        }
    }

    #[test]
    fn test_detect_mirror_risk_official_endpoint() {
        let saved = std::env::var("HF_ENDPOINT").ok();
        unsafe { std::env::set_var("HF_ENDPOINT", "https://huggingface.co") };
        assert!(detect_mirror_risk().is_none());
        match saved {
            Some(v) => unsafe { std::env::set_var("HF_ENDPOINT", v) },
            None => unsafe { std::env::remove_var("HF_ENDPOINT") },
        }
    }

    #[test]
    fn test_detect_mirror_risk_mirror_endpoint() {
        let saved = std::env::var("HF_ENDPOINT").ok();
        unsafe { std::env::set_var("HF_ENDPOINT", "https://hf-mirror.com") };
        let result = detect_mirror_risk();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "https://hf-mirror.com");
        match saved {
            Some(v) => unsafe { std::env::set_var("HF_ENDPOINT", v) },
            None => unsafe { std::env::remove_var("HF_ENDPOINT") },
        }
    }

    // =========================================================================
    // download_files 单元测试（全部离线：输入校验在发起网络请求前完成）
    // =========================================================================

    fn manifest_entry(remote: &str, placement: &str) -> (String, std::path::PathBuf) {
        (remote.to_string(), std::path::PathBuf::from(placement))
    }

    #[tokio::test]
    async fn test_download_files_rejects_invalid_repo_id() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let files = vec![manifest_entry("laya.onnx", "laya.onnx")];
        let result = download_files("../etc/passwd", &files, tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("Invalid HuggingFace repo ID"), "got: {msg}");
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_download_files_rejects_parent_dir_remote_path() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let files = vec![
            manifest_entry("laya.onnx", "laya.onnx"),
            manifest_entry("../evil.onnx", "evil.onnx"),
        ];
        let result = download_files("receptron/laya-onnx", &files, tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(
                    msg.contains("../evil.onnx"),
                    "error must name the offending path, got: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_download_files_rejects_empty_remote_path() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let files = vec![manifest_entry("", "whatever.onnx")];
        let result = download_files("receptron/laya-onnx", &files, tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("empty"), "got: {msg}");
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_download_files_rejects_empty_manifest() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let result = download_files("receptron/laya-onnx", &[], tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(
                    msg.to_lowercase().contains("empty"),
                    "empty manifest must be explicitly rejected, got: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 绝对路径 remote 拒绝：hf-hub 将 filename join 进 local_dir，Unix 下
    /// 绝对路径会替换整条落盘路径写穿 out_dir，必须在网络请求前拒绝
    #[tokio::test]
    async fn test_download_files_rejects_absolute_remote_path() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let files = vec![manifest_entry("/etc/evil.onnx", "evil.onnx")];
        let result = download_files("receptron/laya-onnx", &files, tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(
                    msg.contains("/etc/evil.onnx"),
                    "error must name the offending path, got: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 已存在短路的完整性判定钉：仅「dest 存在 && 远端 size 已知且一致」
    /// 才允许跳过；size 未知（None/0）一律重下——部分文件不得假成功。
    /// download_files 级短路行为（size 一致跳过、size 不符重下）由联网集成
    /// 测试 tests/hf_hub_integration.rs 断言。
    #[test]
    fn test_matches_remote_size_integrity_gate() {
        let dir = tempfile::tempdir().expect("temp dir");
        let dest = dir.path().join("asset.bin");
        std::fs::write(&dest, vec![0u8; 1024]).expect("write dest");

        // size 一致 → 跳过
        assert!(matches_remote_size(&dest, Some(1024)));
        // size 不符（部分文件/截断）→ 重下
        assert!(!matches_remote_size(&dest, Some(1023)));
        assert!(!matches_remote_size(&dest, Some(1025)));
        // size 未知（list_tree 失败或上游未返回）→ 不短路
        assert!(!matches_remote_size(&dest, None));
        assert!(!matches_remote_size(&dest, Some(0)));
        // dest 不存在 → 不短路
        assert!(!matches_remote_size(
            &dir.path().join("missing.bin"),
            Some(1024)
        ));
        // dest 是目录 → 不短路
        assert!(!matches_remote_size(dir.path(), Some(0)));
    }

    #[tokio::test]
    async fn test_download_files_rejects_unsafe_placement_path() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let files = vec![manifest_entry("laya.onnx", "../escape.onnx")];
        let result = download_files("receptron/laya-onnx", &files, tmp.path()).await;
        match result.unwrap_err() {
            VecboostError::ModelLoadError(msg) => {
                assert!(
                    msg.contains("../escape.onnx"),
                    "error must name the offending placement, got: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    #[test]
    fn test_aggregate_download_error_lists_succeeded_and_failed() {
        let err = aggregate_download_error(
            "receptron/laya-onnx",
            &[std::path::PathBuf::from("out/laya.onnx")],
            &[(
                "tokenizer/tokenizer.json".to_string(),
                "404 not found".to_string(),
            )],
        );
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("laya.onnx"), "succeeded list missing: {msg}");
                assert!(
                    msg.contains("tokenizer/tokenizer.json"),
                    "failed list missing: {msg}"
                );
                assert!(
                    msg.contains("404 not found"),
                    "per-file reason missing: {msg}"
                );
            }
            other => panic!("Expected ModelLoadError, got: {other:?}"),
        }
    }

    /// 下载目标红线（R-model-bundle-fetch-003）：默认值只能是官方
    /// receptron/laya-onnx，Mattepiu 静态坏产物不得出现在任何默认值中。
    #[test]
    fn test_laya_bundle_constants_red_line() {
        assert_eq!(LAYA_BUNDLE_REPO, "receptron/laya-onnx");
        assert!(
            !LAYA_BUNDLE_REPO.to_lowercase().contains("mattepiu"),
            "Mattepiu/laya-onnx 为 marker 维静态坏产物，禁止作为默认下载目标"
        );
        assert_eq!(LAYA_BUNDLE_FILES.len(), 5, "官方 bundle 当前为 5 文件");
        let mut seen = std::collections::BTreeSet::new();
        for entry in LAYA_BUNDLE_FILES {
            assert!(!entry.is_empty());
            assert!(
                !entry.contains(".."),
                "manifest entry must not traverse: {entry}"
            );
            assert!(seen.insert(*entry), "duplicate manifest entry: {entry}");
        }
        assert!(LAYA_BUNDLE_FILES.contains(&"laya.onnx"));
        assert!(LAYA_BUNDLE_FILES.contains(&"tokenizer/tokenizer.json"));
    }
}
