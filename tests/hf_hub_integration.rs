// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 联网集成测试：`download_files` 真实拉取官方 `receptron/laya-onnx` bundle。
//!
//! 默认 SKIP（测试离线可跑红线）；`HF_INTEGRATION=1` 时才执行：
//!
//! ```text
//! HF_INTEGRATION=1 cargo test -p vecboost --test hf_hub_integration
//! ```
//!
//! 断言顺序（spec R-model-bundle-fetch-001）：
//! 1. 内置清单常量 `LAYA_BUNDLE_FILES` 与 repo 实际文件列表一致（上游增删
//!    文件时先行红灯，防漂移）；
//! 2. `download_files` 逐文件落位且字节数 > 0。

use std::collections::BTreeSet;
use std::path::PathBuf;
use vecboost::utils::hf_hub::{LAYA_BUNDLE_FILES, LAYA_BUNDLE_REPO, download_files};

/// repo 内非 bundle 资产的脚手架文件，清单一致性比对时剔除。
const REPO_SCAFFOLDING: &[&str] = &[".gitattributes", "README.md"];

fn integration_enabled() -> bool {
    std::env::var("HF_INTEGRATION").as_deref() == Ok("1")
}

#[tokio::test]
async fn laya_bundle_manifest_matches_upstream_and_downloads_nonempty() {
    if !integration_enabled() {
        eprintln!("hf_hub_integration: HF_INTEGRATION!=1，联网集成测试 skip");
        return;
    }

    let client = hf_hub::HFClientBuilder::new()
        .build_sync()
        .expect("HF client");
    let repo = client.model("receptron", "laya-onnx");
    let entries = repo
        .list_tree()
        .recursive(true)
        .send()
        .expect("list tree of receptron/laya-onnx");
    let actual_files: BTreeSet<String> = entries
        .iter()
        .filter_map(|entry| match entry {
            hf_hub::repository::RepoTreeEntry::File { path, .. } => Some(path.clone()),
            _ => None,
        })
        .collect();
    let expected: BTreeSet<String> = LAYA_BUNDLE_FILES
        .iter()
        .copied()
        .chain(REPO_SCAFFOLDING.iter().copied())
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        expected, actual_files,
        "内置清单与上游 repo 实际文件漂移：先更新 LAYA_BUNDLE_FILES 再发版（actual={actual_files:?}）"
    );

    let tmp = tempfile::tempdir().expect("temp dir for bundle download");
    let manifest: Vec<(String, PathBuf)> = LAYA_BUNDLE_FILES
        .iter()
        .map(|f| ((*f).to_string(), PathBuf::from(f)))
        .collect();
    let downloaded = download_files(LAYA_BUNDLE_REPO, &manifest, tmp.path())
        .await
        .expect("download official laya bundle");

    assert_eq!(downloaded.len(), LAYA_BUNDLE_FILES.len());
    for path in &downloaded {
        let meta = std::fs::metadata(path).expect("downloaded file must exist");
        assert!(meta.is_file(), "not a regular file: {path:?}");
        assert!(meta.len() > 0, "downloaded file is empty: {path:?}");
    }
    assert!(
        tmp.path().join("tokenizer/tokenizer.json").is_file(),
        "tokenizer/ 子目录结构必须原样保留"
    );

    // ── download_files 级短路断言（第二轮调用）──
    // hf-hub 的 local_dir 下载直写最终路径，中断会留部分文件；短路必须
    // 以「本地字节数 == 远端 size」为准，而非仅 dest 存在。
    let small = tmp.path().join("tokenizer/tokenizer_config.json");
    let original = std::fs::read(&small).expect("read downloaded tokenizer_config");
    // 同 size 篡改：size 校验通过即跳过——本地内容不被远端覆写
    let same_len_fake = vec![b'x'; original.len()];
    std::fs::write(&small, &same_len_fake).expect("tamper same-size");
    // 异 size 截断：模拟下载中断残留的部分文件
    let laya_config = tmp.path().join("laya_config.json");
    let laya_original_len = std::fs::metadata(&laya_config).expect("meta").len();
    std::fs::write(&laya_config, vec![b'y'; (laya_original_len - 1) as usize])
        .expect("tamper truncated");

    let redownloaded = download_files(LAYA_BUNDLE_REPO, &manifest, tmp.path())
        .await
        .expect("second download round with size-verified short-circuit");
    assert_eq!(redownloaded.len(), LAYA_BUNDLE_FILES.len());
    assert_eq!(
        std::fs::read(&small).expect("read back tokenizer_config"),
        same_len_fake,
        "size 一致的落位文件必须短路保留（不重下）"
    );
    assert_eq!(
        std::fs::metadata(&laya_config)
            .expect("meta after redownload")
            .len(),
        laya_original_len,
        "截断的部分文件必须被 size 校验识别并重下恢复"
    );
}
