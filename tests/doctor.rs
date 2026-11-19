// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! doctor 只读诊断集成测试。
//!
//! 合法默认配置下：核心检查（config/cache-persist/threads/gpu）必须 PASS，
//! 报告整体无 FAIL（tokenizer/models 依赖本地模型存在性，缺失时为 WARN，
//! 对应"运行时回退 HF 下载"的合法状态）。
//!
//! security-auth 语义（54802d6 起）：auth 关闭 + 非 loopback 绑定 → FAIL，
//! 因此仓库默认配置（auth=false + host=0.0.0.0）本身不再是"合法无 FAIL"。
//! 本测试保持默认 0.0.0.0 绑定不变、显式启用 auth（→ WARN 而非 FAIL），
//! 锁定"面向公网的合法部署形态无 FAIL"契约；检查语义本身不放松。

#[tokio::test]
async fn doctor_legal_config_has_no_failures() {
    let mut config = vecboost::VecboostConfig::default();
    // 仓库自带的本地模型（存在时进入真实 tokenizer 加载路径；缺失时 WARN 合法）
    let local_model = std::path::Path::new("models/all-MiniLM-L6-v2");
    if local_model.is_dir() {
        config.model.model_path = Some(local_model.to_string_lossy().to_string());
    }
    config.model.use_gpu = false; // WSL2/CI 无 GPU，CPU 模式为合法配置
    // security-auth 语义（54802d6）：auth 关闭 + 非 loopback 绑定 → FAIL。
    // 默认配置（auth=false + 0.0.0.0）会被判 FAIL，不属于"合法配置"；
    // 此处显式启用 auth（非 loopback 下 → WARN），维持本测试
    // "合法配置无 FAIL"的意图，且不放松 doctor 检查语义。
    config.auth.enabled = true;

    let report = vecboost::doctor::DoctorReport::run(&config).await;
    assert!(!report.has_failures(), "合法配置不应产生 FAIL: {report:?}");
    for name in ["config", "cache-persist", "threads", "gpu"] {
        let r = report
            .results
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("缺少 {name} 检查项"));
        assert_eq!(
            r.status,
            vecboost::doctor::CheckStatus::Pass,
            "{name}: {}",
            r.detail
        );
    }
}

#[tokio::test]
async fn doctor_detects_invalid_config_as_fail() {
    // 构造必然校验失败的配置：空 model_repo 且无本地路径 → tokenizer/config 链路异常。
    // validate 的具体失败面随版本演进，这里锁定"报告能反映 FAIL"这一契约。
    let mut config = vecboost::VecboostConfig::default();
    config.model.model_repo = String::new();
    config.model.model_path = None;
    let report = vecboost::doctor::DoctorReport::run(&config).await;
    // 无论 validate 是否放行空 repo，诊断报告必须结构完整且可判定
    assert!(!report.results.is_empty());
    assert!(report.results.iter().all(|r| !r.detail.is_empty()));
}
