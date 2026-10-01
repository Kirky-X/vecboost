// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// 对拍侧 EngineType::Onnx 挂 onnx feature 门：无 onnx feature 的构建下本
// 测试目标整体为空（feature 门盲区惯例，`cargo check --features
// quantized-gguf --tests` 不受影响）
#![cfg(feature = "onnx")]

//! 对齐闸门（spec R-candle-decision-head-003）：candle 原生决策头与 onnx
//! `DecisionPipeline` 同题对照，choice（≥3 选项）/score/noul 各一题，断言
//! 逐有效 marker `|Δlogit| ≤ 1e-4`（未校准 logits，温度施加前）。
//!
//! 先决条件（checkpoint safetensors、ORT_DYLIB_PATH、onnx bundle）缺失时
//! 整文件 SKIP（探测集中于 `tests/common/mod.rs`）；测试不依赖 Python 与
//! 外部进程——双实现同进程纯 Rust 对拍。
//!
//! ```text
//! cargo test -p vecboost --features onnx --test candle_decision_parity
//! ```

mod common;

use std::path::PathBuf;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::domain::DecisionRequest;
use vecboost::engine::{EngineFactory, InferenceEngine};

/// 硬闸门（spec Constraints）：调整阈值必须在 design.md 记录理由
const PARITY_TOLERANCE: f32 = 1e-4;

fn decision_config(model_path: PathBuf, engine_type: EngineType) -> ModelConfig {
    ModelConfig {
        name: format!("parity-{engine_type}"),
        engine_type,
        model_path,
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        task: ModelTask::Decision,
        quantized: false,
        decision_params: None,
    }
}

/// 三题型同题输入：choice 恰 3 选项（spec 下界）、score 固定 5 级、noul
/// 固定 2 marker；state 为同一请求共享段（两路管线同一预处理协议）。
fn parity_request() -> DecisionRequest {
    serde_json::from_str(
        r#"{"state":"the user has booked flights for a weekend trip and needs to pick lodging",
            "questions":[
              {"name":"destination","qtype":"choice","instructions":"Which lodging fits this trip best?","options":["beach resort","mountain cabin","city hotel"]},
              {"name":"urgency","qtype":"score","instructions":"How urgent is this decision?"},
              {"name":"churn_risk","qtype":"noul","instructions":"Is the user likely to cancel the booking?"}
            ]}"#,
    )
    .expect("parity request")
}

/// 逐 marker 硬闸门：|Δlogit| ≤ 1e-4（选择 logit 差而非概率差的理由：
/// softmax 后的概率抹平 logits 的平移自由度且压缩数值动态范围，logit 是
/// 模型输出的最小失真对照面）。
#[test]
fn candle_vs_onnx_parity_gate() {
    let assets = match common::candle_parity_assets() {
        Ok(assets) => assets,
        Err(missing) => {
            eprintln!("SKIP [candle_vs_onnx_parity_gate]: {missing}");
            return;
        }
    };
    assets.ensure_ort_env();

    let candle_bundle = assets
        .checkpoint
        .parent()
        .expect("checkpoint path has parent dir")
        .to_path_buf();
    let onnx = EngineFactory::create(
        EngineType::Onnx,
        &decision_config(assets.onnx_bundle.clone(), EngineType::Onnx),
    )
    .expect("onnx 对拍侧加载（bundle 与 ORT_DYLIB_PATH 已由探测守卫）");
    let candle = EngineFactory::create(
        EngineType::Candle,
        &decision_config(candle_bundle, EngineType::Candle),
    )
    .expect("candle 对拍侧加载（checkpoint 已由探测守卫）");

    let req = parity_request();
    let onnx_logits = onnx.decide_logits(&req).expect("onnx logits");
    let candle_logits = candle.decide_logits(&req).expect("candle logits");

    assert_eq!(
        onnx_logits.len(),
        candle_logits.len(),
        "两路题数必须一致（同一请求同一预处理协议）"
    );
    for (q, (onnx_row, candle_row)) in onnx_logits.iter().zip(&candle_logits).enumerate() {
        assert_eq!(
            onnx_row.len(),
            candle_row.len(),
            "题 {q} 的有效 marker 数必须一致"
        );
        for (m, (a, b)) in onnx_row.iter().zip(candle_row).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= PARITY_TOLERANCE,
                "题 {q} marker {m} 超出对齐闸门：|{a} - {b}| = {diff} > {PARITY_TOLERANCE}"
            );
        }
    }

    // wire 侧顺带对照：同输入经 decide 后处理的答案题数一致（后处理/校准
    // 协议两路同源，非本闸门的数值断言对象）
    let onnx_resp = onnx.decide(&req).expect("onnx decide");
    let candle_resp = candle.decide(&req).expect("candle decide");
    assert_eq!(onnx_resp.answers.len(), candle_resp.answers.len());
}
