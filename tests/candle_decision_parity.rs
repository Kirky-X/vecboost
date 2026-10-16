// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// 对拍侧双引擎门：onnx（DecisionPipeline）与 candle（原生决策头）均
// 在编译内时本测试目标才有对拍语义（onnx-primary 下 candle 为 opt-in
// feature）；无任一 feature 的构建下整体为空（feature 门盲区惯例，
// `cargo check --features quantized-gguf --tests` 不受影响）
#![cfg(all(feature = "onnx", feature = "candle"))]

//! 对齐闸门（spec R-candle-decision-head-003）：candle 原生决策头与 onnx
//! `DecisionPipeline` 同题对照，choice（≥3 选项）/score/noul 各一题，断言
//! 逐有效 marker `|Δlogit| ≤ 1e-4`（未校准 logits，温度施加前）。
//!
//! 先决条件（checkpoint safetensors、ORT_DYLIB_PATH、onnx bundle）缺失时
//! 整文件 SKIP（探测函数 `candle_parity_assets` 内联于本文件、单一消费者，
//! 见下方「对拍先决条件资产」注记）；测试不依赖 Python 与
//! 外部进程——双实现同进程纯 Rust 对拍。
//!
//! ```text
//! cargo test -p vecboost --features onnx --test candle_decision_parity
//! ```

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 对拍先决条件资产（原 tests/common 共享段，仅本测试目标消费，内联消除
// 跨二进制共享导致的 dead_code 抑制）
// ---------------------------------------------------------------------------

/// onnxruntime 动态库的平台默认落位（`ort/load-dynamic` 运行时依赖，
/// 见 3rdparty/onnxruntime/README.md）。
#[cfg(target_os = "macos")]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.dylib";
#[cfg(target_os = "windows")]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.dll";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const ORT_DEFAULT_DYLIB: &str = "3rdparty/onnxruntime/libonnxruntime.so";

/// candle 原生决策头与 onnx `DecisionPipeline` 同题对拍的先决资产。
struct CandleParityAssets {
    /// 官方 `convaiinnovations/laya` checkpoint（获取步骤见
    /// docs/USER_GUIDE.md「Candle 原生决策路径」）。
    checkpoint: PathBuf,
    /// onnx 对拍侧会话加载所需的动态库路径。
    ort_dylib: PathBuf,
    /// onnx 对拍侧官方 bundle（receptron/laya-onnx 产物）。
    onnx_bundle: PathBuf,
}

impl CandleParityAssets {
    /// 确保进程内 `ORT_DYLIB_PATH` 环境变量就位（未显式设置时回填默认落位；
    /// ort/load-dynamic 在首个 Session 创建时按该变量加载动态库）。
    fn ensure_ort_env(&self) {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", &self.ort_dylib);
            }
        }
    }
}

/// 聚合探测 checkpoint safetensors、onnxruntime 动态库与 onnx 对拍 bundle；
/// `Err` 携带缺失原因清单，调用方据此 SKIP（离线红线：资产缺失不硬失败）。
fn candle_parity_assets() -> Result<CandleParityAssets, String> {
    let checkpoint = PathBuf::from("models/laya-pytorch/model.safetensors");
    let ort_dylib = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(ORT_DEFAULT_DYLIB),
    };
    let onnx_bundle = PathBuf::from("models/laya");
    let mut missing = Vec::new();
    if !checkpoint.is_file() {
        missing.push(format!(
            "checkpoint 缺失：{}（获取步骤见 docs/USER_GUIDE.md）",
            checkpoint.display()
        ));
    }
    if !ort_dylib.is_file() {
        missing.push(format!(
            "onnxruntime 动态库缺失：{}（ORT_DYLIB_PATH 或默认落位）",
            ort_dylib.display()
        ));
    }
    if !onnx_bundle.join("laya.onnx").is_file() {
        missing.push(format!(
            "onnx 对拍 bundle 缺失：{}/laya.onnx（receptron/laya-onnx，获取步骤见 \
             docs/USER_GUIDE.md）",
            onnx_bundle.display()
        ));
    }
    if missing.is_empty() {
        Ok(CandleParityAssets {
            checkpoint,
            ort_dylib,
            onnx_bundle,
        })
    } else {
        Err(missing.join("; "))
    }
}

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
        translation_params: None,
        transcription_params: None,
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
    let assets = match candle_parity_assets() {
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
