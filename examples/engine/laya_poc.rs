// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! P0 数值对照 PoC：经 EngineFactory→AnyEngine→InferenceEngine::decide 全链
//! 调用 Laya 决策管线，与 Python onnxruntime 导出的 golden 概率逐值对照
//! （文档 §六 P0 验收：概率差 < 1e-4）。
//!
//! 用法：
//! ```bash
//! cargo run -p vecboost-examples --bin laya_poc --features onnx
//! ```
//!
//! 环境变量：
//! - `LAYA_BUNDLE`：bundle 目录（默认 `models/laya`）。模型文件按
//!   `model.onnx → model_quantized.onnx → laya.onnx → laya_int8.onnx` 探测；
//!   tokenizer 按根目录 `tokenizer.json` → `tokenizer/tokenizer.json` 探测；
//!   可选 `laya_config.json`（per-cardinality 温度校准，缺失时 warn 并回退 1.2）。
//! - `LAYA_GOLDEN_JSON`：golden 文件路径（缺省仅打印管线输出供人工对照）。
//!   格式（Python onnxruntime 复现「预处理→推理→温度校准→softmax」全管线后导出）：
//!   ```json
//!   {"answers": [{"question": "churn_risk", "probabilities": {"false": 0.1, "true": 0.9}}]}
//!   ```
//!   概率为校准后最终概率，与管线输出同口径；逐值断言 |diff| < 1e-4。
//!
//! bundle 缺失时打印下载指引后正常退出（SKIP 语义，不报错）。

use std::collections::BTreeMap;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::domain::{DecisionAnswerBody, DecisionRequest};
use vecboost::engine::{EngineFactory, InferenceEngine};

/// golden 概率断言阈值（文档 §六 P0 验收口径）
const GOLDEN_TOLERANCE: f32 = 1e-4;

fn default_request() -> Result<DecisionRequest, serde_json::Error> {
    // 三题型各一条：choice / score / noul 全覆盖
    serde_json::from_str(
        r#"{
        "state": "Hi, we were billed twice for March. The duplicate charge is still pending refund.",
        "questions": [
            {
                "name": "department",
                "qtype": "choice",
                "instructions": "Which department should handle this ticket?",
                "options": ["billing", "technical", "other"]
            },
            {
                "name": "urgency",
                "qtype": "score",
                "instructions": "How urgent is this issue on a scale from 0 (not urgent) to 4 (blocking)?"
            },
            {
                "name": "churn_risk",
                "qtype": "noul",
                "instructions": "Is this customer at risk of churning?"
            }
        ]
    }"#,
    )
}

/// 管线答案体归一化为概率表，与 golden 的 `probabilities` map 同构：
/// choice→各选项概率、score→等级分布、noul→{false, true}
fn probability_map(body: &DecisionAnswerBody) -> BTreeMap<String, f32> {
    match body {
        DecisionAnswerBody::Choice { probabilities, .. } => probabilities.clone(),
        DecisionAnswerBody::Score { distribution, .. } => distribution.clone(),
        DecisionAnswerBody::Noul { p_true } => BTreeMap::from([
            ("false".to_string(), 1.0 - p_true),
            ("true".to_string(), *p_true),
        ]),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Laya 决策管线 P0 数值对照示例");
    println!("==================================\n");

    let bundle = std::env::var("LAYA_BUNDLE").unwrap_or_else(|_| "models/laya".to_string());
    let bundle_path = std::path::Path::new(&bundle);
    if !bundle_path.is_dir() {
        println!("⚠️ bundle 目录不存在: {bundle}");
        println!("   SKIP 语义：打印下载指引后正常退出。");
        println!("\n📥 下载指引（任选一种）：");
        println!("   huggingface-cli download Mattepiu/laya-onnx --local-dir {bundle}");
        println!("   # 或从 convaiinnovations/laya 系列 checkpoint 导出后放置：");
        println!("   #   {bundle}/model.onnx | laya.onnx（模型图 + 外部权重数据文件同目录）");
        println!("   #   {bundle}/tokenizer.json 或 {bundle}/tokenizer/tokenizer.json");
        println!("   #   {bundle}/laya_config.json（可选，per-cardinality 温度校准）");
        return Ok(());
    }

    let req = default_request()?;
    req.validate()?;

    // 全链入口：EngineFactory → AnyEngine → trait decide。
    // 不得绕过工厂/trie 直构管线（decision mod 为 pub(crate)）——
    // 此形态同时把 AnyEngine::Decision 变体与转发臂纳入在线闸门。
    let config = ModelConfig {
        name: "laya".to_string(),
        engine_type: EngineType::Onnx,
        model_path: bundle_path.to_path_buf(),
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
    };

    println!("🔧 EngineFactory::create(Onnx, task=decision) ...");
    // 全链入口：EngineFactory（task 分派臂 → DecisionPipeline）→
    // AnyEngine::Decision → trait decide。不得绕过工厂直构管线
    // （decision mod 为 pub(crate)）——此形态同时把分派臂、Decision
    // 变体与转发臂纳入在线闸门。失败时可诊断：区分 bundle 资产缺失
    // 与 UnsupportedTask（构建未含决策管线）两类成因。
    let engine = EngineFactory::create(EngineType::Onnx, &config).map_err(|e| {
        format!(
            "EngineFactory::create(task=decision) 失败：{e}\n\
             排查：① bundle 是否含模型文件（model.onnx|model_quantized.onnx|\
             laya.onnx|laya_int8.onnx）与 tokenizer.json（或 tokenizer/ 子目录）\n\
             ② UnsupportedTask 则说明本次构建未含决策管线（需 onnx feature）"
        )
    })?;

    println!(
        "🧠 InferenceEngine::decide（{} 题）...",
        req.questions.len()
    );
    let response = InferenceEngine::decide(&engine, &req)?;
    println!("\n📋 管线输出（温度校准后概率）：");
    println!("{}", serde_json::to_string_pretty(&response)?);

    // golden 对照：LAYA_GOLDEN_JSON 未设时仅打印供人工对照
    let Ok(golden_path) = std::env::var("LAYA_GOLDEN_JSON") else {
        println!("\n✅ 未设 LAYA_GOLDEN_JSON，跳过逐值断言（人工对照上方输出）");
        return Ok(());
    };

    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&golden_path)?)?;
    let mut mismatches: Vec<String> = Vec::new();
    for golden_answer in golden["answers"]
        .as_array()
        .ok_or("golden 格式错误：缺少 answers 数组")?
    {
        let qname = golden_answer["question"]
            .as_str()
            .ok_or("golden 格式错误：answer 缺少 question 名")?;
        let Some(answer) = response.answers.iter().find(|a| a.question == qname) else {
            mismatches.push(format!("{qname}: 管线输出无此问题"));
            continue;
        };
        let got = probability_map(&answer.answer);
        let Some(want) = golden_answer["probabilities"].as_object() else {
            mismatches.push(format!("{qname}: golden 缺少 probabilities 对象"));
            continue;
        };
        for (key, want_p) in want {
            let want_p = want_p.as_f64().ok_or("golden 概率非数字")? as f32;
            match got.get(key) {
                Some(got_p) => {
                    let diff = (got_p - want_p).abs();
                    if diff >= GOLDEN_TOLERANCE {
                        mismatches.push(format!(
                            "{qname}[{key}]: got {got_p:.6}, want {want_p:.6}, diff {diff:.2e} ≥ {GOLDEN_TOLERANCE}"
                        ));
                    }
                }
                None => mismatches.push(format!("{qname}[{key}]: 管线输出缺少该键")),
            }
        }
    }

    if mismatches.is_empty() {
        println!("\n✅ golden 对照通过：全部概率差 < {GOLDEN_TOLERANCE}");
    } else {
        eprintln!("\n❌ golden 对照失败（{} 处）：", mismatches.len());
        for m in &mismatches {
            eprintln!("   {m}");
        }
        std::process::exit(1);
    }
    Ok(())
}
