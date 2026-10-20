// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// onnx embedding 引擎真机端到端（onnx-primary 主路径回归防线）。
//
// 背景：onnx embedding 此前无真实资产测试（探测类测试不触发 Session），
// 二〇二六年十月基准真机首跑暴露三处兼容缺陷（均已修复，本文件钉住）：
// 1) forward_pass 单条输入曾为 rank-1 [seq]，标准 BERT ONNX 图要求
// [batch, seq]（rank 2），被图拒绝；
// 2) 三输入图（optimum/transformers.js 导出）要求 token_type_ids，
// 引擎需按 session.inputs 探测后补喂全零（segment 0）；
// 3) rank-2 输入对应 rank-3 输出 [1, seq, hidden]，单条路径索引口径。
//
// 先决条件（缺任一整文件 SKIP，与 translate_e2e 同款离线红线）：
// - 本地 bundle models/Xenova-bge-small-en-v1.5（model.onnx + tokenizer.json，
// 获取：cargo run -p vecboost-examples --bin download_model -- --onnx Xenova/bge-small-en-v1.5）；
// - onnxruntime 动态库 3rdparty/onnxruntime/libonnxruntime.so（ort/load-dynamic）。
//
// ```text
// cargo test --features onnx --test onnx_embed_e2e
// ```

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};

const BUNDLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);
const ORT_DYLIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnxruntime/libonnxruntime.so"
);

fn assets_ready() -> bool {
    let bundle = PathBuf::from(BUNDLE);
    bundle.join("model.onnx").exists()
        && bundle.join("tokenizer.json").exists()
        && PathBuf::from(ORT_DYLIB).exists()
}

/// ORT load-dynamic 定位：未显式设置时钉本仓 3rdparty 动态库。
/// OnceLock 串行化：同进程多测试（并行线程）同时到达也只 set_var 一次，
/// 最小化 setenv/getenv 竞争窗口（项目已知 pitfall：测试中 set_var 污染）。
fn ensure_ort_dylib() {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            #[allow(unsafe_code, reason = "OnceLock 串行化的进程级环境变量设置")]
            unsafe {
                std::env::set_var("ORT_DYLIB_PATH", ORT_DYLIB);
            }
        }
    });
}

fn config() -> ModelConfig {
    ModelConfig {
        name: "xenova-bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from(BUNDLE),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 8,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        task: ModelTask::Embedding,
        quantized: false,
        decision_params: None,
        translation_params: None,
        transcription_params: None,
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (l2_norm(a) * l2_norm(b))
}

/// 单条与批量真机前向：维度 / 归一化 / 有限值 / 单条-批量一致性
/// （三处兼容修复的联合回归钉）。
#[test]
fn onnx_embed_single_and_batch_real_inference() {
    if !assets_ready() {
        eprintln!(
            "SKIP onnx_embed_single_and_batch_real_inference: Xenova bundle 或 ORT 动态库缺失"
        );
        return;
    }
    // ORT load-dynamic 定位（未显式设置时按平台默认落位，显式钉本仓 3rdparty）
    ensure_ort_dylib();
    vecboost::i18n::init();
    let engine = AnyEngine::new(&config(), EngineType::Onnx, Precision::Fp32)
        .expect("onnx embedding 引擎必须能加载 Xenova bundle");

    let texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn representations from data.",
    ];

    // 单条：维度/归一化/有限值（rank-2 输入 + rank-3 输出索引 + 三输入喂参）
    let single_a = engine.embed(texts[0]).expect("single embed");
    assert_eq!(single_a.len(), 384, "维度必须等于 expected_dimension");
    assert!(
        (l2_norm(&single_a) - 1.0).abs() < 1e-3,
        "出口向量必须 L2 归一化"
    );
    assert!(single_a.iter().all(|v| v.is_finite()));

    let single_b = engine.embed(texts[1]).expect("single embed (second text)");
    assert!(
        cosine(&single_a, &single_b) < 0.99,
        "不同文本不得产生近似相同向量（区分度防线）"
    );

    // 批量：形状/归一化 + 单条-批量一致性（batch 路径 padding 后逐条 pooling
    // 应与单条一致；容差放宽到余弦 0.999 覆盖 padding 位置的浮点差异）
    let batch = engine
        .embed_batch(&texts.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .expect("batch embed");
    assert_eq!(batch.len(), 2);
    for (i, v) in batch.iter().enumerate() {
        assert_eq!(v.len(), 384);
        assert!((l2_norm(v) - 1.0).abs() < 1e-3);
        assert!(v.iter().all(|x| x.is_finite()));
        let single = if i == 0 { &single_a } else { &single_b };
        assert!(
            cosine(v, single) > 0.999,
            "批量第 {i} 条与单条余弦应≈1，实际 {}",
            cosine(v, single)
        );
    }
}
