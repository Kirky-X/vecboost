// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 引擎加载与推理变体：SHA256 校验、池化模式、显存上限触发的 OOM 回退。
//!
//! 依赖本地 ONNX 模型与 ORT dylib（`3rdparty/onnxruntime`），缺席时 SKIP。

#![cfg(feature = "onnx")]

use std::path::Path;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, PoolingMode};
use vecboost::engine::EngineFactory;
use vecboost::library::{LibraryConfig, VecBoostLibrary};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);
const ORT_DYLIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnxruntime/libonnxruntime.so"
);

fn model_ready() -> bool {
    Path::new(MODEL_PATH).join("model.onnx").exists()
}

fn base_config() -> ModelConfig {
    ModelConfig {
        name: "Xenova/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Onnx,
        model_path: MODEL_PATH.into(),
        device: DeviceType::Cpu,
        max_batch_size: 4,
        expected_dimension: Some(384),
        task: ModelTask::Embedding,
        ..ModelConfig::default()
    }
}

fn set_ort_env() {
    if Path::new(ORT_DYLIB).is_file() {
        // SAFETY: 测试进程内设置 ORT dylib 路径，无并发写。
        unsafe { std::env::set_var("ORT_DYLIB_PATH", ORT_DYLIB) };
    }
}

#[test]
fn sha256_mismatch_rejects_model() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    set_ort_env();
    let mut cfg = base_config();
    cfg.model_sha256 = Some("0".repeat(64));
    let err = EngineFactory::create(EngineType::Onnx, &cfg)
        .err()
        .expect("SHA256 不匹配必须拒绝");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("SHA256") || msg.contains("sha256"),
        "错误应点名 SHA256 校验: {msg}"
    );
}

#[tokio::test]
async fn pooling_max_and_mean_paths() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    set_ort_env();
    for mode in [PoolingMode::Max, PoolingMode::Mean, PoolingMode::Cls] {
        let mut cfg = base_config();
        cfg.pooling_mode = Some(mode.clone());
        let lib = VecBoostLibrary::new(LibraryConfig::from_model_config(cfg))
            .await
            .expect("library init");
        let out = lib.embed("pooling variant").await.expect("embed");
        assert_eq!(out.embedding.len(), 384, "pooling={mode:?}");
        assert!(out.embedding.iter().all(|v| v.is_finite()));
    }
}

/// 极小显存上限 + OOM 回退：不得 panic；成功时向量维度保持契约。
#[tokio::test]
async fn oom_fallback_is_explicit_and_non_panicking() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    set_ort_env();
    let mut cfg = base_config();
    cfg.memory_limit_bytes = Some(1);
    cfg.oom_fallback_enabled = true;
    let lib = VecBoostLibrary::new(LibraryConfig::from_model_config(cfg))
        .await
        .expect("library init");
    match lib.embed("oom fallback").await {
        Ok(out) => assert_eq!(out.embedding.len(), 384),
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(!msg.is_empty());
        }
    }
}
