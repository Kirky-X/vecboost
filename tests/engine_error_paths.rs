// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 引擎工厂错误路径：坏模型目录/缺失 tokenizer/未编译后端/任务不支持。
//!
//! 不依赖真实模型资产，全部走 `EngineFactory::create` 的 fail-closed 分支，
//! 用于覆盖 `onnx_engine.rs` 的加载期校验与 `factory.rs` 的路由拒绝。

#![cfg(feature = "onnx")]

use std::path::PathBuf;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::engine::EngineFactory;

fn base_config() -> ModelConfig {
    ModelConfig {
        name: "missing-model".to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from("/definitely/missing/model-dir"),
        device: DeviceType::Cpu,
        max_batch_size: 4,
        expected_dimension: Some(384),
        task: ModelTask::Embedding,
        ..ModelConfig::default()
    }
}

#[test]
fn onnx_engine_rejects_missing_model_dir() {
    let err = EngineFactory::create(EngineType::Onnx, &base_config())
        .err()
        .expect("缺失模型目录必须拒绝");
    let msg = format!("{err:?}");
    assert!(!msg.is_empty());
}

#[test]
fn onnx_engine_rejects_dir_without_onnx_file() {
    let dir = std::env::temp_dir().join(format!("vecboost-empty-model-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let mut cfg = base_config();
    cfg.model_path = dir.clone();
    let err = EngineFactory::create(EngineType::Onnx, &cfg)
        .err()
        .expect("空模型目录必须拒绝");
    assert!(!format!("{err:?}").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn onnx_engine_rejects_missing_tokenizer() {
    let dir = std::env::temp_dir().join(format!("vecboost-no-tok-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    std::fs::write(dir.join("model.onnx"), b"not-a-real-onnx").expect("write model");
    let mut cfg = base_config();
    cfg.model_path = dir.clone();
    let err = EngineFactory::create(EngineType::Onnx, &cfg)
        .err()
        .expect("缺 tokenizer 必须拒绝");
    assert!(!format!("{err:?}").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn candle_engine_rejected_when_not_compiled() {
    // 未启用 candle feature 时，显式选择 candle 必须拒绝而不是静默降级
    let mut cfg = base_config();
    cfg.engine_type = EngineType::Candle;
    let result = EngineFactory::create(EngineType::Candle, &cfg);
    assert!(
        result.is_err(),
        "未编译 candle 时必须拒绝启动（不得静默降级 onnx）"
    );
}

#[test]
fn decision_task_with_missing_bundle_rejected() {
    let mut cfg = base_config();
    cfg.task = ModelTask::Decision;
    let result = EngineFactory::create(EngineType::Onnx, &cfg);
    assert!(result.is_err(), "决策任务缺 bundle 必须拒绝");
}

#[test]
fn transcribe_task_with_missing_bundle_rejected() {
    #[cfg(feature = "transcribe")]
    {
        let mut cfg = base_config();
        cfg.task = ModelTask::Transcription;
        let result = EngineFactory::create(EngineType::Onnx, &cfg);
        assert!(result.is_err(), "转录任务缺三图 bundle 必须拒绝");
    }
}
