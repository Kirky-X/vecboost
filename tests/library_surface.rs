// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Library 模式集成测试：`VecBoostLibrary` 直接 SDK 面（无 HTTP/gRPC 进程）。
//!
//! 覆盖 `src/library/mod.rs` 的异步与同步入口、批量聚合与空输入错误分支；
//! 模型缺席时 SKIP。

#![cfg(feature = "onnx")]

use std::path::Path;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::library::{LibraryConfig, VecBoostLibrary};

const MODEL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/models/Xenova-bge-small-en-v1.5"
);

fn model_ready() -> bool {
    Path::new(MODEL_PATH).join("model.onnx").exists()
        && Path::new(MODEL_PATH).join("tokenizer.json").exists()
}

fn model_config() -> ModelConfig {
    let mut cfg = ModelConfig::default();
    cfg.name = "Xenova/bge-small-en-v1.5".to_string();
    cfg.engine_type = EngineType::Onnx;
    cfg.model_path = MODEL_PATH.into();
    cfg.device = DeviceType::Cpu;
    cfg.max_batch_size = 8;
    cfg.expected_dimension = Some(384);
    cfg.task = ModelTask::Embedding;
    cfg
}

#[tokio::test]
async fn library_async_and_sync_surface() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失（models/Xenova-bge-small-en-v1.5）");
        return;
    }
    // SAFETY: 单测进程内独占设置 ORT dylib 路径，无并发读写该变量。
    unsafe {
        std::env::set_var(
            "ORT_DYLIB_PATH",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/3rdparty/onnxruntime/libonnxruntime.so"
            ),
        );
    }

    let lib = VecBoostLibrary::new(LibraryConfig::from_model_config(model_config()))
        .await
        .expect("library init");

    let single = lib.embed("hello vecboost").await.expect("embed");
    assert_eq!(single.embedding.len(), 384);

    let batch = lib
        .embed_batch(&["a".to_string(), "b".to_string()])
        .await
        .expect("batch embed");
    assert_eq!(batch.embeddings.len(), 2);

    // 空输入必须拒绝（错误分支）
    assert!(lib.embed("").await.is_err(), "空文本必须拒绝");
    assert!(lib.embed_batch(&[]).await.is_err(), "空批次必须拒绝");

    // rerank 未配置重排模型时走错误分支，不应 panic
    let _ = lib.rerank("query", &["doc".to_string()], Some(1)).await;

    // 同步入口在 tokio 运行时内必须拒绝（防 deadlock 的显式守卫）
    let sync_err = lib.embed_sync("sync path").unwrap_err();
    let sync_debug = format!("{sync_err:?}");
    assert!(
        sync_debug.contains("tokio runtime"),
        "同步 API 在运行时内应拒绝: {sync_debug}"
    );
    let batch_err = lib.embed_batch_sync(&["x".to_string()]).unwrap_err();
    assert!(format!("{batch_err:?}").contains("tokio runtime"));
}

/// 同步入口在运行时之外可用（独立 runtime + block_on）。
#[test]
fn library_sync_api_outside_runtime() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失（models/Xenova-bge-small-en-v1.5）");
        return;
    }
    unsafe {
        std::env::set_var(
            "ORT_DYLIB_PATH",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/3rdparty/onnxruntime/libonnxruntime.so"
            ),
        );
    }
    let lib = {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(VecBoostLibrary::new(LibraryConfig::from_model_config(
            model_config(),
        )))
        .expect("library init")
    };
    let single = lib.embed_sync("sync path").expect("embed_sync");
    assert_eq!(single.embedding.len(), 384);
    let batch = lib
        .embed_batch_sync(&["x".to_string()])
        .expect("embed_batch_sync");
    assert_eq!(batch.embeddings.len(), 1);
}
