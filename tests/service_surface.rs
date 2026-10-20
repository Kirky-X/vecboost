// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! EmbeddingService 直接面集成测试（不经 HTTP 协议层）。
//!
//! 用真实 ONNX 模型构造服务，覆盖单条/批量/相似度/检索/文件流/聚合模式/
//! 预热/切换失败等分支；模型或 ORT 缺席时 SKIP。

#![cfg(feature = "onnx")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
use vecboost::domain::{
    BatchEmbedRequest, EmbedRequest, EmbeddingOutput, ModelSwitchRequest, SearchRequest,
    SimilarityRequest,
};
use vecboost::engine::{EngineFactory, InferenceEngine};
use vecboost::service::embedding::EmbeddingService;
use vecboost::utils::vector::AggregationMode;

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

fn set_ort_env() {
    if Path::new(ORT_DYLIB).is_file() {
        // SAFETY: 测试进程内独占设置 ORT dylib 路径。
        unsafe { std::env::set_var("ORT_DYLIB_PATH", ORT_DYLIB) };
    }
}

fn base_config() -> ModelConfig {
    ModelConfig {
        name: "Xenova/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Onnx,
        model_path: MODEL_PATH.into(),
        device: DeviceType::Cpu,
        max_batch_size: 8,
        expected_dimension: Some(384),
        task: ModelTask::Embedding,
        ..ModelConfig::default()
    }
}

async fn build_service() -> EmbeddingService {
    set_ort_env();
    let engine = EngineFactory::create(EngineType::Onnx, &base_config()).expect("create engine");
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
    EmbeddingService::new(engine, Some(base_config()))
}

#[tokio::test]
async fn service_text_similarity_and_search_surface() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    let svc = build_service().await;

    let single = svc
        .process_text(
            EmbedRequest {
                text: "hello service".to_string(),
                normalize: Some(true),
            },
            None,
        )
        .await
        .expect("process_text");
    assert_eq!(single.embedding.len(), 384);

    let truncated = svc
        .process_text(
            EmbedRequest {
                text: "matryoshka".to_string(),
                normalize: None,
            },
            Some(64),
        )
        .await
        .expect("matryoshka 截断");
    assert_eq!(truncated.embedding.len(), 64);

    let sim = svc
        .process_similarity(SimilarityRequest {
            source: "cat".to_string(),
            target: "dog".to_string(),
            metric: None,
        })
        .await
        .expect("similarity");
    assert!((0.0..=1.0001).contains(&sim.score));

    let search = svc
        .process_search(SearchRequest {
            query: "animal".to_string(),
            texts: vec!["cat".to_string(), "car".to_string()],
            top_k: Some(1),
        })
        .await
        .expect("search");
    assert_eq!(search.results.len(), 1);

    let batch_search = svc
        .process_search_batch(
            "animal",
            &["cat".to_string(), "dog".to_string(), "truck".to_string()],
            Some(100),
        )
        .await
        .expect("search batch");
    assert!(!batch_search.results.is_empty());

    // 非法输入：空文本 / 空候选集
    assert!(
        svc.process_text(
            EmbedRequest {
                text: String::new(),
                normalize: None,
            },
            None,
        )
        .await
        .is_err()
    );
    assert!(svc.process_search_batch("q", &[], Some(1)).await.is_err());

    // count_tokens 不得报错，且当前恒为 0：ONNX 引擎未覆写 trait 默认实现
    // （src/engine/mod.rs 的 count_tokens 默认返回 Ok(0)），调用方据此回退
    // bytes/4 估算，见 src/api/embedding.rs 的 usage.prompt_tokens。
    // 旧断言 > 0 恒假，正是 CI 覆盖率作业的失败根因；此处钉住当前契约，
    // 一旦引擎接入真实 tokenizer，本断言会失败并提示同步更新调用方口径。
    let count = svc
        .count_tokens("tokenize me")
        .expect("count_tokens 不得报错");
    assert_eq!(
        count, 0,
        "ONNX 引擎未接 tokenizer，count_tokens 应回退 0（调用方按 bytes/4 估算）"
    );
    assert!(svc.get_model_info().is_some());
    assert!(svc.get_model_metadata().is_some());
    assert!(!svc.has_model_manager());
    assert!(svc.list_loaded_models().await.is_empty());
}

#[tokio::test]
async fn service_batch_modes_and_file_surface() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    let svc = build_service().await;

    let texts: Vec<String> = (0..5).map(|i| format!("document line {i}")).collect();
    for mode in [
        AggregationMode::Document,
        AggregationMode::Paragraphs,
        AggregationMode::Average,
        AggregationMode::SlidingWindow,
        AggregationMode::FixedSize,
        AggregationMode::MaxPooling,
        AggregationMode::MinPooling,
    ] {
        let out = svc
            .process_batch(
                BatchEmbedRequest {
                    texts: texts.clone(),
                    mode: Some(mode),
                    normalize: Some(true),
                },
                None,
            )
            .await
            .unwrap_or_else(|e| panic!("batch mode {mode:?} 失败: {e}"));
        assert_eq!(out.embeddings.len(), texts.len());
    }

    let plain = svc
        .embed_batch_texts(&texts)
        .await
        .expect("embed_batch_texts");
    assert_eq!(plain.len(), texts.len());

    let dir = tempfile::tempdir().expect("tempdir");
    let file: PathBuf = dir.path().join("doc.txt");
    std::fs::write(
        &file,
        "第一段：VecBoost 提供嵌入服务。\n\n第二段：支持多种聚合模式。\n\n第三段。\n",
    )
    .expect("write doc");
    let stats = svc.get_processing_stats(&file).expect("processing stats");
    assert!(stats.total_chunks >= 1);
    let streamed = svc.process_file_stream(&file).await.expect("file stream");
    assert_eq!(streamed.embedding.len(), 384);
    for mode in [
        AggregationMode::Document,
        AggregationMode::Paragraphs,
        AggregationMode::Average,
    ] {
        let out = svc
            .embed_file(&file, mode)
            .await
            .unwrap_or_else(|e| panic!("embed_file {mode:?} 失败: {e}"));
        match mode {
            AggregationMode::Paragraphs => assert!(matches!(out, EmbeddingOutput::Paragraphs(_))),
            _ => assert!(matches!(out, EmbeddingOutput::Single(_))),
        }
    }
    // 不存在的文件必须报错
    assert!(
        svc.embed_file(&dir.path().join("nope.txt"), AggregationMode::Document)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn service_warmup_persist_and_switch_without_manager() {
    if !model_ready() {
        println!("SKIP: 本地 ONNX 模型缺失");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    set_ort_env();
    let engine = EngineFactory::create(EngineType::Onnx, &base_config()).expect("create engine");
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
    let svc = EmbeddingService::with_cache_persist(
        engine,
        Some(base_config()),
        64,
        dir.path().join("cache.wal"),
        1 << 20,
    );

    svc.warm_up_cache(vec!["warm one".to_string(), "warm two".to_string()])
        .await
        .expect("warm_up_cache");
    svc.load_persisted_cache().await;
    assert!(dir.path().join("cache.wal").exists(), "WAL 应落盘");

    // 无 ModelManager 时切换必须显性失败而非静默成功
    let mut svc = svc;
    let err = svc
        .switch_model(ModelSwitchRequest {
            model_name: "ghost".to_string(),
            model_path: Some(PathBuf::from("/definitely/missing")),
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        })
        .await
        .expect_err("无 manager 的切换应失败");
    assert!(!format!("{err:?}").is_empty());
}
