// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 重排吞吐基线：单对 `rerank` 与批量 `rerank_batch`（1 query × 32 documents）。
//!
//! 重排不是独立 task（`ModelTask` 无 Rerank 变体），其成本由
//! [`InferenceEngine::rerank`] / [`InferenceEngine::rerank_batch`] 的默认实现派生
//! ——bi-encoder：query 与 documents 各 embed 一次 + 逐对 cosine + sigmoid。
//! 故本 bench 的读数与 `embed_throughput_bench` 同模型可直接对账
//! （单对应 2 次前向；批量对应 1 + N 次前向，而非 2N 次）。
//!
//! 模型经 `VECBOOST_BENCH_MODEL` 指定（未设置时 skip 并提示，`cargo bench` 不失败）；
//! 引擎经 `VECBOOST_BENCH_ENGINE`（`onnx`（缺省）/`candle`）；执行设备经
//! `VECBOOST_BENCH_DEVICE`（`cpu`（缺省）/`cuda`）。三者未知值均显性退出——
//! 静默回退会把基准数字记到错误档位名下（与 embed_throughput_bench 同一口径）。
//!
//! 基线数字记入 `docs/benchmarks/`，必须标注主机配置（CPU 型号/核数/内存）与
//! ONNX Runtime 动态库版本（跨库版本的速度与精度均不可外推）。
//!
//! 运行：`cargo bench --features onnx,candle --bench rerank_bench`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use vecboost::config::model::{EngineType, ModelConfig};
use vecboost::engine::{EngineFactory, InferenceEngine};

// 执行设备选择器（`VECBOOST_BENCH_DEVICE`，缺省 CPU）与同目录 bench 共享
#[path = "common/mod.rs"]
mod common;

fn bench_model_dir() -> Option<PathBuf> {
    std::env::var("VECBOOST_BENCH_MODEL")
        .ok()
        .map(PathBuf::from)
}

/// `VECBOOST_BENCH_ENGINE` → 引擎类型（缺省 Onnx；未知值显性退出）。
/// 与 `embed_throughput_bench` 同构，保证两 bench 的引擎口径一致可比。
fn bench_engine_type() -> EngineType {
    match std::env::var("VECBOOST_BENCH_ENGINE").as_deref() {
        Ok("candle") => {
            #[cfg(feature = "candle")]
            {
                EngineType::Candle
            }
            #[cfg(not(feature = "candle"))]
            {
                eprintln!("rerank_bench: candle 引擎需 --features candle 构建");
                std::process::exit(1);
            }
        }
        Ok(other) if other != "onnx" => {
            eprintln!(
                "rerank_bench: 未知 VECBOOST_BENCH_ENGINE={other}（可选值: onnx/candle），拒绝执行"
            );
            std::process::exit(1);
        }
        _ => {
            #[cfg(feature = "onnx")]
            {
                EngineType::Onnx
            }
            #[cfg(not(feature = "onnx"))]
            {
                eprintln!("rerank_bench: onnx 引擎需 --features onnx 构建");
                std::process::exit(1);
            }
        }
    }
}

fn load_engine(
    engine_type: EngineType,
    model_dir: &std::path::Path,
) -> Option<Box<dyn InferenceEngine>> {
    let config = ModelConfig {
        name: "bench-rerank".to_string(),
        engine_type: engine_type.clone(),
        model_path: model_dir.to_path_buf(),
        tokenizer_path: None,
        device: common::bench_device("rerank_bench", matches!(engine_type, EngineType::Candle)),
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        task: vecboost::config::model::ModelTask::Embedding,
        quantized: false,
        decision_params: None,
        translation_params: None,
        transcription_params: None,
    };
    match EngineFactory::create(engine_type, &config) {
        Ok(engine) => Some(Box::new(engine) as Box<dyn InferenceEngine>),
        Err(e) => {
            // Debug 形态携原始 detail（与 embed_throughput_bench 同口径：bench 可读性优先）
            eprintln!(
                "rerank_bench: 模型加载失败 {}: {:?}",
                model_dir.display(),
                e
            );
            None
        }
    }
}

fn documents(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "document candidate number {i}: retrieval augmented generation ranks passages \
                 against a user query before the model answers"
            )
        })
        .collect()
}

fn bench_rerank(c: &mut Criterion) {
    let Some(model_dir) = bench_model_dir() else {
        eprintln!(
            "rerank_bench: 未设置 VECBOOST_BENCH_MODEL，skip（设置本地模型目录后重跑以采集基线）"
        );
        return;
    };
    let engine_type = bench_engine_type();
    let Some(engine) = load_engine(engine_type.clone(), &model_dir) else {
        return;
    };
    // ID 含引擎名（同 embed_throughput_bench）：两引擎各跑各的基线，
    // criterion 不会把引擎/设备切换误报为同 ID 的 regression/improvement
    let engine_name = engine_type.to_string();
    let query = "how does retrieval augmented generation improve answer quality?".to_string();
    let docs = documents(32);
    let one_doc = docs[0].clone();

    let mut group = c.benchmark_group("rerank_throughput");
    // expect 硬防线：rerank 返回 Err 时的快速失败 µs 级数字会污染吞吐读数
    // （历史教训同 embed_throughput_bench：三输入图缺 token_type_ids 时 run 立即报错）
    group.bench_function(
        BenchmarkId::new(format!("{engine_name}_rerank_single"), "1"),
        |b| {
            b.iter(|| {
                engine
                    .rerank(black_box(&query), black_box(&one_doc))
                    .expect("bench rerank must succeed")
            })
        },
    );
    group.bench_function(
        BenchmarkId::new(format!("{engine_name}_rerank_batch"), "32"),
        |b| {
            b.iter(|| {
                engine
                    .rerank_batch(black_box(&query), black_box(&docs))
                    .expect("bench rerank_batch must succeed")
            })
        },
    );
    group.finish();
}

criterion_group!(benches, bench_rerank);
criterion_main!(benches);
