// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 吞吐基线基准：单文本 embed 与 32 文本 embed_batch 两场景。
//!
//! 模型经环境变量 `VECBOOST_BENCH_MODEL` 指定（本地模型目录或 HF repo id）；
//! 未设置时 skip 并输出提示（`cargo bench` 不失败）。
//! 引擎经 `VECBOOST_BENCH_ENGINE` 选择（`onnx`（默认，本地 bundle 目录或 repo id
//! 经 HF 拉取，如 Xenova/bge-small-en-v1.5）/`candle`（safetensors 本地目录，opt-in
//! 回退后端））；未知值显性退出而非静默回退——回退会把基准数字记到错误引擎名下。
//! 两引擎同机对比时保持模型同源（bge-small-en-v1.5 系列）与相同输入集。
//! 基线数字记入 `docs/benchmarks/`，必须标注主机配置（CPU 型号/核数/内存）。

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

/// `VECBOOST_BENCH_ENGINE` → 引擎类型（onnx-primary：缺省 Onnx；未知值显性
/// 退出而非静默回退——静默回退会把基准数字记到错误引擎名下）。
fn bench_engine_type() -> EngineType {
    match std::env::var("VECBOOST_BENCH_ENGINE").as_deref() {
        Ok("candle") => {
            #[cfg(feature = "candle")]
            {
                EngineType::Candle
            }
            #[cfg(not(feature = "candle"))]
            {
                eprintln!("embed_throughput_bench: candle 引擎需 --features candle 构建");
                std::process::exit(1);
            }
        }
        // 未设置（Err(NotPresent)）与 "onnx" → Onnx（默认主引擎，onnx-primary
        // 对齐 main.rs resolve_engine_model_config 的 None 缺省臂）；其余值显性拒绝
        Ok(other) if other != "onnx" => {
            eprintln!(
                "embed_throughput_bench: 未知 VECBOOST_BENCH_ENGINE={other}（可选值: onnx/candle），拒绝执行"
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
                eprintln!("embed_throughput_bench: onnx 引擎需 --features onnx 构建");
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
        name: "bench".to_string(),
        engine_type: engine_type.clone(),
        model_path: model_dir.to_path_buf(),
        tokenizer_path: None,
        device: common::bench_device(
            "embed_throughput_bench",
            matches!(engine_type, EngineType::Candle),
        ),
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
            // 错误文案链依赖 i18n 初始化（未 init 时 Display 输出键名）；
            // Debug 形态携原始 detail，bench 可读性优先
            eprintln!(
                "embed_throughput_bench: 模型加载失败 {}: {:?}",
                model_dir.display(),
                e
            );
            None
        }
    }
}

fn texts(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("benchmark text number {} for throughput measurement", i))
        .collect()
}

fn bench_embed(c: &mut Criterion) {
    let Some(model_dir) = bench_model_dir() else {
        eprintln!(
            "embed_throughput_bench: 未设置 VECBOOST_BENCH_MODEL，skip（设置本地模型目录后重跑以采集基线）"
        );
        return;
    };
    let engine_type = bench_engine_type();
    let Some(engine) = load_engine(engine_type.clone(), &model_dir) else {
        return;
    };
    // ID 含引擎名（随 EngineType Display 派生，与引擎选择单一事实源）：
    // candle/onnx 两次跑各自独立基线，criterion 不把引擎切换误报为
    // 同 ID 的 regression/improvement
    let engine_name = engine_type.to_string();
    let single = "benchmark single text for throughput measurement".to_string();
    let batch = texts(32);

    let mut group = c.benchmark_group("embed_throughput");
    // expect 硬防线：embed 返回 Err 时快速失败的 µs 级耗时会污染吞吐数字
    // （历史教训：三输入 ONNX 图缺 token_type_ids 时 run 立即报错，
    // criterion 测到的是错误路径）；基准数字必须来自成功推理
    group.bench_function(
        BenchmarkId::new(format!("{engine_name}_single"), "1"),
        |b| b.iter(|| engine.embed(&single).expect("bench embed must succeed")),
    );
    group.bench_function(
        BenchmarkId::new(format!("{engine_name}_batch"), "32"),
        |b| {
            b.iter(|| {
                engine
                    .embed_batch(&batch)
                    .expect("bench embed_batch must succeed")
            })
        },
    );
    group.finish();
}

criterion_group!(benches, bench_embed);
criterion_main!(benches);
