//! 吞吐基线基准：单文本 embed 与 32 文本 embed_batch 两场景。
//!
//! 模型经环境变量 `VECBOOST_BENCH_MODEL` 指定（本地模型目录）；未设置时
//! skip 并输出提示（`cargo bench` 不失败）。基线数字记入 `docs/benchmarks/`，
//! 必须标注主机配置（CPU 型号/核数/内存）。

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::path::PathBuf;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig};
use vecboost::engine::{EngineFactory, InferenceEngine};

fn bench_model_dir() -> Option<PathBuf> {
    std::env::var("VECBOOST_BENCH_MODEL")
        .ok()
        .map(PathBuf::from)
}

fn load_engine(model_dir: &std::path::Path) -> Option<Box<dyn InferenceEngine>> {
    let config = ModelConfig {
        name: "bench".to_string(),
        engine_type: EngineType::Candle,
        model_path: model_dir.to_path_buf(),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        quantized: false,
    };
    match EngineFactory::create(EngineType::Candle, &config) {
        Ok(engine) => Some(Box::new(engine) as Box<dyn InferenceEngine>),
        Err(e) => {
            eprintln!(
                "embed_throughput_bench: 模型加载失败 {}: {}",
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
    let Some(engine) = load_engine(&model_dir) else {
        return;
    };
    let single = "benchmark single text for throughput measurement".to_string();
    let batch = texts(32);

    let mut group = c.benchmark_group("embed_throughput");
    group.bench_function(BenchmarkId::new("single", "1"), |b| {
        b.iter(|| engine.embed(&single))
    });
    group.bench_function(BenchmarkId::new("batch", "32"), |b| {
        b.iter(|| engine.embed_batch(&batch))
    });
    group.finish();
}

criterion_group!(benches, bench_embed);
criterion_main!(benches);
