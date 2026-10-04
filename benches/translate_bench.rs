// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 翻译与语言检测基线基准：langdetect（短文本 / 10k 上限文本）、翻译端到端
//! （en→zh 短/中/长、en→es 短）、tokenizer 计数。
//!
//! 构建约束：`[[bench]]` 配置 `required-features = ["onnx"]`——`EngineType::Onnx`
//! 为 cfg 门控变体，且 harness = false 目标是无 harness 注入的普通 bin，
//! 缺 feature 时必须整体跳过而非清空编译（否则产出无 main 的空 bin）。
//! dylib/bundle 缺失时运行期 skip 并输出提示（`cargo bench` 不失败，
//! embed_throughput_bench 同款语义）。
//! 基线数字记入 `docs/benchmarks/translate_baseline_results.md`，
//! 必须标注主机配置（CPU 型号/核数/内存）。
//!
//! 运行：`cargo bench --features onnx --bench translate_bench`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, TranslationParams};
use vecboost::detect_language;
use vecboost::domain::TranslateRequest;
use vecboost::engine::{AnyEngine, EngineFactory, InferenceEngine};

const EN_ZH_BUNDLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-zh");
const EN_ES_BUNDLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/opus-mt-en-es");

fn dylib_ready() -> Option<String> {
    let dylib = std::env::var("ORT_DYLIB_PATH")
        .unwrap_or_else(|_| "3rdparty/onnxruntime/libonnxruntime.so".to_string());
    if Path::new(&dylib).is_file() {
        Some(dylib)
    } else {
        None
    }
}

fn bundle_ready(path: &str) -> bool {
    Path::new(path).join("onnx/encoder_model.onnx").exists()
        && Path::new(path)
            .join("onnx/decoder_model_merged.onnx")
            .exists()
        && Path::new(path).join("tokenizer.json").exists()
}

fn load_pair_engine(bundle_path: &str) -> AnyEngine {
    let config = ModelConfig {
        name: Path::new(bundle_path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from(bundle_path),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 1,
        oom_fallback_enabled: false,
        task: ModelTask::Translate,
        translation_params: Some(TranslationParams::default()),
        ..Default::default()
    };
    EngineFactory::create(EngineType::Onnx, &config).expect("bundle 就绪时必须可加载")
}

fn bench_langdetect(c: &mut Criterion) {
    let short = "The weather is nice today.".to_string();
    let max_len = "El equipo internacional publicó un estudio amplio sobre el cambio climático. "
        .repeat(101)
        .chars()
        .take(10_000)
        .collect::<String>();

    let mut group = c.benchmark_group("langdetect");
    group.bench_function(BenchmarkId::new("short", short.len()), |b| {
        b.iter(|| black_box(detect_language(black_box(&short), None)))
    });
    group.bench_function(BenchmarkId::new("max-10k", max_len.len()), |b| {
        b.iter(|| black_box(detect_language(black_box(&max_len), None)))
    });
    group.finish();
}

fn bench_translate(c: &mut Criterion) {
    let Some(dylib) = dylib_ready() else {
        eprintln!(
            "translate_bench: onnxruntime dylib not found（设置 ORT_DYLIB_PATH 或落位 \
             3rdparty/onnxruntime/libonnxruntime.so），skip"
        );
        return;
    };
    for path in [EN_ZH_BUNDLE, EN_ES_BUNDLE] {
        if !bundle_ready(path) {
            eprintln!("translate_bench: translation bundle missing: {path}，skip");
            return;
        }
    }
    // SAFETY: 两个前提共同保证 env 回填先于任何 env 读取——criterion 组按
    // 声明顺序在主线程串行执行（langdetect 组在前、本组在后），且 ORT
    // Session（含其线程池）在本函数引擎创建（下方 load_pair_engine）时才
    // 启动、晚于本次 set_var；重排组顺序或引入并行执行前必须重新评估
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &dylib);
        }
    }

    let engine_zh = load_pair_engine(EN_ZH_BUNDLE);
    let engine_es = load_pair_engine(EN_ES_BUNDLE);

    let short = "The weather is nice today.".to_string();
    let medium =
        "She walked to the market and bought fresh bread and vegetables for dinner.".to_string();
    let long = "The international research team published a comprehensive study on climate \
                change, highlighting the urgent need for global cooperation and sustainable \
                development policies across all industrial sectors."
        .to_string();

    // 翻译解码为 O(L²) 重算（已知限制 4），秒级单次调用：小样本 + 加长热身，
    // 避免 criterion 缺省 100 样本跑数十分钟
    let mut group = c.benchmark_group("translate");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(30));

    let bench_pair = |group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                      id: &str,
                      len: usize,
                      engine: &AnyEngine,
                      text: &str,
                      target: &str| {
        group.bench_function(BenchmarkId::new(id, len), |b| {
            b.iter(|| {
                black_box(
                    engine
                        .translate(&TranslateRequest {
                            text: black_box(text.to_string()),
                            source_language: Some("en".to_string()),
                            target_language: target.to_string(),
                        })
                        .expect("bundle 就绪时翻译必须成功"),
                )
            })
        });
    };
    bench_pair(
        &mut group,
        "en-zh/short",
        short.len(),
        &engine_zh,
        &short,
        "zh",
    );
    bench_pair(
        &mut group,
        "en-zh/medium",
        medium.len(),
        &engine_zh,
        &medium,
        "zh",
    );
    bench_pair(
        &mut group,
        "en-zh/long",
        long.len(),
        &engine_zh,
        &long,
        "zh",
    );
    bench_pair(
        &mut group,
        "en-es/short",
        short.len(),
        &engine_es,
        &short,
        "es",
    );
    group.bench_function(BenchmarkId::new("en-zh/count_tokens", medium.len()), |b| {
        b.iter(|| {
            engine_zh
                .count_tokens(black_box(&medium))
                .expect("bundle 就绪时计数必须成功")
        })
    });
    group.finish();
}

criterion_group!(benches, bench_langdetect, bench_translate);
criterion_main!(benches);
