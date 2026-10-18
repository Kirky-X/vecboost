// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// 双引擎后端对比（onnx-primary：onnx 主引擎 vs candle 回退后端）——
// 回答"切换后精度与速度"的量化对账面，与 parity 闸门（决策 1e-4 对齐）
// 互补：本文件聚焦 embedding 数值一致性 + 决策性能观测。
//
// 1) `embedding_precision_parity`：同源模型（bge-small-en-v1.5）两格式导出
//    （BAAI safetensors vs Xenova onnx）逐文本余弦。两侧 pooling 经构造期
//    同一 `infer_pooling_mode` 解析（bge→Cls）——历史缺陷（onnx 硬编码
//    mean）已修复；阈值 0.999 严于任何 pooling/结构错配（mean vs cls 错配
//    实测掉至 0.7~0.95 区间），宽于框架层浮点实现差异（典型 1e-5~1e-3）。
// 2) `decision_engine_latency_informational`：同 laya 模型双实现（candle
//    原生决策头 vs onnx DecisionPipeline）单题 decide 耗时观测——仅打印
//    中位耗时不断言倍数（CI 环境噪声下性能断言不稳定）。
//
// 先决条件（缺失即 SKIP）：models/BAAI-bge-small-en-v1.5、
// models/Xenova-bge-small-en-v1.5、models/laya、models/laya-pytorch、
// 3rdparty/onnxruntime 动态库。
//
// ```text
// cargo test --features candle,onnx --test engine_backend_parity
// ```

#![cfg(all(feature = "onnx", feature = "candle"))]

use std::path::PathBuf;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, Precision};
use vecboost::domain::DecisionRequest;
use vecboost::engine::{AnyEngine, EngineFactory, InferenceEngine};

const ORT_DYLIB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnxruntime/libonnxruntime.so"
);

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

fn embedding_config(engine: EngineType, name: &str, path: &str) -> ModelConfig {
    ModelConfig {
        name: name.to_string(),
        engine_type: engine,
        model_path: PathBuf::from(path),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// embedding 精度对账：同源权重双导出，逐文本余弦 ≥ 0.999。
#[test]
fn embedding_precision_parity() {
    let candle_dir = PathBuf::from("models/BAAI-bge-small-en-v1.5");
    let onnx_dir = PathBuf::from("models/Xenova-bge-small-en-v1.5");
    if !candle_dir.join("model.safetensors").exists()
        || !onnx_dir.join("model.onnx").exists()
        || !PathBuf::from(ORT_DYLIB).exists()
    {
        eprintln!("SKIP embedding_precision_parity: M1/Xenova bundle 或 ORT 动态库缺失");
        return;
    }
    ensure_ort_dylib();
    vecboost::i18n::init();

    let candle = AnyEngine::new(
        &embedding_config(
            EngineType::Candle,
            "BAAI/bge-small-en-v1.5",
            "models/BAAI-bge-small-en-v1.5",
        ),
        EngineType::Candle,
        Precision::Fp32,
    )
    .expect("candle 引擎加载 M1 必须成功");
    let onnx = AnyEngine::new(
        &embedding_config(
            EngineType::Onnx,
            "Xenova/bge-small-en-v1.5",
            "models/Xenova-bge-small-en-v1.5",
        ),
        EngineType::Onnx,
        Precision::Fp32,
    )
    .expect("onnx 引擎加载 Xenova bundle 必须成功");

    let texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn representations from data.",
        "Retrieval augmented generation combines search with generation.",
        "A quiet afternoon in the park with a good book.",
        "Vector databases store high-dimensional embeddings for similarity search.",
    ];

    let mut min_cos = f32::MAX;
    let mut sum_cos = 0.0f32;
    for (i, text) in texts.iter().enumerate() {
        let v_candle = candle.embed(text).expect("candle embed");
        let v_onnx = onnx.embed(text).expect("onnx embed");
        assert_eq!(v_candle.len(), 384);
        assert_eq!(v_onnx.len(), 384);
        let c = cosine(&v_candle, &v_onnx);
        eprintln!("[{i}] cosine={c:.6}  text={text:.40}");
        min_cos = min_cos.min(c);
        sum_cos += c;
        assert!(
            c >= 0.999,
            "第 {i} 条同源模型双引擎余弦 {c:.6} < 0.999（pooling/结构错配排查）"
        );
    }
    let mean = sum_cos / texts.len() as f32;
    eprintln!(
        "embedding_precision_parity: min={min_cos:.6} mean={mean:.6} over {} texts",
        texts.len()
    );
}

/// ONNX 量化档（int8 / q4） vs fp32 的**精度对账**：同一 Xenova bundle 的
/// `-int8` / `-q4` 导出与 fp32 基线逐文本余弦（同模型、同 pooling、同输入）。
/// 阈值沿用仓内量化口径（`tests/quantized_parity.rs` 的 GGUF 门：Q8_0 ≥ 0.98、
/// Q4_K ≥ 0.95），并打印 min/median 供基线文档引用；维度不一致直接失败
/// （量化不应改输出维度）。任一资产缺失即 SKIP（大体积资产不入库）。
#[test]
fn onnx_quantized_vs_fp32_parity() {
    let fp32 = PathBuf::from("models/Xenova-bge-small-en-v1.5");
    let int8 = PathBuf::from("models/Xenova-bge-small-en-v1.5-int8");
    let q4 = PathBuf::from("models/Xenova-bge-small-en-v1.5-q4");
    if !fp32.join("model.onnx").exists() || !PathBuf::from(ORT_DYLIB).exists() {
        eprintln!("SKIP onnx_quantized_vs_fp32_parity: fp32 bundle 或 ORT 动态库缺失");
        return;
    }
    ensure_ort_dylib();
    vecboost::i18n::init();

    fn load(path: &str) -> AnyEngine {
        AnyEngine::new(
            &embedding_config(EngineType::Onnx, path, path),
            EngineType::Onnx,
            Precision::Fp32,
        )
        .expect("量化档加载必须成功")
    }

    let texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn representations from data.",
        "Retrieval augmented generation combines search with generation.",
        "A quiet afternoon in the park with a good book.",
        "Vector databases store high-dimensional embeddings for similarity search.",
    ];
    let base = load("models/Xenova-bge-small-en-v1.5");
    let base_vecs: Vec<Vec<f32>> = texts
        .iter()
        .map(|t| base.embed(t).expect("fp32 embed"))
        .collect();

    for (label, dir, gate) in [("int8", &int8, 0.98f32), ("q4", &q4, 0.95f32)] {
        if !dir.join("model.onnx").exists() {
            eprintln!("SKIP {label}: 资产缺失 {}", dir.display());
            continue;
        }
        let engine = load(dir.to_str().expect("资产路径须为 UTF-8"));
        let mut cos = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let v = engine.embed(text).expect("quantized embed");
            assert_eq!(
                v.len(),
                base_vecs[i].len(),
                "{label} 输出维度必须与 fp32 一致"
            );
            let c = cosine(&base_vecs[i], &v);
            eprintln!("[{label}][{i}] cosine={c:.6}");
            cos.push(c);
        }
        cos.sort_by(|a, b| a.partial_cmp(b).expect("余弦不得为 NaN"));
        let min = cos[0];
        let median = cos[cos.len() / 2];
        eprintln!("onnx_quantized_vs_fp32_parity[{label}]: min={min:.6} median={median:.6}");
        assert!(
            median >= gate,
            "{label} 中位余弦 {median:.6} < {gate}（量化损伤超阈值）"
        );
    }
}

/// candle 单条 vs 单元素批量路径耗时对比（informational）：历史基准中
/// 单条 72ms / batch32 单价 5.5ms 的 13× 差异——实测结论（2026-10-16）：
/// single 与 batch1 耗时完全相等（85.56ms/85.56ms，median×10）——单条路径
/// 无固定开销，差异根源是 **candle CPU 对小 batch 前向的固有低效**（大 batch
/// 摊薄 GEMM/逐算子开销），非路径缺陷；candle 已降为回退后端，不投入
/// 引擎内部优化，本测试保留为长期观测。
#[test]
fn candle_single_vs_batch_path_informational() {
    if !PathBuf::from("models/BAAI-bge-small-en-v1.5/model.safetensors").exists() {
        eprintln!("SKIP candle_single_vs_batch_path_informational: M1 缺失");
        return;
    }
    vecboost::i18n::init();
    let engine = AnyEngine::new(
        &embedding_config(
            EngineType::Candle,
            "BAAI/bge-small-en-v1.5",
            "models/BAAI-bge-small-en-v1.5",
        ),
        EngineType::Candle,
        Precision::Fp32,
    )
    .expect("candle 引擎加载必须成功");

    let text = "benchmark single text for throughput measurement".to_string();
    const N: usize = 10;

    let mut single_samples = Vec::with_capacity(N);
    for _ in 0..N {
        let t = std::time::Instant::now();
        engine.embed(&text).expect("embed");
        single_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut batch_samples = Vec::with_capacity(N);
    for _ in 0..N {
        let t = std::time::Instant::now();
        engine
            .embed_batch(std::slice::from_ref(&text))
            .expect("embed_batch x1");
        batch_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    single_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    batch_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "candle single={:.2}ms batch1={:.2}ms (median over {N}; informational)",
        single_samples[N / 2],
        batch_samples[N / 2]
    );
}

/// 决策性能 informational：双实现单题 decide 中位耗时（仅打印，不断言倍数）。
#[test]
fn decision_engine_latency_informational() {
    let candle_ckpt = PathBuf::from("models/laya-pytorch/model.safetensors");
    let onnx_bundle = PathBuf::from("models/laya/laya.onnx");
    if !candle_ckpt.exists() || !onnx_bundle.exists() || !PathBuf::from(ORT_DYLIB).exists() {
        eprintln!(
            "SKIP decision_engine_latency_informational: laya checkpoint/bundle 或 ORT 动态库缺失"
        );
        return;
    }
    ensure_ort_dylib();
    vecboost::i18n::init();

    let decision_config = |engine: EngineType, path: &str| ModelConfig {
        engine_type: engine.clone(),
        model_path: PathBuf::from(path),
        task: ModelTask::Decision,
        expected_dimension: Some(1024),
        ..embedding_config(engine, "laya-multilingual", path)
    };

    // 决策任务必须经 EngineFactory（task 分派臂选决策引擎）；
    // AnyEngine::new 是 embedding 分派，拿 decision config 会走错加载流程
    let candle = EngineFactory::create(
        EngineType::Candle,
        &decision_config(EngineType::Candle, "models/laya-pytorch"),
    )
    .expect("candle 决策引擎加载必须成功");
    let onnx = EngineFactory::create(
        EngineType::Onnx,
        &decision_config(EngineType::Onnx, "models/laya"),
    )
    .expect("onnx 决策引擎加载必须成功");

    let req: DecisionRequest = serde_json::from_str(
        r#"{"state":"a quiet afternoon in the park","questions":[{"name":"confident","qtype":"noul","instructions":"state your p(true)","options":[]}]}"#,
    )
    .expect("decision request");

    const N: usize = 20;
    let measure = |engine: &AnyEngine, label: &str| {
        let mut samples = Vec::with_capacity(N);
        for _ in 0..N {
            let t = std::time::Instant::now();
            let resp = engine.decide(&req).expect("decide must succeed");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(resp.answers.len(), 1);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[N / 2];
        eprintln!(
            "decision {label}: median={median:.2}ms over {N} runs (min={:.2}ms)",
            samples[0]
        );
        median
    };
    let candle_ms = measure(&candle, "candle-native-head");
    let onnx_ms = measure(&onnx, "onnx-pipeline");
    eprintln!(
        "decision_engine_latency_informational: candle={candle_ms:.2}ms onnx={onnx_ms:.2}ms (informational)"
    );
}
