// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![cfg(feature = "quantized-gguf")]
//! 量化质量门（feature `quantized-gguf`）：golden 语料上量化 vs
//! fp32 余弦相似度中位数 Q8_0 ≥ 0.98、Q4_K ≥ 0.95 方可发布为推荐配置。
//!
//! 模型来源优先级：
//! 1. env `VECBOOST_GGUF_MODEL`（真实 llama.cpp 转换文件，走 gguf 加载路径）；
//! 2. 本地 `models/BAAI-bge-small-en-v1.5` 自产 GGUF（写出门 + 加载桥端到端，
//!    零网络依赖；feature `quantized-gguf` 必须启用）。
//! 两者都不可用时 skip。余弦计算内嵌实现，不依赖 GPU 与外部服务。

use std::path::{Path, PathBuf};
use std::time::Instant;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn load_corpus() -> Vec<String> {
    let text = std::fs::read_to_string("tests/fixtures/golden_corpus.txt")
        .expect("golden 语料 tests/fixtures/golden_corpus.txt 必须存在");
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    assert!(
        lines.len() >= 32,
        "golden 语料至少 32 条，实际 {} 条",
        lines.len()
    );
    lines
}

#[test]
fn test_cosine_median_helper() {
    let v = vec![1.0f32, 0.0, 0.0];
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    assert!((median(vec![0.1, 0.9, 0.5]) - 0.5).abs() < 1e-12);
    assert!((median(vec![0.1, 0.9]) - 0.5).abs() < 1e-12);
}

#[test]
fn test_golden_corpus_has_32_mixed_lines() {
    let lines = load_corpus();
    assert!(lines.len() >= 32);
    assert!(
        lines
            .iter()
            .any(|l| l.chars().any(|c| c as u32 > 0x4E00 && (c as u32) < 0x9FFF))
    );
    assert!(lines.iter().any(|l| l.chars().all(|c| (c as u32) < 128)));
}

fn local_model_dir() -> Option<PathBuf> {
    let dir = Path::new("models/BAAI-bge-small-en-v1.5");
    if dir.join("model.safetensors").exists() && dir.join("config.json").exists() {
        Some(dir.to_path_buf())
    } else {
        None
    }
}

/// 按本地模型自产指定 dtype 的 GGUF（质量门用）。
fn self_produce(name: &str) -> Option<PathBuf> {
    let dir = local_model_dir()?;
    let dtype = match name {
        "q8_0" => candle_core::quantized::GgmlDType::Q8_0,
        "q4_k" => candle_core::quantized::GgmlDType::Q4K,
        _ => unreachable!("未知的 dtype 名"),
    };
    // 写进模型目录：引擎从 GGUF 同目录加载 tokenizer.json（约定）
    let out = dir.join(format!("vecboost-parity-bge-small-{name}.gguf"));
    if !out.exists() {
        let stats =
            vecboost::engine::quantized_engine::write_gguf_from_safetensors(&dir, &out, dtype)
                .unwrap_or_else(|e| panic!("自产 GGUF({name}) 失败: {e:?}"));
        eprintln!(
            "quantized_parity: 自产 {name} 完成（二维张量 {}，F16 兜底 {}）",
            stats.total_tensors, stats.f16_fallback
        );
    }
    Some(out)
}

fn fp32_engine(model_path: &Path) -> AnyEngine {
    let config = ModelConfig {
        name: "parity-fp32".to_string(),
        engine_type: EngineType::Candle,
        model_path: model_path.to_path_buf(),
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
    AnyEngine::new(&config, EngineType::Candle, Precision::Fp32).expect("fp32 引擎加载失败")
}

fn quantized_engine(gguf_path: &Path) -> AnyEngine {
    let config = ModelConfig {
        name: "parity-quantized".to_string(),
        engine_type: EngineType::Candle,
        model_path: gguf_path.to_path_buf(),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
        quantized: true,
    };
    // 必须经 EngineFactory（量化路由在此）；AnyEngine::new 是不走路由的原始构造
    vecboost::engine::EngineFactory::create(EngineType::Candle, &config)
        .expect("量化引擎加载失败（路由→QuantizedCandleEngine）")
}

/// 跑质量门：返回余弦中位数与逐条余弦。
fn run_gate(gguf: &Path) -> (f64, Vec<f64>) {
    let corpus = load_corpus();
    let fp32 = fp32_engine(Path::new("models/BAAI-bge-small-en-v1.5"));
    let quant = quantized_engine(gguf);
    let started = Instant::now();
    let reference: Vec<Vec<f32>> = corpus
        .iter()
        .map(|t| InferenceEngine::embed(&fp32, t).expect("fp32 embed 失败"))
        .collect();
    let quantized: Vec<Vec<f32>> = corpus
        .iter()
        .map(|t| InferenceEngine::embed(&quant, t).expect("量化 embed 失败"))
        .collect();
    let cosines: Vec<f64> = reference
        .iter()
        .zip(quantized.iter())
        .map(|(a, b)| cosine(a, b))
        .collect();
    let med = median(cosines.clone());
    eprintln!(
        "quantized_parity: {} 条语料、{} 耗时 {:.1}s，余弦中位数 {:.4}（min {:.4}）",
        corpus.len(),
        gguf.file_name().unwrap_or_default().to_string_lossy(),
        started.elapsed().as_secs_f32(),
        med,
        cosines.iter().cloned().fold(f64::INFINITY, f64::min),
    );
    (med, cosines)
}

/// Q8_0 质量门：余弦中位数 ≥ 0.98。
#[test]
fn test_q8_0_parity_gate() {
    let Some(gguf) = std::env::var("VECBOOST_GGUF_MODEL")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| self_produce("q8_0"))
    else {
        eprintln!("quantized_parity: 无 VECBOOST_GGUF_MODEL 且本地 bge-small 缺失，Q8_0 门 skip");
        return;
    };
    let (med, _) = run_gate(&gguf);
    assert!(
        med >= 0.98,
        "Q8_0 质量门未通过：余弦中位数 {med:.4} < 0.98（不得标注为推荐配置）"
    );
}

/// Q4_K 质量门：余弦中位数 ≥ 0.95。
#[test]
fn test_q4_k_parity_gate() {
    let Some(gguf) = self_produce("q4_k") else {
        eprintln!(
            "quantized_parity: 本地 bge-small 缺失，Q4_K 门 skip（提供 VECBOOST_GGUF_MODEL 或模型目录）"
        );
        return;
    };
    let (med, _) = run_gate(&gguf);
    assert!(
        med >= 0.95,
        "Q4_K 质量门未通过：余弦中位数 {med:.4} < 0.95（不得标注为推荐配置）"
    );
}
