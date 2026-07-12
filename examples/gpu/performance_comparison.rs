// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::path::PathBuf;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::engine::{AnyEngine, InferenceEngine};

mod utils;

#[cfg(feature = "onnx")]
fn run_onnx_test(
    model_path: &str,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, f64), Box<dyn std::error::Error>> {
    let model_config = ModelConfig {
        name: "bge-m3".to_string(),
        engine_type: EngineType::Onnx,
        model_path: PathBuf::from(model_path),
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(1024),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    let mut engine = AnyEngine::new(&model_config, EngineType::Onnx, Precision::Fp32)?;
    let (embeddings, duration) = utils::measure_time(|| engine.embed_batch(texts))?;
    Ok((embeddings, duration))
}

fn run_candle_test(
    model_path: &str,
    texts: &[String],
) -> Result<(Vec<Vec<f32>>, f64), Box<dyn std::error::Error>> {
    let config = ModelConfig {
        name: "bge-m3".to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from(model_path),
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(1024),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    let engine = CandleEngine::new(&config, Precision::Fp16)?;
    let (embeddings, duration) = utils::measure_time(|| engine.embed_batch(texts))?;
    Ok((embeddings, duration))
}

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU 引擎性能对比测试");
    println!("=========================\n");

    if !utils::check_cuda_available() {
        println!("❌ CUDA 不可用，请确保已安装 CUDA 驱动和 cuDNN");
        return Ok(());
    }

    let model_path = match utils::find_model_path(utils::DEFAULT_MODEL_PATHS) {
        Some(path) => path,
        None => {
            println!("❌ 未找到模型文件");
            return Ok(());
        }
    };

    let texts = utils::get_test_texts();
    println!("📊 测试配置:");
    println!("  文本数量: {}", texts.len());
    println!("  模型路径: {}", model_path);

    println!("\n🔧 测试 Candle 引擎...");
    let (candle_embeddings, candle_duration) = run_candle_test(&model_path, &texts)?;
    let candle_metrics = utils::calculate_metrics(candle_duration, texts.len());
    utils::print_metrics(&candle_metrics, "Candle 引擎");

    println!("\n🔧 测试 ONNX Runtime 引擎...");
    let (onnx_embeddings, onnx_duration) = run_onnx_test(&model_path, &texts)?;
    let onnx_metrics = utils::calculate_metrics(onnx_duration, texts.len());
    utils::print_metrics(&onnx_metrics, "ONNX Runtime 引擎");

    println!("\n📈 性能对比:");
    let speedup = onnx_duration / candle_duration;
    if speedup > 1.0 {
        println!("  Candle 比 ONNX 快 {:.2}x", speedup);
    } else {
        println!("  ONNX 比 Candle 快 {:.2}x", 1.0 / speedup);
    }

    println!("\n✅ 性能对比测试完成");
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    println!("❌ 此示例需要启用 'onnx' feature");
    println!("   运行命令: cargo run --example performance_comparison --features onnx");
}
