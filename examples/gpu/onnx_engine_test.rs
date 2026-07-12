// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::path::PathBuf;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};

mod utils;

#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ONNX Runtime GPU 测试");
    println!("=========================\n");

    if !utils::check_cuda_available() {
        println!("❌ CUDA 不可用，请确保已安装 CUDA 驱动和 cuDNN");
        return Ok(());
    }

    let model_path = match utils::find_model_path(utils::DEFAULT_MODEL_PATHS) {
        Some(path) => PathBuf::from(path),
        None => {
            println!("❌ 未找到模型文件");
            return Ok(());
        }
    };

    let model_config = ModelConfig {
        name: "bge-m3".to_string(),
        engine_type: EngineType::Onnx,
        model_path,
        tokenizer_path: None,
        device: DeviceType::Cuda,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(1024),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    println!("📊 模型配置:");
    println!("  引擎: {:?}", model_config.engine_type);
    println!("  批次大小: {}", model_config.max_batch_size);
    println!("  期望维度: {:?}", model_config.expected_dimension);

    println!("\n🔧 初始化 ONNX 引擎...");
    let mut engine = AnyEngine::new(&model_config, EngineType::Onnx, Precision::Fp32)?;
    println!("✅ 引擎初始化成功");

    println!("\n📝 单文本嵌入测试...");
    let single_text = utils::get_single_test_text();
    println!("  文本: \"{}\"", single_text);

    let (embedding, duration) = utils::measure_time(|| engine.embed(single_text))?;
    println!("  耗时: {:.2} ms", duration);

    utils::validate_embedding(&embedding, model_config.expected_dimension.unwrap_or(1024));
    utils::print_embedding_preview(&embedding, 5);

    println!("\n📝 批量文本嵌入测试...");
    let texts = utils::get_test_texts();
    println!("  文本数量: {}", texts.len());

    let (embeddings, duration) = utils::measure_time(|| engine.embed_batch(&texts))?;
    println!("  耗时: {:.2} ms", duration);

    let metrics = utils::calculate_metrics(duration, texts.len());
    utils::print_metrics(&metrics, "批量嵌入");

    for (i, embedding) in embeddings.iter().enumerate() {
        println!("\n  文本 {}:", i + 1);
        utils::validate_embedding(embedding, model_config.expected_dimension.unwrap_or(1024));
    }

    println!("\n✅ ONNX Runtime GPU 测试完成");
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn main() {
    println!("❌ 此示例需要启用 'onnx' feature");
    println!("   运行命令: cargo run --example onnx_engine_test --features onnx");
}
