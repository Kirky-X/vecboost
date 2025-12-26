use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::candle_engine::CandleEngine;
use vecboost::engine::InferenceEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    println!("=== VecBoost GPU 向量化测试 ===\n");

    let model_path = PathBuf::from("./model");
    if !model_path.exists() {
        eprintln!("错误: 模型目录不存在: {:?}", model_path);
        eprintln!("请先下载或转换模型到 ./model 目录");
        eprintln!("\n使用方法:");
        eprintln!("1. 从HuggingFace下载模型:");
        eprintln!("   python -c \"from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5').save_pretrained('./model')\"");
        eprintln!("2. 或转换现有模型为safetensors格式");
        std::process::exit(1);
    }

    let config = ModelConfig {
        model_path,
        max_batch_size: 32,
        max_sequence_length: 512,
        device: DeviceType::Cuda,
        engine: EngineType::Candle,
        precision: Precision::Fp16,
        normalize: true,
        quantize: None,
        pooling: None,
    };

    println!("模型配置:");
    println!("  - 设备: {:?}", config.device);
    println!("  - 引擎: {:?}", config.engine);
    println!("  - 精度: {:?}", config.precision);
    println!();

    println!("正在初始化Candle引擎（GPU模式）...");
    let start_init = Instant::now();
    let engine = CandleEngine::new(&config, config.precision.clone())?;
    let init_time = start_init.elapsed();
    println!("引擎初始化完成，耗时: {:.2}秒\n", init_time.as_secs_f64());

    let device_type = engine.device_type();
    let precision = engine.precision();
    let supports_mixed = engine.supports_mixed_precision();

    println!("引擎状态:");
    println!("  - 设备类型: {:?}", device_type);
    println!("  - 精度设置: {:?}", precision);
    println!("  - 混合精度支持: {}", supports_mixed);
    println!();

    let test_texts = vec![
        "人工智能技术正在快速发展".to_string(),
        "机器学习和深度学习是AI的核心".to_string(),
        "向量数据库支持高效的相似性搜索".to_string(),
        "自然语言处理应用广泛".to_string(),
        "GPU加速显著提升计算性能".to_string(),
    ];

    println!("测试文本数量: {}", test_texts.len());
    println!();

    println!("执行单文本向量化测试...");
    let single_text = "这是一个用于测试GPU向量化的人工智能文本示例";
    let start_single = Instant::now();
    let embedding = engine.embed(single_text)?;
    let single_time = start_single.elapsed();

    println!("  - 文本: \"{}\"", single_text);
    println!("  - 向量维度: {}", embedding.len());
    println!("  - 向量范数: {:.4}", embedding.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!("  - 耗时: {:.2}毫秒", single_time.as_secs_f64() * 1000.0);
    println!();

    println!("执行批量向量化测试（{}个文本）...", test_texts.len());
    let start_batch = Instant::now();
    let embeddings = engine.embed_batch(&test_texts)?;
    let batch_time = start_batch.elapsed();

    println!("  - 成功处理: {} 个文本", embeddings.len());
    println!("  - 向量维度: {}", embeddings[0].len());
    println!("  - 总耗时: {:.2}毫秒", batch_time.as_secs_f64() * 1000.0);
    println!("  - 平均每文本: {:.2}毫秒", batch_time.as_secs_f64() * 1000.0 / test_texts.len() as f64);
    println!();

    let throughput = test_texts.len() as f64 / batch_time.as_secs_f64();
    println!("性能统计:");
    println!("  - 批量吞吐量: {:.2} 文本/秒", throughput);
    println!("  - 单文本延迟: {:.2}毫秒", single_time.as_secs_f64() * 1000.0);
    println!();

    println!("=== 测试完成 ===");
    println!("\n验证GPU使用情况:");
    println!("请在另一个终端运行: nvidia-smi");
    println!("如果GPU-Util或Memory-Usage有显著变化，说明正在使用GPU");

    Ok(())
}
