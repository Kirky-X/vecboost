// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use vecboost::EmbeddingService;
use vecboost::VecboostError;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, PoolingMode, Precision};
use vecboost::domain::EmbedRequest;
use vecboost::engine::InferenceEngine;

struct MockEngine {
    dimension: usize,
}

impl MockEngine {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(vec![0.5; self.dimension])
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
    }

    fn precision(&self) -> &Precision {
        static PRECISION: Precision = Precision::Fp32;
        &PRECISION
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Candle 引擎初始化与推理示例");
    println!("==============================\n");

    let model_config = ModelConfig {
        name: "mock-candle".to_string(),
        engine_type: EngineType::Candle,
        model_path: std::path::PathBuf::from("/nonexistent/model"),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: Some(PoolingMode::Mean),
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    println!("📊 模型配置:");
    println!("  引擎: {:?}", model_config.engine_type);
    println!("  设备: {:?}", model_config.device);
    println!("  期望维度: {:?}", model_config.expected_dimension);

    println!("\n🔧 初始化 MockEngine(模拟 CandleEngine)...");
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(384)));
    let service = EmbeddingService::new(engine, Some(model_config));
    println!("✅ EmbeddingService 创建成功");

    println!("\n📝 调用 process_text 执行嵌入推理...");
    let req = EmbedRequest {
        text: "hello world".to_string(),
        normalize: Some(true),
    };
    let resp = service.process_text(req, None).await?;
    println!("  输入文本: \"hello world\"");
    println!("  返回维度: {}", resp.dimension);
    println!("  耗时: {} ms", resp.processing_time_ms);
    println!("  embedding 前 5 维: {:?}", &resp.embedding[..5]);

    println!("\n✅ 示例完成");
    Ok(())
}
