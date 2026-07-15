// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use vecboost::EmbeddingService;
use vecboost::VecboostError;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::domain::ModelSwitchRequest;
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
    println!("🚀 运行时模型切换示例");
    println!("======================\n");

    let model_config = ModelConfig {
        name: "mock-initial".to_string(),
        engine_type: EngineType::Candle,
        model_path: std::path::PathBuf::from("/nonexistent/model"),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };
    println!("📊 初始模型: {}", model_config.name);

    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(384)));
    let mut service = EmbeddingService::new(engine, Some(model_config));
    println!("✅ EmbeddingService 创建成功");

    println!("\n🔧 切换到同名模型(触发 early-return 成功路径)...");
    let req = ModelSwitchRequest {
        model_name: "mock-initial".to_string(),
        model_path: None,
        tokenizer_path: None,
        device: None,
        max_batch_size: None,
        pooling_mode: None,
        expected_dimension: None,
        memory_limit_bytes: None,
        oom_fallback_enabled: None,
    };
    let resp = service.switch_model(req).await?;
    println!("  success: {}", resp.success);
    println!("  previous_model: {:?}", resp.previous_model);
    println!("  current_model: {}", resp.current_model);
    println!("  message: {}", resp.message);

    println!("\n🔧 切换到新模型(需要真实模型路径,预期失败)...");
    let req = ModelSwitchRequest {
        model_name: "mock-v2".to_string(),
        model_path: Some(std::path::PathBuf::from("/nonexistent/model-v2")),
        tokenizer_path: None,
        device: Some(DeviceType::Cpu),
        max_batch_size: Some(64),
        pooling_mode: None,
        expected_dimension: Some(768),
        memory_limit_bytes: None,
        oom_fallback_enabled: Some(true),
    };
    match service.switch_model(req).await {
        Ok(resp) => println!("  ✅ 切换成功: {}", resp.message),
        Err(e) => println!("  ❌ 切换失败(预期行为,需真实模型): {e}"),
    }

    println!("\n✅ 示例完成");
    Ok(())
}
