// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

use async_trait::async_trait;
use vecboost::VecboostError;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};

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
    println!("🚀 ONNX 引擎初始化示例");
    println!("========================\n");

    #[cfg(feature = "onnx")]
    {
        println!("✅ ONNX feature 已启用");
        println!("\n🔧 尝试通过 AnyEngine::new 创建 ONNX 引擎...");
        let config = ModelConfig {
            name: "mock-onnx".to_string(),
            engine_type: EngineType::Onnx,
            model_path: std::path::PathBuf::from("/nonexistent/onnx-model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        };
        match AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32) {
            Ok(engine) => {
                println!("✅ ONNX 引擎创建成功");
                let embedding = engine.embed("hello")?;
                println!("  embedding 维度: {}", embedding.len());
            }
            Err(e) => {
                println!("❌ ONNX 引擎创建失败(需要真实模型文件): {e}");
            }
        }
    }

    #[cfg(not(feature = "onnx"))]
    {
        println!("⚠️ ONNX feature 未启用");
        println!("   启用方式: cargo run -p vecboost-examples --bin onnx --features onnx");
    }

    println!("\n📝 使用 MockEngine 演示 InferenceEngine trait...");
    let engine = MockEngine::new(1024);
    let embedding = engine.embed("hello world")?;
    println!("  MockEngine embed 维度: {}", embedding.len());
    println!("  前 3 维: {:?}", &embedding[..3]);

    println!("\n✅ 示例完成");
    Ok(())
}
