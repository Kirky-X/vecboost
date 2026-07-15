// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use vecboost::VecboostError;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::domain::EmbedRequest;
use vecboost::engine::InferenceEngine;
use vecboost::service::embedding::EmbeddingService;

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
        &Precision::Fp32
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
    println!("⚡ 推理性能监控示例");
    println!("====================\n");

    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(8)));
    let service = EmbeddingService::new(engine, None);

    let texts = [
        "hello world",
        "vecboost performance test",
        "rust async embedding service",
        "inference benchmark sample",
        "vector similarity search",
    ];

    let mut times: Vec<u128> = Vec::new();

    println!("逐条推理结果:");
    for text in &texts {
        let req = EmbedRequest {
            text: text.to_string(),
            normalize: Some(true),
        };
        let response = service.process_text(req, None).await?;
        times.push(response.processing_time_ms);
        println!(
            "  \"{}\" → {} ms (dim={})",
            text, response.processing_time_ms, response.dimension
        );
    }

    let avg = times.iter().sum::<u128>() as f64 / times.len() as f64;
    let max = *times.iter().max().unwrap();
    let min = *times.iter().min().unwrap();

    println!("\n性能报告:");
    println!("  样本数: {}", times.len());
    println!("  平均耗时: {:.2} ms", avg);
    println!("  最大耗时: {} ms", max);
    println!("  最小耗时: {} ms", min);
    println!("  耗时波动: {} ms", max - min);

    println!("\n✅ 性能监控示例完成");
    Ok(())
}
