// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 余弦相似度示例 — 使用 MockEngine 演示 api::compute_similarity 用法
//!
//! 运行: cargo run -p vecboost-examples --bin similarity --features http

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::api;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::engine::InferenceEngine;
use vecboost::{EmbeddingService, SimilarityRequest, VecboostError};

/// 模拟推理引擎 — 返回固定维度向量，避免依赖真实模型文件
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
    println!("🚀 VecBoost 余弦相似度示例");
    println!("============================\n");

    let dimension = 384;
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(dimension)));
    let service = EmbeddingService::new(engine, None);

    let source = "人工智能改变世界";
    let target = "AI transforms the world";
    println!("📝 源文本: \"{}\"", source);
    println!("📝 目标文本: \"{}\"", target);

    let req = SimilarityRequest {
        source: source.to_string(),
        target: target.to_string(),
    };

    println!("\n🔧 调用 api::compute_similarity...");
    let resp = api::compute_similarity(&service, req).await?;

    println!("✅ 相似度计算完成");
    println!("  余弦相似度: {:.4}", resp.score);
    // MockEngine 对所有文本返回相同向量，故 score = 1.0；真实引擎会给出有区分度的分数

    Ok(())
}
