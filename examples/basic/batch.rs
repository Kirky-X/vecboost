// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 批量嵌入示例 — 使用 MockEngine 演示 api::embed_batch 用法
//!
//! 运行: cargo run -p vecboost-examples --bin batch --features http

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::api;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::domain::BatchEmbedRequest;
use vecboost::engine::InferenceEngine;
use vecboost::{EmbeddingService, VecboostError};

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
    println!("🚀 VecBoost 批量嵌入示例");
    println!("==========================\n");

    let dimension = 384;
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(dimension)));
    let service = EmbeddingService::new(engine, None);

    let texts = vec![
        "机器学习是人工智能的核心".to_string(),
        "向量嵌入将文本转化为数值表示".to_string(),
        "余弦相似度衡量向量方向差异".to_string(),
    ];
    println!("📝 批量输入 ({} 条文本):", texts.len());
    for (i, t) in texts.iter().enumerate() {
        println!("  [{}] \"{}\"", i + 1, t);
    }

    let req = BatchEmbedRequest {
        texts,
        mode: None,
        normalize: Some(true),
    };

    println!("\n🔧 调用 api::embed_batch...");
    let resp = api::embed_batch(&service, req).await?;

    println!("✅ 批量嵌入成功");
    println!("  统一维度: {}", resp.dimension);
    println!("  耗时: {} ms", resp.processing_time_ms);
    for (i, result) in resp.embeddings.iter().enumerate() {
        println!(
            "  文本 {} — 预览: \"{}\", 维度: {}",
            i + 1,
            result.text_preview,
            result.embedding.len()
        );
    }

    Ok(())
}
