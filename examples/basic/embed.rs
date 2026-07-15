// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 单文本嵌入示例 — 使用 MockEngine 演示 EmbeddingService + api::embed 用法
//!
//! 运行: cargo run -p vecboost-examples --bin embed --features http

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::api;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::engine::InferenceEngine;
use vecboost::{EmbedRequest, EmbeddingService, VecboostError};

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
    println!("🚀 VecBoost 单文本嵌入示例");
    println!("==========================\n");

    let dimension = 384;
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockEngine::new(dimension)));
    let service = EmbeddingService::new(engine, None);

    let text = "VecBoost 是高性能向量嵌入服务";
    println!("📝 输入文本: \"{}\"", text);

    let req = EmbedRequest {
        text: text.to_string(),
        normalize: Some(true),
    };

    println!("🔧 调用 api::embed...");
    let resp = api::embed(&service, req).await?;

    println!("✅ 嵌入成功");
    println!("  维度: {}", resp.dimension);
    println!("  耗时: {} ms", resp.processing_time_ms);
    let preview: Vec<String> = resp
        .embedding
        .iter()
        .take(5)
        .map(|v| format!("{:.4}", v))
        .collect();
    println!("  前 5 个值: [{}]", preview.join(", "));

    Ok(())
}
