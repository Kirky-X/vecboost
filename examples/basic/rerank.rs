// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 重排序（Rerank）示例 — 使用 MockEngine 演示 RerankService 用法
//!
//! 运行: cargo run -p vecboost-examples --bin rerank

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;
use vecboost::VecboostError;
use vecboost::config::model::{ModelConfig, Precision};
use vecboost::domain::RerankRequest;
use vecboost::engine::InferenceEngine;
use vecboost::service::rerank::RerankService;

/// 模拟推理引擎 — rerank 分数与文档长度正相关（确定性输出）
struct MockRerankEngine;

#[async_trait]
impl InferenceEngine for MockRerankEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(vec![0.5; 384])
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|_| vec![0.5; 384]).collect())
    }

    fn precision(&self) -> &Precision {
        static PRECISION: Precision = Precision::Fp32;
        &PRECISION
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    fn rerank(&self, _query: &str, document: &str) -> Result<f32, VecboostError> {
        // 模拟分数与文档长度正相关
        Ok(document.len() as f32 / 100.0)
    }

    fn rerank_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, VecboostError> {
        documents
            .iter()
            .map(|doc| self.rerank(query, doc))
            .collect()
    }

    fn supports_rerank(&self) -> bool {
        true
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 VecBoost 重排序（Rerank）示例");
    println!("=================================\n");

    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(MockRerankEngine));
    let service = RerankService::new(engine, None);

    let query = "什么是机器学习？";
    let documents = vec![
        "机器学习是人工智能的一个分支".to_string(),
        "今天天气很好".to_string(),
        "深度学习使用神经网络进行训练".to_string(),
        "Python 是一种流行的编程语言".to_string(),
        "强化学习通过奖励机制优化策略".to_string(),
    ];

    println!("📝 查询: \"{}\"", query);
    println!("📄 文档列表 ({} 条):", documents.len());
    for (i, doc) in documents.iter().enumerate() {
        println!("  [{}] \"{}\"", i, doc);
    }

    // 1) 基本重排序（返回全部结果）
    println!("\n🔧 基本重排序:");
    let req = RerankRequest {
        query: query.to_string(),
        documents: documents.clone(),
        top_k: None,
        return_documents: Some(false),
    };
    let resp = service.process_rerank(req, 100, 8192).await?;
    println!("  返回 {} 条结果 (按相关性降序):", resp.results.len());
    for result in &resp.results {
        println!(
            "    [{}] score={:.4}  (原文档: \"{}\")",
            result.index, result.score, documents[result.index]
        );
    }
    println!("  耗时: {} ms", resp.processing_time_ms);

    // 2) Top-K 截断
    println!("\n🔧 Top-3 截断:");
    let req = RerankRequest {
        query: query.to_string(),
        documents: documents.clone(),
        top_k: Some(3),
        return_documents: Some(true),
    };
    let resp = service.process_rerank(req, 100, 8192).await?;
    println!("  返回 {} 条结果:", resp.results.len());
    for result in &resp.results {
        println!(
            "    score={:.4}  document=\"{}\"",
            result.score,
            result.document.as_deref().unwrap_or("-")
        );
    }

    println!("\n✅ 重排序示例完成");
    Ok(())
}
