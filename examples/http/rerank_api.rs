// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! HTTP POST /api/v1/rerank 客户端示例
//!
//! 需要 vecboost 服务已启动: cargo run --features http

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct RerankRequestBody {
    query: String,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    score: f32,
    document: Option<String>,
}

#[derive(Deserialize)]
struct RerankResponseBody {
    results: Vec<RerankResult>,
    processing_time_ms: u128,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HTTP POST /api/v1/rerank 示例");
    println!("=================================\n");

    let client = Client::new();
    let url = "http://localhost:9002/api/v1/rerank";
    let body = RerankRequestBody {
        query: "what is machine learning?".to_string(),
        documents: vec![
            "Machine learning is a subset of artificial intelligence".to_string(),
            "The weather forecast predicts rain tomorrow".to_string(),
            "Deep learning uses neural networks for training".to_string(),
            "Rust is a systems programming language".to_string(),
            "Reinforcement learning optimizes policies through rewards".to_string(),
        ],
        top_k: Some(3),
        return_documents: Some(true),
    };
    let json = serde_json::to_string_pretty(&body)?;

    println!("📝 请求: POST {}", url);
    println!("  body: {}\n", json);

    let resp = client
        .post(url)
        .header("content-type", "application/json")
        .body(json)
        .send()
        .await?;
    if !resp.status().is_success() {
        println!("❌ 请求失败: HTTP {}", resp.status());
        let text = resp.text().await?;
        println!("  响应: {}", text);
        return Ok(());
    }

    let text = resp.text().await?;
    let result: RerankResponseBody = serde_json::from_str(&text)?;
    println!("✅ 请求成功");
    println!("  返回 {} 条结果 (top_k=3):", result.results.len());
    println!("  耗时: {} ms\n", result.processing_time_ms);
    for (i, item) in result.results.iter().enumerate() {
        let doc_preview = item.document.as_deref().unwrap_or("-");
        println!(
            "  [{}] index={}  score={:.4}  document=\"{}\"",
            i + 1,
            item.index,
            item.score,
            if doc_preview.len() > 50 {
                format!("{}...", &doc_preview[..50])
            } else {
                doc_preview.to_string()
            }
        );
    }

    Ok(())
}
