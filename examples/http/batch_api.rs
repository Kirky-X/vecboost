// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct BatchEmbedRequestBody {
    texts: Vec<String>,
    normalize: bool,
}

#[derive(Deserialize)]
struct BatchEmbeddingResult {
    text_preview: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchEmbedResponseBody {
    embeddings: Vec<BatchEmbeddingResult>,
    dimension: usize,
    processing_time_ms: u128,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HTTP POST /api/v1/embed/batch 示例");
    println!("=====================================\n");

    let client = Client::new();
    let url = "http://localhost:9002/api/v1/embed/batch";
    let body = BatchEmbedRequestBody {
        texts: vec!["hello".to_string(), "world".to_string()],
        normalize: true,
    };
    let json = serde_json::to_string(&body)?;

    println!("📝 请求: POST {}", url);
    println!("  body: {}", json);

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
    let result: BatchEmbedResponseBody = serde_json::from_str(&text)?;
    println!("\n✅ 请求成功");
    println!("  返回 embeddings 数量: {}", result.embeddings.len());
    println!("  维度: {}", result.dimension);
    println!("  耗时: {} ms", result.processing_time_ms);
    for (i, item) in result.embeddings.iter().enumerate() {
        println!(
            "  [{}] text_preview=\"{}\" embedding 前 3 维: {:?}",
            i,
            item.text_preview,
            &item.embedding[..3]
        );
    }

    Ok(())
}
