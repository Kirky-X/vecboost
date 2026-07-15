// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct SimilarityRequestBody {
    source: String,
    target: String,
}

#[derive(Deserialize)]
struct SimilarityResponseBody {
    score: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 HTTP POST /api/v1/similarity 示例");
    println!("====================================\n");

    let client = Client::new();
    let url = "http://localhost:9002/api/v1/similarity";
    let body = SimilarityRequestBody {
        source: "hello".to_string(),
        target: "world".to_string(),
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
    let result: SimilarityResponseBody = serde_json::from_str(&text)?;
    println!("\n✅ 请求成功");
    println!("  相似度 score: {}", result.score);

    Ok(())
}
