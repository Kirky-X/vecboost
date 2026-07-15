// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbedRequestBody {
    text: String,
    normalize: bool,
}

#[derive(Deserialize)]
struct EmbedResponseBody {
    embedding: Vec<f32>,
    dimension: usize,
    processing_time_ms: u128,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 VecBoost 服务客户端示例");
    println!("==========================\n");

    let url = "http://localhost:9002/embed";
    println!("调用: POST {}", url);

    let body = EmbedRequestBody {
        text: "hello".to_string(),
        normalize: true,
    };

    let json_body = serde_json::to_string(&body)?;
    let client = reqwest::Client::new();
    let resp = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(json_body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("请求失败: HTTP {} - {}", status, text).into());
    }

    let text = resp.text().await?;
    let result: EmbedResponseBody = serde_json::from_str(&text)?;

    println!("\n响应结果:");
    println!("  维度: {}", result.dimension);
    println!("  耗时: {} ms", result.processing_time_ms);
    let preview = &result.embedding[..5.min(result.embedding.len())];
    println!("  Embedding 前 5 维: {:?}", preview);

    println!("\n✅ 客户端示例完成");
    Ok(())
}
