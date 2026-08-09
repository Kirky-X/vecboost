// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! VecBoostLibrary SDK 示例 — 演示 library 模式的异步与同步 API
//!
//! library 模式允许在不启动 HTTP/gRPC 服务器的情况下，
//! 将 VecBoost 作为嵌入式向量化和重排序模块直接集成到 Rust 应用中。
//!
//! 运行: cargo run -p vecboost-examples --bin library_usage
//!
//! 注意: 需要先下载模型文件:
//!   cargo run -p vecboost-examples --bin download_model -- --small

use vecboost::{LibraryConfig, VecBoostLibrary};
use vecboost::config::model::{DeviceType, EngineType, ModelConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 VecBoostLibrary SDK 示例（library 模式）");
    println!("============================================\n");

    // 1. 配置模型参数
    let model_config = ModelConfig {
        name: "BAAI/bge-small-en-v1.5".to_string(),
        engine_type: EngineType::Candle,
        model_path: std::path::PathBuf::from("models/BAAI-bge-small-en-v1.5"),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 32,
        pooling_mode: None,
        expected_dimension: Some(384),
        memory_limit_bytes: None,
        oom_fallback_enabled: false,
        model_sha256: None,
    };

    let config = LibraryConfig {
        model_config,
        cache_size: 100, // 启用缓存，容量 100 条
        rerank_config: None,
    };

    println!("📊 LibraryConfig:");
    println!("  模型: BAAI/bge-small-en-v1.5");
    println!("  引擎: Candle (CPU)");
    println!("  缓存容量: 100");

    // 2. 初始化 VecBoostLibrary（加载模型，可能需要几秒）
    println!("\n🔧 初始化 VecBoostLibrary...");
    let lib = match VecBoostLibrary::new(config).await {
        Ok(lib) => {
            println!("✅ VecBoostLibrary 初始化成功\n");
            lib
        }
        Err(e) => {
            println!("❌ 初始化失败（需要真实模型文件）: {}", e);
            println!("\n💡 请先下载模型:");
            println!("   cargo run -p vecboost-examples --bin download_model -- --small");
            println!("\n📝 API 用法参考（以下为伪代码）:");
            println!("  let resp = lib.embed(\"hello world\").await?;");
            println!("  let batch = lib.embed_batch(&texts).await?;");
            println!("  let ranked = lib.rerank(\"query\", &docs, Some(3)).await?;");
            println!("  let sync_resp = lib.embed_sync(\"hello\"); // 同步 API");
            return Ok(());
        }
    };

    // 3. 异步单文本嵌入
    println!("📝 异步单文本嵌入:");
    let resp = lib.embed("hello world").await?;
    println!("  维度: {}", resp.dimension);
    println!("  耗时: {} ms", resp.processing_time_ms);
    let preview: Vec<String> = resp.embedding.iter().take(5)
        .map(|v| format!("{:.4}", v))
        .collect();
    println!("  前 5 个值: [{}]", preview.join(", "));

    // 4. 异步批量嵌入
    println!("\n📝 异步批量嵌入:");
    let texts = vec![
        "machine learning is fascinating".to_string(),
        "vector embeddings power semantic search".to_string(),
        "Rust provides memory safety without garbage collection".to_string(),
    ];
    let batch_resp = lib.embed_batch(&texts).await?;
    println!("  返回 {} 条嵌入", batch_resp.embeddings.len());
    println!("  统一维度: {}", batch_resp.dimension);

    // 5. 异步重排序
    println!("\n📝 异步重排序:");
    let query = "what is machine learning?";
    let documents = vec![
        "Machine learning is a subset of AI".to_string(),
        "The weather is nice today".to_string(),
        "Deep learning uses neural networks".to_string(),
    ];
    let rerank_resp = lib.rerank(query, &documents, Some(2)).await?;
    println!("  查询: \"{}\"", query);
    println!("  返回 {} 条结果 (top_k=2):", rerank_resp.results.len());
    for result in &rerank_resp.results {
        println!("    [{}] score={:.4}", result.index, result.score);
    }

    // 6. 同步 API（适用于非异步上下文）
    println!("\n📝 同步 API 调用:");
    let sync_resp = lib.embed_sync("synchronous embedding")?;
    println!("  embed_sync 维度: {}", sync_resp.dimension);

    let sync_batch = lib.embed_batch_sync(&["text a".to_string(), "text b".to_string()])?;
    println!("  embed_batch_sync 返回 {} 条", sync_batch.embeddings.len());

    println!("\n✅ VecBoostLibrary SDK 示例完成");
    Ok(())
}
