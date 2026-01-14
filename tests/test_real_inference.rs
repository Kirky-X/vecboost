// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 真实推理集成测试
//!
//! 测试 RealTestEngine 的真实推理能力和回退机制。

mod real_engine;

use std::sync::Arc;
use tokio::sync::RwLock;

use vecboost::domain::{EmbedRequest, SimilarityRequest};
use vecboost::service::embedding::EmbeddingService;

use crate::real_engine::{RealTestEngine, create_test_engine, create_test_engine_with_dimension};

/// 测试 RealTestEngine 基本功能
#[tokio::test]
async fn test_real_test_engine_basic() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "Hello, world!".to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    // 打印引擎信息（通过测试模式）
    let mode = real_engine::TestMode::from_env();
    println!("Engine mode: {:?}", mode);

    // 验证向量
    assert!(!result.embedding.is_empty());
    assert!(result.embedding.iter().all(|&x| x.is_finite()));
}

/// 测试不同维度
#[tokio::test]
async fn test_real_test_engine_different_dimensions() {
    let dimensions = [384, 768, 1024];

    for dim in dimensions {
        let engine = create_test_engine_with_dimension(dim).unwrap();
        let service = EmbeddingService::new(engine, None);

        let result = service
            .process_text(EmbedRequest {
                text: "Test text".to_string(),
                normalize: Some(true),
            })
            .await
            .unwrap();

        assert_eq!(result.embedding.len(), dim, "Dimension should be {}", dim);
    }
}

/// 测试确定性
#[tokio::test]
async fn test_real_test_engine_determinism() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let text = "Deterministic test text";

    let result1 = service
        .process_text(EmbedRequest {
            text: text.to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    let result2 = service
        .process_text(EmbedRequest {
            text: text.to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    assert_eq!(
        result1.embedding, result2.embedding,
        "Same text should produce same embedding"
    );
}

/// 测试相似度计算
#[tokio::test]
async fn test_real_test_engine_similarity() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let similar_text1 = "Machine learning is a subset of artificial intelligence";
    let similar_text2 = "ML is part of AI technology";
    let different_text = "The weather is nice today";

    let sim_similar = service
        .process_similarity(SimilarityRequest {
            source: similar_text1.to_string(),
            target: similar_text2.to_string(),
        })
        .await
        .unwrap();

    let sim_different = service
        .process_similarity(SimilarityRequest {
            source: similar_text1.to_string(),
            target: different_text.to_string(),
        })
        .await
        .unwrap();

    println!("Similarity (similar): {:.4}", sim_similar.score);
    println!("Similarity (different): {:.4}", sim_different.score);

    assert!(
        sim_similar.score > sim_different.score,
        "Similar texts should have higher similarity"
    );

    // 验证相似度范围
    assert!(sim_similar.score >= -1.0 && sim_similar.score <= 1.0);
    assert!(sim_different.score >= -1.0 && sim_different.score <= 1.0);
}

/// 测试回退机制
#[tokio::test]
async fn test_real_test_engine_fallback() {
    let engine = RealTestEngine::with_dimension(384);
    let service = EmbeddingService::new(Arc::new(RwLock::new(engine)), None);

    // 由于没有真实模型，应该使用回退
    let result = service
        .process_text(EmbedRequest {
            text: "Fallback test".to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    assert_eq!(result.embedding.len(), 384);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));
}

/// 测试中文文本
#[tokio::test]
async fn test_real_test_engine_chinese() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "人工智能是未来的发展方向".to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    assert!(!result.embedding.is_empty());
    assert!(result.embedding.iter().all(|&x| x.is_finite()));
}

/// 测试归一化
#[tokio::test]
async fn test_real_test_engine_normalization() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "Test normalized embedding vector".to_string(),
            normalize: Some(true),
        })
        .await
        .unwrap();

    // 验证归一化
    let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be normalized, got norm: {}",
        norm
    );
}

/// 测试并发请求
#[tokio::test]
async fn test_real_test_engine_concurrent() {
    let engine = create_test_engine().unwrap();

    let texts: Vec<String> = (0..10)
        .map(|i| format!("Concurrent test text {}", i))
        .collect();

    let mut handles = Vec::new();
    for text in texts {
        let engine = Arc::clone(&engine);
        let text = text.clone();
        let handle = tokio::spawn(async move {
            let service = EmbeddingService::new(engine, None);
            service
                .process_text(EmbedRequest {
                    text,
                    normalize: Some(true),
                })
                .await
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            success_count += 1;
        }
    }

    assert_eq!(success_count, 10, "All concurrent requests should succeed");
}

/// 测试模式检测
#[test]
fn test_test_mode_detection() {
    // 测试默认模式
    unsafe { std::env::remove_var("TEST_MODE") };
    let engine = RealTestEngine::new();
    assert_eq!(engine.engine_info(), "mock");

    // 测试 Mock 模式
    unsafe { std::env::set_var("TEST_MODE", "mock") };
    let engine = RealTestEngine::new();
    assert_eq!(engine.engine_info(), "mock");

    // 测试 Light 模式
    unsafe { std::env::set_var("TEST_MODE", "light") };
    // 取决于是否能够加载真实引擎
    let engine = RealTestEngine::new();
    let info = engine.engine_info();
    assert!(info == "mock" || info == "real");

    // 测试 Full 模式
    unsafe { std::env::set_var("TEST_MODE", "full") };
    let engine = RealTestEngine::new();
    let info = engine.engine_info();
    assert!(info == "mock" || info == "real");

    // 清理
    unsafe { std::env::remove_var("TEST_MODE") };
}

/// 性能测试（仅在 full 模式下运行较慢）
#[tokio::test]
async fn test_real_test_engine_performance() {
    let engine = create_test_engine().unwrap();
    let service = EmbeddingService::new(engine, None);

    let texts: Vec<String> = (0..100)
        .map(|i| format!("Performance test text number {}", i))
        .collect();

    let start = std::time::Instant::now();
    for text in texts {
        let _ = service
            .process_text(EmbedRequest {
                text,
                normalize: Some(true),
            })
            .await;
    }
    let elapsed = start.elapsed();

    println!("Processed 100 embeddings in {:.2?}", elapsed);

    // 在 Mock 模式下，应该很快（< 1秒）
    // 在真实模式下，可能需要更长时间
    if RealTestEngine::new().is_using_fallback() {
        assert!(elapsed < std::time::Duration::from_secs(1));
    }
}
