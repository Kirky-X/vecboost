// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 集成测试
//!
//! 使用 RealTestEngine 进行集成测试，支持真实推理和 Mock 回退。

use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

use vecboost::domain::{EmbedRequest, SearchRequest, SimilarityRequest};
use vecboost::service::embedding::EmbeddingService;

// 使用新的 RealTestEngine
mod real_engine;
use real_engine::create_test_engine;

const MOCK_DIMENSION: usize = 1024;

#[tokio::test]
async fn test_e2e_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "Artificial intelligence is the future of technology".to_string(),
            normalize: Some(true),
        })
        .await?;

    assert_eq!(
        result.embedding.len(),
        MOCK_DIMENSION,
        "Embedding dimension should be 1024"
    );
    assert!(
        result.embedding.iter().all(|&x| x.is_finite()),
        "All values should be finite"
    );

    println!(
        "[PASS] End-to-end text embedding test passed - dimension: {}",
        result.dimension
    );
    Ok(())
}

#[tokio::test]
async fn test_e2e_chinese_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "人工智能是未来的发展方向".to_string(),
            normalize: Some(true),
        })
        .await?;

    assert_eq!(result.embedding.len(), MOCK_DIMENSION);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!(
        "[PASS] Chinese text embedding test passed - dimension: {}",
        result.dimension
    );
    Ok(())
}

#[tokio::test]
async fn test_e2e_mixed_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "AI人工智能技术正在快速发展".to_string(),
            normalize: Some(true),
        })
        .await?;

    assert_eq!(result.embedding.len(), MOCK_DIMENSION);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!("[PASS] Mixed text embedding test passed");
    Ok(())
}

#[tokio::test]
async fn test_similarity_calculation() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SimilarityRequest {
        source: "machine learning".to_string(),
        target: "deep learning".to_string(),
    };

    let result = service.process_similarity(req).await?;

    assert!(
        result.score >= -1.0 && result.score <= 1.0,
        "Similarity should be between -1 and 1"
    );
    assert!(
        result.score.is_finite(),
        "Similarity score should be finite"
    );

    println!(
        "[PASS] Similarity calculation test passed - score: {:.4}",
        result.score
    );
    Ok(())
}

#[tokio::test]
async fn test_embedding_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req1 = SimilarityRequest {
        source: "aaaa".to_string(),
        target: "aaab".to_string(),
    };

    let req2 = SimilarityRequest {
        source: "aaaa".to_string(),
        target: "bbbb".to_string(),
    };

    let sim_similar = service.process_similarity(req1).await?.score;
    let sim_different = service.process_similarity(req2).await?.score;

    assert!(
        sim_similar > sim_different,
        "Same seed should produce higher similarity. Got similar: {:.4}, different: {:.4}",
        sim_similar,
        sim_different
    );

    println!(
        "[PASS] Embedding determinism test passed - similar: {:.4}, different: {:.4}",
        sim_similar, sim_different
    );
    Ok(())
}

#[tokio::test]
async fn test_search_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SearchRequest {
        query: "programming language selection".to_string(),
        texts: vec![
            "Python is an easy-to-learn programming language".to_string(),
            "Java is the首选 for enterprise applications".to_string(),
            "The weather is nice today, good for going outside".to_string(),
            "Rust is a systems programming language focused on safety".to_string(),
        ],
        top_k: Some(2),
    };

    let result = service.process_search(req).await?;

    assert_eq!(result.results.len(), 2, "Should return top 2 results");
    assert!(
        result.results[0].score >= result.results[1].score,
        "Results should be sorted by score in descending order"
    );
    assert!(result.results[0].score.is_finite());

    println!(
        "[PASS] Search functionality test passed - top scores: {:.4}, {:.4}",
        result.results[0].score, result.results[1].score
    );
    Ok(())
}

#[tokio::test]
async fn test_search_with_less_results_than_topk() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SearchRequest {
        query: "tech topic".to_string(),
        texts: vec![
            "AI is popular".to_string(),
            "blockchain technology".to_string(),
        ],
        top_k: Some(10),
    };

    let result = service.process_search(req).await?;

    assert_eq!(
        result.results.len(),
        2,
        "Should return all available results"
    );
    println!(
        "[PASS] Search result count test passed - returned {} results",
        result.results.len()
    );
    Ok(())
}

#[tokio::test]
async fn test_batch_search_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let texts: Vec<String> = (0..20)
        .map(|i| format!("This is test document #{} about technical topics{}", i, i))
        .collect();
    let query = "technical topics related content".to_string();

    let result = service
        .process_search_batch(&query, &texts, Some(5))
        .await?;

    assert_eq!(result.results.len(), 5, "Should return top 5 results");
    assert!(result.results[0].score >= result.results[1].score);

    println!(
        "[PASS] Batch search test passed - returned {} results",
        result.results.len()
    );
    Ok(())
}

#[tokio::test]
async fn test_file_streaming_processing() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("test_text.txt");

    let content = vec![
        "First line of text, about machine learning content",
        "Second line, about deep learning applications",
        "Third line, about natural language processing",
        "Fourth line, about computer vision",
        "Fifth line, about reinforcement learning",
    ];

    let mut file = File::create(&file_path).await?;
    for line in &content {
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
    }
    file.sync_all().await?;

    let result = service.process_file_stream(&file_path).await?;

    assert_eq!(result.embedding.len(), MOCK_DIMENSION);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!(
        "[PASS] File streaming processing test passed - dimension: {}",
        result.dimension
    );
    Ok(())
}

#[tokio::test]
async fn test_empty_file_processing() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("empty.txt");

    let mut file = File::create(&file_path).await?;
    file.write_all(b"").await?;
    file.sync_all().await?;

    let result = service.process_file_stream(&file_path).await;
    assert!(result.is_err(), "Empty file should return error");

    println!("[PASS] Empty file processing test passed - correctly returned error");
    Ok(())
}

#[tokio::test]
async fn test_embedding_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let text = "Consistency test text".to_string();

    let result1 = service
        .process_text(EmbedRequest {
            text: text.clone(),
            normalize: Some(true),
        })
        .await?;
    let result2 = service
        .process_text(EmbedRequest {
            text,
            normalize: Some(true),
        })
        .await?;

    assert_eq!(result1.embedding.len(), result2.embedding.len());
    assert!(
        result1
            .embedding
            .iter()
            .zip(result2.embedding.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "Same input should produce same embedding"
    );

    println!("[PASS] Embedding consistency test passed");
    Ok(())
}

#[tokio::test]
async fn test_embedding_normalization() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "Test normalized embedding vector".to_string(),
            normalize: Some(true),
        })
        .await?;

    let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be normalized, got norm: {}",
        norm
    );

    println!(
        "[PASS] Embedding normalization test passed - L2 norm: {:.6}",
        norm
    );
    Ok(())
}

#[tokio::test]
async fn test_concurrent_embedding_requests() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;

    let texts: Vec<String> = (0..4)
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

    let mut embeddings_count = 0;
    for handle in handles {
        let result = handle.await?;
        if result.is_ok() {
            embeddings_count += 1;
        }
    }

    assert_eq!(embeddings_count, 4);
    println!(
        "[PASS] Concurrent embedding requests test passed - {} requests processed",
        embeddings_count
    );
    Ok(())
}

#[tokio::test]
async fn test_long_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let long_text = vec![
        "This is a very long text used to test the model's ability to handle long text inputs.";
        50
    ]
    .join(" ");

    let result = service
        .process_text(EmbedRequest {
            text: long_text,
            normalize: Some(true),
        })
        .await?;

    assert_eq!(result.embedding.len(), MOCK_DIMENSION);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!(
        "[PASS] Long text embedding test passed - dimension: {}",
        result.dimension
    );
    Ok(())
}
