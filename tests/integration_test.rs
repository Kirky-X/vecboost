// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use vecboost::{
    config::model::Precision,
    domain::{EmbedRequest, SearchRequest, SimilarityRequest},
    engine::InferenceEngine,
    service::embedding::EmbeddingService,
};

use async_trait::async_trait;
use vecboost::error::AppError;

const MOCK_DIMENSION: usize = 1024;

#[derive(Clone)]
struct MockEngine {
    dimension: usize,
}

impl MockEngine {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn generate_deterministic_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];
        let bytes = text.as_bytes();

        let mut hash: u64 = 1469598103934665603;
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }

        let seed = hash;

        let mut state = seed;
        for val in embedding.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let float_val = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *val = float_val;
        }

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in embedding.iter_mut() {
                *val /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl InferenceEngine for MockEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        Ok(self.generate_deterministic_embedding(text))
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| self.generate_deterministic_embedding(t))
            .collect();
        Ok(embeddings)
    }

    fn precision(&self) -> Precision {
        Precision::Fp32
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }
}

fn create_mock_engine(
) -> Result<Arc<RwLock<dyn InferenceEngine + Send + Sync>>, Box<dyn std::error::Error>> {
    Ok(Arc::new(RwLock::new(MockEngine::new(MOCK_DIMENSION))))
}

fn create_test_engine(
) -> Result<Arc<RwLock<dyn InferenceEngine + Send + Sync>>, Box<dyn std::error::Error>> {
    create_mock_engine()
}

#[tokio::test]
async fn test_e2e_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "人工智能是未来的发展方向".to_string(),
        })
        .await?;

    assert_eq!(
        result.embedding.len(),
        1024,
        "Embedding dimension should be 1024"
    );
    assert!(
        result.embedding.iter().all(|&x| x.is_finite()),
        "All values should be finite"
    );

    println!("✅ 端到端文本向量化测试通过 - 维度: {}", result.dimension);
    Ok(())
}

#[tokio::test]
async fn test_e2e_english_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "Machine learning is transforming the technology industry".to_string(),
        })
        .await?;

    assert_eq!(result.embedding.len(), 1024);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!("✅ 英文文本向量化测试通过");
    Ok(())
}

#[tokio::test]
async fn test_e2e_mixed_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "AI人工智能技术正在快速发展".to_string(),
        })
        .await?;

    assert_eq!(result.embedding.len(), 1024);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!("✅ 混合文本向量化测试通过");
    Ok(())
}

#[tokio::test]
async fn test_similarity_calculation() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SimilarityRequest {
        source: "机器学习".to_string(),
        target: "深度学习".to_string(),
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

    println!("✅ 相似度计算测试通过 - 相似度: {:.4}", result.score);
    Ok(())
}

#[tokio::test]
async fn test_similarity_semantic_coherence() -> Result<(), Box<dyn std::error::Error>> {
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
        "Similar texts should have higher similarity. Got similar: {:.4}, different: {:.4}",
        sim_similar,
        sim_different
    );

    println!(
        "✅ 相似度一致性测试通过 - 相似文本: {:.4}, 不同文本: {:.4}",
        sim_similar, sim_different
    );
    Ok(())
}

#[tokio::test]
async fn test_search_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SearchRequest {
        query: "关于编程语言的选择".to_string(),
        texts: vec![
            "Python是一门易学的编程语言".to_string(),
            "Java是企业级应用的首选".to_string(),
            "今天天气很好，适合外出".to_string(),
            "Rust是一门注重安全的系统编程语言".to_string(),
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
        "✅ 搜索功能测试通过 - Top结果: {:.4}, {:.4}",
        result.results[0].score, result.results[1].score
    );
    Ok(())
}

#[tokio::test]
async fn test_search_with_less_results_than_topk() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let req = SearchRequest {
        query: "技术话题".to_string(),
        texts: vec!["人工智能很热门".to_string(), "区块链技术".to_string()],
        top_k: Some(10),
    };

    let result = service.process_search(req).await?;

    assert_eq!(
        result.results.len(),
        2,
        "Should return all available results"
    );
    println!(
        "✅ 搜索结果数量测试通过 - 返回 {} 个结果",
        result.results.len()
    );
    Ok(())
}

#[tokio::test]
async fn test_batch_search_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let texts: Vec<String> = (0..20)
        .map(|i| format!("这是第{}个测试文档，内容关于技术主题{}", i, i))
        .collect();
    let query = "技术主题相关内容".to_string();

    let result = service
        .process_search_batch(&query, &texts, Some(5))
        .await?;

    assert_eq!(result.results.len(), 5, "Should return top 5 results");
    assert!(result.results[0].score >= result.results[1].score);

    println!("✅ 批量搜索测试通过 - 返回 {} 个结果", result.results.len());
    Ok(())
}

#[tokio::test]
async fn test_file_streaming_processing() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join("test_text.txt");

    let content = vec![
        "第一行文本，关于机器学习的内容",
        "第二行文本，关于深度学习的应用",
        "第三行文本，关于自然语言处理",
        "第四行文本，关于计算机视觉",
        "第五行文本，关于强化学习",
    ];

    let mut file = File::create(&file_path).await?;
    for line in &content {
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
    }
    file.sync_all().await?;

    let result = service.process_file_stream(&file_path).await?;

    assert_eq!(result.embedding.len(), 1024);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!("✅ 文件流式处理测试通过 - 维度: {}", result.dimension);
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

    println!("✅ 空文件处理测试通过 - 正确返回错误");
    Ok(())
}

#[tokio::test]
async fn test_embedding_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let text = "一致性测试文本".to_string();

    let result1 = service
        .process_text(EmbedRequest { text: text.clone() })
        .await?;
    let result2 = service.process_text(EmbedRequest { text }).await?;

    assert_eq!(result1.embedding.len(), result2.embedding.len());
    assert!(
        result1
            .embedding
            .iter()
            .zip(result2.embedding.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "Same input should produce same embedding"
    );

    println!("✅ 向量一致性测试通过");
    Ok(())
}

#[tokio::test]
async fn test_embedding_normalization() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let result = service
        .process_text(EmbedRequest {
            text: "测试归一化功能的向量".to_string(),
        })
        .await?;

    let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be normalized, got norm: {}",
        norm
    );

    println!("✅ 向量归一化测试通过 - L2范数: {:.6}", norm);
    Ok(())
}

#[tokio::test]
async fn test_concurrent_embedding_requests() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;

    let texts: Vec<String> = (0..4).map(|i| format!("并发测试文本 {}", i)).collect();

    let mut handles = Vec::new();
    for text in texts {
        let engine = Arc::clone(&engine);
        let text = text.clone();
        let handle = tokio::spawn(async move {
            let service = EmbeddingService::new(engine, None);
            service.process_text(EmbedRequest { text }).await
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
    println!("✅ 并发请求测试通过 - 成功处理 {} 个请求", embeddings_count);
    Ok(())
}

#[tokio::test]
async fn test_long_text_embedding() -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_test_engine()?;
    let service = EmbeddingService::new(engine, None);

    let long_text =
        vec!["这是一个很长的文本，用于测试模型处理长文本的能力。"; 50].join(" ");

    let result = service
        .process_text(EmbedRequest { text: long_text })
        .await?;

    assert_eq!(result.embedding.len(), 1024);
    assert!(result.embedding.iter().all(|&x| x.is_finite()));

    println!("✅ 长文本处理测试通过 - 维度: {}", result.dimension);
    Ok(())
}
