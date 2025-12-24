// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::domain::{
    EmbedRequest, EmbedResponse, SearchRequest, SearchResponse, SearchResult, SimilarityRequest,
    SimilarityResponse,
};
use crate::engine::InferenceEngine;
use crate::error::AppError;
use crate::model::config::ModelConfig;
use crate::utils::{
    cosine_similarity, normalize_l2, InputValidator, TextValidator, DEFAULT_TOP_K, MAX_BATCH_SIZE,
    MAX_TOP_K,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tracing::warn;

pub struct EmbeddingService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
}

impl EmbeddingService {
    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
        }
    }

    pub fn with_validator(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self {
            engine,
            validator,
            model_config,
        }
    }

    fn validate_dimension(&self, actual_dimension: usize) {
        if let Some(ref config) = self.model_config {
            if let Some(expected) = config.expected_dimension {
                if actual_dimension != expected {
                    warn!(
                        "Dimension mismatch: expected {}, got {}. Model '{}' may have been configured incorrectly or the wrong model was loaded.",
                        expected,
                        actual_dimension,
                        config.name
                    );
                }
            }
        }
    }

    /// 处理单文本向量化
    pub async fn process_text(&self, req: EmbedRequest) -> Result<EmbedResponse, AppError> {
        self.validator.validate_text(&req.text)?;

        let mut embedding = self.engine.write().unwrap().embed(&req.text)?;
        normalize_l2(&mut embedding);

        let dimension = embedding.len();
        self.validate_dimension(dimension);

        Ok(EmbedResponse {
            dimension,
            embedding,
        })
    }

    /// 处理相似度计算
    pub async fn process_similarity(
        &self,
        req: SimilarityRequest,
    ) -> Result<SimilarityResponse, AppError> {
        self.validator.validate_text(&req.source)?;
        self.validator.validate_text(&req.target)?;

        let engine1 = Arc::clone(&self.engine);
        let engine2 = Arc::clone(&self.engine);

        let f1 = async move { engine1.write().unwrap().embed(&req.source) };
        let f2 = async move { engine2.write().unwrap().embed(&req.target) };

        let (mut v1, mut v2) = tokio::try_join!(f1, f2)?;

        normalize_l2(&mut v1);
        normalize_l2(&mut v2);

        let score = cosine_similarity(&v1, &v2)?;
        Ok(SimilarityResponse { score })
    }

    /// 处理大文件流式向量化 (简单实现：按行平均)
    pub async fn process_file_stream(&self, path: &Path) -> Result<EmbedResponse, AppError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut total_embedding: Option<Vec<f32>> = None;
        let mut count = 0;

        for line in reader.lines() {
            let text = line?;
            if text.trim().is_empty() {
                continue;
            }

            let vec = self.engine.write().unwrap().embed(&text)?;

            match &mut total_embedding {
                None => total_embedding = Some(vec),
                Some(acc) => {
                    for (i, val) in vec.iter().enumerate() {
                        acc[i] += val;
                    }
                }
            }
            count += 1;
        }

        if let Some(mut final_vec) = total_embedding {
            if count > 0 {
                for x in final_vec.iter_mut() {
                    *x /= count as f32;
                }
            }
            normalize_l2(&mut final_vec);

            let dimension = final_vec.len();
            self.validate_dimension(dimension);

            Ok(EmbedResponse {
                dimension,
                embedding: final_vec,
            })
        } else {
            Err(AppError::InvalidInput("File is empty".to_string()))
        }
    }

    /// 处理 1对N 检索：给定查询文本，在候选文本列表中找到最相似的文本
    pub async fn process_search(&self, req: SearchRequest) -> Result<SearchResponse, AppError> {
        self.validator
            .validate_search(&req.query, &req.texts, req.top_k)?;

        let top_k = std::cmp::min(req.top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K);

        let query_embedding = {
            let mut embedding = self.engine.write().unwrap().embed(&req.query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(req.texts.len());

        for (idx, text) in req.texts.iter().enumerate() {
            let mut embedding = self.engine.write().unwrap().embed(text)?;
            normalize_l2(&mut embedding);

            let score = cosine_similarity(&query_embedding, &embedding)?;
            results.push((idx, score, text.clone()));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_results: Vec<SearchResult> = results
            .into_iter()
            .take(top_k)
            .map(|(idx, score, text)| SearchResult {
                text,
                score,
                index: idx,
            })
            .collect();

        Ok(SearchResponse {
            results: top_results,
        })
    }

    /// 批量处理 1对N 检索（更高效的版本，使用批量推理）
    pub async fn process_search_batch(
        &self,
        query: &str,
        texts: &[String],
        top_k: Option<usize>,
    ) -> Result<SearchResponse, AppError> {
        self.validator.validate_search(query, texts, top_k)?;

        let top_k = std::cmp::min(top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K);

        let query_embedding = {
            let mut embedding = self.engine.write().unwrap().embed(query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let chunk_embeddings = self.engine.write().unwrap().embed_batch(chunk)?;
            for mut emb in chunk_embeddings {
                normalize_l2(&mut emb);
                embeddings.push(emb);
            }
        }

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(texts.len());

        for (idx, (text, emb)) in texts.iter().zip(embeddings.iter()).enumerate() {
            let score = cosine_similarity(&query_embedding, emb)?;
            results.push((idx, score, text.clone()));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_results: Vec<SearchResult> = results
            .into_iter()
            .take(top_k)
            .map(|(idx, score, text)| SearchResult {
                text,
                score,
                index: idx,
            })
            .collect();

        Ok(SearchResponse {
            results: top_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::{DeviceType, EngineType};
    use std::path::PathBuf;
    use tempfile::tempdir;

    struct MockEngine {
        embedding: Vec<f32>,
    }

    impl MockEngine {
        fn new(dimension: usize) -> Self {
            Self {
                embedding: vec![0.1; dimension],
            }
        }
    }

    impl InferenceEngine for MockEngine {
        fn embed(&mut self, _text: &str) -> Result<Vec<f32>, AppError> {
            Ok(self.embedding.clone())
        }

        fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
            Ok(vec![self.embedding.clone(); texts.len()])
        }
    }

    #[tokio::test]
    async fn test_embedding_service_with_matching_dimension() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
        };

        let result = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_with_mismatching_dimension() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
        };

        let result = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_without_dimension_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
        };

        let result = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_without_model_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "Hello world".to_string(),
        };

        let result = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[test]
    fn test_dimension_validation_with_mismatch() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        service.validate_dimension(384);
    }

    #[test]
    fn test_dimension_validation_with_match() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(1024);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        service.validate_dimension(1024);
    }

    #[test]
    fn test_dimension_validation_with_none_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        service.validate_dimension(384);
    }

    #[tokio::test]
    async fn test_embedding_service_with_custom_validator() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let validator = InputValidator::with_default();

        let service = EmbeddingService::with_validator(engine, validator, Some(model_config));

        let req = EmbedRequest {
            text: "Test text for embedding".to_string(),
        };

        let result = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
        assert_eq!(response.embedding.len(), 384);
    }
}
