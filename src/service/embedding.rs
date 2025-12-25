// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::config::model::ModelConfig;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, BatchEmbeddingResult, EmbedRequest, EmbedResponse,
    EmbeddingOutput, FileProcessingStats, ModelInfo, ModelListResponse, ModelMetadata,
    ModelSwitchRequest, ModelSwitchResponse, ParagraphEmbedding, SearchRequest, SearchResponse,
    SearchResult, SimilarityRequest, SimilarityResponse,
};
use crate::engine::{AnyEngine, InferenceEngine};
use crate::error::AppError;
use crate::model::manager::ModelManager;
use crate::utils::{
    cosine_similarity, normalize_l2, AggregationMode, FileValidator, InputValidator, TextValidator,
    DEFAULT_TOP_K, MAX_BATCH_SIZE, MAX_TOP_K,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

pub struct EmbeddingService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    model_manager: Option<Arc<ModelManager>>,
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
            model_manager: None,
        }
    }

    pub fn with_manager(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            model_manager: Some(model_manager),
        }
    }

    pub fn with_validator_and_manager(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
    ) -> Self {
        Self {
            engine,
            validator,
            model_config,
            model_manager,
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

        let mut embedding = self.engine.write().await.embed(&req.text)?;
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

        let f1 = async move { engine1.write().await.embed(&req.source) };
        let f2 = async move { engine2.write().await.embed(&req.target) };

        let (mut v1, mut v2) = tokio::try_join!(f1, f2)?;

        normalize_l2(&mut v1);
        normalize_l2(&mut v2);

        let score = cosine_similarity(&v1, &v2)?;
        Ok(SimilarityResponse { score })
    }

    /// 处理大文件流式向量化 (简单实现：按行平均)
    pub async fn process_file_stream(&self, path: &Path) -> Result<EmbedResponse, AppError> {
        let path_str = path.to_str().ok_or_else(|| {
            AppError::InvalidInput("Invalid path encoding: path contains invalid UTF-8".to_string())
        })?;
        self.validator.validate_file(path_str)?;

        let start_time = std::time::Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let result = self.process_stream_internal(reader, start_time).await;

        if let Err(ref e) = result {
            tracing::error!("File streaming failed: {:?}", e);
        }

        result
    }

    /// 处理文件向量化，支持多种聚合模式
    pub async fn embed_file(
        &self,
        path: &Path,
        mode: AggregationMode,
    ) -> Result<EmbeddingOutput, AppError> {
        let path_str = path.to_str().ok_or_else(|| {
            AppError::InvalidInput("Invalid path encoding: path contains invalid UTF-8".to_string())
        })?;
        self.validator.validate_file(path_str)?;

        let start_time = std::time::Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        match mode {
            AggregationMode::Document => {
                let result = self.process_stream_internal(reader, start_time).await?;
                Ok(EmbeddingOutput::Single(result))
            }
            AggregationMode::Paragraph => self.process_paragraphs(reader, start_time).await,
            AggregationMode::Paragraphs => self.process_paragraphs(reader, start_time).await,
            AggregationMode::Average => {
                let result = self.process_stream_internal(reader, start_time).await?;
                Ok(EmbeddingOutput::Single(result))
            }
            _ => {
                let result = self.process_stream_internal(reader, start_time).await?;
                Ok(EmbeddingOutput::Single(result))
            }
        }
    }

    async fn process_stream_internal(
        &self,
        reader: BufReader<File>,
        start_time: std::time::Instant,
    ) -> Result<EmbedResponse, AppError> {
        let mut total_embedding: Option<Vec<f32>> = None;
        let mut count = 0;

        for line in reader.lines() {
            let text = line?;
            if text.trim().is_empty() {
                continue;
            }

            let vec = self.engine.write().await.embed(&text)?;

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

            let processing_time = start_time.elapsed();
            tracing::info!(
                "Processed {} lines in {:.2}ms",
                count,
                processing_time.as_millis() as f64
            );

            Ok(EmbedResponse {
                dimension,
                embedding: final_vec,
            })
        } else {
            Err(AppError::InvalidInput("File is empty".to_string()))
        }
    }

    async fn process_paragraphs(
        &self,
        reader: BufReader<File>,
        start_time: std::time::Instant,
    ) -> Result<EmbeddingOutput, AppError> {
        use std::io::Read;
        let mut content = String::new();
        reader.into_inner().read_to_string(&mut content)?;

        let paragraphs: Vec<String> = content
            .split("\n\n")
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();

        if paragraphs.is_empty() {
            return Err(AppError::InvalidInput(
                "No paragraphs found in file".to_string(),
            ));
        }

        let mut paragraph_embeddings: Vec<ParagraphEmbedding> =
            Vec::with_capacity(paragraphs.len());

        for (idx, para) in paragraphs.iter().enumerate() {
            if para.trim().is_empty() {
                continue;
            }

            let mut embedding = self.engine.write().await.embed(para)?;
            normalize_l2(&mut embedding);

            let preview = if para.len() > 100 { &para[..100] } else { para };

            paragraph_embeddings.push(ParagraphEmbedding {
                embedding,
                position: idx,
                text_preview: preview.to_string(),
            });
        }

        let processing_time = start_time.elapsed();
        tracing::info!(
            "Processed {} paragraphs in {:.2}ms",
            paragraph_embeddings.len(),
            processing_time.as_millis() as f64
        );

        Ok(EmbeddingOutput::Paragraphs(paragraph_embeddings))
    }

    /// 获取处理统计信息
    pub fn get_processing_stats(&self, path: &Path) -> Result<FileProcessingStats, AppError> {
        let start_time = std::time::Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut lines = 0;
        let mut paragraphs = 0;
        let mut current_para_empty = true;

        for line in reader.lines() {
            let line = line?;
            lines += 1;

            if line.trim().is_empty() {
                if !current_para_empty {
                    paragraphs += 1;
                    current_para_empty = true;
                }
            } else {
                current_para_empty = false;
            }
        }

        if !current_para_empty {
            paragraphs += 1;
        }

        let processing_time = start_time.elapsed();

        Ok(FileProcessingStats {
            total_chunks: lines + paragraphs,
            successful_chunks: lines + paragraphs,
            failed_chunks: 0,
            processing_time_ms: processing_time.as_millis(),
        })
    }

    /// 处理 1对N 检索：给定查询文本，在候选文本列表中找到最相似的文本
    pub async fn process_search(&self, req: SearchRequest) -> Result<SearchResponse, AppError> {
        self.validator
            .validate_search(&req.query, &req.texts, req.top_k)?;

        let top_k = std::cmp::min(req.top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K);

        let query_embedding = {
            let mut embedding = self.engine.write().await.embed(&req.query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(req.texts.len());

        for (idx, text) in req.texts.iter().enumerate() {
            let mut embedding = self.engine.write().await.embed(text)?;
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
            let mut embedding = self.engine.write().await.embed(query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let chunk_embeddings = self.engine.write().await.embed_batch(chunk)?;
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

    /// 处理批量向量化请求
    pub async fn process_batch(
        &self,
        req: BatchEmbedRequest,
    ) -> Result<BatchEmbedResponse, AppError> {
        let start_time = std::time::Instant::now();

        self.validator.validate_batch(&req.texts)?;

        let texts = req.texts;
        let mut results: Vec<BatchEmbeddingResult> = Vec::with_capacity(texts.len());
        let mut dimension: Option<usize> = None;

        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let chunk_embeddings = self.engine.write().await.embed_batch(chunk)?;

            for (text, mut embedding) in chunk.iter().zip(chunk_embeddings.into_iter()) {
                normalize_l2(&mut embedding);

                let text_preview = if text.len() > 100 { &text[..100] } else { text }.to_string();

                let dim = embedding.len();
                if dimension.is_none() {
                    dimension = Some(dim);
                }
                self.validate_dimension(dim);

                results.push(BatchEmbeddingResult {
                    text_preview: text_preview.to_string(),
                    embedding,
                });
            }
        }

        let processing_time = start_time.elapsed();

        Ok(BatchEmbedResponse {
            embeddings: results,
            dimension: dimension.unwrap_or(0),
            processing_time_ms: processing_time.as_millis(),
        })
    }

    pub fn get_model_info(&self) -> Option<ModelInfo> {
        self.model_config.as_ref().map(|config| ModelInfo {
            name: config.name.clone(),
            engine_type: config.engine_type.to_string(),
            dimension: config.expected_dimension,
            is_loaded: true,
        })
    }

    pub fn get_model_metadata(&self) -> Option<ModelMetadata> {
        self.model_config.as_ref().map(|config| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let loaded_at = Some(format!("{}", now));

            ModelMetadata {
                name: config.name.clone(),
                version: "1.0.0".to_string(),
                engine_type: config.engine_type.to_string(),
                dimension: config.expected_dimension,
                max_input_length: 512,
                is_loaded: true,
                loaded_at,
            }
        })
    }

    pub async fn switch_model(
        &mut self,
        req: ModelSwitchRequest,
    ) -> Result<ModelSwitchResponse, AppError> {
        let previous_model = self.model_config.as_ref().map(|c| c.name.clone());

        tracing::info!(
            "Switching model from {:?} to {}",
            previous_model,
            req.model_name
        );

        if let Some(ref prev_name) = previous_model {
            if prev_name == &req.model_name {
                return Ok(ModelSwitchResponse {
                    previous_model: previous_model.clone(),
                    current_model: req.model_name,
                    success: true,
                    message: "Already using this model".to_string(),
                });
            }
        }

        let model_config = ModelConfig {
            name: req.model_name.clone(),
            engine_type: self
                .model_config
                .as_ref()
                .map(|c| c.engine_type.clone())
                .unwrap_or(crate::config::model::EngineType::Candle),
            model_path: req.model_path.clone().unwrap_or_else(|| std::path::PathBuf::from(&req.model_name)),
            tokenizer_path: req.tokenizer_path.clone().or_else(|| {
                self.model_config
                    .as_ref()
                    .and_then(|c| c.tokenizer_path.clone())
            }),
            device: req.device.clone().or_else(|| {
                self.model_config
                    .as_ref()
                    .map(|c| c.device.clone())
            }).unwrap_or(crate::config::model::DeviceType::Cpu),
            max_batch_size: req.max_batch_size.unwrap_or_else(|| {
                self.model_config
                    .as_ref()
                    .map(|c| c.max_batch_size)
                    .unwrap_or(32)
            }),
            pooling_mode: req.pooling_mode.clone().or_else(|| {
                self.model_config
                    .as_ref()
                    .and_then(|c| c.pooling_mode.clone())
            }),
            expected_dimension: req.expected_dimension.or_else(|| {
                self.model_config
                    .as_ref()
                    .and_then(|c| c.expected_dimension)
            }),
        };

        if let Some(ref manager) = self.model_manager {
            tracing::debug!("Using ModelManager for model switching");
            let _loaded_model = manager.load(&model_config).await?;

            if let Some(ref prev_name) = previous_model {
                tracing::info!("Unloading previous model: {}", prev_name);
                let _ = manager.unload(prev_name).await;
            }
        }

        let new_engine = AnyEngine::new(
            &model_config,
            model_config.engine_type.clone(),
            crate::config::model::Precision::Fp32,
        )?;

        self.engine = Arc::new(RwLock::new(new_engine));
        self.model_config = Some(model_config);

        tracing::info!("Model switched successfully to {}", req.model_name);

        Ok(ModelSwitchResponse {
            previous_model,
            current_model: req.model_name,
            success: true,
            message: "Model switched successfully".to_string(),
        })
    }

    pub async fn unload_model(&mut self, name: &str) -> Result<(), AppError> {
        if let Some(ref manager) = self.model_manager {
            manager.unload(name).await?;
            tracing::info!("Model {} unloaded via ModelManager", name);
        }

        if self.model_config.as_ref().map(|c| &c.name) == Some(&name.to_string()) {
            self.model_config = None;
            tracing::info!("Local model config cleared for {}", name);
        }

        Ok(())
    }

    pub async fn list_loaded_models(&self) -> Vec<String> {
        if let Some(ref manager) = self.model_manager {
            manager.list_loaded().await
        } else {
            vec![]
        }
    }

    pub fn has_model_manager(&self) -> bool {
        self.model_manager.is_some()
    }

    pub fn list_available_models(&self) -> ModelListResponse {
        let model_info = ModelInfo {
            name: self
                .model_config
                .as_ref()
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "default".to_string()),
            engine_type: self
                .model_config
                .as_ref()
                .map(|c| c.engine_type.to_string())
                .unwrap_or_else(|| "candle".to_string()),
            dimension: self
                .model_config
                .as_ref()
                .and_then(|c| c.expected_dimension),
            is_loaded: true,
        };

        let models = vec![model_info];
        let total_count = models.len();

        ModelListResponse {
            models,
            total_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, Precision};
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

        fn precision(&self) -> Precision {
            Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn test_process_text_with_model_config() {
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
        let _temp_dir = tempdir().unwrap();
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
        let _temp_dir = tempdir().unwrap();
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

        let service = EmbeddingService::with_validator_and_manager(
            engine,
            validator,
            Some(model_config),
            None,
        );

        let req = EmbedRequest {
            text: "Test text for embedding".to_string(),
        };

        let result: Result<EmbedResponse, AppError> = service.process_text(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
        assert_eq!(response.embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_embed_file_document_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("test.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2\nLine 3").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::Document)
            .await;

        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Single(response) => {
                assert_eq!(response.dimension, 384);
                assert_eq!(response.embedding.len(), 384);
            }
            _ => panic!("Expected Single embedding output"),
        }
    }

    #[tokio::test]
    async fn test_embed_file_paragraph_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_content =
            "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.";
        let test_file_path = temp_dir.path().join("test_paragraphs.txt");
        std::fs::write(&test_file_path, test_content).unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::Paragraph)
            .await;

        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Paragraphs(paragraphs) => {
                assert_eq!(paragraphs.len(), 3);
                for (idx, para) in paragraphs.iter().enumerate() {
                    assert_eq!(para.position, idx);
                    assert!(para.embedding.len() == 384);
                    assert!(!para.text_preview.is_empty());
                }
            }
            _ => panic!("Expected Paragraphs embedding output"),
        }
    }

    #[tokio::test]
    async fn test_embed_file_paragraphs_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_content =
            "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let test_file_path = temp_dir.path().join("test_multi_para.txt");
        std::fs::write(&test_file_path, test_content).unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::Paragraphs)
            .await;

        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Paragraphs(paragraphs) => {
                assert_eq!(paragraphs.len(), 3);
                for para in &paragraphs {
                    assert_eq!(para.embedding.len(), 384);
                }
            }
            _ => panic!("Expected Paragraphs embedding output"),
        }
    }

    #[tokio::test]
    async fn test_embed_file_average_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("test_avg.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2\nLine 3").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::Average)
            .await;

        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Single(response) => {
                assert_eq!(response.dimension, 384);
                assert_eq!(response.embedding.len(), 384);
            }
            _ => panic!("Expected Single embedding output"),
        }
    }

    #[tokio::test]
    async fn test_process_paragraphs_empty_file() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("empty.txt");
        std::fs::write(&test_file_path, "").unwrap();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let reader = std::io::BufReader::new(file);
        let start_time = std::time::Instant::now();

        let result = service.process_paragraphs(reader, start_time).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_processing_stats() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let test_content = "Line 1\nLine 2\n\nParagraph 2 Line 1\nParagraph 2 Line 2";
        let test_file_path = temp_dir.path().join("stats_test.txt");
        std::fs::write(&test_file_path, test_content).unwrap();

        let result = service.get_processing_stats(&test_file_path);

        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!(stats.total_chunks > 0);
        assert!(stats.total_chunks > 0);
    }

    #[tokio::test]
    async fn test_process_batch_basic() {
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

        let texts = vec![
            "Hello world".to_string(),
            "Rust is great".to_string(),
            "Embedding vectors".to_string(),
        ];

        let req = BatchEmbedRequest {
            texts: texts.clone(),
            mode: None,
        };

        let result = service.process_batch(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 3);
        assert_eq!(response.dimension, 384);

        for (i, emb) in response.embeddings.iter().enumerate() {
            assert_eq!(emb.embedding.len(), 384);
            assert_eq!(emb.text_preview, texts[i]);
        }
    }

    #[tokio::test]
    async fn test_process_batch_empty() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec![],
            mode: None,
        };

        let result = service.process_batch(req).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_batch_large_with_chunking() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(_temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 3,
            pooling_mode: None,
            expected_dimension: Some(384),
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let texts: Vec<String> = (0..10).map(|i| format!("Test text {}", i)).collect();

        let req = BatchEmbedRequest {
            texts: texts.clone(),
            mode: None,
        };

        let result = service.process_batch(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 10);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_process_batch_single() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let texts = vec!["Single text".to_string()];

        let req = BatchEmbedRequest { texts, mode: None };

        let result = service.process_batch(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_process_batch_with_long_text_preview() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = MockEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let long_text = "A".repeat(200);
        let texts = vec![long_text.clone()];

        let req = BatchEmbedRequest { texts, mode: None };

        let result = service.process_batch(req).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].text_preview.len(), 100);
    }
}
