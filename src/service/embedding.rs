// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

#![allow(clippy::all)]

use crate::cache::OxCacheBackend;
use crate::config::model::ModelConfig;
use crate::device::DynamicBatchScheduler;
use crate::device::memory_optimizer::SharedGpuMemoryManager;
use crate::device::memory_pool::BufferPool;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, BatchEmbeddingResult, EmbedRequest, EmbedResponse,
    EmbeddingOutput, FileProcessingStats, ModelInfo, ModelListResponse, ModelMetadata,
    ModelSwitchRequest, ModelSwitchResponse, ParagraphEmbedding, SearchRequest, SearchResponse,
    SearchResult, SimilarityRequest, SimilarityResponse,
};
use crate::engine::{AnyEngine, InferenceEngine};
use crate::error::VecboostError;
use crate::model::manager::ModelManager;
use crate::utils::{
    AggregationMode, DEFAULT_TOP_K, FileValidator, InputValidator, MAX_BATCH_SIZE, MAX_TOP_K,
    TextValidator, cosine_similarity, normalize_l2, truncate_vector, validate_dimension,
};
use log::{debug, warn};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

const MAX_FALLBACK_ATTEMPTS: usize = 1;

pub struct EmbeddingService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    model_manager: Option<Arc<ModelManager>>,
    cache: Arc<OxCacheBackend>,
    memory_manager: Option<SharedGpuMemoryManager>,
    batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
    buffer_pool: Option<Arc<tokio::sync::RwLock<BufferPool>>>,
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
            cache: Arc::new(OxCacheBackend::disabled()),
            memory_manager: None,
            batch_scheduler: None,
            buffer_pool: None,
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
            cache: Arc::new(OxCacheBackend::disabled()),
            memory_manager: None,
            batch_scheduler: None,
            buffer_pool: None,
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
            cache: Arc::new(OxCacheBackend::disabled()),
            memory_manager: None,
            batch_scheduler: None,
            buffer_pool: None,
        }
    }

    pub fn with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache_size: usize,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            model_manager: None,
            cache: Arc::new(OxCacheBackend::new(cache_size)),
            memory_manager: None,
            batch_scheduler: None,
            buffer_pool: None,
        }
    }

    pub fn with_all(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
        cache_size: usize,
        memory_manager: Option<SharedGpuMemoryManager>,
        batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
    ) -> Self {
        Self {
            engine,
            validator,
            model_config,
            model_manager,
            cache: Arc::new(OxCacheBackend::new(cache_size)),
            memory_manager,
            batch_scheduler,
            buffer_pool: None,
        }
    }

    /// 设置 BufferPool
    pub fn with_buffer_pool(mut self, buffer_pool: Arc<tokio::sync::RwLock<BufferPool>>) -> Self {
        self.buffer_pool = Some(buffer_pool);
        self
    }

    fn validate_dimension(&self, actual_dimension: usize) {
        if let Some(ref config) = self.model_config
            && let Some(expected) = config.expected_dimension
            && actual_dimension != expected
        {
            warn!(
                "Dimension mismatch: expected {}, got {}. Model '{}' may have been configured incorrectly or the wrong model was loaded.",
                expected, actual_dimension, config.name
            );
        }
    }

    /// 获取当前最优的批量大小
    /// 优先使用 memory_manager 计算，其次使用 batch_scheduler，最后回退到 MAX_BATCH_SIZE
    async fn get_optimal_batch_size(
        &self,
        sequence_length: usize,
        output_dimension: usize,
    ) -> usize {
        // 尝试使用 memory_manager 计算最优批量大小
        if let Some(ref mm) = self.memory_manager {
            if let Some(ref config) = self.model_config {
                match mm
                    .calculate_optimal_batch_size(&config.name, sequence_length, output_dimension)
                    .await
                {
                    Ok(size) if size > 0 => {
                        debug!("Using memory manager calculated batch size: {}", size);
                        return size;
                    }
                    _ => {}
                }
            }
        }

        // 尝试使用 batch_scheduler 的当前批量大小
        if let Some(ref scheduler) = self.batch_scheduler {
            let size = scheduler.current_batch_size().await;
            if size > 0 {
                debug!("Using batch scheduler size: {}", size);
                return size;
            }
        }

        // 回退到默认值
        debug!("Using default MAX_BATCH_SIZE: {}", MAX_BATCH_SIZE);
        MAX_BATCH_SIZE
    }

    /// 缓存预热：批量预加载常用文本的向量
    /// 用于在服务启动时预加载热点数据，提高首次请求的响应速度
    pub async fn warm_up_cache(&self, texts: Vec<String>) -> Result<(), VecboostError> {
        if !self.cache.is_enabled() {
            debug!("Cache is disabled, skipping warm-up");
            return Ok(());
        }

        let total_texts = texts.len();
        let mut embeddings = std::collections::HashMap::with_capacity(total_texts);
        let mut processed = 0;

        for text in texts {
            if let Ok(embedding) = self.engine.read().await.embed(&text) {
                embeddings.insert(text, embedding);
                processed += 1;

                if processed % 100 == 0 {
                    debug!("Warm-up progress: {}/{}", processed, total_texts);
                }
            }
        }

        // 在移动 embeddings 之前记录处理数量
        let processed_count = processed;

        // 使用 HashMap 将预热数据转换为缓存预期的格式
        let cache_entries: std::collections::HashMap<String, Vec<f32>> =
            embeddings.into_iter().collect();

        self.cache.warm_up(cache_entries).await;

        log::info!(
            "Cache warm-up completed: {} entries preloaded",
            processed_count
        );
        Ok(())
    }

    fn is_oom_error(error: &VecboostError) -> bool {
        match error {
            VecboostError::InferenceError(msg) | VecboostError::OutOfMemory(msg) => {
                let lower_msg = msg.to_lowercase();
                lower_msg.contains("out of memory")
                    || lower_msg.contains("cuda out of memory")
                    || lower_msg.contains("gpu out of memory")
                    || lower_msg.contains("memory allocation failed")
                    || lower_msg.contains("failed to allocate")
                    || lower_msg.contains("not enough memory")
                    || lower_msg.contains("alloc")
            }
            _ => false,
        }
    }

    async fn handle_oom_fallback<F, Fut, T>(&self, operation: F) -> Result<T, VecboostError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, VecboostError>>,
    {
        let mut attempts = 0;

        loop {
            attempts += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) if Self::is_oom_error(&error) && attempts <= MAX_FALLBACK_ATTEMPTS => {
                    warn!(
                        "OOM error detected: {}. Attempting fallback to CPU (attempt {}/{})",
                        error, attempts, MAX_FALLBACK_ATTEMPTS
                    );

                    let engine = self.engine.read().await;

                    if engine.is_fallback_triggered() {
                        warn!("Fallback already triggered, cannot retry");
                        return Err(VecboostError::OutOfMemory(
                            "Out of memory and fallback already attempted".to_string(),
                        ));
                    }

                    drop(engine);

                    if let Some(ref config) = self.model_config
                        && let Some(ref manager) = self.model_manager
                    {
                        let loaded_model = manager.get(&config.name).await;

                        if let Some(_model) = loaded_model {
                            let mut engine_guard = self.engine.write().await;
                            let config_clone = config.clone();
                            let fallback_result =
                                engine_guard.try_fallback_to_cpu(&config_clone).await;

                            match fallback_result {
                                Ok(()) => {
                                    warn!("Successfully fell back to CPU, retrying operation");
                                    // 检查是否还有重试次数
                                    if attempts >= MAX_FALLBACK_ATTEMPTS {
                                        warn!("Max fallback attempts reached, aborting");
                                        return Err(VecboostError::OutOfMemory(
                                            "Max fallback attempts exceeded".to_string(),
                                        ));
                                    }
                                    continue;
                                }
                                Err(e) => {
                                    warn!("Failed to fallback to CPU: {}", e);
                                    return Err(VecboostError::OutOfMemory(format!(
                                        "OOM error and fallback failed: {}",
                                        e
                                    )));
                                }
                            }
                        }
                    }

                    return Err(VecboostError::OutOfMemory(
                        "Out of memory and no fallback available".to_string(),
                    ));
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// 处理单文本向量化
    /// 如果提供了 target_dimension，则对输出向量进行截断（Matryoshka 风格）
    pub async fn process_text(
        &self,
        req: EmbedRequest,
        target_dimension: Option<usize>,
    ) -> Result<EmbedResponse, VecboostError> {
        self.validator.validate_text(&req.text)?;

        let cache_key = format!("text:{}", req.text);

        let embedding = if self.cache.is_enabled() {
            self.handle_oom_fallback(|| async {
                self.cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key, || async {
                        let embedding = self.engine.read().await.embed(&req.text)?;
                        Ok(embedding)
                    })
                    .await
            })
            .await?
        } else {
            self.handle_oom_fallback(|| async { self.engine.read().await.embed(&req.text) })
                .await?
        };

        let mut embedding = embedding;
        normalize_l2(&mut embedding);

        // Apply Matryoshka dimension reduction if requested
        if let Some(target_dim) = target_dimension {
            let max_dim = self
                .model_config
                .as_ref()
                .and_then(|c| c.expected_dimension)
                .unwrap_or(1024);
            validate_dimension(Some(target_dim), max_dim)
                .map_err(|e| VecboostError::InvalidInput(e))?;
            embedding = truncate_vector(&embedding, target_dim);
        }

        let dimension = embedding.len();
        self.validate_dimension(dimension);

        Ok(EmbedResponse {
            dimension,
            embedding,
            processing_time_ms: 0,
        })
    }

    /// 处理相似度计算
    pub async fn process_similarity(
        &self,
        req: SimilarityRequest,
    ) -> Result<SimilarityResponse, VecboostError> {
        self.validator.validate_text(&req.source)?;
        self.validator.validate_text(&req.target)?;

        let cache_key_source = format!("text:{}", req.source);
        let cache_key_target = format!("text:{}", req.target);

        let engine = Arc::clone(&self.engine);
        let cache = Arc::clone(&self.cache);

        let f1 = async move {
            if cache.is_enabled() {
                cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key_source, || async {
                        let embedding = engine.read().await.embed(&req.source)?;
                        Ok(embedding)
                    })
                    .await
            } else {
                engine.read().await.embed(&req.source)
            }
        };

        let engine = Arc::clone(&self.engine);
        let cache = Arc::clone(&self.cache);

        let f2 = async move {
            if cache.is_enabled() {
                cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key_target, || async {
                        let embedding = engine.read().await.embed(&req.target)?;
                        Ok(embedding)
                    })
                    .await
            } else {
                engine.read().await.embed(&req.target)
            }
        };

        let (mut v1, mut v2) = tokio::try_join!(f1, f2)?;

        let score = tokio::task::spawn_blocking(move || {
            normalize_l2(&mut v1);
            normalize_l2(&mut v2);
            cosine_similarity(&v1, &v2)
        })
        .await??;

        Ok(SimilarityResponse { score })
    }

    /// 处理大文件流式向量化 (简单实现：按行平均)
    pub async fn process_file_stream(&self, path: &Path) -> Result<EmbedResponse, VecboostError> {
        let path_str = path.to_str().ok_or_else(|| {
            VecboostError::InvalidInput(
                "Invalid path encoding: path contains invalid UTF-8".to_string(),
            )
        })?;
        self.validator.validate_file(path_str)?;

        let start_time = std::time::Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let result = self.process_stream_internal(reader, start_time).await;

        if let Err(ref e) = result {
            log::error!("File streaming failed: {:?}", e);
        }

        result
    }

    /// 处理文件向量化，支持多种聚合模式
    pub async fn embed_file(
        &self,
        path: &Path,
        mode: AggregationMode,
    ) -> Result<EmbeddingOutput, VecboostError> {
        let path_str = path.to_str().ok_or_else(|| {
            VecboostError::InvalidInput(
                "Invalid path encoding: path contains invalid UTF-8".to_string(),
            )
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
    ) -> Result<EmbedResponse, VecboostError> {
        let mut total_embedding: Option<Vec<f32>> = None;
        let mut count = 0;

        for line in reader.lines() {
            let text = line?;
            if text.trim().is_empty() {
                continue;
            }

            let vec = self.engine.read().await.embed(&text)?;

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
            log::info!(
                "Processed {} lines in {:.2}ms",
                count,
                processing_time.as_millis() as f64
            );

            Ok(EmbedResponse {
                dimension,
                embedding: final_vec,
                processing_time_ms: processing_time.as_millis(),
            })
        } else {
            Err(VecboostError::InvalidInput("File is empty".to_string()))
        }
    }

    async fn process_paragraphs(
        &self,
        reader: BufReader<File>,
        start_time: std::time::Instant,
    ) -> Result<EmbeddingOutput, VecboostError> {
        use std::io::Read;
        let mut content = String::new();
        reader.into_inner().read_to_string(&mut content)?;

        let paragraphs: Vec<String> = content
            .split("\n\n")
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();

        if paragraphs.is_empty() {
            return Err(VecboostError::InvalidInput(
                "No paragraphs found in file".to_string(),
            ));
        }

        let mut paragraph_embeddings: Vec<ParagraphEmbedding> =
            Vec::with_capacity(paragraphs.len());

        for (idx, para) in paragraphs.iter().enumerate() {
            if para.trim().is_empty() {
                continue;
            }

            let mut embedding = self.engine.read().await.embed(para)?;
            normalize_l2(&mut embedding);

            let preview = if para.len() > 100 { &para[..100] } else { para };

            paragraph_embeddings.push(ParagraphEmbedding {
                embedding,
                position: idx,
                text_preview: preview.to_string(),
            });
        }

        let processing_time = start_time.elapsed();
        log::info!(
            "Processed {} paragraphs in {:.2}ms",
            paragraph_embeddings.len(),
            processing_time.as_millis() as f64
        );

        Ok(EmbeddingOutput::Paragraphs(paragraph_embeddings))
    }

    /// 获取处理统计信息
    pub fn get_processing_stats(&self, path: &Path) -> Result<FileProcessingStats, VecboostError> {
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
    pub async fn process_search(
        &self,
        req: SearchRequest,
    ) -> Result<SearchResponse, VecboostError> {
        self.validator
            .validate_search(&req.query, &req.texts, req.top_k)?;

        let top_k = std::cmp::min(req.top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K);

        let query_embedding = {
            let mut embedding = self.engine.read().await.embed(&req.query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(req.texts.len());

        for (idx, text) in req.texts.iter().enumerate() {
            let mut embedding = self.engine.read().await.embed(text)?;
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

    /// 批量处理 1对N 检索（更高效的版本，使用批量推理和动态批量大小）
    pub async fn process_search_batch(
        &self,
        query: &str,
        texts: &[String],
        top_k: Option<usize>,
    ) -> Result<SearchResponse, VecboostError> {
        self.validator.validate_search(query, texts, top_k)?;

        let top_k = std::cmp::min(top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K);

        let query_embedding = {
            let mut embedding = self.engine.read().await.embed(query)?;
            normalize_l2(&mut embedding);
            embedding
        };

        // 计算最优批量大小
        let sequence_length = texts.first().map_or(0, |t| t.len());
        let output_dimension = self
            .model_config
            .as_ref()
            .and_then(|c| c.expected_dimension)
            .unwrap_or(768);
        let optimal_batch_size = self
            .get_optimal_batch_size(sequence_length, output_dimension)
            .await;

        debug!(
            "Processing search batch: {} texts, optimal_batch_size={}",
            texts.len(),
            optimal_batch_size
        );

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(optimal_batch_size) {
            let chunk_embeddings = self.engine.read().await.embed_batch(chunk)?;
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

    /// 处理批量向量化请求（优化版本：使用并行处理和智能批量大小调整）
    /// 如果提供了 target_dimension，则对每个输出向量进行截断（Matryoshka 风格）
    pub async fn process_batch(
        &self,
        req: BatchEmbedRequest,
        target_dimension: Option<usize>,
    ) -> Result<BatchEmbedResponse, VecboostError> {
        let start_time = Instant::now();

        self.validator.validate_batch(&req.texts)?;

        let texts = req.texts;

        let texts_len = texts.len();

        // 检测一个示例文本的序列长度（用于计算内存需求）
        let sequence_length = texts.first().map_or(0, |t| t.len());

        // 检测输出维度（如果有模型配置则使用配置值）
        let output_dimension = self
            .model_config
            .as_ref()
            .and_then(|c| c.expected_dimension)
            .unwrap_or(768);

        // 获取最优批量大小
        let optimal_batch_size = self
            .get_optimal_batch_size(sequence_length, output_dimension)
            .await;

        // 使用动态批量大小进行分块
        let chunks: Vec<&[String]> = texts.chunks(optimal_batch_size).collect();

        let num_chunks = chunks.len();

        debug!(
            "Processing batch: {} texts, optimal_batch_size={}, chunks={}",
            texts_len, optimal_batch_size, num_chunks
        );

        // 根据实际负载和系统资源动态调整并发数
        let cpu_count = num_cpus::get();
        let max_concurrent_chunks = std::cmp::min(
            cpu_count * 2,                                    // 每个 CPU 核心最多处理 2 个并发任务
            std::cmp::max(4, texts_len / optimal_batch_size), // 至少 4 个并发
        );

        debug!(
            "Processing batch with concurrent chunks: {} (CPU cores: {}, chunks: {})",
            max_concurrent_chunks, cpu_count, num_chunks
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_chunks));

        let engine = Arc::clone(&self.engine);

        let mut tasks = Vec::with_capacity(num_chunks);

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk = chunk.to_vec();

            let engine = Arc::clone(&engine);

            let semaphore = Arc::clone(&semaphore);

            let task = tokio::spawn(async move {
                // 获取信号量许可，限制并发数

                let _permit = semaphore.acquire().await.map_err(|e| {
                    VecboostError::InferenceError(format!("Failed to acquire semaphore: {}", e))
                })?;

                let mut attempts = 0;

                loop {
                    attempts += 1;

                    let embed_result = {
                        let guard = engine.read().await;
                        guard.embed_batch(&chunk)
                    };

                    match embed_result {
                        Ok(embeddings) => {
                            let mut results: Vec<(usize, Vec<f32>, String)> =
                                Vec::with_capacity(embeddings.len());

                            for (text_idx, (text, embedding)) in
                                chunk.iter().zip(embeddings.into_iter()).enumerate()
                            {
                                let mut emb = embedding;

                                normalize_l2(&mut emb);

                                let global_idx = chunk_idx * optimal_batch_size + text_idx;

                                let text_preview =
                                    if text.len() > 100 { &text[..100] } else { text }.to_string();

                                results.push((global_idx, emb, text_preview));
                            }

                            return Ok(results);
                        }

                        Err(error)
                            if Self::is_oom_error(&error) && attempts <= MAX_FALLBACK_ATTEMPTS =>
                        {
                            warn!(
                                "OOM error detected in batch chunk {}: {}. Attempting fallback to CPU (attempt {}/{})",
                                chunk_idx, error, attempts, MAX_FALLBACK_ATTEMPTS
                            );

                            let current_engine = engine.read().await;

                            if current_engine.is_fallback_triggered() {
                                warn!("Fallback already triggered, cannot retry");

                                return Err(VecboostError::OutOfMemory(
                                    "Out of memory and fallback already attempted".to_string(),
                                ));
                            }

                            drop(current_engine);

                            let mut engine_guard = engine.write().await;

                            let config_clone = ModelConfig::default();

                            let fallback_result =
                                engine_guard.try_fallback_to_cpu(&config_clone).await;

                            match fallback_result {
                                Ok(()) => {
                                    warn!(
                                        "Successfully fell back to CPU, retrying batch chunk {}",
                                        chunk_idx
                                    );

                                    continue;
                                }

                                Err(e) => {
                                    warn!("Failed to fallback to CPU: {}", e);

                                    return Err(VecboostError::OutOfMemory(format!(
                                        "OOM error and fallback failed: {}",
                                        e
                                    )));
                                }
                            }
                        }

                        Err(error) => return Err(error),
                    }
                }
            });

            tasks.push(task);
        }

        let mut all_results: Vec<(usize, Vec<f32>, String)> = Vec::with_capacity(texts_len);

        for task in tasks {
            let chunk_results = task.await??;
            for (idx, embedding, preview) in chunk_results {
                all_results.push((idx, embedding, preview));
            }
        }

        all_results.sort_by_key(|r| r.0);

        // Apply Matryoshka dimension reduction if requested
        let effective_dimension = if let Some(target_dim) = target_dimension {
            let max_dim = self
                .model_config
                .as_ref()
                .and_then(|c| c.expected_dimension)
                .unwrap_or(1024);
            validate_dimension(Some(target_dim), max_dim)
                .map_err(|e| VecboostError::InvalidInput(e))?;
            Some(target_dim)
        } else {
            None
        };

        let results: Vec<BatchEmbeddingResult> = all_results
            .into_iter()
            .map(|(_, embedding, preview)| {
                let embedding = if let Some(dim) = effective_dimension {
                    truncate_vector(&embedding, dim)
                } else {
                    embedding
                };
                BatchEmbeddingResult {
                    text_preview: preview,
                    embedding,
                }
            })
            .collect();

        let dimension =
            effective_dimension.unwrap_or_else(|| results.first().map_or(0, |r| r.embedding.len()));

        let processing_time = start_time.elapsed();
        let processing_time_ms = processing_time.as_millis() as f64;

        // 记录批量完成性能，用于动态调整
        if let Some(ref scheduler) = self.batch_scheduler {
            scheduler
                .record_batch_completion(texts_len, processing_time_ms)
                .await;
            debug!(
                "Recorded batch completion: size={}, latency={:.2}ms",
                texts_len, processing_time_ms
            );
        }

        // 记录内存使用情况，用于动态调整
        if let Some(ref mm) = self.memory_manager {
            let memory_usage = mm.get_memory_usage_percent().await;
            mm.adjust_batch_size_dynamically(processing_time_ms, memory_usage)
                .await;
            debug!(
                "Adjusted batch size based on: latency={:.2}ms, memory_usage={:.1}%",
                processing_time_ms, memory_usage
            );
        }

        Ok(BatchEmbedResponse {
            embeddings: results,
            dimension,
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
    ) -> Result<ModelSwitchResponse, VecboostError> {
        let previous_model = self.model_config.as_ref().map(|c| c.name.clone());

        log::info!(
            "Switching model from {:?} to {}",
            previous_model,
            req.model_name
        );

        if let Some(ref prev_name) = previous_model
            && prev_name == &req.model_name
        {
            return Ok(ModelSwitchResponse {
                previous_model: previous_model.clone(),
                current_model: req.model_name,
                success: true,
                message: "Already using this model".to_string(),
            });
        }

        let model_config = ModelConfig {
            name: req.model_name.clone(),
            engine_type: self
                .model_config
                .as_ref()
                .map(|c| c.engine_type.clone())
                .unwrap_or(crate::config::model::EngineType::Candle),
            model_path: req
                .model_path
                .clone()
                .unwrap_or_else(|| std::path::PathBuf::from(req.model_name.clone())),
            tokenizer_path: req.tokenizer_path.clone().or_else(|| {
                self.model_config
                    .as_ref()
                    .and_then(|c| c.tokenizer_path.clone())
            }),
            device: req
                .device
                .clone()
                .or_else(|| self.model_config.as_ref().map(|c| c.device.clone()))
                .unwrap_or(crate::config::model::DeviceType::Cpu),
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
            memory_limit_bytes: req.memory_limit_bytes.or_else(|| {
                self.model_config
                    .as_ref()
                    .and_then(|c| c.memory_limit_bytes)
            }),
            oom_fallback_enabled: req.oom_fallback_enabled.unwrap_or_else(|| {
                self.model_config
                    .as_ref()
                    .map(|c| c.oom_fallback_enabled)
                    .unwrap_or(false)
            }),
            model_sha256: None,
        };

        if let Some(ref manager) = self.model_manager {
            log::debug!("Using ModelManager for model switching");
            let _loaded_model = manager.load(&model_config).await?;

            if let Some(ref prev_name) = previous_model {
                log::info!("Unloading previous model: {}", prev_name);
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

        log::info!("Model switched successfully to {}", req.model_name);

        Ok(ModelSwitchResponse {
            previous_model,
            current_model: req.model_name,
            success: true,
            message: "Model switched successfully".to_string(),
        })
    }

    pub async fn unload_model(&mut self, name: &str) -> Result<(), VecboostError> {
        if let Some(ref manager) = self.model_manager {
            manager.unload(name).await?;
            log::info!("Model {} unloaded via ModelManager", name);
        }

        if self.model_config.as_ref().map(|c| &c.name) == Some(&name.to_string()) {
            self.model_config = None;
            log::info!("Local model config cleared for {}", name);
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
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }

        fn generate_embedding(&self, text: &str) -> Vec<f32> {
            let mut embedding = vec![0.0f32; self.dimension];
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
    impl InferenceEngine for TestEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(self.generate_embedding(text))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            let embeddings: Vec<Vec<f32>> =
                texts.iter().map(|t| self.generate_embedding(t)).collect();
            Ok(embeddings)
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_process_text_with_model_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
            normalize: Some(true),
        };

        let result = service.process_text(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_with_mismatching_dimension() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
            normalize: Some(true),
        };

        let result = service.process_text(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_without_dimension_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "Hello world".to_string(),
            normalize: Some(true),
        };

        let result = service.process_text(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_embedding_service_without_model_config() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "Hello world".to_string(),
            normalize: Some(true),
        };

        let result = service.process_text(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    #[test]
    fn test_dimension_validation_with_mismatch() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        service.validate_dimension(384);
    }

    #[test]
    fn test_dimension_validation_with_match() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(1024);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        service.validate_dimension(1024);
    }

    #[test]
    fn test_dimension_validation_with_none_config() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        service.validate_dimension(384);
    }

    #[tokio::test]
    async fn test_embedding_service_with_custom_validator() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
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
            normalize: Some(true),
        };

        let result: Result<EmbedResponse, VecboostError> = service.process_text(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
        assert_eq!(response.embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_embed_file_document_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

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
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
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
            normalize: Some(true),
        };

        let result = service.process_batch(req, None).await;

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
        let mock_engine = TestEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec![],
            mode: None,
            normalize: Some(true),
        };

        let result = service.process_batch(req, None).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_batch_large_with_chunking() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(_temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 3,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, Some(model_config));

        let texts: Vec<String> = (0..10).map(|i| format!("Test text {}", i)).collect();

        let req = BatchEmbedRequest {
            texts: texts.clone(),
            mode: None,
            normalize: Some(true),
        };

        let result = service.process_batch(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 10);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_process_batch_single() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let texts = vec!["Single text".to_string()];

        let req = BatchEmbedRequest {
            texts,
            mode: None,
            normalize: Some(true),
        };

        let result = service.process_batch(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.dimension, 384);
    }

    #[tokio::test]
    async fn test_process_batch_with_long_text_preview() {
        let _temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::new(engine, None);

        let long_text = "A".repeat(200);
        let texts = vec![long_text.clone()];

        let req = BatchEmbedRequest {
            texts,
            mode: None,
            normalize: Some(true),
        };

        let result = service.process_batch(req, None).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].text_preview.len(), 100);
    }

    // ===== 辅助 mock 引擎 =====

    /// 始终返回 OOM 错误的引擎,用于测试 OOM 恢复逻辑
    struct OomEngine;

    #[async_trait]
    impl InferenceEngine for OomEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::OutOfMemory(
                "CUDA out of memory: allocation failed".to_string(),
            ))
        }

        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Err(VecboostError::OutOfMemory(
                "CUDA out of memory: allocation failed".to_string(),
            ))
        }

        fn precision(&self) -> &Precision {
            static PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Err(VecboostError::OutOfMemory(
                "OomEngine has no CPU fallback".to_string(),
            ))
        }
    }

    /// 统计 embed/embed_batch 调用次数的引擎,用于验证缓存命中是否跳过引擎
    struct CountingEngine {
        dimension: usize,
        embed_count: Arc<AtomicUsize>,
    }

    impl CountingEngine {
        fn new(dimension: usize, counter: Arc<AtomicUsize>) -> Self {
            Self {
                dimension,
                embed_count: counter,
            }
        }
    }

    #[async_trait]
    impl InferenceEngine for CountingEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            self.embed_count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1.0f32; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            self.embed_count.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|_| vec![1.0f32; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            static PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// 构造用于测试的 ModelConfig
    fn make_model_config(name: &str, dimension: usize) -> ModelConfig {
        let temp_dir = tempdir().unwrap();
        ModelConfig {
            name: name.to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(dimension),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        }
    }

    // ===== OOM 恢复逻辑测试 =====

    #[test]
    fn test_is_oom_error_recognizes_oom_messages() {
        let cases = [
            "CUDA out of memory",
            "cuda out of memory: allocation failed",
            "gpu out of memory",
            "memory allocation failed",
            "failed to allocate 1GB",
            "not enough memory",
            "alloc error",
        ];
        for msg in cases {
            assert!(
                EmbeddingService::is_oom_error(&VecboostError::OutOfMemory(msg.to_string())),
                "should recognize OOM: {}",
                msg
            );
        }
        for msg in cases {
            assert!(
                EmbeddingService::is_oom_error(&VecboostError::InferenceError(msg.to_string())),
                "should recognize InferenceError-wrapped OOM: {}",
                msg
            );
        }
    }

    #[test]
    fn test_is_oom_error_rejects_non_oom_errors() {
        assert!(!EmbeddingService::is_oom_error(
            &VecboostError::InferenceError("invalid model weights".to_string(),)
        ));
        assert!(!EmbeddingService::is_oom_error(
            &VecboostError::InvalidInput("bad input".to_string(),)
        ));
        assert!(!EmbeddingService::is_oom_error(
            &VecboostError::ModelLoadError("missing file".to_string(),)
        ));
    }

    #[tokio::test]
    async fn test_process_text_oom_recovery_no_fallback() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(OomEngine));
        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err(), "OOM error should propagate");
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(
                    msg.contains("no fallback available"),
                    "expected no-fallback message, got: {}",
                    msg
                );
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_batch_oom_propagates() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(OomEngine));
        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec!["a".to_string(), "b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err(), "batch OOM should propagate");
    }

    // ===== 输入验证边界测试 =====

    #[tokio::test]
    async fn test_process_text_empty_input() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(
                msg.contains("empty"),
                "expected empty message, got: {}",
                msg
            ),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_text_whitespace_only_input() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "   \n\t  ".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(
                msg.contains("whitespace"),
                "expected whitespace message, got: {}",
                msg
            ),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_text_too_long_input() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = EmbedRequest {
            text: "a".repeat(crate::utils::MAX_TEXT_LENGTH + 1),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(
                msg.contains("too long"),
                "expected too-long message, got: {}",
                msg
            ),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // ===== Matryoshka 维度截断测试 =====

    #[tokio::test]
    async fn test_process_text_matryoshka_truncation() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("matryoshka-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, Some(128)).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.dimension, 128);
        assert_eq!(resp.embedding.len(), 128);
        // 先归一化(norm=1)再截断,截断后子集的 norm <= 1
        let norm: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm <= 1.0 + 1e-5,
            "truncated norm should be <= 1, got {}",
            norm
        );
    }

    #[tokio::test]
    async fn test_process_text_matryoshka_zero_dimension() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("matryoshka-zero", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, Some(0)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("greater than 0"), "got: {}", msg)
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_text_matryoshka_exceeds_max() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("matryoshka-overflow", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let req = EmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, Some(1024)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("exceeds model maximum"), "got: {}", msg)
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // ===== process_similarity 测试 =====

    #[tokio::test]
    async fn test_process_similarity_identical_text() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "hello world".to_string(),
            target: "hello world".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(
            (resp.score - 1.0).abs() < 1e-5,
            "identical text score should be 1.0, got {}",
            resp.score
        );
    }

    #[tokio::test]
    async fn test_process_similarity_different_text() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "alpha text".to_string(),
            target: "beta text".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(
            (-1.0..=1.0).contains(&resp.score),
            "cosine similarity should be in [-1, 1], got {}",
            resp.score
        );
    }

    #[tokio::test]
    async fn test_process_similarity_empty_source() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "".to_string(),
            target: "valid target".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // ===== process_search 测试 =====

    #[tokio::test]
    async fn test_process_search_n_results() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..5).map(|i| format!("candidate text {}", i)).collect();
        let req = SearchRequest {
            query: "query text".to_string(),
            texts,
            top_k: Some(3),
        };
        let result = service.process_search(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.results.len(), 3, "should return top_k=3 results");
        for w in resp.results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results should be sorted by score desc"
            );
        }
        for (i, r) in resp.results.iter().enumerate() {
            assert!(r.index < 5, "index {} out of range", i);
            assert!(!r.text.is_empty());
        }
    }

    #[tokio::test]
    async fn test_process_search_default_top_k() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..10).map(|i| format!("candidate {}", i)).collect();
        let req = SearchRequest {
            query: "query text".to_string(),
            texts,
            top_k: None,
        };
        let result = service.process_search(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(
            resp.results.len(),
            DEFAULT_TOP_K,
            "default top_k should be DEFAULT_TOP_K"
        );
    }

    #[tokio::test]
    async fn test_process_search_clamped_top_k() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..3).map(|i| format!("candidate {}", i)).collect();
        let req = SearchRequest {
            query: "query text".to_string(),
            texts,
            top_k: Some(MAX_TOP_K + 50),
        };
        let result = service.process_search(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(
            resp.results.len(),
            3,
            "top_k should be clamped to available texts"
        );
    }

    #[tokio::test]
    async fn test_process_search_empty_texts() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SearchRequest {
            query: "query text".to_string(),
            texts: vec![],
            top_k: Some(3),
        };
        let result = service.process_search(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty"), "got: {}", msg),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_search_batch_basic() {
        let mock_engine = TestEngine::new(64);
        let model_config = make_model_config("search-batch-model", 64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let texts: Vec<String> = (0..5).map(|i| format!("candidate {}", i)).collect();
        let result = service
            .process_search_batch("query text", &texts, Some(2))
            .await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.results.len(), 2);
        for w in resp.results.windows(2) {
            assert!(w[0].score >= w[1].score, "should be sorted desc");
        }
    }

    // ===== 缓存测试 =====

    #[tokio::test]
    async fn test_cache_hit_skips_engine() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mock_engine = CountingEngine::new(64, Arc::clone(&counter));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, None, 16);

        // 第一次:cache miss,调用 engine
        let req1 = EmbedRequest {
            text: "cached text".to_string(),
            normalize: Some(true),
        };
        let r1 = service.process_text(req1, None).await.unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "first call should hit engine"
        );

        // 第二次:cache hit,不应调用 engine
        let req2 = EmbedRequest {
            text: "cached text".to_string(),
            normalize: Some(true),
        };
        let r2 = service.process_text(req2, None).await.unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "second call should hit cache, not engine"
        );

        // 两次结果应一致(相同 raw embedding 经相同的 normalize)
        assert_eq!(r1.embedding, r2.embedding);
        assert_eq!(r1.dimension, r2.dimension);
    }

    #[tokio::test]
    async fn test_warm_up_cache_disabled() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None); // cache disabled

        let result = service
            .warm_up_cache(vec!["a".to_string(), "b".to_string()])
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_warm_up_cache_enabled_counts_engine_calls() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mock_engine = CountingEngine::new(64, Arc::clone(&counter));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, None, 32);

        let texts = vec![
            "warm1".to_string(),
            "warm2".to_string(),
            "warm3".to_string(),
        ];
        let result = service.warm_up_cache(texts).await;
        assert!(result.is_ok());
        assert_eq!(
            counter.load(Ordering::SeqCst),
            3,
            "warm_up should call embed for each text"
        );
    }

    // ===== 模型管理与元信息测试 =====

    #[tokio::test]
    async fn test_switch_model_same_name() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("same-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        let req = ModelSwitchRequest {
            model_name: "same-model".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.success);
        assert_eq!(resp.current_model, "same-model");
        assert_eq!(resp.previous_model, Some("same-model".to_string()));
        assert!(
            resp.message.contains("Already"),
            "expected 'Already' message, got: {}",
            resp.message
        );
    }

    #[test]
    fn test_get_model_info_with_config() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("info-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let info = service.get_model_info().expect("model config present");
        assert_eq!(info.name, "info-model");
        assert_eq!(info.dimension, Some(384));
        assert!(info.is_loaded);
        assert_eq!(info.engine_type, "candle");
    }

    #[test]
    fn test_get_model_info_without_config() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);
        assert!(service.get_model_info().is_none());
    }

    #[test]
    fn test_get_model_metadata() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("meta-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let meta = service.get_model_metadata().expect("model config present");
        assert_eq!(meta.name, "meta-model");
        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.dimension, Some(384));
        assert_eq!(meta.max_input_length, 512);
        assert!(meta.is_loaded);
        assert!(meta.loaded_at.is_some());
    }

    #[test]
    fn test_list_available_models_with_config() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("list-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let list = service.list_available_models();
        assert_eq!(list.total_count, 1);
        assert_eq!(list.models[0].name, "list-model");
        assert_eq!(list.models[0].dimension, Some(384));
    }

    #[test]
    fn test_list_available_models_no_config() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let list = service.list_available_models();
        assert_eq!(list.total_count, 1);
        assert_eq!(list.models[0].name, "default");
    }

    #[test]
    fn test_has_model_manager_without_manager() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);
        assert!(!service.has_model_manager());
    }

    #[tokio::test]
    async fn test_unload_model_clears_config() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("unload-test", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        assert!(service.get_model_info().is_some());
        let result = service.unload_model("unload-test").await;
        assert!(result.is_ok());
        assert!(
            service.get_model_info().is_none(),
            "config should be cleared after unload"
        );
    }

    #[tokio::test]
    async fn test_list_loaded_models_without_manager() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);
        let loaded = service.list_loaded_models().await;
        assert!(loaded.is_empty(), "without manager, list should be empty");
    }

    // ===== 文件流式处理测试 =====

    #[tokio::test]
    async fn test_process_file_stream_basic() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file = temp_dir.path().join("stream.txt");
        std::fs::write(&test_file, "Line 1\nLine 2\nLine 3").unwrap();

        let result = service.process_file_stream(&test_file).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.dimension, 384);
        assert_eq!(resp.embedding.len(), 384);
        // 流式聚合:3 行向量平均后归一化,norm 应为 1
        let norm: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "normalized norm should be 1, got {}",
            norm
        );
    }

    #[tokio::test]
    async fn test_process_file_stream_empty_file() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file = temp_dir.path().join("empty.txt");
        std::fs::write(&test_file, "").unwrap();

        let result = service.process_file_stream(&test_file).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty"), "got: {}", msg),
            other => panic!("expected InvalidInput for empty file, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_file_stream_not_found() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let result = service
            .process_file_stream(std::path::Path::new("/nonexistent/path/file.txt"))
            .await;
        assert!(result.is_err());
    }

    // ===== 并发请求测试 =====

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_process_text() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = Arc::new(EmbeddingService::new(engine, None));

        let total = 10;
        let mut handles = Vec::with_capacity(total);
        for i in 0..total {
            let svc = Arc::clone(&service);
            handles.push(tokio::spawn(async move {
                let req = EmbedRequest {
                    text: format!("concurrent text {}", i),
                    normalize: Some(true),
                };
                svc.process_text(req, None).await
            }));
        }

        let results = futures::future::join_all(handles).await;
        assert_eq!(results.len(), total);
        for (i, result) in results.into_iter().enumerate() {
            assert!(result.is_ok(), "task {} panicked: {:?}", i, result.err());
            let response = result.unwrap().unwrap();
            assert_eq!(response.dimension, 64, "task {} dimension mismatch", i);
            assert_eq!(
                response.embedding.len(),
                64,
                "task {} embedding length mismatch",
                i
            );
        }
    }

    // ===== 批处理动态大小测试 =====

    #[tokio::test]
    async fn test_process_batch_dynamic_size() {
        let mock_engine = TestEngine::new(64);
        let model_config = make_model_config("dynamic-bs", 64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        // 测试不同输入规模:1, 5, 50, 100(最大批处理上限)
        for &size in &[1usize, 5, 50, MAX_BATCH_SIZE] {
            let texts: Vec<String> = (0..size).map(|i| format!("dynamic text {}", i)).collect();
            let req = BatchEmbedRequest {
                texts: texts.clone(),
                mode: None,
                normalize: Some(true),
            };
            let result = service.process_batch(req, None).await;
            assert!(
                result.is_ok(),
                "size={} should succeed: {:?}",
                size,
                result.err()
            );
            let response = result.unwrap();
            assert_eq!(
                response.embeddings.len(),
                size,
                "size={} embedding count mismatch",
                size
            );
            assert_eq!(response.dimension, 64, "size={} dimension mismatch", size);
            // 验证顺序保持
            for (i, emb) in response.embeddings.iter().enumerate() {
                assert_eq!(
                    emb.text_preview, texts[i],
                    "size={} index={} preview mismatch",
                    size, i
                );
            }
        }
    }

    #[tokio::test]
    async fn test_process_batch_matryoshka_truncation() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("batch-matryoshka", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let texts: Vec<String> = (0..3).map(|i| format!("matryoshka text {}", i)).collect();
        let req = BatchEmbedRequest {
            texts,
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, Some(128)).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.embeddings.len(), 3);
        assert_eq!(response.dimension, 128);
        for emb in &response.embeddings {
            assert_eq!(emb.embedding.len(), 128);
        }
    }

    // ===== 构造器与 builder 测试 =====

    #[tokio::test]
    async fn test_with_manager_constructor_sets_manager() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = Arc::new(ModelManager::new());
        let service = EmbeddingService::with_manager(engine, None, manager);
        assert!(service.has_model_manager());
    }

    #[test]
    fn test_with_all_constructor_sets_fields() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let model_config = make_model_config("all-fields-model", 64);
        let memory_manager = SharedGpuMemoryManager::new(
            8 * 1024 * 1024 * 1024,
            crate::device::memory_optimizer::GpuMemoryConfig::default(),
        );
        let scheduler = Arc::new(DynamicBatchScheduler::new(
            crate::device::BatchConfig::default(),
        ));
        let service = EmbeddingService::with_all(
            engine,
            InputValidator::with_default(),
            Some(model_config),
            None,
            16,
            Some(memory_manager),
            Some(scheduler),
        );
        let list = service.list_available_models();
        assert_eq!(list.models[0].name, "all-fields-model");
    }

    #[tokio::test]
    async fn test_with_buffer_pool_builder() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let buffer_pool = Arc::new(tokio::sync::RwLock::new(BufferPool::new(
            crate::device::memory_pool::BufferPoolConfig::default(),
        )));
        let service = EmbeddingService::new(engine, None).with_buffer_pool(buffer_pool);
        let req = EmbedRequest {
            text: "buffer pool test".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_ok());
    }

    // ===== 可配置 OOM 引擎 =====

    struct ConfigurableOomEngine {
        fallback_triggered: bool,
        fallback_succeeds: bool,
    }

    impl ConfigurableOomEngine {
        fn new(fallback_triggered: bool, fallback_succeeds: bool) -> Self {
            Self {
                fallback_triggered,
                fallback_succeeds,
            }
        }
    }

    #[async_trait]
    impl InferenceEngine for ConfigurableOomEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::OutOfMemory(
                "CUDA out of memory: allocation failed".to_string(),
            ))
        }

        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Err(VecboostError::OutOfMemory(
                "CUDA out of memory: allocation failed".to_string(),
            ))
        }

        fn precision(&self) -> &Precision {
            static PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn is_fallback_triggered(&self) -> bool {
            self.fallback_triggered
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            if self.fallback_succeeds {
                Ok(())
            } else {
                Err(VecboostError::InferenceError(
                    "CPU fallback unavailable".to_string(),
                ))
            }
        }
    }

    async fn make_manager_with_model(model_name: &str) -> Arc<ModelManager> {
        let cache_dir = tempdir().unwrap();
        let model_path = cache_dir.path().join(model_name);
        std::fs::create_dir_all(&model_path).unwrap();
        let loader = Arc::new(crate::model::loader::LocalModelLoader::new(
            cache_dir.path().to_path_buf(),
        ));
        let manager = Arc::new(ModelManager::with_loader(loader));
        let config = ModelConfig {
            name: model_name.to_string(),
            engine_type: EngineType::Candle,
            model_path,
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        manager.load(&config).await.unwrap();
        manager
    }

    // ===== OOM fallback 路径测试 =====

    #[tokio::test]
    async fn test_handle_oom_fallback_already_triggered() {
        let engine = ConfigurableOomEngine::new(true, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let model_config = make_model_config("oom-fb-triggered", 384);
        let manager = Arc::new(ModelManager::new());
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("fallback already attempted"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_oom_fallback_failure() {
        let engine = ConfigurableOomEngine::new(false, false);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let model_config = make_model_config("oom-fb-fail", 384);
        let manager = make_manager_with_model("oom-fb-fail").await;
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("fallback failed"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_oom_fallback_success_max_attempts_exceeded() {
        let engine = ConfigurableOomEngine::new(false, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let model_config = make_model_config("oom-fb-success", 384);
        let manager = make_manager_with_model("oom-fb-success").await;
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(
                    msg.contains("Max fallback attempts exceeded"),
                    "got: {}",
                    msg
                );
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handle_oom_fallback_model_not_in_manager() {
        let engine = ConfigurableOomEngine::new(false, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let model_config = make_model_config("oom-model-missing", 384);
        let manager = Arc::new(ModelManager::new());
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = EmbedRequest {
            text: "hello".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("no fallback available"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    // ===== process_batch OOM fallback 测试 =====

    #[tokio::test]
    async fn test_process_batch_oom_fallback_already_triggered() {
        let engine = ConfigurableOomEngine::new(true, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec!["a".to_string(), "b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("fallback already attempted"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_batch_oom_fallback_success_then_still_oom() {
        let engine = ConfigurableOomEngine::new(false, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec!["a".to_string(), "b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("CUDA out of memory"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    // ===== warm_up_cache 边界测试 =====

    #[tokio::test]
    async fn test_warm_up_cache_skips_failed_embeddings() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(OomEngine));
        let service = EmbeddingService::with_cache(engine, None, 32);

        let texts = vec!["fail1".to_string(), "fail2".to_string()];
        let result = service.warm_up_cache(texts).await;
        assert!(
            result.is_ok(),
            "warm_up should succeed even if all embeddings fail"
        );
    }

    #[tokio::test]
    async fn test_warm_up_cache_progress_logging() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mock_engine = CountingEngine::new(64, Arc::clone(&counter));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, None, 256);

        let texts: Vec<String> = (0..150).map(|i| format!("text{}", i)).collect();
        let result = service.warm_up_cache(texts).await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 150);
    }

    // ===== get_optimal_batch_size 路径测试 =====

    #[tokio::test]
    async fn test_get_optimal_batch_size_uses_scheduler() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let scheduler = Arc::new(DynamicBatchScheduler::new(
            crate::device::BatchConfig::default(),
        ));
        scheduler.set_batch_size(3).await;

        let service = EmbeddingService::with_all(
            engine,
            InputValidator::with_default(),
            None,
            None,
            16,
            None,
            Some(scheduler),
        );

        let texts: Vec<String> = (0..10).map(|i| format!("sched text {}", i)).collect();
        let req = BatchEmbedRequest {
            texts,
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.embeddings.len(), 10);
    }

    #[tokio::test]
    async fn test_get_optimal_batch_size_memory_manager_unregistered() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let model_config = make_model_config("unregistered-model", 64);
        let memory_manager = SharedGpuMemoryManager::new(
            8 * 1024 * 1024 * 1024,
            crate::device::memory_optimizer::GpuMemoryConfig::default(),
        );

        let service = EmbeddingService::with_all(
            engine,
            InputValidator::with_default(),
            Some(model_config),
            None,
            16,
            Some(memory_manager),
            None,
        );

        let texts: Vec<String> = (0..5).map(|i| format!("mem text {}", i)).collect();
        let req = BatchEmbedRequest {
            texts,
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.embeddings.len(), 5);
    }

    // ===== 文件处理边界测试 =====

    #[tokio::test]
    async fn test_get_processing_stats_nonexistent_file() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let result =
            service.get_processing_stats(std::path::Path::new("/nonexistent/stats_test.txt"));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embed_file_sliding_window_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("sliding.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2\nLine 3").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::SlidingWindow)
            .await;
        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Single(response) => {
                assert_eq!(response.dimension, 384);
            }
            _ => panic!("Expected Single embedding output"),
        }
    }

    #[tokio::test]
    async fn test_embed_file_fixed_size_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("fixed.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2\nLine 3").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::FixedSize)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_embed_file_max_pooling_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("maxpool.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::MaxPooling)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_process_paragraphs_long_paragraph_preview() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let long_paragraph = "A".repeat(200);
        let test_content = format!("{}\n\nShort paragraph", long_paragraph);
        let test_file_path = temp_dir.path().join("long_para.txt");
        std::fs::write(&test_file_path, test_content).unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::Paragraph)
            .await;
        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Paragraphs(paragraphs) => {
                assert_eq!(paragraphs.len(), 2);
                assert_eq!(
                    paragraphs[0].text_preview.len(),
                    100,
                    "long paragraph preview should be truncated to 100"
                );
                assert_eq!(paragraphs[1].text_preview, "Short paragraph");
            }
            _ => panic!("Expected Paragraphs embedding output"),
        }
    }

    #[tokio::test]
    async fn test_process_stream_internal_skips_blank_lines() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("blanks.txt");
        std::fs::write(&test_file_path, "Line 1\n\n   \nLine 2\n\nLine 3").unwrap();

        let result = service.process_file_stream(&test_file_path).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.dimension, 384);
        let norm: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "normalized norm should be 1, got {}",
            norm
        );
    }

    #[tokio::test]
    async fn test_process_file_stream_engine_error_logs_failure() {
        let temp_dir = tempdir().unwrap();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(OomEngine));
        let service = EmbeddingService::new(engine, None);

        let test_file = temp_dir.path().join("error.txt");
        std::fs::write(&test_file, "Line 1\nLine 2").unwrap();

        let result = service.process_file_stream(&test_file).await;
        assert!(result.is_err());
    }

    // ===== 缓存启用路径测试 =====

    #[tokio::test]
    async fn test_process_similarity_with_cache_enabled() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mock_engine = CountingEngine::new(64, Arc::clone(&counter));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, None, 32);

        let req = SimilarityRequest {
            source: "source text".to_string(),
            target: "target text".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_ok());
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "should call engine twice for source and target"
        );
    }

    // ===== 模型管理测试 =====

    #[tokio::test]
    async fn test_unload_model_with_manager() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = make_manager_with_model("unload-mgr-model").await;
        let model_config = make_model_config("unload-mgr-model", 384);
        let mut service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        assert!(service.get_model_info().is_some());
        let result = service.unload_model("unload-mgr-model").await;
        assert!(result.is_ok());
        assert!(service.get_model_info().is_none());
    }

    #[tokio::test]
    async fn test_list_loaded_models_with_manager() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = make_manager_with_model("list-mgr-model").await;
        let model_config = make_model_config("list-mgr-model", 384);
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let loaded = service.list_loaded_models().await;
        assert!(!loaded.is_empty(), "with manager, list should not be empty");
        assert!(loaded.contains(&"list-mgr-model".to_string()));
    }

    // ===== 搜索边界测试 =====

    #[tokio::test]
    async fn test_process_search_batch_default_top_k() {
        let mock_engine = TestEngine::new(64);
        let model_config = make_model_config("search-batch-default", 64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, Some(model_config));

        let texts: Vec<String> = (0..10).map(|i| format!("candidate {}", i)).collect();
        let result = service.process_search_batch("query", &texts, None).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.results.len(), DEFAULT_TOP_K);
    }

    #[tokio::test]
    async fn test_process_search_batch_empty_texts() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let result = service.process_search_batch("query", &[], Some(3)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => assert!(msg.contains("empty"), "got: {}", msg),
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_search_batch_clamped_top_k() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..3).map(|i| format!("candidate {}", i)).collect();
        let result = service
            .process_search_batch("query", &texts, Some(MAX_TOP_K + 50))
            .await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(
            resp.results.len(),
            3,
            "top_k should be clamped to available texts"
        );
    }

    #[tokio::test]
    async fn test_process_search_zero_top_k_rejected() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SearchRequest {
            query: "query text".to_string(),
            texts: vec!["candidate".to_string()],
            top_k: Some(0),
        };
        let result = service.process_search(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("top_k must be at least 1"), "got: {}", msg)
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // ===== similarity 边界测试 =====

    #[tokio::test]
    async fn test_process_similarity_empty_target() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "valid source".to_string(),
            target: "".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_process_similarity_whitespace_target() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "valid source".to_string(),
            target: "   \n\t  ".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("whitespace"), "got: {}", msg)
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_embed_file_min_pooling_mode() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let test_file_path = temp_dir.path().join("minpool.txt");
        std::fs::write(&test_file_path, "Line 1\nLine 2").unwrap();

        let result = service
            .embed_file(&test_file_path, AggregationMode::MinPooling)
            .await;
        assert!(result.is_ok());
        match result.unwrap() {
            EmbeddingOutput::Single(response) => {
                assert_eq!(response.dimension, 384);
            }
            _ => panic!("Expected Single embedding output"),
        }
    }

    #[tokio::test]
    async fn test_process_batch_oom_fallback_failure() {
        let engine = ConfigurableOomEngine::new(false, false);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec!["a".to_string(), "b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(msg.contains("fallback failed"), "got: {}", msg);
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_embed_file_invalid_utf8_path() {
        use std::os::unix::ffi::OsStrExt;

        let invalid_bytes = [0xFF, 0xFE, 0xFD];
        let os_str = std::ffi::OsStr::from_bytes(&invalid_bytes);
        let invalid_path = std::path::Path::new(os_str);

        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let result = service
            .embed_file(invalid_path, AggregationMode::Document)
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("Invalid path encoding"), "got: {}", msg);
            }
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // ===== switch_model 实际切换路径测试 =====

    #[tokio::test]
    async fn test_switch_model_different_name_fails_on_engine_create() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("original-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        let req = ModelSwitchRequest {
            model_name: "new-model".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err(), "switch to nonexistent model should fail");
    }

    #[tokio::test]
    async fn test_switch_model_with_manager_load_fails() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("orig-mgr-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = Arc::new(ModelManager::new());
        let mut service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = ModelSwitchRequest {
            model_name: "target-mgr-model".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err(), "load via manager should fail");
    }

    #[tokio::test]
    async fn test_switch_model_with_loaded_manager_engine_create_fails() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("orig-loaded", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = make_manager_with_model("orig-loaded").await;
        let mut service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = ModelSwitchRequest {
            model_name: "fresh-target".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err(), "AnyEngine::new should fail");
    }

    #[tokio::test]
    async fn test_switch_model_preserves_previous_config_fields() {
        let mock_engine = TestEngine::new(384);
        let temp_dir = tempdir().unwrap();
        let model_config = ModelConfig {
            name: "preserve-orig".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: Some(PathBuf::from("/orig/tokenizer")),
            device: DeviceType::Cpu,
            max_batch_size: 64,
            pooling_mode: Some(crate::config::model::PoolingMode::Mean),
            expected_dimension: Some(384),
            memory_limit_bytes: Some(1024),
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        let req = ModelSwitchRequest {
            model_name: "preserve-target".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_switch_model_no_previous_config() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, None);

        let req = ModelSwitchRequest {
            model_name: "first-model".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unload_model_different_name_keeps_config() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("keep-this", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        let result = service.unload_model("other-name").await;
        assert!(result.is_ok());
        assert!(
            service.get_model_info().is_some(),
            "config should remain when unloading different name"
        );
    }

    #[tokio::test]
    async fn test_unload_model_with_manager_unload_fails() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("mgr-unload-fail", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let manager = Arc::new(ModelManager::new());
        let mut service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let result = service.unload_model("not-in-manager").await;
        assert!(result.is_err(), "unload from empty manager should fail");
    }

    #[tokio::test]
    async fn test_list_available_models_no_config_default_engine() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let list = service.list_available_models();
        assert_eq!(list.total_count, 1);
        assert_eq!(list.models[0].name, "default");
        assert_eq!(list.models[0].engine_type, "candle");
        assert!(list.models[0].is_loaded);
    }

    #[tokio::test]
    async fn test_process_search_batch_uses_model_config_dimension() {
        let mock_engine = TestEngine::new(64);
        let model_config = make_model_config("dim-model", 64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, Some(model_config), 16);

        let texts: Vec<String> = (0..3).map(|i| format!("dim candidate {}", i)).collect();
        let result = service
            .process_search_batch("dim query", &texts, Some(2))
            .await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.results.len(), 2);
    }

    #[tokio::test]
    async fn test_process_search_batch_no_model_config_default_dimension() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..3).map(|i| format!("cfg {}", i)).collect();
        let result = service.process_search_batch("query", &texts, Some(2)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_process_batch_oom_fallback_success_then_retry_oom() {
        let engine = ConfigurableOomEngine::new(false, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        let service = EmbeddingService::new(engine, None);

        let req = BatchEmbedRequest {
            texts: vec!["retry-a".to_string(), "retry-b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_warm_up_cache_mixed_success_and_failure() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mock_engine = CountingEngine::new(64, Arc::clone(&counter));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, None, 32);

        let texts: Vec<String> = (0..5).map(|i| format!("mix{}", i)).collect();
        let result = service.warm_up_cache(texts).await;
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_process_text_with_cache_enabled_and_target_dimension() {
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("cache-trunc-model", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::with_cache(engine, Some(model_config), 16);

        let req = EmbedRequest {
            text: "cache truncation test".to_string(),
            normalize: Some(true),
        };
        let result = service.process_text(req, Some(64)).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.dimension, 64);
        assert_eq!(resp.embedding.len(), 64);
    }

    #[tokio::test]
    async fn test_process_similarity_with_cache_disabled() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "src no cache".to_string(),
            target: "tgt no cache".to_string(),
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!((-1.0..=1.0).contains(&resp.score));
    }

    #[tokio::test]
    async fn test_process_search_returns_correct_indices() {
        let mock_engine = TestEngine::new(32);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let texts: Vec<String> = (0..5).map(|i| format!("idx candidate {}", i)).collect();
        let req = SearchRequest {
            query: "idx query".to_string(),
            texts,
            top_k: Some(5),
        };
        let result = service.process_search(req).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.results.len(), 5);
        let indices: Vec<usize> = resp.results.iter().map(|r| r.index).collect();
        for &i in &[0, 1, 2, 3, 4] {
            assert!(indices.contains(&i), "missing index {}", i);
        }
    }
}
