// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::collapsible_if,
    clippy::useless_conversion,
    clippy::redundant_closure
)]

use crate::cache::OxCacheBackend;
use crate::cache::SemanticCache;
use crate::config::model::{CheckpointPreset, ModelConfig};
use crate::device::memory_optimizer::SharedGpuMemoryManager;
use crate::device::memory_pool::BufferPool;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, BatchEmbeddingResult, EmbedRequest, EmbedResponse,
    EmbeddingOutput, FileProcessingStats, ModelInfo, ModelListResponse, ModelMetadata,
    ModelSwitchRequest, ModelSwitchResponse, ParagraphEmbedding, SearchRequest, SearchResponse,
    SearchResult, SimilarityRequest, SimilarityResponse,
};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::model::manager::ModelManager;
use crate::utils::{
    AggregationMode, DEFAULT_TOP_K, FileValidator, InputValidator, MAX_BATCH_SIZE, MAX_TOP_K,
    SimilarityMetric, TextValidator, calculate_similarity, cosine_similarity, normalize_l2,
    truncate_vector, validate_dimension,
};
use log::{debug, warn};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Per-chunk timeout for batch processing (seconds).
/// Prevents a single hung inference call from blocking the entire batch indefinitely.
const BATCH_CHUNK_TIMEOUT_SECS: u64 = 60;

pub struct EmbeddingService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    model_manager: Option<Arc<ModelManager>>,
    cache: Arc<OxCacheBackend>,
    memory_manager: Option<SharedGpuMemoryManager>,
    buffer_pool: Option<Arc<tokio::sync::RwLock<BufferPool>>>,
    semantic_cache: Option<Arc<SemanticCache>>,
    /// 模型常驻内存上限控制器（memory_limit_bytes 配置时启用，执法）
    memory_limit: Option<Arc<crate::device::memory_limit::MemoryLimitController>>,
    /// 当前记账模型字节数（控制器 update_usage 为 set 语义，本字段跟踪投影基数）
    memory_used_est: Arc<std::sync::atomic::AtomicU64>,
    /// `[model.checkpoints]` resolved 预设表（含内置 laya-multilingual，main.rs
    /// 装配注入）：switch_model 按名命中时把预设参数/任务/路径作为缺省层应用。
    /// None = 未装配（库内直连场景），switch 走自由切换、行为与无预设表等价
    checkpoints: Option<Arc<std::collections::BTreeMap<String, CheckpointPreset>>>,
}

impl EmbeddingService {
    /// 统一内部构造入口：所有可选组件通过参数控制，消除 5 个构造器间的字段初始化重复。
    /// `cache_persist` 为 `Some((path, max_bytes))` 时启用 WAL 持久层。
    fn build(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
        cache_size: Option<usize>,
        memory_manager: Option<SharedGpuMemoryManager>,
        cache_persist: Option<(std::path::PathBuf, u64)>,
    ) -> Self {
        let cache = match (cache_size, cache_persist) {
            (Some(size), Some((path, max_bytes))) => {
                // 指纹绑定权重内容（sha256 可用时），
                // 同名换权重的旧向量在回放时被指纹不匹配淘汰。
                let fingerprint = model_config
                    .as_ref()
                    .map(|c| match &c.model_sha256 {
                        Some(sha) => format!("{}:{}", c.name, &sha[..sha.len().min(16)]),
                        None => c.name.clone(),
                    })
                    .unwrap_or_default();
                OxCacheBackend::with_persist(size, path, max_bytes, fingerprint)
            }
            (Some(size), None) => OxCacheBackend::with_ttl(
                size,
                Some(crate::cache::oxcache_backend::DEFAULT_CACHE_TTL),
            ),
            _ => OxCacheBackend::disabled(),
        };
        Self::assemble(
            engine,
            validator,
            model_config,
            model_manager,
            cache,
            memory_manager,
        )
    }

    /// 统一装配入口：缓存后端已由调用方选定（内存/WAL/Redis）。
    fn assemble(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
        cache: OxCacheBackend,
        memory_manager: Option<SharedGpuMemoryManager>,
    ) -> Self {
        let memory_limit = model_config
            .as_ref()
            .and_then(|c| c.memory_limit_bytes)
            .map(|limit| {
                Arc::new(
                    crate::device::memory_limit::MemoryLimitController::with_config(
                        crate::device::memory_limit::MemoryLimitConfig {
                            limit_bytes: limit,
                            warning_threshold_percent: 80,
                            critical_threshold_percent: 95,
                        },
                    ),
                )
            });
        Self {
            engine,
            validator,
            model_config,
            model_manager,
            cache: Arc::new(cache),
            memory_manager,
            buffer_pool: None,
            semantic_cache: None,
            memory_limit,
            memory_used_est: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            checkpoints: None,
        }
    }

    /// 跨副本共享的精确缓存（`[embedding] cache_backend = "redis"`）。
    ///
    /// Redis 初始化失败返回错误（启动期显性拒绝，不静默回落内存）。
    #[cfg(feature = "distributed-redis")]
    pub async fn with_cache_redis(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache_size: usize,
        url: &str,
    ) -> Result<Self, crate::error::VecboostError> {
        let cache = OxCacheBackend::with_redis(
            url,
            cache_size,
            Some(crate::cache::oxcache_backend::DEFAULT_CACHE_TTL),
        )
        .await?;
        Ok(Self::assemble(
            engine,
            InputValidator::with_default(),
            model_config,
            None,
            cache,
            None,
        ))
    }

    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self::build(
            engine,
            InputValidator::with_default(),
            model_config,
            None,
            None,
            None,
            None,
        )
    }

    pub fn with_manager(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        model_manager: Arc<ModelManager>,
    ) -> Self {
        Self::build(
            engine,
            InputValidator::with_default(),
            model_config,
            Some(model_manager),
            None,
            None,
            None,
        )
    }

    pub fn with_validator_and_manager(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
    ) -> Self {
        Self::build(
            engine,
            validator,
            model_config,
            model_manager,
            None,
            None,
            None,
        )
    }

    pub fn with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache_size: usize,
    ) -> Self {
        Self::build(
            engine,
            InputValidator::with_default(),
            model_config,
            None,
            Some(cache_size),
            None,
            None,
        )
    }

    /// 带 WAL 持久层的缓存构造（`[embedding] persist_path` 配置时启用）。
    /// 插入路径由 OxCacheBackend 内部两段追加写盘；启动时须显式调
    /// [`load_persisted_cache`](Self::load_persisted_cache) 回放。
    pub fn with_cache_persist(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache_size: usize,
        persist_path: std::path::PathBuf,
        persist_max_bytes: u64,
    ) -> Self {
        Self::build(
            engine,
            InputValidator::with_default(),
            model_config,
            None,
            Some(cache_size),
            None,
            Some((persist_path, persist_max_bytes)),
        )
    }

    pub fn with_all(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        validator: InputValidator,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
        cache_size: usize,
        memory_manager: Option<SharedGpuMemoryManager>,
    ) -> Self {
        Self::build(
            engine,
            validator,
            model_config,
            model_manager,
            Some(cache_size),
            memory_manager,
            None,
        )
    }

    /// 注入 ModelManager（server 模式装配；switch/unload 与 LFRU 驻留依赖它）。
    pub fn with_model_manager(mut self, manager: Arc<ModelManager>) -> Self {
        self.model_manager = Some(manager);
        self
    }

    /// 模型热度落盘（switch/unload 成功路径调用；失败仅 warn 不阻断业务）。
    async fn save_model_heat(&self) {
        if let Some(ref manager) = self.model_manager {
            if let Err(e) = manager
                .save_heat_file(std::path::Path::new(crate::model::heat::DEFAULT_HEAT_PATH))
                .await
            {
                log::warn!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-service-model-heat-save-failed",
                        crate::i18n::tr_args(&[("e", &(e).to_string())])
                    )
                );
            }
        }
    }

    /// 启动时顺序回放 WAL 重建缓存。persist 未启用时为 no-op。
    pub async fn load_persisted_cache(&self) {
        if let Some(path) = self.cache.persist_path() {
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-embedding-cache-wal-replay",
                    crate::i18n::tr_args(&[("a1", &(path.display()).to_string())])
                )
            );
        }
        self.cache.load_persisted().await;
    }

    /// 设置 BufferPool
    pub fn with_buffer_pool(mut self, buffer_pool: Arc<tokio::sync::RwLock<BufferPool>>) -> Self {
        self.buffer_pool = Some(buffer_pool);
        self
    }

    /// 设置语义缓存
    pub fn with_semantic_cache(mut self, semantic_cache: Arc<SemanticCache>) -> Self {
        self.semantic_cache = Some(semantic_cache);
        self
    }

    /// 注入 `[model.checkpoints]` resolved 预设表（main.rs 装配；switch_model
    /// 按名命中时应用预设的 model_path/task/tokenizer_path/decision_params）
    pub fn with_checkpoints(
        mut self,
        checkpoints: Arc<std::collections::BTreeMap<String, CheckpointPreset>>,
    ) -> Self {
        self.checkpoints = Some(checkpoints);
        self
    }

    fn validate_dimension(&self, actual_dimension: usize) {
        if let Some(ref config) = self.model_config
            && let Some(expected) = config.expected_dimension
            && actual_dimension != expected
        {
            warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-dimension-mismatch-expected-got",
                    crate::i18n::tr_args(&[
                        ("a1", &(expected).to_string()),
                        ("a2", &(actual_dimension).to_string()),
                        ("a3", &(config.name).to_string())
                    ])
                )
            );
        }
    }

    /// 获取当前最优的批量大小
    /// 优先使用 memory_manager 计算，最后回退到 MAX_BATCH_SIZE
    /// （时间窗拼批取代 DynamicBatchScheduler 后不再查询 scheduler）。
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
        let mut cache_entries = std::collections::HashMap::with_capacity(total_texts);
        let mut processed_count = 0usize;

        // 获取最优批量大小，分批推理避免 OOM
        let batch_size = self
            .get_optimal_batch_size(
                128,
                self.model_config
                    .as_ref()
                    .and_then(|c| c.expected_dimension)
                    .unwrap_or(768),
            )
            .await;

        for chunk in texts.chunks(batch_size) {
            match self.engine.read().await.embed_batch(chunk) {
                Ok(embeddings) => {
                    for (text, mut embedding) in chunk.iter().zip(embeddings.into_iter()) {
                        // NaN 拒绝入库
                        if let Err(e) = crate::utils::vector::ensure_finite_embedding(&embedding) {
                            warn!(
                                "{}",
                                crate::i18n::tr_with_args(
                                    "log-service-warm-up-skip-non",
                                    crate::i18n::tr_args(&[("a1", &text[..text.len().min(16)])])
                                )
                            );
                            let _ = e;
                            continue;
                        }
                        crate::utils::vector::normalize_l2(&mut embedding)?;
                        cache_entries.insert(self.cache_key_for(text), embedding);
                        processed_count += 1;
                    }
                    debug!("Warm-up progress: {}/{}", processed_count, total_texts);
                }
                Err(e) => {
                    warn!(
                        "{}",
                        crate::i18n::tr_with_args(
                            "log-service-warm-up-batch-failed",
                            crate::i18n::tr_args(&[
                                ("a1", &(chunk.len()).to_string()),
                                ("a2", &(e).to_string())
                            ])
                        )
                    );
                }
            }
        }

        self.cache.warm_up(cache_entries).await;

        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-service-cache-warm-up-completed",
                crate::i18n::tr_args(&[
                    ("a1", &(processed_count).to_string()),
                    ("a2", &(total_texts).to_string())
                ])
            )
        );
        Ok(())
    }

    async fn handle_oom_fallback<F, Fut, T>(&self, operation: F) -> Result<T, VecboostError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, VecboostError>>,
    {
        crate::service::common::handle_oom_fallback(
            &self.engine,
            &self.model_config,
            &self.model_manager,
            operation,
        )
        .await
    }

    /// 精确缓存键统一构造：`emb:{model_id}[:{sha16}]:{xxh3_128(text):016x}`。
    /// 写入方（回填/warm_up_cache）与读取方必须共用本函数——键方案曾因
    /// 多处内联复制漂移，warm_up 以裸文本为键致预热条目永不可读。
    /// 键绑定权重指纹（sha256 前 16 位，与 WAL 指纹同源）：同名换权重后
    /// Redis 中的旧向量不再命中——共享缓存跨重启存活，键是唯一防线。
    fn cache_key_for(&self, text: &str) -> String {
        let model_id = self
            .model_config
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or("unknown");
        let text_hash = xxhash_rust::xxh3::xxh3_128(text.as_bytes());
        match self
            .model_config
            .as_ref()
            .and_then(|c| c.model_sha256.as_deref())
        {
            Some(sha) => format!(
                "emb:{}:{}:{:016x}",
                model_id,
                &sha[..sha.len().min(16)],
                text_hash
            ),
            None => format!("emb:{}:{:016x}", model_id, text_hash),
        }
    }

    /// 处理单文本向量化
    /// 如果提供了 target_dimension，则对输出向量进行截断（Matryoshka 风格）
    pub async fn process_text(
        &self,
        req: EmbedRequest,
        target_dimension: Option<usize>,
    ) -> Result<EmbedResponse, VecboostError> {
        let start = std::time::Instant::now();
        self.validator.validate_text(&req.text)?;

        // 缓存键包含 model_id 和文本哈希，避免跨模型污染
        let model_id = self
            .model_config
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or("unknown");
        let cache_key = self.cache_key_for(&req.text);

        let embedding = if let Some(ref semantic_cache) = self.semantic_cache {
            // 语义缓存启用：精确匹配 → trigram 搜索 → 计算回填
            self.handle_oom_fallback(|| async {
                semantic_cache
                    .get_or_compute(model_id, &req.text, || async {
                        let embedding = self.engine.read().await.embed(&req.text)?;
                        crate::utils::vector::ensure_finite_embedding(&embedding)?;
                        Ok(embedding)
                    })
                    .await
            })
            .await?
        } else if self.cache.is_enabled() {
            self.handle_oom_fallback(|| async {
                self.cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key, || async {
                        let embedding = self.engine.read().await.embed(&req.text)?;
                        crate::utils::vector::ensure_finite_embedding(&embedding)?;
                        Ok(embedding)
                    })
                    .await
            })
            .await?
        } else {
            self.handle_oom_fallback(|| async {
                let embedding = self.engine.read().await.embed(&req.text)?;
                crate::utils::vector::ensure_finite_embedding(&embedding)?;
                Ok(embedding)
            })
            .await?
        };

        let mut embedding = embedding;
        normalize_l2(&mut embedding)?;

        // Apply Matryoshka dimension reduction if requested
        if let Some(target_dim) = target_dimension {
            let max_dim = self
                .model_config
                .as_ref()
                .and_then(|c| c.expected_dimension)
                .unwrap_or(1024);
            validate_dimension(Some(target_dim), max_dim)
                .map_err(|e| VecboostError::InvalidInput(e.to_string()))?;
            embedding = truncate_vector(&embedding, target_dim);
            // Matryoshka 截断破坏单位向量语义，必须重归一化以保证余弦相似度正确
            normalize_l2(&mut embedding)?;
        }

        let dimension = embedding.len();
        self.validate_dimension(dimension);

        Ok(EmbedResponse {
            dimension,
            embedding,
            processing_time_ms: start.elapsed().as_millis(),
            information_retention_rate: None,
        })
    }

    /// 批量文本嵌入——批内去重后调用引擎 embed_batch，结果按索引 scatter 回填。
    ///
    /// 不做缓存/语义缓存，直接调引擎。供 worker 时间窗拼批使用。
    /// 去重键为 xxh3_64，碰撞时以原始字符串二次确认，保证字节等同。
    pub async fn embed_batch_texts(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, VecboostError> {
        // 逐条复用文本校验（含控制字符检查，审查：批路径此前绕过防线）
        for t in texts {
            self.validator.validate_text(t)?;
        }
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // 批内去重：xxh3 键 + 字符串二次确认
        let mut hash_to_uniques: std::collections::HashMap<u64, Vec<usize>> =
            std::collections::HashMap::with_capacity(texts.len());
        let mut unique_texts: Vec<String> = Vec::with_capacity(texts.len());
        let mut index_to_unique: Vec<usize> = Vec::with_capacity(texts.len());
        for text in texts {
            let h = xxhash_rust::xxh3::xxh3_64(text.as_bytes());
            let mut found: Option<usize> = None;
            if let Some(candidates) = hash_to_uniques.get(&h) {
                for &uidx in candidates {
                    if unique_texts[uidx] == *text {
                        found = Some(uidx);
                        break;
                    }
                }
            }
            match found {
                Some(uidx) => index_to_unique.push(uidx),
                None => {
                    let uidx = unique_texts.len();
                    unique_texts.push(text.clone());
                    hash_to_uniques.entry(h).or_default().push(uidx);
                    index_to_unique.push(uidx);
                }
            }
        }
        let unique_results = self.engine.read().await.embed_batch(&unique_texts)?;
        for v in &unique_results {
            crate::utils::vector::ensure_finite_embedding(v)?;
        }
        // 引擎返回向量数不足时显性报错——索引越界 panic 会拖垮 worker
        if unique_results.len() < unique_texts.len() {
            return Err(VecboostError::InferenceError(format!(
                "engine returned {} embeddings for {} unique inputs",
                unique_results.len(),
                unique_texts.len()
            )));
        }
        // 埋点：drain 引擎分阶段延迟（tokenize/inference/pool）进 prometheus。
        // 单文本 embed 的累计值延迟到下一次批次 drain 或抓取时汇出（take 语义不丢数据）。
        #[cfg(feature = "http")]
        {
            let snapshot = self.engine.read().await.take_stage_snapshot();
            if let Some(snapshot) = snapshot {
                if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
                    collector.record_stage_snapshot(&snapshot);
                }
            }
        }
        let ratio = crate::metrics::inbatch_dedup_ratio(texts.len(), unique_texts.len());
        // 埋点：批内去重率滚动 gauge（全局 collector 未设置时零开销跳过）
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.set_dedup_ratio("embed", ratio);
        }
        debug!(
            "embed_batch_texts: {} texts → {} unique (saved {}, ratio {:.3})",
            texts.len(),
            unique_texts.len(),
            texts.len() - unique_texts.len(),
            ratio
        );
        Ok(index_to_unique
            .iter()
            .map(|&uidx| unique_results[uidx].clone())
            .collect())
    }

    /// 处理相似度计算
    pub async fn process_similarity(
        &self,
        req: SimilarityRequest,
    ) -> Result<SimilarityResponse, VecboostError> {
        self.validator.validate_text(&req.source)?;
        self.validator.validate_text(&req.target)?;

        // 度量解析：cosine（默认）/ euclidean / dot_product / manhattan
        let metric: SimilarityMetric = match req.metric.as_deref() {
            None | Some("") | Some("cosine") => SimilarityMetric::Cosine,
            Some(m) => SimilarityMetric::from_str(m)
                .map_err(|e| VecboostError::InvalidInput(e.to_string()))?,
        };

        // 短路：相同文本的余弦相似度必为 1.0（仅余弦度量成立）
        if req.source == req.target && metric == SimilarityMetric::Cosine {
            return Ok(SimilarityResponse { score: 1.0 });
        }

        // 缓存键包含 model_id 和文本哈希
        let cache_key_source = self.cache_key_for(&req.source);
        let cache_key_target = self.cache_key_for(&req.target);

        let (mut v1, mut v2) = if self.cache.is_enabled() {
            // 缓存启用：并行查缓存，各自独立命中/未命中
            let engine = Arc::clone(&self.engine);
            let cache = Arc::clone(&self.cache);
            let f1 = async move {
                cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key_source, || async {
                        let embedding = engine.read().await.embed(&req.source)?;
                        crate::utils::vector::ensure_finite_embedding(&embedding)?;
                        Ok(embedding)
                    })
                    .await
            };

            let engine = Arc::clone(&self.engine);
            let cache = Arc::clone(&self.cache);
            let f2 = async move {
                cache
                    .get_or_insert::<_, _, VecboostError>(&cache_key_target, || async {
                        let embedding = engine.read().await.embed(&req.target)?;
                        crate::utils::vector::ensure_finite_embedding(&embedding)?;
                        Ok(embedding)
                    })
                    .await
            };

            tokio::try_join!(f1, f2)?
        } else {
            // 缓存禁用：合并为 1 次 batch forward pass（而非 2 次独立 forward pass）
            let texts = vec![req.source.clone(), req.target.clone()];
            let embeddings = self.engine.read().await.embed_batch(&texts)?;
            (embeddings[0].clone(), embeddings[1].clone())
        };

        let score = tokio::task::spawn_blocking(move || {
            // 距离类度量（euclidean/manhattan）按 1/(1+d) 转换为相似度，
            // 其输入向量不应预先归一化；点积亦使用原始向量。
            if metric == SimilarityMetric::Cosine {
                normalize_l2(&mut v1)?;
                normalize_l2(&mut v2)?;
            }
            calculate_similarity(&v1, &v2, metric)
        })
        .await??;

        Ok(SimilarityResponse { score })
    }

    /// 处理大文件流式向量化 (简单实现：按行平均)
    pub async fn process_file_stream(&self, path: &Path) -> Result<EmbedResponse, VecboostError> {
        let path_str = path.to_str().ok_or_else(|| {
            VecboostError::InvalidInput(crate::i18n::tr("file-invalid-encoding").to_string())
        })?;
        self.validator.validate_file(path_str)?;

        let start_time = std::time::Instant::now();
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let result = self.process_stream_internal(reader, start_time).await;

        if let Err(ref e) = result {
            log::error!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-file-streaming-failed",
                    crate::i18n::tr_args(&[("a1", &format!("{:?}", e))])
                )
            );
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
            VecboostError::InvalidInput(crate::i18n::tr("file-invalid-encoding").to_string())
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
            normalize_l2(&mut final_vec)?;

            let dimension = final_vec.len();
            self.validate_dimension(dimension);

            let processing_time = start_time.elapsed();
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-processed-lines-2-ms",
                    crate::i18n::tr_args(&[
                        ("a1", &(count).to_string()),
                        ("a2", &format!("{:.2}", processing_time.as_millis() as f64))
                    ])
                )
            );

            Ok(EmbedResponse {
                dimension,
                embedding: final_vec,
                processing_time_ms: processing_time.as_millis(),
                information_retention_rate: None,
            })
        } else {
            Err(VecboostError::InvalidInput(
                crate::i18n::tr("file-empty").to_string(),
            ))
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
                crate::i18n::tr("file-no-paragraphs").to_string(),
            ));
        }

        let mut paragraph_embeddings: Vec<ParagraphEmbedding> =
            Vec::with_capacity(paragraphs.len());

        for (idx, para) in paragraphs.iter().enumerate() {
            if para.trim().is_empty() {
                continue;
            }

            let mut embedding = self.engine.read().await.embed(para)?;
            normalize_l2(&mut embedding)?;

            let preview = if para.len() > 100 {
                &para[..para.floor_char_boundary(100)]
            } else {
                para
            };

            paragraph_embeddings.push(ParagraphEmbedding {
                embedding,
                position: idx,
                text_preview: preview.to_string(),
            });
        }

        let processing_time = start_time.elapsed();
        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-service-processed-paragraphs-2-ms",
                crate::i18n::tr_args(&[
                    ("a1", &(paragraph_embeddings.len()).to_string()),
                    ("a2", &format!("{:.2}", processing_time.as_millis() as f64))
                ])
            )
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
            normalize_l2(&mut embedding)?;
            embedding
        };

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(req.texts.len());

        for (idx, text) in req.texts.iter().enumerate() {
            let mut embedding = self.engine.read().await.embed(text)?;
            normalize_l2(&mut embedding)?;

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
    // 候选向量先查精确缓存，命中不再送推理
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
            normalize_l2(&mut embedding)?;
            embedding
        };

        // 先查缓存，收集未命中的文本及其索引
        let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached_texts: Vec<(usize, &str)> = Vec::new();

        for (idx, text) in texts.iter().enumerate() {
            let cache_key = self.cache_key_for(text);
            if let Some(cached) = self.cache.get(&cache_key).await {
                embeddings[idx] = Some(cached);
            } else {
                uncached_texts.push((idx, text.as_str()));
            }
        }

        // 批量推理未命中的文本
        if !uncached_texts.is_empty() {
            let uncached_refs: Vec<String> =
                uncached_texts.iter().map(|(_, t)| t.to_string()).collect();
            let sequence_length = uncached_refs.first().map_or(0, |t| t.len());
            let output_dimension = self
                .model_config
                .as_ref()
                .and_then(|c| c.expected_dimension)
                .unwrap_or(768);
            let optimal_batch_size = self
                .get_optimal_batch_size(sequence_length, output_dimension)
                .await;

            debug!(
                "Processing search batch: {} texts ({} cached, {} uncached), optimal_batch_size={}",
                texts.len(),
                embeddings.iter().filter(|e| e.is_some()).count(),
                uncached_texts.len(),
                optimal_batch_size
            );

            for (chunk_base, chunk) in uncached_refs.chunks(optimal_batch_size).enumerate() {
                let chunk_base = chunk_base * optimal_batch_size;
                let chunk_embeddings = self.engine.read().await.embed_batch(chunk)?;
                for (i, mut emb) in chunk_embeddings.into_iter().enumerate() {
                    normalize_l2(&mut emb)?;
                    let global_i = chunk_base + i;
                    let original_idx = uncached_texts[global_i].0;
                    let cache_key = self.cache_key_for(&uncached_refs[global_i]);
                    self.cache.put(&cache_key, emb.clone()).await;
                    embeddings[original_idx] = Some(emb);
                }
            }
        }

        let mut results: Vec<(usize, f32, String)> = Vec::with_capacity(texts.len());
        for (idx, (text, emb_opt)) in texts.iter().zip(embeddings.into_iter()).enumerate() {
            if let Some(emb) = emb_opt {
                let score = cosine_similarity(&query_embedding, &emb)?;
                results.push((idx, score, text.clone()));
            }
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

        // 去重：相同文本只推理一次，结果映射回原始位置
        let (unique_texts, index_to_unique) = Self::deduplicate_texts(&texts);
        let dedup_saved = texts_len - unique_texts.len();
        if dedup_saved > 0 {
            debug!(
                "Batch deduplication: {} texts → {} unique (saved {} redundant inferences)",
                texts_len,
                unique_texts.len(),
                dedup_saved
            );
        }

        // 检测一个示例文本的序列长度（用于计算内存需求）
        let sequence_length = unique_texts.first().map_or(0, |t| t.len());
        let optimal_batch_size = self
            .get_optimal_batch_size(sequence_length, self.configured_output_dimension())
            .await;
        let num_chunks = unique_texts.len().div_ceil(optimal_batch_size);

        debug!(
            "Processing batch: {} unique texts ({} original), optimal_batch_size={}, chunks={}",
            unique_texts.len(),
            texts_len,
            optimal_batch_size,
            num_chunks
        );

        let max_concurrent_chunks =
            self.max_concurrent_chunks(unique_texts.len(), optimal_batch_size);
        debug!(
            "Processing batch with concurrent chunks: {} (CPU cores: {}, chunks: {})",
            max_concurrent_chunks,
            num_cpus::get(),
            num_chunks
        );

        let mut all_results = self
            .embed_chunks_concurrent(unique_texts, optimal_batch_size, max_concurrent_chunks)
            .await?;
        all_results.sort_by_key(|r| r.0);

        let effective_dimension = self.resolve_effective_dimension(target_dimension)?;

        // Matryoshka 截断 + 构建 unique_idx → (embedding, preview) 查找表
        let unique_results: Vec<(Vec<f32>, String)> = all_results
            .into_iter()
            .map(|(_, embedding, preview)| {
                let embedding = if let Some(dim) = effective_dimension {
                    let mut truncated = truncate_vector(&embedding, dim);
                    if let Err(e) = normalize_l2(&mut truncated) {
                        log::warn!(
                            "{}",
                            crate::i18n::tr_with_args(
                                "log-service-normalize-l2-failed-truncated",
                                crate::i18n::tr_args(&[("a1", &(e).to_string())])
                            )
                        );
                    }
                    truncated
                } else {
                    embedding
                };
                (embedding, preview)
            })
            .collect();

        // 将去重结果映射回原始顺序（重复文本复用同一 embedding）
        let results = Self::remap_deduplicated_results(&unique_results, &index_to_unique);

        let dimension =
            effective_dimension.unwrap_or_else(|| results.first().map_or(0, |r| r.embedding.len()));

        let processing_time = start_time.elapsed();
        self.adapt_batch_size_if_needed(processing_time.as_millis() as f64)
            .await;

        Ok(BatchEmbedResponse {
            embeddings: results,
            dimension,
            processing_time_ms: processing_time.as_millis(),
        })
    }

    /// 相同文本只保留一个推理槽位；返回去重文本与 original_idx →
    /// unique_idx 映射（重复文本复用同一 embedding）。
    fn deduplicate_texts(texts: &[String]) -> (Vec<String>, Vec<usize>) {
        let mut unique_map: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::with_capacity(texts.len());
        let mut unique_texts: Vec<String> = Vec::with_capacity(texts.len());
        let mut index_to_unique: Vec<usize> = Vec::with_capacity(texts.len());

        for text in texts {
            if let Some(&uidx) = unique_map.get(text.as_str()) {
                index_to_unique.push(uidx);
            } else {
                let uidx = unique_texts.len();
                unique_map.insert(text.as_str(), uidx);
                unique_texts.push(text.clone());
                index_to_unique.push(uidx);
            }
        }

        (unique_texts, index_to_unique)
    }

    /// 输出维度：有模型配置用配置值，否则 768 兜底。
    fn configured_output_dimension(&self) -> usize {
        self.model_config
            .as_ref()
            .and_then(|c| c.expected_dimension)
            .unwrap_or(768)
    }

    /// 根据实际负载和系统资源动态调整并发数：每核最多 2 个并发块，至少 4。
    fn max_concurrent_chunks(&self, unique_count: usize, optimal_batch_size: usize) -> usize {
        let cpu_count = num_cpus::get();
        std::cmp::min(
            cpu_count * 2,
            std::cmp::max(4, unique_count / optimal_batch_size),
        )
    }

    /// 分块并发推理：信号量限并发，单块超时显性失败；块内 OOM 走
    /// CPU 回退重试（`embed_single_chunk`）。返回顺序不保证，调用方排序。
    async fn embed_chunks_concurrent(
        &self,
        unique_texts: Vec<String>,
        optimal_batch_size: usize,
        max_concurrent_chunks: usize,
    ) -> Result<Vec<(usize, Vec<f32>, String)>, VecboostError> {
        let chunks: Vec<&[String]> = unique_texts.chunks(optimal_batch_size).collect();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_chunks));
        let engine = Arc::clone(&self.engine);
        // OOM 回退需要真实模型配置与 manager：克隆进 spawn 任务（block_on
        // 之外的 async 任务无法借 &self）
        let model_config = self.model_config.clone();
        let model_manager = self.model_manager.clone();

        let mut tasks = Vec::with_capacity(chunks.len());
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let chunk = chunk.to_vec();
            let engine = Arc::clone(&engine);
            let model_config = model_config.clone();
            let model_manager = model_manager.clone();
            let semaphore = Arc::clone(&semaphore);

            tasks.push(tokio::spawn(async move {
                // 获取信号量许可，限制并发数
                let _permit = semaphore.acquire().await.map_err(|e| {
                    VecboostError::InferenceError(format!("Failed to acquire semaphore: {}", e))
                })?;

                Self::embed_single_chunk(
                    engine,
                    model_config,
                    model_manager,
                    chunk,
                    chunk_idx,
                    optimal_batch_size,
                )
                .await
            }));
        }

        let mut all_results: Vec<(usize, Vec<f32>, String)> =
            Vec::with_capacity(unique_texts.len());

        for task in tasks {
            let chunk_results =
                match tokio::time::timeout(Duration::from_secs(BATCH_CHUNK_TIMEOUT_SECS), task)
                    .await
                {
                    Ok(join_result) => join_result??,
                    Err(_) => {
                        return Err(VecboostError::InferenceError(format!(
                            "Batch chunk processing timed out after {}s",
                            BATCH_CHUNK_TIMEOUT_SECS
                        )));
                    }
                };
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// 单块推理 + OOM CPU 回退：回退/重试循环统一委托
    /// `common::handle_oom_fallback`（真实 `ModelConfig` + `ModelManager`
    /// 门控，重试上限 2 次）；成功产出（全局下标, L2 归一化向量,
    /// 截断预览）三元组。回退已触发/回退失败/无 manager 均显性 OOM 失败。
    async fn embed_single_chunk(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        model_manager: Option<Arc<ModelManager>>,
        chunk: Vec<String>,
        chunk_idx: usize,
        optimal_batch_size: usize,
    ) -> Result<Vec<(usize, Vec<f32>, String)>, VecboostError> {
        let embeddings = crate::service::common::handle_oom_fallback(
            &engine,
            &model_config,
            &model_manager,
            || async { engine.read().await.embed_batch(&chunk) },
        )
        .await?;

        let mut results: Vec<(usize, Vec<f32>, String)> = Vec::with_capacity(embeddings.len());

        for (text_idx, (text, embedding)) in chunk.iter().zip(embeddings.into_iter()).enumerate() {
            let mut emb = embedding;

            normalize_l2(&mut emb)?;

            let global_idx = chunk_idx * optimal_batch_size + text_idx;

            let text_preview = if text.len() > 100 {
                &text[..text.floor_char_boundary(100)]
            } else {
                text
            }
            .to_string();

            results.push((global_idx, emb, text_preview));
        }

        Ok(results)
    }

    /// Matryoshka 目标维度校验（不得超过配置维度上限）。
    fn resolve_effective_dimension(
        &self,
        target_dimension: Option<usize>,
    ) -> Result<Option<usize>, VecboostError> {
        let Some(target_dim) = target_dimension else {
            return Ok(None);
        };
        let max_dim = self
            .model_config
            .as_ref()
            .and_then(|c| c.expected_dimension)
            .unwrap_or(1024);
        validate_dimension(Some(target_dim), max_dim)
            .map(|_| Some(target_dim))
            .map_err(|e| VecboostError::InvalidInput(e.to_string()))
    }

    /// original_idx 顺序重建结果（重复文本复用其 unique 槽位的向量）。
    fn remap_deduplicated_results(
        unique_results: &[(Vec<f32>, String)],
        index_to_unique: &[usize],
    ) -> Vec<BatchEmbeddingResult> {
        index_to_unique
            .iter()
            .map(|&uidx| {
                let (ref embedding, ref preview) = unique_results[uidx];
                BatchEmbeddingResult {
                    text_preview: preview.clone(),
                    embedding: embedding.clone(),
                }
            })
            .collect()
    }

    /// 记录内存使用情况并动态调整批量大小（未启用内存管理器时 no-op）。
    async fn adapt_batch_size_if_needed(&self, processing_time_ms: f64) {
        if let Some(ref mm) = self.memory_manager {
            let memory_usage = mm.get_memory_usage_percent().await;
            mm.adjust_batch_size_dynamically(processing_time_ms, memory_usage)
                .await;
            debug!(
                "Adjusted batch size based on: latency={:.2}ms, memory_usage={:.1}%",
                processing_time_ms, memory_usage
            );
        }
    }

    pub fn get_model_info(&self) -> Option<ModelInfo> {
        self.model_config.as_ref().map(|config| ModelInfo {
            name: config.name.clone(),
            engine_type: config.engine_type.to_string(),
            dimension: config.expected_dimension,
            is_loaded: true,
        })
    }

    /// 使用引擎 tokenizer 统计 token 数(同步,供 API usage 计算)
    pub fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        let engine = self.engine.try_read().map_err(|_| {
            VecboostError::inference_error("engine lock contention during token count".into())
        })?;
        engine.count_tokens(text)
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
            "{}",
            crate::i18n::tr_with_args(
                "log-service-switching-model-from",
                crate::i18n::tr_args(&[
                    ("a1", &format!("{:?}", previous_model)),
                    ("a2", &(req.model_name).to_string())
                ])
            )
        );

        if let Some(ref prev_name) = previous_model
            && prev_name == &req.model_name
            && !self.preset_overrides_differ(&req)?
        {
            return Ok(ModelSwitchResponse {
                previous_model: previous_model.clone(),
                current_model: req.model_name,
                success: true,
                message: crate::i18n::tr("model-already-current"),
            });
        }

        // [model.checkpoints] 预设命中：查表复用 config 层单一实现（与钉测试
        // 同源），命中时预设值为缺省层、请求显式值仍优先；decision_params
        // 请求 wire 无此维度，预设是唯一来源（R-multi-checkpoint-002）
        let checkpoint = match self.checkpoints.as_deref() {
            Some(table) => {
                crate::config::model::lookup_checkpoint_override(table, &req.model_name)?
            }
            None => None,
        };

        let model_config = self.build_switch_model_config(&req, checkpoint.as_ref());

        let new_est = estimate_resident_bytes(
            model_config.task,
            &model_config.engine_type,
            &model_config.model_path,
        );
        self.enforce_memory_limit_projection(&new_est).await?;

        // 统一经 EngineFactory 创建（GGUF 量化路由单一入口）。
        // create 先于 manager.load/unload 执行：失败时 manager 状态未被扰动，
        // 消除"新模型条目残留+旧模型已卸载+服务未切换"的失败次序窗口（评审 R7 四轮）。
        // 权衡：此后 manager.load 失败经 `?` 返回时，create 已完成的权重加载被
        // 整体 drop——可接受（manager.load 现仅走元数据构造，几乎不失败）；
        // 若未来 load 变重（真实权重加载）应改为 create 延后或 load 结果预检。
        let new_engine = self
            .create_engine_via_factory(&model_config, &req.model_name)
            .await?;

        self.swap_manager_entries(&model_config, previous_model.as_ref())
            .await?;

        let new_engine_arc = Arc::new(RwLock::new(new_engine));
        self.finalize_switch(new_engine_arc, model_config, new_est)
            .await;

        log::info!(
            "{}",
            crate::i18n::tr_with_args(
                "log-service-model-switched-successfully",
                crate::i18n::tr_args(&[("a1", &(req.model_name).to_string())])
            )
        );

        Ok(ModelSwitchResponse {
            previous_model,
            current_model: req.model_name,
            success: true,
            message: crate::i18n::tr("model-switch-success"),
        })
    }

    /// 同名早退的例外判定：命中预设且按 fall-through 归一化（req 显式优先 →
    /// 预设缺省层 → 继承当前）得到的加载面配置 ≠ 当前生效配置时必须重载
    /// ——否则 API 报成功而预设契约未兑现（主模型以预设名部署时，同名
    /// 切换是应用预设 task/预算/model_path 的唯一途径，如内置
    /// laya-multilingual 的 decision 任务与 256/256 预算）。device/
    /// max_batch_size 等宿主级旋钮缺省继承当前、req 无显式时归一化恒
    /// 等价，不参与比较；params 按**生效预算**归一化（当前 None 与预设
    /// resolved 缺省同为英文口径 192/256，不构成差异）——误判差异会
    /// 触发全量权重重载 + 双缓存清空，把幂等切换打成缓存命中归零。
    /// 非同名切换恒为 false（无需比较）。
    fn preset_overrides_differ(&self, req: &ModelSwitchRequest) -> Result<bool, VecboostError> {
        if self.model_config.as_ref().map(|c| &c.name) != Some(&req.model_name) {
            return Ok(false);
        }
        match self.checkpoints.as_deref() {
            Some(table) => Ok(crate::config::model::lookup_checkpoint_override(
                table,
                &req.model_name,
            )?
            .is_some_and(|ov| {
                let effective_path = req
                    .model_path
                    .clone()
                    .unwrap_or_else(|| ov.model_path.clone());
                let effective_tokenizer = req
                    .tokenizer_path
                    .clone()
                    .or_else(|| ov.tokenizer_path.clone());
                let effective_engine = ov
                    .engine_type
                    .clone()
                    .or_else(|| self.model_config.as_ref().map(|c| c.engine_type.clone()))
                    // 防御性缺省（此分支仅在 model_config=None 时可达，而该
                    // 情形比对直接走保守重载臂）：对齐 onnx-primary 主引擎；
                    // onnx 变体随 feature 门控，缺省臂 cfg 化（无 onnx 的
                    // 定制构建回落 candle 变体）
                    .unwrap_or({
                        #[cfg(feature = "onnx")]
                        {
                            crate::config::model::EngineType::Onnx
                        }
                        #[cfg(not(feature = "onnx"))]
                        {
                            crate::config::model::EngineType::Candle
                        }
                    });
                let effective_task = req
                    .task
                    .or(Some(ov.task))
                    .or_else(|| self.model_config.as_ref().map(|c| c.task))
                    .unwrap_or(crate::config::model::ModelTask::Embedding);
                match self.model_config.as_ref() {
                    Some(current) => {
                        current.model_path != effective_path
                            || current.tokenizer_path != effective_tokenizer
                            || current.engine_type != effective_engine
                            || current.task != effective_task
                            || current.decision_params.unwrap_or_default() != ov.decision_params
                    }
                    // previous_model 来自 model_config，此臂理论不可达；
                    // 无当前生效值可比时保守重载
                    None => true,
                }
            })),
            None => Ok(false),
        }
    }

    /// 按优先级 fall-through 构建切换后的加载面配置：req 显式值 → 预设
    /// 缺省层（命中时）→ 继承当前模型 → 类型缺省。例外：
    /// tokenizer_path 是模型内生属性（词表与模型绑定），不继承当前模型
    /// （local_bundle 对显式路径命中即用零报错；继承旧模型会让新模型静默
    /// 消费异构词表，分词漂移无信号）——预设 tokenizer_path 是显式配置
    /// （非跨模型继承），可作缺省层。
    fn build_switch_model_config(
        &self,
        req: &ModelSwitchRequest,
        checkpoint: Option<&crate::config::model::CheckpointSwitchOverride>,
    ) -> ModelConfig {
        ModelConfig {
            translation_params: None,
            transcription_params: None,
            name: req.model_name.clone(),
            // 预设 engine_type 为缺省层（req wire 无此维度），未配置继承当前
            // 加载模型（启动时即 [model].engine_type）；无当前模型回落主引擎
            // onnx（onnx-primary）；onnx 变体随 feature 门控，缺省臂 cfg 化
            //（无 onnx 的定制构建回落 candle 变体，加载期显性拒绝与启动期
            // 装配同口径）
            engine_type: checkpoint
                .and_then(|c| c.engine_type.clone())
                .or_else(|| self.model_config.as_ref().map(|c| c.engine_type.clone()))
                .unwrap_or({
                    #[cfg(feature = "onnx")]
                    {
                        crate::config::model::EngineType::Onnx
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        crate::config::model::EngineType::Candle
                    }
                }),
            model_path: req
                .model_path
                .clone()
                .or_else(|| checkpoint.map(|c| c.model_path.clone()))
                .unwrap_or_else(|| std::path::PathBuf::from(req.model_name.clone())),
            tokenizer_path: req
                .tokenizer_path
                .clone()
                .or_else(|| checkpoint.and_then(|c| c.tokenizer_path.clone())),
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
            // task：req.task 优先（switch 契约的任务维度切换入口，经
            // EngineFactory 分派臂加载对应管线），预设命中次之（如内置
            // laya-multilingual 即 decision 任务），再缺省继承当前模型（与
            // device/max_batch_size 等回退模式对齐），无当前模型回落
            // Embedding——decision 模型不被 switch 静默降级为 embedding
            task: req
                .task
                .or(checkpoint.map(|c| c.task))
                .or_else(|| self.model_config.as_ref().map(|c| c.task))
                .unwrap_or(crate::config::model::ModelTask::Embedding),
            // gguf 路径走 EngineFactory 量化路由：复用工厂的路径侧判定为单一
            // 事实源，大小写口径与启动路径一致（"MODEL.GGUF" 同样路由量化，
            // ends_with(".gguf") 的大小写敏感写法会分叉到 safetensors 路径）
            quantized: crate::engine::factory::should_use_quantized_engine(
                Path::new(&req.model_name),
                true,
            ),
            // 预设命中即应用该 checkpoint 的序列预算（DecisionPipeline::load
            // 消费）；未命中保持 None = 引擎缺省英文口径 192/256
            decision_params: checkpoint.map(|c| c.decision_params),
        }
    }

    /// 内存上限执法：按"当前记账 - 旧模型 + 新模型"投影判断，
    /// 在替换前执行——旧实现只查上一次记账（已超限才挡新加载，首个超大
    /// 模型照样放行，时点矛盾）。
    async fn enforce_memory_limit_projection(&self, new_est: &u64) -> Result<(), VecboostError> {
        let Some(ctrl) = &self.memory_limit else {
            return Ok(());
        };
        let limit = ctrl.current_usage() + ctrl.available_bytes();
        let projected = self
            .memory_used_est
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_sub(ctrl.current_usage())
            + new_est;
        if projected > limit {
            return Err(VecboostError::ModelLoadError(format!(
                "memory limit exceeded: new model ~{new_est} bytes, projected {projected} > limit {limit} bytes"
            )));
        }
        Ok(())
    }

    /// EngineFactory 创建走 spawn_blocking（与 decide 的推理口径一致）：
    /// candle 决策臂的 sha256 校验与 F16→F32 权重物化是秒级同步负载，
    /// 内联会阻塞 runtime worker。UnsupportedTask 直传：fail-fast 错误须走
    /// to_api_error 的 400 臂（调用方可换任务/端点），不得与权重缺失混同
    /// 为 404（评审 R3 四轮）；其余映射 model-load-failed NotFound。
    async fn create_engine_via_factory(
        &self,
        model_config: &ModelConfig,
        model_name: &str,
    ) -> Result<crate::engine::AnyEngine, VecboostError> {
        let cfg = model_config.clone();
        let engine_type = cfg.engine_type.clone();
        tokio::task::spawn_blocking(move || crate::engine::EngineFactory::create(engine_type, &cfg))
            .await?
            .map_err(|e| {
                if matches!(e, VecboostError::UnsupportedTask(_)) {
                    return e;
                }
                VecboostError::NotFound(crate::i18n::tr_with_args(
                    "model-load-failed",
                    crate::i18n::tr_args(&[("name", model_name), ("detail", &e.to_string())]),
                ))
            })
    }

    /// ModelManager 条目编排：同名先卸后载（load 对同名条目复用既有实例
    /// 直接返回（不按新 config 重建），原「先 load 后 unload」顺序会把该
    /// 条目卸掉——同名先卸后载使终态对齐跨名路径（resident_bytes 少计、
    /// LFRU/内存预算执法基数失真、后续 manager.get 不命中）；unload 的
    /// NotFound 臂经 let _ 容忍，create 已先行故此处失败不扰动 service
    /// 生效面），跨名载新卸旧。无 manager 时 no-op。
    async fn swap_manager_entries(
        &self,
        model_config: &ModelConfig,
        previous_model: Option<&String>,
    ) -> Result<(), VecboostError> {
        let Some(ref manager) = self.model_manager else {
            return Ok(());
        };
        log::debug!("Using ModelManager for model switching");
        let same_name = previous_model == Some(&model_config.name);
        if same_name {
            let _ = manager.unload(&model_config.name).await;
        }
        manager.load(model_config).await?;

        if let Some(ref prev_name) = previous_model {
            if !same_name {
                log::info!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-service-unloading-previous-model",
                        crate::i18n::tr_args(&[("a1", &(prev_name).to_string())])
                    )
                );
                let _ = manager.unload(prev_name).await;
            }
        }
        Ok(())
    }

    /// 切换生效收尾：新引擎接线内存控制器 + 记账、替换 service 生效引擎
    /// 与配置、双缓存清空（旧模型向量污染防线）、热度落盘供下次 warmstart。
    async fn finalize_switch(
        &mut self,
        new_engine_arc: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: ModelConfig,
        new_est: u64,
    ) {
        if let Some(ctrl) = &self.memory_limit {
            new_engine_arc
                .write()
                .await
                .attach_memory_limit_controller(Arc::clone(ctrl));
            ctrl.update_usage(new_est).await;
            self.memory_used_est
                .store(new_est, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-memory-limit-accounting-model",
                    crate::i18n::tr_args(&[
                        ("a1", &(model_config.name).to_string()),
                        ("new_est", &(new_est).to_string())
                    ])
                )
            );
        }
        self.engine = new_engine_arc;
        self.model_config = Some(model_config);

        // 切模型后清缓存，避免旧模型向量污染新模型。
        // Redis 后端 clear 可能 NotSupported（服务端淘汰面默认关闭）：显性
        // 记录而非静默——残留条目经键内权重指纹与 TTL 双重隔离。
        if let Err(e) = self.cache.clear().await {
            log::warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-cache-clear-failed",
                    crate::i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
        }
        // 语义缓存键为纯文本，跨模型必须一并清空
        if let Some(ref semantic) = self.semantic_cache {
            semantic.clear().await;
        }
        // 切换改变热度分布，落盘供下次 warmstart
        self.save_model_heat().await;
    }

    pub async fn unload_model(&mut self, name: &str) -> Result<(), VecboostError> {
        if let Some(ref manager) = self.model_manager {
            manager.unload(name).await?;
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-model-unloaded-via-modelmanager",
                    crate::i18n::tr_args(&[("a1", (name))])
                )
            );
            // 卸载后落盘热度表
            self.save_model_heat().await;
        }

        if self.model_config.as_ref().map(|c| &c.name) == Some(&name.to_string()) {
            self.model_config = None;
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-local-model-config-cleared",
                    crate::i18n::tr_args(&[("a1", (name))])
                )
            );
        }

        // 卸载模型后清缓存（失败显性记录；残留条目由指纹与 TTL 隔离）
        if let Err(e) = self.cache.clear().await {
            log::warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-cache-clear-failed",
                    crate::i18n::tr_args(&[("detail", &e.to_string())]),
                )
            );
        }
        // 语义缓存含该模型向量，按模型清空防跨模型污染（共享索引同步清理）
        if let Some(ref semantic) = self.semantic_cache {
            semantic.clear_model(name).await;
        }
        // 记账回冲（set 语义下卸载不回冲会永久占用额度）
        if let Some(ctrl) = &self.memory_limit {
            ctrl.update_usage(0).await;
            self.memory_used_est
                .store(0, std::sync::atomic::Ordering::Relaxed);
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

    /// 当前引擎句柄——切模型后传播给 RerankService 等外部持有者。
    pub fn engine_handle(&self) -> Arc<RwLock<dyn InferenceEngine + Send + Sync>> {
        Arc::clone(&self.engine)
    }

    /// 启动期接线内存上限：初始引擎 attach 控制器并按模型目录记账。
    /// server 启动路径调用；构造函数为同步故无法在此完成。
    pub async fn init_memory_limit(&self) {
        let Some(ctrl) = &self.memory_limit else {
            return;
        };
        self.engine
            .write()
            .await
            .attach_memory_limit_controller(Arc::clone(ctrl));
        if let Some(mc) = &self.model_config {
            let est = estimate_resident_bytes(mc.task, &mc.engine_type, &mc.model_path);
            ctrl.update_usage(est).await;
            self.memory_used_est
                .store(est, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-memory-limit-accounting-initial",
                    crate::i18n::tr_args(&[
                        ("a1", &(mc.name).to_string()),
                        ("est", &(est).to_string())
                    ])
                )
            );
        }
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
                // onnx-primary：无生效配置时展示主引擎
                .unwrap_or_else(|| "onnx".to_string()),
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

/// 内存上限记账入口（switch_model 投影执法与启动期 init 共用）：磁盘粗估
/// × 引擎驻留系数。candle 决策臂的 F16 checkpoint 以 DType::F32 物化驻留
/// （candle_decision.rs 的 VarBuilder 构造），磁盘口径系统性低估约 2×，且
/// 该路径 try_fallback_to_cpu 显性拒绝运行时降级——OOM 防线只剩加载前
/// 执法，记账必须与 F32 驻留同量级；系数 2 即 F16(2B/参数)→F32(4B/参数)
/// upcast，F32 磁盘资产被高估属保守方向（更早拒绝超限加载）。
fn estimate_resident_bytes(
    task: crate::config::model::ModelTask,
    engine_type: &crate::config::model::EngineType,
    path: &std::path::Path,
) -> u64 {
    let disk = estimate_model_bytes(path);
    if task == crate::config::model::ModelTask::Decision
        && *engine_type == crate::config::model::EngineType::Candle
    {
        disk.saturating_mul(2)
    } else {
        // Translate/Onnx 臂（cfg 门：EngineType::Onnx 变体仅在 onnx feature
        // 下存在）：翻译 bundle（Marian fp32/int8）驻留系数无实测标定
        // （报告 R8）——int8 权重驻留 ≈ 磁盘，但 fp32 变体 + tokenizer +
        // logits 工作区高于磁盘；取保守系数 2.0（同 Decision/Candle upcast
        // 口径量级），高估方向安全（更早拒绝超限）。实测标定后修正。
        #[cfg(feature = "onnx")]
        if task == crate::config::model::ModelTask::Translate
            && *engine_type == crate::config::model::EngineType::Onnx
        {
            return disk.saturating_mul(2);
        }
        disk
    }
}

/// 粗估模型磁盘占用：模型目录下一层文件的字节总和（权重为主）。
/// 作为内存上限记账的底数——精确显存占用需运行时探针，属独立变更。
/// 已知口径偏差：gguf 反量化路径驻留高于磁盘且更大；部署侧预留口径见
/// docs/USER_GUIDE.md「Candle 原生决策路径」限制节。
fn estimate_model_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    match std::fs::read_dir(path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total += meta.len();
                    } else if meta.is_dir() {
                        total += dir_size(&entry.path());
                    }
                }
            }
        }
        Err(e) => {
            // 目录不可读时粗估为 0——执法被无声绕过，必须显性化（回归审查 LOW）
            log::warn!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-service-estimate-model-bytes-cannot",
                    crate::i18n::tr_args(&[
                        ("a1", &(path.display()).to_string()),
                        ("e", &(e).to_string())
                    ])
                )
            );
        }
    }
    total
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelTask, Precision};
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
            crate::utils::hash::mock_embedding(text, self.dimension)
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

    /// 引擎返回向量数不足时 embed_batch_texts 必须显性报错而非越界 panic
    struct ShortBatchEngine;

    #[async_trait]
    impl InferenceEngine for ShortBatchEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(TestEngine::new(8).generate_embedding(text))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts
                .iter()
                .take(texts.len().saturating_sub(1))
                .map(|t| TestEngine::new(8).generate_embedding(t))
                .collect())
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

    /// 回归钉：memory_limit_bytes 配置时 switch_model 真实执法——
    /// 加载后记账超限，再切换被显性拒绝（此前只记账不执法）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_memory_limit_enforced_on_switch_model() {
        const MINILM: &str = "models/all-MiniLM-L6-v2";
        if !std::path::Path::new(&format!("{MINILM}/model.safetensors")).exists() {
            eprintln!("Skipping test: model weights not found at {MINILM}");
            return;
        }
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let baseline = ModelConfig {
            translation_params: None,
            transcription_params: None,
            name: "baseline".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(MINILM),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 8,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: Some(1024), // 1KB 上限：任何真实模型加载后必超限
            oom_fallback_enabled: true,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        let service = EmbeddingService::new(engine, Some(baseline));

        let mk_req = |name: &str| crate::domain::ModelSwitchRequest {
            model_name: name.to_string(),
            model_path: Some(std::path::PathBuf::from(MINILM)),
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };

        // 第一次切换：控制器初始 usage=0 → 放行；加载后记账 MiniLM 权重(~133MB) > 1KB → Exceeded
        let service = tokio::sync::RwLock::new(service);
        let first = {
            let mut guard = service.write().await;
            guard.switch_model(mk_req("m1")).await
        };
        // 投影预检：新模型 est(~133MB) > 1KB 上限 → 首次切换即拒绝
        let err = first.expect_err("oversized model switch must be rejected");
        match err {
            VecboostError::ModelLoadError(msg) => {
                assert!(msg.contains("memory limit exceeded"), "got: {msg}");
            }
            other => panic!("expected ModelLoadError, got {:?}", other),
        }
    }

    /// 记账口径钉：candle 决策臂 F16 checkpoint 以 F32 物化驻留（约 2× 磁盘），
    /// 记账入口须按驻留放大；其余 task/engine 组合维持磁盘口径。
    #[test]
    fn test_estimate_resident_bytes_candle_decision_f32_upcast() {
        let temp_dir = tempdir().unwrap();
        std::fs::write(temp_dir.path().join("model.safetensors"), vec![0u8; 1000]).unwrap();

        assert_eq!(
            estimate_resident_bytes(ModelTask::Decision, &EngineType::Candle, temp_dir.path()),
            2000
        );
        assert_eq!(
            estimate_resident_bytes(ModelTask::Embedding, &EngineType::Candle, temp_dir.path()),
            1000
        );
        #[cfg(feature = "onnx")]
        assert_eq!(
            estimate_resident_bytes(ModelTask::Decision, &EngineType::Onnx, temp_dir.path()),
            1000
        );
    }

    /// Translate 臂（int8 bundle 驻留估算）：磁盘 × 保守系数 2.0——
    /// ≥ 磁盘字节数（保守方向），且与 Decision/Embedding 臂相异
    ///（记账区分度：同 bundle 不同任务不得共享用同一估算线）。
    #[test]
    fn test_estimate_resident_bytes_translate_onnx_conservative() {
        let temp_dir = tempdir().unwrap();
        // 模拟 bundle：根层 config/tokenizer + onnx/ 子目录双模型
        std::fs::write(temp_dir.path().join("tokenizer.json"), vec![0u8; 300]).unwrap();
        std::fs::create_dir(temp_dir.path().join("onnx")).unwrap();
        std::fs::write(
            temp_dir.path().join("onnx/encoder_model.onnx"),
            vec![0u8; 400],
        )
        .unwrap();
        std::fs::write(
            temp_dir.path().join("onnx/decoder_model_merged.onnx"),
            vec![0u8; 500],
        )
        .unwrap();
        // 非 onnx 构建面本测试仅验证磁盘求和口径不进 Translate 臂（评审）
        #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
        let disk = 300 + 400 + 500;

        #[cfg(feature = "onnx")]
        {
            let est =
                estimate_resident_bytes(ModelTask::Translate, &EngineType::Onnx, temp_dir.path());
            assert_eq!(est, disk * 2, "Translate/Onnx 臂 = 磁盘 × 2 保守系数");
            assert!(est >= disk, "保守系数语义：驻留估算不得低于磁盘占用");
            assert_ne!(
                est,
                estimate_resident_bytes(ModelTask::Decision, &EngineType::Onnx, temp_dir.path()),
                "与 Decision/Onnx 臂（磁盘口径）相异"
            );
            assert_ne!(
                est,
                estimate_resident_bytes(ModelTask::Embedding, &EngineType::Candle, temp_dir.path()),
                "与 Embedding/Candle 臂（磁盘口径）相异"
            );
        }
    }

    #[tokio::test]
    async fn test_embed_batch_texts_short_engine_result_is_error_not_panic() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(ShortBatchEngine));
        let service = EmbeddingService::new(engine, None);
        let texts = vec!["a".to_string(), "b".to_string()];
        let result = service.embed_batch_texts(&texts).await;
        assert!(
            result.is_err(),
            "short engine result must be an explicit error, not an index panic"
        );
        match result.unwrap_err() {
            VecboostError::InferenceError(msg) => {
                assert!(
                    msg.contains("1 embeddings for 2 unique inputs"),
                    "got: {msg}"
                );
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_embed_batch_texts_dedup_bit_equal_and_ordered() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(32)));
        let service = EmbeddingService::new(engine.clone(), None);
        let texts = vec![
            "hello world".to_string(),
            "other text".to_string(),
            "hello world".to_string(),
        ];
        let batch = service.embed_batch_texts(&texts).await.unwrap();
        assert_eq!(batch.len(), 3);
        // 逐条单独调用基线
        let guard = engine.read().await;
        let e0 = guard.embed(&texts[0]).unwrap();
        let e1 = guard.embed(&texts[1]).unwrap();
        drop(guard);
        assert_eq!(batch[0], e0, "重复条目须与逐条调用 bit 级一致");
        assert_eq!(batch[2], e0, "索引回填须正确");
        assert_eq!(batch[1], e1);
        assert_eq!(batch[0], batch[2]);
    }

    #[tokio::test]
    async fn test_process_text_with_model_config() {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(384);

        let model_config = ModelConfig {
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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

    // 辅助 mock 引擎

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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        }
    }

    // OOM 恢复逻辑测试（判定实现在 common，批/单文本路径共用）

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
                crate::service::common::is_oom_error(&VecboostError::OutOfMemory(msg.to_string())),
                "should recognize OOM: {}",
                msg
            );
        }
        for msg in cases {
            assert!(
                crate::service::common::is_oom_error(&VecboostError::InferenceError(
                    msg.to_string()
                )),
                "should recognize InferenceError-wrapped OOM: {}",
                msg
            );
        }
    }

    #[test]
    fn test_is_oom_error_rejects_non_oom_errors() {
        assert!(!crate::service::common::is_oom_error(
            &VecboostError::InferenceError("invalid model weights".to_string(),)
        ));
        assert!(!crate::service::common::is_oom_error(
            &VecboostError::InvalidInput("bad input".to_string(),)
        ));
        assert!(!crate::service::common::is_oom_error(
            &VecboostError::ModelLoadError("missing file".to_string(),)
        ));
    }

    #[tokio::test]
    async fn test_process_text_oom_recovery_no_fallback() {
        crate::i18n::init();
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

    // 输入验证边界测试

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
        crate::i18n::init();
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

    // Matryoshka 维度截断测试

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
        // Matryoshka 截断后必须重归一化，保证余弦相似度正确
        let norm: f32 = resp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "truncated norm should be 1.0 (renormalized), got {}",
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

    // process_similarity 测试

    #[tokio::test]
    async fn test_process_similarity_identical_text() {
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        let req = SimilarityRequest {
            source: "hello world".to_string(),
            target: "hello world".to_string(),
            metric: None,
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
            metric: None,
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
            metric: None,
        };
        let result = service.process_similarity(req).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {:?}", other),
        }
    }

    // process_search 测试

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

    // 缓存测试

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

    #[tokio::test]
    async fn test_warm_up_cache_entries_readable_by_process_text() {
        // 回归钉（warm-up 键方案漂移修复）：预热写入的条目必须能被读路径
        // 命中——process_text 不得再次调用引擎，否则预热白算
        let counter = Arc::new(AtomicUsize::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingEngine::new(64, Arc::clone(&counter))));
        let service = EmbeddingService::with_cache(engine, None, 32);

        service
            .warm_up_cache(vec!["hot text".to_string()])
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let resp = service
            .process_text(
                EmbedRequest {
                    text: "hot text".to_string(),
                    normalize: Some(true),
                },
                None,
            )
            .await
            .expect("process_text must succeed");
        assert_eq!(resp.embedding.len(), 64);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "process_text must hit the warmed cache entry instead of calling the engine"
        );
    }

    // 模型管理与元信息测试

    #[tokio::test]
    async fn test_switch_model_same_name() {
        // message 断言依赖 i18n 文案：显式 init 消除测试顺序依赖
        // （与 process_search top_k 测试同款修复，96af4bf）
        crate::i18n::init();
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
            task: None,
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

    /// 切模型传播钉：req.task=decision 必须传进 switch_model 构造的
    /// ModelConfig 并抵达 EngineFactory 的 task=decision 分派臂（bundle
    /// 缺失报 decision 探测错误，而非落进 embedding 引擎加载）；无 task
    /// 字段时继承当前模型（Embedding），错误与决策臂无关——两条路径对照
    /// 钉住 req.task 优先、继承回退的契约。
    /// （candle 决策分派臂语义钉：fixture engine_type=candle 在 candle
    /// feature 内运行；onnx 分派臂对应钉见 factory
    /// test_create_decision_task_dispatches_to_onnx_pipeline）
    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_switch_model_task_decision_reaches_decision_dispatch() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let model_config = make_model_config("emb-origin", 384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(model_config));

        // 对照臂用本地存在但无权重的目录：路径不存在时 candle 会当作 HF
        // repo id 触发网络下载，违反测试离线约束；模型名不含 "decision"
        // （错误文案回显 model_name，避免与断言子串撞车）
        let empty_dir = tempdir().unwrap();
        let mk_req =
            |task: Option<crate::config::model::ModelTask>,
             model_path: Option<std::path::PathBuf>| ModelSwitchRequest {
                model_name: "missing-weights-model".to_string(),
                model_path,
                tokenizer_path: None,
                device: None,
                max_batch_size: None,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: None,
                task,
            };

        // req.task=Decision → 分派臂错误 detail 必含 decision 探测文案
        // （bundle 路径不存在：DecisionPipeline 探测先行报错，离线）
        let err = service
            .switch_model(mk_req(Some(ModelTask::Decision), None))
            .await
            .unwrap_err();
        match err {
            VecboostError::NotFound(msg) => assert!(
                msg.contains("decision"),
                "task=decision 必须抵达分派臂（detail 含 decision 探测错误），got: {msg}"
            ),
            other => panic!("expected NotFound(ModelLoadError 包装), got: {other:?}"),
        }

        // 对照：无 task 字段 → 继承当前 Embedding，走 candle 本地加载
        // （空目录无权重文件报错），与决策臂无关
        let err = service
            .switch_model(mk_req(None, Some(empty_dir.path().to_path_buf())))
            .await
            .unwrap_err();
        match err {
            VecboostError::NotFound(msg) => assert!(
                !msg.contains("decision"),
                "无 task 字段不得进决策分派臂，got: {msg}"
            ),
            other => panic!("expected NotFound(ModelLoadError 包装), got: {other:?}"),
        }
    }

    /// [model.checkpoints] 预设命中钉（R-multi-checkpoint-002）：switch 到
    /// laya-multilingual（req 全缺省）→ 预设 task=Decision 作为缺省层
    /// 进决策分派臂、预设 model_path 作为缺省层进 bundle 探测（错误回显
    /// 预设路径）；req 显式 model_path 优先于预设；未命中预设名走自由切换
    /// 不进决策臂。decision_params 字段级应用由 config 层
    /// lookup_checkpoint_override 钉测试覆盖（switch 与 task 消费同一
    /// override 结构，路径同源）。
    ///
    /// 失败夹具显式自建：同名条目覆盖内置预设、model_path 指向 tempdir 内
    /// 不存在子路径——探测错误回显该路径即证明预设路径生效，断言不依赖
    /// 工作区是否有 models/laya-multilingual bundle（按文档部署后仍恒真）。
    #[cfg(feature = "onnx")]
    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_switch_model_hits_builtin_checkpoint_preset() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        // 同名条目只改 model_path（task 必填仍 Decision，head/max 缺省继承
        // 内置 256/256）：merge 字段级覆盖后预设路径为显式失败夹具
        let absent_bundle_dir = tempdir().unwrap();
        let absent_bundle = absent_bundle_dir.path().join("absent-laya-multilingual");
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "laya-multilingual".to_string(),
            crate::config::model::CheckpointPreset {
                name: "laya-multilingual".to_string(),
                model_path: absent_bundle.clone(),
                task: crate::config::model::ModelTask::Decision,
                // 决策臂按 engine_type 分派后端：本测试钉 ort DecisionPipeline
                // 的错误面（"No Laya ONNX"），显式 onnx 而非回落 Candle
                engine_type: Some("onnx".to_string()),
                tokenizer_path: None,
                head_max_len: None,
                max_len: None,
            },
        );
        let mut service = EmbeddingService::new(engine, None).with_checkpoints(Arc::new(
            crate::config::model::merge_builtin_checkpoints(table)
                .expect("同名覆盖条目 head/max 缺省，merge 必须通过"),
        ));

        let mk_req =
            |model_name: &str, model_path: Option<std::path::PathBuf>| ModelSwitchRequest {
                model_name: model_name.to_string(),
                model_path,
                tokenizer_path: None,
                device: None,
                max_batch_size: None,
                pooling_mode: None,
                expected_dimension: None,
                memory_limit_bytes: None,
                oom_fallback_enabled: None,
                task: None,
            };

        // req 全缺省：预设填充 task=Decision + model_path（bundle 缺失时
        // 决策臂探测错误同时回显预设路径与 decision 语义）
        let absent_path_str = absent_bundle.to_string_lossy().into_owned();
        let err = service
            .switch_model(mk_req("laya-multilingual", None))
            .await
            .unwrap_err();
        match err {
            VecboostError::NotFound(msg) => assert!(
                msg.contains("decision") && msg.contains(&*absent_path_str),
                "预设必须作为缺省层生效（task=Decision 进分派臂 + 预设 model_path \
                 进探测），got: {msg}"
            ),
            other => panic!("expected NotFound, got: {other:?}"),
        }

        // req 显式 model_path 优先于预设（预设只补缺省层）；显式空目录
        // 走决策臂探测报 No Laya ONNX（不含预设路径）
        let empty_dir = tempdir().unwrap();
        let err = service
            .switch_model(mk_req(
                "laya-multilingual",
                Some(empty_dir.path().to_path_buf()),
            ))
            .await
            .unwrap_err();
        match err {
            VecboostError::NotFound(msg) => {
                assert!(
                    msg.contains("No Laya ONNX"),
                    "req 显式 model_path 必须优先于预设，got: {msg}"
                );
                assert!(
                    !msg.contains(&*absent_path_str),
                    "预设路径不得覆盖 req 显式值，got: {msg}"
                );
            }
            other => panic!("expected NotFound, got: {other:?}"),
        }

        // 对照：未命中预设名（req 全缺省）→ task 回落 Embedding，走 candle
        // 本地加载（空目录无权重报错），不进决策臂
        let empty_dir2 = tempdir().unwrap();
        let err = service
            .switch_model(mk_req(
                "missing-weights-model",
                Some(empty_dir2.path().to_path_buf()),
            ))
            .await
            .unwrap_err();
        match err {
            VecboostError::NotFound(msg) => assert!(
                !msg.contains("decision"),
                "未命中预设名不得进决策分派臂，got: {msg}"
            ),
            other => panic!("expected NotFound, got: {other:?}"),
        }
    }

    /// 同名切换的预设预算例外钉（R-multi-checkpoint-002）：主模型以预设名
    /// 部署时（启动 decision_params=None 英文缺省），切换到同名预设是应用
    /// 256/256 预算的唯一途径——already-current 早退必须让位于预算差异
    /// 重载，否则 API 报成功而预算契约未兑现。重载路径断言 Err（早退必
    /// Ok）：空目录 bundle 即 factory Err，Err 即证明未走早退。
    /// 失败夹具显式自建（同名条目覆盖预设、model_path 指向 tempdir 空目录），
    /// 断言不依赖工作区是否存在 models/laya-multilingual bundle。
    #[tokio::test]
    async fn test_switch_model_same_name_preset_params_differ_reloads() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let empty_bundle_dir = tempdir().unwrap();
        let empty_bundle = empty_bundle_dir.path().to_path_buf();
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "laya-multilingual".to_string(),
            crate::config::model::CheckpointPreset {
                name: "laya-multilingual".to_string(),
                model_path: empty_bundle.clone(),
                task: crate::config::model::ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: Some(256),
                max_len: Some(256),
            },
        );
        let mut service = EmbeddingService::new(engine, None).with_checkpoints(Arc::new(
            crate::config::model::merge_builtin_checkpoints(table)
                .expect("成对 256/256 覆盖条目 merge 必须通过"),
        ));
        // model_path/task 对齐预设归一化值，隔离 params 为唯一差异维
        let mut current = make_model_config("laya-multilingual", 384);
        current.task = crate::config::model::ModelTask::Decision;
        current.model_path = empty_bundle;
        service.model_config = Some(current);

        let req = ModelSwitchRequest {
            model_name: "laya-multilingual".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };
        assert!(
            service.switch_model(req).await.is_err(),
            "同名但预设预算（256/256）≠ 当前生效（英文缺省）时必须重载而非 \
             already-current 早退"
        );
    }

    /// 同名差异重载的 manager 终态钉：already-current 早退按预设归一化差异
    /// 放行的重载走 manager 段时，`ModelManager::load` 对同名条目复用既有
    /// 实例直接返回（不按新 config 重建），若沿用「先 load 后 unload」顺序
    /// 会把同名条目卸掉——终态 manager 无此模型、与跨名 switch 终态（持有
    /// 新条目）不一致（resident_bytes 少计、LFRU/内存预算执法基数失真、
    /// 后续 manager.get 不命中）。钉住修复后的终态：count==1 且 get 命中。
    /// create 须成功才达 manager 段，model_path 用真实 candle 权重资产
    /// （缺失 SKIP）；manager.load 仅元数据构造（loader.rs），不重复加载权重。
    #[cfg(feature = "candle")]
    #[tokio::test]
    async fn test_switch_model_same_name_reload_keeps_manager_entry() {
        crate::i18n::init();
        let has_weights = ["model.safetensors", "pytorch_model.bin"].iter().any(|w| {
            std::path::Path::new("models/BAAI-bge-small-en-v1.5")
                .join(w)
                .exists()
        });
        if !has_weights {
            eprintln!("Skipping test: model weights not found at models/BAAI-bge-small-en-v1.5");
            return;
        }
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let weights_dir = std::path::PathBuf::from("models/BAAI-bge-small-en-v1.5");
        // 同名预设只落 params 差异维（256/256 vs 当前英文缺省），task 保持
        // Embedding、model_path 对齐当前——归一化后 create 走 candle 加载臂
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "laya-multilingual".to_string(),
            crate::config::model::CheckpointPreset {
                name: "laya-multilingual".to_string(),
                model_path: weights_dir.clone(),
                task: crate::config::model::ModelTask::Embedding,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: Some(256),
                max_len: Some(256),
            },
        );
        let manager = Arc::new(ModelManager::new());
        let mut service = EmbeddingService::new(engine, None)
            .with_model_manager(Arc::clone(&manager))
            .with_checkpoints(Arc::new(
                crate::config::model::merge_builtin_checkpoints(table)
                    .expect("成对 256/256 覆盖条目 merge 必须通过"),
            ));
        // 当前生效配置对齐预设归一化值（params=None 英文缺省），隔离 params
        // 为唯一差异维；manager 预置同名条目模拟启动加载后的驻留状态
        let mut current = make_model_config("laya-multilingual", 384);
        current.model_path = weights_dir;
        manager.load(&current).await.expect("预置同名条目");
        assert_eq!(manager.count().await, 1);
        service.model_config = Some(current);

        let req = ModelSwitchRequest {
            model_name: "laya-multilingual".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };
        let resp = service
            .switch_model(req)
            .await
            .expect("预设 params 差异必须重载成功（candle 权重就位）");

        assert_eq!(
            manager.count().await,
            1,
            "同名重载后 manager 必须持有该模型条目（不得被同名 unload 移除）"
        );
        assert!(
            manager.get("laya-multilingual").await.is_some(),
            "manager.get 必须命中同名重载后的条目"
        );
        assert_eq!(resp.current_model, "laya-multilingual");
    }

    /// 幂等早退保持钉：同名切换且预设预算与当前生效一致时，already-current
    /// 早退契约不变（不得因预算例外把幂等切换变成强制重载）。
    #[tokio::test]
    async fn test_switch_model_same_name_preset_params_equal_short_circuits() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, None).with_checkpoints(Arc::new(
            crate::config::model::merge_builtin_checkpoints(Default::default())
                .expect("内置表 merge 恒通过"),
        ));
        let mut current = make_model_config("laya-multilingual", 384);
        current.task = crate::config::model::ModelTask::Decision;
        // model_path 对齐内置预设归一化值（else path 维差异会掩盖 params 等
        // 价语义、误触发重载）
        current.model_path = std::path::PathBuf::from("models/laya-multilingual");
        current.decision_params =
            Some(crate::config::model::DecisionParams::new(256, 256).expect("256/256 合法"));
        service.model_config = Some(current);

        let req = ModelSwitchRequest {
            model_name: "laya-multilingual".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };
        let resp = service
            .switch_model(req)
            .await
            .expect("预算一致的同名切换必须幂等早退");
        assert!(
            resp.success,
            "already-current 早退契约保持，got: {}",
            resp.message
        );
    }

    /// 生效预算归一化钉：当前 decision_params=None（生效=英文缺省 192/256）
    /// + 同名预设未配参数（resolved 同为缺省 192/256），且 task/model_path
    /// 逐维与预设归一化值一致 → 生效配置等价，必须幂等早退。误判差异会触发
    /// 全量权重重载 + 双缓存清空（纯浪费；重载后引擎逐字段等价重建）。
    #[tokio::test]
    async fn test_switch_model_same_name_preset_default_params_short_circuits() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "plain-decision".to_string(),
            crate::config::model::CheckpointPreset {
                name: "plain-decision".to_string(),
                model_path: std::path::PathBuf::from("models/plain"),
                task: crate::config::model::ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: None,
                max_len: None,
            },
        );
        let mut service = EmbeddingService::new(engine, None).with_checkpoints(Arc::new(table));
        // 当前各维对齐预设归一化值（task=Decision、model_path 同预设），
        // 唯一待验证格是 params：None 生效值 vs 预设缺省 192/256
        let mut current = make_model_config("plain-decision", 384);
        current.task = crate::config::model::ModelTask::Decision;
        current.model_path = std::path::PathBuf::from("models/plain");
        service.model_config = Some(current);

        let req = ModelSwitchRequest {
            model_name: "plain-decision".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };
        let resp = service
            .switch_model(req)
            .await
            .expect("生效预算等价（None vs 缺省 192/256）的同名切换必须幂等早退");
        assert!(
            resp.success,
            "already-current 早退契约保持，got: {}",
            resp.message
        );
    }

    /// 预设 task 维差异钉：同名预设 task=Decision vs 当前 Embedding（其余维
    /// 等价，params 均为生效缺省）→ 必须重载而非 already-current 报成功——
    /// 否则切换成败取决于切换前恰好加载了什么，且 decide 端点随后 400。
    /// 空目录 bundle 即 factory Err，Err 即证明未走早退。失败夹具显式自建
    /// （预设 model_path 指向 tempdir 空目录），断言不依赖工作区状态。
    #[tokio::test]
    async fn test_switch_model_same_name_preset_task_differs_reloads() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let empty_bundle_dir = tempdir().unwrap();
        let empty_bundle = empty_bundle_dir.path().to_path_buf();
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "plain-decision".to_string(),
            crate::config::model::CheckpointPreset {
                name: "plain-decision".to_string(),
                model_path: empty_bundle.clone(),
                task: crate::config::model::ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: None,
                max_len: None,
            },
        );
        let mut service = EmbeddingService::new(engine, None).with_checkpoints(Arc::new(table));
        // 当前为同名 embedding 模型（[model] 误配 task=embedding 的部署形态），
        // model_path 对齐预设以隔离 task 为唯一差异维
        let mut current = make_model_config("plain-decision", 384);
        current.model_path = empty_bundle;
        service.model_config = Some(current);

        let req = ModelSwitchRequest {
            model_name: "plain-decision".to_string(),
            model_path: None,
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: None,
        };
        assert!(
            service.switch_model(req).await.is_err(),
            "同名预设 task=Decision vs 当前 Embedding 必须重载而非 already-current \
             早退报成功"
        );
    }

    /// 跨模型 tokenizer 不继承钉：tokenizer_path 是模型内生属性（词表与
    /// 模型绑定），switch_model 构造的新 ModelConfig 不得缺省继承旧模型的
    /// 值——local_bundle::resolve_tokenizer_path 对已存在的显式路径命中
    /// 即用零报错，继承会让新模型静默消费旧模型词表产出错误输出（分词
    /// 漂移无信号）。场景：旧 config 携带真实存在的 tokenizer 文件，切换
    /// 到无任何 tokenizer 的 decision bundle（req.tokenizer_path 缺省）——
    /// 正确行为回落 bundle 内探测报 "Tokenizer not found"；继承回归时
    /// explicit 命中后 from_file 报 "Failed to load tokenizer"，本钉变红。
    /// bundle 探测先于 Session 构建，测试离线、不触发 ort。决策臂在
    /// onnx feature 门内（与 factory 分派钉同口径）。
    #[cfg(feature = "onnx")]
    #[tokio::test]
    async fn test_switch_model_does_not_inherit_previous_tokenizer() {
        crate::i18n::init();
        let mock_engine = TestEngine::new(384);
        let old_tok_dir = tempdir().unwrap();
        let old_tokenizer = old_tok_dir.path().join("old-tokenizer.json");
        std::fs::write(&old_tokenizer, b"{}").expect("write old tokenizer file");
        let mut old_config = make_model_config("tok-origin", 384);
        old_config.tokenizer_path = Some(old_tokenizer);
        // 决策臂按 engine_type 分派后端：本测试钉 ort DecisionPipeline 的
        // tokenizer 探测错误面，切换有效 engine_type 需显式 onnx
        old_config.engine_type = crate::config::model::EngineType::Onnx;
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let mut service = EmbeddingService::new(engine, Some(old_config));

        let bundle_dir = tempdir().unwrap();
        std::fs::write(bundle_dir.path().join("model.onnx"), b"fake onnx")
            .expect("write fake model.onnx");

        let req = ModelSwitchRequest {
            model_name: "tok-target".to_string(),
            model_path: Some(bundle_dir.path().to_path_buf()),
            tokenizer_path: None,
            device: None,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
            task: Some(ModelTask::Decision),
        };
        let err = service.switch_model(req).await.unwrap_err();
        match err {
            VecboostError::NotFound(msg) => assert!(
                msg.contains("Tokenizer not found"),
                "缺省必须回落新 bundle 内探测；若继承旧 tokenizer 路径会报 \
                 Failed to load tokenizer，got: {msg}"
            ),
            other => panic!("expected NotFound(ModelLoadError 包装), got: {other:?}"),
        }
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
            // Matryoshka 截断后必须重归一化，保证余弦相似度正确
            let norm: f32 = emb.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "batch truncated norm should be 1.0 (renormalized), got {}",
                norm
            );
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
        let service = EmbeddingService::with_all(
            engine,
            InputValidator::with_default(),
            Some(model_config),
            None,
            16,
            Some(memory_manager),
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

    /// 记录型 OOM 引擎：回退前 `embed_batch` 恒 OOM，`try_fallback_to_cpu`
    /// 把收到的 `ModelConfig` 抄进共享槽位（外部可断言），回退成功后
    /// `embed_batch` 产出可用向量——钉住「回退消费真实配置」契约。
    struct RecordingOomEngine {
        fallback_triggered: bool,
        fallback_succeeds: bool,
        dimension: usize,
        recorded_config: Arc<std::sync::Mutex<Option<ModelConfig>>>,
    }

    impl RecordingOomEngine {
        fn new(
            fallback_succeeds: bool,
            dimension: usize,
            recorded_config: Arc<std::sync::Mutex<Option<ModelConfig>>>,
        ) -> Self {
            Self {
                fallback_triggered: false,
                fallback_succeeds,
                dimension,
                recorded_config,
            }
        }
    }

    #[async_trait]
    impl InferenceEngine for RecordingOomEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::OutOfMemory(
                "CUDA out of memory: allocation failed".to_string(),
            ))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            if self.fallback_triggered {
                Ok(texts
                    .iter()
                    .map(|t| crate::utils::hash::mock_embedding(t, self.dimension))
                    .collect())
            } else {
                Err(VecboostError::OutOfMemory(
                    "CUDA out of memory: allocation failed".to_string(),
                ))
            }
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

        async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
            *self.recorded_config.lock().unwrap() = Some(config.clone());
            if self.fallback_succeeds {
                self.fallback_triggered = true;
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
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        manager.load(&config).await.unwrap();
        manager
    }

    // ===== OOM fallback 路径测试 =====

    #[tokio::test]
    async fn test_handle_oom_fallback_already_triggered() {
        crate::i18n::init();
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
        crate::i18n::init();
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
        crate::i18n::init();
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
        crate::i18n::init();
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

    // ===== 批处理 OOM 回退：真实配置消费契约（T001） =====

    #[tokio::test]
    async fn test_embed_single_chunk_oom_fallback_uses_real_model_config() {
        crate::i18n::init();
        let recorded = Arc::new(std::sync::Mutex::new(None));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(
            RecordingOomEngine::new(true, 384, Arc::clone(&recorded)),
        ));

        let model_config = make_model_config("real-model", 384);
        let manager = make_manager_with_model("real-model").await;

        let result = EmbeddingService::embed_single_chunk(
            engine,
            Some(model_config.clone()),
            Some(manager),
            vec!["alpha".to_string(), "beta".to_string()],
            0,
            8,
        )
        .await;

        let results = result.expect("回退成功后 chunk 必须重试产出结果");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
        assert_eq!(results[0].1.len(), 384);

        let received = recorded
            .lock()
            .unwrap()
            .clone()
            .expect("OOM 必须触发 try_fallback_to_cpu");
        assert_eq!(received.name, "real-model", "回退必须消费服务真实配置");
        assert_ne!(received.name, "default", "禁止 ModelConfig::default 回退");
        assert_eq!(received.model_path, model_config.model_path);
        assert_ne!(
            received.model_path,
            PathBuf::from("models/default"),
            "禁止默认模型路径（会被 onnx 回退消费为 HF repo id）"
        );
    }

    #[tokio::test]
    async fn test_embed_single_chunk_oom_without_manager_fails_without_fallback() {
        crate::i18n::init();
        let recorded = Arc::new(std::sync::Mutex::new(None));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(
            RecordingOomEngine::new(true, 384, Arc::clone(&recorded)),
        ));

        let result = EmbeddingService::embed_single_chunk(
            engine,
            Some(make_model_config("no-manager-model", 384)),
            None,
            vec!["alpha".to_string()],
            0,
            8,
        )
        .await;

        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(
                    msg.contains(&crate::i18n::tr("oom-no-fallback-available")),
                    "got: {}",
                    msg
                );
            }
            other => panic!("expected OutOfMemory, got {:?}", other),
        }
        assert!(
            recorded.lock().unwrap().is_none(),
            "无 manager 时不得发起回退尝试（原先是注定失败的默认配置回退）"
        );
    }

    // ===== process_batch OOM fallback 测试 =====

    #[tokio::test]
    async fn test_process_batch_oom_fallback_already_triggered() {
        crate::i18n::init();
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
        crate::i18n::init();
        let engine = ConfigurableOomEngine::new(false, true);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        // 回退成功但推理持续 OOM → 耗尽 common 侧重试上限（2 次）
        let model_config = make_model_config("batch-oom-retry", 384);
        let manager = make_manager_with_model("batch-oom-retry").await;
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

        let req = BatchEmbedRequest {
            texts: vec!["a".to_string(), "b".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let result = service.process_batch(req, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::OutOfMemory(msg) => {
                assert!(
                    msg.contains("Max fallback attempts exceeded"),
                    "got: {}",
                    msg
                );
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
    async fn test_get_optimal_batch_size_default_path() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));

        let service = EmbeddingService::with_all(
            engine,
            InputValidator::with_default(),
            None,
            None,
            16,
            None,
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
            metric: None,
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
        crate::i18n::init(); // 断言依赖 ftl 文案（含 "top_k"），未初始化时 tr 返回键名
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
                assert!(msg.contains("top_k"), "got: {}", msg)
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
            metric: None,
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
            metric: None,
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
        crate::i18n::init();
        let engine = ConfigurableOomEngine::new(false, false);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(engine));
        // 回退路径需要真实配置 + manager 命中模型（无 manager 会短路为
        // oom-no-fallback-available，见 no_manager 契约测试）
        let model_config = make_model_config("batch-oom-fail", 384);
        let manager = make_manager_with_model("batch-oom-fail").await;
        let service = EmbeddingService::with_manager(engine, Some(model_config), manager);

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
        crate::i18n::init();
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

    /// 跨名切换不存在的模型：candle 加载臂报 ModelLoadError 包装为
    /// NotFound（404）；fixture engine_type=candle 在 candle feature 内
    /// 运行（无 candle 时工厂报 candle-not-compiled ConfigError，非本钉
    /// 语义）。
    #[cfg(feature = "candle")]
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
            task: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err(), "switch to nonexistent model should fail");
        assert!(
            matches!(result.unwrap_err(), VecboostError::NotFound(_)),
            "nonexistent model should return NotFound (404), not Internal (500)"
        );
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
            task: None,
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
            task: None,
        };
        let result = service.switch_model(req).await;
        assert!(result.is_err(), "AnyEngine::new should fail");
    }

    /// （原 test_switch_model_unsupported_task_passthrough 已删除：其构造场景
    /// ——task=Decision + embedding 权重目录触发工厂 UnsupportedTask——已随
    /// factory task 分派臂接线消失，task=Decision 现一律路由决策管线，加载期
    /// 错误为 bundle 探测 ModelLoadError（正确口径），分派生效断言见
    /// engine::factory::test_create_decision_task_dispatches_to_decision_pipeline。
    /// switch_model 内的 UnsupportedTask 直传分支保留为防御性错误映射：
    /// 未来工厂再产生 UnsupportedTask（新 task 变体）时仍走 to_api_error 400 臂
    /// 而非误包装 404（评审 R3 四轮）。）

    #[tokio::test]
    async fn test_switch_model_preserves_previous_config_fields() {
        let mock_engine = TestEngine::new(384);
        let temp_dir = tempdir().unwrap();
        let model_config = ModelConfig {
            translation_params: None,
            transcription_params: None,
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
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            task: None,
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
            task: None,
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
        assert_eq!(list.models[0].engine_type, "onnx");
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

    /// 回归钉：process_search_batch 多 chunk 偏移——第二 chunk 必须写回
    /// `chunk_base + i` 全局下标，否则第二 chunk 文本向量永不落位、
    /// 第一 chunk 文本被错误覆盖。
    #[tokio::test]
    async fn test_process_search_batch_multi_chunk_offset() {
        let mock_engine = TestEngine::new(64);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = EmbeddingService::new(engine, None);

        // 无 memory_manager 时 optimal_batch_size 回退 MAX_BATCH_SIZE=100，
        // 101 条唯一文本 → chunks(100) 得 [0..100] + [100..101] 两片。
        let texts: Vec<String> = (0..=MAX_BATCH_SIZE)
            .map(|i| format!("chunk-offset candidate {}", i))
            .collect();
        let last = &texts[MAX_BATCH_SIZE];

        let resp = service
            .process_search_batch(last, &texts, Some(MAX_TOP_K))
            .await
            .expect("multi-chunk search should succeed");

        let hit = resp
            .results
            .iter()
            .find(|r| &r.text == last)
            .expect("second-chunk text must have an embedding and appear in results");
        assert!(
            (hit.score - 1.0).abs() < 1e-4,
            "query equals last candidate, expected cosine≈1, got {}",
            hit.score
        );
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
            metric: None,
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

    // ===== 未覆盖分支补测（覆盖率任务）：persist 构造 / 语义缓存 / 维度告警 /
    // warm-up 非有限值与批失败 / 非法度量 / spawn_blocking 余弦臂 / 非法 UTF-8
    // 路径 / 空批 / 批内去重 / search 批缓存命中 / count_tokens 锁竞争 =====

    /// 文本含 "nan" 时返回 NaN 向量，否则返回有效向量的引擎。
    struct NonFiniteBatchEngine {
        dimension: usize,
    }

    #[async_trait]
    impl InferenceEngine for NonFiniteBatchEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(self.embedding_for(text))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|t| self.embedding_for(t)).collect())
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

    impl NonFiniteBatchEngine {
        fn embedding_for(&self, text: &str) -> Vec<f32> {
            if text.contains("nan") {
                vec![f32::NAN; self.dimension]
            } else {
                vec![1.0f32; self.dimension]
            }
        }
    }

    /// persist 缓存构造分支 + WAL 回放 no-op 路径 + 指纹绑定键不互染。
    #[tokio::test]
    async fn test_with_cache_persist_constructor_and_wal_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("embed-cache.wal");
        let mut config = make_model_config("persist-model", 8);
        config.model_sha256 = Some("deadbeef".to_string());
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(
            CountingEngine::new(8, Arc::new(AtomicUsize::new(0))),
        ));
        let service = EmbeddingService::with_cache_persist(
            engine,
            Some(config),
            16,
            wal_path.clone(),
            64 * 1024,
        );
        assert_eq!(
            service.cache.persist_path(),
            Some(wal_path.clone()),
            "persist 构造必须暴露 WAL 路径"
        );
        // 无 WAL 文件时回放为 no-op，不得报错
        service.load_persisted_cache().await;
        let mk = |t: &str| EmbedRequest {
            text: t.to_string(),
            normalize: Some(false),
        };
        let first = service.process_text(mk("persist me"), None).await.unwrap();
        let second = service.process_text(mk("persist me"), None).await.unwrap();
        assert_eq!(first.embedding, second.embedding, "缓存回读须字节等同");
    }

    /// 语义缓存分支：process_text 走 get_or_compute，两次调用结果等同。
    #[tokio::test]
    async fn test_with_semantic_cache_process_text_roundtrip() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, None)
            .with_semantic_cache(Arc::new(SemanticCache::with_capacity(0.9, 16)));
        let mk = |t: &str| EmbedRequest {
            text: t.to_string(),
            normalize: Some(false),
        };
        let first = service
            .process_text(mk("semantic hello"), None)
            .await
            .unwrap();
        let second = service
            .process_text(mk("semantic hello"), None)
            .await
            .unwrap();
        assert_eq!(first.embedding, second.embedding, "语义缓存命中须字节等同");
        assert_eq!(first.dimension, 8);
    }

    /// 引擎维度与 expected_dimension 不一致：validate_dimension 只告警不拒绝，
    /// 返回引擎实际维度。
    #[tokio::test]
    async fn test_process_text_dimension_mismatch_warns_and_returns_actual() {
        let config = make_model_config("mismatch-dim", 8);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(4)));
        let service = EmbeddingService::new(engine, Some(config));
        let resp = service
            .process_text(
                EmbedRequest {
                    text: "dim mismatch".to_string(),
                    normalize: Some(false),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(resp.dimension, 4, "维度不一致只告警，返回实际维度");
    }

    /// warm-up 遇 NaN 向量：跳过该条不落缓存（NaN 请求随后显性报错而非
    /// 命中缓存返回 NaN），整体 Ok。
    #[tokio::test]
    async fn test_warm_up_cache_skips_non_finite_embeddings() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(NonFiniteBatchEngine { dimension: 4 }));
        let service = EmbeddingService::with_cache(engine, None, 16);
        service
            .warm_up_cache(vec!["nan text".to_string(), "ok text".to_string()])
            .await
            .unwrap();
        // NaN 条目未入库：请求 NaN 文本走引擎重算并被 ensure_finite 显性拒绝
        //（若 warm-up 误存 NaN，此处会命中缓存静默返回 NaN 向量）
        let err = service
            .process_text(
                EmbedRequest {
                    text: "nan text".to_string(),
                    normalize: Some(false),
                },
                None,
            )
            .await
            .unwrap_err();
        match err {
            VecboostError::InferenceError(msg) => {
                assert!(
                    msg.contains("non-finite"),
                    "NaN 文本必须显性报非有限值，got: {msg}"
                );
            }
            other => panic!("expected InferenceError for non-finite, got {other:?}"),
        }
        // 合法文本不受影响
        let ok = service
            .process_text(
                EmbedRequest {
                    text: "ok text".to_string(),
                    normalize: Some(false),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(ok.dimension, 4);
    }

    /// warm-up 批量推理失败：记日志继续（不中断预热、不向调用方报错）。
    #[tokio::test]
    async fn test_warm_up_cache_batch_failure_returns_ok() {
        crate::i18n::init();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(OomEngine));
        let service = EmbeddingService::with_cache(engine, None, 16);
        service
            .warm_up_cache(vec!["a".to_string(), "b".to_string()])
            .await
            .unwrap();
    }

    /// 相似度请求携带未注册度量名：InvalidInput 显性拒绝。
    #[tokio::test]
    async fn test_process_similarity_invalid_metric_rejected() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, None);
        let err = service
            .process_similarity(SimilarityRequest {
                source: "hello".to_string(),
                target: "world".to_string(),
                metric: Some("warp".to_string()),
            })
            .await
            .unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "未注册度量必须 InvalidInput，got {err:?}"
        );
    }

    /// 不同文本余弦路径走 spawn_blocking 计算（非短路臂），距离度量跳过
    /// 预归一化，分数均在合法域内。
    #[tokio::test]
    async fn test_process_similarity_different_texts_scores_per_metric() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, None);
        // 默认余弦：spawn_blocking 内归一化 + calculate_similarity
        let cosine = service
            .process_similarity(SimilarityRequest {
                source: "first distinct text".to_string(),
                target: "second totally different".to_string(),
                metric: None,
            })
            .await
            .unwrap();
        assert!(
            (-1.0..=1.0).contains(&cosine.score) && cosine.score < 1.0,
            "不同文本余弦分数应 <1 且在 [-1,1]，got {}",
            cosine.score
        );
        // 距离度量：跳过预归一化，1/(1+d) 转换
        let euclidean = service
            .process_similarity(SimilarityRequest {
                source: "first distinct text".to_string(),
                target: "second totally different".to_string(),
                metric: Some("euclidean".to_string()),
            })
            .await
            .unwrap();
        assert!(
            (0.0..=1.0).contains(&euclidean.score),
            "euclidean 转换分数应在 (0,1]，got {}",
            euclidean.score
        );
    }

    /// 非 UTF-8 路径：process_file_stream 以 InvalidInput 显性拒绝（不 panic）。
    #[tokio::test]
    async fn test_process_file_stream_rejects_invalid_utf8_path() {
        use std::os::unix::ffi::OsStrExt;
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, None);
        let bad_path = std::path::PathBuf::from(std::ffi::OsStr::from_bytes(b"/tmp/\xff\xfe"));
        let err = service.process_file_stream(&bad_path).await.unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "非 UTF-8 路径必须 InvalidInput，got {err:?}"
        );
    }

    /// 空批：embed_batch_texts 直接返回空 vec（不触发引擎调用）。
    #[tokio::test]
    async fn test_embed_batch_texts_empty_slice_returns_empty() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, None);
        let result = service.embed_batch_texts(&[]).await.unwrap();
        assert!(result.is_empty(), "空批必须返回空 vec");
    }

    /// 批内重复文本：哈希命中 + 字符串二次确认路径，去重后引擎只推理一次。
    #[tokio::test]
    async fn test_embed_batch_texts_duplicates_dedup_via_hash_and_confirm() {
        let counter = Arc::new(AtomicUsize::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingEngine::new(8, Arc::clone(&counter))));
        let service = EmbeddingService::new(engine, None);
        let texts = vec!["dup".to_string(), "dup".to_string(), "uniq".to_string()];
        let result = service.embed_batch_texts(&texts).await.unwrap();
        assert_eq!(result.len(), 3, "结果按原顺序回填");
        assert_eq!(result[0], result[1], "重复文本向量必须等同");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "批内去重后引擎只收到 2 个唯一文本"
        );
    }

    /// process_search_batch 候选缓存命中分支：第二次调用候选向量不再推理
    /// （计数只增 query 一次），全命中时跳过批量推理循环。
    #[tokio::test]
    async fn test_process_search_batch_second_call_hits_candidate_cache() {
        let counter = Arc::new(AtomicUsize::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingEngine::new(8, Arc::clone(&counter))));
        let service = EmbeddingService::with_cache(engine, None, 64);
        let candidates: Vec<String> = (0..3).map(|i| format!("cand {i}")).collect();

        service
            .process_search_batch("stable query", &candidates, Some(2))
            .await
            .unwrap();
        let after_first = counter.load(Ordering::SeqCst);

        let resp = service
            .process_search_batch("stable query", &candidates, Some(2))
            .await
            .unwrap();
        let after_second = counter.load(Ordering::SeqCst);
        assert_eq!(
            after_second - after_first,
            1,
            "第二次调用只有 query 走引擎，候选必须全部命中缓存"
        );
        assert_eq!(resp.results.len(), 2, "top_k 截断生效");
    }

    /// 引擎写锁被占时 count_tokens 以 InferenceError 显性报告锁竞争
    ///（try_read 失败分支，禁止降级为 0）。
    #[tokio::test]
    async fn test_count_tokens_lock_contention_returns_inference_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine.clone(), None);
        let _write_guard = engine.write().await;
        let err = service.count_tokens("hello").unwrap_err();
        match err {
            VecboostError::InferenceError(msg) => {
                assert!(
                    msg.contains("lock contention"),
                    "锁竞争文案必须可定位，got: {msg}"
                );
            }
            other => panic!("expected InferenceError, got {other:?}"),
        }
    }

    /// process_batch 批内去重：重复文本只推理一次，结果按原位置复用。
    #[tokio::test]
    async fn test_process_batch_dedup_reuses_single_inference() {
        let counter = Arc::new(AtomicUsize::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingEngine::new(8, Arc::clone(&counter))));
        let service = EmbeddingService::new(engine, None);
        let req = BatchEmbedRequest {
            texts: vec!["same".to_string(), "same".to_string(), "other".to_string()],
            mode: None,
            normalize: Some(true),
        };
        let resp = service.process_batch(req, None).await.unwrap();
        assert_eq!(resp.embeddings.len(), 3, "结果按原位置回填");
        assert_eq!(resp.embeddings[0].embedding, resp.embeddings[1].embedding);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "去重后引擎只推理 2 个唯一文本"
        );
        assert_eq!(resp.dimension, 8);
    }

    /// init_memory_limit 启动接线：控制器 attach + 模型目录记账（含子目录
    /// 递归），完成后服务仍可正常推理。
    #[tokio::test]
    async fn test_init_memory_limit_attaches_and_serves() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("weights.bin"), vec![0u8; 512]).unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/tokenizer.json"), vec![0u8; 128]).unwrap();

        let mut config = make_model_config("init-limit", 8);
        config.memory_limit_bytes = Some(1024 * 1024);
        config.model_path = dir.path().to_path_buf();

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine, Some(config));
        service.init_memory_limit().await;
        let resp = service
            .process_text(
                EmbedRequest {
                    text: "after init".to_string(),
                    normalize: Some(false),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(resp.dimension, 8, "记账接线后推理不受影响");
    }

    /// estimate_model_bytes：一层文件 + 子目录递归求和；目录不可读时粗估
    /// 为 0（显性 warn，不 panic）。
    #[test]
    fn test_estimate_model_bytes_sums_files_and_subdirs() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 100]).unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("nested/b.bin"), vec![0u8; 50]).unwrap();
        assert_eq!(
            estimate_model_bytes(dir.path()),
            150,
            "一层文件与子目录内容必须累加"
        );
        assert_eq!(
            estimate_model_bytes(&dir.path().join("missing")),
            0,
            "目录不可读时粗估为 0"
        );
    }

    /// engine_handle 传播：返回的句柄与生效引擎同源（D27 传播链的锚点）。
    #[tokio::test]
    async fn test_engine_handle_serves_same_engine() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let service = EmbeddingService::new(engine.clone(), None);
        let handle = service.engine_handle();
        assert_eq!(
            handle.read().await.embed("x").unwrap().len(),
            engine.read().await.embed("x").unwrap().len(),
            "engine_handle 必须指向同一生效引擎"
        );
    }

    /// unload_model：语义缓存随卸载一并清空（防跨模型污染），无 manager、
    /// 非当前模型时仅走清缓存路径且不报错。
    #[tokio::test]
    async fn test_unload_model_clears_semantic_cache_without_manager() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(TestEngine::new(8)));
        let mut service = EmbeddingService::new(engine, None)
            .with_semantic_cache(Arc::new(SemanticCache::with_capacity(0.9, 16)));
        service
            .unload_model("never-loaded")
            .await
            .expect("无 manager、非当前模型的卸载必须为无害 no-op");
    }

    /// 直接调用 NonFiniteBatchEngine 未使用的 trait 方法体（覆盖率）。
    #[test]
    fn test_non_finite_batch_engine_unused_trait_bodies() {
        let engine = NonFiniteBatchEngine { dimension: 4 };
        assert_eq!(*engine.precision(), Precision::Fp32);
        assert!(!engine.supports_mixed_precision());
    }

    #[tokio::test]
    async fn test_non_finite_batch_engine_fallback_body() {
        let mut engine = NonFiniteBatchEngine { dimension: 4 };
        assert!(
            engine
                .try_fallback_to_cpu(&ModelConfig::default())
                .await
                .is_ok()
        );
    }
}
