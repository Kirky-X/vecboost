// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Rerank 服务实现

use crate::cache::OxCacheBackend;
use crate::config::model::ModelConfig;
use crate::domain::{RerankRequest, RerankResponse, RerankResult};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::service::common;
use crate::utils::InputValidator;
use crate::utils::validator::input::TextValidator;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// Rerank 服务：引擎 + 输入校验 + 语义缓存聚合。
pub struct RerankService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    cache: Arc<OxCacheBackend>,
    /// 调用方隔离模式（默认 Off = 缓存键行为不变）。
    isolation: crate::cache::CallerIsolation,
}

impl RerankService {
    /// 统一内部构造入口，消除多个构造器间的字段初始化重复。
    fn build(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            cache,
            isolation: crate::cache::CallerIsolation::Off,
        }
    }

    /// 设置调用方隔离模式（builder）。默认 Off。
    pub fn with_caller_isolation(mut self, isolation: crate::cache::CallerIsolation) -> Self {
        self.isolation = isolation;
        self
    }

    /// 当前调用方隔离模式（装配断言/诊断用：生产构造点必须按
    /// `[semantic_cache] caller_isolation` 注入，见 `main.rs::build_rerank_service`）。
    pub fn caller_isolation(&self) -> crate::cache::CallerIsolation {
        self.isolation
    }

    /// 替换底层引擎——EmbeddingService::switch_model 成功后由 API 层传播，
    /// 修复切模型后 rerank 仍用旧引擎的失联。
    pub fn replace_engine(&mut self, new_engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>) {
        self.engine = new_engine;
    }

    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self::build(engine, model_config, Arc::new(OxCacheBackend::disabled()))
    }

    /// 缓存启用的生产构造（`[rerank] cache_enabled=true` 装配消费点）。
    ///
    /// 进程内精确缓存，与语义缓存 L1 装配先例同构
    /// （`SemanticCache::with_capacity` 内建 `OxCacheBackend::new`）：纯内存、
    /// 无 WAL/Redis——RerankConfig 无持久化字段，不发明第二配置来源。
    /// 装配侧必须以 `cache_enabled && cache_size > 0` 门控后才调用本构造
    ///（见 `main.rs::build_rerank_service`）；`cache_is_enabled()` 供装配断言。
    pub fn with_cache_capacity(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        capacity: usize,
    ) -> Self {
        Self::build(
            engine,
            model_config,
            Arc::new(OxCacheBackend::new(capacity)),
        )
    }

    /// 由 `[rerank]` 配置构造生产服务：内部单点消费
    /// `cache_enabled && cache_size > 0`，true 按容量构建进程内精确缓存、
    /// false 保持 disabled（默认零变化）。装配侧不再重复该条件判断；
    /// `cache_is_enabled()` 供日志与诊断查询构造结果。
    pub fn from_rerank_config(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        config: &crate::RerankConfig,
    ) -> Self {
        if config.cache_enabled && config.cache_size > 0 {
            Self::with_cache_capacity(engine, model_config, config.cache_size)
        } else {
            Self::new(engine, model_config)
        }
    }

    /// rerank 精确缓存是否启用（装配断言/诊断用）。
    pub fn cache_is_enabled(&self) -> bool {
        self.cache.is_enabled()
    }

    #[allow(private_interfaces)]
    pub fn with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
    ) -> Self {
        Self::build(engine, model_config, cache)
    }

    /// 请求校验链：query 非空/长度上限、documents 非空/数量上限/逐条
    /// 文本防线（此前 rerank 文档面无长度/控制字符防线）、
    /// top_k 有效性（0 无意义，应拒绝）。
    fn validate_rerank_request(
        &self,
        req: &RerankRequest,
        max_documents: usize,
        max_query_length: usize,
    ) -> Result<(), VecboostError> {
        self.validator.validate_text(&req.query)?;

        if req.query.len() > max_query_length {
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "rerank-query-too-long",
                crate::i18n::tr_args(&[
                    ("length", &req.query.len().to_string()),
                    ("max", &max_query_length.to_string()),
                ]),
            )));
        }

        // 验证 documents 非空
        if req.documents.is_empty() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("rerank-empty-docs").to_string(),
            ));
        }

        // 验证 documents 数量不超过限制
        if req.documents.len() > max_documents {
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "rerank-too-many-docs",
                crate::i18n::tr_args(&[
                    ("count", &req.documents.len().to_string()),
                    ("max", &max_documents.to_string()),
                ]),
            )));
        }

        // 逐条校验 documents（此前 rerank 文档面无长度/控制字符防线）
        for doc in &req.documents {
            self.validator.validate_text(doc)?;
        }

        // 验证 top_k 有效性（0 无意义，应拒绝）
        if let Some(top_k) = req.top_k
            && top_k == 0
        {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("rerank-invalid-top-k").to_string(),
            ));
        }

        Ok(())
    }

    /// 分数计算：缓存启用时逐文档查缓存（键 = model + query/doc 各自哈希，
    /// 裸拼接 ("a","b:c") 与 ("a:b","c") 同键互染），未命中批量
    /// 走引擎并回填；禁用时整批直通。OOM 降级经 handle_oom_fallback 包装。
    async fn compute_rerank_scores(
        &self,
        query: &str,
        documents: &[String],
        caller_id: &str,
    ) -> Result<Vec<f32>, VecboostError> {
        let engine = self.engine.clone();
        let query = query.to_string();
        let documents_vec = documents.to_vec();
        if !self.cache.is_enabled() {
            // 缓存禁用：直接批量调用引擎。caller_prefix 只在缓存路径使用，
            // 早退前不计算，避免禁用态白付 SHA-256 与 String 分配。
            return common::handle_oom_fallback(&engine, &self.model_config, &None, || {
                let engine = engine.clone();
                let query = query.clone();
                let docs = documents_vec.clone();
                async move {
                    let engine = engine.read().await;
                    engine.rerank_batch(&query, &docs)
                }
            })
            .await;
        }

        // user 隔离模式：缓存键前缀 {caller_digest}:——caller 经 SHA-256 全长
        // 十六进制摘要（与语义缓存 L1 键、Redis 层 digest 粒度一致，raw caller
        // 不落键）；off 模式前缀为空（既有键格式逐字节不变）。
        let caller_prefix = if self.isolation.is_enabled() {
            format!("{}:", crate::utils::hash::sha256_hex(caller_id.as_bytes()))
        } else {
            String::new()
        };

        // 缓存启用：逐个检查缓存，仅对未命中的 document 调用引擎
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(documents_vec.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_docs = Vec::new();

        let model_id = self
            .model_config
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or("unknown");
        let query_hash = xxhash_rust::xxh3::xxh3_128(query.as_bytes());
        for (i, doc) in documents_vec.iter().enumerate() {
            let cache_key = format!(
                "rerank:{}{}:{}:{:016x}",
                caller_prefix,
                model_id,
                query_hash,
                xxhash_rust::xxh3::xxh3_128(doc.as_bytes())
            );
            if let Some(cached) = self.cache.get(&cache_key).await {
                // 缓存命中：单元素 Vec<f32>
                scores.push((i, cached.first().copied().unwrap_or(0.0)));
            } else {
                uncached_indices.push(i);
                uncached_docs.push(doc.clone());
            }
        }

        if !uncached_docs.is_empty() {
            // 对未命中的 document 批量调用引擎
            let uncached_scores =
                common::handle_oom_fallback(&engine, &self.model_config, &None, || {
                    let engine = engine.clone();
                    let query = query.clone();
                    let docs = uncached_docs.clone();
                    async move {
                        let engine = engine.read().await;
                        engine.rerank_batch(&query, &docs)
                    }
                })
                .await?;

            // 存入缓存并收集结果
            for (idx, &score) in uncached_indices.iter().zip(uncached_scores.iter()) {
                let cache_key = format!(
                    "rerank:{}{}:{}:{:016x}",
                    caller_prefix,
                    model_id,
                    query_hash,
                    xxhash_rust::xxh3::xxh3_128(documents_vec[*idx].as_bytes())
                );
                self.cache.put(&cache_key, vec![score]).await;
                scores.push((*idx, score));
            }
        }

        // 按原始顺序排列分数
        scores.sort_by_key(|(i, _)| *i);
        Ok(scores.into_iter().map(|(_, s)| s).collect())
    }

    /// 执行重排序：对 query 和 documents 列表计算相关性分数。
    /// 原签名保留（既有调用点零改动），caller 默认 anonymous
    /// （归并桶字面量统一走 `crate::caller::ANONYMOUS_CALLER`）。
    pub async fn process_rerank(
        &self,
        req: RerankRequest,
        max_documents: usize,
        max_query_length: usize,
    ) -> Result<RerankResponse, VecboostError> {
        self.process_rerank_with_caller(
            req,
            max_documents,
            max_query_length,
            crate::caller::ANONYMOUS_CALLER,
        )
        .await
    }

    /// 带调用方身份的重排序（user 隔离模式的真实入口）。
    pub async fn process_rerank_with_caller(
        &self,
        req: RerankRequest,
        max_documents: usize,
        max_query_length: usize,
        caller_id: &str,
    ) -> Result<RerankResponse, VecboostError> {
        self.validate_rerank_request(&req, max_documents, max_query_length)?;

        let documents = &req.documents;

        let start = Instant::now();

        // 检查引擎是否支持 rerank
        let engine_read = self.engine.read().await;
        if !engine_read.supports_rerank() {
            return Err(VecboostError::InternalError(
                crate::i18n::tr("rerank-unsupported").to_string(),
            ));
        }
        drop(engine_read);

        let scores = self
            .compute_rerank_scores(&req.query, documents, caller_id)
            .await?;

        let return_documents = req.return_documents.unwrap_or(false);
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .zip(scores.iter())
            .map(|((idx, doc), &score)| RerankResult {
                index: idx,
                score,
                document: if return_documents {
                    Some(doc.clone())
                } else {
                    None
                },
            })
            .collect();

        // 按分数降序排序（NaN 值排到末尾，保证排序稳定性）
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        // 应用 top_k 截断
        if let Some(top_k) = req.top_k
            && top_k < results.len()
        {
            results.truncate(top_k);
        }

        let processing_time_ms = start.elapsed().as_millis();

        Ok(RerankResponse {
            results,
            processing_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// 计数引擎：每次 rerank_batch 调用 +1，用调用次数区分缓存命中/未命中。
    struct CountingRerankEngine {
        calls: Arc<AtomicU64>,
    }

    #[async_trait]
    impl InferenceEngine for CountingRerankEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 128])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        fn rerank(&self, _query: &str, doc: &str) -> Result<f32, VecboostError> {
            Ok((doc.len() as f32) / 100.0)
        }
        fn rerank_batch(&self, query: &str, docs: &[String]) -> Result<Vec<f32>, VecboostError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            docs.iter().map(|d| self.rerank(query, d)).collect()
        }
        fn supports_rerank(&self) -> bool {
            true
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_rerank_caller_isolation_blocks_cross_caller_hit() {
        let calls = Arc::new(AtomicU64::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingRerankEngine {
                calls: calls.clone(),
            }));
        let cache = Arc::new(OxCacheBackend::with_ttl(100, None));
        let service = RerankService::with_cache(engine, None, cache)
            .with_caller_isolation(crate::cache::CallerIsolation::User);
        let mk = || RerankRequest {
            query: "same query".to_string(),
            documents: vec!["same doc".to_string()],
            top_k: None,
            return_documents: Some(false),
        };

        service
            .process_rerank_with_caller(mk(), 10, 8192, "alice")
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "alice 首次必然计算");
        service
            .process_rerank_with_caller(mk(), 10, 8192, "alice")
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "alice 再次命中缓存");
        service
            .process_rerank_with_caller(mk(), 10, 8192, "bob")
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "bob 必须 miss 并触发自身计算"
        );
    }

    /// 键格式钉（安全 LOW-1）：user 隔离模式下 rerank 缓存键 caller 前缀
    /// 必须是 SHA-256 摘要，raw caller 不得出现在键面；off 模式键格式
    /// 逐字节不变。以预填 digest 键命中（engine 零调用）正向钉住。
    #[tokio::test]
    async fn test_rerank_cache_key_caller_digest_in_user_mode() {
        use crate::utils::hash::sha256_hex;
        let calls = Arc::new(AtomicU64::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingRerankEngine {
                calls: calls.clone(),
            }));
        let cache = Arc::new(OxCacheBackend::with_ttl(100, None));
        let service = RerankService::with_cache(engine, None, Arc::clone(&cache))
            .with_caller_isolation(crate::cache::CallerIsolation::User);

        // 预填 digest 键（model_config=None → model 段为 "unknown"）。
        let digest_key = format!(
            "rerank:{}:unknown:{}:{:016x}",
            sha256_hex(b"alice"),
            xxhash_rust::xxh3::xxh3_128(b"same query"),
            xxhash_rust::xxh3::xxh3_128(b"same doc"),
        );
        cache.put(&digest_key, vec![0.42]).await;

        let resp = service
            .process_rerank_with_caller(
                RerankRequest {
                    query: "same query".to_string(),
                    documents: vec!["same doc".to_string()],
                    top_k: None,
                    return_documents: Some(false),
                },
                10,
                8192,
                "alice",
            )
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "digest 键必须直接命中（键前缀为 SHA-256 摘要）"
        );
        assert!(
            (resp.results[0].score - 0.42).abs() < 1e-6,
            "命中值必须来自预填 digest 键"
        );

        // raw caller 键不得被命中：移除 digest 键后预填 raw 键，必须 miss
        // 计算回填（若查找误用 raw 键，将命中 0.99 且零计算）。
        cache.remove(&digest_key).await;
        let raw_key = format!(
            "rerank:alice:unknown:{}:{:016x}",
            xxhash_rust::xxh3::xxh3_128(b"same query"),
            xxhash_rust::xxh3::xxh3_128(b"same doc"),
        );
        cache.put(&raw_key, vec![0.99]).await;
        let resp = service
            .process_rerank_with_caller(
                RerankRequest {
                    query: "same query".to_string(),
                    documents: vec!["same doc".to_string()],
                    top_k: None,
                    return_documents: Some(false),
                },
                10,
                8192,
                "alice",
            )
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "raw caller 键不得被 user 模式查找命中"
        );
        assert!(
            (resp.results[0].score - 0.08).abs() < 1e-6,
            "miss 后必须计算回填（引擎计分 8/100），got: {}",
            resp.results[0].score
        );
    }

    /// off 模式键格式逐字节不变钉：预填既有裸键（无 caller 前缀）必须命中。
    #[tokio::test]
    async fn test_rerank_cache_key_off_mode_byte_identical() {
        let calls = Arc::new(AtomicU64::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingRerankEngine {
                calls: calls.clone(),
            }));
        let cache = Arc::new(OxCacheBackend::with_ttl(100, None));
        let service = RerankService::with_cache(engine, None, Arc::clone(&cache))
            .with_caller_isolation(crate::cache::CallerIsolation::Off);

        let off_key = format!(
            "rerank:unknown:{}:{:016x}",
            xxhash_rust::xxh3::xxh3_128(b"same query"),
            xxhash_rust::xxh3::xxh3_128(b"same doc"),
        );
        cache.put(&off_key, vec![0.77]).await;
        let resp = service
            .process_rerank_with_caller(
                RerankRequest {
                    query: "same query".to_string(),
                    documents: vec!["same doc".to_string()],
                    top_k: None,
                    return_documents: Some(false),
                },
                10,
                8192,
                "alice",
            )
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "off 模式键必须与既有格式逐字节一致（caller 传入被忽略）"
        );
        assert!((resp.results[0].score - 0.77).abs() < 1e-6);
    }

    /// Mock engine that returns deterministic rerank scores based on document length
    struct MockRerankEngine;

    #[async_trait]
    impl InferenceEngine for MockRerankEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 128])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn rerank(&self, _query: &str, document: &str) -> Result<f32, VecboostError> {
            // Score proportional to document length for deterministic testing
            Ok((document.len() as f32) / 100.0)
        }

        fn rerank_batch(
            &self,
            query: &str,
            documents: &[String],
        ) -> Result<Vec<f32>, VecboostError> {
            documents
                .iter()
                .map(|doc| self.rerank(query, doc))
                .collect()
        }

        fn supports_rerank(&self) -> bool {
            true
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// Engine that does NOT support rerank
    struct NoRerankEngine;

    #[async_trait]
    impl InferenceEngine for NoRerankEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 128])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn supports_rerank(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_service(engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>) -> RerankService {
        RerankService::new(engine, None)
    }

    /// 回归钉：("a","b:c") 与 ("a:b","c") 的缓存键不得互染。
    /// Mock 分数按 doc 长度：前者 0.03，后者 0.01；旧裸拼接键使两次同键，
    /// 第二次命中返回 0.03。
    #[tokio::test]
    async fn test_rerank_cache_keys_distinguish_query_doc_boundaries() {
        let cache = Arc::new(OxCacheBackend::with_ttl(100, None));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = RerankService::with_cache(engine, None, Arc::clone(&cache));

        let mk = |query: &str, doc: &str| RerankRequest {
            query: query.to_string(),
            documents: vec![doc.to_string()],
            top_k: None,
            return_documents: Some(false),
        };

        let r1 = service
            .process_rerank(mk("a", "b:c"), 10, 8192)
            .await
            .unwrap();
        assert!(
            (r1.results[0].score - 0.03).abs() < 1e-6,
            "b:c len=3 → 0.03"
        );

        let r2 = service
            .process_rerank(mk("a:b", "c"), 10, 8192)
            .await
            .unwrap();
        assert!(
            (r2.results[0].score - 0.01).abs() < 1e-6,
            "boundary-shifted pair must compute fresh (0.01), got cached {:?}",
            r2.results[0].score
        );
    }

    /// 回归钉：replace_engine 后 rerank 使用新引擎。
    #[tokio::test]
    async fn test_replace_engine_switches_scoring_source() {
        struct LenEngineA;
        struct LenEngineB;
        macro_rules! impl_len_engine {
            ($t:ident, $scale:expr) => {
                #[async_trait]
                impl InferenceEngine for $t {
                    fn embed(&self, _t: &str) -> Result<Vec<f32>, VecboostError> {
                        Ok(vec![0.0; 8])
                    }
                    fn embed_batch(&self, ts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
                        Ok(ts.iter().map(|_| vec![0.0; 8]).collect())
                    }
                    fn precision(&self) -> &Precision {
                        &Precision::Fp32
                    }
                    fn supports_mixed_precision(&self) -> bool {
                        false
                    }
                    fn rerank(&self, _q: &str, d: &str) -> Result<f32, VecboostError> {
                        Ok(d.len() as f32 * $scale)
                    }
                    fn rerank_batch(
                        &self,
                        q: &str,
                        ds: &[String],
                    ) -> Result<Vec<f32>, VecboostError> {
                        Ok(ds.iter().map(|d| self.rerank(q, d).unwrap()).collect())
                    }
                    async fn try_fallback_to_cpu(
                        &mut self,
                        _: &crate::config::model::ModelConfig,
                    ) -> Result<(), VecboostError> {
                        Ok(())
                    }
                }
            };
        }
        impl_len_engine!(LenEngineA, 1.0);
        impl_len_engine!(LenEngineB, 100.0);

        let engine_a: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(LenEngineA));
        let mut service = make_service(engine_a);
        let req = RerankRequest {
            query: "q".to_string(),
            documents: vec!["abcd".to_string()],
            top_k: None,
            return_documents: Some(false),
        };
        let r1 = service.process_rerank(req.clone(), 10, 8192).await.unwrap();
        assert!((r1.results[0].score - 4.0).abs() < 1e-6);

        let engine_b: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(LenEngineB));
        service.replace_engine(engine_b);
        let r2 = service.process_rerank(req, 10, 8192).await.unwrap();
        assert!(
            (r2.results[0].score - 400.0).abs() < 1e-6,
            "must score via new engine"
        );
    }

    #[tokio::test]
    async fn test_rerank_valid_input_returns_sorted_scores() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "what is rust?".to_string(),
            documents: vec![
                "short".to_string(),
                "a much longer document about programming".to_string(),
                "medium length doc".to_string(),
            ],
            top_k: None,
            return_documents: Some(false),
        };

        let result = service.process_rerank(req, 100, 8192).await.unwrap();

        // Results should be sorted by score descending
        assert_eq!(result.results.len(), 3);
        for i in 1..result.results.len() {
            assert!(
                result.results[i - 1].score >= result.results[i].score,
                "Results not sorted: {} < {}",
                result.results[i - 1].score,
                result.results[i].score
            );
        }
        // Documents should not be returned
        assert!(result.results[0].document.is_none());
    }

    #[tokio::test]
    async fn test_rerank_empty_query_returns_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "".to_string(),
            documents: vec!["doc1".to_string()],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(
                    msg.to_lowercase().contains("empty") || msg.to_lowercase().contains("text")
                );
            }
            other => panic!("Expected InvalidInput, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rerank_empty_documents_returns_error() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "test query".to_string(),
            documents: vec![],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(
                    msg.to_lowercase().contains("empty") || msg.to_lowercase().contains("document")
                );
            }
            other => panic!("Expected InvalidInput, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rerank_top_k_truncation() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec![
                "a".to_string(),
                "bb".to_string(),
                "ccc".to_string(),
                "dddd".to_string(),
                "eeeee".to_string(),
            ],
            top_k: Some(2),
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await.unwrap();
        assert_eq!(result.results.len(), 2);
    }

    #[tokio::test]
    async fn test_rerank_top_k_zero_rejected() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec!["a".to_string(), "b".to_string()],
            top_k: Some(0),
            return_documents: None,
        };

        let err = service.process_rerank(req, 100, 8192).await.unwrap_err();
        assert!(
            matches!(err, VecboostError::InvalidInput(_)),
            "top_k=0 should return InvalidInput, got: {:?}",
            err
        );
    }

    #[tokio::test]
    async fn test_rerank_return_documents_flag() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec!["hello world".to_string(), "foo bar".to_string()],
            top_k: None,
            return_documents: Some(true),
        };

        let result = service.process_rerank(req, 100, 8192).await.unwrap();
        for r in &result.results {
            assert!(r.document.is_some(), "document should be returned");
        }
    }

    #[tokio::test]
    async fn test_rerank_unsupported_engine_returns_error() {
        crate::i18n::init();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(NoRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec!["doc".to_string()],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InternalError(msg) => {
                assert!(msg.contains("does not support rerank"), "got: {}", msg);
            }
            other => panic!("Expected InternalError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rerank_query_length_exceeded_returns_error() {
        crate::i18n::init();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        let req = RerankRequest {
            query: "x".repeat(10000),
            documents: vec!["doc".to_string()],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("exceeds maximum"), "got: {}", msg);
            }
            other => panic!("Expected InvalidInput for query length, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rerank_documents_exceeding_limit_returns_error() {
        crate::i18n::init();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service(engine);

        // 5 documents but max_documents = 3
        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec![
                "a".to_string(),
                "bb".to_string(),
                "ccc".to_string(),
                "dddd".to_string(),
                "eeeee".to_string(),
            ],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 3, 8192).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.contains("exceeds max documents"), "got: {}", msg);
            }
            other => panic!("Expected InvalidInput, got: {:?}", other),
        }
    }

    // -- Cache-enabled path tests --

    fn make_service_with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    ) -> RerankService {
        let cache = Arc::new(OxCacheBackend::new(100));
        RerankService::with_cache(engine, None, cache)
    }

    #[tokio::test]
    async fn test_rerank_with_cache_first_call_misses() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service_with_cache(engine);

        let req = RerankRequest {
            query: "cache test query".to_string(),
            documents: vec!["doc one".to_string(), "doc two".to_string()],
            top_k: None,
            return_documents: None,
        };

        let result = service.process_rerank(req, 100, 8192).await.unwrap();
        assert_eq!(result.results.len(), 2);
    }

    #[tokio::test]
    async fn test_rerank_with_cache_second_call_hits() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let service = make_service_with_cache(engine.clone());

        // First call: cache miss, computes and stores scores
        let req1 = RerankRequest {
            query: "same query".to_string(),
            documents: vec!["cached doc".to_string()],
            top_k: None,
            return_documents: None,
        };
        let result1 = service.process_rerank(req1, 100, 8192).await.unwrap();

        // Second call with same query+doc: should hit cache
        let req2 = RerankRequest {
            query: "same query".to_string(),
            documents: vec!["cached doc".to_string()],
            top_k: None,
            return_documents: None,
        };
        let result2 = service.process_rerank(req2, 100, 8192).await.unwrap();

        // Scores should match
        assert_eq!(result1.results.len(), result2.results.len());
        assert!((result1.results[0].score - result2.results[0].score).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_rerank_with_cache_constructor() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let cache = Arc::new(OxCacheBackend::new(50));
        let service = RerankService::with_cache(engine, None, cache);

        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec!["doc".to_string()],
            top_k: None,
            return_documents: None,
        };
        let result = service.process_rerank(req, 100, 8192).await.unwrap();
        assert_eq!(result.results.len(), 1);
    }

    // -- Direct mock method calls to cover unused trait impls --
    #[test]
    fn test_mock_rerank_engine_embed() {
        let engine = MockRerankEngine;
        let vec = engine.embed("test").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_mock_rerank_engine_embed_batch() {
        let engine = MockRerankEngine;
        let texts = vec!["a".to_string(), "b".to_string()];
        let vecs = engine.embed_batch(&texts).unwrap();
        assert_eq!(vecs.len(), 2);
    }

    #[test]
    fn test_mock_rerank_engine_precision() {
        let engine = MockRerankEngine;
        assert_eq!(*engine.precision(), Precision::Fp32);
    }

    #[test]
    fn test_mock_rerank_engine_supports_mixed_precision() {
        let engine = MockRerankEngine;
        assert!(!engine.supports_mixed_precision());
    }

    #[tokio::test]
    async fn test_mock_rerank_engine_try_fallback() {
        let mut engine = MockRerankEngine;
        let config = crate::config::model::ModelConfig {
            translation_params: None,
            transcription_params: None,
            name: "test".to_string(),
            engine_type: crate::config::model::EngineType::Candle,
            model_path: std::path::PathBuf::from("/tmp"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        assert!(engine.try_fallback_to_cpu(&config).await.is_ok());
    }

    #[test]
    fn test_no_rerank_engine_embed() {
        let engine = NoRerankEngine;
        let vec = engine.embed("test").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_no_rerank_engine_embed_batch() {
        let engine = NoRerankEngine;
        let texts = vec!["a".to_string()];
        let vecs = engine.embed_batch(&texts).unwrap();
        assert_eq!(vecs.len(), 1);
    }

    #[test]
    fn test_no_rerank_engine_precision() {
        assert_eq!(*NoRerankEngine.precision(), Precision::Fp32);
    }

    #[test]
    fn test_no_rerank_engine_supports_mixed_precision() {
        assert!(!NoRerankEngine.supports_mixed_precision());
    }

    #[tokio::test]
    async fn test_no_rerank_engine_try_fallback() {
        let mut engine = NoRerankEngine;
        let config = crate::config::model::ModelConfig {
            translation_params: None,
            transcription_params: None,
            name: "test".to_string(),
            engine_type: crate::config::model::EngineType::Candle,
            model_path: std::path::PathBuf::from("/tmp"),
            tokenizer_path: None,
            device: crate::config::model::DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
            task: crate::config::model::ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        assert!(engine.try_fallback_to_cpu(&config).await.is_ok());
    }
}
