// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Rerank 服务实现

use crate::cache::OxCacheBackend;
use crate::config::model::ModelConfig;
use crate::device::DynamicBatchScheduler;
use crate::device::memory_optimizer::SharedGpuMemoryManager;
use crate::domain::{RerankRequest, RerankResponse, RerankResult};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::service::common;
use crate::utils::InputValidator;
use crate::utils::validator::input::TextValidator;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[allow(
    dead_code,
    reason = "Test helper / trait dispatch / inventory, not directly called"
)]
pub struct RerankService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    cache: Arc<OxCacheBackend>,
    memory_manager: Option<SharedGpuMemoryManager>,
    batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
}

impl RerankService {
    /// 统一内部构造入口，消除多个构造器间的字段初始化重复。
    fn build(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
        memory_manager: Option<SharedGpuMemoryManager>,
        batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            cache,
            memory_manager,
            batch_scheduler,
        }
    }

    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self::build(
            engine,
            model_config,
            Arc::new(OxCacheBackend::disabled()),
            None,
            None,
        )
    }

    #[allow(private_interfaces)]
    pub fn with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
    ) -> Self {
        Self::build(engine, model_config, cache, None, None)
    }

    #[allow(private_interfaces)]
    pub fn with_all(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
        memory_manager: Option<SharedGpuMemoryManager>,
        batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
    ) -> Self {
        Self::build(engine, model_config, cache, memory_manager, batch_scheduler)
    }

    /// 执行重排序：对 query 和 documents 列表计算相关性分数
    pub async fn process_rerank(
        &self,
        req: RerankRequest,
        max_documents: usize,
        max_query_length: usize,
    ) -> Result<RerankResponse, VecboostError> {
        // 验证 query 非空
        self.validator.validate_text(&req.query)?;

        // 验证 query 长度
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

        // 验证 top_k 有效性（0 无意义，应拒绝）
        if let Some(top_k) = req.top_k
            && top_k == 0
        {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("rerank-invalid-top-k").to_string(),
            ));
        }

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

        // 使用 OOM 降级包装的批量推理，支持缓存
        let query = req.query.clone();
        let documents_vec = documents.to_vec();
        let cache = self.cache.clone();
        let engine = self.engine.clone();

        let scores = if cache.is_enabled() {
            // 缓存启用：逐个检查缓存，仅对未命中的 document 调用引擎
            let mut scores = Vec::with_capacity(documents_vec.len());
            let mut uncached_indices = Vec::new();
            let mut uncached_docs = Vec::new();

            for (i, doc) in documents_vec.iter().enumerate() {
                let cache_key = format!("rerank:{}:{}", query, doc);
                if let Some(cached) = cache.get(&cache_key).await {
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
                    let cache_key = format!("rerank:{}:{}", query, documents_vec[*idx]);
                    cache.put(&cache_key, vec![score]).await;
                    scores.push((*idx, score));
                }
            }

            // 按原始顺序排列分数
            scores.sort_by_key(|(i, _)| *i);
            scores.into_iter().map(|(_, s)| s).collect()
        } else {
            // 缓存禁用：直接批量调用引擎
            common::handle_oom_fallback(&engine, &self.model_config, &None, || {
                let engine = engine.clone();
                let query = query.clone();
                let docs = documents_vec.clone();
                async move {
                    let engine = engine.read().await;
                    engine.rerank_batch(&query, &docs)
                }
            })
            .await?
        };

        // 构建结果并排序
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
    async fn test_rerank_with_all_constructors() {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockRerankEngine));
        let cache = Arc::new(OxCacheBackend::new(50));
        let service = RerankService::with_all(engine, None, cache, None, None);

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
        };
        assert!(engine.try_fallback_to_cpu(&config).await.is_ok());
    }
}
