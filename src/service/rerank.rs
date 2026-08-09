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

#[allow(dead_code)]
pub struct RerankService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    validator: InputValidator,
    model_config: Option<ModelConfig>,
    cache: Arc<OxCacheBackend>,
    memory_manager: Option<SharedGpuMemoryManager>,
    batch_scheduler: Option<Arc<DynamicBatchScheduler>>,
}

impl RerankService {
    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            cache: Arc::new(OxCacheBackend::disabled()),
            memory_manager: None,
            batch_scheduler: None,
        }
    }

    #[allow(private_interfaces)]
    pub fn with_cache(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
        cache: Arc<OxCacheBackend>,
    ) -> Self {
        Self {
            engine,
            validator: InputValidator::with_default(),
            model_config,
            cache,
            memory_manager: None,
            batch_scheduler: None,
        }
    }

    #[allow(private_interfaces)]
    pub fn with_all(
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
            return Err(VecboostError::InvalidInput(format!(
                "Query length {} exceeds maximum allowed length {}",
                req.query.len(),
                max_query_length
            )));
        }

        // 验证 documents 非空
        if req.documents.is_empty() {
            return Err(VecboostError::InvalidInput(
                "Documents list cannot be empty".to_string(),
            ));
        }

        // 验证 documents 数量不超过限制
        if req.documents.len() > max_documents {
            return Err(VecboostError::InvalidInput(format!(
                "Documents count {} exceeds max documents per query {}",
                req.documents.len(),
                max_documents
            )));
        }

        let documents = &req.documents;

        let start = Instant::now();

        // 检查引擎是否支持 rerank
        let engine_read = self.engine.read().await;
        if !engine_read.supports_rerank() {
            return Err(VecboostError::InternalError(
                "Current engine does not support rerank".to_string(),
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
                let uncached_scores = common::handle_oom_fallback(
                    &engine,
                    &self.model_config,
                    &None,
                    || {
                        let engine = engine.clone();
                        let query = query.clone();
                        let docs = uncached_docs.clone();
                        async move {
                            let engine = engine.read().await;
                            engine.rerank_batch(&query, &docs)
                        }
                    },
                )
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
            common::handle_oom_fallback(
                &engine,
                &self.model_config,
                &None,
                || {
                    let engine = engine.clone();
                    let query = query.clone();
                    let docs = documents_vec.clone();
                    async move {
                        let engine = engine.read().await;
                        engine.rerank_batch(&query, &docs)
                    }
                },
            )
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

        // 按分数降序排序
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // 应用 top_k 截断
        if let Some(top_k) = req.top_k {
            if top_k < results.len() {
                results.truncate(top_k);
            }
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
            documents.iter().map(|doc| self.rerank(query, doc)).collect()
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
                assert!(msg.to_lowercase().contains("empty") || msg.to_lowercase().contains("text"));
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
                assert!(msg.to_lowercase().contains("empty") || msg.to_lowercase().contains("document"));
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
}
