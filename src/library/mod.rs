// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Library Interface — 轻量级 SDK 入口
//!
//! 提供 `VecBoostLibrary` 作为 library 模式的唯一入口，使 VecBoost 可以在不启动
//! HTTP/gRPC 服务器的情况下作为嵌入式向量化(embedding)和重排序(rerank)模块
//! 被其他 Rust 应用程序直接集成调用。
//!
//! 通过 trait-kit `AsyncKit` 管理 `EmbeddingModule` 和 `RerankModule` 的注册、
//! 构建和生命周期，复用现有 `EmbeddingService` / `RerankService` 的完整推理逻辑。
//!
//! # 异步 API
//!
//! ```ignore
//! let lib = VecBoostLibrary::new(config).await?;
//! let response = lib.embed("hello world").await?;
//! ```
//!
//! # 同步 API
//!
//! ```ignore
//! let lib = VecBoostLibrary::new(config).await?;
//! let response = lib.embed_sync("hello world")?;
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::model::ModelConfig;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, RerankRequest,
    RerankResponse,
};
use crate::engine::{AnyEngine, EngineFactory};
use crate::error::VecboostError;
use crate::module_registry::{EmbeddingModule, RerankModule};
use crate::service::embedding::EmbeddingService;
use crate::service::rerank::RerankService;
use crate::RerankConfig;

// ---------------------------------------------------------------------------
// LibraryConfig
// ---------------------------------------------------------------------------

/// Library 模式配置
///
/// 封装模型配置和可选的缓存/重排序参数，用于初始化 `VecBoostLibrary`。
#[derive(Debug, Clone)]
pub struct LibraryConfig {
    /// 模型配置（引擎类型、模型路径、设备等）
    pub model_config: ModelConfig,
    /// 嵌入缓存大小（0 = 禁用缓存）
    pub cache_size: usize,
    /// 重排序配置（None = 使用默认值）
    pub rerank_config: Option<RerankConfig>,
}

impl Default for LibraryConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            cache_size: 0,
            rerank_config: None,
        }
    }
}

impl LibraryConfig {
    /// 从 `ModelConfig` 快速创建，其余参数使用默认值
    pub fn from_model_config(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// VecBoostLibrary
// ---------------------------------------------------------------------------

/// VecBoost Library 入口 — 提供无网络服务的向量化和重排序 API
///
/// 内部通过 trait-kit `AsyncKit` 管理模块生命周期，复用 `EmbeddingService`
/// 和 `RerankService` 的完整推理逻辑。
///
/// 同时提供异步和同步 API：
/// - 异步方法（`embed`、`embed_batch`、`rerank`）适用于 tokio 运行时
/// - 同步方法（`embed_sync`、`embed_batch_sync`、`rerank_sync`）适用于非异步上下文
pub struct VecBoostLibrary {
    kit: Arc<trait_kit::AsyncKit<trait_kit::AsyncReady>>,
}

impl VecBoostLibrary {
    /// 创建 library 实例
    ///
    /// 通过 `EngineFactory` 创建推理引擎，构建 `EmbeddingService` + `RerankService`，
    /// 注册到最小 `AsyncKit`（仅 `EmbeddingModule` + `RerankModule`）。
    pub async fn new(config: LibraryConfig) -> Result<Self, VecboostError> {
        // 1. 创建推理引擎
        let engine = EngineFactory::create(
            config.model_config.engine_type.clone(),
            &config.model_config,
        )?;
        let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(engine));

        // 2. 创建 EmbeddingService（可选缓存）
        let embedding_service = if config.cache_size > 0 {
            Arc::new(RwLock::new(EmbeddingService::with_cache(
                engine.clone(),
                Some(config.model_config.clone()),
                config.cache_size,
            )))
        } else {
            Arc::new(RwLock::new(EmbeddingService::new(
                engine.clone(),
                Some(config.model_config.clone()),
            )))
        };

        // 3. 创建 RerankService
        let rerank_service = Arc::new(RwLock::new(RerankService::new(
            engine,
            Some(config.model_config),
        )));

        // 4. 构建最小 AsyncKit（仅 EmbeddingModule + RerankModule）
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(embedding_service);
        kit.set_config(rerank_service);
        kit.set_config(config.rerank_config.unwrap_or_default());
        kit
            .register::<EmbeddingModule>()
            .map_err(|e| VecboostError::InternalError(format!("Failed to register EmbeddingModule: {}", e)))?;
        kit
            .register::<RerankModule>()
            .map_err(|e| VecboostError::InternalError(format!("Failed to register RerankModule: {}", e)))?;
        kit.register_lifecycle::<EmbeddingModule>();
        kit.register_lifecycle::<RerankModule>();

        let kit = kit
            .build()
            .await
            .map_err(|e| VecboostError::InternalError(format!("Failed to build AsyncKit: {}", e)))?;

        Ok(Self {
            kit: Arc::new(kit),
        })
    }

    // -----------------------------------------------------------------------
    // 异步 API
    // -----------------------------------------------------------------------

    /// 异步向量化：将单条文本转换为向量表示
    pub async fn embed(&self, text: &str) -> Result<EmbedResponse, VecboostError> {
        let service = self
            .kit
            .require::<EmbeddingModule>()
            .map_err(|e| VecboostError::InternalError(format!("Failed to require EmbeddingModule: {}", e)))?;
        let svc = service.read().await;
        svc.process_text(
            EmbedRequest {
                text: text.to_string(),
                normalize: None,
            },
            None,
        )
        .await
    }

    /// 异步批量向量化：将多条文本转换为向量表示
    pub async fn embed_batch(
        &self,
        texts: &[String],
    ) -> Result<BatchEmbedResponse, VecboostError> {
        let service = self
            .kit
            .require::<EmbeddingModule>()
            .map_err(|e| VecboostError::InternalError(format!("Failed to require EmbeddingModule: {}", e)))?;
        let svc = service.read().await;
        svc.process_batch(
            BatchEmbedRequest {
                texts: texts.to_vec(),
                mode: None,
                normalize: None,
            },
            None,
        )
        .await
    }

    /// 异步重排序：对查询和文档列表计算相关性分数，返回按分数降序排列的结果
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<RerankResponse, VecboostError> {
        let service = self
            .kit
            .require::<RerankModule>()
            .map_err(|e| VecboostError::InternalError(format!("Failed to require RerankModule: {}", e)))?;
        let rerank_config = self
            .kit
            .config::<RerankConfig>()
            .ok()
            .unwrap_or_default();
        let svc = service.read().await;
        svc.process_rerank(
            RerankRequest {
                query: query.to_string(),
                documents: documents.to_vec(),
                top_k,
                return_documents: None,
            },
            rerank_config.max_documents_per_query,
            rerank_config.max_query_length,
        )
        .await
    }

    // -----------------------------------------------------------------------
    // 同步 API
    // -----------------------------------------------------------------------

    /// 同步向量化：将单条文本转换为向量表示
    ///
    /// 内部创建临时 tokio runtime 执行异步推理。
    /// 注意：不要在已有 tokio runtime 的异步上下文中调用此方法，应使用 `embed()` 代替。
    pub fn embed_sync(&self, text: &str) -> Result<EmbedResponse, VecboostError> {
        let future = self.embed(text);
        Self::block_on_future(future)
    }

    /// 同步批量向量化：将多条文本转换为向量表示
    pub fn embed_batch_sync(
        &self,
        texts: &[String],
    ) -> Result<BatchEmbedResponse, VecboostError> {
        let future = self.embed_batch(texts);
        Self::block_on_future(future)
    }

    /// 同步重排序：对查询和文档列表计算相关性分数
    pub fn rerank_sync(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<RerankResponse, VecboostError> {
        let future = self.rerank(query, documents, top_k);
        Self::block_on_future(future)
    }

    /// 内部辅助：执行异步 future 并阻塞等待结果
    ///
    /// 创建临时 `current_thread` runtime 执行 future。
    ///
    /// **注意**：不要在已有 tokio runtime 的异步上下文中调用 sync API，
    /// 应使用对应的异步方法（`embed` / `embed_batch` / `rerank`）代替。
    fn block_on_future<F: std::future::Future>(future: F) -> F::Output {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("Failed to create tokio runtime for sync API");
        rt.block_on(future)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;

    /// Mock engine that returns deterministic vectors for testing
    struct MockEngine {
        dimension: usize,
    }

    impl MockEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![1.0; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts
                .iter()
                .map(|_| vec![1.0; self.dimension])
                .collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn rerank(&self, _query: &str, document: &str) -> Result<f32, VecboostError> {
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
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// 测试辅助：直接注入 mock engine 构建 VecBoostLibrary
    async fn make_test_library() -> VecBoostLibrary {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine::new(128)));

        let embedding_service =
            Arc::new(RwLock::new(EmbeddingService::new(engine.clone(), None)));
        let rerank_service =
            Arc::new(RwLock::new(RerankService::new(engine, None)));

        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(embedding_service);
        kit.set_config(rerank_service);
        kit.set_config(RerankConfig::default());
        kit.register::<EmbeddingModule>().unwrap();
        kit.register::<RerankModule>().unwrap();
        let kit = kit.build().await.unwrap();

        VecBoostLibrary {
            kit: Arc::new(kit),
        }
    }

    // -------------------------------------------------------------------------
    // 构造测试
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_library_construction_succeeds() {
        let lib = make_test_library().await;
        assert!(lib.kit.contains::<EmbeddingModule>());
        assert!(lib.kit.contains::<RerankModule>());
    }

    // -------------------------------------------------------------------------
    // 异步 API 测试
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_embed_returns_correct_dimension() {
        let lib = make_test_library().await;
        let response = lib.embed("hello world").await.unwrap();
        assert_eq!(response.dimension, 128);
        assert_eq!(response.embedding.len(), 128);
    }

    #[tokio::test]
    async fn test_embed_batch_returns_correct_count() {
        let lib = make_test_library().await;
        let texts = vec!["hello".to_string(), "world".to_string()];
        let response = lib.embed_batch(&texts).await.unwrap();
        assert_eq!(response.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_rerank_returns_sorted_results() {
        let lib = make_test_library().await;
        let response = lib
            .rerank(
                "what is rust?",
                &[
                    "short".to_string(),
                    "a much longer document about programming".to_string(),
                    "medium length doc".to_string(),
                ],
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.results.len(), 3);
        // Results should be sorted by score descending
        for i in 1..response.results.len() {
            assert!(
                response.results[i - 1].score >= response.results[i].score,
                "Results not sorted by score descending"
            );
        }
    }

    #[tokio::test]
    async fn test_rerank_with_top_k() {
        let lib = make_test_library().await;
        let response = lib
            .rerank(
                "test",
                &[
                    "a".to_string(),
                    "bb".to_string(),
                    "ccc".to_string(),
                ],
                Some(2),
            )
            .await
            .unwrap();
        assert_eq!(response.results.len(), 2);
    }

    // -------------------------------------------------------------------------
    // 同步 API 测试
    // -------------------------------------------------------------------------

    #[test]
    fn test_sync_embed_outside_runtime() {
        // 普通 #[test] 线程无活跃 tokio runtime
        // 先创建临时 runtime 构建 library，然后 drop
        let lib = {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(make_test_library())
            // rt 在此 drop，无活跃 runtime
        };
        // 此处无活跃 tokio runtime，sync API 可安全调用
        let response = lib.embed_sync("hello").unwrap();
        assert_eq!(response.dimension, 128);
    }

    #[test]
    fn test_sync_rerank_outside_runtime() {
        let lib = {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(make_test_library())
        };
        let response = lib
            .rerank_sync(
                "query",
                &["doc a".to_string(), "doc longer b".to_string()],
                None,
            )
            .unwrap();
        assert_eq!(response.results.len(), 2);
    }

    // -------------------------------------------------------------------------
    // LibraryConfig 测试
    // -------------------------------------------------------------------------

    #[test]
    fn test_library_config_default() {
        let config = LibraryConfig::default();
        assert_eq!(config.cache_size, 0);
        assert!(config.rerank_config.is_none());
    }

    #[test]
    fn test_library_config_from_model_config() {
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            expected_dimension: Some(768),
            ..Default::default()
        };
        let config = LibraryConfig::from_model_config(model_config);
        assert_eq!(config.model_config.name, "test-model");
        assert_eq!(config.model_config.expected_dimension, Some(768));
        assert_eq!(config.cache_size, 0);
    }
}
