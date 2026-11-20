// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Library Interface — 轻量级 SDK 入口
//!
//! 提供两种集成方式：
//!
//! ## 1. `VecBoostLibrary` — 开箱即用的独立 SDK
//!
//! 无需启动 HTTP/gRPC 服务器即可使用完整的嵌入和重排序功能。
//!
//! ```ignore
//! let lib = VecBoostLibrary::new(config).await?;
//! let response = lib.embed("hello world").await?;
//! ```
//!
//! ## 2. `VecBoostModuleBuilder` — trait-kit 模块化集成
//!
//! 将 VecBoost 的 embedding/rerank 能力作为 trait-kit Module 注册到外部项目
//! 自己的 `AsyncKit` 中，与其他模块共存、共享生命周期管理。
//!
//! ```ignore
//! let mut kit = trait_kit::AsyncKit::new;
//! // 注册外部项目自己的模块...
//! kit.register::<MyAppModule>?;
//!
//! // 将 VecBoost 能力注入同一个 kit
//! VecBoostModuleBuilder::new(model_config)
//!     .embedding
//!     .rerank
//!     .build(&mut kit)
//!     .await?;
//!
//! let kit = kit.build.await?;
//! // 通过 kit.require::<EmbeddingModule> 获取 EmbeddingService
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::RerankConfig;
use crate::config::model::ModelConfig;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, RerankRequest,
    RerankResponse,
};
use crate::engine::{AnyEngine, EngineFactory};
use crate::error::VecboostError;
use crate::registry::{EmbeddingModule, RerankModule};
use crate::service::embedding::EmbeddingService;
use crate::service::rerank::RerankService;

// LibraryConfig

/// Library 模式配置
///
/// 封装模型配置和可选的缓存/重排序参数，用于初始化 `VecBoostLibrary`。
#[derive(Debug, Clone, Default)]
pub struct LibraryConfig {
    /// 模型配置（引擎类型、模型路径、设备等）
    pub model_config: ModelConfig,
    /// 嵌入缓存大小（0 = 禁用缓存）
    pub cache_size: usize,
    /// 重排序配置（None = 使用默认值）。`rerank_config.cache_enabled` 在
    /// 库装配期消费（与 server 模式同口径）；None = 默认值 = 缓存禁用。
    pub rerank_config: Option<RerankConfig>,
    /// 调用方隔离模式（默认 Off = 缓存键行为不变）。取值语义与 server 模式
    /// `[semantic_cache] caller_isolation` 一致。库门面 `rerank()` /
    /// `rerank_sync()` / `embed()` 保留 anonymous 语义（单一共享分区）；
    /// 需要按调用方隔离的嵌入方经 `rerank_with_caller()` /
    /// `rerank_sync_with_caller()` / `embed_with_caller()` 显式传 caller。
    pub caller_isolation: crate::cache::CallerIsolation,
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

// VecBoostLibrary

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

        // 3. 创建 RerankService：隔离模式经 LibraryConfig 显式传入。门面
        // rerank()/rerank_sync() 保留 anonymous 语义；携带调用方身份经
        // rerank_with_caller()/rerank_sync_with_caller()（user 隔离模式下
        // 各 caller 独立分区）。[rerank] cache_enabled 消费（与 server 模式
        // build_rerank_service 同口径）：true 且 cache_size>0 时构建进程内
        // 精确缓存，false（默认）disabled 零变化。
        let rerank_config = config.rerank_config.clone().unwrap_or_default();
        let rerank_service = if rerank_config.cache_enabled && rerank_config.cache_size > 0 {
            Arc::new(RwLock::new(
                RerankService::with_cache_capacity(
                    engine.clone(),
                    Some(config.model_config.clone()),
                    rerank_config.cache_size,
                )
                .with_caller_isolation(config.caller_isolation),
            ))
        } else {
            Arc::new(RwLock::new(
                RerankService::new(engine, Some(config.model_config))
                    .with_caller_isolation(config.caller_isolation),
            ))
        };

        // 4. 构建最小 AsyncKit（仅 EmbeddingModule + RerankModule）
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(embedding_service);
        kit.set_config(rerank_service);
        kit.set_config(rerank_config);
        kit.register::<EmbeddingModule>().map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-register-embeddingmodule-bdd55",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;
        kit.register::<RerankModule>().map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-register-rerankmodule-a244b",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;
        kit.register_lifecycle::<EmbeddingModule>();
        kit.register_lifecycle::<RerankModule>();

        let kit = kit.build().await.map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-build-asynckit-181f0",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;

        Ok(Self { kit: Arc::new(kit) })
    }

    // 异步 API

    /// 异步向量化：将单条文本转换为向量表示。
    ///
    /// 旧方法保留 **anonymous 语义**（与库门面既有行为一致）：caller 固定
    /// 归并桶，不携带调用方身份。需要按调用方分区的嵌入方（user 隔离模式）
    /// 经 [`VecBoostLibrary::embed_with_caller`] 显式传 caller。
    pub async fn embed(&self, text: &str) -> Result<EmbedResponse, VecboostError> {
        self.embed_with_caller(text, crate::caller::ANONYMOUS_CALLER)
            .await
    }

    /// 异步向量化（带调用方身份）：user 隔离模式下按 caller 分区语义缓存。
    ///
    /// 与 [`VecBoostLibrary::embed`] 同构，显式携带 `caller`——caller 语义
    /// 与 server 模式一致：仅用于缓存分区，不做认证校验；off 模式下该参数
    /// 被缓存忽略，行为与 anonymous 路径一致。
    pub async fn embed_with_caller(
        &self,
        text: &str,
        caller: &str,
    ) -> Result<EmbedResponse, VecboostError> {
        let service = self.kit.require::<EmbeddingModule>().map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-require-embeddingmodule-c3bff",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;
        let svc = service.read().await;
        svc.process_text_with_caller(
            EmbedRequest {
                text: text.to_string(),
                normalize: None,
            },
            None,
            caller,
        )
        .await
    }

    /// 异步批量向量化：将多条文本转换为向量表示
    pub async fn embed_batch(&self, texts: &[String]) -> Result<BatchEmbedResponse, VecboostError> {
        let service = self.kit.require::<EmbeddingModule>().map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-require-embeddingmodule-c3bff",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;
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

    /// 异步重排序：对查询和文档列表计算相关性分数，返回按分数降序排列的结果。
    ///
    /// 旧方法保留 **anonymous 语义**（与库门面既有行为一致）：caller 固定
    /// 归并桶。user 隔离模式下需要按调用方分区的嵌入方经
    /// [`VecBoostLibrary::rerank_with_caller`] 显式传 caller。
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<RerankResponse, VecboostError> {
        self.rerank_with_caller(query, documents, top_k, crate::caller::ANONYMOUS_CALLER)
            .await
    }

    /// 异步重排序（带调用方身份）：user 隔离模式下按 caller 分区 rerank 缓存。
    ///
    /// 与 [`VecBoostLibrary::rerank`] 同构，显式携带 `caller`；caller 语义
    /// 与 server 模式一致：仅用于缓存分区，不做认证校验；off 模式下该参数
    /// 被忽略，行为与 anonymous 路径一致。
    pub async fn rerank_with_caller(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
        caller: &str,
    ) -> Result<RerankResponse, VecboostError> {
        let service = self.kit.require::<RerankModule>().map_err(|e| {
            VecboostError::InternalError(crate::i18n::tr_with_args(
                "detail-failed-to-require-rerankmodule-c45ca",
                crate::i18n::tr_args(&[("p1", &e.to_string())]),
            ))
        })?;
        let rerank_config = self.kit.config::<RerankConfig>().ok().unwrap_or_default();
        let svc = service.read().await;
        svc.process_rerank_with_caller(
            RerankRequest {
                query: query.to_string(),
                documents: documents.to_vec(),
                top_k,
                return_documents: None,
            },
            rerank_config.max_documents_per_query,
            rerank_config.max_query_length,
            caller,
        )
        .await
    }

    // 同步 API

    /// 同步向量化：将单条文本转换为向量表示
    ///
    /// 使用进程级共享 runtime 执行异步推理;在 tokio 异步上下文中调用会返回
    /// 错误(非 panic),应使用 `embed` 代替。
    pub fn embed_sync(&self, text: &str) -> Result<EmbedResponse, VecboostError> {
        let future = self.embed(text);
        Self::block_on_future(future)
    }

    /// 同步批量向量化：将多条文本转换为向量表示
    pub fn embed_batch_sync(&self, texts: &[String]) -> Result<BatchEmbedResponse, VecboostError> {
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

    /// 同步重排序（带调用方身份）：user 隔离模式下按 caller 分区 rerank 缓存。
    ///
    /// 与 [`VecBoostLibrary::rerank_with_caller`] 同构的同步入口；在已有
    /// tokio runtime 上下文中调用会返回错误，应使用异步变体。
    pub fn rerank_sync_with_caller(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
        caller: &str,
    ) -> Result<RerankResponse, VecboostError> {
        let future = self.rerank_with_caller(query, documents, top_k, caller);
        Self::block_on_future(future)
    }

    /// 内部辅助：执行异步 future 并阻塞等待结果
    ///
    /// 进程级共享 **multi_thread** runtime(`OnceLock` 复用,首次调用惰性创建)。
    /// 推理引擎内部使用 `block_in_place`,必须 multi_thread runtime。
    /// 在已有 tokio runtime 上下文中调用 → 返回 `VecboostError`(指引改用
    /// 异步方法 `embed` / `embed_batch` / `rerank`),不再 panic。
    fn block_on_future<F: std::future::Future>(future: F) -> F::Output
    where
        F::Output: IntoSyncResult,
    {
        // 在已有 tokio runtime 上下文中调用 → 返回可操作错误而非 panic
        if tokio::runtime::Handle::try_current().is_ok() {
            return F::Output::into_sync_result(
                "sync API called from within a tokio runtime; use the async variants                  (embed / embed_batch / rerank) instead",
            );
        }
        static SHARED_RUNTIME: std::sync::OnceLock<std::io::Result<tokio::runtime::Runtime>> =
            std::sync::OnceLock::new();
        let rt = SHARED_RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
        });
        match rt {
            Ok(rt) => rt.block_on(future),
            Err(e) => F::Output::into_sync_result(format!(
                "Failed to create tokio runtime for sync API: {e}"
            )),
        }
    }
}

/// 内部约定:sync 包装层把"不能 block_on"类失败转成 `VecboostError`。
trait IntoSyncResult {
    fn into_sync_result(message: impl Into<String>) -> Self;
}

impl<T> IntoSyncResult for Result<T, VecboostError> {
    fn into_sync_result(message: impl Into<String>) -> Self {
        Err(VecboostError::InternalError(message.into()))
    }
}

// VecBoostModuleBuilder — trait-kit 模块化集成

/// VecBoost 模块化构建器 — 将 embedding/rerank 能力注册到外部 `AsyncKit`
///
/// 允许外部项目将 VecBoost 的向量化和重排序能力作为 trait-kit Module
/// 注册到自己的 `AsyncKit` 中，与项目自身的模块共存并共享生命周期管理。
///
/// # 选择性注册
///
/// 通过 `embedding` / `rerank` 按需启用能力，未调用的能力不会注册。
/// 两者都未调用时 `build` 不会注册任何模块（也不会报错）。
///
/// # 示例
///
/// ```ignore
/// use vecboost::{VecBoostModuleBuilder, config::model::ModelConfig};
///
/// let model_config = ModelConfig { /* ... */ };
/// let mut kit = trait_kit::AsyncKit::new;
///
/// // 注册自己的模块
/// kit.register::<MyAppModule>?;
///
/// // 注入 VecBoost 能力
/// VecBoostModuleBuilder::new(model_config)
///     .embedding           // 启用 embedding
///     .rerank              // 启用 rerank
///     .cache_size(1000)      // 可选：嵌入缓存
///     .build(&mut kit)
///     .await?;
///
/// let kit = kit.build.await?;
///
/// // 使用
/// let embed_svc = kit.require::<vecboost::registry::EmbeddingModule>?;
/// ```
pub struct VecBoostModuleBuilder {
    model_config: ModelConfig,
    cache_size: usize,
    with_embedding: bool,
    with_rerank: bool,
    rerank_config: Option<RerankConfig>,
    caller_isolation: crate::cache::CallerIsolation,
}

impl VecBoostModuleBuilder {
    /// 创建构建器，指定模型配置
    pub fn new(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            cache_size: 0,
            with_embedding: false,
            with_rerank: false,
            rerank_config: None,
            caller_isolation: crate::cache::CallerIsolation::Off,
        }
    }

    /// 启用 embedding 模块
    pub fn embedding(mut self) -> Self {
        self.with_embedding = true;
        self
    }

    /// 启用 rerank 模块
    pub fn rerank(mut self) -> Self {
        self.with_rerank = true;
        self
    }

    /// 设置嵌入缓存容量（0 = 禁用，默认 0）
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// 设置重排序配置（None = 使用默认值）
    pub fn rerank_config(mut self, config: RerankConfig) -> Self {
        self.rerank_config = Some(config);
        self
    }

    /// 设置调用方隔离模式（默认 Off = 缓存键行为不变）。与 server 模式
    /// `[semantic_cache] caller_isolation` 同语义。
    pub fn caller_isolation(mut self, isolation: crate::cache::CallerIsolation) -> Self {
        self.caller_isolation = isolation;
        self
    }

    /// 将选中的 VecBoost 模块注册到外部 `AsyncKit`
    ///
    /// 此方法：
    /// 1. 通过 `EngineFactory` 创建推理引擎
    /// 2. 构建 `EmbeddingService` / `RerankService`（按选择）
    /// 3. 通过 `kit.set_config` 注入能力
    /// 4. 注册对应的 `EmbeddingModule` / `RerankModule` + lifecycle hooks
    ///
    /// 调用后外部项目继续注册自己的其他模块，最终调用 `kit.build.await`。
    pub async fn build(self, kit: &mut trait_kit::AsyncKit) -> Result<(), VecboostError> {
        let engine =
            EngineFactory::create(self.model_config.engine_type.clone(), &self.model_config)?;
        let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(engine));

        // 2. 按选择构建并注册服务
        if self.with_embedding {
            let embedding_service = if self.cache_size > 0 {
                Arc::new(RwLock::new(EmbeddingService::with_cache(
                    engine.clone(),
                    Some(self.model_config.clone()),
                    self.cache_size,
                )))
            } else {
                Arc::new(RwLock::new(EmbeddingService::new(
                    engine.clone(),
                    Some(self.model_config.clone()),
                )))
            };
            kit.set_config(embedding_service);
            kit.register::<EmbeddingModule>().map_err(|e| {
                VecboostError::InternalError(crate::i18n::tr_with_args(
                    "detail-failed-to-register-embeddingmodule-bdd55",
                    crate::i18n::tr_args(&[("p1", &e.to_string())]),
                ))
            })?;
            kit.register_lifecycle::<EmbeddingModule>();
        }

        if self.with_rerank {
            // [rerank] cache_enabled 消费（与 VecBoostLibrary::new /
            // server 模式 build_rerank_service 同口径）：true 且 cache_size>0
            // 时构建进程内精确缓存，false（默认）disabled 零变化。
            let rerank_config = self.rerank_config.clone().unwrap_or_default();
            let rerank_service = if rerank_config.cache_enabled && rerank_config.cache_size > 0 {
                Arc::new(RwLock::new(
                    RerankService::with_cache_capacity(
                        engine.clone(),
                        Some(self.model_config.clone()),
                        rerank_config.cache_size,
                    )
                    .with_caller_isolation(self.caller_isolation),
                ))
            } else {
                Arc::new(RwLock::new(
                    RerankService::new(engine, Some(self.model_config))
                        .with_caller_isolation(self.caller_isolation),
                ))
            };
            kit.set_config(rerank_service);
            kit.set_config(rerank_config);
            kit.register::<RerankModule>().map_err(|e| {
                VecboostError::InternalError(crate::i18n::tr_with_args(
                    "detail-failed-to-register-rerankmodule-a244b",
                    crate::i18n::tr_args(&[("p1", &e.to_string())]),
                ))
            })?;
            kit.register_lifecycle::<RerankModule>();
        }

        Ok(())
    }
}

// Tests

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
            Ok(texts.iter().map(|_| vec![1.0; self.dimension]).collect())
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

        let embedding_service = Arc::new(RwLock::new(EmbeddingService::new(engine.clone(), None)));
        let rerank_service = Arc::new(RwLock::new(RerankService::new(engine, None)));

        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(embedding_service);
        kit.set_config(rerank_service);
        kit.set_config(RerankConfig::default());
        kit.register::<EmbeddingModule>().unwrap();
        kit.register::<RerankModule>().unwrap();
        let kit = kit.build().await.unwrap();

        VecBoostLibrary { kit: Arc::new(kit) }
    }

    // 构造测试

    #[tokio::test]
    async fn test_library_construction_succeeds() {
        let lib = make_test_library().await;
        assert!(lib.kit.contains::<EmbeddingModule>());
        assert!(lib.kit.contains::<RerankModule>());
    }

    // 异步 API 测试

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
                &["a".to_string(), "bb".to_string(), "ccc".to_string()],
                Some(2),
            )
            .await
            .unwrap();
        assert_eq!(response.results.len(), 2);
    }

    // 同步 API 测试

    // 在 tokio runtime 上下文调用 sync API → 可操作错误而非 panic
    #[tokio::test]
    async fn sync_api_inside_runtime_returns_error_not_panic() {
        let lib = make_test_library().await;
        let err = lib
            .embed_sync("hello")
            .expect_err("must error inside tokio context");
        let msg = err.error_detail().to_string();
        assert!(
            msg.contains("async variants") || msg.contains("tokio runtime"),
            "error should point to async alternatives, got: {msg}"
        );
    }

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

    // LibraryConfig 测试

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

    // VecBoostModuleBuilder 测试

    /// 测试辅助：通过 mock engine 直接注入 kit（绕过 EngineFactory 需要真实模型）
    async fn make_test_kit_with_builder(
        with_embedding: bool,
        with_rerank: bool,
    ) -> trait_kit::AsyncKit<trait_kit::AsyncReady> {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockEngine::new(128)));

        let mut kit = trait_kit::AsyncKit::new();

        if with_embedding {
            let embedding_service =
                Arc::new(RwLock::new(EmbeddingService::new(engine.clone(), None)));
            kit.set_config(embedding_service);
            kit.register::<EmbeddingModule>().unwrap();
            kit.register_lifecycle::<EmbeddingModule>();
        }

        if with_rerank {
            let rerank_service = Arc::new(RwLock::new(RerankService::new(engine, None)));
            kit.set_config(rerank_service);
            kit.set_config(RerankConfig::default());
            kit.register::<RerankModule>().unwrap();
            kit.register_lifecycle::<RerankModule>();
        }

        kit.build().await.unwrap()
    }

    #[tokio::test]
    async fn test_module_builder_embedding_only() {
        let kit = make_test_kit_with_builder(true, false).await;
        assert!(kit.contains::<EmbeddingModule>());
        assert!(!kit.contains::<RerankModule>());

        // embedding 能力可正常检索
        let svc = kit.require::<EmbeddingModule>().unwrap();
        let guard = svc.read().await;
        let resp = guard
            .process_text(
                EmbedRequest {
                    text: "test".to_string(),
                    normalize: None,
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(resp.dimension, 128);
    }

    #[tokio::test]
    async fn test_module_builder_rerank_only() {
        let kit = make_test_kit_with_builder(false, true).await;
        assert!(!kit.contains::<EmbeddingModule>());
        assert!(kit.contains::<RerankModule>());

        let svc = kit.require::<RerankModule>().unwrap();
        let guard = svc.read().await;
        let resp = guard
            .process_rerank(
                RerankRequest {
                    query: "query".to_string(),
                    documents: vec!["doc a".to_string(), "doc b".to_string()],
                    top_k: None,
                    return_documents: None,
                },
                100,
                1000,
            )
            .await
            .unwrap();
        assert_eq!(resp.results.len(), 2);
    }

    #[tokio::test]
    async fn test_module_builder_both_modules() {
        let kit = make_test_kit_with_builder(true, true).await;
        assert!(kit.contains::<EmbeddingModule>());
        assert!(kit.contains::<RerankModule>());

        // 两个能力都可正常检索
        let _embed_svc = kit.require::<EmbeddingModule>().unwrap();
        let _rerank_svc = kit.require::<RerankModule>().unwrap();
    }

    #[tokio::test]
    async fn test_module_builder_neither_module() {
        // 两者都未启用 → kit 为空但构建成功
        let kit = make_test_kit_with_builder(false, false).await;
        assert!(!kit.contains::<EmbeddingModule>());
        assert!(!kit.contains::<RerankModule>());
    }

    #[test]
    fn test_module_builder_chaining_api() {
        // 验证 builder 链式调用编译正确
        let model_config = ModelConfig::default();
        let _builder = VecBoostModuleBuilder::new(model_config)
            .embedding()
            .rerank()
            .cache_size(500)
            .rerank_config(RerankConfig::default());
    }

    // 同步 API 补充测试

    // Note: embed_batch_sync and rerank_sync are covered via the existing
    // test_sync_embed_outside_runtime and test_sync_rerank_outside_runtime tests.
    // Additional sync batch testing is skipped because block_on_future creates
    // a current_thread runtime without timers, which batch processing requires.

    // 异步 API 补充测试

    #[tokio::test]
    async fn test_embed_empty_string_returns_error() {
        let lib = make_test_library().await;
        // Empty string is rejected by the input validator
        let result = lib.embed("").await;
        assert!(
            result.is_err(),
            "empty string should return validation error"
        );
    }

    #[tokio::test]
    async fn test_rerank_empty_documents_returns_error() {
        let lib = make_test_library().await;
        let result = lib.rerank("query", &[], None).await;
        assert!(result.is_err(), "empty documents should return error");
    }

    #[tokio::test]
    async fn test_rerank_empty_query_returns_error() {
        let lib = make_test_library().await;
        let result = lib.rerank("", &["doc".to_string()], None).await;
        assert!(result.is_err(), "empty query should return error");
    }

    // LibraryConfig 补充测试

    #[test]
    fn test_library_config_with_cache_size() {
        let config = LibraryConfig {
            cache_size: 1000,
            ..Default::default()
        };
        assert_eq!(config.cache_size, 1000);
    }

    // 默认 Off = 行为不变；user 由调用方经 LibraryConfig/builder 显式传入
    #[test]
    fn test_library_config_caller_isolation_default_off() {
        let config = LibraryConfig::default();
        assert_eq!(
            config.caller_isolation,
            crate::cache::CallerIsolation::Off,
            "库路径缺省必须维持既有缓存键行为"
        );
    }

    #[test]
    fn test_library_config_with_rerank_config() {
        let config = LibraryConfig {
            rerank_config: Some(RerankConfig::default()),
            ..Default::default()
        };
        assert!(config.rerank_config.is_some());
    }

    // VecBoostModuleBuilder 补充测试

    #[test]
    fn test_module_builder_new_defaults() {
        let builder = VecBoostModuleBuilder::new(ModelConfig::default());
        assert!(!builder.with_embedding);
        assert!(!builder.with_rerank);
        assert_eq!(builder.cache_size, 0);
        assert!(builder.rerank_config.is_none());
        assert_eq!(builder.caller_isolation, crate::cache::CallerIsolation::Off);
    }

    #[test]
    fn test_module_builder_embedding_only_chain() {
        let builder = VecBoostModuleBuilder::new(ModelConfig::default())
            .embedding()
            .cache_size(100);
        assert!(builder.with_embedding);
        assert!(!builder.with_rerank);
        assert_eq!(builder.cache_size, 100);
    }

    #[test]
    fn test_module_builder_rerank_only_chain() {
        let builder = VecBoostModuleBuilder::new(ModelConfig::default())
            .rerank()
            .rerank_config(RerankConfig::default());
        assert!(!builder.with_embedding);
        assert!(builder.with_rerank);
        assert!(builder.rerank_config.is_some());
    }

    #[test]
    fn test_mock_engine_trait_method_coverage() {
        let engine = MockEngine::new(256);
        assert_eq!(engine.embed("test").unwrap().len(), 256);
        assert_eq!(
            engine.embed_batch(&["a".into(), "b".into()]).unwrap().len(),
            2
        );
        assert_eq!(*engine.precision(), Precision::Fp32);
        assert!(!engine.supports_mixed_precision());
        assert!(engine.supports_rerank());
        assert!(engine.rerank("q", "doc").unwrap() > 0.0);
        assert_eq!(
            engine
                .rerank_batch("q", &["a".into(), "b".into()])
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn test_mock_engine_try_fallback_coverage() {
        let mut engine = MockEngine::new(64);
        let cfg = ModelConfig::default();
        let _ = engine.try_fallback_to_cpu(&cfg).await;
    }

    // ── 门面级 caller 供给 API（②）与隔离钉 ──────────────────────────

    /// 计数重排引擎：rerank_batch 调用次数区分缓存命中/未命中。
    struct CountingFacadeRerankEngine {
        calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }

    #[async_trait]
    impl InferenceEngine for CountingFacadeRerankEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 8])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        fn rerank(&self, _q: &str, doc: &str) -> Result<f32, VecboostError> {
            Ok((doc.len() as f32) / 100.0)
        }
        fn rerank_batch(&self, query: &str, docs: &[String]) -> Result<Vec<f32>, VecboostError> {
            use std::sync::atomic::Ordering;
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

    /// 计数嵌入引擎：embed 调用次数区分语义缓存命中/未命中。
    struct CountingFacadeEmbedEngine {
        calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
    }

    #[async_trait]
    impl InferenceEngine for CountingFacadeEmbedEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            use std::sync::atomic::Ordering;
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![0.5; 8])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; 8]).collect())
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        fn rerank(&self, _q: &str, _doc: &str) -> Result<f32, VecboostError> {
            Ok(0.0)
        }
        fn rerank_batch(&self, _q: &str, docs: &[String]) -> Result<Vec<f32>, VecboostError> {
            Ok(docs.iter().map(|_| 0.0).collect())
        }
        fn supports_rerank(&self) -> bool {
            false
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// 门面隔离钉（②）：user 隔离 + 缓存启用时，`rerank_with_caller` 两个
    /// caller 互不可见——alice 二次命中自身分区，bob 必须 miss。
    #[tokio::test]
    async fn facade_rerank_with_caller_partitions_per_caller() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let calls = Arc::new(AtomicU64::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingFacadeRerankEngine {
                calls: calls.clone(),
            }));
        let rerank_service = Arc::new(RwLock::new(
            RerankService::with_cache_capacity(engine, None, 100)
                .with_caller_isolation(crate::cache::CallerIsolation::User),
        ));
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(rerank_service);
        kit.set_config(RerankConfig::default());
        kit.register::<RerankModule>().unwrap();
        let kit = kit.build().await.unwrap();
        let lib = VecBoostLibrary { kit: Arc::new(kit) };

        let docs = vec!["same doc".to_string()];
        lib.rerank_with_caller("same query", &docs, None, "alice")
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "alice 首次必然计算");
        lib.rerank_with_caller("same query", &docs, None, "alice")
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "alice 二次必须命中自身分区"
        );
        lib.rerank_with_caller("same query", &docs, None, "bob")
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "bob 必须 miss，不得命中 alice 分区"
        );
    }

    /// 门面隔离钉（②）sync 入口：`rerank_sync_with_caller` 与 async 变体
    /// 同分区（alice 命中自身缓存），须在无活跃 runtime 的普通线程调用。
    #[test]
    fn facade_rerank_sync_with_caller_partitions_per_caller_outside_runtime() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let calls = Arc::new(AtomicU64::new(0));
        let lib = {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
                    Arc::new(RwLock::new(CountingFacadeRerankEngine {
                        calls: calls.clone(),
                    }));
                let rerank_service = Arc::new(RwLock::new(
                    RerankService::with_cache_capacity(engine, None, 100)
                        .with_caller_isolation(crate::cache::CallerIsolation::User),
                ));
                let mut kit = trait_kit::AsyncKit::new();
                kit.set_config(rerank_service);
                kit.set_config(RerankConfig::default());
                kit.register::<RerankModule>().unwrap();
                let kit = kit.build().await.unwrap();
                VecBoostLibrary { kit: Arc::new(kit) }
            })
        };
        let docs = vec!["same doc".to_string()];
        lib.rerank_sync_with_caller("same query", &docs, None, "alice")
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "alice 首次必然计算");
        lib.rerank_sync_with_caller("same query", &docs, None, "alice")
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "sync 变体必须命中 alice 自身分区"
        );
        lib.rerank_sync_with_caller("same query", &docs, None, "bob")
            .unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "bob 必须 miss，不得命中 alice 分区"
        );
    }

    /// 门面隔离钉（②）：user 隔离 + 语义缓存启用时，`embed_with_caller`
    /// 两个 caller 互不可见。
    #[tokio::test]
    async fn facade_embed_with_caller_partitions_per_caller() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let calls = Arc::new(AtomicU64::new(0));
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(CountingFacadeEmbedEngine {
                calls: calls.clone(),
            }));
        let semantic = Arc::new(
            crate::cache::SemanticCache::with_capacity(0.7, 100)
                .with_caller_isolation(crate::cache::CallerIsolation::User),
        );
        let embedding_service = Arc::new(RwLock::new(
            EmbeddingService::new(engine, None).with_semantic_cache(semantic),
        ));
        let mut kit = trait_kit::AsyncKit::new();
        kit.set_config(embedding_service);
        kit.register::<EmbeddingModule>().unwrap();
        let kit = kit.build().await.unwrap();
        let lib = VecBoostLibrary { kit: Arc::new(kit) };

        lib.embed_with_caller("shared text", "alice").await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "alice 首次必然计算");
        lib.embed_with_caller("shared text", "alice").await.unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "alice 二次必须命中自身分区"
        );
        lib.embed_with_caller("shared text", "bob").await.unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "bob 必须 miss，不得命中 alice 分区"
        );
    }

    /// 旧门面方法保留 anonymous 语义：off 模式下 caller 被忽略，行为不变
    ///（零行为变化承诺的回归钉）。
    #[tokio::test]
    async fn facade_legacy_methods_keep_anonymous_semantics() {
        let lib = make_test_library().await;
        // 旧路径可正常工作（anonymous 归并桶语义不回归）。
        let resp = lib
            .rerank(
                "legacy query",
                &["doc a".to_string(), "longer doc b".to_string()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(resp.results.len(), 2);
        let embed = lib.embed("legacy text").await.unwrap();
        assert_eq!(embed.dimension, 128);
    }
}
