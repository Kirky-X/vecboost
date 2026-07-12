// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 路由模块
//!
//! 本模块包含所有 API 路由定义，将路由从 main.rs 中分离出来以提高可维护性。

// 内部子模块 - 只在 routes 模块内使用，不暴露给外部
pub(crate) mod embedding;
pub(crate) mod health;
pub(crate) mod impl_;
pub(crate) mod openai_embedding;

use std::time::Duration;
use utoipa::OpenApi;

/// Default request timeout (30 seconds)
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// VecBoost API OpenAPI 文档
///
/// 注意：此结构体仅在 crate 内部使用，不暴露给外部库
#[derive(OpenApi)]
#[openapi(
    info(
        title = "VecBoost API",
        version = "0.1.0",
        description = "高性能向量嵌入服务 API 文档",
        contact(
            name = "VecBoost Team",
            email = "support@vecboost.io"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    ),
    paths(
        health::health_check,
        health::metrics_endpoint,
        embedding::embed_handler,
        embedding::batch_embed_handler,
        embedding::similarity_handler,
        embedding::file_embed_handler,
    ),
    components(
        schemas(
            crate::domain::EmbedRequest,
            crate::domain::EmbedResponse,
            crate::domain::SimilarityRequest,
            crate::domain::SimilarityResponse,
            crate::domain::SearchRequest,
            crate::domain::SearchResponse,
            crate::domain::SearchResult,
            crate::domain::ParagraphEmbedding,
            crate::domain::EmbeddingOutput,
            crate::domain::FileProcessingStats,
            crate::domain::FileEmbedRequest,
            crate::domain::FileEmbedResponse,
            crate::domain::BatchEmbedRequest,
            crate::domain::BatchEmbedResponse,
            crate::domain::BatchEmbeddingResult,
            crate::domain::ModelInfo,
            crate::domain::ModelMetadata,
            crate::domain::ModelListResponse,
        )
    ),
    servers(
        (url = "http://localhost:9000", description = "本地开发服务器")
    ),
    tags(
        (name = "health", description = "健康检查"),
        (name = "auth", description = "认证"),
        (name = "embedding", description = "嵌入"),
        (name = "model", description = "模型管理")
    )
)]
pub(crate) struct ApiDoc;

pub use impl_::create_router;
