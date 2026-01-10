// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 路由模块
//!
//! 本模块包含所有 API 路由定义，将路由从 main.rs 中分离出来以提高可维护性。

// 内部子模块 - 只在 routes 模块内使用，不暴露给外部
pub(crate) mod embedding;
pub(crate) mod health;

use axum::{Router, middleware, routing::get, routing::post};
use tower_http::timeout::TimeoutLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::AppState;
use std::time::Duration;

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

/// 创建 OpenAPI 配置
///
/// 注意：此函数仅在 crate 内部使用，不暴露给外部库
pub(crate) fn create_openapi() -> utoipa::openapi::OpenApi {
    ApiDoc::openapi()
}

/// Create unified router
///
/// Creates different route combinations based on authentication configuration:
/// - If authentication is enabled, separate auth routes and protected routes
/// - If authentication is disabled, all routes are public
///
/// Note: This function is exposed to main.rs and other crates using this library
pub fn create_router(app_state: AppState) -> Router {
    let openapi = create_openapi();

    // Add ConnectInfo middleware to capture client IP addresses
    // This is required for IP-based rate limiting
    let mut app = Router::new()
        .route("/health", get(health::health_check))
        .route("/metrics", get(health::metrics_endpoint))
        .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", openapi))
        .with_state(app_state.clone());

    if app_state.auth_enabled {
        // Authentication routes (with timeout)
        let auth_routes = Router::new()
            .route("/api/v1/auth/login", post(crate::auth::login_handler))
            .route("/api/v1/auth/logout", post(crate::auth::logout_handler))
            .route(
                "/api/v1/auth/refresh",
                post(crate::auth::refresh_token_handler),
            )
            .layer(TimeoutLayer::new(DEFAULT_TIMEOUT))
            .with_state(app_state.clone());

        // Protected routes (require authentication, with timeout)
        let protected_routes = Router::new()
            .route("/api/v1/embed", post(embedding::embed_handler))
            .route("/api/v1/embed/batch", post(embedding::batch_embed_handler))
            .route("/api/v1/similarity", post(embedding::similarity_handler))
            .route("/api/v1/embed/file", post(embedding::file_embed_handler))
            .with_state(app_state.clone())
            .layer(middleware::from_fn_with_state(
                app_state.clone(),
                crate::auth::auth_middleware,
            ))
            .layer(TimeoutLayer::new(DEFAULT_TIMEOUT));

        // Apply CSRF protection (if enabled)
        let protected_routes = if let (Some(csrf_config), Some(csrf_token_store)) = (
            app_state.csrf_config.clone(),
            app_state.csrf_token_store.clone(),
        ) {
            // If CSRF token validation is enabled, use combined middleware
            if csrf_config.token_validation_enabled {
                protected_routes.layer(middleware::from_fn_with_state(
                    (csrf_config, csrf_token_store),
                    crate::auth::middleware::csrf_combined_middleware,
                ))
            } else {
                // Otherwise use only Origin validation
                protected_routes.layer(middleware::from_fn_with_state(
                    csrf_config,
                    crate::auth::csrf_origin_middleware,
                ))
            }
        } else if let Some(csrf_config) = app_state.csrf_config.clone() {
            // Only Origin validation
            protected_routes.layer(middleware::from_fn_with_state(
                csrf_config,
                crate::auth::csrf_origin_middleware,
            ))
        } else {
            protected_routes
        };

        // Merge routes
        app = app.merge(auth_routes).merge(protected_routes);
    } else {
        // No authentication mode - all routes public (with timeout)
        app = app
            .route("/api/v1/embed", post(embedding::embed_handler))
            .route("/api/v1/embed/batch", post(embedding::batch_embed_handler))
            .route("/api/v1/similarity", post(embedding::similarity_handler))
            .route("/api/v1/embed/file", post(embedding::file_embed_handler))
            .layer(TimeoutLayer::new(DEFAULT_TIMEOUT))
    }

    app.with_state(app_state)
}
