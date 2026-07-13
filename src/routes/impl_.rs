// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 路由实现块

#[cfg(feature = "auth")]
use axum::middleware;
use axum::{Router, routing::get, routing::post};
use tower_http::timeout::TimeoutLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::AppState;

use super::{ApiDoc, DEFAULT_TIMEOUT};

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
        .route("/health", get(super::health::health_check))
        .route("/metrics", get(super::health::metrics_endpoint))
        .merge(SwaggerUi::new("/api-docs").url("/api-docs/openapi.json", openapi))
        .with_state(app_state.clone());

    #[cfg(feature = "auth")]
    {
        if app_state.auth_enabled {
            // Authentication routes (with timeout)
            let auth_routes = Router::new()
                .route("/api/v1/auth/login", post(crate::auth::login_handler))
                .route("/api/v1/auth/logout", post(crate::auth::logout_handler))
                .route(
                    "/api/v1/auth/refresh",
                    post(crate::auth::refresh_token_handler),
                )
                .layer(
                    #[allow(deprecated)]
                    TimeoutLayer::new(DEFAULT_TIMEOUT),
                )
                .with_state(app_state.clone());

            // Protected routes (require authentication, with timeout)
            let protected_routes = Router::new()
                .route("/api/v1/embed", post(super::embedding::embed_handler))
                .route(
                    "/api/v1/embed/batch",
                    post(super::embedding::batch_embed_handler),
                )
                .route(
                    "/api/v1/similarity",
                    post(super::embedding::similarity_handler),
                )
                .route(
                    "/api/v1/embed/file",
                    post(super::embedding::file_embed_handler),
                )
                .with_state(app_state.clone())
                .layer(middleware::from_fn_with_state(
                    app_state.clone(),
                    crate::auth::auth_middleware,
                ))
                .layer(
                    #[allow(deprecated)]
                    TimeoutLayer::new(DEFAULT_TIMEOUT),
                );

            // OpenAI compatible routes (always public, like OpenAI API)
            let openai_routes = Router::new()
                .route(
                    "/v1/embeddings",
                    post(super::openai_embedding::openai_embed_handler),
                )
                .layer(
                    #[allow(deprecated)]
                    TimeoutLayer::new(DEFAULT_TIMEOUT),
                )
                .with_state(app_state.clone());

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
            app = app
                .merge(auth_routes)
                .merge(protected_routes)
                .merge(openai_routes);
            return app.with_state(app_state);
        }
    }

    // No authentication mode - all routes public (with timeout)
    app = app
        .route("/api/v1/embed", post(super::embedding::embed_handler))
        .route(
            "/api/v1/embed/batch",
            post(super::embedding::batch_embed_handler),
        )
        .route(
            "/api/v1/similarity",
            post(super::embedding::similarity_handler),
        )
        .route(
            "/api/v1/embed/file",
            post(super::embedding::file_embed_handler),
        )
        .layer(
            #[allow(deprecated)]
            TimeoutLayer::new(DEFAULT_TIMEOUT),
        )
        // OpenAI compatible routes (always public)
        .route(
            "/v1/embeddings",
            post(super::openai_embedding::openai_embed_handler),
        );

    app.with_state(app_state)
}
