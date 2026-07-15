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

use crate::VecboostState;

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
pub fn create_router(app_state: VecboostState) -> Router {
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

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use crate::error::VecboostError;
    use crate::pipeline::{
        PriorityCalculator, PriorityConfig, PriorityRequestQueue, ResponseChannel, WorkerConfig,
        WorkerManager,
    };
    use crate::rate_limit::LimiteronAdapter;
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    /// Deterministic mock engine that returns a fixed-dimension vector.
    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for TestEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
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

    /// Build an `Arc<UserStore>` for testing (conditional on `db` feature).
    #[cfg(all(feature = "db", feature = "auth"))]
    async fn make_user_store_arc() -> Arc<crate::auth::UserStore> {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_routes.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = crate::db::DbPool::new(&url).await.unwrap();
        crate::db::init_schema(&pool).await.unwrap();
        std::mem::forget(temp_dir);
        Arc::new(crate::auth::UserStore::new(Arc::new(pool)))
    }

    #[cfg(all(not(feature = "db"), feature = "auth"))]
    async fn make_user_store_arc() -> Arc<crate::auth::UserStore> {
        Arc::new(crate::auth::UserStore::new())
    }

    /// Build a minimal `VecboostState` for testing, with auth optionally enabled.
    async fn make_test_app_state(auth_enabled: bool) -> VecboostState {
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
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            engine,
            Some(model_config),
        )));

        // Keep temp dir alive for the test duration
        std::mem::forget(temp_dir);

        let rate_limiter = Arc::new(LimiteronAdapter::with_default_config());
        let pipeline_queue = Arc::new(PriorityRequestQueue::new(0));
        let response_channel = Arc::new(ResponseChannel::new());
        let priority_calculator = Arc::new(PriorityCalculator::new(PriorityConfig::default()));
        let worker_manager = Arc::new(WorkerManager::new(
            Arc::new(PriorityRequestQueue::new(0)),
            response_channel.clone(),
            WorkerConfig::default(),
            service.clone(),
        ));

        #[cfg(feature = "auth")]
        {
            let jwt_secret = "test_secret_key_for_router_tests_must_be_long_enough_abcdef123456";
            let jwt_manager = if auth_enabled {
                Some(Arc::new(
                    crate::auth::JwtManager::new(jwt_secret.to_string()).unwrap(),
                ))
            } else {
                None
            };
            let user_store = if auth_enabled {
                Some(make_user_store_arc().await)
            } else {
                None
            };

            VecboostState {
                service,
                jwt_manager,
                user_store,
                auth_enabled,
                csrf_config: None,
                csrf_token_store: None,
                metrics_collector: None,
                prometheus_collector: None,
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator,
                worker_manager,
                kit: None,
            }
        }

        #[cfg(not(feature = "auth"))]
        {
            let _ = auth_enabled;
            VecboostState {
                service,
                auth_enabled: false,
                metrics_collector: None,
                prometheus_collector: None,
                rate_limiter,
                ip_whitelist: vec![],
                rate_limit_enabled: false,
                audit_logger: None,
                pipeline_enabled: false,
                pipeline_queue,
                response_channel,
                priority_calculator,
                worker_manager,
                kit: None,
            }
        }
    }

    fn build_request(method: Method, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn test_create_openapi_returns_valid_doc() {
        let openapi = create_openapi();
        let info = openapi.info.clone();
        assert_eq!(info.title, "VecBoost API");
        assert!(!info.version.is_empty());
    }

    #[tokio::test]
    async fn test_create_router_no_auth_health_endpoint() {
        let state = make_test_app_state(false).await;
        let app = create_router(state);

        let response = app
            .oneshot(build_request(Method::GET, "/health"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_router_no_auth_embed_route_exists() {
        let state = make_test_app_state(false).await;
        let app = create_router(state);

        // Sending an empty body should produce a 400 (bad JSON) rather than 404,
        // proving the route is registered.
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/embed")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_router_no_auth_openai_route_exists() {
        let state = make_test_app_state(false).await;
        let app = create_router(state);

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_router_with_auth_protects_embed_endpoint() {
        let state = make_test_app_state(true).await;
        let app = create_router(state);

        // POST /api/v1/embed without Authorization header should be blocked
        // by auth_middleware with 401 Unauthorized.
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/embed")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_router_with_auth_login_endpoint_exists() {
        let state = make_test_app_state(true).await;
        let app = create_router(state);

        // POST /api/v1/auth/login with empty body should produce a non-404 response,
        // proving the auth route is registered.
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/auth/login")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_router_with_auth_openai_route_public() {
        let state = make_test_app_state(true).await;
        let app = create_router(state);

        // OpenAI-compatible route should remain public even when auth is enabled.
        // An empty-body request should produce 400 (bad request) not 401 (unauthorized),
        // proving the auth middleware is NOT applied to /v1/embeddings.
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_router_with_csrf_origin_only() {
        // Build state with CSRF config (Origin validation only, no token validation)
        let jwt_secret = "test_secret_key_for_csrf_origin_tests_long_enough_abc";
        let csrf_config = Arc::new(crate::auth::CsrfConfig::new(vec![
            "https://example.com".to_string(),
        ]));

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
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            engine,
            Some(model_config),
        )));
        std::mem::forget(temp_dir);

        let state = VecboostState {
            service,
            jwt_manager: Some(Arc::new(
                crate::auth::JwtManager::new(jwt_secret.to_string()).unwrap(),
            )),
            user_store: Some(make_user_store_arc().await),
            auth_enabled: true,
            csrf_config: Some(csrf_config),
            csrf_token_store: None,
            metrics_collector: None,
            prometheus_collector: None,
            rate_limiter: Arc::new(LimiteronAdapter::with_default_config()),
            ip_whitelist: vec![],
            rate_limit_enabled: false,
            audit_logger: None,
            pipeline_enabled: false,
            pipeline_queue: Arc::new(PriorityRequestQueue::new(0)),
            response_channel: Arc::new(ResponseChannel::new()),
            priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
            worker_manager: Arc::new(WorkerManager::new(
                Arc::new(PriorityRequestQueue::new(0)),
                Arc::new(ResponseChannel::new()),
                WorkerConfig::default(),
                Arc::new(RwLock::new(EmbeddingService::new(
                    Arc::new(RwLock::new(TestEngine::new(384))),
                    None,
                ))),
            )),
            kit: None,
        };

        // Router should build successfully with CSRF Origin-only config
        let app = create_router(state);

        // POST /api/v1/embed without Origin → 400 (CSRF middleware is the outer
        // layer and runs before auth_middleware, so it rejects first)
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/embed")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_router_with_csrf_token_validation() {
        // Build state with CSRF config (both Origin + token validation enabled)
        let jwt_secret = "test_secret_key_for_csrf_token_tests_long_enough_xyz";
        let csrf_config = Arc::new(
            crate::auth::CsrfConfig::new(vec!["https://example.com".to_string()])
                .with_token_validation(true),
        );
        let csrf_token_store = Arc::new(crate::auth::CsrfTokenStore::new());

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
        let service = Arc::new(RwLock::new(EmbeddingService::new(
            engine,
            Some(model_config),
        )));
        std::mem::forget(temp_dir);

        let state = VecboostState {
            service,
            jwt_manager: Some(Arc::new(
                crate::auth::JwtManager::new(jwt_secret.to_string()).unwrap(),
            )),
            user_store: Some(make_user_store_arc().await),
            auth_enabled: true,
            csrf_config: Some(csrf_config),
            csrf_token_store: Some(csrf_token_store),
            metrics_collector: None,
            prometheus_collector: None,
            rate_limiter: Arc::new(LimiteronAdapter::with_default_config()),
            ip_whitelist: vec![],
            rate_limit_enabled: false,
            audit_logger: None,
            pipeline_enabled: false,
            pipeline_queue: Arc::new(PriorityRequestQueue::new(0)),
            response_channel: Arc::new(ResponseChannel::new()),
            priority_calculator: Arc::new(PriorityCalculator::new(PriorityConfig::default())),
            worker_manager: Arc::new(WorkerManager::new(
                Arc::new(PriorityRequestQueue::new(0)),
                Arc::new(ResponseChannel::new()),
                WorkerConfig::default(),
                Arc::new(RwLock::new(EmbeddingService::new(
                    Arc::new(RwLock::new(TestEngine::new(384))),
                    None,
                ))),
            )),
            kit: None,
        };

        // Router should build successfully with CSRF token validation config
        let app = create_router(state);

        // POST /api/v1/embed without Origin → 400 (CSRF combined middleware is
        // the outer layer and runs before auth_middleware, so it rejects first)
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api/v1/embed")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
