// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    body::Body,
    extract::State,
    routing::{get, post},
    Json, Router,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::sync::RwLock;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use vecboost::{
    config::model::{EngineType, ModelConfig},
    domain::{
        BatchEmbedRequest, EmbedRequest, FileEmbedRequest, ModelInfo, ModelListResponse,
        ModelMetadata, ModelSwitchRequest, SimilarityRequest,
    },
    engine::AnyEngine,
    service::embedding::EmbeddingService,
    utils::AggregationMode,
    AppConfig,
};

const RATE_LIMIT_REQUESTS: u64 = 100;
const RATE_LIMIT_WINDOW_SECS: u64 = 60;

#[derive(Clone)]
struct RateLimitState {
    requests: Arc<AtomicU64>,
    window_start: Arc<AtomicU64>,
}

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<EmbeddingService>>,
    rate_limit: RateLimitState,
}

impl RateLimitState {
    fn new() -> Self {
        Self {
            requests: Arc::new(AtomicU64::new(0)),
            window_start: Arc::new(AtomicU64::new(0)),
        }
    }

    async fn check_rate_limit(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let window_start = self.window_start.load(Ordering::Relaxed);
        if now - window_start > RATE_LIMIT_WINDOW_SECS {
            self.window_start.store(now, Ordering::Relaxed);
            self.requests.store(0, Ordering::Relaxed);
        }

        let requests = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        requests <= RATE_LIMIT_REQUESTS
    }
}

#[allow(dead_code)]
async fn rate_limit_middleware(
    State(state): State<RateLimitState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, std::convert::Infallible> {
    if !state.check_rate_limit().await {
        return Ok(axum::response::Response::builder()
            .status(429)
            .body(Body::from("Too Many Requests"))
            .unwrap());
    }
    Ok(next.run(request).await)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Rust Embedding Service...");

    let config = AppConfig::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    let model_config = ModelConfig {
        name: config.model.model_repo.clone(),
        engine_type: EngineType::Candle,
        model_path: std::path::PathBuf::from(&config.model.model_repo),
        tokenizer_path: None,
        device: if config.model.use_gpu {
            vecboost::config::model::DeviceType::Cuda
        } else {
            vecboost::config::model::DeviceType::Cpu
        },
        max_batch_size: config.model.batch_size,
        pooling_mode: None,
        expected_dimension: config.model.expected_dimension,
    };

    tracing::info!("Initializing Inference Engine (this may take a while to download models)...");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(AnyEngine::new(
        &model_config,
        EngineType::Candle,
        vecboost::config::model::Precision::Fp32,
    )?));
    let service = EmbeddingService::new(engine, Some(model_config));
    let service = Arc::new(RwLock::new(service));

    let rate_limit_state = RateLimitState::new();
    let app_state = AppState {
        service,
        rate_limit: rate_limit_state.clone(),
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/embed", post(embed_handler))
        .route("/api/v1/embed/batch", post(batch_embed_handler))
        .route("/api/v1/similarity", post(similarity_handler))
        .route("/api/v1/embed/file", post(file_embed_handler))
        .route("/api/v1/model/switch", post(model_switch_handler))
        .route("/api/v1/model/current", get(current_model_handler))
        .route("/api/v1/model/info", get(model_info_handler))
        .route("/api/v1/model/list", get(model_list_handler))
        .layer(TraceLayer::new_for_http())
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_CONTENT_TYPE_OPTIONS,
            axum::http::HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_FRAME_OPTIONS,
            axum::http::HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::X_XSS_PROTECTION,
            axum::http::HeaderValue::from_static("1; mode=block"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            axum::http::header::STRICT_TRANSPORT_SECURITY,
            axum::http::HeaderValue::from_static("max-age=31536000; includeSubDomains"),
        ))
        .with_state(app_state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> &'static str {
    "OK"
}

async fn embed_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<vecboost::domain::EmbedResponse>, vecboost::error::AppError> {
    if !state.rate_limit.check_rate_limit().await {
        return Err(vecboost::error::AppError::InvalidInput(
            "Rate limit exceeded".to_string(),
        ));
    }
    let service_guard = state.service.read().await;
    let res = service_guard.process_text(req).await?;
    Ok(Json(res))
}

async fn batch_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<BatchEmbedRequest>,
) -> Result<Json<vecboost::domain::BatchEmbedResponse>, vecboost::error::AppError> {
    if !state.rate_limit.check_rate_limit().await {
        return Err(vecboost::error::AppError::InvalidInput(
            "Rate limit exceeded".to_string(),
        ));
    }
    let service_guard = state.service.read().await;
    let res = service_guard.process_batch(req).await?;
    Ok(Json(res))
}

async fn similarity_handler(
    State(state): State<AppState>,
    Json(req): Json<SimilarityRequest>,
) -> Result<Json<vecboost::domain::SimilarityResponse>, vecboost::error::AppError> {
    if !state.rate_limit.check_rate_limit().await {
        return Err(vecboost::error::AppError::InvalidInput(
            "Rate limit exceeded".to_string(),
        ));
    }
    let service_guard = state.service.read().await;
    let res = service_guard.process_similarity(req).await?;
    Ok(Json(res))
}

async fn file_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<FileEmbedRequest>,
) -> Result<Json<vecboost::domain::FileEmbedResponse>, vecboost::error::AppError> {
    if !state.rate_limit.check_rate_limit().await {
        return Err(vecboost::error::AppError::InvalidInput(
            "Rate limit exceeded".to_string(),
        ));
    }
    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    let service_guard = state.service.read().await;
    let stats = service_guard.get_processing_stats(&path)?;
    let output = service_guard.embed_file(&path, mode).await?;

    drop(service_guard);

    match output {
        vecboost::domain::EmbeddingOutput::Single(response) => {
            Ok(Json(vecboost::domain::FileEmbedResponse {
                mode,
                stats,
                embedding: Some(response.embedding),
                paragraphs: None,
            }))
        }
        vecboost::domain::EmbeddingOutput::Paragraphs(paragraphs) => {
            Ok(Json(vecboost::domain::FileEmbedResponse {
                mode,
                stats,
                embedding: None,
                paragraphs: Some(paragraphs),
            }))
        }
    }
}

async fn model_switch_handler(
    State(state): State<AppState>,
    Json(req): Json<ModelSwitchRequest>,
) -> Result<Json<vecboost::domain::ModelSwitchResponse>, vecboost::error::AppError> {
    let mut service_guard = state.service.write().await;
    let result = service_guard.switch_model(req).await?;
    Ok(Json(result))
}

async fn current_model_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelInfo>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let info = service_guard
        .get_model_info()
        .ok_or_else(|| vecboost::error::AppError::NotFound("No model loaded".to_string()))?;
    Ok(Json(info))
}

async fn model_info_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelMetadata>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let metadata = service_guard
        .get_model_metadata()
        .ok_or_else(|| vecboost::error::AppError::NotFound("No model loaded".to_string()))?;
    Ok(Json(metadata))
}

async fn model_list_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelListResponse>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let result = service_guard.list_available_models();
    Ok(Json(result))
}
