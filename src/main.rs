// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use vecboost::{
    config::model::{EngineType, ModelConfig},
    domain::{
        BatchEmbedRequest, EmbedRequest, FileEmbedRequest, ModelInfo, ModelListResponse,
        ModelSwitchRequest, SimilarityRequest,
    },
    engine::AnyEngine,
    service::embedding::EmbeddingService,
    utils::AggregationMode,
    AppConfig,
};

#[derive(Clone)]
struct AppState {
    service: Arc<RwLock<EmbeddingService>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 初始化日志
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Rust Embedding Service...");

    // 2. 加载配置
    let config = AppConfig::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    // 3. 创建模型配置用于维度验证
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

    // 4. 初始化推理引擎
    tracing::info!("Initializing Inference Engine (this may take a while to download models)...");
    let engine: Arc<RwLock<AnyEngine>> = Arc::new(RwLock::new(AnyEngine::new(&model_config, EngineType::Candle)?));
    let service = EmbeddingService::new(engine, Some(model_config));
    let service = Arc::new(RwLock::new(service));

    let app_state = AppState { service };

    // 4. 构建路由
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/embed", post(embed_handler))
        .route("/api/v1/embed/batch", post(batch_embed_handler))
        .route("/api/v1/similarity", post(similarity_handler))
        .route("/api/v1/embed/file", post(file_embed_handler))
        .route("/api/v1/model/switch", post(model_switch_handler))
        .route("/api/v1/model/current", get(current_model_handler))
        .route("/api/v1/model/list", get(model_list_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    // 5. 启动服务
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
    let service_guard = state.service.read().await;
    let res = service_guard.process_text(req).await?;
    Ok(Json(res))
}

async fn batch_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<BatchEmbedRequest>,
) -> Result<Json<vecboost::domain::BatchEmbedResponse>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let res = service_guard.process_batch(req).await?;
    Ok(Json(res))
}

async fn similarity_handler(
    State(state): State<AppState>,
    Json(req): Json<SimilarityRequest>,
) -> Result<Json<vecboost::domain::SimilarityResponse>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let res = service_guard.process_similarity(req).await?;
    Ok(Json(res))
}

async fn file_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<FileEmbedRequest>,
) -> Result<Json<vecboost::domain::FileEmbedResponse>, vecboost::error::AppError> {
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


async fn model_list_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelListResponse>, vecboost::error::AppError> {
    let service_guard = state.service.read().await;
    let result = service_guard.list_available_models();
    Ok(Json(result))
}
