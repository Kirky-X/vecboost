// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    routing::{get, post},
    Extension, Json, Router,
};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tower_http::trace::TraceLayer;
use vecboost::{
    config::model::ModelConfig,
    domain::{BatchEmbedRequest, EmbedRequest, FileEmbedRequest, SimilarityRequest},
    engine::candle_engine::CandleEngine,
    service::embedding::EmbeddingService,
    utils::AggregationMode,
    AppConfig,
};

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
        engine_type: vecboost::config::model::EngineType::Candle,
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
    let engine: Arc<RwLock<dyn vecboost::engine::InferenceEngine + Send + Sync>> =
        Arc::new(RwLock::new(CandleEngine::new(&model_config)?));
    let service = Arc::new(EmbeddingService::new(engine, Some(model_config)));

    // 4. 构建路由
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/api/v1/embed", post(embed_handler))
        .route("/api/v1/embed/batch", post(batch_embed_handler))
        .route("/api/v1/similarity", post(similarity_handler))
        .route("/api/v1/embed/file", post(file_embed_handler))
        .layer(TraceLayer::new_for_http())
        .layer(Extension(service));

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
    Extension(service): Extension<Arc<EmbeddingService>>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<vecboost::domain::EmbedResponse>, vecboost::error::AppError> {
    let res = service.process_text(req).await?;
    Ok(Json(res))
}

async fn batch_embed_handler(
    Extension(service): Extension<Arc<EmbeddingService>>,
    Json(req): Json<BatchEmbedRequest>,
) -> Result<Json<vecboost::domain::BatchEmbedResponse>, vecboost::error::AppError> {
    let res = service.process_batch(req).await?;
    Ok(Json(res))
}

async fn similarity_handler(
    Extension(service): Extension<Arc<EmbeddingService>>,
    Json(req): Json<SimilarityRequest>,
) -> Result<Json<vecboost::domain::SimilarityResponse>, vecboost::error::AppError> {
    let res = service.process_similarity(req).await?;
    Ok(Json(res))
}

async fn file_embed_handler(
    Extension(service): Extension<Arc<EmbeddingService>>,
    Json(req): Json<FileEmbedRequest>,
) -> Result<Json<vecboost::domain::FileEmbedResponse>, vecboost::error::AppError> {
    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    let stats = service.get_processing_stats(&path)?;
    let output = service.embed_file(&path, mode).await?;

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
