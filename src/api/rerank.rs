// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Rerank forge handlers — HTTP/CLI/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/embedding.rs` 相同模式：协议无关的 `*_handler` 函数包含
//! 业务逻辑，`forge_*` / `cli_*` / `grpc_*` 仅为薄包装。

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
use crate::domain::{
    BatchRerankRequest, BatchRerankResponse, RerankRequest, RerankResponse,
};
use crate::error::VecboostError;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::module_registry::RerankModule;
use crate::service::rerank::RerankService;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use std::sync::Arc;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use tokio::sync::RwLock;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions
// =============================================================================

pub async fn rerank(
    svc: &RerankService,
    req: RerankRequest,
    max_documents: usize,
    max_query_length: usize,
) -> Result<RerankResponse, VecboostError> {
    svc.process_rerank(req, max_documents, max_query_length).await
}

pub async fn rerank_batch(
    svc: &RerankService,
    req: BatchRerankRequest,
    max_documents: usize,
    max_query_length: usize,
) -> Result<BatchRerankResponse, VecboostError> {
    let mut responses = Vec::with_capacity(req.queries.len());
    for q in req.queries {
        match svc.process_rerank(q, max_documents, max_query_length).await {
            Ok(resp) => responses.push(resp),
            Err(e) => {
                log::warn!("Batch rerank: individual query failed, skipping: {}", e);
                // 单个 query 失败不影响其他 query 的处理
            }
        }
    }
    Ok(BatchRerankResponse { responses })
}

// =============================================================================
// Protocol-agnostic handlers
// =============================================================================

/// Load rerank service and limits from the global kit.
///
/// Returns the `Arc<RwLock<RerankService>>` capability together with the
/// configured `max_documents_per_query` and `max_query_length` limits so
/// callers only need to acquire a read-guard and dispatch.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn load_rerank_service(
) -> Result<(Arc<RwLock<RerankService>>, usize, usize), ApiError> {
    let st = state().map_err(to_api_error)?;
    let rerank_config = st
        .kit
        .config::<crate::config::app::RerankConfig>()
        .unwrap_or_default();
    let svc = st
        .kit
        .require::<RerankModule>()
        .map_err(kit_internal_error)?;
    Ok((
        svc,
        rerank_config.max_documents_per_query,
        rerank_config.max_query_length,
    ))
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn rerank_handler(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    let (svc, max_documents, max_query_length) = load_rerank_service().await?;
    let guard = svc.read().await;
    rerank(&guard, req, max_documents, max_query_length)
        .await
        .map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn rerank_batch_handler(
    req: BatchRerankRequest,
) -> Result<BatchRerankResponse, ApiError> {
    let (svc, max_documents, max_query_length) = load_rerank_service().await?;
    let guard = svc.read().await;
    rerank_batch(&guard, req, max_documents, max_query_length)
        .await
        .map_err(to_api_error)
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "rerank",
    version = 1,
    path = "/rerank",
    method = "POST",
    tool_name = "rerank",
    description = "Rerank documents by relevance to a query"
)]
pub async fn forge_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "rerank_batch",
    version = 1,
    path = "/rerank/batch",
    method = "POST",
    tool_name = "rerank_batch",
    description = "Batch rerank multiple queries against document sets"
)]
pub async fn forge_rerank_batch(
    req: BatchRerankRequest,
) -> Result<BatchRerankResponse, ApiError> {
    rerank_batch_handler(req).await
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "rerank",
    version = 1,
    cli = true,
    description = "Rerank documents by relevance to a query"
)]
pub async fn cli_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

// =============================================================================
// gRPC forge handlers
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "rerank",
    version = 1,
    grpc_method = "vecboost.rerank",
    description = "Rerank documents by relevance to a query"
)]
pub async fn grpc_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "rerank_batch",
    version = 1,
    grpc_method = "vecboost.rerank_batch",
    description = "Batch rerank multiple queries against document sets"
)]
pub async fn grpc_rerank_batch(
    req: BatchRerankRequest,
) -> Result<BatchRerankResponse, ApiError> {
    rerank_batch_handler(req).await
}
