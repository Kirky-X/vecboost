// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Multi-protocol API layer — sdforge integration
//!
//! This module defines the unified API functions that can be exposed via
//! HTTP (Axum), MCP (rmcp), or CLI (clap) protocols. The functions wrap
//! `EmbeddingService` methods and are protocol-agnostic.
//!
//! When the `http` feature is enabled, these functions are wired into Axum
//! routes at `/api/v1/<endpoint>` via the `#[forge]` macro. When `mcp` is
//! enabled, they are registered as MCP tools. When `cli` is enabled, they
//! back the CLI subcommands.

// sdforge #[forge] 宏内部引用 `feature = "mcp"` 门控(由本 crate 的 `mcp` feature 启用,
// 进而开启 `sdforge/mcp`,后者依赖 `rmcp ~2.1`)。此 allow 抑制第三方宏的 check-cfg 警告。
#![allow(unexpected_cfgs)]

#[cfg(test)]
mod tests;

use crate::EmbeddingService;
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, SimilarityRequest,
    SimilarityResponse,
};
use crate::error::VecboostError;
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;

// sdforge imports (available when http or cli feature is enabled)
#[cfg(any(feature = "http", feature = "cli"))]
use sdforge::prelude::*;

/// Global service holder — set once during startup by `init_service`.
///
/// Uses `OnceLock` for thread-safe one-time initialization. The service
/// is stored as `Arc<RwLock<EmbeddingService>>` to allow shared access
/// from `#[forge]`-annotated functions which cannot accept service
/// parameters (the macro generates Axum handlers that deserialize
/// parameters from the request body).
static SERVICE: OnceLock<Arc<RwLock<EmbeddingService>>> = OnceLock::new();

/// Initialize the global service reference.
///
/// Called from `main.rs` after `EmbeddingService` is constructed.
/// Subsequent `#[forge]`-annotated functions retrieve the service via
/// `service()` which reads from this global.
///
/// # Panics
///
/// Panics if called more than once (the service is already initialized).
pub fn init_service(svc: Arc<RwLock<EmbeddingService>>) {
    if SERVICE.set(svc).is_err() {
        panic!("init_service called more than once");
    }
}

/// Retrieve the global service reference.
///
/// # Errors
///
/// Returns `VecboostError::InternalError` if `init_service` was not called.
#[allow(dead_code)]
fn service() -> Result<Arc<RwLock<EmbeddingService>>, VecboostError> {
    SERVICE
        .get()
        .cloned()
        .ok_or_else(|| VecboostError::InternalError("EmbeddingService not initialized".to_string()))
}

// ---------------------------------------------------------------------------
// Core API functions (backward-compatible, used by CLI and direct callers)
// ---------------------------------------------------------------------------

/// Embed a single text into a vector representation.
///
/// Wraps `EmbeddingService::process_text` with no Matryoshka dimension reduction.
///
/// # Errors
///
/// Returns `VecboostError::InvalidInput` if `text` is empty or exceeds limits.
pub async fn embed(
    service: &EmbeddingService,
    req: EmbedRequest,
) -> Result<EmbedResponse, VecboostError> {
    service.process_text(req, None).await
}

/// Embed multiple texts in batch.
///
/// Wraps `EmbeddingService::process_batch` with no Matryoshka dimension reduction.
///
/// # Errors
///
/// Returns `VecboostError::InvalidInput` if `texts` is empty or any text is invalid.
pub async fn embed_batch(
    service: &EmbeddingService,
    req: BatchEmbedRequest,
) -> Result<BatchEmbedResponse, VecboostError> {
    service.process_batch(req, None).await
}

/// Compute cosine similarity between two texts.
///
/// Wraps `EmbeddingService::process_similarity`.
///
/// # Errors
///
/// Returns `VecboostError::InvalidInput` if either text is empty or invalid.
pub async fn compute_similarity(
    service: &EmbeddingService,
    req: SimilarityRequest,
) -> Result<SimilarityResponse, VecboostError> {
    service.process_similarity(req).await
}

// ---------------------------------------------------------------------------
// VecboostError → ApiError conversion
// ---------------------------------------------------------------------------

/// Convert `VecboostError` to `sdforge::prelude::ApiError`.
#[cfg(any(feature = "http", feature = "cli"))]
#[allow(dead_code)]
fn to_api_error(e: &VecboostError) -> ApiError {
    match e {
        VecboostError::InvalidInput(msg) => ApiError::InvalidInput {
            message: msg.clone(),
            field: None,
            value: None,
        },
        VecboostError::ModelLoadError(msg) => ApiError::Internal {
            message: format!("Model load error: {}", msg),
            error_id: uuid_like_id(),
            source: None,
            context: None,
        },
        other => ApiError::Internal {
            message: other.to_string(),
            error_id: uuid_like_id(),
            source: None,
            context: None,
        },
    }
}

/// Generate a simple error ID (without uuid dependency for non-http features).
#[cfg(any(feature = "http", feature = "cli"))]
#[allow(dead_code)]
fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("err-{}", nanos)
}

// ---------------------------------------------------------------------------
// #[forge]-annotated functions (sdforge multi-protocol)
//
// These functions use the global OnceLock service and are registered via
// inventory for HTTP/MCP/CLI protocol generation by sdforge.
// ---------------------------------------------------------------------------

/// Generate embedding vector for input text.
///
/// HTTP: POST /api/v1/embed
/// MCP: tool "embed_text" (registered via sdforge inventory when `mcp` is on)
#[cfg(feature = "http")]
#[forge(
    name = "embed",
    version = "v1",
    path = "/embed",
    method = "POST",
    tool_name = "embed_text",
    description = "Generate embedding vector for input text"
)]
pub async fn forge_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
    let svc_guard = svc.read().await;
    embed(&svc_guard, req).await.map_err(|e| to_api_error(&e))
}

/// Embed multiple texts in batch.
///
/// HTTP: POST /api/v1/embed_batch
/// MCP: tool "embed_batch" (registered via sdforge inventory when `mcp` is on)
#[cfg(feature = "http")]
#[forge(
    name = "embed_batch",
    version = "v1",
    path = "/embed_batch",
    method = "POST",
    tool_name = "embed_batch",
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn forge_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
    let svc_guard = svc.read().await;
    embed_batch(&svc_guard, req)
        .await
        .map_err(|e| to_api_error(&e))
}

/// Compute cosine similarity between two texts.
///
/// HTTP: POST /api/v1/similarity
/// MCP: tool "compute_similarity" (registered via sdforge inventory when `mcp` is on)
#[cfg(feature = "http")]
#[forge(
    name = "compute_similarity",
    version = "v1",
    path = "/similarity",
    method = "POST",
    tool_name = "compute_similarity",
    description = "Compute cosine similarity between two texts"
)]
pub async fn forge_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
    let svc_guard = svc.read().await;
    compute_similarity(&svc_guard, req)
        .await
        .map_err(|e| to_api_error(&e))
}

// ---------------------------------------------------------------------------
// CLI-specific #[forge(cli = true)] functions
//
// These functions take primitive types (String) that implement FromStr,
// required by sdforge's CLI argument parser.
// ---------------------------------------------------------------------------

/// CLI command: embed a single text.
///
/// Usage: `vecboost cli_embed --text "Hello"`
#[cfg(feature = "cli")]
#[forge(
    name = "cli_embed",
    version = "v1",
    description = "Embed a single text into a vector",
    cli = true
)]
pub async fn cli_embed(text: String) -> Result<String, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
    let svc_guard = svc.read().await;
    let req = EmbedRequest {
        text,
        normalize: Some(true),
    };
    let response = embed(&svc_guard, req).await.map_err(|e| to_api_error(&e))?;
    serde_json::to_string(&response).map_err(|e| ApiError::Internal {
        message: format!("JSON serialization failed: {}", e),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    })
}

/// CLI command: compute similarity between two texts.
///
/// Usage: `vecboost cli_similarity --source "a" --target "b"`
#[cfg(feature = "cli")]
#[forge(
    name = "cli_similarity",
    version = "v1",
    description = "Compute cosine similarity between two texts",
    cli = true
)]
pub async fn cli_similarity(source: String, target: String) -> Result<String, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
    let svc_guard = svc.read().await;
    let req = SimilarityRequest { source, target };
    let response = compute_similarity(&svc_guard, req)
        .await
        .map_err(|e| to_api_error(&e))?;
    serde_json::to_string(&response).map_err(|e| ApiError::Internal {
        message: format!("JSON serialization failed: {}", e),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    })
}

// Verify that the `#[forge(tool_name = ...)]` macros register the embedding
// tools into sdforge's MCP inventory (the tools are served by `sdforge::mcp`).
#[cfg(all(test, feature = "mcp"))]
mod mcp_registration_tests {
    #[test]
    fn forge_macros_register_embed_tools() {
        let tools = sdforge::mcp::get_mcp_tools();
        let names: Vec<String> = tools.iter().map(|t| t.tool().name().to_string()).collect();
        assert!(
            names.iter().any(|n| n == "embed_text"),
            "embed_text not registered; tools: {:?}",
            names
        );
        assert!(
            names.iter().any(|n| n == "embed_batch"),
            "embed_batch not registered; tools: {:?}",
            names
        );
        assert!(
            names.iter().any(|n| n == "compute_similarity"),
            "compute_similarity not registered; tools: {:?}",
            names
        );
    }
}
