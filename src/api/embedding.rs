// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Embedding forge handlers — HTTP/MCP/CLI/gRPC protocol-agnostic.
//!
//! All handlers access `EmbeddingService` via
//! `state()?.kit.require::<EmbeddingModule>()` (returns `Arc<RwLock<EmbeddingService>>`).
//!
//! # Architecture
//!
//! Protocol-specific `forge_*` / `cli_*` / `grpc_*` handlers are thin wrappers
//! that only attach `#[forge(...)]` macros; the actual business logic lives in
//! protocol-agnostic `*_handler` functions below. This eliminates ~96 lines of
//! duplicated state-acquire/validate/dispatch code across the three protocols.

use crate::api::init::state;
use crate::domain::openai_embedding::{
    EmbeddingObject, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, EmbeddingOutput,
    FileEmbedRequest, FileEmbedResponse, ModelInfo, ModelListResponse, ModelMetadata,
    ModelSwitchRequest, ModelSwitchResponse, SimilarityRequest, SimilarityResponse,
};
use crate::error::VecboostError;
use crate::module_registry::EmbeddingModule;
use crate::utils::{AggregationMode, PathValidator};
use std::path::PathBuf;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions — used by examples and external callers
// =============================================================================

pub async fn embed(
    svc: &crate::service::embedding::EmbeddingService,
    req: EmbedRequest,
) -> Result<EmbedResponse, VecboostError> {
    svc.process_text(req, None).await
}

pub async fn embed_batch(
    svc: &crate::service::embedding::EmbeddingService,
    req: BatchEmbedRequest,
) -> Result<BatchEmbedResponse, VecboostError> {
    svc.process_batch(req, None).await
}

pub async fn compute_similarity(
    svc: &crate::service::embedding::EmbeddingService,
    req: SimilarityRequest,
) -> Result<SimilarityResponse, VecboostError> {
    svc.process_similarity(req).await
}

// =============================================================================
// Error conversion helpers
// =============================================================================

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn to_api_error(e: VecboostError) -> ApiError {
    match e {
        VecboostError::InvalidInput(msg) => ApiError::InvalidInput {
            message: msg,
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

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("err-{}", nanos)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn kit_internal_error(e: impl std::fmt::Display) -> ApiError {
    ApiError::Internal {
        message: e.to_string(),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    }
}

// =============================================================================
// Validation helpers
// =============================================================================

/// Validate that no text exceeds the maximum allowed byte length.
///
/// Returns `ValidationError` on the first offending text, including index,
/// limit, and actual length for diagnostics. Byte length (`str::len`) is used
/// to match tokenizer input boundaries and prevent resource exhaustion.
fn validate_text_length(texts: &[String], max: usize) -> Result<(), VecboostError> {
    for (idx, text) in texts.iter().enumerate() {
        if text.len() > max {
            return Err(VecboostError::ValidationError(format!(
                "text at index {} exceeds max length {} (got {})",
                idx,
                max,
                text.len()
            )));
        }
    }
    Ok(())
}

/// Validate that batch size does not exceed the configured maximum, preventing
/// resource exhaustion via oversized batch requests.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn validate_batch_size(texts_len: usize, max: usize) -> Result<(), VecboostError> {
    if texts_len > max {
        return Err(VecboostError::ValidationError(format!(
            "batch size {} exceeds max {} (config embedding.max_batch_size)",
            texts_len, max
        )));
    }
    Ok(())
}

/// Retrieve `max_text_length` from kit config, falling back to the default
/// (`EmbeddingConfig::default().max_text_length` = 8192) when config is absent.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn max_text_length_from_kit(kit: &trait_kit::AsyncKit<trait_kit::AsyncReady>) -> usize {
    kit.config::<crate::config::app::EmbeddingConfig>()
        .unwrap_or_default()
        .max_text_length
}

/// Retrieve `max_batch_size` from kit config, falling back to the default
/// (`EmbeddingConfig::default().max_batch_size` = 64) when config is absent.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn max_batch_size_from_kit(kit: &trait_kit::AsyncKit<trait_kit::AsyncReady>) -> usize {
    kit.config::<crate::config::app::EmbeddingConfig>()
        .unwrap_or_default()
        .max_batch_size
}

/// Build a `PathValidator` from `[server] grpc_allowed_roots` config.
///
/// When `grpc_allowed_roots` is `Some`, those paths are used as the allowed
/// roots. When `None`, falls back to the current working directory — but
/// refuses sensitive directories (`/`, `/etc`, `/root`, `/var`, `/usr`, ...)
/// to prevent accidental filesystem-wide exposure.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn build_path_validator() -> Result<PathValidator, ApiError> {
    let st = state().map_err(to_api_error)?;
    let server_cfg = st
        .kit
        .config::<crate::config::app::ServerConfig>()
        .unwrap_or_default();

    if let Some(roots) = &server_cfg.grpc_allowed_roots
        && !roots.is_empty()
    {
        let mut validator = PathValidator::new();
        for root in roots {
            validator = validator.add_allowed_root(root);
        }
        return Ok(validator);
    }

    // Fallback: current working directory with sensitive-dir guard.
    let cwd = std::env::current_dir().map_err(|e| ApiError::Internal {
        message: format!("Failed to get current directory: {}", e),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    })?;

    const SENSITIVE_DIRS: &[&str] = &[
        "/", "/etc", "/root", "/var", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc",
    ];
    let cwd_str = cwd.to_string_lossy();
    if SENSITIVE_DIRS.iter().any(|s| cwd_str.as_ref() == *s) {
        return Err(ApiError::Internal {
            message: format!(
                "Refusing to use sensitive directory '{}' as allowed root; \
                 configure [server] grpc_allowed_roots explicitly",
                cwd_str
            ),
            error_id: uuid_like_id(),
            source: None,
            context: None,
        });
    }

    Ok(PathValidator::new().add_allowed_root(&cwd))
}

// =============================================================================
// Protocol-agnostic business-logic handlers
//
// Each `*_handler` performs state acquisition, input validation, and service
// dispatch. Protocol-specific `forge_*` / `cli_*` / `grpc_*` functions below
// delegate to these helpers, keeping each protocol's surface to a single line
// of delegation plus the `#[forge(...)]` macro registration.
// =============================================================================

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn embed_handler(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    validate_text_length(
        std::slice::from_ref(&req.text),
        max_text_length_from_kit(&st.kit),
    )
    .map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn embed_batch_handler(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    validate_batch_size(req.texts.len(), max_batch_size_from_kit(&st.kit)).map_err(to_api_error)?;
    validate_text_length(&req.texts, max_text_length_from_kit(&st.kit)).map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed_batch(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn compute_similarity_handler(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    compute_similarity(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn embed_file_handler(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    let validator = build_path_validator()?;
    let validated_path = validator
        .validate_file(&path)
        .map_err(|e| ApiError::InvalidInput {
            message: format!("Path validation failed: {}", e),
            field: Some("path".to_string()),
            value: Some(serde_json::Value::String(req.path.clone())),
        })?;

    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    let stats = guard
        .get_processing_stats(&validated_path)
        .map_err(to_api_error)?;
    let output = guard
        .embed_file(&validated_path, mode)
        .await
        .map_err(to_api_error)?;
    drop(guard);

    Ok(match output {
        EmbeddingOutput::Single(response) => FileEmbedResponse {
            mode,
            stats,
            embedding: Some(response.embedding),
            paragraphs: None,
        },
        EmbeddingOutput::Paragraphs(paragraphs) => FileEmbedResponse {
            mode,
            stats,
            embedding: None,
            paragraphs: Some(paragraphs),
        },
    })
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn model_switch_handler(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let mut guard = svc.write().await;
    guard.switch_model(req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn get_current_model_handler() -> Result<ModelInfo, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    guard.get_model_info().ok_or_else(|| ApiError::NotFound {
        resource: "model".to_string(),
        resource_id: None,
    })
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn get_model_info_handler() -> Result<ModelMetadata, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    guard
        .get_model_metadata()
        .ok_or_else(|| ApiError::NotFound {
            resource: "model_metadata".to_string(),
            resource_id: None,
        })
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn list_models_handler() -> Result<ModelListResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    Ok(guard.list_available_models())
}

/// Unified health-check response — minimal, no sensitive info.
///
/// Returns only `{"status": "OK"}`. Detailed runtime info (version, uptime,
/// model name) is intentionally omitted to avoid information leakage on
/// unauthenticated endpoints.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn health_handler() -> Result<serde_json::Value, ApiError> {
    Ok(serde_json::json!({ "status": "OK" }))
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

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
    embed_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "embed_batch",
    version = "v1",
    path = "/embed/batch",
    method = "POST",
    tool_name = "embed_batch",
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn forge_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

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
    compute_similarity_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "file_embed",
    version = "v1",
    path = "/embed/file",
    method = "POST",
    tool_name = "file_embed",
    description = "Embed text from a file with path validation"
)]
pub async fn forge_file_embed(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    embed_file_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "health",
    version = "v1",
    path = "/health",
    method = "GET",
    no_prefix = true,
    tool_name = "health",
    description = "Service health check"
)]
pub async fn forge_health() -> Result<serde_json::Value, ApiError> {
    health_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "model_switch",
    version = "v1",
    path = "/model/switch",
    method = "POST",
    tool_name = "model_switch",
    description = "Switch the currently loaded model"
)]
pub async fn forge_model_switch(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    model_switch_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "get_current_model",
    version = "v1",
    path = "/model/current",
    method = "GET",
    tool_name = "get_current_model",
    description = "Get information about the currently loaded model"
)]
pub async fn forge_get_current_model() -> Result<ModelInfo, ApiError> {
    get_current_model_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "get_model_info",
    version = "v1",
    path = "/model/info",
    method = "GET",
    tool_name = "get_model_info",
    description = "Get metadata about the currently loaded model"
)]
pub async fn forge_get_model_info() -> Result<ModelMetadata, ApiError> {
    get_model_info_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "list_models",
    version = "v1",
    path = "/models",
    method = "GET",
    tool_name = "list_models",
    description = "List all available models"
)]
pub async fn forge_list_models() -> Result<ModelListResponse, ApiError> {
    list_models_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "openai_embed",
    version = "v1",
    path = "/v1/embeddings",
    method = "POST",
    no_prefix = true,
    tool_name = "openai_embed",
    description = "OpenAI-compatible embeddings endpoint"
)]
pub async fn forge_openai_embed(req: OpenAIEmbedRequest) -> Result<OpenAIEmbedResponse, ApiError> {
    if req.input.is_empty() {
        return Err(ApiError::InvalidInput {
            message: "input cannot be empty".to_string(),
            field: Some("input".to_string()),
            value: None,
        });
    }
    if req.input.len() > 2048 {
        return Err(ApiError::InvalidInput {
            message: "input array too large (max 2048 items)".to_string(),
            field: Some("input".to_string()),
            value: None,
        });
    }

    let st = state().map_err(to_api_error)?;
    validate_batch_size(req.input.len(), max_batch_size_from_kit(&st.kit)).map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;

    let texts = req.input.to_vec();
    validate_text_length(&texts, max_text_length_from_kit(&st.kit)).map_err(to_api_error)?;
    // 预先计算 total_chars，避免后续 move texts 到 batch_req 后再访问
    let total_chars: usize = texts.iter().map(|s| s.len()).sum();
    let batch_req = BatchEmbedRequest {
        texts,
        mode: None,
        normalize: Some(true),
    };
    let batch_response = guard
        .process_batch(batch_req, req.dimensions)
        .await
        .map_err(to_api_error)?;

    let embedding_objects: Vec<EmbeddingObject> = batch_response
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(idx, result)| EmbeddingObject {
            object: "embedding".to_string(),
            embedding: result.embedding,
            index: idx,
        })
        .collect();

    let prompt_tokens = (total_chars / 4) as u32;

    Ok(OpenAIEmbedResponse {
        object: "list".to_string(),
        data: embedding_objects,
        model: req.model.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "embed",
    version = "v1",
    cli = true,
    description = "Generate embedding vector for input text"
)]
pub async fn cli_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}

#[cfg(feature = "cli")]
#[forge(
    name = "embed_batch",
    version = "v1",
    cli = true,
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn cli_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

#[cfg(feature = "cli")]
#[forge(
    name = "compute_similarity",
    version = "v1",
    cli = true,
    description = "Compute cosine similarity between two texts"
)]
pub async fn cli_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    compute_similarity_handler(req).await
}

// =============================================================================
// gRPC forge handlers — sdforge unified `Call` protocol
//
// Each function is registered via `#[forge(grpc_method = "...")]` and invoked
// through sdforge's `SdForgeService/Call` RPC with the corresponding method
// name. Request payloads are JSON-serialized domain types passed via
// `CallRequest.data`; responses are JSON-serialized domain types returned in
// `CallResponse.data`.
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed",
    version = "v1",
    grpc_method = "vecboost.embed",
    description = "Generate embedding vector for input text"
)]
pub async fn grpc_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed_batch",
    version = "v1",
    grpc_method = "vecboost.embed_batch",
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn grpc_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_compute_similarity",
    version = "v1",
    grpc_method = "vecboost.compute_similarity",
    description = "Compute similarity between two texts"
)]
pub async fn grpc_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    compute_similarity_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed_file",
    version = "v1",
    grpc_method = "vecboost.embed_file",
    description = "Embed text from a file with path validation"
)]
pub async fn grpc_embed_file(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    embed_file_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_model_switch",
    version = "v1",
    grpc_method = "vecboost.model_switch",
    description = "Switch the currently loaded model"
)]
pub async fn grpc_model_switch(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    model_switch_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_get_current_model",
    version = "v1",
    grpc_method = "vecboost.get_current_model",
    description = "Get information about the currently loaded model"
)]
pub async fn grpc_get_current_model() -> Result<ModelInfo, ApiError> {
    get_current_model_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_get_model_info",
    version = "v1",
    grpc_method = "vecboost.get_model_info",
    description = "Get metadata about the currently loaded model"
)]
pub async fn grpc_get_model_info() -> Result<ModelMetadata, ApiError> {
    get_model_info_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_list_models",
    version = "v1",
    grpc_method = "vecboost.list_models",
    description = "List all available models"
)]
pub async fn grpc_list_models() -> Result<ModelListResponse, ApiError> {
    list_models_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_health_check",
    version = "v1",
    grpc_method = "vecboost.health_check",
    description = "Service health check"
)]
pub async fn grpc_health_check() -> Result<serde_json::Value, ApiError> {
    health_handler().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_text_length_under_limit_passes() {
        let texts = vec!["short".to_string(), "also short".to_string()];
        assert!(validate_text_length(&texts, 100).is_ok());
    }

    #[test]
    fn test_validate_text_length_at_boundary_passes() {
        let text = "a".repeat(8192);
        let texts = vec![text];
        assert!(validate_text_length(&texts, 8192).is_ok());
    }

    #[test]
    fn test_validate_text_length_exceeds_limit_returns_error() {
        let texts = vec!["ok".to_string(), "x".repeat(101)];
        let err = validate_text_length(&texts, 100).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                assert!(
                    msg.contains("index 1"),
                    "error should mention index 1: {msg}"
                );
                assert!(
                    msg.contains("max length 100"),
                    "error should mention limit: {msg}"
                );
                assert!(
                    msg.contains("got 101"),
                    "error should mention actual length: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_text_length_empty_slice_passes() {
        let texts: Vec<String> = vec![];
        assert!(validate_text_length(&texts, 100).is_ok());
    }

    #[test]
    fn test_validate_text_length_first_offending_text_reported() {
        let texts = vec!["x".repeat(51), "x".repeat(52)];
        let err = validate_text_length(&texts, 50).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                assert!(
                    msg.contains("index 0"),
                    "should report first offending index: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_under_limit_passes() {
        assert!(validate_batch_size(10, 64).is_ok());
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_exceeds_limit_returns_error() {
        let err = validate_batch_size(100, 64).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                assert!(
                    msg.contains("batch size 100"),
                    "error should mention actual size: {msg}"
                );
                assert!(msg.contains("max 64"), "error should mention limit: {msg}");
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_at_boundary_passes() {
        assert!(validate_batch_size(64, 64).is_ok());
    }
}
