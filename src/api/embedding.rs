// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Embedding forge handlers — HTTP/MCP/CLI protocol-agnostic.
//!
//! All handlers access `EmbeddingService` via
//! `state()?.kit.require::<EmbeddingModule>()` (returns `Arc<RwLock<EmbeddingService>>`).

use crate::api::init::state;
use crate::domain::openai_embedding::{
    EmbeddingObject, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, EmbeddingOutput,
    FileEmbedRequest, FileEmbedResponse, SimilarityRequest, SimilarityResponse,
};
use crate::error::VecboostError;
use crate::module_registry::EmbeddingModule;
use crate::utils::{AggregationMode, PathValidator};
use std::path::PathBuf;

#[cfg(any(feature = "http", feature = "cli"))]
use sdforge::prelude::*;

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

#[cfg(any(feature = "http", feature = "cli"))]
pub(crate) fn to_api_error(e: &VecboostError) -> ApiError {
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

#[cfg(any(feature = "http", feature = "cli"))]
pub(crate) fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("err-{}", nanos)
}

#[cfg(any(feature = "http", feature = "cli"))]
pub(crate) fn kit_internal_error(e: impl std::fmt::Display) -> ApiError {
    ApiError::Internal {
        message: e.to_string(),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    }
}

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

/// Retrieve `max_text_length` from kit config, falling back to the default
/// (`EmbeddingConfig::default().max_text_length` = 8192) when config is absent.
#[cfg(any(feature = "http", feature = "cli"))]
fn max_text_length_from_kit(kit: &trait_kit::AsyncKit<trait_kit::AsyncReady>) -> usize {
    kit.config::<crate::config::app::EmbeddingConfig>()
        .unwrap_or_default()
        .max_text_length
}

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
    let st = state().map_err(|e| to_api_error(&e))?;
    validate_text_length(
        std::slice::from_ref(&req.text),
        max_text_length_from_kit(&st.kit),
    )
    .map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed(&guard, req).await.map_err(|e| to_api_error(&e))
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
    let st = state().map_err(|e| to_api_error(&e))?;
    validate_text_length(&req.texts, max_text_length_from_kit(&st.kit))
        .map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed_batch(&guard, req).await.map_err(|e| to_api_error(&e))
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
    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    compute_similarity(&guard, req)
        .await
        .map_err(|e| to_api_error(&e))
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
    Ok(serde_json::json!({ "status": "healthy" }))
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

    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;

    let texts = req.input.to_vec();
    validate_text_length(&texts, max_text_length_from_kit(&st.kit))
        .map_err(|e| to_api_error(&e))?;
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
        .map_err(|e| to_api_error(&e))?;

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
    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    let current_dir = std::env::current_dir().map_err(|e| ApiError::Internal {
        message: format!("Failed to get current directory: {}", e),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    })?;

    let path_validator = PathValidator::new().add_allowed_root(&current_dir);

    let validated_path =
        path_validator
            .validate_file(&path)
            .map_err(|e| ApiError::InvalidInput {
                message: format!("Path validation failed: {}", e),
                field: Some("path".to_string()),
                value: Some(serde_json::Value::String(req.path.clone())),
            })?;

    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    let stats = guard
        .get_processing_stats(&validated_path)
        .map_err(|e| to_api_error(&e))?;
    let output = guard
        .embed_file(&validated_path, mode)
        .await
        .map_err(|e| to_api_error(&e))?;
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

#[cfg(feature = "cli")]
#[forge(
    name = "embed",
    version = "v1",
    cli = true,
    description = "Generate embedding vector for input text"
)]
pub async fn cli_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed(&guard, req).await.map_err(|e| to_api_error(&e))
}

#[cfg(feature = "cli")]
#[forge(
    name = "embed_batch",
    version = "v1",
    cli = true,
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn cli_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed_batch(&guard, req).await.map_err(|e| to_api_error(&e))
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
    let st = state().map_err(|e| to_api_error(&e))?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    compute_similarity(&guard, req)
        .await
        .map_err(|e| to_api_error(&e))
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
}
