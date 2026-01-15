// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! OpenAI Compatible Embedding Routes
//!
//! Provides API endpoints that conform to the OpenAI Embeddings API specification.

use crate::domain::openai_embedding::{
    EmbeddingObject, EncodingFormat, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::{AppState, error::AppError};
use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::response::IntoResponse;
use std::net::SocketAddr;

/// Extract the real client IP address
fn extract_real_ip(addr: SocketAddr) -> String {
    addr.ip().to_string()
}

/// OpenAI-compatible embedding handler
///
/// This handler provides an endpoint that conforms to the OpenAI Embeddings API.
/// See: https://platform.openai.com/docs/api-reference/embeddings
pub async fn openai_embed_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<OpenAIEmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Validate input
    if req.input.is_empty() {
        return Err(AppError::InvalidInput("input cannot be empty".to_string()));
    }

    // Validate input array size (OpenAI limit is 2048)
    if req.input.len() > 2048 {
        return Err(AppError::InvalidInput(
            "input array too large (max 2048 items)".to_string(),
        ));
    }

    // Extract encoding format (reserved for future base64 support)
    let _encoding_format = req
        .encoding_format
        .as_deref()
        .and_then(EncodingFormat::parse)
        .unwrap_or(EncodingFormat::Float);

    // Extract client IP for rate limiting
    let ip = extract_real_ip(addr);

    // Check rate limiting if enabled
    if state.rate_limit_enabled {
        let global_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Global)
            .await;
        let ip_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Ip(ip.clone()))
            .await;

        if global_remaining == 0 || ip_remaining == 0 {
            return Err(AppError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    // Process the embedding request using existing service
    let service_guard = state.service.read().await;

    let texts = req.input.to_vec();

    // Use batch processing for both single and multiple inputs
    let batch_req = crate::domain::BatchEmbedRequest {
        texts: texts.clone(),
        mode: None,
        normalize: Some(true),
    };

    let batch_response = service_guard.process_batch(batch_req).await?;

    // Build OpenAI-formatted response
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

    // Calculate token usage (approximate based on text length)
    let total_chars: usize = texts.iter().map(|s| s.len()).sum();
    let prompt_tokens = (total_chars / 4) as u32; // Rough approximation: 4 chars per token

    let response = OpenAIEmbedResponse {
        object: "list".to_string(),
        data: embedding_objects,
        model: req.model.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    };

    Ok(Json(response))
}
