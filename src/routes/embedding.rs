// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Embedding-related routes
//!
//! Provides API endpoints for text embedding, batch embedding, similarity calculation, and file embedding

use crate::AppState;
use crate::domain::{
    BatchEmbedRequest, EmbedRequest, FileEmbedRequest, FileEmbedResponse, SimilarityRequest,
};
use crate::error::AppError;
use crate::utils::{AggregationMode, PathValidator};
use axum::http::HeaderMap;
use axum::{Json, extract::ConnectInfo, extract::State, response::IntoResponse};
use std::net::SocketAddr;
use std::path::PathBuf;

/// Extract the real client IP address, considering proxy headers
/// Returns the IP address as a string
fn extract_real_ip(addr: SocketAddr) -> String {
    addr.ip().to_string()
}

/// Check if an IP is in the whitelist
#[allow(clippy::collapsible_if)]
fn is_ip_whitelisted(ip: &str, whitelist: &[String]) -> bool {
    whitelist.iter().any(|whitelist_ip| {
        // Exact match
        if ip == *whitelist_ip {
            return true;
        }

        // CIDR match (完整实现)
        if let Some((network_str, prefix_len_str)) = whitelist_ip.split_once('/') {
            if let Ok(prefix_len) = prefix_len_str.parse::<u8>() {
                // 尝试解析 IP 地址
                if let Ok(ip_addr) = ip.parse::<std::net::IpAddr>() {
                    if let Ok(network_addr) = network_str.parse::<std::net::IpAddr>() {
                        // 使用标准库的 IpAddr 进行匹配
                        return match (ip_addr, network_addr) {
                            (std::net::IpAddr::V4(ip_v4), std::net::IpAddr::V4(network_v4)) => {
                                // IPv4 CIDR 匹配
                                let ip_u32 = u32::from_be_bytes(ip_v4.octets());
                                let network_u32 = u32::from_be_bytes(network_v4.octets());
                                let mask = if prefix_len >= 32 {
                                    0xFFFFFFFFu32
                                } else {
                                    0xFFFFFFFFu32 << (32 - prefix_len)
                                };
                                (ip_u32 & mask) == (network_u32 & mask)
                            }
                            (std::net::IpAddr::V6(ip_v6), std::net::IpAddr::V6(network_v6)) => {
                                // IPv6 CIDR 匹配
                                let ip_u128 = u128::from_be_bytes(ip_v6.octets());
                                let network_u128 = u128::from_be_bytes(network_v6.octets());
                                let mask = if prefix_len >= 128 {
                                    u128::MAX
                                } else {
                                    u128::MAX << (128u8 - prefix_len)
                                };
                                (ip_u128 & mask) == (network_u128 & mask)
                            }
                            _ => false,
                        };
                    }
                }
            }
        }

        false
    })
}

/// Add rate limit headers to response
async fn add_rate_limit_headers(headers: &mut HeaderMap, state: &AppState, ip: &str) {
    use axum::http::HeaderValue;

    if state.rate_limit_enabled {
        let global_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Global)
            .await;
        let ip_remaining = state
            .rate_limiter
            .get_remaining(crate::rate_limit::RateLimitDimension::Ip(ip.to_string()))
            .await;

        // Use the more restrictive limit
        let remaining = std::cmp::min(global_remaining, ip_remaining);

        if let Ok(limit_val) = HeaderValue::from_str("1000") {
            headers.insert("x-ratelimit-limit", limit_val);
        }
        if let Ok(remaining_val) = HeaderValue::from_str(&remaining.to_string()) {
            headers.insert("x-ratelimit-remaining", remaining_val);
        }
        if let Ok(reset_val) = HeaderValue::from_str("60") {
            headers.insert("x-ratelimit-reset", reset_val);
        }
    }
}

/// Single text embedding handler
///
/// Converts a single text to vector representation
#[utoipa::path(
    post,
    path = "/api/v1/embed",
    tag = "embedding",
    request_body = EmbedRequest,
    responses(
        (status = 200, description = "Embedding successful", body = crate::domain::EmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_text",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn embed_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    if state.rate_limit_enabled && !is_ip_whitelisted(&ip, &state.ip_whitelist) {
        // Check both global and IP rate limits
        if !state
            .rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(AppError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            ));
        }
    }

    // 如果启用了流水线，使用流水线处理
    if state.pipeline_enabled {
        let ip_clone = ip.clone();
        let res = crate::pipeline::handle_pipeline_request(state.clone(), req, ip_clone).await?;
        let mut response = res.into_response();
        add_rate_limit_headers(response.headers_mut(), &state, &ip).await;
        return Ok(response);
    }

    // 否则直接调用服务
    let service_guard = state.service.read().await;
    let res = service_guard.process_text(req).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// Batch text embedding handler
///
/// Converts multiple texts to vector representations in batch
#[utoipa::path(
    post,
    path = "/api/v1/embed/batch",
    tag = "embedding",
    request_body = BatchEmbedRequest,
    responses(
        (status = 200, description = "Batch embedding successful", body = crate::domain::BatchEmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_batch",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn batch_embed_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<BatchEmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    if state.rate_limit_enabled && !is_ip_whitelisted(&ip, &state.ip_whitelist) {
        // Check both global and IP rate limits
        if !state
            .rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(AppError::InvalidInput("Rate limit exceeded".to_string()));
        }
    }

    let service_guard = state.service.read().await;
    let res = service_guard.process_batch(req).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// Similarity calculation handler
///
/// Calculates the similarity between two texts
#[utoipa::path(
    post,
    path = "/api/v1/similarity",
    tag = "embedding",
    request_body = SimilarityRequest,
    responses(
        (status = 200, description = "Similarity calculation successful", body = crate::domain::SimilarityResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "compute_similarity",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn similarity_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<SimilarityRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    if state.rate_limit_enabled && !is_ip_whitelisted(&ip, &state.ip_whitelist) {
        // Check both global and IP rate limits
        if !state
            .rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(AppError::InvalidInput("Rate limit exceeded".to_string()));
        }
    }

    let service_guard = state.service.read().await;
    let res = service_guard.process_similarity(req).await?;

    // Create response with rate limit headers
    let mut response = Json(res).into_response();
    add_rate_limit_headers(response.headers_mut(), &state, &ip).await;

    Ok(response)
}

/// File embedding handler
///
/// Converts file content to vector representation
#[utoipa::path(
    post,
    path = "/api/v1/embed/file",
    tag = "embedding",
    request_body = FileEmbedResponse,
    responses(
        (status = 200, description = "File embedding successful", body = crate::domain::FileEmbedResponse),
        (status = 400, description = "Invalid request"),
        (status = 429, description = "Rate limit exceeded")
    ),
    operation_id = "embed_file",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn file_embed_handler(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(req): Json<FileEmbedRequest>,
) -> Result<impl IntoResponse, AppError> {
    // Extract real client IP
    let ip = extract_real_ip(addr);

    // Check if rate limiting is enabled and IP is not whitelisted
    if state.rate_limit_enabled && !is_ip_whitelisted(&ip, &state.ip_whitelist) {
        // Check both global and IP rate limits
        if !state
            .rate_limiter
            .check_rate_limit(vec![
                crate::rate_limit::RateLimitDimension::Global,
                crate::rate_limit::RateLimitDimension::Ip(ip.clone()),
            ])
            .await
        {
            return Err(AppError::InvalidInput("Rate limit exceeded".to_string()));
        }
    }

    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    // Create path validator, only allow file access within current working directory
    let current_dir = std::env::current_dir()
        .map_err(|e| AppError::IoError(format!("Failed to get current directory: {}", e)))?;

    let path_validator = PathValidator::new()
        .add_allowed_root(&current_dir)
        .add_allowed_root("/tmp"); // Allow temporary directory access

    // Validate path to prevent path traversal attacks
    let validated_path = path_validator
        .validate_file(&path)
        .map_err(|e| AppError::InvalidInput(format!("Path validation failed: {}", e)))?;

    let service_guard = state.service.read().await;
    let stats = service_guard.get_processing_stats(&validated_path)?;
    let output = service_guard.embed_file(&validated_path, mode).await?;

    drop(service_guard);

    let response = match output {
        crate::domain::EmbeddingOutput::Single(response) => crate::domain::FileEmbedResponse {
            mode,
            stats,
            embedding: Some(response.embedding),
            paragraphs: None,
        },
        crate::domain::EmbeddingOutput::Paragraphs(paragraphs) => {
            crate::domain::FileEmbedResponse {
                mode,
                stats,
                embedding: None,
                paragraphs: Some(paragraphs),
            }
        }
    };

    // Create response with rate limit headers
    let mut resp = Json(response).into_response();
    add_rate_limit_headers(resp.headers_mut(), &state, &ip).await;

    Ok(resp)
}
