// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Multi-protocol API layer — sdforge integration.
//!
//! `#[forge]`-annotated functions are registered via sdforge inventory for
//! HTTP/MCP/CLI protocol generation. Service is injected via `init_service`
//! into a process-wide `OnceLock` so all protocols share the same instance.

#![allow(unexpected_cfgs)]

#[cfg(test)]
mod tests;

#[cfg(feature = "auth")]
use crate::VecboostState;
#[cfg(feature = "auth")]
use crate::auth::middleware::AuthContext;
#[cfg(feature = "auth")]
use crate::auth::{AuthResponse, LoginRequest, RefreshTokenRequest, validate_username_format};
use crate::domain::openai_embedding::{
    EmbeddingObject, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, FileEmbedRequest,
    FileEmbedResponse, SimilarityRequest, SimilarityResponse,
};
use crate::error::VecboostError;
#[cfg(feature = "auth")]
use crate::module_registry::{AuditModule, AuthModule, UserStoreModule};
use crate::utils::{AggregationMode, PathValidator};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use tokio::sync::RwLock;
type Service = Arc<RwLock<crate::service::embedding::EmbeddingService>>;

#[cfg(any(feature = "http", feature = "cli"))]
use sdforge::prelude::*;

static SERVICE: OnceLock<Service> = OnceLock::new();

#[cfg(feature = "auth")]
static STATE: OnceLock<VecboostState> = OnceLock::new();

pub fn init_service(svc: Service) {
    let _ = SERVICE.set(svc);
}

fn service() -> Result<Service, VecboostError> {
    SERVICE
        .get()
        .cloned()
        .ok_or_else(|| VecboostError::InternalError("init_service not called".to_string()))
}

#[cfg(feature = "auth")]
pub fn init_state(state: VecboostState) {
    let _ = STATE.set(state);
}

#[cfg(feature = "auth")]
fn state() -> Result<VecboostState, VecboostError> {
    STATE
        .get()
        .cloned()
        .ok_or_else(|| VecboostError::InternalError("init_state not called".to_string()))
}

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

#[cfg(any(feature = "http", feature = "cli"))]
fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("err-{}", nanos)
}

// sdforge `#[forge]` functions — protocol-agnostic, service from OnceLock.
// HTTP: POST /api/v1/<path>; MCP: tool "<tool_name>"; CLI: subcommand "<name>".

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
    let svc = service().map_err(|e| to_api_error(&e))?;
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
    let svc = service().map_err(|e| to_api_error(&e))?;
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

    let svc = service().map_err(|e| to_api_error(&e))?;
    let guard = svc.read().await;

    let texts = req.input.to_vec();
    let batch_req = BatchEmbedRequest {
        texts: texts.clone(),
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

    let total_chars: usize = texts.iter().map(|s| s.len()).sum();
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

    let svc = service().map_err(|e| to_api_error(&e))?;
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
        crate::domain::EmbeddingOutput::Single(response) => FileEmbedResponse {
            mode,
            stats,
            embedding: Some(response.embedding),
            paragraphs: None,
        },
        crate::domain::EmbeddingOutput::Paragraphs(paragraphs) => FileEmbedResponse {
            mode,
            stats,
            embedding: None,
            paragraphs: Some(paragraphs),
        },
    })
}

// Auth forge functions — login/refresh use body, logout/me use extension AuthContext.
// AuthContext is filled by auth_middleware (token + user). In MCP mode, logout/me
// generate stubs returning "MCP tool with State parameters is not supported".

#[cfg(all(feature = "http", feature = "auth"))]
fn kit_internal_error(e: impl std::fmt::Display) -> ApiError {
    ApiError::Internal {
        message: e.to_string(),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
#[forge(
    name = "login",
    version = "v1",
    path = "/auth/login",
    method = "POST",
    tool_name = "login",
    description = "User login with username and password"
)]
pub async fn forge_login(req: LoginRequest) -> Result<AuthResponse, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let user_store = st
        .kit
        .require::<UserStoreModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let jwt_manager = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    validate_username_format(&req.username).map_err(|e| ApiError::InvalidInput {
        message: e.to_string(),
        field: Some("username".to_string()),
        value: None,
    })?;

    match user_store
        .verify_password(&req.username, &req.password)
        .await
    {
        Ok(user) => {
            let token = jwt_manager
                .generate_token(&user)
                .map_err(|e| to_api_error(&e))?;
            if let Some(logger) = audit_logger {
                logger.log_login_success(&req.username, None);
            }
            Ok(AuthResponse {
                token,
                token_type: "Bearer".to_string(),
                expires_in: jwt_manager.get_token_expiration(),
            })
        }
        Err(e) => {
            if let Some(logger) = audit_logger {
                logger.log_login_failed(&req.username, None, &e.to_string());
            }
            Err(to_api_error(&e))
        }
    }
}

#[cfg(all(feature = "http", feature = "auth"))]
#[forge(
    name = "refresh",
    version = "v1",
    path = "/auth/refresh",
    method = "POST",
    tool_name = "refresh_token",
    description = "Refresh JWT token"
)]
pub async fn forge_refresh(req: RefreshTokenRequest) -> Result<AuthResponse, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let jwt_manager = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    let new_token = jwt_manager
        .refresh_token(&req.refresh_token)
        .await
        .map_err(|e| to_api_error(&e))?;

    if let Some(logger) = audit_logger
        && let Ok(claims) = jwt_manager.validate_token(&req.refresh_token).await
    {
        logger.log_token_refresh(&claims.username, None);
    }

    Ok(AuthResponse {
        token: new_token,
        token_type: "Bearer".to_string(),
        expires_in: jwt_manager.get_token_expiration(),
    })
}

#[cfg(all(feature = "http", feature = "auth"))]
#[forge(
    name = "logout",
    version = "v1",
    path = "/auth/logout",
    method = "POST",
    tool_name = "logout",
    description = "Logout and revoke JWT token"
)]
pub async fn forge_logout(
    #[param(kind = "extension")] auth_ctx: AuthContext,
) -> Result<String, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let jwt_manager = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    match jwt_manager.revoke_token(&auth_ctx.token).await {
        Ok(()) => log::info!("Token successfully revoked on logout"),
        Err(e) => log::debug!("Logout token could not be revoked: {}", e),
    }

    if let Some(logger) = audit_logger {
        logger.log_logout("unknown", None);
    }

    Ok("Logout successful. Token has been revoked.".to_string())
}

#[cfg(all(feature = "http", feature = "auth"))]
#[forge(
    name = "me",
    version = "v1",
    path = "/auth/me",
    method = "GET",
    tool_name = "get_current_user",
    description = "Get current authenticated user info"
)]
pub async fn forge_me(
    #[param(kind = "extension")] auth_ctx: AuthContext,
) -> Result<serde_json::Value, ApiError> {
    let st = state().map_err(|e| to_api_error(&e))?;
    let user_store = st
        .kit
        .require::<UserStoreModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;

    let username = auth_ctx.user.username.clone();
    let user = user_store
        .get_user(&username)
        .await
        .map_err(|e| to_api_error(&e))?
        .ok_or_else(|| ApiError::InvalidInput {
            message: format!("User '{}' not found", username),
            field: Some("username".to_string()),
            value: None,
        })?;

    Ok(serde_json::json!({
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions
    }))
}

#[cfg(feature = "cli")]
#[forge(
    name = "embed",
    version = "v1",
    cli = true,
    description = "Generate embedding vector for input text"
)]
pub async fn cli_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    let svc = service().map_err(|e| to_api_error(&e))?;
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
    let svc = service().map_err(|e| to_api_error(&e))?;
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
    let svc = service().map_err(|e| to_api_error(&e))?;
    let guard = svc.read().await;
    compute_similarity(&guard, req)
        .await
        .map_err(|e| to_api_error(&e))
}

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
