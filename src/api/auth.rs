// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Auth forge handlers — HTTP protocol-agnostic.
//!
//! All handlers access kit capabilities via `state()?.kit.require::<Module>()`:
//! - `AuthModule` → `Option<Arc<JwtManager>>`
//! - `UserStoreModule` → `Option<Arc<UserStore>>`
//! - `AuditModule` → `Option<Arc<AuditLogger>>`

use crate::api::embedding::{kit_internal_error, to_api_error};
use crate::api::init::state;
use crate::auth::middleware::AuthContext;
use crate::auth::{AuthResponse, LoginRequest, RefreshTokenRequest, validate_username_format};
use crate::module_registry::{AuditModule, AuthModule, UserStoreModule};
use std::net::SocketAddr;

#[cfg(feature = "http")]
use axum::extract::ConnectInfo;

#[cfg(feature = "http")]
use sdforge::prelude::*;

#[cfg(feature = "http")]
#[forge(
    name = "login",
    version = "v1",
    path = "/auth/login",
    method = "POST",
    tool_name = "login",
    description = "User login with username and password"
)]
pub async fn forge_login(
    #[param(kind = "extension")] connect_info: ConnectInfo<SocketAddr>,
    req: LoginRequest,
) -> Result<AuthResponse, ApiError> {
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

    let peer_ip = connect_info.0.ip().to_string();

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
                logger.log_login_success(&req.username, Some(peer_ip.clone()));
            }
            Ok(AuthResponse {
                token,
                token_type: "Bearer".to_string(),
                expires_in: jwt_manager.get_token_expiration(),
            })
        }
        Err(e) => {
            if let Some(logger) = audit_logger {
                logger.log_login_failed(&req.username, Some(peer_ip.clone()), &e.to_string());
            }
            Err(to_api_error(&e))
        }
    }
}

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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
    #[param(kind = "extension")] connect_info: ConnectInfo<SocketAddr>,
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
        let peer_ip = connect_info.0.ip().to_string();
        logger.log_logout(&auth_ctx.user.username, Some(peer_ip));
    }

    Ok("Logout successful. Token has been revoked.".to_string())
}

#[cfg(feature = "http")]
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
