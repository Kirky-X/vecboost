// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Auth forge handlers — HTTP protocol-agnostic.
//!
//! All handlers access kit capabilities via `state()?.kit.require::<Module>()`:
//! - `AuthModule` → `Option<Arc<GarrisonHandle>>`
//! - `AuditModule` → `Option<Arc<AuditLogger>>`
//!
//! 认证操作通过 garrison `GarrisonUtil` 静态方法执行（全局单例）。

use crate::api::embedding::{kit_internal_error, to_api_error};
use crate::api::init::state;
use crate::auth::middleware::AuthContext;
use crate::auth::{AuthResponse, GarrisonUtil, LoginRequest, RefreshTokenRequest};
use crate::registry::{AuditModule, AuthModule};
use std::net::SocketAddr;

#[cfg(feature = "http")]
use axum::extract::ConnectInfo;

#[cfg(feature = "http")]
use sdforge::prelude::*;

#[cfg(feature = "http")]
#[forge(
    name = "login",
    version = 1,
    path = "/auth/login",
    method = "POST",
    tool_name = "login",
    description = "User login with username and password"
)]
pub async fn forge_login(
    #[param(kind = "extension")] connect_info: ConnectInfo<SocketAddr>,
    req: LoginRequest,
) -> Result<AuthResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let _auth = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    let peer_ip = connect_info.0.ip().to_string();

    crate::auth::validate_username_format(&req.username).map_err(|e| ApiError::InvalidInput {
        message: e.to_string(),
        field: Some("username".to_string()),
        value: None,
    })?;

    // 拒绝空密码（基本安全检查）
    if req.password.is_empty() {
        return Err(ApiError::InvalidInput {
            message: "password must not be empty".to_string(),
            field: Some("password".to_string()),
            value: None,
        });
    }

    // 通过 garrison 创建会话（login_id = username）
    // TODO: 凭证校验需集成 garrison account-credential 系统
    // GarrisonUtil::login 不接受密码参数，需通过 DAO 查询用户存储的密码哈希后
    // 使用 PasswordHasher::verify 校验。当前仅验证 username 格式合法。
    // SECURITY: 密码未校验即颁发 token，仅限受信任网络环境使用。
    log::warn!(
        "SECURITY: forge_login issued token for user '{}' without password verification \
         (garrison credential store not integrated)",
        &req.username
    );
    match GarrisonUtil::login_simple(&req.username).await {
        Ok(token) => {
            if let Some(logger) = audit_logger {
                logger.log_login_success(&req.username, Some(peer_ip.clone()));
            }
            Ok(AuthResponse {
                token,
                token_type: "Bearer".to_string(),
                expires_in: 3600, // TODO: 从 garrison config TTL 读取；当前默认 1h
            })
        }
        Err(e) => {
            if let Some(logger) = audit_logger {
                logger.log_login_failed(&req.username, Some(peer_ip.clone()), "authentication failed");
            }
            Err(to_api_error(e.into()))
        }
    }
}

#[cfg(feature = "http")]
#[forge(
    name = "refresh",
    version = 1,
    path = "/auth/refresh",
    method = "POST",
    tool_name = "refresh_token",
    description = "Refresh JWT token"
)]
pub async fn forge_refresh(
    #[param(kind = "extension")] connect_info: ConnectInfo<SocketAddr>,
    req: RefreshTokenRequest,
) -> Result<AuthResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let _auth = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    // 输入验证：拒绝空 refresh_token
    if req.refresh_token.is_empty() {
        return Err(ApiError::InvalidInput {
            message: "refresh_token must not be empty".to_string(),
            field: Some("refresh_token".to_string()),
            value: None,
        });
    }

    // 通过旧 token 获取 login_id，然后创建新会话
    // 注意：先创建新会话，再撤销旧 token，避免并发 revoke 导致竞态
    let login_id = GarrisonUtil::get_login_id_by_token(&req.refresh_token)
        .await
        .map_err(|e| {
            if let Some(logger) = &audit_logger {
                logger.log_login_failed("<unknown>", None, "refresh token validation failed");
            }
            to_api_error(e.into())
        })?
        .ok_or_else(|| ApiError::InvalidInput {
            message: "Invalid or expired token".to_string(),
            field: Some("refresh_token".to_string()),
            value: None,
        })?;

    // 先创建新会话
    let new_token = GarrisonUtil::login_simple(&login_id)
        .await
        .map_err(|e| to_api_error(e.into()))?;

    // 再撤销旧 token（失败不影响新 token 颁发）
    if let Err(e) = GarrisonUtil::revoke_token(&req.refresh_token).await {
        log::warn!(
            "Failed to revoke old refresh token during refresh for login_id '{}': {}",
            &login_id,
            e
        );
    }

    let peer_ip = connect_info.0.ip().to_string();
    if let Some(logger) = audit_logger {
        logger.log_token_refresh(&login_id, Some(peer_ip));
    }

    Ok(AuthResponse {
        token: new_token,
        token_type: "Bearer".to_string(),
        expires_in: 3600, // TODO: 从 garrison config TTL 读取；当前默认 1h
    })
}

#[cfg(feature = "http")]
#[forge(
    name = "logout",
    version = 1,
    path = "/auth/logout",
    method = "POST",
    tool_name = "logout",
    description = "Logout and revoke JWT token"
)]
pub async fn forge_logout(
    #[param(kind = "extension")] auth_ctx: AuthContext,
    #[param(kind = "extension")] connect_info: ConnectInfo<SocketAddr>,
) -> Result<String, ApiError> {
    let st = state().map_err(to_api_error)?;
    let _auth = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error("auth disabled at runtime"))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    // 通过 garrison 撤销 token（best-effort：失败时仍返回成功，避免阻断客户端登出）
    if let Err(e) = GarrisonUtil::revoke_token(&auth_ctx.token).await {
        log::warn!("Logout token revocation failed (best-effort): {}", e);
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
    version = 1,
    path = "/auth/me",
    method = "GET",
    tool_name = "get_current_user",
    description = "Get current authenticated user info"
)]
pub async fn forge_me(
    #[param(kind = "extension")] auth_ctx: AuthContext,
) -> Result<serde_json::Value, ApiError> {
    // 通过 garrison 查询完整权限/角色列表（task_local token 已由 auth_middleware 设置）
    let has_all_perms = GarrisonUtil::has_permission("*").await
        .map_err(|e| to_api_error(e.into()))?;
    let is_admin = GarrisonUtil::has_role("admin").await
        .map_err(|e| to_api_error(e.into()))?;

    let role = if is_admin { "admin" } else { "user" };
    let perms: Vec<&str> = if has_all_perms {
        vec!["*"]
    } else {
        vec!["embedding:read", "embedding:write"]
    };

    Ok(serde_json::json!({
        "username": auth_ctx.user.username,
        "role": role,
        "permissions": perms
    }))
}
