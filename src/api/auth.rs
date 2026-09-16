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
    let auth_handle = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error(crate::i18n::tr("auth-disabled")))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    let peer_ip = connect_info.0.ip().to_string();

    crate::auth::validate_username_format(&req.username).map_err(|e| ApiError::InvalidInput {
        message: e.error_detail().to_string(),
        field: Some("username".to_string()),
        value: None,
    })?;

    if req.password.is_empty() {
        return Err(ApiError::InvalidInput {
            message: crate::i18n::tr("auth-password-empty"),
            field: Some("password".to_string()),
            value: None,
        });
    }

    // 单管理员登录判定(纯函数,见 verify_login_decision 测试):
    // 非 admin 用户名/错误密码 → 401;无哈希配置(启动闸门漏网)→ 503。
    let decision = crate::auth::verify_login_decision(&auth_handle, &req.username, &req.password);
    // garrison metrics-prometheus：garrison_login_total 计数（vecboost 登录走自有
    // forge_login，不进 garrison 内部 login 路径，此处补齐指标口径）
    let metrics = garrison::observability::GarrisonMetrics::new();
    match decision {
        crate::auth::LoginDecision::Authenticated => {}
        crate::auth::LoginDecision::InvalidCredentials => {
            metrics.record_login(false);
            if let Some(logger) = audit_logger {
                logger.log_login_failed(
                    &req.username,
                    Some(peer_ip.clone()),
                    "invalid credentials",
                );
            }
            return Err(ApiError::AuthenticationFailed {
                reason: crate::i18n::tr("auth-invalid-credentials"),
            });
        }
        crate::auth::LoginDecision::AdminPasswordMissing => {
            log::error!(
                "Login rejected: auth is enabled but no admin password is configured (missing \
                 VECBOOST_ADMIN_PASSWORD). This should have failed at startup."
            );
            return Err(ApiError::service_unavailable_with_source(
                "auth",
                None,
                std::io::Error::other(crate::i18n::tr("auth-admin-password-missing")),
            ));
        }
    }

    // 通过 garrison 创建会话（login_id = username）
    match GarrisonUtil::login_simple(&req.username).await {
        Ok(token) => {
            metrics.record_login(true);
            if let Some(logger) = audit_logger {
                logger.log_login_success(&req.username, Some(peer_ip.clone()));
            }
            Ok(AuthResponse {
                token,
                token_type: "Bearer".to_string(),
                expires_in: auth_handle.token_timeout_secs as u64,
            })
        }
        Err(e) => {
            if let Some(logger) = audit_logger {
                logger.log_login_failed(
                    &req.username,
                    Some(peer_ip.clone()),
                    "authentication failed",
                );
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
    let auth_handle = st
        .kit
        .require::<AuthModule>()
        .map_err(kit_internal_error)?
        .ok_or_else(|| kit_internal_error(crate::i18n::tr("auth-disabled")))?;
    let audit_logger = st
        .kit
        .require::<AuditModule>()
        .map_err(kit_internal_error)?;

    if req.refresh_token.is_empty() {
        return Err(ApiError::InvalidInput {
            message: crate::i18n::tr("auth-refresh-token-empty"),
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
            message: crate::i18n::tr("auth-invalid-token"),
            field: Some("refresh_token".to_string()),
            value: None,
        })?;

    let new_token = GarrisonUtil::login_simple(&login_id)
        .await
        .map_err(|e| to_api_error(e.into()))?;

    // 再撤销旧 token（失败不影响新 token 颁发）
    if let Err(e) = GarrisonUtil::revoke_token(&req.refresh_token).await {
        log::warn!(
            "Failed to revoke old refresh token during refresh for login_id '{}': {}",
            login_id,
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
        expires_in: auth_handle.token_timeout_secs as u64,
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
        .ok_or_else(|| kit_internal_error(crate::i18n::tr("auth-disabled")))?;
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

    Ok(crate::i18n::tr("logout-success"))
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
    let has_all_perms = GarrisonUtil::has_permission("*")
        .await
        .map_err(|e| to_api_error(e.into()))?;
    let is_admin = GarrisonUtil::has_role("admin")
        .await
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
