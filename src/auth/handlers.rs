#![allow(unused)]

use crate::auth::JwtManager;
use crate::auth::types::{AuthResponse, LoginRequest, RefreshTokenRequest};
use crate::auth::user_store::{UserStore, validate_username_format};
use crate::error::AppError;
use axum::{Json, extract::State};
use std::sync::Arc;
use utoipa::ToSchema;

/// 用户登录处理器
///
/// 验证用户凭据并返回 JWT 令牌
#[utoipa::path(
    post,
    path = "/api/v1/auth/login",
    tag = "auth",
    request_body = LoginRequest,
    responses(
        (status = 200, description = "登录成功", body = AuthResponse),
        (status = 401, description = "认证失败")
    ),
    operation_id = "login"
)]
pub async fn login_handler(
    State(user_store): State<Arc<UserStore>>,
    State(jwt_manager): State<Arc<JwtManager>>,
    State(audit_logger): State<Option<Arc<crate::audit::AuditLogger>>>,
    Json(login_request): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, AppError> {
    // 验证用户名格式
    validate_username_format(&login_request.username)?;

    match user_store.verify_password(&login_request.username, &login_request.password) {
        Ok(user) => {
            let token = jwt_manager.generate_token(&user)?;

            // Log successful login
            if let Some(logger) = audit_logger {
                logger.log_login_success(&login_request.username, None);
            }

            Ok(Json(AuthResponse {
                token,
                token_type: "Bearer".to_string(),
                expires_in: jwt_manager.get_token_expiration(),
            }))
        }
        Err(e) => {
            // Log failed login
            if let Some(logger) = audit_logger {
                logger.log_login_failed(&login_request.username, None, &e.to_string());
            }
            Err(e)
        }
    }
}

/// 用户登出处理器
///
/// 注销当前用户（客户端应丢弃令牌）
#[utoipa::path(
    post,
    path = "/api/v1/auth/logout",
    tag = "auth",
    responses(
        (status = 200, description = "登出成功", body = String)
    ),
    operation_id = "logout",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn logout_handler(
    State(audit_logger): State<Option<Arc<crate::audit::AuditLogger>>>,
) -> &'static str {
    // Note: In a real implementation, you would extract the username from the JWT token
    // For now, we log with a placeholder username
    if let Some(logger) = audit_logger {
        logger.log_logout("unknown", None);
    }
    "Logout successful. Client should discard the token."
}

/// 刷新令牌处理器
///
/// 使用刷新令牌获取新的访问令牌
#[utoipa::path(
    post,
    path = "/api/v1/auth/refresh",
    tag = "auth",
    request_body = RefreshTokenRequest,
    responses(
        (status = 200, description = "刷新成功", body = AuthResponse),
        (status = 401, description = "刷新令牌无效或已过期")
    ),
    operation_id = "refresh_token"
)]
pub async fn refresh_token_handler(
    State(jwt_manager): State<Arc<JwtManager>>,
    State(audit_logger): State<Option<Arc<crate::audit::AuditLogger>>>,
    Json(refresh_request): Json<RefreshTokenRequest>,
) -> Result<Json<AuthResponse>, AppError> {
    let new_token = jwt_manager
        .refresh_token(&refresh_request.refresh_token)
        .await?;

    // Try to extract username from the token for logging
    if let Some(logger) = audit_logger
        && let Ok(claims) = jwt_manager
            .validate_token(&refresh_request.refresh_token)
            .await
    {
        logger.log_token_refresh(&claims.username, None);
    }

    Ok(Json(AuthResponse {
        token: new_token,
        token_type: "Bearer".to_string(),
        expires_in: jwt_manager.get_token_expiration(),
    }))
}

/// 获取当前用户信息处理器
///
/// 返回当前认证用户的信息
#[utoipa::path(
    get,
    path = "/api/v1/auth/me",
    tag = "auth",
    responses(
        (status = 200, description = "成功返回用户信息", body = serde_json::Value),
        (status = 401, description = "未认证")
    ),
    operation_id = "get_current_user",
    security(
        ("bearer_auth" = [])
    )
)]
pub async fn me_handler(
    State(user_store): State<Arc<UserStore>>,
    State(jwt_manager): State<Arc<JwtManager>>,
    Json(refresh_request): Json<RefreshTokenRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let claims = jwt_manager
        .validate_token(&refresh_request.refresh_token)
        .await?;
    let user = user_store.get_user(&claims.username)?.ok_or_else(|| {
        AppError::AuthenticationError(format!("User '{}' not found", claims.username))
    })?;

    Ok(Json(serde_json::json!({
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions
    })))
}
