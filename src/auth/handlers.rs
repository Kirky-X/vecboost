// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::auth::{
    AuthResponse, JwtManager, LoginRequest, RefreshTokenRequest, UserStore,
    validate_username_format,
};
use crate::error::VecboostError;
use axum::{Json, extract::State};
use std::sync::Arc;

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
) -> Result<Json<AuthResponse>, VecboostError> {
    // 验证用户名格式
    validate_username_format(&login_request.username)?;

    match user_store
        .verify_password(&login_request.username, &login_request.password)
        .await
    {
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
) -> Result<Json<AuthResponse>, VecboostError> {
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
) -> Result<Json<serde_json::Value>, VecboostError> {
    let claims = jwt_manager
        .validate_token(&refresh_request.refresh_token)
        .await?;
    let user = user_store
        .get_user(&claims.username)
        .await?
        .ok_or_else(|| {
            VecboostError::AuthenticationError(format!("User '{}' not found", claims.username))
        })?;

    Ok(Json(serde_json::json!({
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions
    })))
}

#[cfg(all(test, feature = "auth"))]
mod tests {
    use super::*;
    use crate::audit::{AuditConfig, AuditLogger};
    use crate::auth::{User, create_default_admin_user};

    const TEST_JWT_SECRET: &str =
        "test_secret_key_for_handler_tests_must_be_long_enough_abcdef123456";

    fn make_jwt_manager() -> Arc<JwtManager> {
        Arc::new(JwtManager::new(TEST_JWT_SECRET.to_string()).unwrap())
    }

    #[cfg(feature = "db")]
    async fn make_empty_user_store() -> UserStore {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_handlers.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = crate::db::DbPool::new(&url).await.unwrap();
        crate::db::init_schema(&pool).await.unwrap();
        std::mem::forget(temp_dir);
        UserStore::new(Arc::new(pool))
    }

    #[cfg(not(feature = "db"))]
    async fn make_empty_user_store() -> UserStore {
        UserStore::new()
    }

    async fn make_user_store_with_admin() -> Arc<UserStore> {
        let store = make_empty_user_store().await;
        let admin = create_default_admin_user("admin", "TestPassword123!").unwrap();
        store.add_user(admin).await.unwrap();
        Arc::new(store)
    }

    fn make_disabled_audit_logger() -> Arc<AuditLogger> {
        Arc::new(AuditLogger::new(AuditConfig {
            enabled: false,
            ..Default::default()
        }))
    }

    // =========================================================================
    // login_handler tests
    // =========================================================================

    #[tokio::test]
    async fn test_login_handler_valid_credentials_returns_token() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();

        let request = LoginRequest {
            username: "admin".to_string(),
            password: "TestPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager.clone()),
            State(None),
            Json(request),
        )
        .await;

        let response = result.expect("login should succeed");
        assert!(!response.token.is_empty());
        assert_eq!(response.token_type, "Bearer");
        assert_eq!(response.expires_in, jwt_manager.get_token_expiration());

        let claims = jwt_manager.validate_token(&response.token).await.unwrap();
        assert_eq!(claims.username, "admin");
        assert_eq!(claims.role, "admin");
    }

    #[tokio::test]
    async fn test_login_handler_invalid_password_returns_auth_error() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();

        let request = LoginRequest {
            username: "admin".to_string(),
            password: "WrongPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager),
            State(None),
            Json(request),
        )
        .await;

        match result {
            Err(VecboostError::AuthenticationError(msg)) => {
                assert!(msg.contains("Invalid password"));
            }
            _ => panic!("expected AuthenticationError, got {:?}", result.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn test_login_handler_invalid_username_format_returns_validation_error() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();

        // Username starting with a digit violates the format rule.
        let request = LoginRequest {
            username: "1invalid".to_string(),
            password: "TestPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager),
            State(None),
            Json(request),
        )
        .await;

        match result {
            Err(VecboostError::ValidationError(msg)) => {
                assert!(msg.contains("字母开头"));
            }
            _ => panic!("expected ValidationError, got {:?}", result.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn test_login_handler_nonexistent_user_returns_auth_error() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();

        let request = LoginRequest {
            username: "ghost".to_string(),
            password: "TestPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager),
            State(None),
            Json(request),
        )
        .await;

        match result {
            Err(VecboostError::AuthenticationError(msg)) => {
                assert!(msg.contains("User not found"));
            }
            _ => panic!("expected AuthenticationError, got {:?}", result.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn test_login_handler_with_audit_logger_does_not_panic() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();
        let audit_logger: Option<Arc<AuditLogger>> = Some(make_disabled_audit_logger());

        let request = LoginRequest {
            username: "admin".to_string(),
            password: "TestPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager),
            State(audit_logger),
            Json(request),
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_login_handler_failed_login_with_audit_logger_does_not_panic() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();
        let audit_logger: Option<Arc<AuditLogger>> = Some(make_disabled_audit_logger());

        let request = LoginRequest {
            username: "admin".to_string(),
            password: "WrongPassword123!".to_string(),
        };

        let result = login_handler(
            State(user_store),
            State(jwt_manager),
            State(audit_logger),
            Json(request),
        )
        .await;

        assert!(result.is_err());
    }

    // =========================================================================
    // logout_handler tests
    // =========================================================================

    #[tokio::test]
    async fn test_logout_handler_returns_success_message() {
        let result = logout_handler(State(None::<Arc<AuditLogger>>)).await;
        assert!(result.contains("Logout successful"));
    }

    #[tokio::test]
    async fn test_logout_handler_with_audit_logger_does_not_panic() {
        let audit_logger: Option<Arc<AuditLogger>> = Some(make_disabled_audit_logger());
        let result = logout_handler(State(audit_logger)).await;
        assert!(result.contains("Logout successful"));
    }

    // =========================================================================
    // refresh_token_handler tests
    // =========================================================================

    #[tokio::test]
    async fn test_refresh_token_handler_valid_token_returns_new_token() {
        let jwt_manager = make_jwt_manager();
        let user = User {
            username: "admin".to_string(),
            role: "admin".to_string(),
            permissions: vec![],
        };
        let refresh_token = jwt_manager.generate_refresh_token(&user).unwrap();

        let request = RefreshTokenRequest { refresh_token };

        let result =
            refresh_token_handler(State(jwt_manager.clone()), State(None), Json(request)).await;

        let response = result.expect("refresh should succeed");
        assert!(!response.token.is_empty());
        assert_eq!(response.token_type, "Bearer");
        assert_eq!(response.expires_in, jwt_manager.get_token_expiration());

        let claims = jwt_manager.validate_token(&response.token).await.unwrap();
        assert_eq!(claims.username, "admin");
    }

    #[tokio::test]
    async fn test_refresh_token_handler_invalid_token_returns_auth_error() {
        let jwt_manager = make_jwt_manager();

        let request = RefreshTokenRequest {
            refresh_token: "invalid.token.here".to_string(),
        };

        let result = refresh_token_handler(State(jwt_manager), State(None), Json(request)).await;

        match result {
            Err(VecboostError::AuthenticationError(msg)) => {
                assert!(msg.contains("Token validation failed"));
            }
            _ => panic!("expected AuthenticationError, got {:?}", result.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn test_refresh_token_handler_with_audit_logger_does_not_panic() {
        let jwt_manager = make_jwt_manager();
        let user = User {
            username: "admin".to_string(),
            role: "admin".to_string(),
            permissions: vec![],
        };
        let refresh_token = jwt_manager.generate_refresh_token(&user).unwrap();
        let audit_logger: Option<Arc<AuditLogger>> = Some(make_disabled_audit_logger());

        let request = RefreshTokenRequest { refresh_token };

        let result =
            refresh_token_handler(State(jwt_manager), State(audit_logger), Json(request)).await;

        assert!(result.is_ok());
    }

    // =========================================================================
    // me_handler tests
    // =========================================================================

    #[tokio::test]
    async fn test_me_handler_valid_token_existing_user_returns_user_info() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();
        let user = User {
            username: "admin".to_string(),
            role: "admin".to_string(),
            permissions: vec!["embedding:read".to_string()],
        };
        let token = jwt_manager.generate_token(&user).unwrap();

        let request = RefreshTokenRequest {
            refresh_token: token,
        };

        let result = me_handler(State(user_store), State(jwt_manager), Json(request)).await;

        let value = result.expect("me should succeed");
        assert_eq!(value["username"], "admin");
        assert_eq!(value["role"], "admin");
        assert!(value["permissions"].is_array());
    }

    #[tokio::test]
    async fn test_me_handler_valid_token_nonexistent_user_returns_auth_error() {
        let user_store = Arc::new(make_empty_user_store().await);
        let jwt_manager = make_jwt_manager();
        let user = User {
            username: "ghost".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };
        let token = jwt_manager.generate_token(&user).unwrap();

        let request = RefreshTokenRequest {
            refresh_token: token,
        };

        let result = me_handler(State(user_store), State(jwt_manager), Json(request)).await;

        match result {
            Err(VecboostError::AuthenticationError(msg)) => {
                assert!(msg.contains("User 'ghost' not found"));
            }
            _ => panic!("expected AuthenticationError, got {:?}", result.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn test_me_handler_invalid_token_returns_auth_error() {
        let user_store = make_user_store_with_admin().await;
        let jwt_manager = make_jwt_manager();

        let request = RefreshTokenRequest {
            refresh_token: "invalid.token.here".to_string(),
        };

        let result = me_handler(State(user_store), State(jwt_manager), Json(request)).await;

        match result {
            Err(VecboostError::AuthenticationError(msg)) => {
                assert!(msg.contains("Token validation failed"));
            }
            _ => panic!("expected AuthenticationError, got {:?}", result.map(|_| ())),
        }
    }
}
