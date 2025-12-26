use crate::auth::JwtManager;
use crate::auth::types::{AuthResponse, LoginRequest};
use crate::auth::user_store::UserStore;
use crate::error::AppError;
use axum::{Json, extract::State};
use std::sync::Arc;

pub async fn login_handler(
    State(user_store): State<Arc<UserStore>>,
    State(jwt_manager): State<Arc<JwtManager>>,
    Json(login_request): Json<LoginRequest>,
) -> Result<Json<AuthResponse>, AppError> {
    let user = user_store.verify_password(&login_request.username, &login_request.password)?;

    let token = jwt_manager.generate_token(&user)?;

    Ok(Json(AuthResponse {
        token,
        token_type: "Bearer".to_string(),
        expires_in: jwt_manager.get_token_expiration(),
    }))
}

pub async fn logout_handler() -> &'static str {
    "Logout successful. Client should discard the token."
}

pub async fn refresh_token_handler(
    State(_jwt_manager): State<Arc<JwtManager>>,
    State(_user_store): State<Arc<UserStore>>,
) -> Result<Json<AuthResponse>, AppError> {
    Err(AppError::AuthenticationError(
        "Token refresh not implemented. Please login again.".to_string(),
    ))
}

pub async fn me_handler(
    State(_user_store): State<Arc<UserStore>>,
    State(_jwt_manager): State<Arc<JwtManager>>,
) -> Result<Json<serde_json::Value>, AppError> {
    Err(AppError::AuthenticationError("Not implemented".to_string()))
}
