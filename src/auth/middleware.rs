use crate::auth::jwt::JwtManager;
use crate::auth::types::User;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use axum_extra::{
    TypedHeader,
    headers::{Authorization, authorization::Bearer},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct AuthContext {
    pub user: User,
}

#[derive(Clone)]
pub struct JwtAuthLayer {
    #[allow(dead_code)]
    jwt_manager: Arc<JwtManager>,
}

impl JwtAuthLayer {
    pub fn new(jwt_manager: Arc<JwtManager>) -> Self {
        Self { jwt_manager }
    }
}

pub async fn auth_middleware(
    State(jwt_manager): State<Arc<JwtManager>>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = auth.token();

    match jwt_manager.validate_token(token) {
        Ok(claims) => {
            let user = User {
                username: claims.username,
                role: claims.role,
                permissions: claims.permissions,
            };

            let mut request = request;
            request.extensions_mut().insert(AuthContext { user });

            Ok(next.run(request).await)
        }
        Err(_) => Err(StatusCode::UNAUTHORIZED),
    }
}

pub async fn optional_auth_middleware(
    State(jwt_manager): State<Arc<JwtManager>>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Response {
    if let Some(auth_header) = headers.get("authorization")
        && let Ok(auth_str) = auth_header.to_str()
        && let Some(token) = auth_str.strip_prefix("Bearer ")
        && let Ok(claims) = jwt_manager.validate_token(token)
    {
        let user = User {
            username: claims.username,
            role: claims.role,
            permissions: claims.permissions,
        };

        request.extensions_mut().insert(AuthContext { user });
    }

    next.run(request).await
}

pub async fn require_permission_middleware(
    permission: &'static str,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth_context.user.has_permission(permission) {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

pub async fn require_role_middleware(
    role: &'static str,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth_context.user.role == role || auth_context.user.role == "admin" {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}
