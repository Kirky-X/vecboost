// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::audit::AuditLogger;
use crate::auth::csrf::{CsrfConfig, CsrfProtection, CsrfTokenStore, OriginValidator};
use crate::auth::jwt::JwtManager;
use crate::auth::types::User;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::Response,
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
    State(audit_logger): State<Option<Arc<AuditLogger>>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 从 Authorization 头获取 token
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok());

    let ip = request
        .headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());

    let path = request.uri().path().to_string();

    let token = match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            &header[7..] // 跳过 "Bearer "
        }
        _ => {
            // Log unauthorized access
            if let Some(ref logger) = audit_logger {
                logger.log_unauthorized_access(ip.clone(), &path);
            }
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    match jwt_manager.validate_token(token).await {
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
        Err(_) => {
            // Log unauthorized access for invalid token
            if let Some(ref logger) = audit_logger {
                logger.log_unauthorized_access(ip.clone(), &path);
            }
            Err(StatusCode::UNAUTHORIZED)
        }
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
        && let Ok(claims) = jwt_manager.validate_token(token).await
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
    // State(audit_logger): State<Arc<AuditLogger>>,
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
        // Log permission denied
        // audit_logger.log_permission_denied(
        //     &auth_context.user.username,
        //     None,
        //     permission,
        // );
        Err(StatusCode::FORBIDDEN)
    }
}

pub async fn require_role_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if auth_context.user.role == "admin" {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

// ============================================================================
// CSRF Protection Middleware
// ============================================================================

/// CSRF Origin Validation Middleware
///
/// This middleware validates the Origin header for state-changing requests.
/// It is the recommended CSRF protection method for API services.
///
/// How it works:
/// 1. For POST/PUT/DELETE/PATCH requests, checks the Origin header
/// 2. Validates that the Origin is in the allowed origins list
/// 3. Allows same-origin requests if configured
///
/// This middleware should be applied to routes that require CSRF protection.
/// It works well with CORS configuration and JWT authentication.
///
/// # Example
///
/// ```ignore
/// let app = Router::new()
///     .route("/api/v1/data", post(handler))
///     .layer(middleware::from_fn_with_state(
///         csrf_config,
///         csrf_origin_middleware,
///     ));
/// ```
pub async fn csrf_origin_middleware(
    State(csrf_config): State<Arc<CsrfConfig>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Only validate state-changing methods
    if !CsrfProtection::requires_protection(request.method()) {
        return Ok(next.run(request).await);
    }

    // Get request URI for logging
    let uri = request.uri().to_string();

    // Extract and validate Origin header
    let origin = OriginValidator::validate_origin(request.headers(), &csrf_config, &uri)?;

    tracing::debug!(
        "CSRF origin validation passed for origin '{}' on {}",
        origin,
        uri
    );

    // Add origin to request extensions for downstream handlers
    let mut request = request;
    request.extensions_mut().insert(origin);

    Ok(next.run(request).await)
}

/// CSRF Token Validation Middleware
///
/// This middleware validates CSRF tokens for state-changing requests.
/// It is suitable for traditional web applications with form submissions.
///
/// How it works:
/// 1. For POST/PUT/DELETE/PATCH requests, checks for CSRF token
/// 2. Looks for token in X-CSRF-Token header
/// 3. Validates token against stored tokens
/// 4. Removes token after validation (one-time use)
///
/// This middleware requires a CSRF token store and should be used
/// in conjunction with a token generation endpoint.
///
/// # Example
///
/// ```ignore
/// let app = Router::new()
///     .route("/api/v1/data", post(handler))
///     .layer(middleware::from_fn_with_state(
///         csrf_token_store,
///         csrf_middleware,
///     ));
/// ```
pub async fn csrf_middleware(
    State(token_store): State<Arc<CsrfTokenStore>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Only validate state-changing methods
    if !CsrfProtection::requires_protection(request.method()) {
        return Ok(next.run(request).await);
    }

    // Get request URI for logging
    let uri = request.uri().to_string();

    // Extract CSRF token from headers
    let csrf_token = request
        .headers()
        .get("X-CSRF-Token")
        .or_else(|| request.headers().get("x-csrf-token"))
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            tracing::warn!("Missing CSRF token for request to {}", uri);
            StatusCode::BAD_REQUEST
        })?;

    // Validate token (and remove it - one-time use)
    let is_valid = token_store.validate_token(csrf_token).await;

    if !is_valid {
        tracing::warn!("Invalid CSRF token for request to {}", uri);
        return Err(StatusCode::FORBIDDEN);
    }

    tracing::debug!("CSRF token validation passed for {}", uri);

    Ok(next.run(request).await)
}

/// Combined CSRF Protection Middleware
///
/// This middleware provides both Origin validation and CSRF token validation.
/// It is useful when you want maximum security for critical endpoints.
///
/// How it works:
/// 1. Validates Origin header (if provided)
/// 2. Validates CSRF token (if token validation is enabled)
/// 3. Requires both validations to pass if both are configured
///
/// # Example
///
/// ```ignore
/// let app = Router::new()
///     .route("/api/v1/admin/*", post(admin_handler))
///     .layer(middleware::from_fn_with_state(
///         (csrf_config, token_store),
///         csrf_combined_middleware,
///     ));
/// ```
pub async fn csrf_combined_middleware(
    State((csrf_config, token_store)): State<(Arc<CsrfConfig>, Arc<CsrfTokenStore>)>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Only validate state-changing methods
    if !CsrfProtection::requires_protection(request.method()) {
        return Ok(next.run(request).await);
    }

    let uri = request.uri().to_string();

    // Step 1: Validate Origin header (if allowed origins are configured)
    if !csrf_config.allowed_origins.is_empty() {
        let _origin = OriginValidator::validate_origin(request.headers(), &csrf_config, &uri)?;
        tracing::debug!("CSRF origin validation passed for {}", uri);
    }

    // Step 2: Validate CSRF token (if token validation is enabled)
    if csrf_config.token_validation_enabled {
        let csrf_token = request
            .headers()
            .get("X-CSRF-Token")
            .or_else(|| request.headers().get("x-csrf-token"))
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| {
                tracing::warn!("Missing CSRF token for request to {}", uri);
                StatusCode::BAD_REQUEST
            })?;

        let is_valid = token_store.validate_token(csrf_token).await;

        if !is_valid {
            tracing::warn!("Invalid CSRF token for request to {}", uri);
            return Err(StatusCode::FORBIDDEN);
        }

        tracing::debug!("CSRF token validation passed for {}", uri);
    }

    Ok(next.run(request).await)
}

/// CORS Configuration Helper for CSRF Protection
///
/// This function creates a CORS configuration that works well with
/// CSRF Origin validation middleware.
///
/// The configuration:
/// - Allows specified origins
/// - Allows necessary headers for CSRF tokens
/// - Allows necessary methods
/// - Sets credentials support for JWT cookies
///
/// # Example
///
/// ```ignore
/// let cors = create_csrf_cors(vec![
///     "https://example.com".to_string(),
///     "http://localhost:3000".to_string(),
/// ]);
///
/// let app = Router::new()
///     .layer(cors)
///     .layer(middleware::from_fn_with_state(
///         csrf_config,
///         csrf_origin_middleware,
///     ));
/// ```
pub fn create_csrf_cors(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use axum::http::header;
    use tower_http::cors::{Any, CorsLayer};

    let allowed_origins: Vec<HeaderValue> = allowed_origins
        .into_iter()
        .filter_map(|origin| origin.parse().ok())
        .collect();

    if allowed_origins.is_empty() {
        // If no origins specified, allow all (development mode)
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::PATCH,
            ])
            .allow_headers(Any)
            .allow_credentials(true)
            .expose_headers([header::CONTENT_TYPE])
    } else {
        // Production mode: only allow specified origins
        CorsLayer::new()
            .allow_origin(allowed_origins)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::DELETE,
                Method::PATCH,
            ])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
            .allow_credentials(true)
            .expose_headers([header::CONTENT_TYPE])
    }
}
