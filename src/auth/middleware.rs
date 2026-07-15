// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::audit::AuditLogger;
use crate::auth::{CsrfConfig, CsrfProtection, CsrfTokenStore, JwtManager, OriginValidator, User};
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

    log::debug!(
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
            log::warn!("Missing CSRF token for request to {}", uri);
            StatusCode::BAD_REQUEST
        })?;

    // Validate token (and remove it - one-time use)
    let is_valid = token_store.validate_token(csrf_token).await;

    if !is_valid {
        log::warn!("Invalid CSRF token for request to {}", uri);
        return Err(StatusCode::FORBIDDEN);
    }

    log::debug!("CSRF token validation passed for {}", uri);

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
        log::debug!("CSRF origin validation passed for {}", uri);
    }

    // Step 2: Validate CSRF token (if token validation is enabled)
    if csrf_config.token_validation_enabled {
        let csrf_token = request
            .headers()
            .get("X-CSRF-Token")
            .or_else(|| request.headers().get("x-csrf-token"))
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| {
                log::warn!("Missing CSRF token for request to {}", uri);
                StatusCode::BAD_REQUEST
            })?;

        let is_valid = token_store.validate_token(csrf_token).await;

        if !is_valid {
            log::warn!("Invalid CSRF token for request to {}", uri);
            return Err(StatusCode::FORBIDDEN);
        }

        log::debug!("CSRF token validation passed for {}", uri);
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

#[cfg(all(test, feature = "auth", feature = "http"))]
mod tests {
    use super::*;
    use crate::auth::JwtManager;
    use crate::auth::types::User;
    use axum::Router;
    use axum::body::Body;
    use axum::extract::FromRef;
    use axum::http::{Method, Request, StatusCode};
    use axum::middleware;
    use axum::routing::any;
    use std::sync::Arc;
    use tower::ServiceExt;

    const TEST_JWT_SECRET: &str = "test_secret_key_for_middleware_tests_must_be_long_enough_abcdef";

    fn make_jwt_manager() -> Arc<JwtManager> {
        Arc::new(JwtManager::new(TEST_JWT_SECRET.to_string()).unwrap())
    }

    fn make_test_user() -> User {
        User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["embedding:read".to_string()],
        }
    }

    fn make_admin_user() -> User {
        User {
            username: "admin".to_string(),
            role: "admin".to_string(),
            permissions: vec![],
        }
    }

    fn build_request(method: Method, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    fn build_request_with_headers(
        method: Method,
        uri: &str,
        headers: &[(&str, &str)],
    ) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(uri);
        for (key, value) in headers {
            builder = builder.header(*key, *value);
        }
        builder.body(Body::empty()).unwrap()
    }

    fn build_request_with_extensions(
        method: Method,
        uri: &str,
        auth_ctx: Option<AuthContext>,
    ) -> Request<Body> {
        let mut request = build_request(method, uri);
        if let Some(ctx) = auth_ctx {
            request.extensions_mut().insert(ctx);
        }
        request
    }

    /// Test-only state that provides `FromRef` impls for `auth_middleware`'s
    /// two `State<T>` extractors. axum-core 0.4 has no blanket tuple `FromRef`
    /// impls, so a dedicated struct is required.
    #[derive(Clone)]
    struct AuthTestState {
        jwt: Arc<JwtManager>,
        audit: Option<Arc<AuditLogger>>,
    }

    impl FromRef<AuthTestState> for Arc<JwtManager> {
        fn from_ref(s: &AuthTestState) -> Self {
            s.jwt.clone()
        }
    }

    impl FromRef<AuthTestState> for Option<Arc<AuditLogger>> {
        fn from_ref(s: &AuthTestState) -> Self {
            s.audit.clone()
        }
    }

    fn make_auth_state() -> AuthTestState {
        AuthTestState {
            jwt: make_jwt_manager(),
            audit: None,
        }
    }

    // =========================================================================
    // auth_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_auth_middleware_missing_authorization_header_returns_401() {
        let state = make_auth_state();
        let app = Router::new()
            .route("/protected", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/protected"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_middleware_non_bearer_header_returns_401() {
        let state = make_auth_state();
        let app = Router::new()
            .route("/protected", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
            .with_state(());

        let request = build_request_with_headers(
            Method::GET,
            "/protected",
            &[("authorization", "Basic dXNlcjpwYXNz")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_middleware_invalid_token_returns_401() {
        let state = make_auth_state();
        let app = Router::new()
            .route("/protected", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
            .with_state(());

        let request = build_request_with_headers(
            Method::GET,
            "/protected",
            &[("authorization", "Bearer invalid.token.here")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_middleware_valid_token_passes_through() {
        let jwt_manager = make_jwt_manager();
        let token = jwt_manager.generate_token(&make_test_user()).unwrap();
        let state = AuthTestState {
            jwt: jwt_manager,
            audit: None,
        };
        let app = Router::new()
            .route("/protected", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(state, auth_middleware))
            .with_state(());

        let request = build_request_with_headers(
            Method::GET,
            "/protected",
            &[("authorization", &format!("Bearer {}", token))],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // =========================================================================
    // optional_auth_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_optional_auth_middleware_without_token_passes_through() {
        let state = make_jwt_manager();
        let app = Router::new()
            .route("/api", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                optional_auth_middleware,
            ))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/api"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_optional_auth_middleware_with_valid_token_passes_through() {
        let jwt_manager = make_jwt_manager();
        let token = jwt_manager.generate_token(&make_test_user()).unwrap();
        let state = jwt_manager;
        let app = Router::new()
            .route("/api", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                optional_auth_middleware,
            ))
            .with_state(());

        let request = build_request_with_headers(
            Method::GET,
            "/api",
            &[("authorization", &format!("Bearer {}", token))],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_optional_auth_middleware_with_invalid_token_passes_through() {
        let state = make_jwt_manager();
        let app = Router::new()
            .route("/api", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                optional_auth_middleware,
            ))
            .with_state(());

        let request = build_request_with_headers(
            Method::GET,
            "/api",
            &[("authorization", "Bearer invalid.token")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // =========================================================================
    // require_role_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_require_role_middleware_admin_passes() {
        let app = Router::new()
            .route("/admin", any(|| async { "ok" }))
            .layer(middleware::from_fn(require_role_middleware))
            .with_state(());

        let request = build_request_with_extensions(
            Method::GET,
            "/admin",
            Some(AuthContext {
                user: make_admin_user(),
            }),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_require_role_middleware_non_admin_returns_403() {
        let app = Router::new()
            .route("/admin", any(|| async { "ok" }))
            .layer(middleware::from_fn(require_role_middleware))
            .with_state(());

        let request = build_request_with_extensions(
            Method::GET,
            "/admin",
            Some(AuthContext {
                user: make_test_user(),
            }),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_require_role_middleware_no_auth_context_returns_401() {
        let app = Router::new()
            .route("/admin", any(|| async { "ok" }))
            .layer(middleware::from_fn(require_role_middleware))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/admin"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    // =========================================================================
    // require_permission_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_require_permission_middleware_with_permission_passes() {
        let app = Router::new()
            .route("/resource", any(|| async { "ok" }))
            .layer(middleware::from_fn(|req, next| {
                require_permission_middleware("embedding:read", req, next)
            }))
            .with_state(());

        let request = build_request_with_extensions(
            Method::GET,
            "/resource",
            Some(AuthContext {
                user: make_test_user(),
            }),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_require_permission_middleware_without_permission_returns_403() {
        let app = Router::new()
            .route("/resource", any(|| async { "ok" }))
            .layer(middleware::from_fn(|req, next| {
                require_permission_middleware("model:write", req, next)
            }))
            .with_state(());

        let request = build_request_with_extensions(
            Method::GET,
            "/resource",
            Some(AuthContext {
                user: make_test_user(),
            }),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_require_permission_middleware_admin_role_grants_all() {
        // Admin role should bypass permission check (User::has_permission returns true for admin)
        let app = Router::new()
            .route("/resource", any(|| async { "ok" }))
            .layer(middleware::from_fn(|req, next| {
                require_permission_middleware("any:permission", req, next)
            }))
            .with_state(());

        let request = build_request_with_extensions(
            Method::GET,
            "/resource",
            Some(AuthContext {
                user: make_admin_user(),
            }),
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_require_permission_middleware_no_auth_context_returns_401() {
        let app = Router::new()
            .route("/resource", any(|| async { "ok" }))
            .layer(middleware::from_fn(|req, next| {
                require_permission_middleware("embedding:read", req, next)
            }))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/resource"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    // =========================================================================
    // csrf_origin_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_csrf_origin_middleware_get_passes() {
        // GET requests don't require CSRF protection
        let csrf_config = Arc::new(CsrfConfig::new(vec!["https://example.com".to_string()]));
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                csrf_config,
                csrf_origin_middleware,
            ))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/data"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_origin_middleware_post_without_origin_returns_400() {
        let csrf_config = Arc::new(CsrfConfig::new(vec!["https://example.com".to_string()]));
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                csrf_config,
                csrf_origin_middleware,
            ))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::POST, "/data"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_csrf_origin_middleware_post_with_allowed_origin_passes() {
        let csrf_config = Arc::new(CsrfConfig::new(vec!["https://example.com".to_string()]));
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                csrf_config,
                csrf_origin_middleware,
            ))
            .with_state(());

        let request =
            build_request_with_headers(Method::POST, "/data", &[("origin", "https://example.com")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_origin_middleware_post_with_disallowed_origin_returns_403() {
        let csrf_config = Arc::new(CsrfConfig::new(vec!["https://example.com".to_string()]));
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                csrf_config,
                csrf_origin_middleware,
            ))
            .with_state(());

        let request =
            build_request_with_headers(Method::POST, "/data", &[("origin", "https://evil.com")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_csrf_origin_middleware_empty_allowed_origins_allows_all() {
        // Empty allowed_origins means no restriction (development mode)
        let csrf_config = Arc::new(CsrfConfig::new(vec![]));
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                csrf_config,
                csrf_origin_middleware,
            ))
            .with_state(());

        // POST with any origin should pass when allowed_origins is empty
        let request = build_request_with_headers(
            Method::POST,
            "/data",
            &[("origin", "https://any-site.com")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // =========================================================================
    // csrf_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_csrf_middleware_get_passes() {
        let token_store = Arc::new(CsrfTokenStore::new());
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(token_store, csrf_middleware))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/data"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_middleware_post_without_token_returns_400() {
        let token_store = Arc::new(CsrfTokenStore::new());
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(token_store, csrf_middleware))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::POST, "/data"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_csrf_middleware_post_with_valid_token_passes() {
        let token_store = Arc::new(CsrfTokenStore::new());
        token_store.store_token("valid-csrf-token").await;
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(token_store, csrf_middleware))
            .with_state(());

        let request = build_request_with_headers(
            Method::POST,
            "/data",
            &[("x-csrf-token", "valid-csrf-token")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_middleware_post_with_invalid_token_returns_403() {
        let token_store = Arc::new(CsrfTokenStore::new());
        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(token_store, csrf_middleware))
            .with_state(());

        let request = build_request_with_headers(
            Method::POST,
            "/data",
            &[("x-csrf-token", "nonexistent-token")],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_csrf_middleware_token_is_one_time_use() {
        let token_store = Arc::new(CsrfTokenStore::new());
        token_store.store_token("one-time-token").await;

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                token_store.clone(),
                csrf_middleware,
            ))
            .with_state(());

        // First use: token should be valid and consumed
        let request1 = build_request_with_headers(
            Method::POST,
            "/data",
            &[("x-csrf-token", "one-time-token")],
        );
        let response1 = app.clone().oneshot(request1).await.unwrap();
        assert_eq!(response1.status(), StatusCode::OK);

        // Second use: token should be invalid (already consumed)
        let request2 = build_request_with_headers(
            Method::POST,
            "/data",
            &[("x-csrf-token", "one-time-token")],
        );
        let response2 = app.oneshot(request2).await.unwrap();
        assert_eq!(response2.status(), StatusCode::FORBIDDEN);
    }

    // =========================================================================
    // csrf_combined_middleware tests
    // =========================================================================

    #[tokio::test]
    async fn test_csrf_combined_middleware_get_passes() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::GET, "/data"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_post_without_origin_returns_400() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let response = app
            .oneshot(build_request(Method::POST, "/data"))
            .await
            .unwrap();
        // Missing Origin header → BAD_REQUEST
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_post_with_origin_but_no_token_returns_400() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let request =
            build_request_with_headers(Method::POST, "/data", &[("origin", "https://example.com")]);
        let response = app.oneshot(request).await.unwrap();
        // Origin OK but missing CSRF token → BAD_REQUEST
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_post_with_valid_origin_and_token_passes() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        token_store.store_token("valid-combined-token").await;
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let request = build_request_with_headers(
            Method::POST,
            "/data",
            &[
                ("origin", "https://example.com"),
                ("x-csrf-token", "valid-combined-token"),
            ],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_post_with_disallowed_origin_returns_403() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let request =
            build_request_with_headers(Method::POST, "/data", &[("origin", "https://evil.com")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_post_with_valid_origin_invalid_token_returns_403() {
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(true),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        let request = build_request_with_headers(
            Method::POST,
            "/data",
            &[
                ("origin", "https://example.com"),
                ("x-csrf-token", "invalid-token"),
            ],
        );
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_csrf_combined_middleware_no_token_validation_only_checks_origin() {
        // When token_validation_enabled is false, only Origin is checked
        let csrf_config = Arc::new(
            CsrfConfig::new(vec!["https://example.com".to_string()]).with_token_validation(false),
        );
        let token_store = Arc::new(CsrfTokenStore::new());
        let state = (csrf_config, token_store);

        let app = Router::new()
            .route("/data", any(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(
                state,
                csrf_combined_middleware,
            ))
            .with_state(());

        // POST with valid Origin but no CSRF token should pass
        let request =
            build_request_with_headers(Method::POST, "/data", &[("origin", "https://example.com")]);
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    // =========================================================================
    // create_csrf_cors tests
    // =========================================================================

    #[test]
    fn test_create_csrf_cors_empty_origins_allows_any() {
        // Empty origins → development mode, allow any
        let _cors = create_csrf_cors(vec![]);
        // Should not panic; CorsLayer is created successfully
    }

    #[test]
    fn test_create_csrf_cors_specific_origins() {
        let _cors = create_csrf_cors(vec![
            "https://example.com".to_string(),
            "http://localhost:3000".to_string(),
        ]);
        // Should not panic; CorsLayer is created successfully
    }
}
