// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! CSRF (Cross-Site Request Forgery) Protection Module
//!
//! This module provides CSRF protection mechanisms for VecBoost API.
//! Since VecBoost is primarily an API service, we implement two protection strategies:
//!
//! 1. **Origin Header Validation** (Primary - Recommended for APIs)
//!    - Validates the Origin header for state-changing requests
//!    - Suitable for JSON APIs accessed via fetch/XHR
//!    - Requires CORS configuration
//!
//! 2. **CSRF Token Mechanism** (Secondary - For traditional web apps)
//!    - Generates and validates CSRF tokens
//!    - Suitable for form-based submissions
//!    - Requires token storage and validation
//!
//! For API services, Origin validation is generally preferred because:
//! - Browsers automatically include Origin headers for cross-origin requests
//! - No additional token management overhead
//! - Works seamlessly with JWT authentication
//! - Compatible with CORS policies

use crate::error::AppError;
use axum::http::{HeaderMap, StatusCode, header::ORIGIN};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// CSRF Configuration
///
/// Contains settings for CSRF protection, including allowed origins
/// and token configuration.
#[derive(Clone)]
pub struct CsrfConfig {
    /// List of allowed origins for API access
    /// Empty set means no origin validation (not recommended for production)
    pub allowed_origins: Arc<HashSet<String>>,

    /// Whether CSRF token validation is enabled
    pub token_validation_enabled: bool,

    /// Token expiration time in seconds (default: 1 hour)
    pub token_expiration_secs: u64,

    /// Whether to allow same-origin requests without validation
    pub allow_same_origin: bool,
}

impl Default for CsrfConfig {
    fn default() -> Self {
        Self {
            allowed_origins: Arc::new(HashSet::new()),
            token_validation_enabled: false,
            token_expiration_secs: 3600,
            allow_same_origin: true,
        }
    }
}

impl CsrfConfig {
    /// Create a new CSRF configuration with allowed origins
    pub fn new(allowed_origins: Vec<String>) -> Self {
        Self {
            allowed_origins: Arc::new(allowed_origins.into_iter().collect()),
            token_validation_enabled: false,
            token_expiration_secs: 3600,
            allow_same_origin: true,
        }
    }

    /// Enable CSRF token validation
    pub fn with_token_validation(mut self, enabled: bool) -> Self {
        self.token_validation_enabled = enabled;
        self
    }

    /// Set token expiration time
    pub fn with_token_expiration(mut self, secs: u64) -> Self {
        self.token_expiration_secs = secs;
        self
    }

    /// Set whether to allow same-origin requests without validation
    pub fn with_allow_same_origin(mut self, allow: bool) -> Self {
        self.allow_same_origin = allow;
        self
    }

    /// Check if an origin is allowed
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        if self.allowed_origins.is_empty() {
            // No restriction if no origins specified
            return true;
        }

        self.allowed_origins.contains(origin)
    }
}

/// CSRF Token
///
/// Represents a CSRF token with expiration time.
#[derive(Debug, Clone)]
pub struct CsrfToken {
    /// The token value (hashed)
    pub value: String,

    /// Token expiration timestamp (Unix epoch)
    pub expires_at: u64,
}

impl CsrfToken {
    /// Create a new CSRF token
    pub fn new(expires_in_secs: u64) -> Self {
        let value = Self::generate_token();
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + expires_in_secs;

        Self { value, expires_at }
    }

    /// Generate a cryptographically secure random token
    fn generate_token() -> String {
        let bytes: [u8; 32] = rand::random();
        let hash = Sha256::digest(bytes);
        hex::encode(hash)
    }

    /// Check if the token is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now > self.expires_at
    }
}

/// CSRF Token Store trait for different storage backends
#[async_trait::async_trait]
pub trait CsrfTokenStorage: Send + Sync {
    async fn store_token(&self, token: &str) -> Result<(), AppError>;
    async fn validate_token(&self, token: &str) -> bool;
    async fn token_count(&self) -> usize;
}

/// In-memory CSRF token storage
#[derive(Clone)]
pub struct MemoryCsrfStore {
    tokens: Arc<tokio::sync::RwLock<HashSet<String>>>,
}

impl MemoryCsrfStore {
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(tokio::sync::RwLock::new(HashSet::new())),
        }
    }
}

impl Default for MemoryCsrfStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl CsrfTokenStorage for MemoryCsrfStore {
    async fn store_token(&self, token: &str) -> Result<(), AppError> {
        let mut tokens = self.tokens.write().await;
        tokens.insert(token.to_string());
        Ok(())
    }

    async fn validate_token(&self, token: &str) -> bool {
        let mut tokens = self.tokens.write().await;
        tokens.remove(token)
    }

    async fn token_count(&self) -> usize {
        self.tokens.read().await.len()
    }
}

/// Redis-backed CSRF token storage
#[cfg(feature = "redis")]
#[derive(Clone)]
pub struct RedisCsrfStore {
    client: Arc<redis::Client>,
    key_prefix: String,
    expiration_secs: u64,
}

#[cfg(feature = "redis")]
impl RedisCsrfStore {
    pub fn new(client: redis::Client, key_prefix: Option<String>, expiration_secs: u64) -> Self {
        Self {
            client: Arc::new(client),
            key_prefix: key_prefix.unwrap_or_else(|| "vecboost:csrf:".to_string()),
            expiration_secs,
        }
    }

    fn get_key(&self, token: &str) -> String {
        format!("{}{}", self.key_prefix, token)
    }
}

#[cfg(feature = "redis")]
#[async_trait::async_trait]
impl CsrfTokenStorage for RedisCsrfStore {
    async fn store_token(&self, token: &str) -> Result<(), AppError> {
        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| AppError::ConfigError(e.to_string()))?;

        let key = self.get_key(token);
        redis::Cmd::set_ex(&key, "1", self.expiration_secs)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| AppError::ConfigError(e.to_string()))?;

        Ok(())
    }

    async fn validate_token(&self, token: &str) -> bool {
        let mut conn = match self.client.get_async_connection().await {
            Ok(c) => c,
            Err(_) => return false,
        };

        let key = self.get_key(token);
        let existed: bool = match redis::Cmd::exists(&key).query_async(&mut conn).await {
            Ok(e) => e,
            Err(_) => return false,
        };

        if existed {
            // Remove the token after validation (one-time use)
            let _ = redis::Cmd::del(&key).query_async::<()>(&mut conn).await;
        }

        existed
    }

    async fn token_count(&self) -> usize {
        // SCAN would be needed for accurate count - simplified for now
        0
    }
}

/// CSRF Token Store - now wraps a configurable storage backend
#[derive(Clone)]
pub struct CsrfTokenStore {
    storage: Arc<dyn CsrfTokenStorage>,
}

impl CsrfTokenStore {
    /// Create a new CSRF token store with the specified storage backend
    pub fn new() -> Self {
        Self::with_storage(Arc::new(MemoryCsrfStore::new()) as Arc<dyn CsrfTokenStorage>)
    }

    /// Create with a custom storage backend
    pub fn with_storage(storage: Arc<dyn CsrfTokenStorage>) -> Self {
        Self { storage }
    }

    /// Create a CSRF token store with Redis backend
    #[cfg(feature = "redis")]
    pub async fn with_redis(
        redis_url: &str,
        key_prefix: Option<String>,
        expiration_secs: u64,
    ) -> Option<Self> {
        let client = redis::Client::open(redis_url).ok()?;
        if client.get_async_connection().await.is_ok() {
            let storage = Arc::new(RedisCsrfStore::new(client, key_prefix, expiration_secs));
            Some(Self::with_storage(storage))
        } else {
            None
        }
    }

    #[cfg(not(feature = "redis"))]
    pub async fn with_redis(
        _redis_url: &str,
        _key_prefix: Option<String>,
        _expiration_secs: u64,
    ) -> Option<Self> {
        None
    }

    /// Store a token
    pub async fn store_token(&self, token: &str) {
        let _ = self.storage.store_token(token).await;
    }

    /// Validate a token and remove it if valid (one-time use)
    pub async fn validate_token(&self, token: &str) -> bool {
        self.storage.validate_token(token).await
    }

    /// Clean up expired tokens
    pub async fn cleanup_expired(&self, _current_timestamp: u64) {
        // For Redis, expiration is handled by TTL
        // For memory store, this is a no-op
    }

    /// Get the current number of stored tokens
    pub async fn token_count(&self) -> usize {
        self.storage.token_count().await
    }
}

impl Default for CsrfTokenStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Origin Validator
///
/// Validates Origin headers for API requests.
pub struct OriginValidator;

impl OriginValidator {
    /// Extract and validate the Origin header
    ///
    /// Returns Ok(origin) if valid, Err(StatusCode) otherwise.
    pub fn validate_origin(
        headers: &HeaderMap,
        config: &CsrfConfig,
        request_uri: &str,
    ) -> Result<String, StatusCode> {
        // Get the Origin header
        let origin = headers
            .get(ORIGIN)
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| {
                tracing::warn!("Missing Origin header for request to {}", request_uri);
                StatusCode::BAD_REQUEST
            })?;

        // Parse the origin to extract the scheme, host, and port
        let parsed_origin = Self::parse_origin(origin).map_err(|_| {
            tracing::warn!("Invalid Origin header: {}", origin);
            StatusCode::BAD_REQUEST
        })?;

        // Check if same-origin requests are allowed
        if config.allow_same_origin {
            // For same-origin requests, Origin header should match the request host
            // This is handled by CORS middleware, so we just check allowed origins
        }

        // Validate against allowed origins
        if config.is_origin_allowed(&parsed_origin) {
            Ok(parsed_origin)
        } else {
            tracing::warn!(
                "Origin '{}' not in allowed list for request to {}",
                parsed_origin,
                request_uri
            );
            Err(StatusCode::FORBIDDEN)
        }
    }

    /// Parse Origin header to extract scheme, host, and port
    fn parse_origin(origin: &str) -> Result<String, StatusCode> {
        // Basic URL parsing
        if let Some(start) = origin.find("://") {
            let after_scheme = &origin[start + 3..];
            if let Some(end) = after_scheme.find('/') {
                Ok(origin[..start + 3 + end].to_string())
            } else {
                Ok(origin.to_string())
            }
        } else {
            // Invalid origin format
            Err(StatusCode::BAD_REQUEST)
        }
    }

    /// Extract the Referer header as a fallback
    ///
    /// Note: Referer header is less reliable than Origin header
    pub fn get_referer(headers: &HeaderMap) -> Option<String> {
        headers
            .get("referer")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
    }
}

/// CSRF Protection Helper Functions
///
/// Utility functions for CSRF protection.
pub struct CsrfProtection;

impl CsrfProtection {
    /// Check if a request method requires CSRF protection
    ///
    /// State-changing methods (POST, PUT, DELETE, PATCH) require protection
    pub fn requires_protection(method: &axum::http::Method) -> bool {
        matches!(
            *method,
            axum::http::Method::POST
                | axum::http::Method::PUT
                | axum::http::Method::DELETE
                | axum::http::Method::PATCH
        )
    }
    /// Check if a request is a same-origin request
    pub fn is_same_origin(headers: &HeaderMap, host: &str) -> bool {
        // Check Origin header first
        if let Some(origin) = headers.get(ORIGIN).and_then(|h| h.to_str().ok()) {
            return Self::origin_matches_host(origin, host);
        }

        // Fall back to Referer header
        if let Some(referer) = headers.get("referer").and_then(|h| h.to_str().ok()) {
            return Self::referer_matches_host(referer, host);
        }

        // No Origin or Referer header - assume same-origin
        true
    }

    /// Check if Origin matches the host
    fn origin_matches_host(origin: &str, host: &str) -> bool {
        if let Some(start) = origin.find("://") {
            let after_scheme = &origin[start + 3..];
            // Remove path and query
            let origin_host = after_scheme
                .find('/')
                .map(|i| &after_scheme[..i])
                .unwrap_or(after_scheme);

            origin_host == host || origin_host.starts_with(&format!("{}:", host))
        } else {
            false
        }
    }

    /// Check if Referer matches the host
    fn referer_matches_host(referer: &str, host: &str) -> bool {
        if let Some(start) = referer.find("://") {
            let after_scheme = &referer[start + 3..];
            // Remove path and query
            let referer_host = after_scheme
                .find('/')
                .map(|i| &after_scheme[..i])
                .unwrap_or(after_scheme);

            referer_host == host || referer_host.starts_with(&format!("{}:", host))
        } else {
            false
        }
    }

    /// Validate CSRF token from headers
    ///
    /// Looks for the token in:
    /// 1. X-CSRF-Token header (recommended for APIs)
    /// 2. x-csrf-token header (case-insensitive variant)
    pub fn validate_token_from_headers(headers: &HeaderMap, expected_token: &str) -> bool {
        // Check X-CSRF-Token header
        if let Some(token) = headers.get("X-CSRF-Token").and_then(|h| h.to_str().ok()) {
            return token == expected_token;
        }

        // Check x-csrf-token header (case-insensitive)
        if let Some(token) = headers.get("x-csrf-token").and_then(|h| h.to_str().ok()) {
            return token == expected_token;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csrf_token_generation() {
        let token1 = CsrfToken::new(3600);
        let token2 = CsrfToken::new(3600);

        // Tokens should be different
        assert_ne!(token1.value, token2.value);

        // Tokens should be 64 characters (SHA256 hex)
        assert_eq!(token1.value.len(), 64);
    }

    #[test]
    fn test_csrf_token_expiration() {
        let token = CsrfToken::new(1); // 1 second expiration

        // Token should not be expired immediately
        assert!(!token.is_expired());

        // Wait for expiration (simulated)
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(token.is_expired());
    }

    #[test]
    fn test_origin_parsing() {
        let origin = "https://example.com/path";
        let parsed = OriginValidator::parse_origin(origin).unwrap();
        assert_eq!(parsed, "https://example.com");

        let origin = "http://localhost:3000/api/test";
        let parsed = OriginValidator::parse_origin(origin).unwrap();
        assert_eq!(parsed, "http://localhost:3000");
    }

    #[test]
    fn test_config_allowed_origins() {
        let config = CsrfConfig::new(vec![
            "https://example.com".to_string(),
            "http://localhost:3000".to_string(),
        ]);

        assert!(config.is_origin_allowed("https://example.com"));
        assert!(config.is_origin_allowed("http://localhost:3000"));
        assert!(!config.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_requires_protection() {
        assert!(CsrfProtection::requires_protection(
            &axum::http::Method::POST
        ));
        assert!(CsrfProtection::requires_protection(
            &axum::http::Method::PUT
        ));
        assert!(CsrfProtection::requires_protection(
            &axum::http::Method::DELETE
        ));
        assert!(CsrfProtection::requires_protection(
            &axum::http::Method::PATCH
        ));
        assert!(!CsrfProtection::requires_protection(
            &axum::http::Method::GET
        ));
        assert!(!CsrfProtection::requires_protection(
            &axum::http::Method::HEAD
        ));
        assert!(!CsrfProtection::requires_protection(
            &axum::http::Method::OPTIONS
        ));
    }
}
