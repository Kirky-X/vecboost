// Copyright (c) 2025-2026 Kirky.X
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

use crate::error::VecboostError;
use axum::http::{HeaderMap, StatusCode, header::ORIGIN};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use url::Url;

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
            .expect("SystemTime before UNIX EPOCH")
            .as_secs()
            + expires_in_secs;

        Self { value, expires_at }
    }

    /// Generate a cryptographically secure random token using OS entropy source
    fn generate_token() -> String {
        let mut bytes = [0u8; 32];

        use rand::Rng;
        rand::rng().fill_bytes(&mut bytes);

        let hash = Sha256::digest(bytes);
        hex::encode(hash)
    }

    /// Check if the token is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime before UNIX EPOCH")
            .as_secs();
        now > self.expires_at
    }
}

/// CSRF Token Store trait for different storage backends
#[async_trait::async_trait]
pub trait CsrfTokenStorage: Send + Sync {
    async fn store_token(&self, token: &str) -> Result<(), VecboostError>;
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
    async fn store_token(&self, token: &str) -> Result<(), VecboostError> {
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

    pub fn with_custom_prefix(
        client: redis::Client,
        key_prefix: String,
        expiration_secs: u64,
    ) -> Self {
        Self {
            client: Arc::new(client),
            key_prefix,
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
    async fn store_token(&self, token: &str) -> Result<(), VecboostError> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| VecboostError::ConfigError(e.to_string()))?;

        let key = self.get_key(token);
        redis::Cmd::set_ex(&key, "1", self.expiration_secs)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| VecboostError::ConfigError(e.to_string()))?;

        Ok(())
    }

    async fn validate_token(&self, token: &str) -> bool {
        let mut conn = match self.client.get_multiplexed_async_connection().await {
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
        if client.get_multiplexed_async_connection().await.is_ok() {
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
                log::warn!("Missing Origin header for request to {}", request_uri);
                StatusCode::BAD_REQUEST
            })?;

        // Parse the origin to extract the scheme, host, and port
        let parsed_origin = Self::parse_origin(origin).map_err(|_| {
            log::warn!("Invalid Origin header: {}", origin);
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
            log::warn!(
                "Origin '{}' not in allowed list for request to {}",
                parsed_origin,
                request_uri
            );
            Err(StatusCode::FORBIDDEN)
        }
    }

    /// Parse Origin header to extract scheme, host, and port
    fn parse_origin(origin: &str) -> Result<String, StatusCode> {
        // Use url crate for robust URL parsing
        Url::parse(origin)
            .ok()
            .and_then(|url| {
                url.host_str().map(|host| {
                    // Reconstruct origin without path
                    let scheme = url.scheme();
                    if let Some(port) = url.port() {
                        format!("{}://{}:{}", scheme, host, port)
                    } else {
                        format!("{}://{}", scheme, host)
                    }
                })
            })
            .ok_or_else(|| {
                log::warn!("Invalid Origin header format: {}", origin);
                StatusCode::BAD_REQUEST
            })
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

    // ===== CsrfConfig 边界测试 =====

    #[test]
    fn test_csrf_config_default() {
        let config = CsrfConfig::default();
        assert!(config.allowed_origins.is_empty());
        assert!(!config.token_validation_enabled);
        assert_eq!(config.token_expiration_secs, 3600);
        assert!(config.allow_same_origin);
    }

    #[test]
    fn test_csrf_config_builder_chain() {
        let config = CsrfConfig::new(vec!["https://api.example.com".to_string()])
            .with_token_validation(true)
            .with_token_expiration(7200)
            .with_allow_same_origin(false);

        assert!(config.token_validation_enabled);
        assert_eq!(config.token_expiration_secs, 7200);
        assert!(!config.allow_same_origin);
        assert!(config.is_origin_allowed("https://api.example.com"));
    }

    #[test]
    fn test_is_origin_allowed_empty_set() {
        // 空集合表示不限制 — 所有 origin 均允许
        let config = CsrfConfig::default();
        assert!(config.is_origin_allowed("https://anything.com"));
        assert!(config.is_origin_allowed("http://localhost:1234"));
        assert!(config.is_origin_allowed(""));
    }

    #[test]
    fn test_is_origin_allowed_non_empty_set_rejects_unknown() {
        let config = CsrfConfig::new(vec!["https://trusted.com".to_string()]);
        assert!(!config.is_origin_allowed("https://untrusted.com"));
        assert!(!config.is_origin_allowed(""));
    }

    // ===== CsrfToken 边界测试 =====

    #[test]
    fn test_csrf_token_zero_expiration() {
        // 0 秒过期 — 边界条件,token 立即过期
        let token = CsrfToken::new(0);
        // 由于时间精度为秒,0 秒后可能已过期或刚好未过期
        // 关键属性:value 仍应是有效的 SHA256 hex
        assert_eq!(token.value.len(), 64);
    }

    #[tokio::test]
    async fn test_memory_csrf_store_replay_attack_protection() {
        // 一次性使用 token — 重放攻击防护
        let store = MemoryCsrfStore::new();
        let token = "replay-test-token";

        store.store_token(token).await.unwrap();
        assert_eq!(store.token_count().await, 1);

        // 第一次验证 — 应成功并移除 token
        let first_validation = store.validate_token(token).await;
        assert!(first_validation);

        // 第二次验证 — 重放攻击,应失败
        let second_validation = store.validate_token(token).await;
        assert!(!second_validation);

        assert_eq!(store.token_count().await, 0);
    }

    #[tokio::test]
    async fn test_memory_csrf_store_validate_nonexistent_token() {
        let store = MemoryCsrfStore::new();
        // 验证不存在的 token — 应返回 false
        assert!(!store.validate_token("nonexistent").await);
        assert_eq!(store.token_count().await, 0);
    }

    #[tokio::test]
    async fn test_memory_csrf_store_multiple_tokens() {
        let store = MemoryCsrfStore::new();
        store.store_token("token1").await.unwrap();
        store.store_token("token2").await.unwrap();
        store.store_token("token3").await.unwrap();
        assert_eq!(store.token_count().await, 3);

        // 重复存储同一 token 不应增加计数 (HashSet)
        store.store_token("token1").await.unwrap();
        assert_eq!(store.token_count().await, 3);
    }

    #[tokio::test]
    async fn test_csrf_token_store_default() {
        let store = CsrfTokenStore::default();
        assert_eq!(store.token_count().await, 0);

        store.store_token("default-store-token").await;
        assert_eq!(store.token_count().await, 1);
        assert!(store.validate_token("default-store-token").await);
        // 一次性使用后应失效
        assert!(!store.validate_token("default-store-token").await);
    }

    #[tokio::test]
    async fn test_csrf_token_store_cleanup_expired_no_op() {
        // cleanup_expired 对内存存储是 no-op,不应 panic 也不应影响 token
        let store = CsrfTokenStore::new();
        store.store_token("cleanup-test").await;
        store.cleanup_expired(0).await;
        assert_eq!(store.token_count().await, 1);
    }

    #[tokio::test]
    async fn test_csrf_token_store_with_redis_returns_none_without_feature() {
        // 未启用 redis feature 时,with_redis 应返回 None
        let result = CsrfTokenStore::with_redis("redis://localhost:6379", None, 3600).await;
        assert!(result.is_none());
    }

    // ===== OriginValidator 错误路径测试 =====

    #[test]
    fn test_validate_origin_missing_header() {
        let headers = HeaderMap::new();
        let config = CsrfConfig::new(vec!["https://example.com".to_string()]);
        let result = OriginValidator::validate_origin(&headers, &config, "/api/test");
        assert_eq!(result, Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_validate_origin_invalid_format() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "not-a-valid-url".parse().unwrap());
        let config = CsrfConfig::new(vec!["https://example.com".to_string()]);
        let result = OriginValidator::validate_origin(&headers, &config, "/api/test");
        assert_eq!(result, Err(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn test_validate_origin_not_in_allowed_list() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://evil.com".parse().unwrap());
        let config = CsrfConfig::new(vec!["https://example.com".to_string()]);
        let result = OriginValidator::validate_origin(&headers, &config, "/api/test");
        assert_eq!(result, Err(StatusCode::FORBIDDEN));
    }

    #[test]
    fn test_validate_origin_allowed() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://example.com".parse().unwrap());
        let config = CsrfConfig::new(vec!["https://example.com".to_string()]);
        let result = OriginValidator::validate_origin(&headers, &config, "/api/test");
        assert_eq!(result, Ok("https://example.com".to_string()));
    }

    #[test]
    fn test_validate_origin_empty_allowed_set_allows_all() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://any-origin.com".parse().unwrap());
        // 空集合表示不限制
        let config = CsrfConfig::default();
        let result = OriginValidator::validate_origin(&headers, &config, "/api/test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_origin_with_port() {
        let parsed = OriginValidator::parse_origin("http://localhost:8080/path?q=1").unwrap();
        assert_eq!(parsed, "http://localhost:8080");
    }

    #[test]
    fn test_parse_origin_no_path() {
        let parsed = OriginValidator::parse_origin("https://api.example.com").unwrap();
        assert_eq!(parsed, "https://api.example.com");
    }

    #[test]
    fn test_parse_origin_invalid_no_host() {
        // 无 host 的 URL 应解析失败
        let result = OriginValidator::parse_origin("file:///path/to/file");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_origin_empty_string() {
        let result = OriginValidator::parse_origin("");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_referer_present() {
        let mut headers = HeaderMap::new();
        headers.insert("referer", "https://example.com/page".parse().unwrap());
        let referer = OriginValidator::get_referer(&headers);
        assert_eq!(referer, Some("https://example.com/page".to_string()));
    }

    #[test]
    fn test_get_referer_absent() {
        let headers = HeaderMap::new();
        let referer = OriginValidator::get_referer(&headers);
        assert_eq!(referer, None);
    }

    // ===== CsrfProtection 边界测试 =====

    #[test]
    fn test_is_same_origin_with_matching_origin() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://example.com".parse().unwrap());
        assert!(CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_is_same_origin_with_matching_origin_and_port() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://example.com:8443".parse().unwrap());
        assert!(CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_is_same_origin_with_mismatching_origin() {
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "https://evil.com".parse().unwrap());
        assert!(!CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_is_same_origin_fallback_to_referer() {
        let mut headers = HeaderMap::new();
        headers.insert("referer", "https://example.com/some/path".parse().unwrap());
        assert!(CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_is_same_origin_no_headers_assumes_same_origin() {
        // 无 Origin 和 Referer — 默认同源
        let headers = HeaderMap::new();
        assert!(CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_is_same_origin_invalid_origin_format() {
        // 格式错误的 Origin (无 "://") 应返回 false
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "invalid-origin".parse().unwrap());
        assert!(!CsrfProtection::is_same_origin(&headers, "example.com"));
    }

    #[test]
    fn test_validate_token_from_headers_x_csrf_token() {
        let mut headers = HeaderMap::new();
        headers.insert("X-CSRF-Token", "expected-token".parse().unwrap());
        assert!(CsrfProtection::validate_token_from_headers(
            &headers,
            "expected-token"
        ));
    }

    #[test]
    fn test_validate_token_from_headers_lowercase() {
        let mut headers = HeaderMap::new();
        headers.insert("x-csrf-token", "expected-token".parse().unwrap());
        assert!(CsrfProtection::validate_token_from_headers(
            &headers,
            "expected-token"
        ));
    }

    #[test]
    fn test_validate_token_from_headers_mismatch() {
        let mut headers = HeaderMap::new();
        headers.insert("X-CSRF-Token", "wrong-token".parse().unwrap());
        assert!(!CsrfProtection::validate_token_from_headers(
            &headers,
            "expected-token"
        ));
    }

    #[test]
    fn test_validate_token_from_headers_missing() {
        let headers = HeaderMap::new();
        assert!(!CsrfProtection::validate_token_from_headers(
            &headers,
            "expected-token"
        ));
    }

    #[test]
    fn test_validate_token_from_headers_empty_token() {
        let mut headers = HeaderMap::new();
        headers.insert("X-CSRF-Token", "".parse().unwrap());
        // 空 token 不等于 expected_token
        assert!(!CsrfProtection::validate_token_from_headers(
            &headers,
            "expected-token"
        ));
    }

    #[tokio::test]
    async fn test_csrf_token_store_concurrent_access() {
        // 并发存储和验证 — 确保线程安全
        let store = CsrfTokenStore::new();
        let mut handles = Vec::new();

        for i in 0..10 {
            let store_clone = store.clone();
            handles.push(tokio::spawn(async move {
                let token = format!("concurrent-token-{i}");
                store_clone.store_token(&token).await;
                assert!(store_clone.validate_token(&token).await);
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // 所有 token 验证后应已被移除 (一次性使用)
        assert_eq!(store.token_count().await, 0);
    }
}
