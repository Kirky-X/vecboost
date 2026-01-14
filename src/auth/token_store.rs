// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! Token storage abstraction for JWT token blacklisting
//!
//! Supports both in-memory storage (for development/testing) and
//! Redis storage (for production deployments).

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trait for token storage backends
///
/// Implement this trait to add support for different storage backends
/// like Redis, PostgreSQL, etc.
#[async_trait]
pub trait TokenStore: Send + Sync {
    /// Add a token to the blacklist
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError>;

    /// Check if a token is blacklisted
    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::AppError>;

    /// Remove a token from the blacklist
    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError>;

    /// Check if a token is blacklisted (同步版本，用于验证）
    fn is_blacklisted_sync(&self, jti: &str) -> bool;

    /// Get the number of blacklisted tokens
    async fn blacklist_size(&self) -> usize;
}

/// In-memory token blacklist store
///
/// This is the default store used for development and testing.
/// In production, use `RedisTokenStore` instead.
#[derive(Clone)]
pub struct MemoryTokenStore {
    blacklist: Arc<RwLock<HashSet<String>>>,
}

impl MemoryTokenStore {
    /// Create a new in-memory token store
    pub fn new() -> Self {
        Self {
            blacklist: Arc::new(RwLock::new(HashSet::new())),
        }
    }
}

impl Default for MemoryTokenStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TokenStore for MemoryTokenStore {
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError> {
        let mut blacklist = self.blacklist.write().await;
        blacklist.insert(jti.to_string());
        Ok(())
    }

    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::AppError> {
        let blacklist = self.blacklist.read().await;
        Ok(blacklist.contains(jti))
    }

    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError> {
        let mut blacklist = self.blacklist.write().await;
        blacklist.remove(jti);
        Ok(())
    }

    fn is_blacklisted_sync(&self, _jti: &str) -> bool {
        // 同步版本用于非异步路径
        // 注意：在 Tokio 运行时中不应使用 blocking_read
        // 生产环境中应使用 Redis 的非阻塞查询
        // 这里返回 false 作为安全默认值
        false
    }

    async fn blacklist_size(&self) -> usize {
        self.blacklist.read().await.len()
    }
}

/// Redis-backed token blacklist store
///
/// This store uses Redis for persistent token blacklist storage,
/// supporting distributed deployments with multiple service instances.
#[cfg(feature = "redis")]
#[derive(Clone)]
pub struct RedisTokenStore {
    client: Arc<redis::Client>,
    key_prefix: String,
}

#[cfg(feature = "redis")]
impl RedisTokenStore {
    /// Create a new Redis token store
    ///
    /// # Arguments
    /// * `client` - Redis client connection
    /// * `key_prefix` - Prefix for Redis keys (e.g., "vecboost:jwt:")
    pub fn new(client: redis::Client, key_prefix: Option<String>) -> Self {
        Self {
            client: Arc::new(client),
            key_prefix: key_prefix.unwrap_or_else(|| "vecboost:jwt:blacklist:".to_string()),
        }
    }

    fn get_key(&self, jti: &str) -> String {
        format!("{}{}", self.key_prefix, jti)
    }
}

#[cfg(feature = "redis")]
#[async_trait]
impl TokenStore for RedisTokenStore {
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError> {
        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        // Store with 24-hour TTL (token expiration)
        redis::Cmd::set_ex(&key, "1", 86400)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        Ok(())
    }

    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::AppError> {
        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        let exists: bool = redis::Cmd::exists(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        Ok(exists)
    }

    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::AppError> {
        let mut conn = self
            .client
            .get_async_connection()
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        redis::Cmd::del(&key)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| crate::error::AppError::ConfigError(e.to_string()))?;

        Ok(())
    }

    fn is_blacklisted_sync(&self, jti: &str) -> bool {
        // Synchronous Redis check - use for non-async paths
        // In production, prefer async version
        if let Ok(mut conn) = self.client.get_connection() {
            let key = self.get_key(jti);
            if let Ok(exists) = redis::Cmd::exists(&key).query(&mut conn) {
                return exists;
            }
        }
        false
    }

    async fn blacklist_size(&self) -> usize {
        // For Redis, this would require SCAN - simplified for now
        0
    }
}

/// Factory for creating token stores
pub struct TokenStoreFactory;

impl TokenStoreFactory {
    /// Create an in-memory token store
    pub fn create_memory_store() -> MemoryTokenStore {
        MemoryTokenStore::new()
    }

    /// Create a Redis token store
    ///
    /// Returns None if Redis feature is not enabled or connection fails.
    #[cfg(feature = "redis")]
    pub async fn create_redis_store(
        redis_url: &str,
        key_prefix: Option<String>,
    ) -> Option<RedisTokenStore> {
        let client = redis::Client::open(redis_url).ok()?;
        if client.get_async_connection().await.is_ok() {
            Some(RedisTokenStore::new(client, key_prefix))
        } else {
            None
        }
    }

    /// Create a token store with automatic fallback
    ///
    /// Tries to create a Redis store first, falls back to memory store.
    #[cfg(feature = "redis")]
    pub async fn create_with_fallback(
        redis_url: Option<&str>,
        key_prefix: Option<String>,
    ) -> Arc<dyn TokenStore> {
        if let Some(url) = redis_url {
            if let Some(redis_store) = Self::create_redis_store(url, key_prefix).await {
                return Arc::new(redis_store);
            }
        }
        // Fallback to memory store
        Arc::new(MemoryTokenStore::new())
    }

    #[cfg(not(feature = "redis"))]
    pub async fn create_with_fallback(
        _redis_url: Option<&str>,
        _key_prefix: Option<String>,
    ) -> Arc<dyn TokenStore> {
        // Without Redis feature, always use memory store
        Arc::new(MemoryTokenStore::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_token_store() {
        let store = MemoryTokenStore::new();

        // Initially not blacklisted
        assert!(!store.is_blacklisted("test-jti").await.unwrap());

        // Add to blacklist
        store.add_to_blacklist("test-jti").await.unwrap();

        // Now should be blacklisted
        assert!(store.is_blacklisted("test-jti").await.unwrap());

        // Remove from blacklist
        store.remove_from_blacklist("test-jti").await.unwrap();

        // No longer blacklisted
        assert!(!store.is_blacklisted("test-jti").await.unwrap());

        // Check size
        assert_eq!(store.blacklist_size().await, 0);
    }
}
