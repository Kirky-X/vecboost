// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! Token storage abstraction for JWT token blacklisting
//!
//! Supports both in-memory storage (for development/testing) and
//! Redis storage (for production deployments).

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// 内存黑名单默认 TTL(秒)— 与 Redis 版本的 24h TTL 保持一致
const MEMORY_BLACKLIST_TTL_SECS: u64 = 86_400;

/// 获取当前 Unix 时间戳(秒)
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Trait for token storage backends
///
/// Implement this trait to add support for different storage backends
/// like Redis, PostgreSQL, etc.
#[async_trait]
pub trait TokenStore: Send + Sync {
    /// Add a token to the blacklist
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError>;

    /// Check if a token is blacklisted
    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::VecboostError>;

    /// Remove a token from the blacklist
    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError>;

    /// Check if a token is blacklisted (同步版本，用于验证）
    fn is_blacklisted_sync(&self, jti: &str) -> bool;

    /// Get the number of blacklisted tokens
    async fn blacklist_size(&self) -> usize;

    /// Clean up expired tokens from the blacklist
    async fn cleanup_expired_blacklist(&self);
}

/// In-memory token blacklist store
///
/// This is the default store used for development and testing.
/// In production, use `RedisTokenStore` instead.
#[derive(Clone)]
pub struct MemoryTokenStore {
    /// JTI → 过期 Unix 时间戳(秒)
    blacklist: Arc<RwLock<HashMap<String, u64>>>,
}

impl MemoryTokenStore {
    /// Create a new in-memory token store
    pub fn new() -> Self {
        Self {
            blacklist: Arc::new(RwLock::new(HashMap::new())),
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
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError> {
        let mut blacklist = self.blacklist.write().await;
        blacklist.insert(jti.to_string(), now_secs() + MEMORY_BLACKLIST_TTL_SECS);
        Ok(())
    }

    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::VecboostError> {
        let blacklist = self.blacklist.read().await;
        if let Some(expiry) = blacklist.get(jti)
            && *expiry > now_secs()
        {
            return Ok(true);
        }
        Ok(false)
    }

    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError> {
        let mut blacklist = self.blacklist.write().await;
        blacklist.remove(jti);
        Ok(())
    }

    fn is_blacklisted_sync(&self, jti: &str) -> bool {
        // 使用 try_read 避免在异步上下文中阻塞 panic
        // 锁竞争时返回 false(安全默认值,异步路径会再次校验)
        if let Ok(blacklist) = self.blacklist.try_read()
            && let Some(expiry) = blacklist.get(jti)
            && *expiry > now_secs()
        {
            return true;
        }
        false
    }

    async fn blacklist_size(&self) -> usize {
        let blacklist = self.blacklist.read().await;
        let now = now_secs();
        blacklist.values().filter(|exp| **exp > now).count()
    }

    async fn cleanup_expired_blacklist(&self) {
        let mut blacklist = self.blacklist.write().await;
        let now = now_secs();
        blacklist.retain(|_, exp| *exp > now);
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
    async fn add_to_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        // Store with 24-hour TTL (token expiration)
        redis::Cmd::set_ex(&key, "1", 86400)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

        Ok(())
    }

    async fn is_blacklisted(&self, jti: &str) -> Result<bool, crate::error::VecboostError> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        let exists: bool = redis::Cmd::exists(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

        Ok(exists)
    }

    async fn remove_from_blacklist(&self, jti: &str) -> Result<(), crate::error::VecboostError> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

        let key = self.get_key(jti);
        redis::Cmd::del(&key)
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| crate::error::VecboostError::ConfigError(e.to_string()))?;

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

    async fn cleanup_expired_blacklist(&self) {
        // 对于 Redis 存储，过期由 TTL 自动处理
        // 无需手动清理
        tracing::debug!("Redis token store cleanup - expiration handled by TTL");
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
        if client.get_multiplexed_async_connection().await.is_ok() {
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
        if let Some(url) = redis_url
            && let Some(redis_store) = Self::create_redis_store(url, key_prefix).await
        {
            return Arc::new(redis_store);
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

    #[tokio::test]
    async fn test_is_blacklisted_sync_queries_actual_blacklist() {
        let store = MemoryTokenStore::new();
        assert!(!store.is_blacklisted_sync("sync-jti"));
        store.add_to_blacklist("sync-jti").await.unwrap();
        assert!(store.is_blacklisted_sync("sync-jti"));
        store.remove_from_blacklist("sync-jti").await.unwrap();
        assert!(!store.is_blacklisted_sync("sync-jti"));
    }

    #[tokio::test]
    async fn test_cleanup_expired_removes_expired_entries() {
        let store = MemoryTokenStore::new();
        // 手动注入已过期条目
        {
            let mut bl = store.blacklist.write().await;
            bl.insert("expired-jti".to_string(), now_secs().saturating_sub(1));
            bl.insert(
                "valid-jti".to_string(),
                now_secs() + MEMORY_BLACKLIST_TTL_SECS,
            );
        }
        assert_eq!(store.blacklist_size().await, 1);
        store.cleanup_expired_blacklist().await;
        assert!(!store.is_blacklisted("expired-jti").await.unwrap());
        assert!(store.is_blacklisted("valid-jti").await.unwrap());
        assert_eq!(store.blacklist_size().await, 1);
    }

    // ===== MemoryTokenStore 边界测试 =====

    #[tokio::test]
    async fn test_memory_token_store_default() {
        let store = MemoryTokenStore::default();
        assert_eq!(store.blacklist_size().await, 0);
        assert!(!store.is_blacklisted("any").await.unwrap());
    }

    #[tokio::test]
    async fn test_add_duplicate_jti_idempotent() {
        // 重复添加同一 jti 不应增加计数
        let store = MemoryTokenStore::new();
        store.add_to_blacklist("dup-jti").await.unwrap();
        store.add_to_blacklist("dup-jti").await.unwrap();
        assert_eq!(store.blacklist_size().await, 1);
        assert!(store.is_blacklisted("dup-jti").await.unwrap());
    }

    #[tokio::test]
    async fn test_remove_nonexistent_jti_succeeds() {
        // 删除不存在的 jti 不应报错
        let store = MemoryTokenStore::new();
        let result = store.remove_from_blacklist("nonexistent").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_blacklist_size_with_expired_entries() {
        let store = MemoryTokenStore::new();
        // 注入一个已过期和一个有效条目
        {
            let mut bl = store.blacklist.write().await;
            bl.insert("expired".to_string(), now_secs().saturating_sub(100));
            bl.insert("valid".to_string(), now_secs() + 3600);
        }
        // blacklist_size 只计算未过期的
        assert_eq!(store.blacklist_size().await, 1);
    }

    #[tokio::test]
    async fn test_is_blacklisted_with_expired_entry() {
        // 已过期的条目应返回 false
        let store = MemoryTokenStore::new();
        {
            let mut bl = store.blacklist.write().await;
            bl.insert("expired-jti".to_string(), now_secs().saturating_sub(1));
        }
        assert!(!store.is_blacklisted("expired-jti").await.unwrap());
    }

    #[tokio::test]
    async fn test_empty_jti_handled() {
        let store = MemoryTokenStore::new();
        // 空 jti 应正常处理
        store.add_to_blacklist("").await.unwrap();
        assert!(store.is_blacklisted("").await.unwrap());
        assert_eq!(store.blacklist_size().await, 1);
    }

    #[tokio::test]
    async fn test_long_jti_handled() {
        let store = MemoryTokenStore::new();
        let long_jti = "j".repeat(1000);
        store.add_to_blacklist(&long_jti).await.unwrap();
        assert!(store.is_blacklisted(&long_jti).await.unwrap());
    }

    #[tokio::test]
    async fn test_special_chars_jti_handled() {
        let store = MemoryTokenStore::new();
        let special_jti = "jti-with-special:chars/and.symbols#";
        store.add_to_blacklist(special_jti).await.unwrap();
        assert!(store.is_blacklisted(special_jti).await.unwrap());
    }

    #[tokio::test]
    async fn test_is_blacklisted_sync_with_expired_entry() {
        // is_blacklisted_sync 应正确处理过期条目
        let store = MemoryTokenStore::new();
        {
            let mut bl = store.blacklist.write().await;
            bl.insert("expired-sync".to_string(), now_secs().saturating_sub(1));
        }
        assert!(!store.is_blacklisted_sync("expired-sync"));
    }

    #[tokio::test]
    async fn test_is_blacklisted_sync_empty_jti() {
        let store = MemoryTokenStore::new();
        assert!(!store.is_blacklisted_sync(""));
        store.add_to_blacklist("").await.unwrap();
        assert!(store.is_blacklisted_sync(""));
    }

    #[tokio::test]
    async fn test_cleanup_expired_on_empty_store() {
        // 对空存储执行清理不应 panic
        let store = MemoryTokenStore::new();
        store.cleanup_expired_blacklist().await;
        assert_eq!(store.blacklist_size().await, 0);
    }

    #[tokio::test]
    async fn test_cleanup_expired_all_entries() {
        // 所有条目都过期时,清理后应为空
        let store = MemoryTokenStore::new();
        {
            let mut bl = store.blacklist.write().await;
            bl.insert("exp1".to_string(), now_secs().saturating_sub(1));
            bl.insert("exp2".to_string(), now_secs().saturating_sub(100));
            bl.insert("exp3".to_string(), 0); // epoch 时间,已过期
        }
        store.cleanup_expired_blacklist().await;
        assert_eq!(store.blacklist_size().await, 0);
    }

    // ===== TokenStoreFactory 测试 =====

    #[test]
    fn test_create_memory_store() {
        let store = TokenStoreFactory::create_memory_store();
        // MemoryTokenStore 创建后应为空 (需要 async 检查,但 struct 创建本身不失败)
        let _ = store;
    }

    #[tokio::test]
    async fn test_create_with_fallback_no_redis() {
        // 未启用 redis feature 时,始终返回内存存储
        let store = TokenStoreFactory::create_with_fallback(None, None).await;
        assert_eq!(store.blacklist_size().await, 0);
        store.add_to_blacklist("fallback-test").await.unwrap();
        assert!(store.is_blacklisted("fallback-test").await.unwrap());
    }

    #[tokio::test]
    async fn test_create_with_fallback_some_url_no_redis() {
        // 传入 URL 但无 redis feature,仍应回退到内存存储
        let store =
            TokenStoreFactory::create_with_fallback(Some("redis://localhost:6379"), None).await;
        assert_eq!(store.blacklist_size().await, 0);
    }

    // ===== 并发访问测试 =====

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_blacklist_operations() {
        // 并发添加和查询 — 确保线程安全
        let store = Arc::new(MemoryTokenStore::new());
        let mut handles = Vec::new();

        for i in 0..20 {
            let store_clone = store.clone();
            handles.push(tokio::spawn(async move {
                let jti = format!("concurrent-jti-{i}");
                store_clone.add_to_blacklist(&jti).await.unwrap();
                assert!(store_clone.is_blacklisted(&jti).await.unwrap());
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(store.blacklist_size().await, 20);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_add_and_remove() {
        // 并发添加和删除同一 jti — 不应 panic
        let store = Arc::new(MemoryTokenStore::new());
        let jti = "race-jti";

        let store_add = store.clone();
        let store_remove = store.clone();

        let h1 = tokio::spawn(async move {
            for _ in 0..10 {
                store_add.add_to_blacklist(jti).await.unwrap();
            }
        });

        let h2 = tokio::spawn(async move {
            for _ in 0..10 {
                let _ = store_remove.remove_from_blacklist(jti).await;
            }
        });

        h1.await.unwrap();
        h2.await.unwrap();
        // 最终状态不确定 (取决于调度),但不应 panic
        let _ = store.is_blacklisted(jti).await.unwrap();
    }
}
