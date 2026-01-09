// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{Cache, CacheConfig, CacheEntry, CacheStats};

/// LRU 缓存实现
pub struct LruCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// 主存储
    store: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    /// 访问顺序（队首是最近访问的）
    access_order: Arc<RwLock<VecDeque<K>>>,
    /// 配置
    config: CacheConfig,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
}

impl<K, V> LruCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的 LRU 缓存
    pub fn new(config: CacheConfig) -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// 移除最久未使用的条目
    async fn evict_lru(&self) -> Option<(K, CacheEntry<V>)> {
        let mut order = self.access_order.write().await;
        if let Some(key) = order.pop_back() {
            let mut store = self.store.write().await;
            if let Some(entry) = store.remove(&key) {
                {
                    let mut stats = self.stats.write().await;
                    stats.evictions += 1;
                }
                return Some((key, entry));
            }
        }
        None
    }

    /// 更新访问顺序
    async fn update_access_order(&self, key: &K) {
        let mut order = self.access_order.write().await;

        // 移除现有位置（如果存在）
        if let Some(pos) = order.iter().position(|k| k == key) {
            order.remove(pos);
        }

        // 添加到队首（最近访问）
        order.push_front(key.clone());
    }

    /// 清理过期条目
    async fn cleanup_expired(&self) {
        let mut store = self.store.write().await;
        let mut order = self.access_order.write().await;
        let mut to_remove = Vec::new();

        for key in order.iter() {
            if let Some(entry) = store.get(key)
                && entry.is_expired(self.config.ttl_secs)
            {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            store.remove(&key);
            order.retain(|k| k != &key);
            debug!("Expired entry removed from LRU cache: {:?}", key);
        }
    }
}

#[async_trait::async_trait]
impl<K, V> Cache<K, V> for LruCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Option<V> {
        self.cleanup_expired().await;

        let mut store = self.store.write().await;
        if store.contains_key(key) {
            let entry = store.get_mut(key).unwrap();
            entry.record_access();
            let value = entry.value.clone();
            drop(store);

            self.update_access_order(key).await;

            {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                stats.total_requests += 1;
            }

            Some(value)
        } else {
            {
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                stats.total_requests += 1;
            }
            None
        }
    }

    async fn put(&self, key: K, value: V, size: usize) -> Result<(), String> {
        self.cleanup_expired().await;

        let mut store = self.store.write().await;

        // 如果键已存在，更新值
        if store.contains_key(&key) {
            let entry = CacheEntry::new(value, size);
            let key_clone = key.clone();
            store.insert(key, entry);
            drop(store);

            self.update_access_order(&key_clone).await;
            return Ok(());
        }

        // 检查容量
        if store.len() >= self.config.capacity {
            let key_clone = key.clone();
            drop(store);
            if let Some(_) = self.evict_lru().await {
                // 驱逐成功
            }
            let mut store = self.store.write().await;
            let entry = CacheEntry::new(value, size);
            store.insert(key, entry);
            drop(store);

            self.update_access_order(&key_clone).await;
        } else {
            let key_clone = key.clone();
            let entry = CacheEntry::new(value, size);
            store.insert(key, entry);
            drop(store);

            self.update_access_order(&key_clone).await;
        }

        {
            let mut stats = self.stats.write().await;
            let store = self.store.read().await;
            stats.current_size = store.len();
        }

        Ok(())
    }

    async fn remove(&self, key: &K) -> bool {
        let mut store = self.store.write().await;
        let mut order = self.access_order.write().await;

        if store.remove(key).is_some() {
            order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    async fn clear(&self) {
        let mut store = self.store.write().await;
        let mut order = self.access_order.write().await;
        store.clear();
        order.clear();

        {
            let mut stats = self.stats.write().await;
            stats.current_size = 0;
        }
    }

    async fn len(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }

    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    async fn stats(&self) -> CacheStats {
        let mut stats = self.stats.write().await;
        stats.current_size = self.len().await;
        stats.calculate_hit_rate();
        stats.clone()
    }

    async fn warm_up(
        &self,
        entries: std::collections::HashMap<K, (V, usize)>,
    ) -> Result<(), String> {
        let mut store = self.store.write().await;
        let mut order = self.access_order.write().await;

        for (key, (value, size)) in entries {
            if store.len() >= self.config.capacity {
                drop(store);
                drop(order);
                if let Some(_) = self.evict_lru().await {
                    // 驱逐成功
                }
                store = self.store.write().await;
                order = self.access_order.write().await;
            }

            let entry = CacheEntry::new(value, size);
            store.insert(key.clone(), entry);
            order.push_front(key);
        }

        {
            let mut stats = self.stats.write().await;
            stats.current_size = store.len();
        }

        info!("LRU cache warmed up with {} entries", store.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lru_basic() {
        let config = CacheConfig {
            capacity: 3,
            ..Default::default()
        };
        let cache = LruCache::new(config);

        // 插入三个条目
        cache
            .put("key1".to_string(), "value1".to_string(), 10)
            .await
            .unwrap();
        cache
            .put("key2".to_string(), "value2".to_string(), 10)
            .await
            .unwrap();
        cache
            .put("key3".to_string(), "value3".to_string(), 10)
            .await
            .unwrap();

        assert_eq!(cache.len().await, 3);

        // 访问 key1（应该更新为最近访问）
        assert_eq!(
            cache.get(&"key1".to_string()).await,
            Some("value1".to_string())
        );

        // 插入第四个条目，应该驱逐 key2
        cache
            .put("key4".to_string(), "value4".to_string(), 10)
            .await
            .unwrap();
        assert_eq!(cache.len().await, 3);
        assert!(cache.get(&"key2".to_string()).await.is_none());
        assert!(cache.get(&"key1".to_string()).await.is_some());
    }

    #[tokio::test]
    async fn test_lru_stats() {
        let config = CacheConfig::default();
        let cache = LruCache::new(config);

        cache
            .put("key1".to_string(), "value1".to_string(), 10)
            .await
            .unwrap();

        assert!(cache.get(&"key1".to_string()).await.is_some());
        assert!(cache.get(&"key2".to_string()).await.is_none());

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_requests, 2);
    }
}
