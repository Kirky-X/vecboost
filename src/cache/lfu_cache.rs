// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use super::{Cache, CacheConfig, CacheEntry, CacheStats};

/// LFU 缓存实现
pub struct LfuCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// 主存储
    store: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    /// 访问频率计数
    frequency: Arc<RwLock<HashMap<K, usize>>>,
    /// 配置
    config: CacheConfig,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
}

impl<K, V> LfuCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的 LFU 缓存
    pub fn new(config: CacheConfig) -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            frequency: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// 移除最少使用的条目
    async fn evict_lfu(&self) -> Option<(K, CacheEntry<V>)> {
        let mut freq = self.frequency.write().await;
        let mut store = self.store.write().await;

        // 找到访问频率最低的键
        let min_freq_key = freq
            .iter()
            .min_by_key(|(_, count)| **count)
            .map(|(key, _)| key.clone());

        if let Some(key) = min_freq_key {
            freq.remove(&key);
            if let Some(entry) = store.remove(&key) {
                {
                    let mut stats = self.stats.write().await;
                    stats.evictions += 1;
                }
                debug!("LFU evicted entry: {:?}", key);
                return Some((key, entry));
            }
        }

        None
    }

    /// 清理过期条目
    async fn cleanup_expired(&self) {
        let mut store = self.store.write().await;
        let mut freq = self.frequency.write().await;
        let mut to_remove = Vec::new();

        for key in store.keys() {
            if let Some(entry) = store.get(key) {
                if entry.is_expired(self.config.ttl_secs) {
                    to_remove.push(key.clone());
                }
            }
        }

        for key in to_remove {
            store.remove(&key);
            freq.remove(&key);
            debug!("Expired entry removed from LFU cache: {:?}", key);
        }
    }
}

#[async_trait::async_trait]
impl<K, V> Cache<K, V> for LfuCache<K, V>
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

            // 增加访问频率
            let mut freq = self.frequency.write().await;
            *freq.entry(key.clone()).or_insert(0) += 1;
            drop(freq);

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
        let mut freq = self.frequency.write().await;

        // 如果键已存在，更新值
        if store.contains_key(&key) {
            let entry = CacheEntry::new(value, size);
            store.insert(key, entry);
            return Ok(());
        }

        // 检查容量
        if store.len() >= self.config.capacity {
            drop(store);
            drop(freq);
            if let Some(_) = self.evict_lfu().await {
                // 驱逐成功
            }
            store = self.store.write().await;
            freq = self.frequency.write().await;
        }

        let entry = CacheEntry::new(value, size);
        store.insert(key.clone(), entry);
        freq.insert(key, 1);

        {
            let mut stats = self.stats.write().await;
            stats.current_size = store.len();
        }

        Ok(())
    }

    async fn remove(&self, key: &K) -> bool {
        let mut store = self.store.write().await;
        let mut freq = self.frequency.write().await;

        if store.remove(key).is_some() {
            freq.remove(key);
            true
        } else {
            false
        }
    }

    async fn clear(&self) {
        let mut store = self.store.write().await;
        let mut freq = self.frequency.write().await;
        store.clear();
        freq.clear();

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
        let mut freq = self.frequency.write().await;

        for (key, (value, size)) in entries {
            if store.len() >= self.config.capacity {
                drop(store);
                drop(freq);
                if let Some(_) = self.evict_lfu().await {
                    // 驱逐成功
                }
                store = self.store.write().await;
                freq = self.frequency.write().await;
            }

            let entry = CacheEntry::new(value, size);
            store.insert(key.clone(), entry);
            freq.insert(key, 1);
        }

        {
            let mut stats = self.stats.write().await;
            stats.current_size = store.len();
        }

        debug!("LFU cache warmed up with {} entries", store.len());
        Ok(())
    }
}
