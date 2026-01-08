// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use lru::LruCache;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::RwLock;
use xxhash_rust::xxh3::Xxh3;

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub embedding: Vec<f32>,
    pub timestamp: u64,
    pub access_count: u64,
}

pub struct KvCache {
    cache: Arc<RwLock<LruCache<u64, CacheEntry>>>,
    #[allow(dead_code)]
    max_size: NonZeroUsize,
    enabled: bool,
}

impl KvCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::MIN),
            ))),
            max_size: NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::MIN),
            enabled: true,
        }
    }

    #[allow(dead_code)]
    pub fn with_capacity(max_size: usize) -> Self {
        Self::new(max_size)
    }

    pub fn disabled() -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(NonZeroUsize::MIN))),
            max_size: NonZeroUsize::MIN,
            enabled: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }

        let hash = self.compute_hash(key);
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(&hash) {
            entry.timestamp = Self::current_timestamp();
            entry.access_count += 1;
            Some(entry.embedding.clone())
        } else {
            None
        }
    }

    pub async fn put(&self, key: &str, embedding: Vec<f32>) {
        if !self.enabled {
            return;
        }

        let hash = self.compute_hash(key);
        let entry = CacheEntry {
            embedding,
            timestamp: Self::current_timestamp(),
            access_count: 1,
        };

        let mut cache = self.cache.write().await;
        cache.put(hash, entry);
    }

    pub async fn get_or_insert<F, Fut, E>(&self, key: &str, f: F) -> Result<Vec<f32>, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, E>>,
    {
        if let Some(cached) = self.get(key).await {
            return Ok(cached);
        }

        let embedding = f().await?;
        self.put(key, embedding.clone()).await;
        Ok(embedding)
    }

    #[allow(dead_code)]
    pub async fn contains(&self, key: &str) -> bool {
        if !self.enabled {
            return false;
        }

        let hash = self.compute_hash(key);
        let cache = self.cache.read().await;
        cache.contains(&hash)
    }

    #[allow(dead_code)]
    pub async fn remove(&self, key: &str) -> bool {
        if !self.enabled {
            return false;
        }

        let hash = self.compute_hash(key);
        let mut cache = self.cache.write().await;
        cache.pop(&hash).is_some()
    }

    #[allow(dead_code)]
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    #[allow(dead_code)]
    pub async fn len(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }

    #[allow(dead_code)]
    pub async fn is_empty(&self) -> bool {
        let cache = self.cache.read().await;
        cache.is_empty()
    }

    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.max_size.get()
    }

    #[cfg(test)]
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total_access_count: u64 = cache.iter().map(|(_, e)| e.access_count).sum();
        let oldest_timestamp = cache.iter().map(|(_, e)| e.timestamp).min();
        let newest_timestamp = cache.iter().map(|(_, e)| e.timestamp).max();

        CacheStats {
            size: cache.len(),
            capacity: self.max_size.get(),
            total_access_count,
            oldest_timestamp,
            newest_timestamp,
            hit_rate: None,
            miss_count: None,
        }
    }

    /// 缓存预热：批量插入缓存条目
    /// 用于在服务启动时预加载热点数据
    pub async fn warm_up(&self, entries: HashMap<String, Vec<f32>>) {
        if !self.enabled {
            return;
        }

        let mut cache = self.cache.write().await;

        let now = Self::current_timestamp();

        let entries_count = entries.len();

        for (key, embedding) in entries {
            let hash = self.compute_hash(&key);
            let entry = CacheEntry {
                embedding,
                timestamp: now,
                access_count: 1,
            };
            cache.put(hash, entry);
        }

        tracing::info!("Cache warmed up with {} entries", entries_count);
    }

    fn compute_hash(&self, key: &str) -> u64 {
        let mut hasher = Xxh3::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub total_access_count: u64,
    pub oldest_timestamp: Option<u64>,
    pub newest_timestamp: Option<u64>,
    pub hit_rate: Option<f64>,
    pub miss_count: Option<u64>,
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    hits: u64,
    misses: u64,
}

#[cfg(test)]
impl CacheMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.total_requests();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}
