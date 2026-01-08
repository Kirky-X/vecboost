// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

use super::{Cache, CacheConfig, CacheEntry, CacheStats};

/// ARC (Adaptive Replacement Cache) 缓存实现
///
/// ARC 自适应地平衡 LRU 和 LFU 策略
pub struct ArcCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// 主存储
    store: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    /// T1 (最近不频繁使用) 队列
    t1: Arc<RwLock<VecDeque<K>>>,
    /// T2 (最近频繁使用) 队列
    t2: Arc<RwLock<VecDeque<K>>>,
    /// B1 (被驱逐的不频繁使用) 历史记录
    b1: Arc<RwLock<VecDeque<K>>>,
    /// B2 (被驱逐的频繁使用) 历史记录
    b2: Arc<RwLock<VecDeque<K>>>,
    /// T1 的目标大小
    p: Arc<RwLock<usize>>,
    /// 配置
    config: CacheConfig,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
}

impl<K, V> ArcCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的 ARC 缓存
    pub fn new(config: CacheConfig) -> Self {
        let capacity = config.capacity;
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            t1: Arc::new(RwLock::new(VecDeque::new())),
            t2: Arc::new(RwLock::new(VecDeque::new())),
            b1: Arc::new(RwLock::new(VecDeque::new())),
            b2: Arc::new(RwLock::new(VecDeque::new())),
            p: Arc::new(RwLock::new(capacity / 2)),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// 获取键所在的队列
    async fn locate_queue(&self, key: &K) -> Option<(usize, VecDeque<K>)> {
        let t1 = self.t1.read().await;
        if t1.contains(key) {
            return Some((1, VecDeque::new()));
        }

        drop(t1);

        let t2 = self.t2.read().await;
        if t2.contains(key) {
            return Some((2, VecDeque::new()));
        }

        drop(t2);

        let b1 = self.b1.read().await;
        if b1.contains(key) {
            return Some((3, VecDeque::new()));
        }

        drop(b1);

        let b2 = self.b2.read().await;
        if b2.contains(key) {
            return Some((4, VecDeque::new()));
        }

        None
    }

    /// 替换算法
    async fn replace(&self, key: &K) -> Result<(), String> {
        let p = *self.p.read().await;
        let t1_len = self.t1.read().await.len();

        if t1_len >= 1
            && ((self.t1.read().await.contains(key) && t1_len == p)
                || (self.b2.read().await.contains(key) && t1_len > p))
        {
            // 从 T1 驱逐到 B1
            let mut t1 = self.t1.write().await;
            if let Some(evicted_key) = t1.pop_back() {
                let mut b1 = self.b1.write().await;
                b1.push_back(evicted_key.clone());
                if b1.len() > p + self.b2.read().await.len() {
                    b1.pop_front();
                }

                let mut store = self.store.write().await;
                if store.remove(&evicted_key).is_some() {
                    let mut stats = self.stats.write().await;
                    stats.evictions += 1;
                }
                debug!("ARC: evicted from T1 to B1: {:?}", evicted_key);
            }
        } else {
            // 从 T2 驱逐到 B2
            let mut t2 = self.t2.write().await;
            if let Some(evicted_key) = t2.pop_back() {
                let mut b2 = self.b2.write().await;
                b2.push_back(evicted_key.clone());
                if b2.len() > self.config.capacity - p + self.b1.read().await.len() {
                    b2.pop_front();
                }

                let mut store = self.store.write().await;
                if store.remove(&evicted_key).is_some() {
                    let mut stats = self.stats.write().await;
                    stats.evictions += 1;
                }
                debug!("ARC: evicted from T2 to B2: {:?}", evicted_key);
            }
        }

        Ok(())
    }

    /// 清理过期条目
    async fn cleanup_expired(&self) {
        let mut store = self.store.write().await;
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
            self.t1.write().await.retain(|k| k != &key);
            self.t2.write().await.retain(|k| k != &key);
        }
    }
}

#[async_trait::async_trait]
impl<K, V> Cache<K, V> for ArcCache<K, V>
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

            let t1 = self.t1.read().await;
            let exists_in_t1 = t1.contains(key);
            drop(t1);

            // 根据位置移动到 T2
            if exists_in_t1 {
                let mut t1 = self.t1.write().await;
                t1.retain(|k| k != key);
                drop(t1);

                let mut t2 = self.t2.write().await;
                t2.push_front(key.clone());
            } else {
                // 已经在 T2 中，移到队首
                let mut t2 = self.t2.write().await;
                t2.retain(|k| k != key);
                t2.push_front(key.clone());
            }

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
        let b1_contains = self.b1.read().await.contains(&key);
        let b2_contains = self.b2.read().await.contains(&key);

        // 如果键已存在，更新值
        if store.contains_key(&key) {
            let entry = CacheEntry::new(value, size);
            store.insert(key, entry);
            return Ok(());
        }

        // 如果在 B1 中，调整 p
        if b1_contains {
            let delta = self.b2.read().await.len() / self.b1.read().await.len();
            let delta = delta.max(1);
            let new_p = *self.p.read().await + delta;
            *self.p.write().await = new_p.min(self.config.capacity);
        }
        // 如果在 B2 中，调整 p
        else if b2_contains {
            let delta = self.b1.read().await.len() / self.b2.read().await.len();
            let delta = delta.max(1);
            let current_p = *self.p.read().await;
            let new_p = current_p.checked_sub(delta).unwrap_or(0);
            *self.p.write().await = new_p;
        }

        // 检查是否需要替换
        let t1_len = self.t1.read().await.len();
        let t2_len = self.t2.read().await.len();
        let total = t1_len + t2_len;

        if total >= self.config.capacity {
            drop(store);
            self.replace(&key).await?;
            store = self.store.write().await;
        }

        // 插入到 T1
        store.insert(key.clone(), CacheEntry::new(value, size));
        self.t1.write().await.push_front(key);

        {
            let mut stats = self.stats.write().await;
            stats.current_size = store.len();
        }

        Ok(())
    }

    async fn remove(&self, key: &K) -> bool {
        let mut store = self.store.write().await;

        if store.remove(key).is_some() {
            self.t1.write().await.retain(|k| k != key);
            self.t2.write().await.retain(|k| k != key);
            self.b1.write().await.retain(|k| k != key);
            self.b2.write().await.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    async fn clear(&self) {
        let mut store = self.store.write().await;
        self.t1.write().await.clear();
        self.t2.write().await.clear();
        self.b1.write().await.clear();
        self.b2.write().await.clear();
        store.clear();

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
        for (key, (value, size)) in entries {
            if self.len().await >= self.config.capacity {
                self.replace(&key).await?;
            }
            self.put(key, value, size).await?;
        }

        debug!("ARC cache warmed up with {} entries", self.len().await);
        Ok(())
    }
}
