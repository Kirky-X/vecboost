// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::{ArcCache, Cache, CacheConfig, CacheStats, LfuCache, LruCache};

/// 分层缓存级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CacheLevel {
    /// L1 缓存（最快，最小）
    L1 = 1,
    /// L2 缓存
    L2 = 2,
    /// L3 缓存（最慢，最大）
    L3 = 3,
}

/// 分层缓存配置
#[derive(Debug, Clone)]
pub struct TieredCacheConfig {
    /// L1 缓存配置
    pub l1_config: CacheConfig,
    /// L2 缓存配置
    pub l2_config: Option<CacheConfig>,
    /// L3 缓存配置
    pub l3_config: Option<CacheConfig>,
    /// 是否启用自动提升（将数据提升到更高层级）
    pub enable_promotion: bool,
    /// 提升阈值（访问次数达到此值时提升）
    pub promotion_threshold: u64,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            l1_config: CacheConfig {
                capacity: 1000,
                ttl_secs: 600, // 10分钟
                enable_stats: true,
                strategy: Default::default(),
            },
            l2_config: Some(CacheConfig {
                capacity: 10000,
                ttl_secs: 3600, // 1小时
                enable_stats: true,
                strategy: Default::default(),
            }),
            l3_config: Some(CacheConfig {
                capacity: 100000,
                ttl_secs: 86400, // 24小时
                enable_stats: true,
                strategy: Default::default(),
            }),
            enable_promotion: true,
            promotion_threshold: 5,
        }
    }
}

/// 分层缓存（多级缓存）
///
/// L1 -> L2 -> L3
/// 访问 L1 如果未命中则尝试 L2，再未命中则尝试 L3
pub struct TieredCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// L1 缓存
    l1: Arc<Box<dyn Cache<K, V> + Send + Sync>>,
    /// L2 缓存（可选）
    l2: Option<Arc<Box<dyn Cache<K, V> + Send + Sync>>>,
    /// L3 缓存（可选）
    l3: Option<Arc<Box<dyn Cache<K, V> + Send + Sync>>>,
    /// 配置
    config: TieredCacheConfig,
    /// 统计信息
    stats: Arc<RwLock<CacheStats>>,
}

impl<K, V> TieredCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的分层缓存
    pub fn new(config: TieredCacheConfig) -> Self {
        let l1: Arc<Box<dyn Cache<K, V> + Send + Sync>> = match config.l1_config.strategy {
            super::CacheStrategy::Lru => {
                Arc::new(Box::new(LruCache::new(config.l1_config.clone())))
            }
            super::CacheStrategy::Lfu => {
                Arc::new(Box::new(LfuCache::new(config.l1_config.clone())))
            }
            super::CacheStrategy::Arc => {
                Arc::new(Box::new(ArcCache::new(config.l1_config.clone())))
            }
            super::CacheStrategy::TwoQueue => {
                Arc::new(Box::new(LruCache::new(config.l1_config.clone())))
            }
        };

        let l2 = config.l2_config.as_ref().map(|cfg| {
            let cache: Box<dyn Cache<K, V> + Send + Sync> = match cfg.strategy {
                super::CacheStrategy::Lru => Box::new(LruCache::new(cfg.clone())),
                super::CacheStrategy::Lfu => Box::new(LfuCache::new(cfg.clone())),
                super::CacheStrategy::Arc => Box::new(ArcCache::new(cfg.clone())),
                super::CacheStrategy::TwoQueue => Box::new(LruCache::new(cfg.clone())),
            };
            Arc::new(cache)
        });

        let l3 = config.l3_config.as_ref().map(|cfg| {
            let cache: Box<dyn Cache<K, V> + Send + Sync> = match cfg.strategy {
                super::CacheStrategy::Lru => Box::new(LruCache::new(cfg.clone())),
                super::CacheStrategy::Lfu => Box::new(LfuCache::new(cfg.clone())),
                super::CacheStrategy::Arc => Box::new(ArcCache::new(cfg.clone())),
                super::CacheStrategy::TwoQueue => Box::new(LruCache::new(cfg.clone())),
            };
            Arc::new(cache)
        });

        Self {
            l1,
            l2,
            l3,
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// 从指定级别获取缓存
    async fn get_from_level(&self, key: &K, level: CacheLevel) -> Option<V> {
        match level {
            CacheLevel::L1 => self.l1.get(key).await,
            CacheLevel::L2 => self.l2.as_ref()?.get(key).await,
            CacheLevel::L3 => self.l3.as_ref()?.get(key).await,
        }
    }

    /// 提升数据到更高层级
    async fn promote(&self, key: &K, value: V, size: usize, current_level: CacheLevel) {
        if !self.config.enable_promotion {
            return;
        }

        match current_level {
            CacheLevel::L2 => {
                // 从 L2 提升到 L1
                if let Err(e) = self.l1.put(key.clone(), value, size).await {
                    debug!("Failed to promote to L1: {}", e);
                } else {
                    info!("Promoted key {:?} from L2 to L1", key);
                }
            }
            CacheLevel::L3 => {
                // 从 L3 提升到 L2
                if let Some(l2) = &self.l2 {
                    if let Err(e) = l2.put(key.clone(), value.clone(), size).await {
                        debug!("Failed to promote to L2: {}", e);
                    } else {
                        info!("Promoted key {:?} from L3 to L2", key);
                    }
                }
            }
            CacheLevel::L1 => {
                // 已经在最高层级
            }
        }
    }

    /// 获取缓存条目并记录访问
    async fn get_with_entry(&self, key: &K, level: CacheLevel) -> Option<(V, u64)> {
        // 这里需要扩展 Cache trait 以支持获取访问次数
        // 暂时简化实现
        let value = self.get_from_level(key, level).await?;
        Some((value, 1))
    }
}

#[async_trait::async_trait]
impl<K, V> Cache<K, V> for TieredCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Option<V> {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;

        // 尝试从 L1 获取
        if let Some(value) = self.l1.get(key).await {
            stats.hits += 1;
            return Some(value);
        }

        // 尝试从 L2 获取
        if let Some(l2) = &self.l2 {
            if let Some(value) = l2.get(key).await {
                stats.hits += 1;
                drop(stats);

                // 提升到 L1
                let size = 100; // 简化：假设默认大小
                let _ = self.l1.put(key.clone(), value.clone(), size).await;

                info!("Cache hit in L2, promoted to L1: {:?}", key);
                return Some(value);
            }
        }

        // 尝试从 L3 获取
        if let Some(_l3) = &self.l3 {
            if let Some((value, _access_count)) = self.get_with_entry(key, CacheLevel::L3).await {
                stats.hits += 1;
                drop(stats);

                // 提升到 L2
                let size = 100;
                let _ = self.promote(key, value.clone(), size, CacheLevel::L3).await;

                info!("Cache hit in L3, promoted to L2: {:?}", key);
                return Some(value);
            }
        }

        // 未命中
        stats.misses += 1;
        None
    }

    async fn put(&self, key: K, value: V, size: usize) -> Result<(), String> {
        // 默认插入 L1
        self.l1.put(key.clone(), value.clone(), size).await?;

        // 如果 L1 已满，尝试插入 L2
        if let Some(l2) = &self.l2 {
            if let Err(_) = self.l1.put(key.clone(), value.clone(), size).await {
                debug!("L1 full, inserting to L2: {:?}", key);
                if let Err(_e) = l2.put(key.clone(), value.clone(), size).await {
                    // L2 也满了，尝试 L3
                    if let Some(l3) = &self.l3 {
                        debug!("L2 full, inserting to L3: {:?}", key);
                        l3.put(key, value, size).await?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn remove(&self, key: &K) -> bool {
        let mut removed = false;

        if self.l1.remove(key).await {
            removed = true;
        }
        if let Some(l2) = &self.l2 {
            if l2.remove(key).await {
                removed = true;
            }
        }
        if let Some(l3) = &self.l3 {
            if l3.remove(key).await {
                removed = true;
            }
        }

        removed
    }

    async fn clear(&self) {
        self.l1.clear().await;
        if let Some(l2) = &self.l2 {
            l2.clear().await;
        }
        if let Some(l3) = &self.l3 {
            l3.clear().await;
        }

        let mut stats = self.stats.write().await;
        stats.current_size = 0;
    }

    async fn len(&self) -> usize {
        let mut total = self.l1.len().await;
        if let Some(l2) = &self.l2 {
            total += l2.len().await;
        }
        if let Some(l3) = &self.l3 {
            total += l3.len().await;
        }
        total
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
        info!("Warming up tiered cache with {} entries", entries.len());

        // 首先将所有数据预热到 L3（最大层级）
        if let Some(l3) = &self.l3 {
            l3.warm_up(entries.clone()).await?;
        }

        // 然后将热门数据预热到 L2
        if let Some(l2) = &self.l2 {
            // 使用 config 中的容量来决定预热条目数
            let l2_capacity = self
                .config
                .l2_config
                .as_ref()
                .map(|c| c.capacity)
                .unwrap_or(1000);
            let l2_entries: HashMap<_, _> = entries
                .iter()
                .take(l2_capacity / 10)
                .map(|(k, (v, s))| (k.clone(), (v.clone(), *s)))
                .collect();
            l2.warm_up(l2_entries).await?;
        }

        // 最热的数据预热到 L1
        let l1_capacity = self.config.l1_config.capacity;
        let l1_entries: HashMap<_, _> = entries
            .iter()
            .take(l1_capacity / 10)
            .map(|(k, (v, s))| (k.clone(), (v.clone(), *s)))
            .collect();
        self.l1.warm_up(l1_entries).await?;

        info!("Tiered cache warm-up completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tiered_cache_basic() {
        let config = TieredCacheConfig::default();
        let cache = TieredCache::<String, String>::new(config);

        cache
            .put("key1".to_string(), "value1".to_string(), 10)
            .await
            .unwrap();

        assert_eq!(
            cache.get(&"key1".to_string()).await,
            Some("value1".to_string())
        );
        assert_eq!(cache.get(&"key2".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_tiered_cache_stats() {
        let config = TieredCacheConfig::default();
        let cache = TieredCache::<String, String>::new(config);

        cache
            .put("key1".to_string(), "value1".to_string(), 10)
            .await
            .unwrap();

        assert!(cache.get(&"key1".to_string()).await.is_some());
        assert!(cache.get(&"key2".to_string()).await.is_none());

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_tiered_cache_warmup() {
        let config = TieredCacheConfig::default();
        let cache = TieredCache::<String, String>::new(config);

        let mut entries = HashMap::new();
        entries.insert("key1".to_string(), ("value1".to_string(), 10));
        entries.insert("key2".to_string(), ("value2".to_string(), 10));
        entries.insert("key3".to_string(), ("value3".to_string(), 10));

        cache.warm_up(entries).await.unwrap();

        assert!(cache.get(&"key1".to_string()).await.is_some());
        assert!(cache.get(&"key2".to_string()).await.is_some());
        assert!(cache.get(&"key3".to_string()).await.is_some());
    }
}
