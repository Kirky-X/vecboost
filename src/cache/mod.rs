// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod arc_cache;
pub mod kv_cache;
pub mod lfu_cache;
pub mod lru_cache;
pub mod tiered_cache;

pub use kv_cache::KvCache;

// 导出 LRU/LFU/ARC/Tiered 缓存
pub use arc_cache::ArcCache;
pub use lfu_cache::LfuCache;
pub use lru_cache::LruCache;
// TieredCache 暂未使用，保留供将来扩展
// pub use tiered_cache::{TieredCache, TieredCacheConfig, CacheLevel};

// === 通用缓存类型和接口 ===

use std::collections::HashMap;
use std::hash::Hash;

/// 缓存策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// 最近最少使用
    Lru,
    /// 最不经常使用
    Lfu,
    /// 自适应替换缓存
    Arc,
    /// 两队列缓存 (FIFO + LRU)
    TwoQueue,
}

impl Default for CacheStrategy {
    fn default() -> Self {
        Self::Lru
    }
}

/// 缓存配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 缓存容量
    pub capacity: usize,
    /// TTL（秒），0 表示永不过期
    pub ttl_secs: u64,
    /// 是否启用统计
    pub enable_stats: bool,
    /// 缓存策略
    pub strategy: CacheStrategy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            ttl_secs: 3600, // 1小时
            enable_stats: true,
            strategy: CacheStrategy::Lru,
        }
    }
}

/// 缓存统计
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
    /// 总请求数
    pub total_requests: u64,
    /// 命中率
    pub hit_rate: f64,
    /// 当前条目数
    pub current_size: usize,
    /// 驱逐次数
    pub evictions: u64,
}

impl CacheStats {
    /// 计算命中率
    pub fn calculate_hit_rate(&mut self) {
        if self.total_requests > 0 {
            self.hit_rate = self.hits as f64 / self.total_requests as f64;
        }
    }
}

/// 缓存条目
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// 值
    pub value: V,
    /// 创建时间戳
    pub created_at: u64,
    /// 最后访问时间戳
    pub last_accessed: u64,
    /// 访问次数
    pub access_count: u64,
    /// 条目大小（字节）
    pub size: usize,
}

impl<V> CacheEntry<V> {
    /// 创建新的缓存条目
    pub fn new(value: V, size: usize) -> Self {
        let now = Self::current_timestamp();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            size,
        }
    }

    /// 检查是否过期
    pub fn is_expired(&self, ttl_secs: u64) -> bool {
        if ttl_secs == 0 {
            return false;
        }
        let now = Self::current_timestamp();
        now - self.last_accessed > ttl_secs
    }

    /// 更新访问信息
    pub fn record_access(&mut self) {
        self.last_accessed = Self::current_timestamp();
        self.access_count += 1;
    }

    /// 获取当前时间戳（秒）
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

/// 通用缓存接口
#[async_trait::async_trait]
pub trait Cache<K, V>: Send + Sync
where
    K: Hash + Eq + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 获取缓存值
    async fn get(&self, key: &K) -> Option<V>;

    /// 插入缓存值
    async fn put(&self, key: K, value: V, size: usize) -> Result<(), String>;

    /// 移除缓存条目
    async fn remove(&self, key: &K) -> bool;

    /// 清空缓存
    async fn clear(&self);

    /// 获取缓存大小
    async fn len(&self) -> usize;

    /// 检查是否为空
    async fn is_empty(&self) -> bool;

    /// 获取缓存统计
    async fn stats(&self) -> CacheStats;

    /// 预热缓存（批量插入）
    async fn warm_up(&self, entries: HashMap<K, (V, usize)>) -> Result<(), String>;
}

/// 辅助 trait 用于 get_or_insert
#[async_trait::async_trait]
pub trait CacheGetOrInsert<K, V>: Send + Sync
where
    K: Hash + Eq + Send + Sync + std::fmt::Debug + Clone + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get_or_insert(
        &self,
        key: K,
        value: V,
        size: usize,
    ) -> Result<V, Box<dyn std::error::Error + Send + Sync>>;
}

/// 所有缓存实现默认实现 get_or_insert
#[async_trait::async_trait]
impl<K, V, C> CacheGetOrInsert<K, V> for C
where
    K: Hash + Eq + Send + Sync + std::fmt::Debug + Clone + 'static,
    V: Clone + Send + Sync + 'static,
    C: Cache<K, V>,
{
    async fn get_or_insert(
        &self,
        key: K,
        value: V,
        size: usize,
    ) -> Result<V, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(v) = self.get(&key).await {
            return Ok(v);
        }
        self.put(key.clone(), value, size).await?;
        Ok(self.get(&key).await.ok_or("Value not found after insert")?)
    }
}
