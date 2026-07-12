// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

pub(crate) mod arc_cache;
pub(crate) mod bloom_filter;
pub(crate) mod entry;
pub(crate) mod kv_cache;
pub(crate) mod lfu_cache;
pub(crate) mod lru_cache;
pub(crate) mod tiered_cache;
pub(crate) mod trait_impl;

// 内部使用，不对外暴露
pub(crate) use kv_cache::KvCache;

// 导出 LRU/LFU/ARC/Tiered 缓存（仅内部使用）
pub(crate) use arc_cache::ArcCache;
// Bloom Filter 暂未使用，保留供将来扩展
// pub(crate) use bloom_filter::{BloomFilter, BloomFilterConfig};
pub(crate) use lfu_cache::LfuCache;
pub(crate) use lru_cache::LruCache;
// TieredCache 暂未使用，保留供将来扩展
// pub(crate) use tiered_cache::{TieredCache, TieredCacheConfig, CacheLevel};

// === 通用缓存类型和接口 ===

use std::collections::HashMap;
use std::hash::Hash;

/// 缓存策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheStrategy {
    /// 最近最少使用
    #[default]
    Lru,
    /// 最不经常使用
    Lfu,
    /// 自适应替换缓存
    Arc,
    /// 两队列缓存 (FIFO + LRU)
    TwoQueue,
}

/// 缓存配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 缓存容量
    pub(crate) capacity: usize,
    /// TTL（秒），0 表示永不过期
    pub(crate) ttl_secs: u64,
    /// 是否启用统计
    pub(crate) enable_stats: bool,
    /// 缓存策略
    pub(crate) strategy: CacheStrategy,
}

/// 缓存统计
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 命中次数
    pub(crate) hits: u64,
    /// 未命中次数
    pub(crate) misses: u64,
    /// 总请求数
    pub(crate) total_requests: u64,
    /// 命中率
    pub(crate) hit_rate: f64,
    /// 当前条目数
    pub(crate) current_size: usize,
    /// 驱逐次数
    pub(crate) evictions: u64,
}

/// 缓存条目
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// 值
    pub(crate) value: V,
    /// 创建时间戳
    pub(crate) created_at: u64,
    /// 最后访问时间戳
    pub(crate) last_accessed: u64,
    /// 访问次数
    pub(crate) access_count: u64,
    /// 条目大小（字节）
    pub(crate) size: usize,
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
