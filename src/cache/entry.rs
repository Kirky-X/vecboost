// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 缓存条目与统计的实现块

use super::{CacheConfig, CacheEntry, CacheStats, CacheStrategy};

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

impl CacheStats {
    /// 计算命中率
    pub fn calculate_hit_rate(&mut self) {
        if self.total_requests > 0 {
            self.hit_rate = self.hits as f64 / self.total_requests as f64;
        }
    }
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
            .expect("SystemTime before UNIX EPOCH")
            .as_secs()
    }
}
