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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CacheEntry, CacheStats, CacheStrategy};

    #[test]
    fn test_cache_config_default_values() {
        let config = CacheConfig::default();
        assert_eq!(config.capacity, 10000);
        assert_eq!(config.ttl_secs, 3600);
        assert!(config.enable_stats);
        assert_eq!(config.strategy, CacheStrategy::Lru);
    }

    #[test]
    fn test_cache_entry_new_initializes_fields() {
        let entry = CacheEntry::new("value", 42);
        assert_eq!(entry.value, "value");
        assert_eq!(entry.size, 42);
        assert_eq!(entry.access_count, 1);
        assert!(entry.created_at > 0);
        assert_eq!(entry.created_at, entry.last_accessed);
    }

    #[test]
    fn test_cache_entry_new_with_zero_size() {
        let entry = CacheEntry::new(vec![1, 2, 3], 0);
        assert_eq!(entry.value, vec![1, 2, 3]);
        assert_eq!(entry.size, 0);
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_cache_entry_new_with_complex_value() {
        let value = std::collections::HashMap::from([("k".to_string(), 7)]);
        let entry = CacheEntry::new(value.clone(), 64);
        assert_eq!(entry.value, value);
        assert_eq!(entry.size, 64);
    }

    #[test]
    fn test_cache_entry_is_expired_zero_ttl_never_expires() {
        let entry = CacheEntry::new("value", 10);
        assert!(!entry.is_expired(0));
    }

    #[test]
    fn test_cache_entry_is_expired_future_timestamp_not_expired() {
        let entry = CacheEntry::new("value", 10);
        // 刚创建的条目在合理 TTL 内不应过期
        assert!(!entry.is_expired(3600));
    }

    #[test]
    fn test_cache_entry_is_expired_with_past_last_accessed() {
        let mut entry = CacheEntry::new("value", 10);
        // 将 last_accessed 设置为很久以前,模拟过期条目
        // now - last_accessed > ttl_secs → 过期
        entry.last_accessed = entry.created_at.saturating_sub(7200);
        assert!(entry.is_expired(3600));
    }

    #[test]
    fn test_cache_entry_is_expired_boundary_equal_ttl_not_expired() {
        let mut entry = CacheEntry::new("value", 10);
        // now - last_accessed == ttl_secs 不算过期(严格大于)
        entry.last_accessed = entry.created_at.saturating_sub(3600);
        assert!(!entry.is_expired(3600));
    }

    #[test]
    fn test_cache_entry_record_access_increments_count() {
        let mut entry = CacheEntry::new("value", 10);
        let initial_count = entry.access_count;
        let initial_accessed = entry.last_accessed;
        entry.record_access();
        assert_eq!(entry.access_count, initial_count + 1);
        assert!(entry.last_accessed >= initial_accessed);
    }

    #[test]
    fn test_cache_entry_record_access_multiple_times() {
        let mut entry = CacheEntry::new("value", 10);
        for _ in 0..5 {
            entry.record_access();
        }
        assert_eq!(entry.access_count, 6); // 初始 1 + 5 次
    }

    #[test]
    fn test_cache_entry_clone_preserves_all_fields() {
        let entry = CacheEntry::new(vec![1, 2, 3], 24);
        let cloned = entry.clone();
        assert_eq!(entry.value, cloned.value);
        assert_eq!(entry.created_at, cloned.created_at);
        assert_eq!(entry.last_accessed, cloned.last_accessed);
        assert_eq!(entry.access_count, cloned.access_count);
        assert_eq!(entry.size, cloned.size);
    }

    #[test]
    fn test_cache_entry_size_field_preserved() {
        let entry = CacheEntry::new("hello", 5);
        assert_eq!(entry.size, 5);

        let entry2 = CacheEntry::new("hello", 1000);
        assert_eq!(entry2.size, 1000);
    }

    #[test]
    fn test_cache_stats_calculate_hit_rate_with_zero_requests() {
        let mut stats = CacheStats::default();
        stats.calculate_hit_rate();
        assert_eq!(stats.hit_rate, 0.0);
        assert_eq!(stats.total_requests, 0);
    }

    #[test]
    fn test_cache_stats_calculate_hit_rate_all_hits() {
        let mut stats = CacheStats {
            hits: 100,
            total_requests: 100,
            ..Default::default()
        };
        stats.calculate_hit_rate();
        assert!((stats.hit_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_calculate_hit_rate_all_misses() {
        let mut stats = CacheStats {
            misses: 50,
            total_requests: 50,
            ..Default::default()
        };
        stats.calculate_hit_rate();
        assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_calculate_hit_rate_half() {
        let mut stats = CacheStats {
            hits: 50,
            misses: 50,
            total_requests: 100,
            ..Default::default()
        };
        stats.calculate_hit_rate();
        assert!((stats.hit_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_calculate_hit_rate_partial() {
        let mut stats = CacheStats {
            hits: 30,
            misses: 70,
            total_requests: 100,
            ..Default::default()
        };
        stats.calculate_hit_rate();
        assert!((stats.hit_rate - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_default_values() {
        let stats = CacheStats::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.hit_rate, 0.0);
        assert_eq!(stats.current_size, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_cache_entry_current_timestamp_is_recent() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let entry = CacheEntry::new("v", 1);
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(entry.created_at >= before);
        assert!(entry.created_at <= after);
    }
}
