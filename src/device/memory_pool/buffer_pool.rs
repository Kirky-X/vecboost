// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use log::{debug, info, warn};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::BufferPoolConfig;

/// 最大缓冲区大小
const MAX_BUFFER_SIZE: usize = 10000;

/// CPU 缓冲区池
///
/// 用于预分配和复用 CPU 上的文本和向量缓冲区
pub struct BufferPool {
    /// 文本缓冲区池: size -> buffer queue
    text_buffers: HashMap<usize, VecDeque<Vec<String>>>,
    /// 向量缓冲区池: size -> buffer queue
    vector_buffers: HashMap<usize, VecDeque<Vec<Vec<f32>>>>,
    /// 配置
    config: BufferPoolConfig,
    /// 统计信息
    stats: BufferPoolStats,
}

/// 缓冲区池统计信息
#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    /// 文本缓冲区总分配次数
    pub text_allocations: u64,
    /// 文本缓冲区总释放次数
    pub text_releases: u64,
    /// 文本缓冲区缓存命中次数
    pub text_cache_hits: u64,
    /// 文本缓冲区缓存未命中次数
    pub text_cache_misses: u64,
    /// 向量缓冲区总分配次数
    pub vector_allocations: u64,
    /// 向量缓冲区总释放次数
    pub vector_releases: u64,
    /// 向量缓冲区缓存命中次数
    pub vector_cache_hits: u64,
    /// 向量缓冲区缓存未命中次数
    pub vector_cache_misses: u64,
    /// 当前文本缓冲区池大小
    pub current_text_pool_size: usize,
    /// 当前向量缓冲区池大小
    pub current_vector_pool_size: usize,
}

impl BufferPool {
    /// 创建新的缓冲区池
    pub fn new(config: BufferPoolConfig) -> Self {
        info!(
            "Creating BufferPool with text_sizes={:?}, vector_sizes={:?}",
            config.text_buffer_sizes, config.vector_buffer_sizes
        );

        Self {
            text_buffers: HashMap::new(),
            vector_buffers: HashMap::new(),
            config,
            stats: BufferPoolStats::default(),
        }
    }

    /// 预分配缓冲区
    pub fn preallocate(&mut self) {
        info!("Preallocating buffers...");

        // 预分配文本缓冲区
        for &size in &self.config.text_buffer_sizes {
            self.text_buffers.entry(size).or_default();

            let pool = self.text_buffers.get_mut(&size).unwrap();
            let pool_size = self.config.pool_size_per_size;

            for _ in 0..pool_size {
                let buffer = vec![String::new(); size];
                pool.push_back(buffer);
                self.stats.text_allocations += 1;
            }
        }

        // 预分配向量缓冲区
        for &size in &self.config.vector_buffer_sizes {
            self.vector_buffers.entry(size).or_default();

            let pool = self.vector_buffers.get_mut(&size).unwrap();
            let pool_size = self.config.pool_size_per_size;

            for _ in 0..pool_size {
                // 创建空的向量缓冲区
                let buffer = vec![Vec::new(); size];
                pool.push_back(buffer);
                self.stats.vector_allocations += 1;
            }
        }

        info!(
            "Preallocation complete. Text buffers: {}, Vector buffers: {}",
            self.text_buffers.values().map(|q| q.len()).sum::<usize>(),
            self.vector_buffers.values().map(|q| q.len()).sum::<usize>()
        );
    }

    /// 获取文本缓冲区
    pub fn acquire_text_buffer(&mut self, size: usize) -> Vec<String> {
        // 边界检查
        if size > MAX_BUFFER_SIZE {
            warn!(
                "Requested text buffer size {} exceeds maximum {}, using max size instead",
                size, MAX_BUFFER_SIZE
            );
            return vec![String::new(); MAX_BUFFER_SIZE];
        }

        // 尝试从池中获取
        if let Some(pool) = self.text_buffers.get_mut(&size)
            && let Some(buffer) = pool.pop_front()
        {
            self.stats.text_cache_hits += 1;
            self.stats.text_allocations += 1;
            debug!("Acquired text buffer from pool, size={}", size);
            return buffer;
        }

        // 池中没有，创建新的
        self.stats.text_cache_misses += 1;
        self.stats.text_allocations += 1;
        debug!("Creating new text buffer, size={}", size);
        vec![String::new(); size]
    }

    /// 释放文本缓冲区回池
    pub fn release_text_buffer(&mut self, mut buffer: Vec<String>) {
        let size = buffer.capacity();

        // 清空缓冲区内容
        buffer.clear();

        self.text_buffers.entry(size).or_default();

        let pool = self.text_buffers.get_mut(&size).unwrap();

        // 如果池未满，则放回池中
        if pool.len() < self.config.pool_size_per_size {
            pool.push_back(buffer);
            self.stats.text_releases += 1;
            debug!("Released text buffer to pool, size={}", size);
        } else {
            // 池已满，直接丢弃
            self.stats.text_releases += 1;
            debug!("Text buffer pool full for size={}, buffer dropped", size);
        }
    }

    /// 获取向量缓冲区
    pub fn acquire_vector_buffer(&mut self, size: usize) -> Vec<Vec<f32>> {
        // 边界检查
        if size > MAX_BUFFER_SIZE {
            warn!(
                "Requested vector buffer size {} exceeds maximum {}, using max size instead",
                size, MAX_BUFFER_SIZE
            );
            return vec![Vec::new(); MAX_BUFFER_SIZE];
        }

        // 尝试从池中获取
        if let Some(pool) = self.vector_buffers.get_mut(&size)
            && let Some(buffer) = pool.pop_front()
        {
            self.stats.vector_cache_hits += 1;
            self.stats.vector_allocations += 1;
            debug!("Acquired vector buffer from pool, size={}", size);
            return buffer;
        }

        // 池中没有，创建新的
        self.stats.vector_cache_misses += 1;
        self.stats.vector_allocations += 1;
        debug!("Creating new vector buffer, size={}", size);
        vec![Vec::new(); size]
    }

    /// 释放向量缓冲区回池
    pub fn release_vector_buffer(&mut self, mut buffer: Vec<Vec<f32>>) {
        let size = buffer.capacity();

        // 清空缓冲区内容
        buffer.clear();

        self.vector_buffers.entry(size).or_default();

        let pool = self.vector_buffers.get_mut(&size).unwrap();

        // 如果池未满，则放回池中
        if pool.len() < self.config.pool_size_per_size {
            pool.push_back(buffer);
            self.stats.vector_releases += 1;
            debug!("Released vector buffer to pool, size={}", size);
        } else {
            // 池已满，直接丢弃
            self.stats.vector_releases += 1;
            debug!("Vector buffer pool full for size={}, buffer dropped", size);
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> BufferPoolStats {
        let current_text_pool_size = self.text_buffers.values().map(|q| q.len()).sum();
        let current_vector_pool_size = self.vector_buffers.values().map(|q| q.len()).sum();

        BufferPoolStats {
            text_allocations: self.stats.text_allocations,
            text_releases: self.stats.text_releases,
            text_cache_hits: self.stats.text_cache_hits,
            text_cache_misses: self.stats.text_cache_misses,
            vector_allocations: self.stats.vector_allocations,
            vector_releases: self.stats.vector_releases,
            vector_cache_hits: self.stats.vector_cache_hits,
            vector_cache_misses: self.stats.vector_cache_misses,
            current_text_pool_size,
            current_vector_pool_size,
        }
    }

    /// 清空池
    pub fn clear(&mut self) {
        info!("Clearing buffer pool...");
        self.text_buffers.clear();
        self.vector_buffers.clear();
        info!("Buffer pool cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_creation() {
        let config = BufferPoolConfig::default();
        let pool = BufferPool::new(config);

        assert_eq!(pool.get_stats().text_allocations, 0);
        assert_eq!(pool.get_stats().vector_allocations, 0);
    }

    #[test]
    fn test_acquire_release_text_buffer() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16],
            pool_size_per_size: 2,
            ..Default::default()
        };

        let mut pool = BufferPool::new(config);

        // 获取缓冲区
        let buffer = pool.acquire_text_buffer(16);
        assert_eq!(buffer.len(), 16);
        assert_eq!(pool.get_stats().text_cache_misses, 1);

        // 释放缓冲区
        pool.release_text_buffer(buffer);
        assert_eq!(pool.get_stats().text_releases, 1);

        // 再次获取，应该从池中获取
        let _buffer2 = pool.acquire_text_buffer(16);
        assert_eq!(pool.get_stats().text_cache_hits, 1);
    }

    #[test]
    fn test_acquire_release_vector_buffer() {
        let config = BufferPoolConfig {
            vector_buffer_sizes: vec![16],
            pool_size_per_size: 2,
            ..Default::default()
        };

        let mut pool = BufferPool::new(config);

        // 获取缓冲区
        let buffer = pool.acquire_vector_buffer(16);
        assert_eq!(buffer.len(), 16);
        assert_eq!(pool.get_stats().vector_cache_misses, 1);

        // 释放缓冲区
        pool.release_vector_buffer(buffer);
        assert_eq!(pool.get_stats().vector_releases, 1);

        // 再次获取，应该从池中获取
        let _buffer2 = pool.acquire_vector_buffer(16);
        assert_eq!(pool.get_stats().vector_cache_hits, 1);
    }

    #[test]
    fn test_preallocate() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16, 32],
            vector_buffer_sizes: vec![16, 32],
            pool_size_per_size: 2,
            ..Default::default()
        };

        let mut pool = BufferPool::new(config);

        pool.preallocate();

        let stats = pool.get_stats();
        assert!(stats.current_text_pool_size > 0);
        assert!(stats.current_vector_pool_size > 0);
    }

    #[test]
    fn test_buffer_capacity_preservation() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16],
            pool_size_per_size: 2,
            ..Default::default()
        };

        let mut pool = BufferPool::new(config);

        let mut buffer = pool.acquire_text_buffer(16);
        buffer[0] = "test".to_string();
        let capacity = buffer.capacity();

        pool.release_text_buffer(buffer);

        let buffer2 = pool.acquire_text_buffer(16);
        assert_eq!(buffer2.capacity(), capacity);
    }

    #[test]
    fn test_release_text_buffer_pool_full_drop() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16],
            pool_size_per_size: 2,
            ..Default::default()
        };
        let mut pool = BufferPool::new(config);

        let b1 = pool.acquire_text_buffer(16);
        let b2 = pool.acquire_text_buffer(16);
        let b3 = pool.acquire_text_buffer(16);
        assert_eq!(pool.get_stats().text_cache_misses, 3);

        pool.release_text_buffer(b1);
        pool.release_text_buffer(b2);
        assert_eq!(pool.get_stats().current_text_pool_size, 2);

        pool.release_text_buffer(b3);
        assert_eq!(pool.get_stats().current_text_pool_size, 2);
        assert_eq!(pool.get_stats().text_releases, 3);
    }

    #[test]
    fn test_release_vector_buffer_pool_full_drop() {
        let config = BufferPoolConfig {
            vector_buffer_sizes: vec![16],
            pool_size_per_size: 1,
            ..Default::default()
        };
        let mut pool = BufferPool::new(config);

        let b1 = pool.acquire_vector_buffer(16);
        let b2 = pool.acquire_vector_buffer(16);

        pool.release_vector_buffer(b1);
        assert_eq!(pool.get_stats().current_vector_pool_size, 1);

        pool.release_vector_buffer(b2);
        assert_eq!(pool.get_stats().current_vector_pool_size, 1);
        assert_eq!(pool.get_stats().vector_releases, 2);
    }

    #[test]
    fn test_acquire_text_buffer_exceeds_max_size() {
        let config = BufferPoolConfig::default();
        let mut pool = BufferPool::new(config);

        let buffer = pool.acquire_text_buffer(MAX_BUFFER_SIZE + 1);
        assert_eq!(buffer.len(), MAX_BUFFER_SIZE);
    }

    #[test]
    fn test_acquire_vector_buffer_exceeds_max_size() {
        let config = BufferPoolConfig::default();
        let mut pool = BufferPool::new(config);

        let buffer = pool.acquire_vector_buffer(MAX_BUFFER_SIZE + 1);
        assert_eq!(buffer.len(), MAX_BUFFER_SIZE);
    }

    #[test]
    fn test_clear_empties_pools() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16, 32],
            vector_buffer_sizes: vec![16, 32],
            pool_size_per_size: 2,
            ..Default::default()
        };
        let mut pool = BufferPool::new(config);
        pool.preallocate();
        assert!(pool.get_stats().current_text_pool_size > 0);
        assert!(pool.get_stats().current_vector_pool_size > 0);

        pool.clear();

        let stats = pool.get_stats();
        assert_eq!(stats.current_text_pool_size, 0);
        assert_eq!(stats.current_vector_pool_size, 0);
        assert!(stats.text_allocations > 0);
    }

    #[test]
    fn test_preallocate_default_config_all_sizes() {
        let config = BufferPoolConfig::default();
        let mut pool = BufferPool::new(config);
        pool.preallocate();

        let stats = pool.get_stats();
        assert_eq!(stats.current_text_pool_size, 5 * 8);
        assert_eq!(stats.current_vector_pool_size, 5 * 8);
        assert_eq!(stats.text_allocations, 5 * 8);
        assert_eq!(stats.vector_allocations, 5 * 8);
    }

    #[test]
    fn test_acquire_release_mixed_sizes() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16, 32],
            pool_size_per_size: 2,
            ..Default::default()
        };
        let mut pool = BufferPool::new(config);

        let b16 = pool.acquire_text_buffer(16);
        let b32 = pool.acquire_text_buffer(32);
        assert_eq!(b16.len(), 16);
        assert_eq!(b32.len(), 32);

        pool.release_text_buffer(b16);
        pool.release_text_buffer(b32);

        let _ = pool.acquire_text_buffer(16);
        let _ = pool.acquire_text_buffer(32);
        assert_eq!(pool.get_stats().text_cache_hits, 2);
        assert_eq!(pool.get_stats().text_cache_misses, 2);
    }

    #[test]
    fn test_stats_reflect_operations() {
        let config = BufferPoolConfig {
            text_buffer_sizes: vec![16],
            vector_buffer_sizes: vec![16],
            pool_size_per_size: 2,
            ..Default::default()
        };
        let mut pool = BufferPool::new(config);

        let s = pool.get_stats();
        assert_eq!((s.text_allocations, s.vector_allocations), (0, 0));
        assert_eq!((s.text_cache_hits, s.vector_cache_hits), (0, 0));

        let tb = pool.acquire_text_buffer(16);
        let vb = pool.acquire_vector_buffer(16);
        let s = pool.get_stats();
        assert_eq!(s.text_cache_misses, 1);
        assert_eq!(s.vector_cache_misses, 1);

        pool.release_text_buffer(tb);
        pool.release_vector_buffer(vb);

        let _ = pool.acquire_text_buffer(16);
        let _ = pool.acquire_vector_buffer(16);
        let s = pool.get_stats();
        assert_eq!(s.text_cache_hits, 1);
        assert_eq!(s.vector_cache_hits, 1);
        assert_eq!(s.text_releases, 1);
        assert_eq!(s.vector_releases, 1);
    }
}
