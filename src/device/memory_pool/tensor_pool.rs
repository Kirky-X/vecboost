// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use candle_core::{Device, Result as CandleResult, Tensor};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};

use super::config::TensorPoolConfig;
use crate::error::AppError;

/// 池统计信息
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// 总分配次数
    pub total_allocations: u64,
    /// 总释放次数
    pub total_releases: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// 当前池大小
    pub current_pool_size: usize,
    /// 总内存使用量（字节）
    pub total_memory_bytes: u64,
}

/// GPU 张量池
///
/// 用于预分配和复用 GPU 张量，减少频繁的内存分配和释放开销
pub struct TensorPool {
    /// 设备
    device: Device,
    /// 最大批量大小
    max_batch_size: usize,
    /// 最大序列长度
    max_sequence_length: usize,
    /// 池: (batch_size, seq_len) -> 张量队列
    pools: HashMap<(usize, usize), VecDeque<Tensor>>,
    /// 配置
    config: TensorPoolConfig,
    /// 统计信息
    stats: PoolStats,
    /// 总内存使用量（原子）
    total_memory_bytes: AtomicU64,
}

impl TensorPool {
    /// 创建新的张量池
    pub fn new(device: Device, config: TensorPoolConfig) -> Self {
        info!(
            "Creating TensorPool on device {:?} with max_batch_size={}, max_seq_len={}",
            device, config.max_batch_size, config.max_sequence_length
        );

        Self {
            device,
            max_batch_size: config.max_batch_size,
            max_sequence_length: config.max_sequence_length,
            pools: HashMap::new(),
            config,
            stats: PoolStats::default(),
            total_memory_bytes: AtomicU64::new(0),
        }
    }

    /// 预分配张量
    pub fn preallocate(&mut self) -> Result<(), AppError> {
        info!("Preallocating tensors...");

        // 预分配常用的批量大小
        let batch_sizes = vec![1, 4, 8, 16, 32, 64, 128];
        let seq_lengths = vec![128, 256, 512, 1024, 2048, 4096, 8192];

        for &batch_size in &batch_sizes {
            if batch_size > self.max_batch_size {
                continue;
            }

            for &seq_len in &seq_lengths {
                if seq_len > self.max_sequence_length {
                    continue;
                }

                let key = (batch_size, seq_len);
                self.pools.entry(key).or_default();

                let pool = self.pools.get_mut(&key).unwrap();
                let pool_size = self.config.pool_size_per_shape;

                for _ in 0..pool_size {
                    // 先创建张量
                    let tensor_result: Result<Tensor, AppError> = {
                        let size = batch_size * seq_len;
                        let data = vec![0i64; size];

                        let tensor = Tensor::new(data, &self.device)
                            .and_then(|t| t.reshape(&[batch_size, seq_len]))
                            .map_err(|e| {
                                AppError::InferenceError(format!("Failed to create tensor: {}", e))
                            })?;

                        // 更新内存统计
                        let tensor_size = (batch_size * seq_len * 8) as u64; // i64 = 8 bytes
                        self.total_memory_bytes
                            .fetch_add(tensor_size, Ordering::Relaxed);

                        Ok(tensor)
                    };

                    match tensor_result {
                        Ok(tensor) => {
                            pool.push_back(tensor);
                            self.stats.total_allocations += 1;
                        }
                        Err(e) => {
                            warn!(
                                "Failed to preallocate tensor for shape ({}, {}): {}",
                                batch_size, seq_len, e
                            );
                            // 继续尝试其他形状
                        }
                    }
                }
            }
        }

        info!(
            "Preallocation complete. Total tensors: {}",
            self.pools.values().map(|q| q.len()).sum::<usize>()
        );

        Ok(())
    }

    /// 获取张量
    pub fn acquire(&mut self, batch_size: usize, seq_len: usize) -> Result<Tensor, AppError> {
        // 验证参数
        if batch_size > self.max_batch_size {
            return Err(AppError::InvalidInput(format!(
                "Batch size {} exceeds maximum {}",
                batch_size, self.max_batch_size
            )));
        }

        if seq_len > self.max_sequence_length {
            return Err(AppError::InvalidInput(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_sequence_length
            )));
        }

        let key = (batch_size, seq_len);

        // 尝试从池中获取
        if let Some(pool) = self.pools.get_mut(&key)
            && let Some(tensor) = pool.pop_front()
        {
            self.stats.cache_hits += 1;
            self.stats.total_allocations += 1;
            debug!(
                "Acquired tensor from pool for shape ({}, {})",
                batch_size, seq_len
            );
            return Ok(tensor);
        }

        // 池中没有，创建新的
        self.stats.cache_misses += 1;
        self.stats.total_allocations += 1;
        debug!(
            "Creating new tensor for shape ({}, {})",
            batch_size, seq_len
        );

        self.create_tensor(batch_size, seq_len)
    }

    /// 释放张量回池
    pub fn release(&mut self, tensor: Tensor, batch_size: usize, seq_len: usize) {
        let key = (batch_size, seq_len);

        self.pools.entry(key).or_default();

        let pool = self.pools.get_mut(&key).unwrap();

        // 如果池未满，则放回池中
        if pool.len() < self.config.pool_size_per_shape {
            pool.push_back(tensor);
            self.stats.total_releases += 1;
            debug!(
                "Released tensor to pool for shape ({}, {})",
                batch_size, seq_len
            );
        } else {
            // 池已满，直接丢弃（Tensor 会被 Drop）
            self.stats.total_releases += 1;
            // 更新内存统计
            let tensor_size = (batch_size * seq_len * 8) as u64; // i64 = 8 bytes
            self.total_memory_bytes
                .fetch_sub(tensor_size, Ordering::Relaxed);
            debug!(
                "Pool full for shape ({}, {}), tensor dropped, memory reduced by {} bytes",
                batch_size, seq_len, tensor_size
            );
        }
    }

    /// 创建新张量
    fn create_tensor(&self, batch_size: usize, seq_len: usize) -> Result<Tensor, AppError> {
        let size = batch_size * seq_len;
        let data = vec![0i64; size];

        let tensor = Tensor::new(data, &self.device)
            .map_err(|e| AppError::InferenceError(format!("Failed to create tensor: {}", e)))?
            .reshape(&[batch_size, seq_len])
            .map_err(|e| AppError::InferenceError(format!("Failed to reshape tensor: {}", e)))?;

        // 更新内存统计
        let tensor_size = (batch_size * seq_len * 8) as u64; // i64 = 8 bytes
        self.total_memory_bytes
            .fetch_add(tensor_size, Ordering::Relaxed);

        Ok(tensor)
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> PoolStats {
        let current_pool_size = self.pools.values().map(|q| q.len()).sum();
        let total_memory_bytes = self.total_memory_bytes.load(Ordering::Relaxed);

        PoolStats {
            total_allocations: self.stats.total_allocations,
            total_releases: self.stats.total_releases,
            cache_hits: self.stats.cache_hits,
            cache_misses: self.stats.cache_misses,
            current_pool_size,
            total_memory_bytes,
        }
    }

    /// 清空池
    pub fn clear(&mut self) {
        info!("Clearing tensor pool...");
        self.pools.clear();
        self.total_memory_bytes.store(0, Ordering::Relaxed);
        info!("Tensor pool cleared");
    }

    /// 获取设备
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool_creation() {
        let device = Device::Cpu;
        let config = TensorPoolConfig::default();
        let pool = TensorPool::new(device, config);

        assert_eq!(pool.get_stats().total_allocations, 0);
        assert_eq!(pool.get_stats().current_pool_size, 0);
    }

    #[test]
    fn test_acquire_release() {
        let device = Device::Cpu;
        let config = TensorPoolConfig {
            max_batch_size: 32,
            max_sequence_length: 512,
            pool_size_per_shape: 2,
            ..Default::default()
        };

        let mut pool = TensorPool::new(device, config);

        // 获取张量
        let tensor = pool.acquire(16, 256).unwrap();
        assert_eq!(pool.get_stats().cache_misses, 1);

        // 释放张量
        pool.release(tensor, 16, 256);
        assert_eq!(pool.get_stats().total_releases, 1);

        // 再次获取，应该从池中获取
        let _tensor2 = pool.acquire(16, 256).unwrap();
        assert_eq!(pool.get_stats().cache_hits, 1);
    }

    #[test]
    fn test_preallocate() {
        let device = Device::Cpu;
        let config = TensorPoolConfig {
            max_batch_size: 32,
            max_sequence_length: 512,
            pool_size_per_shape: 2,
            preallocate_on_startup: true,
            ..Default::default()
        };

        let mut pool = TensorPool::new(device, config);

        pool.preallocate().unwrap();

        let stats = pool.get_stats();
        assert!(stats.current_pool_size > 0);
        assert!(stats.total_memory_bytes > 0);
    }

    #[test]
    fn test_batch_size_limit() {
        let device = Device::Cpu;
        let config = TensorPoolConfig {
            max_batch_size: 16,
            max_sequence_length: 512,
            ..Default::default()
        };

        let mut pool = TensorPool::new(device, config);

        // 尝试获取超过最大批次的张量
        let result = pool.acquire(32, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_seq_len_limit() {
        let device = Device::Cpu;
        let config = TensorPoolConfig {
            max_batch_size: 32,
            max_sequence_length: 256,
            ..Default::default()
        };

        let mut pool = TensorPool::new(device, config);

        // 尝试获取超过最大序列长度的张量
        let result = pool.acquire(16, 512);
        assert!(result.is_err());
    }
}
