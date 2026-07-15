// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

#![allow(clippy::all)]

use log::{debug, error, info, warn};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::{BufferPoolConfig, CudaPoolConfig, ModelWeightPoolConfig, TensorPoolConfig};
use super::{
    BufferPool, BufferPoolStats, CudaMemoryPool, ModelPoolStats, ModelWeightPool, TensorPool,
};
use crate::error::VecboostError;

/// 内存池管理器
///
/// 统一管理所有内存池
pub struct MemoryPoolManager {
    /// Tensor 池
    tensor_pool: Option<Arc<RwLock<TensorPool>>>,
    /// 缓冲区池
    buffer_pool: Arc<RwLock<BufferPool>>,
    /// 模型权重池
    model_pool: Arc<RwLock<ModelWeightPool>>,
    /// CUDA 池
    cuda_pool: Option<Arc<RwLock<CudaMemoryPool>>>,
    /// 配置
    config: MemoryPoolConfig,
}

/// 内存池配置
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Tensor 池配置
    pub tensor_pool: TensorPoolConfig,
    /// 缓冲区池配置
    pub buffer_pool: BufferPoolConfig,
    /// 模型权重池配置
    pub model_pool: ModelWeightPoolConfig,
    /// CUDA 池配置
    pub cuda_pool: CudaPoolConfig,
}

impl MemoryPoolManager {
    /// 创建新的内存池管理器
    pub fn new(config: MemoryPoolConfig) -> Self {
        info!("Creating MemoryPoolManager");

        Self {
            tensor_pool: None, // 需要设备信息，稍后初始化
            buffer_pool: Arc::new(RwLock::new(BufferPool::new(config.buffer_pool.clone()))),
            model_pool: Arc::new(RwLock::new(ModelWeightPool::new(
                "default".to_string(),
                config.model_pool.clone(),
            ))),
            cuda_pool: None, // 需要设备信息，稍后初始化
            config,
        }
    }

    /// 初始化 Tensor 池
    pub async fn initialize_tensor_pool(
        &mut self,
        device: candle_core::Device,
    ) -> Result<(), VecboostError> {
        if !self.config.tensor_pool.enabled {
            info!("Tensor pool disabled, skipping initialization");
            return Ok(());
        }

        info!("Initializing Tensor pool...");

        let mut pool = TensorPool::new(device, self.config.tensor_pool.clone());

        if self.config.tensor_pool.preallocate_on_startup {
            pool.preallocate()?;
        }

        self.tensor_pool = Some(Arc::new(RwLock::new(pool)));

        info!("Tensor pool initialized successfully");
        Ok(())
    }

    /// 初始化 CUDA 池
    pub async fn initialize_cuda_pool(&mut self, device_id: i32) -> Result<(), VecboostError> {
        if !self.config.cuda_pool.enabled {
            info!("CUDA pool disabled, skipping initialization");
            return Ok(());
        }

        info!("Initializing CUDA pool for device {}...", device_id);

        let pool = CudaMemoryPool::new(device_id, self.config.cuda_pool.clone()).map_err(|e| {
            VecboostError::ConfigError(format!("Failed to initialize CUDA pool: {}", e))
        })?;

        self.cuda_pool = Some(Arc::new(RwLock::new(pool)));

        info!("CUDA pool initialized successfully");
        Ok(())
    }

    /// 初始化所有池

    pub async fn initialize_all(
        &mut self,
        device: Option<candle_core::Device>,
        cuda_device_id: Option<i32>,
    ) -> Result<(), VecboostError> {
        info!("Initializing all memory pools...");

        // 先验证所有必需的参数

        if self.config.tensor_pool.enabled && device.is_none() {
            return Err(VecboostError::ConfigError(
                "Tensor pool requires device but none provided".to_string(),
            ));
        }

        if self.config.cuda_pool.enabled && cuda_device_id.is_none() {
            return Err(VecboostError::ConfigError(
                "CUDA pool requires device_id but none provided".to_string(),
            ));
        }

        // 记录初始化状态

        let mut initialized_pools = Vec::new();

        // 初始化缓冲区池

        if self.config.buffer_pool.enabled {
            let mut buffer_pool = self.buffer_pool.write().await;

            buffer_pool.preallocate();

            initialized_pools.push("buffer_pool");

            info!("Buffer pool initialized");
        }

        // 初始化模型权重池

        if self.config.model_pool.enabled {
            initialized_pools.push("model_pool");

            info!("Model weight pool initialized");
        }

        // 初始化 Tensor 池

        if let Some(device) = device {
            match self.initialize_tensor_pool(device).await {
                Ok(_) => {
                    initialized_pools.push("tensor_pool");

                    info!("Tensor pool initialized");
                }

                Err(e) => {
                    // 回滚已初始化的池

                    error!("Failed to initialize tensor pool: {}, rolling back", e);

                    self.clear_all().await;

                    return Err(VecboostError::ConfigError(format!(
                        "Failed to initialize tensor pool: {}. Rollback completed.",
                        e
                    )));
                }
            }
        }

        // 初始化 CUDA 池

        if let Some(device_id) = cuda_device_id {
            match self.initialize_cuda_pool(device_id).await {
                Ok(_) => {
                    initialized_pools.push("cuda_pool");

                    info!("CUDA pool initialized");
                }

                Err(e) => {
                    // 回滚已初始化的池

                    error!("Failed to initialize CUDA pool: {}, rolling back", e);

                    self.clear_all().await;

                    return Err(VecboostError::ConfigError(format!(
                        "Failed to initialize CUDA pool: {}. Rollback completed.",
                        e
                    )));
                }
            }
        }

        info!(
            "All memory pools initialized successfully: {:?}",
            initialized_pools
        );

        Ok(())
    }

    /// 获取 Tensor 池
    pub async fn get_tensor_pool(&self) -> Option<Arc<RwLock<TensorPool>>> {
        self.tensor_pool.clone()
    }

    /// 获取缓冲区池
    pub async fn get_buffer_pool(&self) -> Arc<RwLock<BufferPool>> {
        Arc::clone(&self.buffer_pool)
    }

    /// 获取模型权重池
    pub async fn get_model_pool(&self) -> Arc<RwLock<ModelWeightPool>> {
        Arc::clone(&self.model_pool)
    }

    /// 获取 CUDA 池
    pub async fn get_cuda_pool(&self) -> Option<Arc<RwLock<CudaMemoryPool>>> {
        self.cuda_pool.clone()
    }

    /// 获取内存统计信息
    pub async fn get_memory_stats(&self) -> MemoryPoolStats {
        let buffer_pool = self.buffer_pool.read().await;
        let model_pool = self.model_pool.read().await;

        let mut stats = MemoryPoolStats {
            tensor_pool_enabled: self.config.tensor_pool.enabled,
            tensor_pool_stats: None,
            buffer_pool_enabled: self.config.buffer_pool.enabled,
            buffer_pool_stats: Some(buffer_pool.get_stats()),
            model_pool_enabled: self.config.model_pool.enabled,
            model_pool_stats: Some(model_pool.get_stats()),
            cuda_pool_enabled: self.config.cuda_pool.enabled,
            cuda_pool_stats: None,
        };

        if let Some(ref tensor_pool) = self.tensor_pool {
            let pool = tensor_pool.read().await;
            stats.tensor_pool_stats = Some(pool.get_stats());
        }

        if let Some(ref cuda_pool) = self.cuda_pool {
            let pool = cuda_pool.read().await;
            let (used, total) = pool.get_memory_usage();
            stats.cuda_pool_stats = Some(CudaPoolStats {
                used_memory_mb: used / 1024 / 1024,
                total_memory_mb: total / 1024 / 1024,
                memory_usage_percent: pool.get_memory_usage_percent(),
            });
        }

        stats
    }

    /// 清空所有池
    pub async fn clear_all(&self) {
        info!("Clearing all memory pools...");

        if let Some(ref tensor_pool) = self.tensor_pool {
            let mut pool = tensor_pool.write().await;
            pool.clear();
        }

        {
            let mut buffer_pool = self.buffer_pool.write().await;
            buffer_pool.clear();
        }

        {
            let mut model_pool = self.model_pool.write().await;
            model_pool.clear();
        }

        if let Some(ref cuda_pool) = self.cuda_pool {
            let mut pool = cuda_pool.write().await;
            pool.clear();
        }

        info!("All memory pools cleared");
    }
}

/// 内存池统计信息
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Tensor 池是否启用
    pub tensor_pool_enabled: bool,
    /// Tensor 池统计
    pub tensor_pool_stats: Option<super::PoolStats>,
    /// 缓冲区池是否启用
    pub buffer_pool_enabled: bool,
    /// 缓冲区池统计
    pub buffer_pool_stats: Option<super::BufferPoolStats>,
    /// 模型权重池是否启用
    pub model_pool_enabled: bool,
    /// 模型权重池统计
    pub model_pool_stats: Option<super::ModelPoolStats>,
    /// CUDA 池是否启用
    pub cuda_pool_enabled: bool,
    /// CUDA 池统计
    pub cuda_pool_stats: Option<CudaPoolStats>,
}

/// CUDA 池统计信息
#[derive(Debug, Clone)]
pub struct CudaPoolStats {
    /// 已使用内存（MB）
    pub used_memory_mb: u64,
    /// 总内存（MB）
    pub total_memory_mb: u64,
    /// 内存使用率（百分比）
    pub memory_usage_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_manager_creation() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };

        let manager = MemoryPoolManager::new(config);
        assert!(manager.tensor_pool.is_none());
        assert!(manager.cuda_pool.is_none());
    }

    #[tokio::test]
    async fn test_get_buffer_pool() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };

        let manager = MemoryPoolManager::new(config);
        let buffer_pool = manager.get_buffer_pool().await;

        let stats = buffer_pool.read().await.get_stats();
        assert_eq!(stats.text_allocations, 0);
    }

    #[tokio::test]
    async fn test_get_model_pool() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };

        let manager = MemoryPoolManager::new(config);
        let model_pool = manager.get_model_pool().await;

        let stats = model_pool.read().await.get_stats();
        assert_eq!(stats.loaded_models, 0);
    }

    #[tokio::test]
    async fn test_initialize_all() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: false, // 禁用 Tensor 池以避免需要设备
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: false, // 禁用 CUDA 池以避免需要 device_id
                ..Default::default()
            },
        };

        let mut manager = MemoryPoolManager::new(config);

        let result = manager.initialize_all(None, None).await;
        assert!(result.is_ok(), "initialize_all failed: {:?}", result);

        let stats = manager.get_memory_stats().await;
        assert!(stats.buffer_pool_stats.is_some());
        assert!(stats.model_pool_stats.is_some());
    }

    #[tokio::test]
    async fn test_clear_all() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };

        let manager = MemoryPoolManager::new(config);
        manager.clear_all().await;
    }

    #[tokio::test]
    async fn test_initialize_tensor_pool_disabled() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: false,
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager
            .initialize_tensor_pool(candle_core::Device::Cpu)
            .await;
        assert!(result.is_ok());
        assert!(manager.tensor_pool.is_none());
    }

    #[tokio::test]
    async fn test_initialize_tensor_pool_enabled_with_preallocate() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                max_batch_size: 8,
                max_sequence_length: 256,
                pool_size_per_shape: 1,
                preallocate_on_startup: true,
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager
            .initialize_tensor_pool(candle_core::Device::Cpu)
            .await;
        assert!(result.is_ok());
        assert!(manager.tensor_pool.is_some());

        let pool = manager.get_tensor_pool().await;
        assert!(pool.is_some());
        let stats = pool.unwrap().read().await.get_stats();
        assert!(stats.current_pool_size > 0);
    }

    #[tokio::test]
    async fn test_initialize_tensor_pool_enabled_without_preallocate() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                preallocate_on_startup: false,
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager
            .initialize_tensor_pool(candle_core::Device::Cpu)
            .await;
        assert!(result.is_ok());
        assert!(manager.tensor_pool.is_some());

        let pool = manager.get_tensor_pool().await.unwrap();
        let stats = pool.read().await.get_stats();
        assert_eq!(stats.current_pool_size, 0);
        assert_eq!(stats.total_allocations, 0);
    }

    #[tokio::test]
    async fn test_initialize_cuda_pool_disabled() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager.initialize_cuda_pool(0).await;
        assert!(result.is_ok());
        assert!(manager.cuda_pool.is_none());
    }

    #[tokio::test]
    async fn test_initialize_cuda_pool_enabled() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: true,
                max_memory_mb: 1024,
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager.initialize_cuda_pool(0).await;
        assert!(result.is_ok());
        assert!(manager.cuda_pool.is_some());
    }

    #[tokio::test]
    async fn test_get_tensor_pool_returns_none_before_init() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        let manager = MemoryPoolManager::new(config);

        assert!(manager.get_tensor_pool().await.is_none());
    }

    #[tokio::test]
    async fn test_get_cuda_pool_returns_none_before_init() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig::default(),
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig::default(),
        };
        let manager = MemoryPoolManager::new(config);

        assert!(manager.get_cuda_pool().await.is_none());
    }

    #[tokio::test]
    async fn test_initialize_all_error_tensor_enabled_no_device() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager.initialize_all(None, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::ConfigError(msg) => {
                assert!(msg.contains("Tensor pool requires device"));
            }
            other => panic!("Expected ConfigError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_initialize_all_error_cuda_enabled_no_device_id() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: false,
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig::default(),
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: true,
                ..Default::default()
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager.initialize_all(None, None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            VecboostError::ConfigError(msg) => {
                assert!(msg.contains("CUDA pool requires device_id"));
            }
            other => panic!("Expected ConfigError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_initialize_all_all_enabled() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                max_batch_size: 8,
                max_sequence_length: 256,
                pool_size_per_shape: 1,
                preallocate_on_startup: true,
            },
            buffer_pool: BufferPoolConfig {
                enabled: true,
                text_buffer_sizes: vec![16],
                vector_buffer_sizes: vec![16],
                pool_size_per_size: 2,
            },
            model_pool: ModelWeightPoolConfig {
                enabled: true,
                ..Default::default()
            },
            cuda_pool: CudaPoolConfig {
                enabled: true,
                max_memory_mb: 1024,
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager
            .initialize_all(Some(candle_core::Device::Cpu), Some(0))
            .await;
        assert!(result.is_ok(), "initialize_all failed: {:?}", result);

        assert!(manager.tensor_pool.is_some());
        assert!(manager.cuda_pool.is_some());
    }

    #[tokio::test]
    async fn test_get_memory_stats_with_all_pools() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                max_batch_size: 8,
                max_sequence_length: 256,
                pool_size_per_shape: 1,
                preallocate_on_startup: true,
            },
            buffer_pool: BufferPoolConfig {
                enabled: true,
                text_buffer_sizes: vec![16],
                vector_buffer_sizes: vec![16],
                pool_size_per_size: 2,
            },
            model_pool: ModelWeightPoolConfig {
                enabled: true,
                ..Default::default()
            },
            cuda_pool: CudaPoolConfig {
                enabled: true,
                max_memory_mb: 1024,
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        manager
            .initialize_all(Some(candle_core::Device::Cpu), Some(0))
            .await
            .unwrap();

        let stats = manager.get_memory_stats().await;
        assert!(stats.tensor_pool_enabled);
        assert!(stats.buffer_pool_enabled);
        assert!(stats.model_pool_enabled);
        assert!(stats.cuda_pool_enabled);
        assert!(stats.tensor_pool_stats.is_some());
        assert!(stats.buffer_pool_stats.is_some());
        assert!(stats.model_pool_stats.is_some());
        assert!(stats.cuda_pool_stats.is_some());

        let tensor_stats = stats.tensor_pool_stats.unwrap();
        assert!(tensor_stats.current_pool_size > 0);
        assert!(tensor_stats.total_memory_bytes > 0);

        let buffer_stats = stats.buffer_pool_stats.unwrap();
        assert!(buffer_stats.current_text_pool_size > 0);

        let cuda_stats = stats.cuda_pool_stats.unwrap();
        assert_eq!(cuda_stats.used_memory_mb, 0);
        // cuda feature 下 CudaMemoryPool 使用 config.max_memory_mb(1024);
        // 非 cuda feature 下走 no-op 分支,get_memory_usage() 返回 (0, 0)。
        #[cfg(feature = "cuda")]
        assert_eq!(cuda_stats.total_memory_mb, 1024);
        #[cfg(not(feature = "cuda"))]
        assert_eq!(cuda_stats.total_memory_mb, 0);
        assert!((cuda_stats.memory_usage_percent - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_get_memory_stats_disabled_pools() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: false,
                ..Default::default()
            },
            buffer_pool: BufferPoolConfig {
                enabled: false,
                ..Default::default()
            },
            model_pool: ModelWeightPoolConfig {
                enabled: false,
                ..Default::default()
            },
            cuda_pool: CudaPoolConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let manager = MemoryPoolManager::new(config);

        let stats = manager.get_memory_stats().await;
        assert!(!stats.tensor_pool_enabled);
        assert!(!stats.buffer_pool_enabled);
        assert!(!stats.model_pool_enabled);
        assert!(!stats.cuda_pool_enabled);
        assert!(stats.tensor_pool_stats.is_none());
        assert!(stats.cuda_pool_stats.is_none());
        assert!(stats.buffer_pool_stats.is_some());
        assert!(stats.model_pool_stats.is_some());
    }

    #[tokio::test]
    async fn test_clear_all_after_initialization() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                max_batch_size: 8,
                max_sequence_length: 256,
                pool_size_per_shape: 1,
                preallocate_on_startup: true,
            },
            buffer_pool: BufferPoolConfig {
                enabled: true,
                text_buffer_sizes: vec![16],
                vector_buffer_sizes: vec![16],
                pool_size_per_size: 2,
            },
            model_pool: ModelWeightPoolConfig::default(),
            cuda_pool: CudaPoolConfig {
                enabled: true,
                ..Default::default()
            },
        };
        let mut manager = MemoryPoolManager::new(config);
        manager
            .initialize_all(Some(candle_core::Device::Cpu), Some(0))
            .await
            .unwrap();

        manager.clear_all().await;

        let stats = manager.get_memory_stats().await;
        let tensor_stats = stats.tensor_pool_stats.as_ref().unwrap();
        assert_eq!(tensor_stats.current_pool_size, 0);
        assert_eq!(tensor_stats.total_memory_bytes, 0);
        let buffer_stats = stats.buffer_pool_stats.as_ref().unwrap();
        assert_eq!(buffer_stats.current_text_pool_size, 0);
        assert_eq!(buffer_stats.current_vector_pool_size, 0);
    }

    #[tokio::test]
    async fn test_initialize_all_with_tensor_only() {
        let config = MemoryPoolConfig {
            tensor_pool: TensorPoolConfig {
                enabled: true,
                max_batch_size: 8,
                max_sequence_length: 256,
                pool_size_per_shape: 1,
                preallocate_on_startup: false,
            },
            buffer_pool: BufferPoolConfig {
                enabled: false,
                ..Default::default()
            },
            model_pool: ModelWeightPoolConfig {
                enabled: false,
                ..Default::default()
            },
            cuda_pool: CudaPoolConfig {
                enabled: false,
                ..Default::default()
            },
        };
        let mut manager = MemoryPoolManager::new(config);

        let result = manager
            .initialize_all(Some(candle_core::Device::Cpu), None)
            .await;
        assert!(result.is_ok());
        assert!(manager.tensor_pool.is_some());
        assert!(manager.cuda_pool.is_none());

        let stats = manager.get_memory_stats().await;
        assert!(!stats.buffer_pool_enabled);
        let buffer_stats = stats.buffer_pool_stats.as_ref().unwrap();
        assert_eq!(buffer_stats.current_text_pool_size, 0);
    }
}
