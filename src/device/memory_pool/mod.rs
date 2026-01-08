// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 内存池模块
//!
//! 提供多种内存池实现，用于优化内存分配和释放性能：
//! - TensorPool: GPU 张量池
//! - BufferPool: CPU 缓冲区池
//! - ModelWeightPool: 模型权重内存池
//! - CudaPool: CUDA 内存池

mod buffer_pool;
mod config;
mod cuda_pool;
mod model_pool;
mod pool_manager;
mod tensor_pool;

pub use buffer_pool::{BufferPool, BufferPoolStats};
pub use config::{
    BufferPoolConfig, CudaPoolConfig, MemoryPoolConfig, ModelWeightPoolConfig, TensorPoolConfig,
};
pub use cuda_pool::CudaMemoryPool;
pub use model_pool::{ModelPoolStats, ModelSlot, ModelWeightPool};
pub use pool_manager::{MemoryPoolManager, MemoryPoolStats};
pub use tensor_pool::{PoolStats, TensorPool};
