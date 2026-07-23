// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};

/// CPU 缓冲区池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 文本缓冲区大小列表
    pub text_buffer_sizes: Vec<usize>,
    /// 向量缓冲区大小列表
    pub vector_buffer_sizes: Vec<usize>,
    /// 每种大小的池大小
    pub pool_size_per_size: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            text_buffer_sizes: vec![16, 32, 64, 128, 256],
            vector_buffer_sizes: vec![16, 32, 64, 128, 256],
            pool_size_per_size: 8,
        }
    }
}

/// 模型权重内存池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 最大内存（MB）
    pub max_memory_mb: usize,
    /// 是否缓存模型
    pub cache_models: bool,
}

impl Default for ModelWeightPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: 8192, // 8GB
            cache_models: true,
        }
    }
}

/// CUDA 内存池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaPoolConfig {
    /// 是否启用
    pub enabled: bool,
    /// 最大内存（MB）
    pub max_memory_mb: usize,
}

impl Default for CudaPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: 4096, // 4GB
        }
    }
}

/// 内存池统一配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryPoolConfig {
    /// CPU 缓冲区池配置
    pub buffer_pool: BufferPoolConfig,
    /// 模型权重内存池配置
    pub model_pool: ModelWeightPoolConfig,
    /// CUDA 内存池配置
    pub cuda_pool: CudaPoolConfig,
}
