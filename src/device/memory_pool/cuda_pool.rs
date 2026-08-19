// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use log::{debug, info, warn};
use std::sync::atomic::{AtomicU64, Ordering};

use super::config::CudaPoolConfig;

/// CUDA 内存池
///
/// 使用 cudarc driver API（`cuMemAlloc_v2` / `cuMemFree_v2`）管理 GPU 内存分配。
/// 构造时初始化 CUDA 驱动并为指定设备创建上下文。
#[cfg(feature = "cuda")]
pub struct CudaMemoryPool {
    /// 设备 ID
    device_id: i32,
    /// 已分配内存（字节）
    allocated_memory: AtomicU64,
    /// 最大内存（字节）
    max_memory: u64,
    /// 配置
    config: CudaPoolConfig,
    /// CUDA 上下文句柄（由 cudarc 管理生命周期）
    _ctx: cudarc::driver::sys::CUcontext,
}

#[cfg(feature = "cuda")]
impl CudaMemoryPool {
    /// 创建新的 CUDA 内存池
    ///
    /// 初始化 CUDA 驱动并为 `device_id` 创建上下文。
    /// 若 CUDA 不可用或设备无效，返回错误。
    pub fn new(device_id: i32, config: CudaPoolConfig) -> Result<Self, String> {
        let max_memory = (config.max_memory_mb * 1024 * 1024) as u64;

        info!(
            "Creating CudaMemoryPool for device {} with max_memory={}MB",
            device_id, config.max_memory_mb
        );

        // 初始化 CUDA 驱动并创建上下文
        cudarc::driver::result::init().map_err(|e| format!("CUDA driver init failed: {}", e))?;

        let device = cudarc::driver::result::device::get(device_id)
            .map_err(|e| format!("CUDA device {} not found: {}", device_id, e))?;

        let ctx = unsafe {
            cudarc::driver::result::primary_ctx::retain(device).map_err(|e| {
                format!(
                    "CUDA context creation failed for device {}: {}",
                    device_id, e
                )
            })?
        };

        info!("CUDA context created for device {}", device_id);

        Ok(Self {
            device_id,
            allocated_memory: AtomicU64::new(0),
            max_memory,
            config,
            _ctx: ctx,
        })
    }

    /// 分配 CUDA 设备内存
    ///
    /// 通过 `cuMemAlloc_v2` 分配真实 GPU 内存。受池最大容量限制。
    pub fn allocate(&mut self, size: usize) -> Result<CudaMemoryPtr, String> {
        let size_u64 = size as u64;

        // 检查是否有足够内存
        let current_allocated = self.allocated_memory.load(Ordering::Relaxed);
        let available = self.max_memory.saturating_sub(current_allocated);

        if size_u64 > available {
            return Err(format!(
                "Insufficient CUDA memory: need {}MB, available {}MB",
                size_u64 / 1024 / 1024,
                available / 1024 / 1024
            ));
        }

        // 通过 cudarc 调用 cuMemAlloc_v2 分配真实 CUDA 内存
        let dev_ptr = unsafe {
            cudarc::driver::result::malloc_sync(size)
                .map_err(|e| format!("CUDA malloc failed for {} bytes: {}", size, e))?
        };

        self.allocated_memory.fetch_add(size_u64, Ordering::Relaxed);

        debug!(
            "Allocated {}MB CUDA memory on device {} at {:?}, total allocated: {}MB",
            size_u64 / 1024 / 1024,
            self.device_id,
            dev_ptr,
            self.allocated_memory.load(Ordering::Relaxed) / 1024 / 1024
        );

        Ok(CudaMemoryPtr {
            device_id: self.device_id,
            size,
            ptr: dev_ptr,
        })
    }

    /// 释放 CUDA 设备内存
    ///
    /// 通过 `cuMemFree_v2` 释放真实 GPU 内存。
    pub fn deallocate(&mut self, ptr: CudaMemoryPtr) {
        if ptr.size == 0 {
            warn!("Attempted to deallocate CUDA memory with size 0");
            return;
        }

        // 通过 cudarc 调用 cuMemFree_v2 释放真实 CUDA 内存
        if let Err(e) = unsafe { cudarc::driver::result::free_sync(ptr.ptr) } {
            warn!("CUDA free failed for device {}: {}", self.device_id, e);
        }

        self.allocated_memory
            .fetch_sub(ptr.size as u64, Ordering::Relaxed);

        debug!(
            "Deallocated {}MB CUDA memory on device {}, total allocated: {}MB",
            ptr.size / 1024 / 1024,
            self.device_id,
            self.allocated_memory.load(Ordering::Relaxed) / 1024 / 1024
        );

        // 防止 Drop 再次释放（ptr 已被 move 消费）
        std::mem::forget(ptr);
    }

    /// 获取内存使用情况
    pub fn get_memory_usage(&self) -> (u64, u64) {
        let used = self.allocated_memory.load(Ordering::Relaxed);
        (used, self.max_memory)
    }

    /// 获取内存使用率
    pub fn get_memory_usage_percent(&self) -> f64 {
        let used = self.allocated_memory.load(Ordering::Relaxed) as f64;
        let total = self.max_memory as f64;
        (used / total) * 100.0
    }

    /// 清空池
    pub fn clear(&mut self) {
        info!("Clearing CUDA memory pool on device {}...", self.device_id);
        self.allocated_memory.store(0, Ordering::Relaxed);
        info!("CUDA memory pool cleared");
    }
}

/// CUDA 设备内存指针
///
/// 持有 `CUdeviceptr`（真实 CUDA 设备指针）。`Drop` 时通过 `cuMemFree_v2` 释放内存，
/// 确保即使未显式调用 `deallocate()` 也不会泄漏 GPU 内存。
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaMemoryPtr {
    /// 设备 ID
    pub device_id: i32,
    /// 大小（字节）
    pub size: usize,
    /// CUDA 设备指针（cuMemAlloc_v2 返回）
    pub ptr: cudarc::driver::sys::CUdeviceptr,
}

#[cfg(feature = "cuda")]
impl Drop for CudaMemoryPtr {
    fn drop(&mut self) {
        if self.size > 0 && self.ptr != cudarc::driver::sys::CUdeviceptr::default() {
            debug!(
                "Dropping CUDA memory ptr on device {} ({} bytes) — freeing via cuMemFree_v2",
                self.device_id, self.size
            );
            if let Err(e) = unsafe { cudarc::driver::result::free_sync(self.ptr) } {
                warn!(
                    "CUDA free on drop failed for device {}: {}",
                    self.device_id, e
                );
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
/// CUDA 内存池（非 CUDA 版本的占位符）
pub struct CudaMemoryPool {
    _device_id: i32,
    _config: CudaPoolConfig,
}

#[cfg(not(feature = "cuda"))]
impl CudaMemoryPool {
    /// 创建新的 CUDA 内存池（非 CUDA 版本）
    pub fn new(_device_id: i32, _config: CudaPoolConfig) -> Result<Self, String> {
        warn!("CUDA feature not enabled, CudaMemoryPool will be no-op");
        Ok(Self {
            _device_id: 0,
            _config,
        })
    }

    /// 分配内存（非 CUDA 版本，返回错误）
    pub fn allocate(&mut self, _size: usize) -> Result<CudaMemoryPtr, String> {
        Err("CUDA feature not enabled".to_string())
    }

    /// 释放内存（非 CUDA 版本）
    pub fn deallocate(&mut self, _ptr: CudaMemoryPtr) {
        // No-op
    }

    /// 获取内存使用情况（非 CUDA 版本）
    pub fn get_memory_usage(&self) -> (u64, u64) {
        (0, 0)
    }

    /// 获取内存使用率（非 CUDA 版本）
    pub fn get_memory_usage_percent(&self) -> f64 {
        0.0
    }

    /// 清空池（非 CUDA 版本）
    pub fn clear(&mut self) {
        // No-op
    }
}

#[cfg(not(feature = "cuda"))]
/// CUDA 内存指针（非 CUDA 版本的占位符）
#[derive(Debug, Clone)]
pub struct CudaMemoryPtr {
    _size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_pool_creation() {
        let config = CudaPoolConfig::default();
        let pool = CudaMemoryPool::new(0, config);

        assert!(pool.is_ok());

        let pool = pool.unwrap();
        let (used, total) = pool.get_memory_usage();
        assert_eq!(used, 0);
        assert!(total > 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_allocate_deallocate() {
        let config = CudaPoolConfig {
            max_memory_mb: 1024,
            ..Default::default()
        };

        let mut pool = CudaMemoryPool::new(0, config).unwrap();

        // 分配 512MB
        let ptr = pool.allocate(512 * 1024 * 1024);
        assert!(ptr.is_ok());

        let (used, _) = pool.get_memory_usage();
        assert_eq!(used, 512 * 1024 * 1024);

        // 释放内存
        let ptr = ptr.unwrap();
        pool.deallocate(ptr);

        let (used, _) = pool.get_memory_usage();
        assert_eq!(used, 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_insufficient_memory() {
        let config = CudaPoolConfig {
            max_memory_mb: 512,
            ..Default::default()
        };

        let mut pool = CudaMemoryPool::new(0, config).unwrap();

        // 分配 512MB
        let _ptr1 = pool.allocate(512 * 1024 * 1024).unwrap();

        // 尝试再分配 512MB，应该失败
        let ptr2 = pool.allocate(512 * 1024 * 1024);
        assert!(ptr2.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_pool_no_cuda() {
        let config = CudaPoolConfig::default();
        let pool = CudaMemoryPool::new(0, config);

        assert!(pool.is_ok());

        let mut pool = pool.unwrap();

        // 尝试分配应该失败
        let ptr = pool.allocate(1024);
        assert!(ptr.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_pool_no_cuda_memory_usage() {
        let config = CudaPoolConfig {
            max_memory_mb: 2048,
            ..Default::default()
        };
        let pool = CudaMemoryPool::new(0, config).unwrap();

        let (used, total) = pool.get_memory_usage();
        assert_eq!(used, 0);
        assert_eq!(total, 0);

        let percent = pool.get_memory_usage_percent();
        assert_eq!(percent, 0.0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_pool_no_cuda_clear_noop() {
        let config = CudaPoolConfig::default();
        let mut pool = CudaMemoryPool::new(1, config).unwrap();

        pool.clear();
        pool.clear();

        let (used, total) = pool.get_memory_usage();
        assert_eq!(used, 0);
        assert_eq!(total, 0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_pool_no_cuda_new_with_different_device_ids() {
        for device_id in [0, 1, -1, 42] {
            let config = CudaPoolConfig::default();
            let pool = CudaMemoryPool::new(device_id, config);
            assert!(
                pool.is_ok(),
                "Failed to create pool for device {}",
                device_id
            );
        }
    }
}
