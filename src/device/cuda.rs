// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::config::model::DeviceType;
use crate::device::memory_optimizer::{
    GpuMemoryConfig, ModelMemoryRequirements, SharedGpuMemoryManager,
};
use crate::device::{DeviceCapability, DeviceInfo, DeviceStatus};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CudaGpuInfo {
    pub device_id: usize,
    pub name: String,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub compute_capability: (u8, u8),
    pub supports_float16: bool,
    pub supports_tensor_cores: bool,
    pub sm_count: u32,
    pub max_threads_per_multiprocessor: u32,
    pub cuda_cores_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaDevice {
    device_id: usize,
    name: String,
    total_memory: u64,
    compute_capability: (u8, u8),
    capability: DeviceCapability,
}

impl CudaDevice {
    #[inline]
    pub fn new(
        device_id: usize,
        name: String,
        total_memory: u64,
        compute_capability: (u8, u8),
    ) -> Self {
        let (major, _minor) = compute_capability;
        let supports_float16 = major >= 7;
        let supports_tensor_cores = major >= 7;

        Self {
            device_id,
            name,
            total_memory,
            compute_capability,
            capability: DeviceCapability {
                supports_float16,
                supports_tensor_cores,
                max_memory_bytes: total_memory,
                compute_capability: Some(compute_capability),
            },
        }
    }

    #[inline]
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    #[inline]
    pub fn compute_capability(&self) -> (u8, u8) {
        self.compute_capability
    }

    #[inline]
    pub fn capability(&self) -> &DeviceCapability {
        &self.capability
    }

    pub fn info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cuda,
            name: self.name.clone(),
            status: DeviceStatus::Available,
            memory_bytes: Some(self.total_memory),
            is_default: self.device_id == 0,
        }
    }
}

#[derive(Clone)]
pub struct CudaDeviceManager {
    devices: Arc<RwLock<Vec<CudaDevice>>>,
    primary_device_id: Arc<RwLock<Option<usize>>>,
    memory_managers: Arc<RwLock<HashMap<usize, SharedGpuMemoryManager>>>,
    initialized: Arc<RwLock<bool>>,
}

impl CudaDeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(Vec::new())),
            primary_device_id: Arc::new(RwLock::new(None)),
            memory_managers: Arc::new(RwLock::new(HashMap::new())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn initialize(&self) -> Result<(), String> {
        let mut initialized = self.initialized.write().await;
        if *initialized {
            return Ok(());
        }

        match self.detect_cuda_devices().await {
            Ok(devices) => {
                let mut devices_guard = self.devices.write().await;
                *devices_guard = devices;

                // 为每个设备创建内存管理器
                let mut memory_managers = self.memory_managers.write().await;
                memory_managers.clear();

                for device in &*devices_guard {
                    let memory_config = GpuMemoryConfig::default();
                    let memory_manager =
                        SharedGpuMemoryManager::new(device.total_memory(), memory_config);
                    memory_managers.insert(device.device_id(), memory_manager);
                    info!(
                        "Created memory manager for device {}: {} MB",
                        device.device_id(),
                        device.total_memory() / (1024 * 1024)
                    );
                }

                if !devices_guard.is_empty() {
                    let mut primary = self.primary_device_id.write().await;
                    *primary = Some(0);
                }

                *initialized = true;
                info!(
                    "CUDA device manager initialized with {} device(s)",
                    devices_guard.len()
                );
                Ok(())
            }
            Err(e) => {
                warn!("CUDA initialization failed: {}", e);
                Err(e)
            }
        }
    }

    async fn detect_cuda_devices(&self) -> Result<Vec<CudaDevice>, String> {
        #[cfg(feature = "cuda")]
        {
            self.detect_with_candle().await
        }

        #[cfg(not(feature = "cuda"))]
        {
            self.detect_compatible().await
        }
    }

    #[cfg(feature = "cuda")]
    async fn detect_with_candle(&self) -> Result<Vec<CudaDevice>, String> {
        let mut devices = Vec::new();

        match candle_core::Device::cuda_if_available(0) {
            Ok(_device) => {
                let cuda_device = CudaDevice::new(
                    0,
                    "CUDA Device (via candle-core)".to_string(),
                    8 * 1024 * 1024 * 1024, // 8GB default
                    (7, 0),
                );

                devices.push(cuda_device.clone());
                info!("Detected CUDA device via candle-core");
                Ok(devices)
            }
            Err(e) => {
                error!("Failed to initialize CUDA device via candle-core: {}", e);
                Err(format!("CUDA device detection failed: {}", e))
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    async fn detect_compatible(&self) -> Result<Vec<CudaDevice>, String> {
        let mut devices = Vec::new();

        if let Ok(nvidia_info) = detect_nvidia_driver().await
            && nvidia_info.cuda_version.is_some()
        {
            let cuda_device = CudaDevice::new(
                0,
                nvidia_info
                    .device_name
                    .unwrap_or_else(|| "Unknown NVIDIA GPU".to_string()),
                nvidia_info.total_vram_bytes,
                nvidia_info.compute_capability.unwrap_or((7, 0)),
            );
            devices.push(cuda_device.clone());
            info!("Detected compatible CUDA device: {}", cuda_device.name());
        }

        if devices.is_empty() {
            debug!("No CUDA devices detected (CUDA feature not enabled or no NVIDIA GPU found)");
        }

        Ok(devices)
    }

    pub async fn devices(&self) -> Vec<CudaDevice> {
        self.devices.read().await.clone()
    }

    pub async fn primary_device(&self) -> Option<CudaDevice> {
        let devices = self.devices.read().await;
        let primary_id = *self.primary_device_id.read().await;

        match primary_id {
            Some(id) if id < devices.len() => Some(devices[id].clone()),
            Some(id) if !devices.is_empty() => Some(devices[0].clone()),
            _ => devices.first().cloned(),
        }
    }

    pub async fn get_device(&self, device_id: usize) -> Option<CudaDevice> {
        let devices = self.devices.read().await;
        devices.get(device_id).cloned()
    }

    pub async fn device_count(&self) -> usize {
        self.devices.read().await.len()
    }

    pub async fn available_memory(&self, device_id: usize) -> Option<u64> {
        let devices = self.devices.read().await;
        devices.get(device_id).map(|d| d.total_memory())
    }

    pub async fn get_optimal_batch_size(&self, device_id: usize) -> usize {
        if let Some(_device) = self.get_device(device_id).await {
            if let Some(memory_manager) = self.memory_managers.read().await.get(&device_id).cloned()
            {
                // 使用智能内存管理器计算批量大小（需要先注册模型需求）
                // 如果没有模型需求注册，使用原始的简单算法
                let stats = memory_manager.get_performance_stats().await;
                if stats.current_batch_size > 0 {
                    stats.current_batch_size
                } else {
                    // 降级到简单算法
                    let available = memory_manager.get_available_memory().await;
                    let available_mb = available / (1024 * 1024);
                    std::cmp::min(32, (available_mb / 2048) as usize + 1)
                }
            } else {
                16
            }
        } else {
            16
        }
    }

    /// 注册模型的内存需求
    pub async fn register_model_memory_requirements(
        &self,
        device_id: usize,
        requirements: ModelMemoryRequirements,
    ) -> Result<(), String> {
        let memory_managers = self.memory_managers.read().await;
        let memory_manager = memory_managers
            .get(&device_id)
            .ok_or_else(|| format!("Device {} not found", device_id))?
            .clone();
        drop(memory_managers);

        memory_manager.register_model(requirements).await;
        Ok(())
    }

    /// 获取设备的内存使用率
    pub async fn get_memory_usage_percent(&self, device_id: usize) -> f64 {
        if let Some(memory_manager) = self.memory_managers.read().await.get(&device_id).cloned() {
            memory_manager.get_memory_usage_percent().await
        } else {
            0.0
        }
    }

    /// 动态调整批量大小
    pub async fn adjust_batch_size_dynamically(
        &self,
        device_id: usize,
        latency_ms: f64,
        memory_usage_percent: f64,
    ) {
        if let Some(memory_manager) = self.memory_managers.read().await.get(&device_id).cloned() {
            memory_manager
                .adjust_batch_size_dynamically(latency_ms, memory_usage_percent)
                .await
        }
    }

    /// 获取性能统计
    pub async fn get_performance_stats(
        &self,
        device_id: usize,
    ) -> Option<crate::device::memory_optimizer::PerformanceStats> {
        if let Some(manager) = self.memory_managers.read().await.get(&device_id).cloned() {
            Some(manager.get_performance_stats().await)
        } else {
            None
        }
    }

    pub async fn is_supported(&self) -> bool {
        !self.devices.read().await.is_empty()
    }

    pub async fn reset_primary_device(&self, device_id: usize) -> Result<(), String> {
        let devices = self.devices.read().await;
        if device_id >= devices.len() {
            return Err(format!("Device {} not found", device_id));
        }

        let mut primary = self.primary_device_id.write().await;
        *primary = Some(device_id);
        info!("Primary CUDA device set to: {}", devices[device_id].name());

        Ok(())
    }
}

impl Default for CudaDeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

async fn detect_nvidia_driver() -> Result<NvidiaDriverInfo, String> {
    use std::process::Command;
    use std::str;

    let mut info = NvidiaDriverInfo {
        cuda_version: None,
        driver_version: None,
        device_name: None,
        total_vram_bytes: 0,
        compute_capability: None,
    };

    let nvidia_smi_output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;

    if nvidia_smi_output.status.success()
        && let Ok(output) = str::from_utf8(&nvidia_smi_output.stdout)
    {
        let parts: Vec<&str> = output.trim().split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            info.device_name = Some(parts[0].to_string());

            if let Ok(vram_mb) = parts[1].parse::<u64>() {
                info.total_vram_bytes = vram_mb * 1024 * 1024;
            }

            let cc_parts: Vec<&str> = parts[2].split('.').collect();
            if cc_parts.len() >= 2
                && let (Ok(major), Ok(minor)) =
                    (cc_parts[0].parse::<u8>(), cc_parts[1].parse::<u8>())
            {
                info.compute_capability = Some((major, minor));
            }
        }
    }

    let cuda_version_output = Command::new("nvidia-smi")
        .arg("--query-gpu=driver_version")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| str::from_utf8(&o.stdout).ok().map(|s| s.trim().to_string()));

    info.driver_version = cuda_version_output;

    if info.total_vram_bytes > 0 {
        info.cuda_version = Some(11);
    }

    Ok(info)
}

#[derive(Debug)]
struct NvidiaDriverInfo {
    cuda_version: Option<u32>,
    driver_version: Option<String>,
    device_name: Option<String>,
    total_vram_bytes: u64,
    compute_capability: Option<(u8, u8)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cuda_device_manager_creation() {
        let manager = CudaDeviceManager::new();

        assert_eq!(manager.device_count().await, 0);
        assert!(!manager.is_supported().await);
    }

    #[tokio::test]
    async fn test_cuda_device_info() {
        let device = CudaDevice::new(
            0,
            "NVIDIA GeForce RTX 3060".to_string(),
            12 * 1024 * 1024 * 1024,
            (8, 6),
        );

        assert_eq!(device.device_id(), 0);
        assert_eq!(device.name(), "NVIDIA GeForce RTX 3060");
        assert_eq!(device.total_memory(), 12 * 1024 * 1024 * 1024);
        assert_eq!(device.compute_capability(), (8, 6));

        let capability = device.capability();
        assert!(capability.supports_float16);
        assert!(capability.supports_tensor_cores);
        assert_eq!(capability.max_memory_bytes, 12 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_cuda_device_info_conversion() {
        let device = CudaDevice::new(
            1,
            "NVIDIA GeForce GTX 1080".to_string(),
            8 * 1024 * 1024 * 1024,
            (6, 1),
        );

        let info = device.info();
        assert_eq!(info.device_type, DeviceType::Cuda);
        assert_eq!(info.name, "NVIDIA GeForce GTX 1080");
        assert_eq!(info.status, DeviceStatus::Available);
        assert_eq!(info.memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert!(!info.is_default);
    }

    #[tokio::test]
    async fn test_compute_capability_support() {
        let sm_70_device = CudaDevice::new(0, "V100".to_string(), 16 * 1024 * 1024 * 1024, (7, 0));
        assert!(sm_70_device.capability.supports_float16);
        assert!(sm_70_device.capability.supports_tensor_cores);

        let sm_60_device =
            CudaDevice::new(0, "Pascal GPU".to_string(), 8 * 1024 * 1024 * 1024, (6, 0));
        assert!(!sm_60_device.capability.supports_float16);
        assert!(!sm_60_device.capability.supports_tensor_cores);

        let sm_50_device =
            CudaDevice::new(0, "Maxwell GPU".to_string(), 4 * 1024 * 1024 * 1024, (5, 0));
        assert!(!sm_50_device.capability.supports_float16);
        assert!(!sm_50_device.capability.supports_tensor_cores);
    }

    #[tokio::test]
    async fn test_reset_primary_device() {
        let manager = CudaDeviceManager::new();

        assert_eq!(manager.primary_device().await, None);

        let result = manager.reset_primary_device(0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_initialize_without_cuda_hardware() {
        let manager = CudaDeviceManager::new();

        let result = manager.initialize().await;
        assert!(
            result.is_ok(),
            "initialize should succeed even without CUDA"
        );
        let count = manager.device_count().await;
        assert_eq!(manager.is_supported().await, count > 0);
    }

    #[tokio::test]
    async fn test_initialize_is_idempotent() {
        let manager = CudaDeviceManager::new();

        manager.initialize().await.unwrap();
        let first_count = manager.device_count().await;

        manager.initialize().await.unwrap();
        let second_count = manager.device_count().await;

        assert_eq!(first_count, second_count);
    }

    #[tokio::test]
    async fn test_devices_returns_empty_after_init() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        let devices = manager.devices().await;
        assert_eq!(devices.len(), manager.device_count().await);
    }

    #[tokio::test]
    async fn test_primary_device_returns_none_when_empty() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        let count = manager.device_count().await;
        let primary = manager.primary_device().await;
        assert_eq!(primary.is_some(), count > 0);
    }

    #[tokio::test]
    async fn test_get_device_returns_none_for_invalid_id() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        assert!(manager.get_device(999).await.is_none());
    }

    #[tokio::test]
    async fn test_available_memory_returns_none_for_invalid_id() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        assert!(manager.available_memory(999).await.is_none());
    }

    #[tokio::test]
    async fn test_get_optimal_batch_size_no_device_returns_default() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        assert_eq!(manager.get_optimal_batch_size(0).await, 16);
        assert_eq!(manager.get_optimal_batch_size(999).await, 16);
    }

    #[tokio::test]
    async fn test_register_model_requirements_unknown_device_errors() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        let requirements = ModelMemoryRequirements {
            model_name: "test-model".to_string(),
            base_memory_bytes: 100_000_000,
            per_token_memory_bytes: 1000,
            per_vector_memory_bytes: 4000,
            max_sequence_length: 512,
        };

        let result = manager
            .register_model_memory_requirements(999, requirements)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_memory_usage_percent_no_manager_returns_zero() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        assert_eq!(manager.get_memory_usage_percent(0).await, 0.0);
        assert_eq!(manager.get_memory_usage_percent(999).await, 0.0);
    }

    #[tokio::test]
    async fn test_adjust_batch_size_dynamically_no_manager_no_panic() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        manager.adjust_batch_size_dynamically(0, 50.0, 60.0).await;
        manager.adjust_batch_size_dynamically(999, 50.0, 60.0).await;
    }

    #[tokio::test]
    async fn test_get_performance_stats_no_manager_returns_none() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        assert!(manager.get_performance_stats(999).await.is_none());
    }

    #[tokio::test]
    async fn test_reset_primary_device_invalid_id_errors() {
        let manager = CudaDeviceManager::new();
        manager.initialize().await.unwrap();

        let result = manager.reset_primary_device(5).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_cuda_device_default_manager_uses_new() {
        let manager = CudaDeviceManager::default();
        assert_eq!(manager.device_count().await, 0);
        assert!(!manager.is_supported().await);
    }

    #[test]
    fn test_cuda_gpu_info_fields() {
        let info = CudaGpuInfo {
            device_id: 0,
            name: "Test GPU".to_string(),
            total_memory_bytes: 8 * 1024 * 1024 * 1024,
            available_memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: (8, 6),
            supports_float16: true,
            supports_tensor_cores: true,
            sm_count: 84,
            max_threads_per_multiprocessor: 1536,
            cuda_cores_count: 5888,
        };

        assert_eq!(info.device_id, 0);
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.total_memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(info.available_memory_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(info.compute_capability, (8, 6));
        assert!(info.supports_float16);
        assert!(info.supports_tensor_cores);
        assert_eq!(info.sm_count, 84);
        assert_eq!(info.max_threads_per_multiprocessor, 1536);
        assert_eq!(info.cuda_cores_count, 5888);
    }

    #[test]
    fn test_cuda_device_info_is_default_for_device_zero() {
        let device = CudaDevice::new(0, "RTX 4090".to_string(), 24 * 1024 * 1024 * 1024, (8, 9));

        let info = device.info();
        assert!(info.is_default);
    }

    #[test]
    fn test_cuda_device_info_not_default_for_nonzero_id() {
        let device = CudaDevice::new(2, "RTX 4090".to_string(), 24 * 1024 * 1024 * 1024, (8, 9));

        let info = device.info();
        assert!(!info.is_default);
    }

    #[test]
    fn test_cuda_device_capability_set_for_high_compute_capability() {
        let device = CudaDevice::new(0, "H100".to_string(), 80 * 1024 * 1024 * 1024, (9, 0));

        let cap = device.capability();
        assert!(cap.supports_float16);
        assert!(cap.supports_tensor_cores);
        assert_eq!(cap.compute_capability, Some((9, 0)));
        assert_eq!(cap.max_memory_bytes, 80 * 1024 * 1024 * 1024);
    }
}
