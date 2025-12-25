// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::config::model::{DeviceType, PoolingMode};
use crate::monitor::{GpuMemoryStats, MemoryMonitor};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DeviceStatus {
    Available,
    Busy,
    LowMemory,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: String,
    pub status: DeviceStatus,
    pub memory_bytes: Option<u64>,
    pub is_default: bool,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            status: DeviceStatus::Available,
            memory_bytes: None,
            is_default: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub device_id: usize,
    pub name: String,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub utilization_percent: f64,
    pub compute_capability: Option<(u8, u8)>,
    pub supports_float16: bool,
    pub status: DeviceStatus,
}

impl GpuInfo {
    pub fn from_gpu_memory_stats(device_id: usize, name: String, stats: GpuMemoryStats) -> Self {
        Self {
            device_id,
            name,
            total_memory_bytes: stats.total_bytes,
            available_memory_bytes: stats.available_bytes,
            utilization_percent: stats.utilization_percent,
            compute_capability: None,
            supports_float16: true,
            status: if stats.utilization_percent > 90.0 {
                DeviceStatus::Busy
            } else if stats.available_bytes < 1024 * 1024 * 1024 {
                DeviceStatus::LowMemory
            } else {
                DeviceStatus::Available
            },
        }
    }
}

#[derive(Clone)]
pub struct DeviceManager {
    devices: Arc<RwLock<Vec<DeviceInfo>>>,
    memory_monitor: Arc<MemoryMonitor>,
    auto_fallback_enabled: bool,
    memory_threshold_percent: u64,
    initialized: Arc<AtomicBool>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self::with_monitor(Arc::new(MemoryMonitor::new()))
    }

    pub fn with_monitor(memory_monitor: Arc<MemoryMonitor>) -> Self {
        Self {
            devices: Arc::new(RwLock::new(Vec::new())),
            memory_monitor,
            auto_fallback_enabled: true,
            memory_threshold_percent: 85,
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn ensure_initialized(&self) {
        if self.initialized.load(Ordering::SeqCst) {
            return;
        }
        self.initialize_devices().await;
        self.initialized.store(true, Ordering::SeqCst);
    }

    async fn initialize_devices(&self) {
        let mut devices = Vec::new();

        devices.push(DeviceInfo {
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            status: DeviceStatus::Available,
            memory_bytes: None,
            is_default: true,
        });

        if let Some(gpu_stats) = self.memory_monitor.get_gpu_stats().await {
            devices.push(DeviceInfo {
                device_type: DeviceType::Cuda,
                name: format!("GPU {}", gpu_stats.device_id),
                status: DeviceStatus::Available,
                memory_bytes: Some(gpu_stats.total_bytes),
                is_default: false,
            });
        }

        let mut devices_rw = self.devices.write().await;
        *devices_rw = devices;
    }

    pub async fn refresh_devices(&self) {
        self.ensure_initialized().await;
        let mut devices = self.devices.write().await;

        if let Some(gpu_stats) = self.memory_monitor.get_gpu_stats().await {
            for device in devices.iter_mut() {
                if let DeviceType::Cuda = device.device_type {
                    device.memory_bytes = Some(gpu_stats.total_bytes);
                    device.status = if gpu_stats.utilization_percent > 90.0 {
                        DeviceStatus::Busy
                    } else if gpu_stats.available_bytes < 1024 * 1024 * 1024 {
                        DeviceStatus::LowMemory
                    } else {
                        DeviceStatus::Available
                    };
                }
            }
        }
    }

    pub async fn select_device(
        &self,
        preferred: &DeviceType,
        require_gpu: bool,
    ) -> DeviceType {
        self.ensure_initialized().await;
        let devices = self.devices.read().await;

        if require_gpu {
            for device in devices.iter() {
                if device.device_type == DeviceType::Cuda {
                    if device.status == DeviceStatus::Available {
                        debug!("Selected GPU device: {}", device.name);
                        return device.device_type.clone();
                    }
                }
            }
            warn!("No available GPU, falling back to CPU");
            return DeviceType::Cpu;
        }

        for device in devices.iter() {
            if device.device_type == *preferred && device.status == DeviceStatus::Available {
                debug!("Selected preferred device: {}", device.name);
                return device.device_type.clone();
            }
        }

        for device in devices.iter() {
            if device.status == DeviceStatus::Available {
                debug!("Fell back to available device: {}", device.name);
                return device.device_type.clone();
            }
        }

        warn!("No available devices, using CPU");
        DeviceType::Cpu
    }

    pub async fn check_device_available(&self, device: &DeviceType) -> bool {
        self.ensure_initialized().await;
        let devices = self.devices.read().await;
        devices.iter().any(|d| {
            d.device_type == *device && d.status == DeviceStatus::Available
        })
    }

    pub async fn get_device_info(&self, device: &DeviceType) -> Option<DeviceInfo> {
        self.ensure_initialized().await;
        let devices = self.devices.read().await;
        devices.iter().find(|d| d.device_type == *device).cloned()
    }

    pub async fn list_devices(&self) -> Vec<DeviceInfo> {
        self.ensure_initialized().await;
        self.devices.read().await.clone()
    }

    pub async fn get_gpu_info(&self) -> Vec<GpuInfo> {
        self.ensure_initialized().await;
        let mut gpu_infos = Vec::new();

        if let Some(gpu_stats) = self.memory_monitor.get_gpu_stats().await {
            let devices = self.devices.read().await;
            for device in devices.iter() {
                if let DeviceType::Cuda = device.device_type {
                    gpu_infos.push(GpuInfo::from_gpu_memory_stats(
                        0,
                        device.name.clone(),
                        gpu_stats.clone(),
                    ));
                }
            }
        }

        gpu_infos
    }

    pub async fn check_memory_pressure(&self) -> bool {
        self.memory_monitor
            .check_oom_risk(self.memory_threshold_percent)
            .await
    }

    pub async fn fallback_to_cpu(&self) -> DeviceType {
        info!("Memory pressure detected, falling back to CPU");
        self.memory_monitor.refresh().await;
        DeviceType::Cpu
    }

    pub fn set_auto_fallback(&mut self, enabled: bool) {
        self.auto_fallback_enabled = enabled;
    }

    pub fn is_auto_fallback_enabled(&self) -> bool {
        self.auto_fallback_enabled
    }

    pub fn set_memory_threshold(&mut self, percent: u64) {
        self.memory_threshold_percent = percent.clamp(50, 99);
    }

    pub fn memory_threshold(&self) -> u64 {
        self.memory_threshold_percent
    }

    pub async fn get_optimal_batch_size(&self, device: &DeviceType) -> usize {
        self.ensure_initialized().await;
        match device {
            DeviceType::Cuda => {
                if let Some(gpu_stats) = self.memory_monitor.get_gpu_stats().await {
                    if let Some(available_mb) = gpu_stats.available_bytes.checked_div(1024 * 1024) {
                        return std::cmp::min(32, (available_mb / 1024) as usize + 1);
                    }
                }
                16
            }
            DeviceType::Cpu => 8,
            DeviceType::Metal => 16,
        }
    }

    pub async fn get_optimal_pooling_mode(&self, device: &DeviceType) -> PoolingMode {
        self.ensure_initialized().await;
        match device {
            DeviceType::Cuda => PoolingMode::Mean,
            DeviceType::Cpu => PoolingMode::Mean,
            DeviceType::Metal => PoolingMode::Mean,
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        let devices = manager.list_devices().await;

        assert!(!devices.is_empty());
        assert!(devices.iter().any(|d| d.device_type == DeviceType::Cpu));
    }

    #[tokio::test]
    async fn test_select_device_cpu() {
        let manager = DeviceManager::new();
        let device = manager.select_device(&DeviceType::Cpu, false).await;

        assert_eq!(device, DeviceType::Cpu);
    }

    #[tokio::test]
    async fn test_check_device_available() {
        let manager = DeviceManager::new();
        let available = manager.check_device_available(&DeviceType::Cpu).await;

        assert!(available);
    }

    #[tokio::test]
    async fn test_get_device_info() {
        let manager = DeviceManager::new();
        let info = manager.get_device_info(&DeviceType::Cpu).await;

        assert!(info.is_some());
        assert_eq!(info.unwrap().device_type, DeviceType::Cpu);
    }

    #[tokio::test]
    async fn test_list_devices() {
        let manager = DeviceManager::new();
        let devices = manager.list_devices().await;

        assert!(!devices.is_empty());
    }

    #[tokio::test]
    async fn test_memory_threshold_setting() {
        let mut manager = DeviceManager::new();
        manager.set_memory_threshold(75);
        assert_eq!(manager.memory_threshold(), 75);

        manager.set_memory_threshold(150);
        assert_eq!(manager.memory_threshold(), 99);
    }

    #[tokio::test]
    async fn test_auto_fallback_setting() {
        let mut manager = DeviceManager::new();
        assert!(manager.is_auto_fallback_enabled());

        manager.set_auto_fallback(false);
        assert!(!manager.is_auto_fallback_enabled());
    }

    #[tokio::test]
    async fn test_get_optimal_batch_size() {
        let manager = DeviceManager::new();

        let cpu_batch = manager.get_optimal_batch_size(&DeviceType::Cpu).await;
        assert!(cpu_batch > 0);
    }

    #[tokio::test]
    async fn test_get_optimal_pooling_mode() {
        let manager = DeviceManager::new();

        let cpu_pooling = manager.get_optimal_pooling_mode(&DeviceType::Cpu).await;
        assert_eq!(cpu_pooling, PoolingMode::Mean);
    }
}
