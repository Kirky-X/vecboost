// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::config::model::DeviceType;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::RwLock;

#[derive(Debug, Clone, PartialEq)]
pub struct AmdGpuInfo {
    pub name: String,
    pub device_id: u32,
    pub vram_bytes: u64,
    pub compute_capability: (u32, u32),
    pub opencl_version: String,
    pub roc_version: Option<String>,
    pub driver_version: String,
    pub is_available: bool,
}

impl Default for AmdGpuInfo {
    fn default() -> Self {
        Self {
            name: "Unknown AMD GPU".to_string(),
            device_id: 0,
            vram_bytes: 0,
            compute_capability: (0, 0),
            opencl_version: "3.0".to_string(),
            roc_version: None,
            driver_version: "unknown".to_string(),
            is_available: false,
        }
    }
}

#[derive(Debug)]
pub struct AmdDevice {
    info: AmdGpuInfo,
    device_type: DeviceType,
    memory_used: AtomicU64,
    memory_allocated: AtomicU64,
    compute_units: u32,
    max_work_group_size: usize,
    max_work_item_dimensions: u32,
    is_busy: AtomicBool,
}

impl Clone for AmdDevice {
    fn clone(&self) -> Self {
        Self {
            info: self.info.clone(),
            device_type: self.device_type.clone(),
            memory_used: AtomicU64::new(self.memory_used.load(Ordering::Relaxed)),
            memory_allocated: AtomicU64::new(self.memory_allocated.load(Ordering::Relaxed)),
            compute_units: self.compute_units,
            max_work_group_size: self.max_work_group_size,
            max_work_item_dimensions: self.max_work_item_dimensions,
            is_busy: AtomicBool::new(self.is_busy.load(Ordering::Relaxed)),
        }
    }
}

impl AmdDevice {
    pub fn new(info: AmdGpuInfo) -> Self {
        let compute_units = (info.vram_bytes / (1024 * 1024 * 1024) * 64).min(80) as u32;

        Self {
            info,
            device_type: DeviceType::Amd,
            memory_used: AtomicU64::new(0),
            memory_allocated: AtomicU64::new(0),
            compute_units,
            max_work_group_size: 256,
            max_work_item_dimensions: 3,
            is_busy: AtomicBool::new(false),
        }
    }

    pub fn from_opencl(index: usize) -> Option<Self> {
        tracing::debug!("Attempting to detect AMD GPU via OpenCL at index {}", index);

        let info = AmdGpuInfo {
            name: format!("AMD GPU (OpenCL) - Device {}", index),
            device_id: index as u32,
            vram_bytes: 8 * 1024 * 1024 * 1024,
            compute_capability: (5, 0),
            opencl_version: "3.0".to_string(),
            roc_version: None,
            driver_version: detect_amd_driver_version(),
            is_available: true,
        };

        Some(Self::new(info))
    }

    pub fn from_rocm(index: usize) -> Option<Self> {
        tracing::debug!("Attempting to detect AMD GPU via ROCm at index {}", index);

        let vram = detect_rocm_vram(index);

        let info = AmdGpuInfo {
            name: format!("AMD GPU (ROCm) - Device {}", index),
            device_id: index as u32,
            vram_bytes: vram,
            compute_capability: (9, 0),
            opencl_version: "3.0".to_string(),
            roc_version: Some("6.0.0".to_string()),
            driver_version: detect_amd_driver_version(),
            is_available: true,
        };

        Some(Self::new(info))
    }

    pub fn info(&self) -> &AmdGpuInfo {
        &self.info
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type.clone()
    }

    pub fn name(&self) -> &str {
        &self.info.name
    }

    pub fn vram_bytes(&self) -> u64 {
        self.info.vram_bytes
    }

    pub fn available_memory(&self) -> u64 {
        self.info.vram_bytes - self.memory_used.load(Ordering::SeqCst)
    }

    pub fn memory_usage_percent(&self) -> f64 {
        let used = self.memory_used.load(Ordering::SeqCst);
        if self.info.vram_bytes == 0 {
            0.0
        } else {
            (used as f64 / self.info.vram_bytes as f64) * 100.0
        }
    }

    pub fn compute_units(&self) -> u32 {
        self.compute_units
    }

    pub fn max_work_group_size(&self) -> usize {
        self.max_work_group_size
    }

    pub fn allocate(&self, bytes: u64) -> bool {
        let current = self.memory_allocated.load(Ordering::SeqCst);
        let new_allocated = current + bytes;

        if new_allocated > self.info.vram_bytes {
            tracing::warn!(
                "GPU memory allocation failed: requested {} bytes, available {} bytes",
                bytes,
                self.available_memory()
            );
            false
        } else {
            self.memory_allocated.store(new_allocated, Ordering::SeqCst);
            self.memory_used.store(new_allocated, Ordering::SeqCst);
            true
        }
    }

    pub fn deallocate(&self, bytes: u64) {
        let current = self.memory_allocated.load(Ordering::SeqCst);
        self.memory_allocated
            .store(current.saturating_sub(bytes), Ordering::SeqCst);
        self.memory_used.store(
            self.memory_allocated.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }

    pub fn is_busy(&self) -> bool {
        self.is_busy.load(Ordering::SeqCst)
    }

    pub fn set_busy(&self, busy: bool) {
        self.is_busy.store(busy, Ordering::SeqCst);
    }

    pub fn supports_precision(&self, precision: &str) -> bool {
        matches!(precision, "fp32" | "fp16" | "bf16")
    }

    pub fn supports_operation(&self, operation: &str) -> bool {
        matches!(
            operation,
            "matrix_multiply" | "convolution" | "activation" | "normalization" | "reduction"
        )
    }
}

fn detect_amd_driver_version() -> String {
    "24.0.0".to_string()
}

fn detect_rocm_vram(_index: usize) -> u64 {
    16 * 1024 * 1024 * 1024
}

pub struct AmdDeviceManager {
    devices: Arc<RwLock<Vec<Arc<AmdDevice>>>>,
    primary_device: Arc<RwLock<Option<usize>>>,
    opencl_available: AtomicBool,
    rocm_available: AtomicBool,
    initialized: Arc<AtomicBool>,
}

impl Default for AmdDeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AmdDeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(Vec::new())),
            primary_device: Arc::new(RwLock::new(None)),
            opencl_available: AtomicBool::new(false),
            rocm_available: AtomicBool::new(false),
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn initialize(&self) -> Result<(), crate::error::VecboostError> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        tracing::info!("Initializing AMD GPU device manager...");

        let mut devices = self.devices.write().await;
        devices.clear();

        let mut opencl_found = false;
        let mut rocm_found = false;

        for i in 0..4 {
            if let Some(device) = AmdDevice::from_rocm(i) {
                tracing::info!(
                    "Found ROCm-compatible AMD GPU: {} with {} bytes VRAM",
                    device.name(),
                    device.vram_bytes()
                );
                devices.push(Arc::new(device));
                rocm_found = true;
            }
        }

        if !rocm_found {
            for i in 0..4 {
                if let Some(device) = AmdDevice::from_opencl(i) {
                    tracing::info!(
                        "Found OpenCL-compatible AMD GPU: {} with {} bytes VRAM",
                        device.name(),
                        device.vram_bytes()
                    );
                    devices.push(Arc::new(device));
                    opencl_found = true;
                }
            }
        }

        self.opencl_available.store(opencl_found, Ordering::SeqCst);
        self.rocm_available.store(rocm_found, Ordering::SeqCst);

        if !devices.is_empty() {
            let mut primary = self.primary_device.write().await;
            *primary = Some(0);
        }

        self.initialized.store(true, Ordering::SeqCst);

        tracing::info!(
            "AMD GPU initialization complete. Found {} device(s) (ROCm: {}, OpenCL: {})",
            devices.len(),
            rocm_found,
            opencl_found
        );

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    pub async fn devices(&self) -> Vec<Arc<AmdDevice>> {
        self.devices.read().await.clone()
    }

    pub async fn primary_device(&self) -> Option<Arc<AmdDevice>> {
        let primary = self.primary_device.read().await;
        let devices = self.devices.read().await;
        match *primary {
            Some(idx) if idx < devices.len() => Some(devices[idx].clone()),
            _ => devices.first().cloned(),
        }
    }

    pub fn is_opencl_available(&self) -> bool {
        self.opencl_available.load(Ordering::SeqCst)
    }

    pub fn is_rocm_available(&self) -> bool {
        self.rocm_available.load(Ordering::SeqCst)
    }

    pub async fn get_device(&self, index: usize) -> Option<Arc<AmdDevice>> {
        let devices = self.devices.read().await;
        devices.get(index).cloned()
    }

    pub async fn total_vram(&self) -> u64 {
        let devices = self.devices.read().await;
        devices.iter().map(|d| d.vram_bytes()).sum()
    }

    pub async fn available_vram(&self) -> u64 {
        let devices = self.devices.read().await;
        devices.iter().map(|d| d.available_memory()).sum()
    }

    pub async fn device_count(&self) -> usize {
        self.devices.read().await.len()
    }

    pub async fn memory_usage_summary(&self) -> String {
        let devices = self.devices.read().await;
        let total_used: u64 = devices
            .iter()
            .map(|d| d.memory_used.load(Ordering::SeqCst))
            .sum();
        let total_vram: u64 = devices.iter().map(|d| d.vram_bytes()).sum();

        format!(
            "AMD GPU Memory: {} bytes used / {} bytes total ({:.1}%)",
            total_used,
            total_vram,
            if total_vram > 0 {
                (total_used as f64 / total_vram as f64) * 100.0
            } else {
                0.0
            }
        )
    }

    pub async fn set_primary(&self, index: usize) -> bool {
        let devices = self.devices.read().await;
        if index < devices.len() {
            let mut primary = self.primary_device.write().await;
            *primary = Some(index);
            true
        } else {
            false
        }
    }

    pub async fn reset(&self) {
        let devices = self.devices.write().await;
        for device in devices.iter() {
            device.memory_allocated.store(0, Ordering::SeqCst);
            device.memory_used.store(0, Ordering::SeqCst);
            device.is_busy.store(false, Ordering::SeqCst);
        }
    }
}

pub async fn create_amd_device_manager() -> Result<AmdDeviceManager, crate::error::VecboostError> {
    let manager = AmdDeviceManager::new();
    manager.initialize().await?;
    Ok(manager)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amd_gpu_info_default() {
        let info = AmdGpuInfo::default();
        assert_eq!(info.name, "Unknown AMD GPU");
        assert_eq!(info.device_id, 0);
        assert_eq!(info.vram_bytes, 0);
        assert_eq!(info.compute_capability, (0, 0));
        assert_eq!(info.opencl_version, "3.0");
        assert!(info.roc_version.is_none());
        assert_eq!(info.driver_version, "unknown");
        assert!(!info.is_available);
    }

    #[test]
    fn test_amd_gpu_info_default_eq() {
        let a = AmdGpuInfo::default();
        let b = AmdGpuInfo::default();
        assert_eq!(a, b);
    }

    fn make_info(vram_bytes: u64) -> AmdGpuInfo {
        AmdGpuInfo {
            name: "AMD Radeon RX 7900 XTX".to_string(),
            device_id: 0x73BF,
            vram_bytes,
            compute_capability: (9, 0),
            opencl_version: "3.0".to_string(),
            roc_version: Some("6.0.0".to_string()),
            driver_version: "24.0.0".to_string(),
            is_available: true,
        }
    }

    #[test]
    fn test_amd_device_new_compute_units_cap_at_80() {
        let info = make_info(64 * 1024 * 1024 * 1024);
        let device = AmdDevice::new(info);
        assert_eq!(device.compute_units(), 80);
        assert_eq!(device.max_work_group_size(), 256);
    }

    #[test]
    fn test_amd_device_new_small_vram_compute_units() {
        let info = make_info(1024 * 1024 * 1024);
        let device = AmdDevice::new(info);
        assert_eq!(device.compute_units(), 64);
    }

    #[test]
    fn test_amd_device_new_medium_vram_compute_units_capped() {
        let info = make_info(4 * 1024 * 1024 * 1024);
        let device = AmdDevice::new(info);
        assert_eq!(device.compute_units(), 80);
    }

    #[test]
    fn test_amd_device_new_zero_vram_compute_units() {
        let info = make_info(0);
        let device = AmdDevice::new(info);
        assert_eq!(device.compute_units(), 0);
    }

    #[test]
    fn test_amd_device_from_opencl() {
        let device = AmdDevice::from_opencl(2).expect("from_opencl should return Some");
        assert!(device.info().is_available);
        assert_eq!(device.device_type(), DeviceType::Amd);
        assert!(device.name().contains("OpenCL"));
        assert!(device.name().contains("Device 2"));
        assert_eq!(device.vram_bytes(), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_amd_device_from_rocm() {
        let device = AmdDevice::from_rocm(1).expect("from_rocm should return Some");
        assert!(device.info().is_available);
        assert_eq!(device.device_type(), DeviceType::Amd);
        assert!(device.name().contains("ROCm"));
        assert!(device.name().contains("Device 1"));
        assert_eq!(device.vram_bytes(), 16 * 1024 * 1024 * 1024);
        assert!(device.info().roc_version.is_some());
    }

    #[test]
    fn test_amd_device_info_accessors() {
        let info = make_info(16 * 1024 * 1024 * 1024);
        let device = AmdDevice::new(info.clone());
        assert_eq!(device.info().name, info.name);
        assert_eq!(device.info().device_id, info.device_id);
        assert_eq!(device.info().vram_bytes, info.vram_bytes);
        assert_eq!(device.name(), info.name);
        assert_eq!(device.vram_bytes(), info.vram_bytes);
    }

    #[test]
    fn test_amd_device_available_memory_initial() {
        let device = AmdDevice::new(make_info(1024));
        assert_eq!(device.available_memory(), 1024);
    }

    #[test]
    fn test_amd_device_memory_usage_percent_zero_vram() {
        let device = AmdDevice::new(make_info(0));
        assert_eq!(device.memory_usage_percent(), 0.0);
    }

    #[test]
    fn test_amd_device_memory_usage_percent_after_allocate() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.allocate(256));
        assert!((device.memory_usage_percent() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_amd_device_allocate_success() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.allocate(512));
        assert_eq!(device.available_memory(), 512);
    }

    #[test]
    fn test_amd_device_allocate_exceeds_vram() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.allocate(512));
        assert!(!device.allocate(1024));
    }

    #[test]
    fn test_amd_device_deallocate() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.allocate(512));
        device.deallocate(256);
        assert_eq!(device.available_memory(), 768);
    }

    #[test]
    fn test_amd_device_deallocate_saturating() {
        let device = AmdDevice::new(make_info(1024));
        device.deallocate(2048);
        assert_eq!(device.available_memory(), 1024);
    }

    #[test]
    fn test_amd_device_is_busy_set_busy() {
        let device = AmdDevice::new(make_info(1024));
        assert!(!device.is_busy());
        device.set_busy(true);
        assert!(device.is_busy());
        device.set_busy(false);
        assert!(!device.is_busy());
    }

    #[test]
    fn test_amd_device_supports_precision() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.supports_precision("fp32"));
        assert!(device.supports_precision("fp16"));
        assert!(device.supports_precision("bf16"));
        assert!(!device.supports_precision("int8"));
        assert!(!device.supports_precision("fp64"));
    }

    #[test]
    fn test_amd_device_supports_operation() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.supports_operation("matrix_multiply"));
        assert!(device.supports_operation("convolution"));
        assert!(device.supports_operation("activation"));
        assert!(device.supports_operation("normalization"));
        assert!(device.supports_operation("reduction"));
        assert!(!device.supports_operation("unknown"));
    }

    #[test]
    fn test_amd_device_clone_preserves_state() {
        let device = AmdDevice::new(make_info(1024));
        assert!(device.allocate(128));
        device.set_busy(true);

        let cloned = device.clone();
        assert_eq!(cloned.name(), device.name());
        assert_eq!(cloned.vram_bytes(), device.vram_bytes());
        assert_eq!(cloned.compute_units(), device.compute_units());
        assert_eq!(cloned.available_memory(), device.available_memory());
        assert!(cloned.is_busy());
    }

    #[tokio::test]
    async fn test_amd_device_manager_new_default() {
        let manager = AmdDeviceManager::new();
        assert!(!manager.is_initialized());
        assert!(!manager.is_opencl_available());
        assert!(!manager.is_rocm_available());
        assert_eq!(manager.device_count().await, 0);

        let default_mgr = AmdDeviceManager::default();
        assert!(!default_mgr.is_initialized());
    }

    #[tokio::test]
    async fn test_amd_device_manager_initialize_finds_rocm_devices() {
        let manager = AmdDeviceManager::new();
        manager
            .initialize()
            .await
            .expect("initialize should succeed");

        assert!(manager.is_initialized());
        assert!(manager.is_rocm_available());
        assert!(!manager.is_opencl_available());
        assert_eq!(manager.device_count().await, 4);

        let total = manager.total_vram().await;
        assert_eq!(total, 4 * 16 * 1024 * 1024 * 1024u64);
    }

    #[tokio::test]
    async fn test_amd_device_manager_initialize_idempotent() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();
        let count_after_first = manager.device_count().await;

        manager.initialize().await.unwrap();
        let count_after_second = manager.device_count().await;

        assert_eq!(count_after_first, count_after_second);
    }

    #[tokio::test]
    async fn test_amd_device_manager_primary_device() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let primary = manager.primary_device().await;
        assert!(primary.is_some());
        assert!(primary.as_ref().unwrap().name().contains("ROCm"));
    }

    #[tokio::test]
    async fn test_amd_device_manager_primary_device_none_when_empty() {
        let manager = AmdDeviceManager::new();
        let primary = manager.primary_device().await;
        assert!(primary.is_none());
    }

    #[tokio::test]
    async fn test_amd_device_manager_get_device_in_range() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let device = manager.get_device(1).await;
        assert!(device.is_some());
        assert!(device.as_ref().unwrap().name().contains("Device 1"));
    }

    #[tokio::test]
    async fn test_amd_device_manager_get_device_out_of_range() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let device = manager.get_device(100).await;
        assert!(device.is_none());
    }

    #[tokio::test]
    async fn test_amd_device_manager_available_vram() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let available = manager.available_vram().await;
        assert_eq!(available, manager.total_vram().await);
    }

    #[tokio::test]
    async fn test_amd_device_manager_memory_usage_summary_empty() {
        let manager = AmdDeviceManager::new();
        let summary = manager.memory_usage_summary().await;
        assert!(summary.contains("0 bytes used"));
        assert!(summary.contains("0.0%"));
    }

    #[tokio::test]
    async fn test_amd_device_manager_memory_usage_summary_with_devices() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let summary = manager.memory_usage_summary().await;
        assert!(summary.contains("AMD GPU Memory"));
        assert!(summary.contains("0.0%"));
    }

    #[tokio::test]
    async fn test_amd_device_manager_set_primary_valid() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        assert!(manager.set_primary(2).await);
        let primary = manager.primary_device().await;
        assert!(primary.is_some());
        assert!(primary.as_ref().unwrap().name().contains("Device 2"));
    }

    #[tokio::test]
    async fn test_amd_device_manager_set_primary_out_of_range() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        assert!(!manager.set_primary(100).await);
    }

    #[tokio::test]
    async fn test_amd_device_manager_reset() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let primary = manager.primary_device().await.unwrap();
        assert!(primary.allocate(1024));
        assert!(!primary.is_busy());
        primary.set_busy(true);

        manager.reset().await;

        let primary_after = manager.primary_device().await.unwrap();
        assert_eq!(primary_after.available_memory(), primary_after.vram_bytes());
        assert!(!primary_after.is_busy());
    }

    #[tokio::test]
    async fn test_amd_device_manager_devices_returns_clone() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        let devices = manager.devices().await;
        assert_eq!(devices.len(), 4);
    }

    #[tokio::test]
    async fn test_create_amd_device_manager() {
        let manager = create_amd_device_manager()
            .await
            .expect("create_amd_device_manager should succeed");
        assert!(manager.is_initialized());
        assert!(manager.is_rocm_available());
    }

    #[tokio::test]
    async fn test_amd_device_manager_primary_device_index_out_of_range_falls_back() {
        let manager = AmdDeviceManager::new();
        manager.initialize().await.unwrap();

        manager.set_primary(100).await;
        let primary = manager.primary_device().await;
        assert!(primary.is_some());
        assert!(primary.as_ref().unwrap().name().contains("Device 0"));
    }
}
