// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::device::{DeviceCapability, DeviceInfo, DeviceStatus};
use crate::config::model::DeviceType;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

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
    initialized: Arc<RwLock<bool>>,
}

impl CudaDeviceManager {
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(Vec::new())),
            primary_device_id: Arc::new(RwLock::new(None)),
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

                if !devices_guard.is_empty() {
                    let mut primary = self.primary_device_id.write().await;
                    *primary = Some(0);
                }

                *initialized = true;
                info!("CUDA device manager initialized with {} device(s)", devices_guard.len());
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
            self.detect_with_cudarc().await
        }

        #[cfg(not(feature = "cuda"))]
        {
            self.detect_compatible().await
        }
    }

    #[cfg(feature = "cuda")]
    async fn detect_with_cudarc(&self) -> Result<Vec<CudaDevice>, String> {
        use cudarc::nvrtc::PtxCompiler;
        use cudarc::driver::{CudaDevice, CudaResult};

        let mut devices = Vec::new();

        match CudaDevice::count() {
            Ok(count) if count > 0 => {
                for i in 0..count {
                    match CudaDevice::new(i) {
                        Ok(device) => {
                            let name = device.name().map_err(|e| format!("Failed to get device name: {}", e))?;
                            let total_memory = device.total_mem().map_err(|e| format!("Failed to get memory: {}", e))?;
                            let compute_capability = device.compute_capability().map_err(|e| format!("Failed to get compute capability: {}", e))?;

                            let cuda_device = CudaDevice::new(
                                i,
                                name.clone(),
                                total_memory,
                                compute_capability,
                            );

                            devices.push(cuda_device.clone());
                            info!("Detected compatible CUDA device: {}", name);
                        }
                        Err(e) => {
                            warn!("Failed to initialize CUDA device {}: {}", i, e);
                        }
                    }
                }

                if devices.is_empty() {
                    Err("No CUDA devices could be initialized".to_string())
                } else {
                    Ok(devices)
                }
            }
            Ok(_) => {
                warn!("CUDA devices reported but none accessible");
                Err("No accessible CUDA devices".to_string())
            }
            Err(e) => {
                error!("Failed to count CUDA devices: {}", e);
                Err(format!("CUDA device count failed: {}", e))
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
                nvidia_info.device_name.unwrap_or_else(|| "Unknown NVIDIA GPU".to_string()),
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
        if let Some(device) = self.get_device(device_id).await {
            let available_mb = device.total_memory() / (1024 * 1024);
            return std::cmp::min(32, (available_mb / 2048) as usize + 1);
        }
        16
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

    if nvidia_smi_output.status.success() {
        if let Ok(output) = str::from_utf8(&nvidia_smi_output.stdout) {
            let parts: Vec<&str> = output.trim().split(',').map(|s| s.trim()).collect();
            if parts.len() >= 3 {
                info.device_name = Some(parts[0].to_string());

                if let Ok(vram_mb) = parts[1].parse::<u64>() {
                    info.total_vram_bytes = vram_mb * 1024 * 1024;
                }

                let cc_parts: Vec<&str> = parts[2].split('.').collect();
                if cc_parts.len() >= 2 {
                    if let (Ok(major), Ok(minor)) = (
                        cc_parts[0].parse::<u8>(),
                        cc_parts[1].parse::<u8>()
                    ) {
                        info.compute_capability = Some((major, minor));
                    }
                }
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

        let sm_60_device = CudaDevice::new(0, "Pascal GPU".to_string(), 8 * 1024 * 1024 * 1024, (6, 0));
        assert!(sm_60_device.capability.supports_float16);
        assert!(!sm_60_device.capability.supports_tensor_cores);

        let sm_50_device = CudaDevice::new(0, "Maxwell GPU".to_string(), 4 * 1024 * 1024 * 1024, (5, 0));
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
}
