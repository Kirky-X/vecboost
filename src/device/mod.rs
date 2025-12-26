// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod manager;
pub mod memory_limit;
pub mod amd;
pub mod cuda;

pub use manager::{DeviceInfo, DeviceManager, DeviceStatus, GpuInfo};
pub use memory_limit::{MemoryLimitConfig, MemoryLimitController, MemoryLimitStatus, block_on_async, block_on_sync};
pub use amd::{AmdDevice, AmdDeviceManager, AmdGpuInfo};
pub use cuda::{CudaDevice, CudaDeviceManager, CudaGpuInfo};

use crate::config::model::DeviceType as ConfigDeviceType;
use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DeviceCategory {
    Cpu,
    Gpu,
}

impl fmt::Display for DeviceCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceCategory::Cpu => write!(f, "CPU"),
            DeviceCategory::Gpu => write!(f, "GPU"),
        }
    }
}

impl From<&ConfigDeviceType> for DeviceCategory {
    fn from(device: &ConfigDeviceType) -> Self {
        match device {
            ConfigDeviceType::Cpu => DeviceCategory::Cpu,
            ConfigDeviceType::Cuda => DeviceCategory::Gpu,
            ConfigDeviceType::Metal => DeviceCategory::Gpu,
            ConfigDeviceType::Amd => DeviceCategory::Gpu,
            ConfigDeviceType::OpenCL => DeviceCategory::Gpu,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct DeviceCapability {
    pub supports_float16: bool,
    pub supports_tensor_cores: bool,
    pub max_memory_bytes: u64,
    pub compute_capability: Option<(u8, u8)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_category_display() {
        assert_eq!(DeviceCategory::Cpu.to_string(), "CPU");
        assert_eq!(DeviceCategory::Gpu.to_string(), "GPU");
    }

    #[test]
    fn test_device_capability_default() {
        let capability = DeviceCapability::default();
        assert!(!capability.supports_float16);
        assert!(!capability.supports_tensor_cores);
        assert_eq!(capability.max_memory_bytes, 0);
        assert!(capability.compute_capability.is_none());
    }
}
