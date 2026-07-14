// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

pub(crate) mod amd;
pub(crate) mod batch_scheduler;
pub(crate) mod cuda;
pub(crate) mod manager;
pub(crate) mod memory_limit;
pub(crate) mod memory_optimizer;
pub(crate) mod memory_pool;

// 重新导出必要的类型供内部使用
pub(crate) use batch_scheduler::{
    BatchConfig, BatchPerformanceStats, BatchPriority, BatchRequest, DynamicBatchScheduler,
};
pub(crate) use manager::{DeviceInfo, DeviceStatus};
pub(crate) use memory_pool::{
    BufferPool, CudaMemoryPool, MemoryPoolConfig, MemoryPoolManager, ModelWeightPool, TensorPool,
};

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

    #[test]
    fn test_device_category_from_config_cpu() {
        let category = DeviceCategory::from(&ConfigDeviceType::Cpu);
        assert_eq!(category, DeviceCategory::Cpu);
    }

    #[test]
    fn test_device_category_from_config_cuda() {
        let category = DeviceCategory::from(&ConfigDeviceType::Cuda);
        assert_eq!(category, DeviceCategory::Gpu);
    }

    #[test]
    fn test_device_category_from_config_metal() {
        let category = DeviceCategory::from(&ConfigDeviceType::Metal);
        assert_eq!(category, DeviceCategory::Gpu);
    }

    #[test]
    fn test_device_category_from_config_amd() {
        let category = DeviceCategory::from(&ConfigDeviceType::Amd);
        assert_eq!(category, DeviceCategory::Gpu);
    }

    #[test]
    fn test_device_category_from_config_opencl() {
        let category = DeviceCategory::from(&ConfigDeviceType::OpenCL);
        assert_eq!(category, DeviceCategory::Gpu);
    }

    #[test]
    fn test_device_capability_with_values() {
        let capability = DeviceCapability {
            supports_float16: true,
            supports_tensor_cores: true,
            max_memory_bytes: 8 * 1024 * 1024 * 1024,
            compute_capability: Some((7, 5)),
        };
        assert!(capability.supports_float16);
        assert!(capability.supports_tensor_cores);
        assert_eq!(capability.max_memory_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(capability.compute_capability, Some((7, 5)));
    }
}
