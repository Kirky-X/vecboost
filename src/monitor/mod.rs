// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub available_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub device_id: usize,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone)]
pub enum DeviceType {
    Cpu,
    Gpu {
        device_id: usize,
        device_name: String,
    },
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

#[derive(Debug, Clone)]
pub struct MemoryMonitor {
    inner: Arc<MemoryMonitorInner>,
}

#[derive(Debug)]
struct MemoryMonitorInner {
    cpu_stats: RwLock<MemoryStats>,
    gpu_stats: RwLock<Option<GpuMemoryStats>>,
    peak_memory: AtomicU64,
    last_update: RwLock<std::time::Instant>,
    check_interval_ms: u64,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MemoryMonitorInner {
                cpu_stats: RwLock::new(MemoryStats::default()),
                gpu_stats: RwLock::new(None),
                peak_memory: AtomicU64::new(0),
                last_update: RwLock::new(std::time::Instant::now()),
                check_interval_ms: 1000,
            }),
        }
    }

    pub fn with_interval(interval_ms: u64) -> Self {
        Self {
            inner: Arc::new(MemoryMonitorInner {
                cpu_stats: RwLock::new(MemoryStats::default()),
                gpu_stats: RwLock::new(None),
                peak_memory: AtomicU64::new(0),
                last_update: RwLock::new(std::time::Instant::now()),
                check_interval_ms: interval_ms,
            }),
        }
    }

    pub async fn refresh(&self) -> MemoryStats {
        let mut stats = MemoryStats::default();

        match sys_info::mem_info() {
            Ok(mem) => {
                stats.total_bytes = mem.total * 1024;
                stats.available_bytes = mem.avail * 1024;
                stats.current_bytes = (mem.total - mem.avail) * 1024;

                let current = stats.current_bytes;
                let peak = self.inner.peak_memory.fetch_max(current, Ordering::SeqCst);
                stats.peak_bytes = std::cmp::max(current, peak);
            }
            Err(e) => {
                warn!("Failed to get memory info: {}", e);
            }
        }

        {
            let mut cpu_stats = self.inner.cpu_stats.write().await;
            *cpu_stats = stats.clone();
        }

        let mut last_update = self.inner.last_update.write().await;
        *last_update = std::time::Instant::now();

        stats
    }

    pub async fn get_memory_stats(&self) -> MemoryStats {
        let cpu_stats = self.inner.cpu_stats.read().await.clone();
        cpu_stats
    }

    pub async fn get_gpu_stats(&self) -> Option<GpuMemoryStats> {
        self.inner.gpu_stats.read().await.clone()
    }

    pub fn get_peak_memory(&self) -> u64 {
        self.inner.peak_memory.load(Ordering::SeqCst)
    }

    pub fn reset_peak(&self) {
        self.inner.peak_memory.store(0, Ordering::SeqCst);
    }

    pub async fn check_oom_risk(&self, threshold_percent: u64) -> bool {
        let stats = self.get_memory_stats().await;
        let usage_percent = if stats.total_bytes > 0 {
            (stats.current_bytes * 100) / stats.total_bytes
        } else {
            0
        };
        usage_percent >= threshold_percent
    }

    pub async fn is_memory_low(&self, threshold_mb: u64) -> bool {
        let stats = self.get_memory_stats().await;
        stats.available_bytes < threshold_mb * 1024 * 1024
    }

    pub async fn update_gpu_memory(&self, used: u64, total: u64) {
        let utilization = if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let stats = GpuMemoryStats {
            device_id: 0,
            used_bytes: used,
            total_bytes: total,
            available_bytes: total.saturating_sub(used),
            utilization_percent: utilization,
        };

        let mut gpu_stats = self.inner.gpu_stats.write().await;
        *gpu_stats = Some(stats);
    }

    pub fn check_interval(&self) -> u64 {
        self.inner.check_interval_ms
    }

    pub async fn should_refresh(&self) -> bool {
        let last_update = self.inner.last_update.read().await;
        last_update.elapsed().as_millis() >= self.inner.check_interval_ms as u128
    }

    #[cfg(feature = "cuda")]
    pub async fn update_gpu_memory_from_candle(&self) {
        if let Ok((used, total)) = candle_core::Device::cuda_memory_info() {
            self.update_gpu_memory(used as u64, total as u64).await;
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub async fn update_gpu_memory_from_candle(&self) {}

    #[cfg(feature = "onnx")]
    pub async fn update_gpu_memory_from_ort(&self) {
        use ort::ExecutionProviderDispatch;

        if let Ok(ep) = ort::ExecutionProviderDispatch::CUDA {
            if ep.is_available() {
                let memory_info = ep.get_memory_info();
                if let Ok(info) = memory_info {
                    self.update_gpu_memory(
                        info.used_memory() as u64,
                        info.total_memory() as u64,
                    )
                    .await;
                }
            }
        }
    }

    #[cfg(not(feature = "onnx"))]
    pub async fn update_gpu_memory_from_ort(&self) {}
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_monitor_creation() {
        let monitor = MemoryMonitor::new();
        assert!(monitor.check_interval() > 0);
    }

    #[tokio::test]
    async fn test_memory_refresh() {
        let monitor = MemoryMonitor::new();
        let stats = monitor.refresh().await;

        assert!(stats.total_bytes > 0, "Total memory should be > 0");
        assert!(stats.available_bytes > 0, "Available memory should be > 0");
    }

    #[tokio::test]
    async fn test_get_memory_stats() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;
        let stats = monitor.get_memory_stats().await;

        assert!(stats.total_bytes > 0);
    }

    #[tokio::test]
    async fn test_peak_memory_tracking() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;

        let peak = monitor.get_peak_memory();
        assert!(peak > 0, "Peak memory should be tracked");
    }

    #[tokio::test]
    async fn test_oom_risk_check() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;

        let is_risk = monitor.check_oom_risk(99).await;
        assert!(!is_risk, "Should not be OOM risk at 99% threshold");
    }

    #[tokio::test]
    async fn test_memory_low_check() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;

        let is_low = monitor.is_memory_low(1).await;
        assert!(!is_low, "Memory should not be low with 1MB threshold");
    }

    #[tokio::test]
    async fn test_gpu_memory_update() {
        let monitor = MemoryMonitor::new();

        monitor.update_gpu_memory(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024).await;

        let stats = monitor.get_gpu_stats().await;
        assert!(stats.is_some());

        let gpu = stats.unwrap();
        assert_eq!(gpu.used_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(gpu.total_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(gpu.utilization_percent, 50.0);
    }

    #[tokio::test]
    async fn test_reset_peak() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;

        assert!(monitor.get_peak_memory() > 0);

        monitor.reset_peak();
        assert_eq!(monitor.get_peak_memory(), 0);
    }
}
