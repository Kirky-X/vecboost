// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use tracing::warn;

#[derive(Debug, Clone, Default)]
pub(crate) struct MemoryStats {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub available_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct GpuMemoryStats {
    pub device_id: usize,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) enum DeviceType {
    #[default]
    Cpu,
    Gpu {
        device_id: usize,
        device_name: String,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryMonitor {
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
        self.inner.cpu_stats.read().await.clone()
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
        let usage_percent = stats
            .current_bytes
            .checked_mul(100)
            .and_then(|v| v.checked_div(stats.total_bytes))
            .unwrap_or(0);
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
        // candle-core 0.9.2 doesn't expose memory_info directly
        // GPU memory tracking via candle requires runtime-specific APIs
    }

    #[cfg(not(feature = "cuda"))]
    pub async fn update_gpu_memory_from_candle(&self) {}

    #[cfg(feature = "metal")]
    pub async fn update_gpu_memory_from_metal(&self) {
        // Metal GPU memory tracking via candle requires runtime-specific APIs
    }

    #[cfg(not(feature = "metal"))]
    pub async fn update_gpu_memory_from_metal(&self) {}

    #[cfg(feature = "onnx")]
    pub async fn update_gpu_memory_from_ort(&self) {
        tracing::debug!("GPU memory update from ONNX Runtime not yet implemented");
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

        monitor
            .update_gpu_memory(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            .await;

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

    #[tokio::test]
    async fn test_with_interval_sets_custom_interval() {
        let monitor = MemoryMonitor::with_interval(500);
        assert_eq!(monitor.check_interval(), 500);
    }

    #[tokio::test]
    async fn test_new_default_interval_is_1000ms() {
        let monitor = MemoryMonitor::new();
        assert_eq!(monitor.check_interval(), 1000);
    }

    #[tokio::test]
    async fn test_default_uses_new() {
        let monitor = MemoryMonitor::default();
        assert_eq!(monitor.check_interval(), 1000);
    }

    #[tokio::test]
    async fn test_get_peak_memory_starts_at_zero() {
        let monitor = MemoryMonitor::new();
        assert_eq!(monitor.get_peak_memory(), 0);
    }

    #[tokio::test]
    async fn test_get_gpu_stats_returns_none_initially() {
        let monitor = MemoryMonitor::new();
        assert!(monitor.get_gpu_stats().await.is_none());
    }

    #[tokio::test]
    async fn test_update_gpu_memory_with_zero_total() {
        let monitor = MemoryMonitor::new();
        monitor.update_gpu_memory(100, 0).await;

        let stats = monitor.get_gpu_stats().await;
        assert!(stats.is_some());
        let gpu = stats.unwrap();
        assert_eq!(gpu.total_bytes, 0);
        assert_eq!(gpu.utilization_percent, 0.0);
    }

    #[tokio::test]
    async fn test_update_gpu_memory_full_usage() {
        let monitor = MemoryMonitor::new();
        let total = 4 * 1024 * 1024 * 1024;
        monitor.update_gpu_memory(total, total).await;

        let gpu = monitor.get_gpu_stats().await.unwrap();
        assert_eq!(gpu.used_bytes, total);
        assert_eq!(gpu.total_bytes, total);
        assert_eq!(gpu.utilization_percent, 100.0);
        assert_eq!(gpu.available_bytes, 0);
    }

    #[tokio::test]
    async fn test_update_gpu_memory_overwrites_previous() {
        let monitor = MemoryMonitor::new();
        monitor
            .update_gpu_memory(1 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
            .await;
        monitor
            .update_gpu_memory(2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            .await;

        let gpu = monitor.get_gpu_stats().await.unwrap();
        assert_eq!(gpu.used_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(gpu.total_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(gpu.utilization_percent, 25.0);
    }

    #[tokio::test]
    async fn test_should_refresh_returns_false_immediately_after_creation() {
        let monitor = MemoryMonitor::with_interval(1000);
        assert!(
            !monitor.should_refresh().await,
            "should not need refresh immediately after creation"
        );
    }

    #[tokio::test]
    async fn test_should_refresh_returns_true_after_interval_elapsed() {
        let monitor = MemoryMonitor::with_interval(1);
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(
            monitor.should_refresh().await,
            "should need refresh after interval elapsed"
        );
    }

    #[tokio::test]
    async fn test_should_refresh_resets_after_refresh_call() {
        let monitor = MemoryMonitor::with_interval(1000);
        let _ = monitor.refresh().await;
        assert!(
            !monitor.should_refresh().await,
            "should not need refresh immediately after refresh() call"
        );
    }

    #[tokio::test]
    async fn test_check_oom_risk_with_zero_threshold_returns_true() {
        let monitor = MemoryMonitor::new();
        let is_risk = monitor.check_oom_risk(0).await;
        assert!(is_risk, "0% threshold should always be OOM risk");
    }

    #[tokio::test]
    async fn test_is_memory_low_with_zero_threshold_returns_false() {
        let monitor = MemoryMonitor::new();
        let is_low = monitor.is_memory_low(0).await;
        assert!(!is_low, "0MB threshold should never be low");
    }

    #[tokio::test]
    async fn test_is_memory_low_with_huge_threshold_returns_true_after_refresh() {
        let monitor = MemoryMonitor::new();
        let _ = monitor.refresh().await;
        let is_low = monitor.is_memory_low(1_000_000).await;
        assert!(
            is_low,
            "1TB threshold should be considered low on any real machine"
        );
    }

    #[tokio::test]
    async fn test_update_gpu_memory_from_candle_is_safe() {
        let monitor = MemoryMonitor::new();
        monitor.update_gpu_memory_from_candle().await;
    }

    #[tokio::test]
    async fn test_update_gpu_memory_from_metal_is_safe() {
        let monitor = MemoryMonitor::new();
        monitor.update_gpu_memory_from_metal().await;
    }

    #[tokio::test]
    async fn test_update_gpu_memory_from_ort_is_safe() {
        let monitor = MemoryMonitor::new();
        monitor.update_gpu_memory_from_ort().await;
    }

    #[test]
    fn test_memory_stats_default_is_all_zero() {
        let stats = MemoryStats::default();
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert_eq!(stats.available_bytes, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    #[test]
    fn test_device_type_default_is_cpu() {
        let dt = DeviceType::default();
        assert!(matches!(dt, DeviceType::Cpu));
    }
}
