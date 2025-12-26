// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum MemoryLimitStatus {
    Ok,
    Warning,
    Critical,
    Exceeded,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryLimitConfig {
    pub limit_bytes: u64,
    pub warning_threshold_percent: u64,
    pub critical_threshold_percent: u64,
}

impl Default for MemoryLimitConfig {
    fn default() -> Self {
        Self {
            limit_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        }
    }
}

#[derive(Debug)]
pub struct MemoryLimitController {
    config: Arc<RwLock<MemoryLimitConfig>>,
    current_usage: AtomicU64,
    peak_usage: AtomicU64,
    status: Arc<RwLock<MemoryLimitStatus>>,
    fallback_triggered: Arc<RwLock<bool>>,
}

impl MemoryLimitController {
    pub fn new() -> Self {
        Self::with_config(MemoryLimitConfig::default())
    }

    pub fn with_config(config: MemoryLimitConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            status: Arc::new(RwLock::new(MemoryLimitStatus::Ok)),
            fallback_triggered: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn update_usage(&self, used_bytes: u64) {
        self.current_usage.store(used_bytes, Ordering::SeqCst);

        let peak = self.peak_usage.fetch_max(used_bytes, Ordering::SeqCst);
        self.peak_usage.store(std::cmp::max(used_bytes, peak), Ordering::SeqCst);

        self.update_status().await;
    }

    async fn update_status(&self) {
        let config = self.config.read().await;
        let current = self.current_usage.load(Ordering::SeqCst);
        let limit = config.limit_bytes;

        let usage_percent = if limit > 0 {
            (current * 100) / limit
        } else {
            0
        };

        let new_status = if current >= limit {
            MemoryLimitStatus::Exceeded
        } else if usage_percent >= config.critical_threshold_percent {
            MemoryLimitStatus::Critical
        } else if usage_percent >= config.warning_threshold_percent {
            MemoryLimitStatus::Warning
        } else {
            MemoryLimitStatus::Ok
        };

        let mut status = self.status.write().await;
        *status = new_status.clone();

        match new_status {
            MemoryLimitStatus::Exceeded => {
                warn!("Memory usage exceeded limit: {} bytes (limit: {} bytes)", current, limit);
            }
            MemoryLimitStatus::Critical => {
                warn!("Memory usage critical: {} bytes ({}%)", current, usage_percent);
            }
            MemoryLimitStatus::Warning => {
                debug!("Memory usage warning: {} bytes ({}%)", current, usage_percent);
            }
            MemoryLimitStatus::Ok => {
                debug!("Memory usage OK: {} bytes ({}%)", current, usage_percent);
            }
        }
    }

    pub async fn check_limit(&self) -> MemoryLimitStatus {
        self.status.read().await.clone()
    }

    pub fn current_usage(&self) -> u64 {
        self.current_usage.load(Ordering::SeqCst)
    }

    pub fn peak_usage(&self) -> u64 {
        self.peak_usage.load(Ordering::SeqCst)
    }

    pub fn available_bytes(&self) -> u64 {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = self.config.read().await;
            let current = self.current_usage.load(Ordering::SeqCst);
            let limit = config.limit_bytes;
            limit.saturating_sub(current)
        })
    }

    pub fn usage_percent(&self) -> f64 {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = self.config.read().await;
            let current = self.current_usage.load(Ordering::SeqCst);
            let limit = config.limit_bytes;

            if limit == 0 {
                0.0
            } else {
                (current as f64 / limit as f64) * 100.0
            }
        })
    }

    pub async fn set_limit(&self, limit_bytes: u64) {
        let mut config = self.config.write().await;
        config.limit_bytes = limit_bytes;
        info!("Memory limit set to: {} bytes", limit_bytes);
        self.update_status().await;
    }

    pub async fn set_warning_threshold(&self, percent: u64) {
        let mut config = self.config.write().await;
        config.warning_threshold_percent = percent.clamp(50, 99);
        self.update_status().await;
    }

    pub async fn set_critical_threshold(&self, percent: u64) {
        let mut config = self.config.write().await;
        config.critical_threshold_percent = percent.clamp(70, 99);
        self.update_status().await;
    }

    pub fn should_fallback(&self) -> bool {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let status = self.status.read().await.clone();
            let fallback = self.fallback_triggered.read().await.clone();
            status == MemoryLimitStatus::Exceeded && !fallback
        })
    }

    pub async fn trigger_fallback(&self) {
        let mut fallback = self.fallback_triggered.write().await;
        if !*fallback {
            info!("Memory limit exceeded, triggering fallback to CPU");
            *fallback = true;
        }
    }

    pub fn reset(&self) {
        self.current_usage.store(0, Ordering::SeqCst);
        self.peak_usage.store(0, Ordering::SeqCst);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut status = self.status.write().await;
            *status = MemoryLimitStatus::Ok;
            let mut fallback = self.fallback_triggered.write().await;
            *fallback = false;
        });
    }

    pub async fn get_config(&self) -> MemoryLimitConfig {
        self.config.read().await.clone()
    }
}

impl Default for MemoryLimitController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_limit_controller_creation() {
        let controller = MemoryLimitController::new();
        let status = controller.check_limit().await;
        assert_eq!(status, MemoryLimitStatus::Ok);
    }

    #[tokio::test]
    async fn test_update_usage() {
        let controller = MemoryLimitController::new();

        controller.update_usage(4 * 1024 * 1024 * 1024).await;

        assert_eq!(controller.current_usage(), 4 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_peak_usage_tracking() {
        let controller = MemoryLimitController::new();

        controller.update_usage(4 * 1024 * 1024 * 1024).await;
        controller.update_usage(6 * 1024 * 1024 * 1024).await;

        assert_eq!(controller.peak_usage(), 6 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_limit_check() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 8 * 1024 * 1024 * 1024,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });

        controller.update_usage(6 * 1024 * 1024 * 1024).await;
        let status = controller.check_limit().await;

        assert_eq!(status, MemoryLimitStatus::Warning);
    }

    #[tokio::test]
    async fn test_limit_exceeded() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 8 * 1024 * 1024 * 1024,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });

        controller.update_usage(9 * 1024 * 1024 * 1024).await;
        let status = controller.check_limit().await;

        assert_eq!(status, MemoryLimitStatus::Exceeded);
    }

    #[tokio::test]
    async fn test_available_bytes() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 8 * 1024 * 1024 * 1024,
            ..Default::default()
        });

        controller.update_usage(4 * 1024 * 1024 * 1024).await;

        assert_eq!(controller.available_bytes(), 4 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_set_limit() {
        let controller = MemoryLimitController::new();

        controller.set_limit(16 * 1024 * 1024 * 1024).await;
        let config = controller.get_config().await;

        assert_eq!(config.limit_bytes, 16 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_reset() {
        let controller = MemoryLimitController::new();

        controller.update_usage(4 * 1024 * 1024 * 1024).await;
        controller.reset();

        assert_eq!(controller.current_usage(), 0);
        assert_eq!(controller.peak_usage(), 0);
    }

    #[tokio::test]
    async fn test_usage_percent() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 8 * 1024 * 1024 * 1024,
            ..Default::default()
        });

        controller.update_usage(4 * 1024 * 1024 * 1024).await;

        assert!((controller.usage_percent() - 50.0).abs() < 0.1);
    }
}
