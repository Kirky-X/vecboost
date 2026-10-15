// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use log::{debug, info, warn};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

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
        self.peak_usage
            .store(std::cmp::max(used_bytes, peak), Ordering::SeqCst);

        self.update_status().await;
    }

    async fn update_status(&self) {
        let config = self.config.read().await;
        let current = self.current_usage.load(Ordering::SeqCst);
        let limit = config.limit_bytes;

        let usage_percent = current
            .checked_mul(100)
            .and_then(|v| v.checked_div(limit))
            .unwrap_or(0);

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
                warn!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-device-memory-usage-exceeded-limit",
                        crate::i18n::tr_args(&[
                            ("a1", &(current).to_string()),
                            ("a2", &(limit).to_string())
                        ])
                    )
                );
            }
            MemoryLimitStatus::Critical => {
                warn!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-device-memory-usage-critical-bytes",
                        crate::i18n::tr_args(&[
                            ("a1", &(current).to_string()),
                            ("a2", &(usage_percent).to_string())
                        ])
                    )
                );
            }
            MemoryLimitStatus::Warning => {
                debug!(
                    "Memory usage warning: {} bytes ({}%)",
                    current, usage_percent
                );
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
        if let Ok(config_guard) = self.config.try_read() {
            let limit = config_guard.limit_bytes;
            let used = self.current_usage.load(Ordering::SeqCst);
            limit.saturating_sub(used)
        } else {
            0
        }
    }

    pub fn usage_percent(&self) -> f64 {
        if let Ok(config_guard) = self.config.try_read() {
            let limit = config_guard.limit_bytes;
            let used = self.current_usage.load(Ordering::SeqCst);
            if limit == 0 {
                0.0
            } else {
                (used as f64 / limit as f64) * 100.0
            }
        } else {
            0.0
        }
    }
}

/// Run a synchronous closure within the current tokio runtime context (if any).
///
/// Does **not** block on the runtime — merely enters its guard so that
/// `Handle::current()` and similar APIs work inside `f`.
pub fn block_on_sync<F: FnOnce() -> T, T>(f: F) -> T {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let _guard = handle.enter();
        f()
    } else {
        f()
    }
}

/// Try to run a future to completion.
///
/// - **Inside a tokio runtime** → returns `None` (blocking would deadlock).
///   Callers should `await` the future directly instead.
/// - **Outside any runtime** → creates a temporary single-threaded runtime,
///   blocks on the future, and returns `Some(result)`.
pub fn block_on_async<F: std::future::Future<Output = T>, T>(f: F) -> Option<T> {
    if tokio::runtime::Handle::try_current().is_ok() {
        // Already inside a running runtime — blocking would deadlock.
        None
    } else {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                log::error!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-device-failed-create-tokio-runtime",
                        crate::i18n::tr_args(&[("a1", &(e).to_string())])
                    )
                );
                return None;
            }
        };
        Some(rt.block_on(f))
    }
}

impl MemoryLimitController {
    pub async fn set_limit(&self, limit_bytes: u64) {
        {
            let mut config = self.config.write().await;
            config.limit_bytes = limit_bytes;
            info!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-device-memory-limit-set-bytes",
                    crate::i18n::tr_args(&[("a1", &(limit_bytes).to_string())])
                )
            );
        }
        self.update_status().await;
    }

    pub async fn set_warning_threshold(&self, percent: u64) {
        {
            let mut config = self.config.write().await;
            config.warning_threshold_percent = percent.clamp(50, 99);
        }
        self.update_status().await;
    }

    pub async fn set_critical_threshold(&self, percent: u64) {
        {
            let mut config = self.config.write().await;
            config.critical_threshold_percent = percent.clamp(70, 99);
        }
        self.update_status().await;
    }

    pub async fn should_fallback(&self) -> bool {
        let status = self.status.read().await;
        let fallback = self.fallback_triggered.read().await;
        *status == MemoryLimitStatus::Exceeded && !*fallback
    }

    pub async fn trigger_fallback(&self) {
        let mut fallback = self.fallback_triggered.write().await;
        if !*fallback {
            info!(
                "{}",
                crate::i18n::tr("log-device-memory-limit-exceeded-triggering")
            );
            *fallback = true;
        }
    }

    pub async fn reset(&self) {
        self.current_usage.store(0, Ordering::SeqCst);
        self.peak_usage.store(0, Ordering::SeqCst);
        let mut status = self.status.write().await;
        *status = MemoryLimitStatus::Ok;
        let mut fallback = self.fallback_triggered.write().await;
        *fallback = false;
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

        controller
            .update_usage(6 * 1024 * 1024 * 1024 + 512 * 1024 * 1024)
            .await;
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
        controller.reset().await;

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

    #[test]
    fn test_memory_limit_config_default() {
        let config = MemoryLimitConfig::default();
        assert_eq!(config.limit_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(config.warning_threshold_percent, 80);
        assert_eq!(config.critical_threshold_percent, 90);
    }

    #[tokio::test]
    async fn test_critical_status() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 10 * 1024 * 1024 * 1024,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });

        controller.update_usage(9 * 1024 * 1024 * 1024).await;
        let status = controller.check_limit().await;
        assert_eq!(status, MemoryLimitStatus::Critical);
    }

    #[tokio::test]
    async fn test_usage_percent_zero_limit() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 0,
            ..Default::default()
        });

        controller.update_usage(1024).await;
        assert_eq!(controller.usage_percent(), 0.0);
    }

    #[tokio::test]
    async fn test_available_bytes_saturating_sub() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1024,
            ..Default::default()
        });

        controller.update_usage(2048).await;
        assert_eq!(controller.available_bytes(), 0);
    }

    #[tokio::test]
    async fn test_set_warning_threshold_clamping() {
        let controller = MemoryLimitController::new();

        controller.set_warning_threshold(10).await;
        let config = controller.get_config().await;
        assert_eq!(config.warning_threshold_percent, 50);

        controller.set_warning_threshold(150).await;
        let config = controller.get_config().await;
        assert_eq!(config.warning_threshold_percent, 99);
    }

    #[tokio::test]
    async fn test_set_critical_threshold_clamping() {
        let controller = MemoryLimitController::new();

        controller.set_critical_threshold(10).await;
        let config = controller.get_config().await;
        assert_eq!(config.critical_threshold_percent, 70);

        controller.set_critical_threshold(150).await;
        let config = controller.get_config().await;
        assert_eq!(config.critical_threshold_percent, 99);
    }

    #[tokio::test]
    async fn test_set_warning_threshold_updates_status() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 10 * 1024 * 1024 * 1024,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });

        controller.update_usage(5 * 1024 * 1024 * 1024).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Ok);

        controller.set_warning_threshold(40).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Warning);
    }

    #[tokio::test]
    async fn test_set_critical_threshold_updates_status() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 10 * 1024 * 1024 * 1024,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });

        controller.update_usage(8 * 1024 * 1024 * 1024).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Warning);

        controller.set_critical_threshold(70).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Critical);
    }

    #[tokio::test]
    async fn test_should_fallback_initially_false() {
        let controller = MemoryLimitController::new();
        assert!(!controller.should_fallback().await);
    }

    #[tokio::test]
    async fn test_should_fallback_after_exceeded_and_trigger() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1024,
            ..Default::default()
        });

        controller.update_usage(2048).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Exceeded);
        assert!(controller.should_fallback().await);

        controller.trigger_fallback().await;
        assert!(!controller.should_fallback().await);
    }

    #[tokio::test]
    async fn test_trigger_fallback_idempotent() {
        let controller = MemoryLimitController::new();

        controller.trigger_fallback().await;
        controller.trigger_fallback().await;

        let fallback = controller.should_fallback().await;
        assert!(!fallback);
    }

    #[tokio::test]
    async fn test_reset_clears_fallback_flag() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1024,
            ..Default::default()
        });

        controller.update_usage(2048).await;
        controller.trigger_fallback().await;
        assert!(!controller.should_fallback().await);

        controller.reset().await;
        assert_eq!(controller.current_usage(), 0);
        assert_eq!(controller.peak_usage(), 0);
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Ok);

        controller.update_usage(2048).await;
        assert!(controller.should_fallback().await);
    }

    #[tokio::test]
    async fn test_set_limit_updates_status() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 10 * 1024 * 1024 * 1024,
            ..Default::default()
        });

        controller.update_usage(5 * 1024 * 1024 * 1024).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Ok);

        controller.set_limit(4 * 1024 * 1024 * 1024).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Exceeded);
    }

    #[test]
    fn test_block_on_sync_without_runtime() {
        let result = block_on_sync(|| 42);
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_block_on_sync_with_runtime() {
        let result = block_on_sync(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_async_without_runtime() {
        let result = block_on_async(async { 42 });
        assert_eq!(result, Some(42));
    }

    #[tokio::test]
    async fn test_block_on_async_inside_runtime_returns_none() {
        // Inside a running runtime, block_on_async must return None (not deadlock).
        let result = block_on_async(async { 42 });
        assert!(result.is_none());
    }

    // =========================================================================
    // 补缺口：日志支路（warn!/info! 参数行）、config 写锁下的 try_read 回退、
    // Default impl
    // =========================================================================

    /// 无 logger 安装时 log 宏默认 max_level=Off 会跳过参数求值；
    /// 测试期提升级别让日志支路真实执行（幂等，无输出副作用）。
    fn enable_test_log_level() {
        log::set_max_level(log::LevelFilter::Trace);
    }

    #[tokio::test]
    async fn test_exceeded_status_logs_warning_args() {
        enable_test_log_level();
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1024,
            ..Default::default()
        });
        controller.update_usage(4096).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Exceeded);
    }

    #[tokio::test]
    async fn test_critical_status_logs_warning_args() {
        enable_test_log_level();
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1000,
            warning_threshold_percent: 80,
            critical_threshold_percent: 90,
        });
        // 95% → Critical（且未达 limit）
        controller.update_usage(950).await;
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Critical);
        assert_eq!(controller.current_usage(), 950);
    }

    #[tokio::test]
    async fn test_set_limit_logs_info_args() {
        enable_test_log_level();
        let controller = MemoryLimitController::new();
        controller.set_limit(2048).await;
        assert_eq!(controller.get_config().await.limit_bytes, 2048);
        assert_eq!(controller.check_limit().await, MemoryLimitStatus::Ok);
    }

    #[tokio::test]
    async fn test_trigger_fallback_logs_info_on_first_call() {
        enable_test_log_level();
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1024,
            ..Default::default()
        });
        controller.update_usage(2048).await;
        assert!(controller.should_fallback().await);
        controller.trigger_fallback().await;
        assert!(!controller.should_fallback().await);
    }

    /// config 写锁被持有时，同步读取器（try_read 失败）必须回退安全值而非
    /// 阻塞或 panic。
    #[tokio::test]
    async fn test_sync_readers_return_zero_while_config_write_locked() {
        let controller = MemoryLimitController::with_config(MemoryLimitConfig {
            limit_bytes: 1000,
            ..Default::default()
        });
        controller.update_usage(500).await;
        {
            let _write_guard = controller.config.write().await;
            assert_eq!(controller.available_bytes(), 0);
            assert_eq!(controller.usage_percent(), 0.0);
        }
        // 锁释放后恢复真实读数
        assert_eq!(controller.available_bytes(), 500);
        assert!((controller.usage_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_default_impl_matches_new() {
        let controller = MemoryLimitController::default();
        assert_eq!(controller.current_usage(), 0);
        assert_eq!(controller.peak_usage(), 0);
    }
}
