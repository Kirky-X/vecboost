//! 测试专用:进程级环境变量的串行访问锁。
//!
//! Rust 测试默认并行,多个测试同时 set/remove 同一环境变量会产生竞态
//! (vecboost 全库测试实测复现)。触碰以下变量的单元测试必须持有
//! [`ENV_LOCK`]:VECBOOST_ENCRYPTION_KEY / VECBOOST_KEY_STORAGE_TYPE /
//! VECBOOST_KEY_FILE_PATH / VECBOOST_ALLOW_INSECURE 等。
//!
//! 仅在 `cfg(test)` 下编译。

#![cfg(test)]

/// 全局环境变量测试锁。持有者可安全操作共享环境变量。
pub static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 便捷获取锁(poisoned 时恢复,测试进程无需 panic 传播)。
pub fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
