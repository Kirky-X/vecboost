// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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

/// 异步测试用:tokio Mutex 版本,可在跨 .await 持有时避免
/// clippy::await_holding_lock 告警。
pub static ENV_LOCK_ASYNC: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// 异步测试便捷获取(tokio guard 可跨 .await)。
pub async fn env_lock_async() -> tokio::sync::MutexGuard<'static, ()> {
    ENV_LOCK_ASYNC.lock().await
}

/// macOS 测试专用:主动归还分配器滞留的自由页。
///
/// 真机 ONNX 会话用例(单会话 ~1GB)释放后,Darwin malloc 以 MADV_FREE
/// 方式惰性回收,进程 RSS 跨测试单调累积(全量 lib 套件实测峰值 7.5GB,
/// 与 rustc 增量编译并发时被内存管理 SIGKILL)。重会话测试结束调用一次,
/// 把进程峰值压回单个会话量级。仅 `cfg(all(test, target_os = "macos"))`
/// 编译;Linux 生产面由 jemalloc decay 防线覆盖,无需此助手。
#[cfg(all(test, target_os = "macos"))]
pub fn release_free_pages() {
    use std::os::raw::{c_int, c_void};

    unsafe extern "C" {
        fn malloc_default_zone() -> *mut c_void;
        // all=1:对 zone 内全部自由页执行 MADV_FREE_REUSABLE 归还
        fn malloc_zone_pressure_relief(zone: *mut c_void, all: c_int) -> usize;
    }
    #[allow(unsafe_code, reason = "测试进程内对自己 malloc zone 的只读压力回收")]
    unsafe {
        malloc_zone_pressure_relief(malloc_default_zone(), 1);
    }
}

/// 重内存会话测试的串行锁。
///
/// 真机 ORT 会话用例（翻译 encoder+decoder、嵌入真 bundle）单个进程峰值
/// 实测 1.5-5.2GB（m2m100 418M 每套 worker 集 ~1.1GB）。测试默认并行跑时
/// 多个重会话同时驻留 + rustc 增量编译抢内存，在 16GB 机器上被内存管理
/// SIGKILL（signal 9，cargo 侧只见 "process didn't exit successfully"）。
/// 持本锁让重会话用例串行进出，进程峰值收敛为单用例量级。
///
/// 只约束「构造真引擎」的用例；纯函数/离线用例不受影响，并行度损失可忽略。
pub static HEAVY_SESSION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 取重会话锁(中毒也继续，与 [`env_lock`] 同口径)。
pub fn heavy_session_lock() -> std::sync::MutexGuard<'static, ()> {
    HEAVY_SESSION_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
