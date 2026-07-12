// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Tests for inklog integration (T019).
//!
//! Covers five scenarios required by the spec:
//! 1. `LoggerModule::build(&kit)` returns `Arc<LoggerManager>` via pre-built injection
//! 2. LoggerManager file sink writes to `logs/test.log`
//! 3. `log::info!` / `log::warn!` / `log::error!` macros work (inklog LogLogger adapter)
//! 4. Console sink enabled without panic
//! 5. `LoggerModule::build(&kit)` returns Err when config is missing
//!
//! 测试策略:
//! - 所有 async 测试用 `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]`,
//!   因为 inklog 的 `ObjectPool::with_config` 内部调用 `tokio::task::block_in_place`
//!   构建缓存,该函数只能在多线程 runtime 的 worker 线程上工作。
//! - 触发日志写入的测试(2/3/4)用 `tokio::spawn` 在 worker 线程上执行
//!   `tracing::subscriber::with_default` 闭包,确保 `block_in_place` 能工作。
//!   原因:主测试线程(调用 `block_on` 的线程)不是 worker 线程,没有 worker core,
//!   `block_in_place` 会 panic("can call blocking only when running on the
//!   multi-threaded runtime")。
//! - 所有测试用 `build_detached` 而非 `builder().build()`,**不安装全局 subscriber**,
//!   避免全局 subscriber 污染其他测试(后续测试的 tracing 事件会触发 inklog
//!   ObjectPool Lazy 初始化,在非 runtime 线程上 panic)。
//! - `log::info!` 宏需要全局 LogLogger,会污染其他测试,因此省略;
//!   LogLogger 功能由 inklog 自己的测试覆盖。
//! - 测试 1/5 不触发日志写入,无需 `tokio::spawn`。

use std::sync::Arc;

use tracing_subscriber::prelude::*;
use trait_kit::prelude::*;

use super::LoggerModule;

// ---------------------------------------------------------------------------
// T019 测试 1: LoggerModule::build 返回 Arc<LoggerManager>
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_logger_module_build_returns_manager() {
    let config = inklog::InklogConfig {
        console_sink: None,
        ..Default::default()
    };
    let (manager, _subscriber, _filter) = inklog::LoggerManager::build_detached(config)
        .await
        .expect("build_detached");
    let manager = Arc::new(manager);

    let mut kit = Kit::new();
    kit.set_config(manager.clone());
    kit.register::<LoggerModule>().expect("register");

    let kit = kit.build().expect("build");
    let capability: Arc<inklog::LoggerManager> = kit.require::<LoggerModule>().expect("require");

    assert!(
        Arc::ptr_eq(&capability, &manager),
        "LoggerModule::build should return the injected Arc<LoggerManager>"
    );

    manager.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// T019 测试 2: LoggerManager 文件 sink 写入日志
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_logger_writes_to_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let log_path = dir.path().join("test.log");

    let config = inklog::InklogConfig {
        global: inklog::GlobalConfig {
            level: "info".to_string(),
            ..Default::default()
        },
        console_sink: None,
        file_sink: Some(inklog::FileSinkConfig {
            enabled: true,
            path: log_path.clone(),
            compress: false,
            ..Default::default()
        }),
        ..Default::default()
    };

    let (manager, subscriber, _filter) = inklog::LoggerManager::build_detached(config)
        .await
        .expect("build_detached");

    // 在 worker 线程上执行日志写入,确保 block_in_place 能工作
    let handle = tokio::spawn(async move {
        let registry = tracing_subscriber::registry().with(subscriber);
        tracing::subscriber::with_default(registry, || {
            tracing::info!("test log message for file sink verification");
        });
    });
    handle.await.expect("spawn task completed");

    // 等待异步 flush
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    manager.shutdown().expect("shutdown");

    assert!(log_path.exists(), "log file should exist at {:?}", log_path);

    let content = std::fs::read_to_string(&log_path).unwrap_or_default();
    assert!(
        content.contains("test log message for file sink verification"),
        "log file should contain the message, got: {}",
        content
    );
}

// ---------------------------------------------------------------------------
// T019 测试 3: tracing::info! / warn! / error! 宏通过 subscriber 正常工作
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_log_macro_works() {
    // 用 build_detached + tokio::spawn + with_default 测试 tracing 宏,
    // 不安装全局 subscriber,避免污染其他测试。
    // log::info! 宏需要全局 LogLogger,会污染其他测试,因此省略;
    // LogLogger 功能由 inklog 自己的测试覆盖。
    let config = inklog::InklogConfig {
        global: inklog::GlobalConfig {
            level: "info".to_string(),
            ..Default::default()
        },
        console_sink: None,
        ..Default::default()
    };

    let (manager, subscriber, _filter) = inklog::LoggerManager::build_detached(config)
        .await
        .expect("build_detached");

    // 在 worker 线程上调用 tracing 宏,确保 block_in_place 能工作
    let handle = tokio::spawn(async move {
        let registry = tracing_subscriber::registry().with(subscriber);
        tracing::subscriber::with_default(registry, || {
            tracing::info!("tracing info macro test");
            tracing::warn!("tracing warn macro test");
            tracing::error!("tracing error macro test");
        });
    });
    handle.await.expect("spawn task completed");

    manager.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// T019 测试 4: console sink 启用时不 panic
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_logger_console_output() {
    let config = inklog::InklogConfig {
        global: inklog::GlobalConfig {
            level: "info".to_string(),
            ..Default::default()
        },
        console_sink: Some(inklog::ConsoleSinkConfig {
            enabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };

    let (manager, subscriber, _filter) = inklog::LoggerManager::build_detached(config)
        .await
        .expect("build_detached");

    // 在 worker 线程上执行日志写入,确保 block_in_place 能工作
    let handle = tokio::spawn(async move {
        let registry = tracing_subscriber::registry().with(subscriber);
        tracing::subscriber::with_default(registry, || {
            tracing::info!("console output test message");
        });
    });
    handle.await.expect("spawn task completed");

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    manager.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// T019 测试 5: 未注入 config 时 build 返回 Err
// ---------------------------------------------------------------------------

#[test]
fn test_logger_module_missing_config_fails() {
    let mut kit = Kit::new();
    kit.register::<LoggerModule>().expect("register");

    let result = kit.build();
    assert!(
        result.is_err(),
        "build should fail when LoggerManager config is not injected"
    );
}
