// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Logger — inklog 集成模块
//!
//! 通过 trait-kit 管理 inklog `LoggerManager` 的构建与注入。
//! 由于 `LoggerManager::builder().build()` 是 async 操作,而 `Kit` 基于同步 `AutoBuilder`,
//! 本模块采用"预构建能力注入"模式:`main.rs` 中预构建 `LoggerManager` 后,
//! 通过 `kit.set_config(Arc::new(manager))` 注入,模块的 `build()` 从 config 检索。
//!
//! 注意:与 `module_registry::AuditModule`(提供 `Option<Arc<AuditLogger>>` 审计能力)
//! 不同,本模块提供 `Arc<inklog::LoggerManager>` 应用日志能力。

pub(crate) mod impl_;

pub use impl_::LoggerModule;

#[cfg(test)]
mod tests;
