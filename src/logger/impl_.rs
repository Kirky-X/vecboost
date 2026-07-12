// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! LoggerModule — `ModuleMeta` + `AutoBuilder` 实现
//!
//! 提供 `Arc<inklog::LoggerManager>` 能力,通过 `Kit::set_config` 注入预构建的实例。

use std::sync::Arc;

use trait_kit::prelude::*;

/// 日志模块 — 提供 `Arc<inklog::LoggerManager>` 能力
///
/// 采用"预构建能力注入"模式:`main.rs` 中
/// `let manager = inklog::LoggerManager::builder().level("info").console(true).file("logs/vecboost.log").build().await?;`
/// 然后 `kit.set_config(Arc::new(manager));`,本模块 `build()` 通过 `kit.config::<Self::Capability>()` 检索。
pub struct LoggerModule;

impl ModuleMeta for LoggerModule {
    const NAME: &'static str = "inklog_logger";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AutoBuilder for LoggerModule {
    type Capability = Arc<inklog::LoggerManager>;
    type Error = TraitKitError;

    fn build(kit: &Kit) -> Result<Self::Capability, Self::Error> {
        kit.config::<Self::Capability>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_module_compiles() {
        let _ = LoggerModule;
    }
}
