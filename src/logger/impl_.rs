// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! LoggerModule — `ModuleMeta` + `AsyncAutoBuilder` 实现
//!
//! 提供 `Arc<inklog::LoggerManager>` 能力,通过 `AsyncKit::set_config` 注入预构建的实例。

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use trait_kit::AsyncKit;
use trait_kit::prelude::*;

/// 日志模块 — 提供 `Arc<inklog::LoggerManager>` 能力
///
/// 采用"预构建能力注入"模式:`main.rs` 中预构建 `LoggerManager` 后,
/// 通过 `kit.set_config(Arc::new(manager))` 注入,模块的 `build()` 从 config 检索。
pub struct LoggerModule;

impl ModuleMeta for LoggerModule {
    const NAME: &'static str = "inklog_logger";

    fn dependencies() -> &'static [(&'static str, std::any::TypeId)] {
        &[]
    }
}

impl AsyncAutoBuilder for LoggerModule {
    type Capability = Arc<inklog::LoggerManager>;
    type Error = TraitKitError;

    fn build<'a>(
        kit: &'a AsyncKit,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Capability, Self::Error>> + Send + 'a>> {
        Box::pin(async move { kit.config::<Self::Capability>() })
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
