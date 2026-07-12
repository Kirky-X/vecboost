// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Module Registry — trait-kit 集成模块
//!
//! 通过 trait-kit 管理模块依赖关系和能力构建。
//! 由于 trait-kit 基于 RefCell 且 !Sync，Kit 仅在启动时用于构建依赖图，
//! 构建完成后提取能力填充 AppState，Kit 本身不放入 AppState。
//!
//! 模块采用"预构建能力注入"模式：需要异步构造的复杂对象（如 EmbeddingService）
//! 在 main.rs 中预构建后通过 `kit.set_config()` 注入，模块的 `build()` 从 config 检索。

mod impl_;
#[cfg(test)]
mod tests;

/// 嵌入服务模块 — 提供 `Arc<RwLock<EmbeddingService>>` 能力
pub struct EmbeddingModule;

/// 认证模块 — 提供 `Option<Arc<JwtManager>>` 能力（需要 auth feature）
#[cfg(feature = "auth")]
pub struct AuthModule;

/// 限流模块 — 提供 `Arc<RateLimiter>` 能力
pub struct RateLimitModule;

/// 缓存模块 — 提供缓存启用状态能力
pub struct CacheModule;

/// 数据库模块 — 提供数据库启用状态能力（占位，db feature 未实现）
pub struct DbModule;

/// 日志模块 — 提供 `Option<Arc<AuditLogger>>` 能力
pub struct LoggerModule;

/// 缓存配置（通过 `Kit::set_config` 注入）
#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub enabled: bool,
    pub size: usize,
}

/// 数据库配置（通过 `Kit::set_config` 注入）
#[derive(Clone, Debug, Default)]
pub struct DbConfig {
    pub enabled: bool,
}
