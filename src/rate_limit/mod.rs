// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 限流模块
//!
//! 基于 limiteron `Governor` 编排层提供多维度限流支持。
//! Governor 内部管理标识符提取、规则匹配、L1 缓存、封禁检查和熔断器。

// limiteron Governor 后端（limiteron 必选，完全接管速率限制）
pub mod ip_whitelist;
pub(crate) mod limiteron_adapter;

pub use ip_whitelist::is_ip_whitelisted;
pub use limiteron_adapter::{LimiteronAdapter, RateLimitSettings};
pub use limiteron::matchers::RequestContext;
