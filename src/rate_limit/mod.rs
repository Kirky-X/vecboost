// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 限流模块
//!
//! 基于 limiteron 提供多维度限流支持,支持全局、IP、用户、API Key 等维度。
//! 内部使用 limiteron 原生 `TokenBucketLimiter`,每个维度维护独立的令牌桶。

// limiteron 后端(limiteron 必选,完全接管速率限制)
pub mod ip_whitelist;
pub(crate) mod limiteron_adapter;

pub use ip_whitelist::is_ip_whitelisted;
pub use limiteron_adapter::{Dimension, LimiteronAdapter, RateLimitSettings};
