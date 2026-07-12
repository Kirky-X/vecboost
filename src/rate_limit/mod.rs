// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 限流模块
//!
//! 提供多维度限流功能，支持全局、IP、用户、API Key 等维度的限流
//! 支持滑动窗口和令牌桶两种限流算法

pub(crate) mod limiter;

#[cfg(feature = "redis")]
pub(crate) mod redis_store;

pub(crate) mod store;
pub(crate) mod token_bucket;

// limiteron 后端(limiteron feature 启用时可用)
#[cfg(feature = "limiteron")]
pub(crate) mod limiteron_adapter;

pub use limiter::{RateLimitConfig, RateLimitDimension, RateLimitStatus, RateLimiter};

pub(crate) use limiter::RateLimitAlgorithm;

#[cfg(feature = "limiteron")]
pub(crate) use limiteron_adapter::LimiteronAdapter;

#[cfg(feature = "redis")]
pub(crate) use redis_store::{RedisConfig, RedisRateLimitStore};

// MemoryRateLimitStore 和 RateLimitStore 需要被二进制 crate (main.rs) 使用，因此保持 pub
pub use store::{MemoryRateLimitStore, RateLimitStore};
pub(crate) use token_bucket::{TokenBucket, TokenBucketConfig, TokenBucketStore};
