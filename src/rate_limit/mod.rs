// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 限流模块
//!
//! 提供多维度限流功能，支持全局、IP、用户、API Key 等维度的限流
//! 支持滑动窗口和令牌桶两种限流算法

pub mod limiter;
pub mod redis_store;
pub mod store;
pub mod token_bucket;

pub use limiter::{
    RateLimitAlgorithm, RateLimitConfig, RateLimitDimension, RateLimitStatus, RateLimiter,
};
pub use redis_store::{RedisConfig, RedisRateLimitStore};
pub use store::{MemoryRateLimitStore, RateLimitStore};
pub use token_bucket::{TokenBucket, TokenBucketConfig, TokenBucketStore};
