// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 限流模块
//!
//! 基于 limiteron 提供多维度限流功能,支持全局、IP、用户、API Key 等维度。
//! 内部使用令牌桶算法,每个维度维护独立的令牌桶。

// limiteron 后端(limiteron feature 启用时可用)
#[cfg(feature = "limiteron")]
pub(crate) mod limiteron_adapter;

#[cfg(feature = "limiteron")]
pub use limiteron_adapter::LimiteronAdapter;

// limiteron feature 关闭时的 no-op stub,保持 VecboostState/路由 API 一致
#[cfg(not(feature = "limiteron"))]
mod limiteron_stub;

#[cfg(not(feature = "limiteron"))]
pub use limiteron_stub::LimiteronAdapter;

// === 限流类型定义 ===

/// 限流算法类型
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RateLimitAlgorithm {
    /// 滑动窗口算法（固定窗口改进版）
    SlidingWindow,
    /// 令牌桶算法（允许突发流量）
    #[default]
    TokenBucket,
}

/// 限流维度
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RateLimitDimension {
    /// 全局限流
    Global,
    /// IP 限流
    Ip(String),
    /// 用户限流
    User(String),
    /// API Key 限流
    ApiKey(String),
}

/// 限流配置
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// 限流算法
    pub algorithm: RateLimitAlgorithm,
    /// 全局请求限制（每分钟）
    pub global_requests_per_minute: u64,
    /// IP 请求限制（每分钟）
    pub ip_requests_per_minute: u64,
    /// 用户请求限制（每分钟）
    pub user_requests_per_minute: u64,
    /// API Key 请求限制（每分钟）
    pub api_key_requests_per_minute: u64,
    /// 窗口大小（秒）- 仅滑动窗口使用
    pub window_secs: u64,
    /// 令牌桶容量 - 仅令牌桶使用
    pub token_bucket_capacity: u64,
    /// 令牌桶补充速率 - 仅令牌桶使用
    pub token_bucket_refill_rate: u64,
    /// 是否允许突发 - 仅令牌桶使用
    pub allow_burst: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::TokenBucket,
            global_requests_per_minute: 1000,
            ip_requests_per_minute: 100,
            user_requests_per_minute: 200,
            api_key_requests_per_minute: 500,
            window_secs: 60,
            token_bucket_capacity: 100,
            token_bucket_refill_rate: 10,
            allow_burst: true,
        }
    }
}

impl RateLimitConfig {
    /// 创建滑动窗口配置
    pub fn sliding_window() -> Self {
        Self {
            algorithm: RateLimitAlgorithm::SlidingWindow,
            ..Default::default()
        }
    }

    /// 创建令牌桶配置
    pub fn token_bucket(capacity: u64, refill_rate: u64) -> Self {
        Self {
            algorithm: RateLimitAlgorithm::TokenBucket,
            token_bucket_capacity: capacity,
            token_bucket_refill_rate: refill_rate,
            ..Default::default()
        }
    }
}

/// 限流状态
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct RateLimitStatus {
    pub dimension: String,
    pub max_requests: u64,
    pub current_count: u64,
    pub remaining: u64,
    pub window_secs: u64,
    #[serde(default)]
    pub algorithm: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_algorithm_default() {
        assert_eq!(
            RateLimitAlgorithm::default(),
            RateLimitAlgorithm::TokenBucket
        );
    }

    #[test]
    fn test_rate_limit_algorithm_equality() {
        assert_eq!(
            RateLimitAlgorithm::SlidingWindow,
            RateLimitAlgorithm::SlidingWindow
        );
        assert_ne!(
            RateLimitAlgorithm::SlidingWindow,
            RateLimitAlgorithm::TokenBucket
        );
    }

    #[test]
    fn test_rate_limit_dimension_equality() {
        assert_eq!(RateLimitDimension::Global, RateLimitDimension::Global);
        assert_eq!(
            RateLimitDimension::Ip("1.2.3.4".to_string()),
            RateLimitDimension::Ip("1.2.3.4".to_string())
        );
        assert_ne!(
            RateLimitDimension::Ip("1.2.3.4".to_string()),
            RateLimitDimension::Ip("5.6.7.8".to_string())
        );
        assert_ne!(
            RateLimitDimension::Global,
            RateLimitDimension::User("u".to_string())
        );
    }

    #[test]
    fn test_rate_limit_dimension_variants() {
        let ip = RateLimitDimension::Ip("192.168.1.1".to_string());
        let user = RateLimitDimension::User("alice".to_string());
        let api_key = RateLimitDimension::ApiKey("key123".to_string());
        assert_ne!(ip, user);
        assert_ne!(user, api_key);
        assert_ne!(ip, api_key);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.algorithm, RateLimitAlgorithm::TokenBucket);
        assert_eq!(config.global_requests_per_minute, 1000);
        assert_eq!(config.ip_requests_per_minute, 100);
        assert_eq!(config.user_requests_per_minute, 200);
        assert_eq!(config.api_key_requests_per_minute, 500);
        assert_eq!(config.window_secs, 60);
        assert_eq!(config.token_bucket_capacity, 100);
        assert_eq!(config.token_bucket_refill_rate, 10);
        assert!(config.allow_burst);
    }

    #[test]
    fn test_rate_limit_config_sliding_window() {
        let config = RateLimitConfig::sliding_window();
        assert_eq!(config.algorithm, RateLimitAlgorithm::SlidingWindow);
        assert_eq!(config.window_secs, 60);
    }

    #[test]
    fn test_rate_limit_config_token_bucket() {
        let config = RateLimitConfig::token_bucket(200, 20);
        assert_eq!(config.algorithm, RateLimitAlgorithm::TokenBucket);
        assert_eq!(config.token_bucket_capacity, 200);
        assert_eq!(config.token_bucket_refill_rate, 20);
    }

    #[test]
    fn test_rate_limit_status_serialization() {
        let status = RateLimitStatus {
            dimension: "global".to_string(),
            max_requests: 100,
            current_count: 30,
            remaining: 70,
            window_secs: 60,
            algorithm: "token_bucket".to_string(),
        };
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: RateLimitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.dimension, "global");
        assert_eq!(deserialized.max_requests, 100);
        assert_eq!(deserialized.remaining, 70);
    }

    #[test]
    fn test_rate_limit_status_default_algorithm() {
        let json = r#"{"dimension":"ip","max_requests":10,"current_count":5,"remaining":5,"window_secs":60}"#;
        let status: RateLimitStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.dimension, "ip");
        assert_eq!(status.algorithm, "");
    }
}
