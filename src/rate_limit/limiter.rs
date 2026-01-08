// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 限流器模块
//!
//! 提供多维度限流功能，支持滑动窗口和令牌桶两种算法

use crate::rate_limit::store::RateLimitStore;
use crate::rate_limit::token_bucket::{TokenBucketConfig, TokenBucketStore};
use std::sync::Arc;

/// 限流算法类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitAlgorithm {
    /// 滑动窗口算法（固定窗口改进版）
    SlidingWindow,
    /// 令牌桶算法（允许突发流量）
    TokenBucket,
}

impl Default for RateLimitAlgorithm {
    fn default() -> Self {
        Self::TokenBucket
    }
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

/// 限流器
pub struct RateLimiter {
    store: Arc<dyn RateLimitStore>,
    token_bucket_store: Option<Arc<dyn TokenBucketStore>>,
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(store: Arc<dyn RateLimitStore>) -> Self {
        Self {
            store,
            token_bucket_store: None,
            config: RateLimitConfig::default(),
        }
    }

    pub fn with_token_bucket_store(
        store: Arc<dyn RateLimitStore>,
        token_bucket_store: Arc<dyn TokenBucketStore>,
    ) -> Self {
        Self {
            store,
            token_bucket_store: Some(token_bucket_store),
            config: RateLimitConfig::default(),
        }
    }

    pub fn with_config(
        store: Arc<dyn RateLimitStore>,
        config: RateLimitConfig,
        token_bucket_store: Option<Arc<dyn TokenBucketStore>>,
    ) -> Self {
        Self {
            store,
            token_bucket_store,
            config,
        }
    }

    /// 检查是否允许请求
    pub async fn check_rate_limit(&self, dimensions: Vec<RateLimitDimension>) -> bool {
        match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => self.check_sliding_window(dimensions).await,
            RateLimitAlgorithm::TokenBucket => self.check_token_bucket(dimensions).await,
        }
    }

    async fn check_sliding_window(&self, dimensions: Vec<RateLimitDimension>) -> bool {
        for dimension in dimensions {
            let (max_requests, key) = match dimension {
                RateLimitDimension::Global => {
                    (self.config.global_requests_per_minute, "global".to_string())
                }
                RateLimitDimension::Ip(ip) => {
                    (self.config.ip_requests_per_minute, format!("ip:{}", ip))
                }
                RateLimitDimension::User(username) => (
                    self.config.user_requests_per_minute,
                    format!("user:{}", username),
                ),
                RateLimitDimension::ApiKey(api_key) => (
                    self.config.api_key_requests_per_minute,
                    format!("apikey:{}", api_key),
                ),
            };

            if !self
                .store
                .check_and_increment(&key, self.config.window_secs, max_requests)
                .await
            {
                return false;
            }
        }

        true
    }

    async fn check_token_bucket(&self, dimensions: Vec<RateLimitDimension>) -> bool {
        let store = match &self.token_bucket_store {
            Some(s) => s.clone(),
            None => return false,
        };

        for dimension in dimensions {
            let (max_requests, key) = match dimension {
                RateLimitDimension::Global => {
                    (self.config.global_requests_per_minute, "global".to_string())
                }
                RateLimitDimension::Ip(ip) => {
                    (self.config.ip_requests_per_minute, format!("ip:{}", ip))
                }
                RateLimitDimension::User(username) => (
                    self.config.user_requests_per_minute,
                    format!("user:{}", username),
                ),
                RateLimitDimension::ApiKey(api_key) => (
                    self.config.api_key_requests_per_minute,
                    format!("apikey:{}", api_key),
                ),
            };

            let _config = TokenBucketConfig::from_requests_per_minute(max_requests);
            let bucket = store.get_bucket(&key).await;

            if !bucket.allows(1).await {
                return false;
            }
        }

        true
    }

    /// 获取限流状态
    pub async fn get_status(&self, dimension: RateLimitDimension) -> RateLimitStatus {
        let key = match &dimension {
            RateLimitDimension::Global => "global".to_string(),
            RateLimitDimension::Ip(ip) => format!("ip:{}", ip),
            RateLimitDimension::User(username) => format!("user:{}", username),
            RateLimitDimension::ApiKey(api_key) => format!("apikey:{}", api_key),
        };

        let max_requests = match dimension {
            RateLimitDimension::Global => self.config.global_requests_per_minute,
            RateLimitDimension::Ip(_) => self.config.ip_requests_per_minute,
            RateLimitDimension::User(_) => self.config.user_requests_per_minute,
            RateLimitDimension::ApiKey(_) => self.config.api_key_requests_per_minute,
        };

        match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => {
                let current_count = self.store.get_count(&key).await;
                let remaining = max_requests.saturating_sub(current_count);
                RateLimitStatus {
                    dimension: format!("{:?}", dimension),
                    max_requests,
                    current_count,
                    remaining,
                    window_secs: self.config.window_secs,
                    algorithm: "sliding_window".to_string(),
                }
            }
            RateLimitAlgorithm::TokenBucket => {
                if let Some(store) = &self.token_bucket_store {
                    let bucket = store.get_bucket(&key).await;
                    let tokens = bucket.tokens().await;
                    RateLimitStatus {
                        dimension: format!("{:?}", dimension),
                        max_requests,
                        current_count: max_requests.saturating_sub(tokens),
                        remaining: tokens,
                        window_secs: self.config.token_bucket_refill_rate,
                        algorithm: "token_bucket".to_string(),
                    }
                } else {
                    RateLimitStatus {
                        dimension: format!("{:?}", dimension),
                        max_requests,
                        current_count: 0,
                        remaining: max_requests,
                        window_secs: 0,
                        algorithm: "token_bucket_unconfigured".to_string(),
                    }
                }
            }
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
