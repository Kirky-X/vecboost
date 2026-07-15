// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron 后端封装:提供与 RateLimiter 兼容的接口,内部委托 limiteron::TokenBucketLimiter。
//!
//! 用于替代自研 RateLimiter,支持多维度限流(Global/Ip/User/ApiKey)。
//! 每个维度维护独立的令牌桶,容量等于该维度的每分钟限额,补充速率为 限额/60(令牌/秒)。

use std::collections::HashMap;
use std::sync::Arc;

use limiteron::limiters::{Limiter, TokenBucketLimiter};
use tokio::sync::Mutex;

use crate::rate_limit::{RateLimitConfig, RateLimitDimension, RateLimitStatus};

/// limiteron 后端,包装多个 `limiteron::TokenBucketLimiter`(按维度 key 管理)。
pub struct LimiteronAdapter {
    buckets: Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
    config: RateLimitConfig,
}

impl LimiteronAdapter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            config,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(RateLimitConfig::default())
    }

    /// 检查是否允许请求(所有维度都必须通过)。
    pub async fn check_rate_limit(&self, dimensions: Vec<RateLimitDimension>) -> bool {
        for dimension in dimensions {
            let key = dimension_to_key(&dimension);
            let max_requests = dimension_limit(&self.config, &dimension);
            // limit=0 表示禁止该维度所有请求
            if max_requests == 0 {
                return false;
            }
            let bucket = self.get_or_create_bucket(&key, max_requests).await;
            match bucket.allow(1).await {
                Ok(true) => continue,
                _ => return false,
            }
        }
        true
    }

    /// 获取限流状态。
    pub async fn get_status(&self, dimension: RateLimitDimension) -> RateLimitStatus {
        let key = dimension_to_key(&dimension);
        let max_requests = dimension_limit(&self.config, &dimension);
        let bucket = self.get_or_create_bucket(&key, max_requests).await;
        let tokens = bucket.tokens();
        RateLimitStatus {
            dimension: format!("{:?}", dimension),
            max_requests,
            current_count: max_requests.saturating_sub(tokens),
            remaining: tokens,
            window_secs: 60, // 限流窗口固定 60 秒(每分钟限额)
            algorithm: "token_bucket".to_string(),
        }
    }

    /// 获取给定维度的剩余请求数。
    pub async fn get_remaining(&self, dimension: RateLimitDimension) -> u64 {
        self.get_status(dimension).await.remaining
    }

    async fn get_or_create_bucket(&self, key: &str, max_requests: u64) -> Arc<TokenBucketLimiter> {
        let mut buckets = self.buckets.lock().await;
        buckets
            .entry(key.to_string())
            .or_insert_with(|| {
                let refill_rate = (max_requests / 60).max(1);
                Arc::new(TokenBucketLimiter::new(max_requests, refill_rate))
            })
            .clone()
    }
}

fn dimension_to_key(dimension: &RateLimitDimension) -> String {
    match dimension {
        RateLimitDimension::Global => "global".to_string(),
        RateLimitDimension::Ip(ip) => format!("ip:{}", ip),
        RateLimitDimension::User(u) => format!("user:{}", u),
        RateLimitDimension::ApiKey(k) => format!("apikey:{}", k),
    }
}

fn dimension_limit(config: &RateLimitConfig, dimension: &RateLimitDimension) -> u64 {
    match dimension {
        RateLimitDimension::Global => config.global_requests_per_minute,
        RateLimitDimension::Ip(_) => config.ip_requests_per_minute,
        RateLimitDimension::User(_) => config.user_requests_per_minute,
        RateLimitDimension::ApiKey(_) => config.api_key_requests_per_minute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> RateLimitConfig {
        RateLimitConfig {
            global_requests_per_minute: 5,
            ip_requests_per_minute: 3,
            user_requests_per_minute: 4,
            api_key_requests_per_minute: 2,
            ..RateLimitConfig::default()
        }
    }

    #[tokio::test]
    async fn test_check_rate_limit_allows_under_limit() {
        let adapter = LimiteronAdapter::new(small_config());
        // global 限额 5,发 5 个请求都应通过
        for _ in 0..5 {
            assert!(
                adapter
                    .check_rate_limit(vec![RateLimitDimension::Global])
                    .await
            );
        }
    }

    #[tokio::test]
    async fn test_check_rate_limit_blocks_over_limit() {
        let adapter = LimiteronAdapter::new(small_config());
        // api_key 限额 2
        assert!(
            adapter
                .check_rate_limit(vec![RateLimitDimension::ApiKey("k1".into())])
                .await
        );
        assert!(
            adapter
                .check_rate_limit(vec![RateLimitDimension::ApiKey("k1".into())])
                .await
        );
        // 第 3 个应被拒绝
        assert!(
            !adapter
                .check_rate_limit(vec![RateLimitDimension::ApiKey("k1".into())])
                .await
        );
    }

    #[tokio::test]
    async fn test_multiple_dimensions_independent() {
        let adapter = LimiteronAdapter::new(small_config());
        // ip 限额 3,user 限额 4 — 不同维度独立计数
        let ip = RateLimitDimension::Ip("1.2.3.4".into());
        let user = RateLimitDimension::User("alice".into());
        for _ in 0..3 {
            assert!(adapter.check_rate_limit(vec![ip.clone()]).await);
        }
        // ip 已耗尽
        assert!(!adapter.check_rate_limit(vec![ip.clone()]).await);
        // user 仍可用
        assert!(adapter.check_rate_limit(vec![user.clone()]).await);
    }

    #[tokio::test]
    async fn test_check_rate_limit_all_dimensions_must_pass() {
        let adapter = LimiteronAdapter::new(small_config());
        // api_key 限额 2,消耗 2 次后,同时检查 global+apikey 应失败(apikey 维度拒绝)
        let api_key = RateLimitDimension::ApiKey("key".into());
        assert!(adapter.check_rate_limit(vec![api_key.clone()]).await);
        assert!(adapter.check_rate_limit(vec![api_key.clone()]).await);
        // apikey 已耗尽,组合维度应返回 false
        assert!(
            !adapter
                .check_rate_limit(vec![RateLimitDimension::Global, api_key.clone()])
                .await
        );
    }

    #[tokio::test]
    async fn test_get_status_returns_correct_info() {
        let adapter = LimiteronAdapter::new(small_config());
        let dim = RateLimitDimension::Ip("10.0.0.1".into());
        // 消耗 1 个令牌
        assert!(adapter.check_rate_limit(vec![dim.clone()]).await);
        let status = adapter.get_status(dim.clone()).await;
        assert_eq!(status.max_requests, 3);
        assert_eq!(status.remaining, 2);
        assert_eq!(status.current_count, 1);
        assert_eq!(status.algorithm, "token_bucket");
        assert!(status.dimension.contains("Ip"));
    }

    #[tokio::test]
    async fn test_get_remaining_returns_tokens() {
        let adapter = LimiteronAdapter::new(small_config());
        let dim = RateLimitDimension::User("bob".into());
        // 初始剩余 = 限额 4
        assert_eq!(adapter.get_remaining(dim.clone()).await, 4);
        // 消耗 2 个
        adapter.check_rate_limit(vec![dim.clone()]).await;
        adapter.check_rate_limit(vec![dim.clone()]).await;
        assert_eq!(adapter.get_remaining(dim.clone()).await, 2);
    }

    #[tokio::test]
    async fn test_global_dimension_key_isolation() {
        let adapter = LimiteronAdapter::new(small_config());
        // global 限额 5,消耗 5 次
        for _ in 0..5 {
            assert!(
                adapter
                    .check_rate_limit(vec![RateLimitDimension::Global])
                    .await
            );
        }
        // 第 6 次应拒绝
        assert!(
            !adapter
                .check_rate_limit(vec![RateLimitDimension::Global])
                .await
        );
        // 但其他维度仍可用
        assert!(
            adapter
                .check_rate_limit(vec![RateLimitDimension::Ip("9.9.9.9".into())])
                .await
        );
    }

    #[tokio::test]
    async fn test_token_refill_restores_capacity() {
        let adapter = LimiteronAdapter::new(small_config());
        let dim = RateLimitDimension::ApiKey("refill".into());
        // 限额 2,消耗完
        adapter.check_rate_limit(vec![dim.clone()]).await;
        adapter.check_rate_limit(vec![dim.clone()]).await;
        // 此时再请求应被拒绝
        assert!(!adapter.check_rate_limit(vec![dim.clone()]).await);
        // 等待令牌补充(refill_rate = max(2/60,1) = 1 令牌/秒,等 1.2 秒)
        tokio::time::sleep(tokio::time::Duration::from_millis(1200)).await;
        // 补充后请求应成功(allow() 触发 refill)
        assert!(
            adapter.check_rate_limit(vec![dim.clone()]).await,
            "expected refill to allow request after wait"
        );
    }

    #[tokio::test]
    async fn test_limit_zero_rejects_all() {
        let mut config = small_config();
        config.global_requests_per_minute = 0;
        let adapter = LimiteronAdapter::new(config);
        // limit=0 时所有请求被拒绝
        assert!(
            !adapter
                .check_rate_limit(vec![RateLimitDimension::Global])
                .await,
            "limit=0 must reject all requests"
        );
    }

    #[tokio::test]
    async fn test_window_secs_is_60() {
        let adapter = LimiteronAdapter::new(small_config());
        let status = adapter.get_status(RateLimitDimension::Global).await;
        assert_eq!(
            status.window_secs, 60,
            "window_secs must be 60 (per-minute window)"
        );
    }
}
