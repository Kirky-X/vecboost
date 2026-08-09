// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron 后端封装：多维度限流适配器。
//!
//! 基于 limiteron 原生 `Limiter` trait + `TokenBucketLimiter`，
//! 支持 Global/Ip/User/ApiKey 四个维度的独立限流。
//! 每个维度维护独立的令牌桶，容量等于该维度的每分钟限额，
//! 补充速率为 限额/60（令牌/秒）。

use std::collections::HashMap;
use std::sync::Arc;

use limiteron::limiters::{Limiter, TokenBucketLimiter};
use tokio::sync::Mutex;

/// 限流维度（VecBoost 特有的多维度路由逻辑）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// 全局限流
    Global,
    /// IP 限流
    Ip(String),
    /// 用户限流
    User(String),
    /// API Key 限流
    ApiKey(String),
}

/// 多维度限流配置
#[derive(Debug, Clone)]
pub struct RateLimitSettings {
    /// 全局请求限制（每分钟）
    pub global_requests_per_minute: u64,
    /// IP 请求限制（每分钟）
    pub ip_requests_per_minute: u64,
    /// 用户请求限制（每分钟）
    pub user_requests_per_minute: u64,
    /// API Key 请求限制（每分钟）
    pub api_key_requests_per_minute: u64,
}

impl Default for RateLimitSettings {
    fn default() -> Self {
        Self {
            global_requests_per_minute: 1000,
            ip_requests_per_minute: 100,
            user_requests_per_minute: 200,
            api_key_requests_per_minute: 500,
        }
    }
}

/// limiteron 后端，按维度管理 `TokenBucketLimiter` 实例。
pub struct LimiteronAdapter {
    buckets: Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
    settings: RateLimitSettings,
}

impl LimiteronAdapter {
    pub fn new(settings: RateLimitSettings) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            settings,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(RateLimitSettings::default())
    }

    /// 检查是否允许请求（所有维度都必须通过）。
    pub async fn check_rate_limit(&self, dimensions: Vec<Dimension>) -> bool {
        for dimension in dimensions {
            let key = dimension_to_key(&dimension);
            let max_requests = self.dimension_limit(&dimension);
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

    /// 获取给定维度的剩余令牌数。
    pub async fn get_remaining(&self, dimension: Dimension) -> u64 {
        let key = dimension_to_key(&dimension);
        let max_requests = self.dimension_limit(&dimension);
        let bucket = self.get_or_create_bucket(&key, max_requests).await;
        bucket.tokens()
    }

    async fn get_or_create_bucket(
        &self,
        key: &str,
        max_requests: u64,
    ) -> Arc<TokenBucketLimiter> {
        let mut buckets = self.buckets.lock().await;
        buckets
            .entry(key.to_string())
            .or_insert_with(|| {
                let refill_rate = (max_requests / 60).max(1);
                Arc::new(TokenBucketLimiter::new(max_requests, refill_rate))
            })
            .clone()
    }

    fn dimension_limit(&self, dimension: &Dimension) -> u64 {
        match dimension {
            Dimension::Global => self.settings.global_requests_per_minute,
            Dimension::Ip(_) => self.settings.ip_requests_per_minute,
            Dimension::User(_) => self.settings.user_requests_per_minute,
            Dimension::ApiKey(_) => self.settings.api_key_requests_per_minute,
        }
    }
}

fn dimension_to_key(dimension: &Dimension) -> String {
    match dimension {
        Dimension::Global => "global".to_string(),
        Dimension::Ip(ip) => format!("ip:{}", ip),
        Dimension::User(u) => format!("user:{}", u),
        Dimension::ApiKey(k) => format!("apikey:{}", k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_settings() -> RateLimitSettings {
        RateLimitSettings {
            global_requests_per_minute: 5,
            ip_requests_per_minute: 3,
            user_requests_per_minute: 4,
            api_key_requests_per_minute: 2,
        }
    }

    #[tokio::test]
    async fn test_check_rate_limit_allows_under_limit() {
        let adapter = LimiteronAdapter::new(small_settings());
        for _ in 0..5 {
            assert!(
                adapter
                    .check_rate_limit(vec![Dimension::Global])
                    .await
            );
        }
    }

    #[tokio::test]
    async fn test_check_rate_limit_blocks_over_limit() {
        let adapter = LimiteronAdapter::new(small_settings());
        assert!(
            adapter
                .check_rate_limit(vec![Dimension::ApiKey("k1".into())])
                .await
        );
        assert!(
            adapter
                .check_rate_limit(vec![Dimension::ApiKey("k1".into())])
                .await
        );
        assert!(
            !adapter
                .check_rate_limit(vec![Dimension::ApiKey("k1".into())])
                .await
        );
    }

    #[tokio::test]
    async fn test_multiple_dimensions_independent() {
        let adapter = LimiteronAdapter::new(small_settings());
        let ip = Dimension::Ip("1.2.3.4".into());
        let user = Dimension::User("alice".into());
        for _ in 0..3 {
            assert!(adapter.check_rate_limit(vec![ip.clone()]).await);
        }
        assert!(!adapter.check_rate_limit(vec![ip.clone()]).await);
        assert!(adapter.check_rate_limit(vec![user.clone()]).await);
    }

    #[tokio::test]
    async fn test_check_rate_limit_all_dimensions_must_pass() {
        let adapter = LimiteronAdapter::new(small_settings());
        let api_key = Dimension::ApiKey("key".into());
        assert!(adapter.check_rate_limit(vec![api_key.clone()]).await);
        assert!(adapter.check_rate_limit(vec![api_key.clone()]).await);
        assert!(
            !adapter
                .check_rate_limit(vec![Dimension::Global, api_key.clone()])
                .await
        );
    }

    #[tokio::test]
    async fn test_get_remaining_returns_tokens() {
        let adapter = LimiteronAdapter::new(small_settings());
        let dim = Dimension::User("bob".into());
        assert_eq!(adapter.get_remaining(dim.clone()).await, 4);
        adapter.check_rate_limit(vec![dim.clone()]).await;
        adapter.check_rate_limit(vec![dim.clone()]).await;
        assert_eq!(adapter.get_remaining(dim.clone()).await, 2);
    }

    #[tokio::test]
    async fn test_global_dimension_key_isolation() {
        let adapter = LimiteronAdapter::new(small_settings());
        for _ in 0..5 {
            assert!(
                adapter
                    .check_rate_limit(vec![Dimension::Global])
                    .await
            );
        }
        assert!(
            !adapter
                .check_rate_limit(vec![Dimension::Global])
                .await
        );
        assert!(
            adapter
                .check_rate_limit(vec![Dimension::Ip("9.9.9.9".into())])
                .await
        );
    }

    #[tokio::test]
    async fn test_token_refill_restores_capacity() {
        let adapter = LimiteronAdapter::new(small_settings());
        let dim = Dimension::ApiKey("refill".into());
        adapter.check_rate_limit(vec![dim.clone()]).await;
        adapter.check_rate_limit(vec![dim.clone()]).await;
        assert!(!adapter.check_rate_limit(vec![dim.clone()]).await);
        // refill_rate = max(2/60,1) = 1 令牌/秒,等 1.2 秒
        tokio::time::sleep(tokio::time::Duration::from_millis(1200)).await;
        assert!(
            adapter.check_rate_limit(vec![dim.clone()]).await,
            "expected refill to allow request after wait"
        );
    }

    #[tokio::test]
    async fn test_limit_zero_rejects_all() {
        let mut settings = small_settings();
        settings.global_requests_per_minute = 0;
        let adapter = LimiteronAdapter::new(settings);
        assert!(
            !adapter
                .check_rate_limit(vec![Dimension::Global])
                .await,
            "limit=0 must reject all requests"
        );
    }
}
