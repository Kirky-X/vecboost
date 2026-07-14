// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron feature 关闭时的 no-op stub。
//!
//! 提供与 `limiteron_adapter::LimiteronAdapter` 相同的方法签名,
//! 但不持有任何令牌桶——`check_rate_limit` 恒为 `true`(允许所有请求),
//! `get_remaining` 返回 `u64::MAX` 表示无限制。

use crate::rate_limit::{RateLimitConfig, RateLimitDimension, RateLimitStatus};

pub struct LimiteronAdapter {
    config: RateLimitConfig,
}

impl LimiteronAdapter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(RateLimitConfig::default())
    }

    pub async fn check_rate_limit(&self, _dimensions: Vec<RateLimitDimension>) -> bool {
        true
    }

    pub async fn get_status(&self, dimension: RateLimitDimension) -> RateLimitStatus {
        let max_requests = match &dimension {
            RateLimitDimension::Global => self.config.global_requests_per_minute,
            RateLimitDimension::Ip(_) => self.config.ip_requests_per_minute,
            RateLimitDimension::User(_) => self.config.user_requests_per_minute,
            RateLimitDimension::ApiKey(_) => self.config.api_key_requests_per_minute,
        };
        RateLimitStatus {
            dimension: format!("{:?}", dimension),
            max_requests,
            current_count: 0,
            remaining: max_requests,
            window_secs: 60, // 限流窗口固定 60 秒(每分钟限额)
            algorithm: "noop".to_string(),
        }
    }

    pub async fn get_remaining(&self, dimension: RateLimitDimension) -> u64 {
        self.get_status(dimension).await.remaining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_window_secs_is_60() {
        let adapter = LimiteronAdapter::with_default_config();
        let status = adapter.get_status(RateLimitDimension::Global).await;
        assert_eq!(
            status.window_secs, 60,
            "window_secs must be 60 (per-minute window)"
        );
    }
}
