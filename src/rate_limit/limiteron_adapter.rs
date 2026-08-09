// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron Governor 后端：多维度限流适配器。
//!
//! 基于 limiteron 原生 `Governor` 编排层，通过 `FlowControlConfig` 规则配置
//! Global/Ip/User/ApiKey 四个维度的限流。每个维度对应一条 Rule，所有匹配
//! 规则按优先级顺序执行 DecisionChain——任一拒绝即整体拒绝（AND 语义）。
//!
//! Governor 内部提供：标识符提取（CompositeExtractor）、L1 缓存、封禁检查、
//! 熔断器、健康检查、统计收集等编排能力。

use std::sync::Arc;

use limiteron::Governor;
use limiteron::matchers::RequestContext;
use limiteron::storage::{MemoryBanStorage, MemoryStorage};

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

/// limiteron Governor 后端，通过 `Governor::check()` 完成限流决策。
pub struct LimiteronAdapter {
    governor: Governor,
}

impl LimiteronAdapter {
    /// 使用指定配置创建 Governor 后端。
    pub async fn new(settings: RateLimitSettings) -> Self {
        let config = build_flow_control_config(&settings);
        let storage: Arc<dyn limiteron::storage::Storage> = MemoryStorage::create_storage();
        let ban_storage: Arc<dyn limiteron::storage::BanStorage> =
            MemoryBanStorage::create_ban_storage();

        let governor = Governor::builder()
            .with_config(config)
            .with_storage(storage)
            .with_ban_storage(ban_storage)
            .with_l1_cache_enabled(false) // 禁用 L1 缓存：限流需要每次检查 limiter
            .build()
            .await
            .expect("Governor build with valid config should succeed");

        Self { governor }
    }

    /// 使用默认配置创建 Governor 后端。
    pub async fn with_defaults() -> Self {
        Self::new(RateLimitSettings::default()).await
    }

    /// 检查请求是否被允许（所有维度规则都必须通过）。
    ///
    /// 调用 `Governor::check()`，将 `Decision::Allowed` 映射为 `true`，
    /// 其余（Rejected/Banned/Err）映射为 `false`。
    pub async fn check_rate_limit(&self, context: &RequestContext) -> bool {
        match self.governor.check(context).await {
            Ok(limiteron::error::Decision::Allowed(_)) => true,
            _ => false,
        }
    }

    /// 获取 Governor 健康状态。
    pub async fn health_status(&self) -> bool {
        self.governor.health_status().await.healthy()
    }

    /// 获取 Governor 统计快照。
    pub async fn stats(&self) -> limiteron::GovernorStats {
        self.governor.stats().await
    }
}

/// 从 RateLimitSettings 构建 FlowControlConfig。
///
/// 为每个维度创建一条 Rule，所有 Rule 使用通配匹配（匹配所有请求）。
/// Rule 按优先级排序：global(100) > ip(90) > user(80) > api_key(70)。
/// Governor 的 check 循环对所有匹配 Rule 的 DecisionChain 依次检查，
/// 任一拒绝即整体拒绝，实现 AND 语义。
fn build_flow_control_config(settings: &RateLimitSettings) -> limiteron::config::FlowControlConfig {
    use limiteron::config::{
        ActionConfig, Action, CacheBackend, FlowControlConfig, GlobalConfig, LimiterConfig,
        Matcher, MetricsBackend, Rule, StorageType, TrustedProxyConfig,
    };

    let make_rule = |id: &str, name: &str, priority: u16, rpm: u64| Rule {
        id: id.to_string(),
        name: name.to_string(),
        priority,
        matchers: vec![Matcher::User {
            user_ids: vec!["*".to_string()],
        }],
        limiters: vec![LimiterConfig::TokenBucket {
            capacity: rpm,
            refill_rate: (rpm / 60).max(1),
        }],
        action: ActionConfig {
            on_exceed: Action::Reject,
            ban: None,
        },
    };

    let mut rules = Vec::new();

    if settings.global_requests_per_minute > 0 {
        rules.push(make_rule(
            "global",
            "Global rate limit",
            100,
            settings.global_requests_per_minute,
        ));
    }
    if settings.ip_requests_per_minute > 0 {
        rules.push(make_rule(
            "ip",
            "IP rate limit",
            90,
            settings.ip_requests_per_minute,
        ));
    }
    if settings.user_requests_per_minute > 0 {
        rules.push(make_rule(
            "user",
            "User rate limit",
            80,
            settings.user_requests_per_minute,
        ));
    }
    if settings.api_key_requests_per_minute > 0 {
        rules.push(make_rule(
            "api_key",
            "API Key rate limit",
            70,
            settings.api_key_requests_per_minute,
        ));
    }

    FlowControlConfig {
        version: "1.0".to_string(),
        global: GlobalConfig {
            storage: StorageType::Memory,
            cache: CacheBackend::Memory,
            metrics: MetricsBackend::Prometheus,
            trusted_proxies: TrustedProxyConfig::default(),
        },
        rules,
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

    fn ctx(ip: &str) -> RequestContext {
        RequestContext {
            client_ip: Some(ip.to_string()),
            path: "/test".to_string(),
            method: "GET".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_check_rate_limit_allows_under_limit() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        // 最严格的维度是 api_key（2 req/min），但所有维度共享同一 limiter
        // 所以前 2 次请求应通过
        assert!(adapter.check_rate_limit(&ctx("1.2.3.4")).await);
        assert!(adapter.check_rate_limit(&ctx("1.2.3.4")).await);
    }

    #[tokio::test]
    async fn test_check_rate_limit_blocks_over_limit() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let context = ctx("1.2.3.4");
        // api_key 维度限制 2/min，是最严格的
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(
            !adapter.check_rate_limit(&context).await,
            "third request should be rejected"
        );
    }

    #[tokio::test]
    async fn test_different_contexts_share_global_limit() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        // Governor 规则的 limiter 是共享的（非 per-key）
        let ctx1 = ctx("1.2.3.4");
        let ctx2 = ctx("5.6.7.8");
        assert!(adapter.check_rate_limit(&ctx1).await);
        assert!(adapter.check_rate_limit(&ctx2).await);
        // 已用完 api_key 维度的 2 个令牌
        assert!(!adapter.check_rate_limit(&ctx1).await);
    }

    #[tokio::test]
    async fn test_global_only_settings() {
        let settings = RateLimitSettings {
            global_requests_per_minute: 3,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 0,
            api_key_requests_per_minute: 0,
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let context = ctx("1.2.3.4");
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(!adapter.check_rate_limit(&context).await);
    }

    #[tokio::test]
    async fn test_health_status_healthy() {
        let adapter = LimiteronAdapter::with_defaults().await;
        assert!(adapter.health_status().await);
    }

    #[tokio::test]
    async fn test_stats_returns_non_zero_after_requests() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let context = ctx("1.2.3.4");
        adapter.check_rate_limit(&context).await;
        adapter.check_rate_limit(&context).await;
        let stats = adapter.stats().await;
        assert!(stats.total_requests >= 2, "stats should show at least 2 requests");
    }
}
