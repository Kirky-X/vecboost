// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron 混合限流后端：Governor 全局编排 + per-key TokenBucketLimiter 维度隔离。
//!
//! 架构分两层：
//! - **Governor**：全局限流 + 封禁检查 + 熔断器 + 健康检查 + 统计。通过
//!   `FlowControlConfig` 配置一条 global 规则，管理系统总吞吐量。
//! - **per-key TokenBucketLimiter**：IP/User/ApiKey 三个维度的独立限流。
//!   每个 key 维护独立的令牌桶，不同 IP/User/ApiKey 互不影响。
//!
//! 检查顺序：Governor（全局）→ per-key（IP → User → ApiKey），
//! 任一拒绝即整体拒绝。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use limiteron::Governor;
use limiteron::limiters::{Limiter, TokenBucketLimiter};
use limiteron::matchers::RequestContext;
use limiteron::storage::{MemoryBanStorage, MemoryStorage};
use tokio::sync::Mutex;

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

/// limiteron 混合限流后端。
///
/// Governor 负责全局限流 + 封禁/熔断/统计；per-key TokenBucketLimiter
/// 负责维度隔离（每个 IP/User/ApiKey 独立限额）。
pub struct LimiteronAdapter {
    governor: Governor,
    ip_buckets: Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
    user_buckets: Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
    api_key_buckets: Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
    settings: RateLimitSettings,
    /// 缓存的健康状态（由异步检查更新，同步方法读取）
    health_cache: AtomicBool,
}

impl LimiteronAdapter {
    /// 使用指定配置创建限流后端。
    pub async fn new(settings: RateLimitSettings) -> Self {
        let config = build_global_flow_control_config(&settings);
        let storage: Arc<dyn limiteron::storage::Storage> = MemoryStorage::create_storage();
        let ban_storage: Arc<dyn limiteron::storage::BanStorage> =
            MemoryBanStorage::create_ban_storage();

        let governor = Governor::builder()
            .with_config(config)
            .with_storage(storage)
            .with_ban_storage(ban_storage)
            .with_l1_cache_enabled(false)
            .build()
            .await
            .expect("Governor build with valid config should succeed");

        Self {
            governor,
            ip_buckets: Mutex::new(HashMap::new()),
            user_buckets: Mutex::new(HashMap::new()),
            api_key_buckets: Mutex::new(HashMap::new()),
            settings,
            health_cache: AtomicBool::new(true),
        }
    }

    /// 使用默认配置创建限流后端。
    pub async fn with_defaults() -> Self {
        Self::new(RateLimitSettings::default()).await
    }

    /// 检查请求是否被允许。
    ///
    /// 检查顺序：Governor 全局 → IP per-key → User per-key → ApiKey per-key。
    /// 任一拒绝即整体拒绝。
    pub async fn check_rate_limit(&self, context: &RequestContext) -> bool {
        // 1. Governor 全局检查（含封禁、熔断）
        match self.governor.check(context).await {
            Ok(limiteron::error::Decision::Allowed(_)) => {}
            _ => return false,
        }

        // 2. per-key 维度检查
        if self.settings.ip_requests_per_minute > 0 {
            if let Some(ip) = &context.client_ip {
                if !self.check_per_key(&self.ip_buckets, ip, self.settings.ip_requests_per_minute).await {
                    return false;
                }
            }
        }

        if self.settings.user_requests_per_minute > 0 {
            if let Some(user_id) = &context.user_id {
                if !self.check_per_key(&self.user_buckets, user_id, self.settings.user_requests_per_minute).await {
                    return false;
                }
            }
        }

        if self.settings.api_key_requests_per_minute > 0 {
            if let Some(api_key) = &context.api_key {
                if !self.check_per_key(&self.api_key_buckets, api_key, self.settings.api_key_requests_per_minute).await {
                    return false;
                }
            }
        }

        true
    }

    /// per-key 令牌桶检查：获取或创建桶，消费 1 个令牌。
    async fn check_per_key(
        &self,
        buckets: &Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
        key: &str,
        rpm: u64,
    ) -> bool {
        let bucket = {
            let mut map = buckets.lock().await;
            map.entry(key.to_string())
                .or_insert_with(|| {
                    let refill_rate = (rpm / 60).max(1);
                    Arc::new(TokenBucketLimiter::new(rpm, refill_rate))
                })
                .clone()
        };
        bucket.allow(1).await.unwrap_or(false)
    }

    /// 异步健康检查：委托 Governor + 更新缓存。
    pub async fn check_health(&self) -> bool {
        let healthy = self.governor.health_status().await.healthy();
        self.health_cache.store(healthy, Ordering::Relaxed);
        healthy
    }

    /// 同步健康检查：读取缓存值（由 `check_health` 异步更新）。
    pub fn is_healthy(&self) -> bool {
        self.health_cache.load(Ordering::Relaxed)
    }

    /// 获取 Governor 统计快照。
    pub async fn stats(&self) -> limiteron::GovernorStats {
        self.governor.stats().await
    }
}

/// 构建 Governor 的 FlowControlConfig（仅全局规则）。
///
/// Governor 负责全局限流（系统总吞吐量管理），per-key 维度隔离由
/// `TokenBucketLimiter` 直接处理（Governor 的 DecisionChain 不支持 per-key）。
fn build_global_flow_control_config(
    settings: &RateLimitSettings,
) -> limiteron::config::FlowControlConfig {
    use limiteron::config::{
        Action, ActionConfig, CacheBackend, FlowControlConfig, GlobalConfig, LimiterConfig,
        Matcher, MetricsBackend, Rule, StorageType, TrustedProxyConfig,
    };

    let rules = if settings.global_requests_per_minute > 0 {
        vec![Rule {
            id: "global".to_string(),
            name: "Global rate limit".to_string(),
            priority: 100,
            matchers: vec![Matcher::User {
                user_ids: vec!["*".to_string()],
            }],
            limiters: vec![LimiterConfig::TokenBucket {
                capacity: settings.global_requests_per_minute,
                refill_rate: (settings.global_requests_per_minute / 60).max(1),
            }],
            action: ActionConfig {
                on_exceed: Action::Reject,
                ban: None,
            },
        }]
    } else {
        vec![]
    };

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
            global_requests_per_minute: 100, // 全局宽松，让 per-key 维度先触发
            ip_requests_per_minute: 3,
            user_requests_per_minute: 4,
            api_key_requests_per_minute: 2,
        }
    }

    fn ctx_ip(ip: &str) -> RequestContext {
        RequestContext {
            client_ip: Some(ip.to_string()),
            path: "/test".to_string(),
            method: "GET".to_string(),
            ..Default::default()
        }
    }

    fn ctx_user(user: &str) -> RequestContext {
        RequestContext {
            user_id: Some(user.to_string()),
            client_ip: Some("1.2.3.4".to_string()),
            path: "/test".to_string(),
            method: "GET".to_string(),
            ..Default::default()
        }
    }

    // ==================== 全局限流测试 ====================

    #[tokio::test]
    async fn test_global_limit_blocks_when_exceeded() {
        let settings = RateLimitSettings {
            global_requests_per_minute: 3,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 0,
            api_key_requests_per_minute: 0,
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let context = ctx_ip("1.2.3.4");
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(!adapter.check_rate_limit(&context).await);
    }

    // ==================== per-key IP 限流测试 ====================

    #[tokio::test]
    async fn test_ip_dimension_blocks_over_limit() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let ctx = ctx_ip("1.2.3.4");
        assert!(adapter.check_rate_limit(&ctx).await);
        assert!(adapter.check_rate_limit(&ctx).await);
        assert!(adapter.check_rate_limit(&ctx).await);
        assert!(
            !adapter.check_rate_limit(&ctx).await,
            "4th request from same IP should be rejected"
        );
    }

    #[tokio::test]
    async fn test_different_ips_have_independent_limits() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let ctx1 = ctx_ip("1.2.3.4");
        let ctx2 = ctx_ip("5.6.7.8");

        // 用尽 IP 1.2.3.4 的 3 个令牌
        assert!(adapter.check_rate_limit(&ctx1).await);
        assert!(adapter.check_rate_limit(&ctx1).await);
        assert!(adapter.check_rate_limit(&ctx1).await);
        assert!(!adapter.check_rate_limit(&ctx1).await);

        // IP 5.6.7.8 仍有独立配额
        assert!(adapter.check_rate_limit(&ctx2).await);
        assert!(adapter.check_rate_limit(&ctx2).await);
    }

    // ==================== per-key User 限流测试 ====================

    #[tokio::test]
    async fn test_user_dimension_independent_from_ip() {
        // 禁用 IP 维度，只测 User per-key 独立性
        let settings = RateLimitSettings {
            global_requests_per_minute: 100,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 4,
            api_key_requests_per_minute: 0,
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let ctx_alice = ctx_user("alice");
        let ctx_bob = RequestContext {
            user_id: Some("bob".to_string()),
            ..ctx_user("charlie")
        };

        // 用尽 alice 的 4 个用户令牌
        for _ in 0..4 {
            assert!(adapter.check_rate_limit(&ctx_alice).await);
        }
        assert!(!adapter.check_rate_limit(&ctx_alice).await);

        // bob 有独立配额
        assert!(adapter.check_rate_limit(&ctx_bob).await);
    }

    // ==================== 健康检查测试 ====================

    #[tokio::test]
    async fn test_health_check_returns_true_on_init() {
        let adapter = LimiteronAdapter::with_defaults().await;
        assert!(adapter.is_healthy(), "health cache should be true on init");
    }

    #[tokio::test]
    async fn test_async_health_check_updates_cache() {
        let adapter = LimiteronAdapter::with_defaults().await;
        let healthy = adapter.check_health().await;
        assert!(healthy);
        assert!(adapter.is_healthy());
    }

    // ==================== 统计测试 ====================

    #[tokio::test]
    async fn test_stats_returns_non_zero_after_requests() {
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let context = ctx_ip("1.2.3.4");
        adapter.check_rate_limit(&context).await;
        adapter.check_rate_limit(&context).await;
        let stats = adapter.stats().await;
        assert!(
            stats.total_requests >= 2,
            "stats should show at least 2 requests"
        );
    }
}
