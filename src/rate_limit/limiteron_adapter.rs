// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use limiteron::Governor;
use limiteron::limiters::{Limiter, TokenBucketLimiter};
use limiteron::matchers::RequestContext;
use limiteron::middleware::RateLimitHeaderValues;
use limiteron::storage::{MemoryBanStorage, MemoryStorage};
use tokio::sync::Mutex;

/// 限流决策详情：布尔结果 + 绑定维度快照。
///
/// `headers` 供 IETF RateLimit-* 响应头渲染：allowed 时为剩余额度最紧维度
/// （全局 Governor 与各 per-key 桶中 remaining 最小者）的快照；rejected 时为
/// 触发拒绝维度的快照（429 响应据此携带 Retry-After）。Governor 封禁
/// （Banned）不携带快照——封禁事件由 Governor 审计流处理。
#[derive(Debug, Clone)]
pub struct RateLimitDecision {
    pub allowed: bool,
    pub headers: Option<RateLimitHeaderValues>,
}

impl RateLimitDecision {
    fn allowed(headers: Option<RateLimitHeaderValues>) -> Self {
        Self {
            allowed: true,
            headers,
        }
    }

    fn rejected(headers: Option<RateLimitHeaderValues>) -> Self {
        Self {
            allowed: false,
            headers,
        }
    }
}

/// 取剩余额度更紧（remaining 更小）的维度快照。
fn tighter(
    a: Option<RateLimitHeaderValues>,
    b: Option<RateLimitHeaderValues>,
) -> Option<RateLimitHeaderValues> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (a, b) => {
            // 此臂 a/b 必为 Some（None 臂已在上方拦截）；用 ? 收窄避免 unwrap
            let (a, b) = (a?, b?);
            Some(if a.remaining <= b.remaining { a } else { b })
        }
    }
}

/// per-key 维度检查结果（携带消费后的非消费快照，供响应头渲染）。
enum KeyOutcome {
    Allowed(Option<RateLimitHeaderValues>),
    Rejected(Option<RateLimitHeaderValues>),
}

/// 当前 Unix 秒（时钟回拨等异常时回退 0，仅影响 RateLimit-Reset 展示）。
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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
            ip_buckets: Mutex::new(HashMap::with_capacity(64)),
            user_buckets: Mutex::new(HashMap::with_capacity(64)),
            api_key_buckets: Mutex::new(HashMap::with_capacity(64)),
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
        self.check_rate_limit_detailed(context).await.allowed
    }

    /// 检查请求是否被允许，并携带绑定维度的限流快照。
    ///
    /// 语义与 [`Self::check_rate_limit`] 完全一致；额外返回剩余额度最紧
    /// 维度（allowed）或触发拒绝维度（rejected）的 `RateLimitHeaderValues`，
    /// 供限流中间件注入 IETF RateLimit-* 响应头。
    pub async fn check_rate_limit_detailed(&self, context: &RequestContext) -> RateLimitDecision {
        // 1. Governor 全局检查（含封禁、熔断）
        let global_headers = match self.governor.check(context).await {
            // limit==0 表示全局规则未启用（build_global_flow_control_config
            // 对 0 配额不建规则），不参与绑定维度比较
            Ok(limiteron::error::Decision::Allowed(meta)) if meta.limit > 0 => {
                Some(RateLimitHeaderValues {
                    limit: meta.limit,
                    remaining: meta.remaining,
                    reset_at: meta.reset_at,
                    retry_after: None,
                    policy: meta.policy,
                })
            }
            Ok(limiteron::error::Decision::Allowed(_)) => None,
            Ok(limiteron::error::Decision::Rejected(meta)) => {
                return RateLimitDecision::rejected(Some(RateLimitHeaderValues {
                    limit: meta.limit,
                    remaining: 0,
                    reset_at: meta.reset_at,
                    retry_after: Some(meta.retry_after),
                    policy: "global".to_string(),
                }));
            }
            // Banned / 内部错误：无头数据（封禁事件走 Governor 审计流）
            _ => return RateLimitDecision::rejected(None),
        };

        // 2. per-key 维度检查
        let mut binding = global_headers;

        if self.settings.ip_requests_per_minute > 0
            && let Some(ip) = &context.client_ip
        {
            match self
                .check_per_key_detailed(
                    &self.ip_buckets,
                    ip,
                    self.settings.ip_requests_per_minute,
                    "ip",
                )
                .await
            {
                KeyOutcome::Allowed(headers) => binding = tighter(binding, headers),
                KeyOutcome::Rejected(headers) => return RateLimitDecision::rejected(headers),
            }
        }

        if self.settings.user_requests_per_minute > 0
            && let Some(user_id) = &context.user_id
        {
            match self
                .check_per_key_detailed(
                    &self.user_buckets,
                    user_id,
                    self.settings.user_requests_per_minute,
                    "user",
                )
                .await
            {
                KeyOutcome::Allowed(headers) => binding = tighter(binding, headers),
                KeyOutcome::Rejected(headers) => return RateLimitDecision::rejected(headers),
            }
        }

        if self.settings.api_key_requests_per_minute > 0
            && let Some(api_key) = &context.api_key
        {
            match self
                .check_per_key_detailed(
                    &self.api_key_buckets,
                    api_key,
                    self.settings.api_key_requests_per_minute,
                    "api-key",
                )
                .await
            {
                KeyOutcome::Allowed(headers) => binding = tighter(binding, headers),
                KeyOutcome::Rejected(headers) => return RateLimitDecision::rejected(headers),
            }
        }

        RateLimitDecision::allowed(binding)
    }

    /// per-key 令牌桶检查：获取或创建桶，消费 1 个令牌，并读取消费后的
    /// 非消费快照（`remaining()`）供响应头渲染。
    async fn check_per_key_detailed(
        &self,
        buckets: &Mutex<HashMap<String, Arc<TokenBucketLimiter>>>,
        key: &str,
        rpm: u64,
        dimension: &str,
    ) -> KeyOutcome {
        let bucket = {
            let mut map = buckets.lock().await;
            map.entry(key.to_string())
                .or_insert_with(|| {
                    let refill_rate = (rpm / 60).max(1);
                    Arc::new(TokenBucketLimiter::new(rpm, refill_rate))
                })
                .clone()
        };
        let allowed = bucket.allow(1).await.unwrap_or(false);
        let headers = bucket.remaining().await.ok().map(|snapshot| {
            let policy = format!("{dimension}-token-bucket");
            if allowed {
                RateLimitHeaderValues {
                    limit: snapshot.limit,
                    remaining: snapshot.remaining,
                    reset_at: unix_now() + snapshot.reset_secs,
                    retry_after: None,
                    policy,
                }
            } else {
                RateLimitHeaderValues {
                    limit: snapshot.limit,
                    remaining: snapshot.remaining,
                    reset_at: unix_now() + snapshot.reset_secs,
                    retry_after: Some(snapshot.reset_secs.max(1)),
                    policy,
                }
            }
        });
        if allowed {
            KeyOutcome::Allowed(headers)
        } else {
            KeyOutcome::Rejected(headers)
        }
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
///
/// `global_requests_per_minute == 0` 语义为"不设全局上限"：但 Governor 要求
/// 至少一条规则（空规则集构建报 ConfigError），故以高容量哨兵规则代替
/// （单实例吞吐远低于 100 万/分钟，等效于不限）。
fn build_global_flow_control_config(
    settings: &RateLimitSettings,
) -> limiteron::config::FlowControlConfig {
    use limiteron::config::{
        Action, ActionConfig, CacheBackend, FlowControlConfig, GlobalConfig, LimiterConfig,
        Matcher, MetricsBackend, Rule, StorageType, TrustedProxyConfig,
    };

    // (容量, 补充速率)：0 配额映射为哨兵高容量规则
    let (capacity, refill_rate) = if settings.global_requests_per_minute > 0 {
        (
            settings.global_requests_per_minute,
            (settings.global_requests_per_minute / 60).max(1),
        )
    } else {
        (1_000_000, 16_666)
    };

    let rules = vec![Rule {
        id: "global".to_string(),
        name: "Global rate limit".to_string(),
        priority: 100,
        matchers: vec![Matcher::User {
            user_ids: vec!["*".to_string()],
        }],
        limiters: vec![LimiterConfig::TokenBucket {
            capacity,
            refill_rate,
        }],
        action: ActionConfig {
            on_exceed: Action::Reject,
            ban: None,
        },
    }];

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

    // ==================== RateLimit-* 响应头详情测试 ====================

    #[tokio::test]
    async fn test_detailed_decision_carries_binding_snapshot() {
        // 只启用 IP 维度，绑定维度必为 ip-token-bucket
        let settings = RateLimitSettings {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 3,
            user_requests_per_minute: 0,
            api_key_requests_per_minute: 0,
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let ctx = ctx_ip("9.9.9.9");

        let decision = adapter.check_rate_limit_detailed(&ctx).await;
        assert!(decision.allowed);
        let headers = decision.headers.expect("allowed 决策应携带绑定维度快照");
        assert_eq!(headers.limit, 3, "limit 应为 IP 维度配额");
        assert_eq!(headers.remaining, 2, "消费 1 个令牌后剩余 2");
        assert!(headers.retry_after.is_none(), "放行不携带 Retry-After");
        assert_eq!(headers.policy, "ip-token-bucket");

        // 用尽配额后：拒绝决策携带 Retry-After
        assert!(adapter.check_rate_limit_detailed(&ctx).await.allowed);
        assert!(adapter.check_rate_limit_detailed(&ctx).await.allowed);
        let rejected = adapter.check_rate_limit_detailed(&ctx).await;
        assert!(!rejected.allowed);
        let headers = rejected.headers.expect("拒绝决策应携带拒绝维度快照");
        assert!(headers.retry_after.is_some(), "拒绝决策应携带 Retry-After");
        assert_eq!(headers.policy, "ip-token-bucket");
    }

    #[tokio::test]
    async fn test_detailed_decision_picks_tightest_dimension() {
        // global 1000 很宽松、ip 3 很紧：绑定维度应取 ip（remaining 更小）
        let adapter = LimiteronAdapter::new(small_settings()).await;
        let ctx = ctx_ip("8.8.8.8");
        let decision = adapter.check_rate_limit_detailed(&ctx).await;
        assert!(decision.allowed);
        let headers = decision.headers.expect("应携带绑定维度快照");
        assert_eq!(
            headers.policy, "ip-token-bucket",
            "绑定维度应为剩余额度最紧的 per-key 桶"
        );
    }

    #[tokio::test]
    async fn test_bool_wrapper_matches_detailed_decision() {
        // 两个入口共享桶状态，需在各自独立的适配器上跑相同序列
        let adapter_bool = LimiteronAdapter::new(small_settings()).await;
        let adapter_detailed = LimiteronAdapter::new(small_settings()).await;
        let ctx = ctx_ip("7.7.7.7");
        let mut bool_results = Vec::new();
        let mut detailed_results = Vec::new();
        for _ in 0..4 {
            bool_results.push(adapter_bool.check_rate_limit(&ctx).await);
            detailed_results.push(
                adapter_detailed
                    .check_rate_limit_detailed(&ctx)
                    .await
                    .allowed,
            );
        }
        assert_eq!(bool_results, detailed_results);
    }
}
