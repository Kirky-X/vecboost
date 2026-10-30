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

#[cfg(feature = "distributed-redis")]
use limiteron::CacheBanStorage;
use limiteron::Governor;
use limiteron::limiters::{Limiter, TokenBucketLimiter};
use limiteron::matchers::{
    ApiKeyExtractor, CompositeExtractor, Identifier, IdentifierExtractor, IpExtractor,
    RequestContext, UserIdExtractor,
};
use limiteron::middleware::RateLimitHeaderValues;
use limiteron::storage::{MemoryBanStorage, MemoryStorage};
use tokio::sync::Mutex;

use crate::config::app::RateLimitBackend;
use crate::error::VecboostError;

/// 把 `RequestContext.user_id` 字段纳入 Governor 标识提取链。
///
/// limiteron 默认提取链为 UserIdExtractor(X-User-Id header/query/默认值) →
/// IP → ApiKey，不读 `user_id` 字段：仅带 `user_id` 的已认证上下文（无 IP、
/// 无头）在全局检查期提取失败，被 adapter 兜底为静默拒绝（无快照），
/// user 维度桶永远不可达。本提取器置于链首位，使已认证上下文可被全局
/// 维度归因；`user_id` 来自服务端认证态，其权威性高于可伪造的请求头，
/// 故优先于 header 提取。
struct ContextUserIdExtractor;

impl IdentifierExtractor for ContextUserIdExtractor {
    fn extract(&self, context: &RequestContext) -> Option<Identifier> {
        context
            .user_id
            .as_ref()
            .map(|u| Identifier::UserId(u.clone()))
    }

    fn name(&self) -> &str {
        "vecboost-context-user-id"
    }
}

/// 复刻 limiteron 默认提取链，并在链首加入 [`ContextUserIdExtractor`]：
/// `context.user_id` → X-User-Id header/query → IP → X-API-Key。
fn build_identifier_extractor() -> Arc<dyn IdentifierExtractor> {
    Arc::new(
        CompositeExtractor::builder()
            .add_extractor(Box::new(ContextUserIdExtractor))
            .add_extractor(Box::new(UserIdExtractor::from_header("X-User-Id")))
            .add_extractor(Box::new(IpExtractor::builder().build()))
            .add_extractor(Box::new(ApiKeyExtractor::from_header("X-API-Key")))
            .build(),
    )
}

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
    /// 存储后端：memory（缺省）或 redis（跨副本共享）
    pub backend: RateLimitBackend,
    /// Redis 连接 URL（backend=redis 必填）
    pub redis_url: Option<String>,
    /// Redis 固定窗口毫秒数（默认 60000）
    pub global_window_ms: u64,
}

impl Default for RateLimitSettings {
    fn default() -> Self {
        Self {
            global_requests_per_minute: 1000,
            ip_requests_per_minute: 100,
            user_requests_per_minute: 200,
            api_key_requests_per_minute: 500,
            backend: RateLimitBackend::Memory,
            redis_url: None,
            global_window_ms: 60_000,
        }
    }
}

/// per-key 桶工厂：内存令牌桶（缺省）或 Redis 固定窗口（跨副本）。
pub trait BucketFactory: Send + Sync {
    fn create(&self, key: &str, capacity: u64, refill_per_sec: u64) -> Arc<dyn Limiter>;
}

struct MemoryBucketFactory;

impl BucketFactory for MemoryBucketFactory {
    fn create(&self, _key: &str, capacity: u64, refill_per_sec: u64) -> Arc<dyn Limiter> {
        Arc::new(TokenBucketLimiter::new(capacity, refill_per_sec))
    }
}

/// Redis 固定窗口限流器（Lua 原子）：
/// `INCR key`，首次窗口设置 `PEXPIRE`，返回 `{count, pttl}`。
/// 全局与 per-key 共用同一实现与同一 oxcache Redis Cache。
#[cfg(feature = "distributed-redis")]
pub struct RedisWindowLimiter {
    cache: Arc<oxcache::Cache<String, String>>,
    key: String,
    limit: u64,
    window_ms: u64,
}

#[cfg(feature = "distributed-redis")]
const REDIS_WINDOW_LUA: &str = r#"
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('PTTL', KEYS[1])
if ttl == -1 then
  -- externally-written key without TTL: re-arm the window so the
  -- dimension cannot be locked out forever by a TTL-less key
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {current, ttl}
"#;

/// 单次 Redis Lua 往返的有界时长：oxcache `eval_lua` 直连无响应超时
/// 且不经熔断器，Redis 半死（接受连接不响应）时若无此界，请求处理
/// 线程将挂起耗尽。超时与 Redis 错误同为 fail-closed 拒绝。
#[cfg(feature = "distributed-redis")]
const EVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// per-key 固定窗口宽度（与「每分钟配额」配置语义一致，独立于
/// `global_window_ms`——后者仅作用于全局维度，防止运维调小全局窗口
/// 时按比例静默放宽 per-key 配额）。
#[cfg(feature = "distributed-redis")]
const PER_KEY_WINDOW_MS: u64 = 60_000;

#[cfg(feature = "distributed-redis")]
impl RedisWindowLimiter {
    pub fn new(
        cache: Arc<oxcache::Cache<String, String>>,
        key: impl Into<String>,
        limit: u64,
        window_ms: u64,
    ) -> Self {
        Self {
            cache,
            key: key.into(),
            limit: limit.max(1),
            window_ms: window_ms.max(1),
        }
    }

    fn script_error(op: &str, detail: impl std::fmt::Display) -> limiteron::error::LimiteronError {
        limiteron::error::LimiteronError::Other(crate::i18n::tr_with_args(
            "rate-limit-redis-window-op-failed",
            crate::i18n::tr_args(&[("op", op), ("detail", &detail.to_string())]),
        ))
    }

    fn parse_count_ttl(
        value: oxcache::redis::Value,
    ) -> Result<(u64, i64), limiteron::error::LimiteronError> {
        let items = match value {
            oxcache::redis::Value::Array(items) => items,
            other => {
                return Err(Self::script_error(
                    "parse",
                    crate::i18n::tr_with_args(
                        "rate-limit-redis-unexpected-reply",
                        crate::i18n::tr_args(&[("detail", &format!("{other:?}"))]),
                    ),
                ));
            }
        };
        let count = match items.first() {
            Some(oxcache::redis::Value::Int(v)) => (*v).max(0) as u64,
            other => {
                return Err(Self::script_error(
                    "count",
                    crate::i18n::tr_with_args(
                        "rate-limit-redis-unexpected-value",
                        crate::i18n::tr_args(&[("detail", &format!("{other:?}"))]),
                    ),
                ));
            }
        };
        let ttl = match items.get(1) {
            Some(oxcache::redis::Value::Int(v)) => *v,
            other => {
                return Err(Self::script_error(
                    "ttl",
                    crate::i18n::tr_with_args(
                        "rate-limit-redis-unexpected-value",
                        crate::i18n::tr_args(&[("detail", &format!("{other:?}"))]),
                    ),
                ));
            }
        };
        Ok((count, ttl))
    }
}

#[cfg(feature = "distributed-redis")]
#[async_trait::async_trait]
impl Limiter for RedisWindowLimiter {
    async fn allow(&self, cost: u64) -> Result<bool, limiteron::error::LimiteronError> {
        let window_ms = self.window_ms.to_string();
        let eval = tokio::time::timeout(
            EVAL_TIMEOUT,
            self.cache.eval_lua(
                REDIS_WINDOW_LUA,
                &[self.key.as_str()],
                &[window_ms.as_str()],
            ),
        )
        .await;
        let value = match eval {
            Ok(Ok(value)) => value,
            Ok(Err(e)) => return Err(RedisWindowLimiter::script_error("eval", e)),
            Err(_) => {
                return Err(RedisWindowLimiter::script_error(
                    "eval",
                    crate::i18n::tr_with_args(
                        "rate-limit-redis-eval-timeout",
                        crate::i18n::tr_args(&[("timeout", &format!("{EVAL_TIMEOUT:?}"))]),
                    ),
                ));
            }
        };
        let (count, _ttl) = RedisWindowLimiter::parse_count_ttl(value)?;
        Ok(count.saturating_add(cost.saturating_sub(1)) <= self.limit)
    }
}

/// Redis per-key 工厂：与全局窗口共享同一个 Cache，窗口固定 60s。
#[cfg(feature = "distributed-redis")]
struct RedisWindowFactory {
    cache: Arc<oxcache::Cache<String, String>>,
}

#[cfg(feature = "distributed-redis")]
impl BucketFactory for RedisWindowFactory {
    fn create(&self, key: &str, capacity: u64, _refill_per_sec: u64) -> Arc<dyn Limiter> {
        Arc::new(RedisWindowLimiter::new(
            self.cache.clone(),
            format!("vecboost:rl:{key}"),
            capacity,
            PER_KEY_WINDOW_MS,
        ))
    }
}

/// limiteron 混合限流后端。
///
/// Governor 负责全局限流 + 封禁/熔断/统计；per-key TokenBucketLimiter
/// 负责维度隔离（每个 IP/User/ApiKey 独立限额）。
pub struct LimiteronAdapter {
    governor: Governor,
    #[cfg(feature = "distributed-redis")]
    global_redis: Option<Arc<RedisWindowLimiter>>,
    bucket_factory: Arc<dyn BucketFactory>,
    ip_buckets: Mutex<HashMap<String, Arc<dyn Limiter>>>,
    user_buckets: Mutex<HashMap<String, Arc<dyn Limiter>>>,
    api_key_buckets: Mutex<HashMap<String, Arc<dyn Limiter>>>,
    settings: RateLimitSettings,
    /// 缓存的健康状态（由异步检查更新，同步方法读取）
    health_cache: AtomicBool,
}

impl LimiteronAdapter {
    /// 使用指定配置创建内存限流后端（保持既有调用面）。
    pub async fn new(settings: RateLimitSettings) -> Self {
        Self::build_memory(settings).await
    }

    async fn build_memory(settings: RateLimitSettings) -> Self {
        #[cfg(feature = "distributed-redis")]
        {
            Self::build(settings, None, Arc::new(MemoryBucketFactory), None).await
        }
        #[cfg(not(feature = "distributed-redis"))]
        {
            Self::build(settings, Arc::new(MemoryBucketFactory)).await
        }
    }

    /// 按 `settings.backend` 创建限流后端：redis 初始化失败显性返回错误。
    pub async fn try_new(settings: RateLimitSettings) -> Result<Self, VecboostError> {
        match settings.backend {
            RateLimitBackend::Memory => Ok(Self::build_memory(settings).await),
            RateLimitBackend::Redis => {
                #[cfg(feature = "distributed-redis")]
                {
                    let url = settings
                        .redis_url
                        .as_deref()
                        .map(str::trim)
                        .filter(|url| !url.is_empty())
                        .ok_or_else(|| {
                            VecboostError::config_error(crate::i18n::tr(
                                "startup-rate-limit-redis-url-missing",
                            ))
                        })?;
                    let backend = oxcache::backend::RedisBackend::builder()
                        .connection_string(url)
                        .build()
                        .await
                        .map_err(|e| {
                            VecboostError::config_error(crate::i18n::tr_with_args(
                                "startup-rate-limit-redis-init-failed",
                                crate::i18n::tr_args(&[("detail", &e.to_string())]),
                            ))
                        })?;
                    let cache = Arc::new(oxcache::Cache::<String, String>::with_dependencies(
                        Arc::new(backend),
                    ));
                    let global_redis = if settings.global_requests_per_minute > 0 {
                        let capacity = u64::try_from(
                            (settings.global_requests_per_minute as u128
                                * settings.global_window_ms as u128)
                                .div_ceil(60_000),
                        )
                        .map_err(|_| {
                            VecboostError::config_error(
                                "rate-limit-global-window-capacity-overflow".to_string(),
                            )
                        })?
                        .max(1);
                        Some(Arc::new(RedisWindowLimiter::new(
                            cache.clone(),
                            "vecboost:rl:global",
                            capacity,
                            settings.global_window_ms,
                        )))
                    } else {
                        None
                    };
                    let factory = Arc::new(RedisWindowFactory { cache });
                    // 封禁清单跨副本共享：独立 Redis 连接 + CacheBanStorage。
                    let ban_backend = oxcache::backend::RedisBackend::builder()
                        .connection_string(url)
                        .build()
                        .await
                        .map_err(|e| {
                            VecboostError::config_error(crate::i18n::tr_with_args(
                                "startup-rate-limit-redis-ban-init-failed",
                                crate::i18n::tr_args(&[("detail", &e.to_string())]),
                            ))
                        })?;
                    let ban_backend: Arc<dyn oxcache::backend::CacheBackend> =
                        Arc::new(ban_backend);
                    let ban_storage: Arc<dyn limiteron::storage::BanStorage> =
                        Arc::new(CacheBanStorage::new(ban_backend));
                    Ok(Self::build(settings, global_redis, factory, Some(ban_storage)).await)
                }
                #[cfg(not(feature = "distributed-redis"))]
                {
                    Err(VecboostError::config_error(crate::i18n::tr(
                        "startup-rate-limit-redis-feature-missing",
                    )))
                }
            }
        }
    }

    async fn build(
        settings: RateLimitSettings,
        #[cfg(feature = "distributed-redis")] global_redis: Option<Arc<RedisWindowLimiter>>,
        bucket_factory: Arc<dyn BucketFactory>,
        #[cfg(feature = "distributed-redis")] ban_storage: Option<
            Arc<dyn limiteron::storage::BanStorage>,
        >,
    ) -> Self {
        let config = build_global_flow_control_config(&settings);
        let storage: Arc<dyn limiteron::storage::Storage> = MemoryStorage::create_storage();
        #[cfg(feature = "distributed-redis")]
        let ban_storage: Arc<dyn limiteron::storage::BanStorage> =
            ban_storage.unwrap_or_else(MemoryBanStorage::create_ban_storage);
        #[cfg(not(feature = "distributed-redis"))]
        let ban_storage: Arc<dyn limiteron::storage::BanStorage> =
            MemoryBanStorage::create_ban_storage();

        let governor = Governor::builder()
            .with_config(config)
            .with_storage(storage)
            .with_ban_storage(ban_storage)
            .with_l1_cache_enabled(false)
            .with_identifier_extractor(build_identifier_extractor())
            .build()
            .await
            .expect("Governor build with valid config should succeed");

        Self {
            governor,
            #[cfg(feature = "distributed-redis")]
            global_redis,
            bucket_factory,
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
        // 0. Redis 全局固定窗口（跨副本共享；memory 后端为 None）
        #[cfg(feature = "distributed-redis")]
        if let Some(global) = &self.global_redis {
            match global.allow(1).await {
                Ok(true) => {}
                Ok(false) => {
                    let window_secs = (self.settings.global_window_ms / 1000).max(1);
                    return RateLimitDecision::rejected(Some(RateLimitHeaderValues {
                        limit: global.limit,
                        remaining: 0,
                        reset_at: unix_now() + window_secs,
                        retry_after: Some(window_secs),
                        policy: "global-redis".to_string(),
                    }));
                }
                Err(_) => return RateLimitDecision::rejected(None),
            }
        }

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
    /// 非消费快照（`remaining`）供响应头渲染。
    async fn check_per_key_detailed(
        &self,
        buckets: &Mutex<HashMap<String, Arc<dyn Limiter>>>,
        key: &str,
        rpm: u64,
        dimension: &str,
    ) -> KeyOutcome {
        let bucket = {
            let mut map = buckets.lock().await;
            let composite = format!("{dimension}:{key}");
            map.entry(composite.clone())
                .or_insert_with(|| {
                    let refill_rate = (rpm / 60).max(1);
                    self.bucket_factory.create(&composite, rpm, refill_rate)
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
            ..RateLimitSettings::default()
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

    // 全局限流测试

    #[tokio::test]
    async fn test_global_limit_blocks_when_exceeded() {
        let settings = RateLimitSettings {
            global_requests_per_minute: 3,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 0,
            api_key_requests_per_minute: 0,
            ..RateLimitSettings::default()
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let context = ctx_ip("1.2.3.4");
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(adapter.check_rate_limit(&context).await);
        assert!(!adapter.check_rate_limit(&context).await);
    }

    // per-key IP 限流测试

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

    // per-key User 限流测试

    #[tokio::test]
    async fn test_user_dimension_independent_from_ip() {
        // 禁用 IP 维度，只测 User per-key 独立性
        let settings = RateLimitSettings {
            global_requests_per_minute: 100,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 4,
            api_key_requests_per_minute: 0,
            ..RateLimitSettings::default()
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

    #[tokio::test]
    async fn test_user_only_context_without_ip_is_allowed() {
        // 回归：limiteron 默认提取链不读 user_id 字段，user-only 上下文在
        // Governor 全局检查期提取失败被兜底静默拒绝——ContextUserIdExtractor
        // 修复后应命中 user 维度桶（文档口径：无 IP 亦可按用户独立计数）。
        // 桶耗尽语义由 test_user_dimension_independent_from_ip 覆盖（带 IP
        // 路径同构）；此处不断言耗尽——refill_rate=max(rpm/60,1)=1 token/s，
        // 跨秒边界的耗尽断言会竞态。首调快照紧跟消费取值，remaining 确定。
        let settings = RateLimitSettings {
            global_requests_per_minute: 100,
            ip_requests_per_minute: 5,
            user_requests_per_minute: 3,
            api_key_requests_per_minute: 0,
            ..RateLimitSettings::default()
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let ctx = RequestContext {
            user_id: Some("alice".to_string()),
            path: "/test".to_string(),
            method: "GET".to_string(),
            ..Default::default()
        };

        let first = adapter.check_rate_limit_detailed(&ctx).await;
        assert!(first.allowed, "user-only 上下文不应被全局检查拒绝");
        let headers = first.headers.expect("allowed 决策应携带绑定维度快照");
        assert_eq!(headers.policy, "user-token-bucket", "绑定维度应为 user 桶");
        assert_eq!(headers.limit, 3, "绑定维度应为 user 配额");
        assert_eq!(headers.remaining, 2, "消费 1 个令牌后剩余 2");

        // 同键后续请求仍放行（桶按 user 持久化，不逐请求重建）
        assert!(adapter.check_rate_limit(&ctx).await);

        // 其他 user-only 用户有独立配额
        let bob = RequestContext {
            user_id: Some("bob".to_string()),
            ..ctx.clone()
        };
        let bob_first = adapter.check_rate_limit_detailed(&bob).await;
        assert!(bob_first.allowed, "bob 应有独立 user 桶");
        assert_eq!(
            bob_first.headers.expect("bob 快照").policy,
            "user-token-bucket"
        );
    }

    #[tokio::test]
    async fn test_user_id_field_takes_precedence_over_header() {
        // context.user_id 来自服务端认证态，权威性高于可伪造的 X-User-Id 头：
        // 两者并存时应归因到认证态用户，而非头中声明值
        let settings = RateLimitSettings {
            global_requests_per_minute: 100,
            ip_requests_per_minute: 0,
            user_requests_per_minute: 2,
            api_key_requests_per_minute: 0,
            ..RateLimitSettings::default()
        };
        let adapter = LimiteronAdapter::new(settings).await;
        let mut ctx = RequestContext {
            user_id: Some("alice".to_string()),
            path: "/test".to_string(),
            method: "GET".to_string(),
            ..Default::default()
        };
        ctx.headers
            .insert("X-User-Id".to_string(), "mallory".to_string());

        // alice 桶限额 2：两次后耗尽——若归因到头值 mallory，alice 不会耗尽
        assert!(adapter.check_rate_limit(&ctx).await);
        assert!(adapter.check_rate_limit(&ctx).await);
        assert!(
            !adapter.check_rate_limit(&ctx).await,
            "认证态 user_id 应主导归因（alice 桶耗尽即拒绝）"
        );

        // mallory 有独立配额
        let mallory_ctx = RequestContext {
            user_id: Some("mallory".to_string()),
            ..ctx.clone()
        };
        assert!(adapter.check_rate_limit(&mallory_ctx).await);
    }

    // 健康检查测试

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

    // 统计测试

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

    // RateLimit-* 响应头详情测试

    #[tokio::test]
    async fn test_detailed_decision_carries_binding_snapshot() {
        // 只启用 IP 维度，绑定维度必为 ip-token-bucket
        let settings = RateLimitSettings {
            global_requests_per_minute: 0,
            ip_requests_per_minute: 3,
            user_requests_per_minute: 0,
            api_key_requests_per_minute: 0,
            ..RateLimitSettings::default()
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

    // ===== 以下用例靠“造假应答”驱动，集成测试结构性跑不到它们 =====

    fn hdr(limit: u64, remaining: u64, policy: &str) -> RateLimitHeaderValues {
        RateLimitHeaderValues {
            limit,
            remaining,
            reset_at: 1_800_000_000,
            retry_after: None,
            policy: policy.to_string(),
        }
    }

    /// 响应头必须取“剩余额度最紧”的那个维度，且不能因传入顺序而飘。
    #[test]
    fn tighter_covers_all_four_arms_and_is_order_independent() {
        assert!(tighter(None, None).is_none(), "双 None 必须返回 None");
        assert_eq!(
            tighter(None, Some(hdr(10, 7, "user")))
                .expect("(None, b) 必须取 b")
                .remaining,
            7
        );
        assert_eq!(
            tighter(Some(hdr(10, 3, "ip")), None)
                .expect("(a, None) 必须取 a")
                .remaining,
            3
        );
        // 两侧 Some：取 remaining 更小者，且左右顺序不得影响结果
        assert_eq!(
            tighter(Some(hdr(10, 3, "ip")), Some(hdr(10, 7, "user")))
                .expect("ab")
                .policy,
            "ip"
        );
        assert_eq!(
            tighter(Some(hdr(10, 7, "user")), Some(hdr(10, 3, "ip")))
                .expect("ba")
                .policy,
            "ip",
            "维度选择不得随传入顺序变化，否则同一请求重试会拿到不同策略名"
        );
        // 并列时保留左侧（稳定选择）
        assert_eq!(
            tighter(Some(hdr(10, 3, "ip")), Some(hdr(10, 3, "api-key")))
                .expect("tie")
                .policy,
            "ip"
        );
    }

    #[cfg(feature = "distributed-redis")]
    fn rints(values: &[i64]) -> oxcache::redis::Value {
        oxcache::redis::Value::Array(
            values
                .iter()
                .map(|v| oxcache::redis::Value::Int(*v))
                .collect(),
        )
    }

    /// 正常应答：(count, ttl) 原值透传。
    #[cfg(feature = "distributed-redis")]
    #[test]
    fn parse_count_ttl_accepts_int_pair() {
        let (count, ttl) =
            RedisWindowLimiter::parse_count_ttl(rints(&[3, 60_000])).expect("合法应答不得报错");
        assert_eq!((count, ttl), (3, 60_000));
    }

    /// 负计数必须被夹到 0：count 会被后续 `saturating_add` 参与额度判定，
    /// 若直接 `as u64` 回绕成巨值，该维度会永久拒绝所有请求。
    #[cfg(feature = "distributed-redis")]
    #[test]
    fn parse_count_ttl_clamps_negative_count_without_wraparound() {
        let (count, ttl) = RedisWindowLimiter::parse_count_ttl(rints(&[-7, 100]))
            .expect("负 count 应被夹住而非报错");
        assert_eq!(count, 0, "负计数不得回绕成 u64 巨值");
        assert_eq!(ttl, 100, "ttl 保留原值供上层诊断");
    }

    /// 非数组应答（如误接 REDIS_ERR / Nil）必须报错，不能当“未超限”放行。
    #[cfg(feature = "distributed-redis")]
    #[test]
    fn parse_count_ttl_rejects_non_array_reply() {
        crate::i18n::init();
        let err = RedisWindowLimiter::parse_count_ttl(oxcache::redis::Value::Nil)
            .expect_err("非数组应答必须显性失败");
        let expected = crate::i18n::tr_with_args(
            "rate-limit-redis-window-op-failed",
            crate::i18n::tr_args(&[("op", "parse"), ("detail", "")]),
        );
        assert!(
            err.to_string().contains(&expected),
            "错误必须点名解析阶段：{err}"
        );
    }

    /// count 位不是整数（或整个缺失）时必须报 count 阶段错误。
    #[cfg(feature = "distributed-redis")]
    #[test]
    fn parse_count_ttl_rejects_non_integer_count() {
        crate::i18n::init();
        let err = RedisWindowLimiter::parse_count_ttl(oxcache::redis::Value::Array(vec![
            oxcache::redis::Value::BulkString(b"nan".to_vec()),
            oxcache::redis::Value::Int(1),
        ]))
        .expect_err("非整数 count 必须拒绝");
        let expected = crate::i18n::tr_with_args(
            "rate-limit-redis-window-op-failed",
            crate::i18n::tr_args(&[("op", "count"), ("detail", "")]),
        );
        assert!(
            err.to_string().contains(&expected),
            "错误必须点名 count 阶段：{err}"
        );

        let err2 = RedisWindowLimiter::parse_count_ttl(oxcache::redis::Value::Array(vec![]))
            .expect_err("空数组必须拒绝");
        assert!(
            err2.to_string().contains(&expected),
            "空应答也应归为 count 阶段：{err2}"
        );
    }

    /// ttl 位缺失或非整数时报 ttl 阶段错误（与 count 阶段分开，运维能定位脚本返回形状）。
    #[cfg(feature = "distributed-redis")]
    #[test]
    fn parse_count_ttl_rejects_missing_or_non_integer_ttl() {
        crate::i18n::init();
        let err = RedisWindowLimiter::parse_count_ttl(rints(&[1]))
            .expect_err("只带 count 的应答必须拒绝");
        let expected = crate::i18n::tr_with_args(
            "rate-limit-redis-window-op-failed",
            crate::i18n::tr_args(&[("op", "ttl"), ("detail", "")]),
        );
        assert!(
            err.to_string().contains(&expected),
            "错误必须点名 ttl 阶段：{err}"
        );

        let err2 = RedisWindowLimiter::parse_count_ttl(oxcache::redis::Value::Array(vec![
            oxcache::redis::Value::Int(1),
            oxcache::redis::Value::Nil,
        ]))
        .expect_err("非整数 ttl 必须拒绝");
        assert!(err2.to_string().contains(&expected), "实际：{err2}");
    }
}
