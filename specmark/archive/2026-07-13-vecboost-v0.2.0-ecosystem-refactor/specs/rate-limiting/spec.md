# Spec — rate-limiting

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 limiteron 限流能力域需求。

## Requirements

### R-rate-limiting-001: limiteron adapter 实现

`src/rate_limit/limiteron_adapter.rs` 实现 `pub(crate) struct LimiteronAdapter` 包装 `limiteron::Governor`,实现 `crate::rate_limit::RateLimiter` trait。启用 limiteron `standard` feature(postgres + ban-manager + quota-control + circuit-breaker)。

**验收标准:**
- `LimiteronAdapter::check(dim, key)` 返回 `RateLimitStatus::Allowed` 或 `RateLimitStatus::Denied`
- 令牌桶配置:10 令牌/秒,11 个请求第 11 个返回 `Denied`
- 维度支持:Global/IP/User/ApiKey

### R-rate-limiting-002: 封禁管理

启用 limiteron `ban-manager` feature,支持 IP/User/MAC 封禁。封禁规则从 `[flow_control].ban_rules` 配置加载,支持 YAML 文件热重载。

**验收标准:**
- `ban(ip="1.2.3.4", duration=3600)` 后,该 IP 的请求在 1 小时内返回 `RateLimitStatus::Banned`
- 自动封禁:1 分钟内 100 次失败请求触发 IP 自动封禁 1 小时
- 封禁列表 YAML 热重载:修改文件后 5 秒内生效

### R-rate-limiting-003: 熔断器

启用 limiteron `circuit-breaker` feature,连续 N 次失败触发熔断,熔断期间请求快速失败。熔断恢复后自动放行。

**验收标准:**
- 连续 5 次下游错误(500)触发熔断,熔断期间请求返回 `RateLimitStatus::CircuitOpen`
- 熔断 30 秒后进入半开状态,允许 1 个探测请求
- 探测成功后关闭熔断,恢复正常流量
- 探测失败重新打开熔断

### R-rate-limiting-004: Tower 中间件

启用 limiteron `tower-middleware` feature,提供 Axum 中间件 `RateLimitLayer`,自动提取 IP/UserId/ApiKey 并调用 `Governor`。

**验收标准:**
- 中间件从 `X-Forwarded-For` 提取 IP(无则用 socket addr)
- 中间件从 `Authorization: Bearer <jwt>` 提取 UserId
- 中间件从 `X-API-Key` 提取 ApiKey
- 超出限制返回 HTTP 429 + `Retry-After` 头

## Constraints

- **v0.2.0 实际 `default = ["http", "oxcache", "limiteron"]`,limiteron 默认启用**(原 spec 写"非 default",已偏离)
- 删除全部自研限流文件(limiter/store/token_bucket/redis_store)
- `RateLimiter` trait + `RateLimitConfig`/`RateLimitDimension`/`RateLimitStatus` 保留为公共接口
- postgres 持久化通过 `postgres` feature 启用(依赖 `db` feature)

## Out of Scope

- 不启用 limiteron OpenTelemetry 追踪(`telemetry` feature 留待下个 change)
- 不启用 limiteron Admin REST API(`admin-api` feature 不启用)
- 不启用 limiteron 地理位置匹配(`geo-matching` feature 不启用)
- 不启用 limiteron 多租户(`multi-tenant` feature 不启用)
