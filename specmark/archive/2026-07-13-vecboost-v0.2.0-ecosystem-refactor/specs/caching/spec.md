# Spec — caching

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 oxcache 缓存能力域需求。

## Requirements

### R-caching-001: oxcache backend 实现

`src/cache/oxcache_backend.rs` 实现 `pub(crate) struct OxCacheBackend` 包装 `oxcache::Cache`,实现 `crate::cache::Cache<K, V>` trait。启用 oxcache `core` feature(L1 Moka 内存 + L2 Redis)。

**验收标准:**
- `OxCacheBackend::put(key, value, size)` + `get(key)` 返回命中值
- L1 命中时延迟 < 1μs
- L2 启用时(redis feature),L1 miss 后查询 L2
- `clear()` 清空 L1 和 L2

### R-caching-002: LRU 驱逐策略

L1 缓存容量达到上限时,按 LRU 策略驱逐最久未访问的条目。容量从 `[cache].l1.max_capacity` 读取,默认 10000。

**验收标准:**
- 容量 3 时,插入 4 个 key 后最久未访问的被驱逐
- `get(key)` 更新访问时间,延长该 key 生命周期
- 驱逐事件记录到 `CacheStats.evictions`

### R-caching-003: TTL 过期

每个缓存条目可设置独立 TTL,过期后自动失效。TTL 从 `put(key, value, size, Some(ttl_secs))` 传入,或使用全局默认 `[cache].ttl`。

**v0.2.0 实际实施状态(部分实现,全局 TTL 默认值推迟)**:
- `OxCacheBackend::put` 签名当前为 `put(key, value, size)`,**未含 TTL 参数**
- per-entry TTL 传入接口推迟到 v0.3.0
- 全局默认 `[cache].ttl` 配置项推迟到 v0.3.0

**验收标准(按实际代码):**
- 当前 `OxCacheBackend::put(key, value, size)` 不支持 per-entry TTL
- 当前不支持全局 TTL 默认值

**推迟到 v0.3.0:**
- `put(key, value, 100, Some(2))` 后 `sleep(3s)`,`get(key)` 返回 `None`
- `put(key, value, 100, None)` 使用全局 TTL(默认 3600s)
- 全局 TTL = 0 表示永不过期

### R-caching-004: 多实例同步(可选)

启用 `oxcache/redis` 时,L1 失效通过 Redis Pub/Sub 广播到其他实例,其他实例收到消息后清除本地 L1。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- 无 Redis Pub/Sub 集成
- 无多实例 L1 失效广播机制
- `redis` feature 存在但仅作为依赖声明

**验收标准(目标,v0.3.0 实现):**
- 实例 A `put(key, value)` 后,实例 B `get(key)` 在 1 秒内命中 L2
- 实例 A `remove(key)` 后,实例 B 的 L1 中该 key 在 1 秒内被清除
- Redis 断连时不影响单实例 L1 操作

## Constraints

- **v0.2.0 实际 `default = ["http", "oxcache", "limiteron"]`,oxcache 默认启用**(原 spec 写"非 default",已偏离)
- 删除全部自研缓存文件(arc_cache/lfu_cache/lru_cache/kv_cache/bloom_filter/tiered_cache)
- `Cache` trait 保留作为公共接口,仅 backend 替换
- 保留 `CacheStats`/`CacheConfig`/`CacheStrategy` 数据结构(改为 `pub(crate)` 字段)

## Out of Scope

- 不启用 oxcache BloomFilter(本轮仅 L1+L2)
- 不启用 oxcache 压缩(compression feature 留待下个 change)
- 不实现缓存预热 API(本轮仅运行时缓存)
