# Cache 示例

VecBoost 使用 oxcache 作为嵌入向量的缓存后端，避免对相同文本重复推理。

> 注意：缓存由 `EmbeddingService` 内部管理（`cache` 模块为 `pub(crate)`），
> 不对外暴露 `CacheStats`，`EmbedResponse.processing_time_ms` 恒为 0。
> 示例通过 `CountingEngine`（`AtomicU64` 计数器）+ 外部计时验证缓存命中。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|--------------|
| `cache_config` | 对比缓存启用/禁用时的引擎调用次数 | 无（oxcache 为主 crate 必选依赖） |
| `ttl` | 验证相同文本第二次调用命中缓存（引擎不被调用） | 无 |

## 运行

```bash
# 注意：vecboost-examples 不定义 oxcache/http feature，
# 加 --features oxcache,http 会被 cargo 拒绝（does not contain these features）
cargo run -p vecboost-examples --bin cache_config
cargo run -p vecboost-examples --bin ttl
```
