# Rate Limiting 示例

VecBoost 使用 limiteron 提供多维度令牌桶限流（全局 / IP / 用户 / API Key）。

## 标识提取链（Governor 全局维度归因）

Governor 全局检查按序提取请求标识：`RequestContext.user_id`（服务端认证态，
权威性最高）→ `X-User-Id` 头/query → 客户端 IP → `X-API-Key` 头。
仅带 `user_id` 的上下文（无 IP、无头）按用户维度归因；全链皆空时全局检查
失败并被兜底拒绝。IP/用户/API Key 的 per-key 独立计数由 vecboost 侧
TokenBucketLimiter 实现，任一维度拒绝即整体拒绝。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|--------------|
| `rate_limit` | 默认配置限流，观察 remaining 递减 | 无（limiteron 为主 crate 必选依赖） |
| `multi_dimension` | 自定义小限额，演示多维度独立限流 | 无 |

## 运行

```bash
# 注意：vecboost-examples 不定义 limiteron/http feature，
# 加 --features limiteron,http 会被 cargo 拒绝（does not contain these features）
cargo run -p vecboost-examples --bin rate_limit
cargo run -p vecboost-examples --bin multi_dimension
```
