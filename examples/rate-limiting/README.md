# Rate Limiting 示例

VecBoost 使用 limiteron 提供多维度令牌桶限流（全局 / IP / 用户 / API Key）。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|--------------|
| `rate_limit` | 默认配置限流，观察 remaining 递减 | `limiteron, http` |
| `multi_dimension` | 自定义小限额，演示多维度独立限流 | `limiteron, http` |

## 运行

```bash
cargo run -p vecboost-examples --bin rate_limit --features limiteron,http
cargo run -p vecboost-examples --bin multi_dimension --features limiteron,http
```
