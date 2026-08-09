# 语义缓存示例

演示 `SemanticCache` — 在精确匹配缓存之上添加 trigram Jaccard 文本相似度检查。

## 示例

| 名称 | 说明 |
|------|------|
| `semantic_cache_demo` | 三级查询策略：精确匹配 → trigram 语义搜索 → 模型推理回填 |

## 运行

```bash
cargo run -p vecboost-examples --bin semantic_cache_demo
```

## 说明

`SemanticCache` 是 VecBoost 的性能优化层，在模型推理前插入零开销的文本相似度检查：

1. **精确匹配**：与 `OxCacheBackend` 精确 key 匹配
2. **语义搜索**：trigram Jaccard 相似度 > threshold 时复用缓存
3. **推理回填**：miss 后调用模型推理并回填到两级缓存

核心 API：
- `SemanticCache::with_capacity(threshold, capacity)` — 便捷构造
- `cache.get_or_compute(text, compute_fn)` — 三级查询
- `cache.stats()` — 统计信息（精确命中/语义命中/miss）
