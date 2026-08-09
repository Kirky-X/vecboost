# Semantic Cache Baseline Results

## Exact Cache Operations

| Operation | Time |
|-----------|------|
| HashMap hit (1000 entries) | ~7.5 ns |
| HashMap miss | ~8.0 ns |

## Trigram Jaccard Brute-Force Search

| Index Size | Time |
|------------|------|
| 100 entries | 84.9 µs |
| 1,000 entries | 872.6 µs |
| 10,000 entries | 8.96 ms |

## Analysis

- 精确缓存操作延迟在纳秒级（HashMap O(1)）
- Trigram 暴力搜索在 10K 条目时 < 10ms，完全可接受
- 搜索延迟与条目数线性增长（O(n)），符合预期
- 10K 条目覆盖大多数生产场景的缓存规模
