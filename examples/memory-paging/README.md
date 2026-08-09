# GPU 显存分页管理示例

演示 `WeightPagingManager` — 基于 LRU-K 策略的模型权重分层管理。

## 示例

| 名称 | 说明 |
|------|------|
| `memory_paging` | 层换入换出、LRU-K 驱逐、预取优化、显存预算控制 |

## 运行

```bash
cargo run -p vecboost-examples --bin memory_paging
```

## 说明

`WeightPagingManager` 管理模型权重在 GPU/CPU 之间的分层调度：

- **热层常驻 GPU**：频繁访问的层保持在 GPU 显存
- **冷层按需换入**：不活跃的层换出到 CPU 内存
- **LRU-K 驱逐**：基于第 K 次最近访问时间选择驱逐候选
- **预取优化**：根据当前推理层预测后续层，提前标记换入

核心 API：
- `WeightPagingManager::new(config)` — 创建分页管理器
- `manager.register_layer(name, size)` — 注册模型层
- `manager.page_in(name)` / `page_out(name)` — 层换入/换出
- `manager.record_access(name)` — 记录访问（影响 LRU-K）
- `manager.get_prefetch_list(current, order)` — 获取预取列表
- `manager.stats()` — 分页统计
