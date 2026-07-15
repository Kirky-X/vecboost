# Pipeline 示例

VecBoost 请求流水线模块示例,演示优先级队列与 Worker 管理器的使用。

## 示例列表

| 示例 | 说明 | 所需 feature |
|------|------|-------------|
| `priority_queue` | 优先级队列入队与出队,验证按 Critical → Low 顺序出队 | `http` |
| `workers` | WorkerManager 启动、手动扩容、优雅关闭 | `http` |

## 运行命令

```bash
# 优先级队列示例
cargo run -p vecboost-examples --bin priority_queue --features http

# Worker 管理器示例
cargo run -p vecboost-examples --bin workers --features http
```

## 依赖说明

- 使用 `MockEngine` 返回固定维度的零向量,不依赖真实模型文件
- `PriorityRequestQueue` 基于优先级 BTreeMap,支持 Critical/High/Normal/Low 四级
- `WorkerManager` 支持自动扩缩容(基于队列压力阈值)和优雅关闭
