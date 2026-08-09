# 审计日志示例

演示 `AuditLogger` — 异步批量写入的安全事件审计日志系统。

## 示例

| 名称 | 说明 |
|------|------|
| `audit_demo` | 安全事件记录、文件轮转、批量性能测试 |

## 运行

```bash
cargo run -p vecboost-examples --bin audit_demo
```

## 说明

`AuditLogger` 通过后台异步批量写入实现高性能审计：

- **非阻塞记录**：`log_*` 方法仅 mpsc send，1000 次调用 < 1ms
- **批量 flush**：后台 writer task 每 1s 或 100 条时批量写入
- **文件轮转**：达到 `max_file_size_mb` 后自动轮转，保留 `max_files` 个历史文件
- **DB 后端**：启用 `db` feature 后可选数据库存储

### 支持的安全事件类型
- `log_login_success` / `log_login_failed` — 登录事件
- `log_logout` — 登出事件
- `log_user_created` / `log_user_updated` / `log_user_deleted` — 用户管理
- `log_permission_denied` — 权限拒绝
- `log_token_refresh` — Token 刷新
- `log_unauthorized_access` — 未授权访问
- `log_rate_limit_exceeded` — 速率限制

核心 API：
- `AuditLogger::new(config)` — 创建文件后端审计日志
- `logger.log_*()` — 记录各类安全事件
- `logger.flush()` — 同步等待 flush 到磁盘
