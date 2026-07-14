# Spec — logging

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 inklog 日志基础设施能力域需求。

## Requirements

### R-logging-001: inklog 替换 tracing-subscriber

`src/logger/mod.rs` 定义 `LoggerModule` 实现 `AutoBuilder`,`build()` 调用 `inklog::LoggerManager::builder().level(level).console(console_enabled).file(file_path).build().await`。启用 inklog `standard` feature(http + cli)。

**验收标准(按实际代码):**
- `LoggerModule::build()` 返回 `Arc<inklog::LoggerManager>`
- `log::info!("Hello")` 输出到 console
- `log::error!("Test")` 写入 `logs/vecboost.log` 文件
- 日志格式包含时间戳、级别、模块路径、消息

### R-logging-002: 三级降级机制

inklog 配置 DB → File → Console 三级降级。DB 不可用时降级到 File,File 不可用时降级到 Console,确保日志不丢失。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- `src/logger/` 仅有基础 inklog 集成
- 无 DB → File → Console 三级降级逻辑
- 无降级事件验证测试

**验收标准(目标,v0.3.0 实现):**
- DB 连接断开时,日志自动降级到 File,无 panic
- File 写入失败(磁盘满)时,降级到 Console,日志仍输出
- 降级事件本身被记录到 Console(避免日志丢失元信息)

### R-logging-003: 敏感数据脱敏

inklog 配置正则脱敏规则:JWT token、密码、API key、邮箱、IP 地址。脱敏后的日志不包含原始敏感数据。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- `src/error.rs` 有 `sanitize_error_message` 函数(错误消息脱敏),但 **logger 无脱敏配置**
- 无正则脱敏规则配置
- 无 JWT/密码/API key/邮箱/IP 脱敏测试

**验收标准(目标,v0.3.0 实现):**
- `log::info!("token=eyJhbG...")` 输出 `token=[REDACTED_JWT]`
- `log::info!("password=secret123")` 输出 `password=[REDACTED_PASSWORD]`
- `log::info!("email=alice@example.com")` 输出 `email=[REDACTED_EMAIL]`
- `log::info!("ip=192.168.1.1")` 输出 `ip=[REDACTED_IP]`

### R-logging-004: 文件轮转

日志文件按大小(默认 100MB)和时间(默认 daily)轮转,旧文件保留 7 天后自动删除。

**v0.2.0 实际实施状态(未验证,推迟到 v0.3.0)**:
- 未验证 inklog 是否启用 100MB + daily 轮转
- 未验证 7 天保留策略
- 当前 `src/logger/` 仅配置基础 file/console sink

**验收标准(目标,v0.3.0 实现):**
- 文件达到 100MB 时自动创建新文件,旧文件重命名为 `vecboost.log.2026-07-13`
- 跨天时自动创建新文件
- 7 天前的日志文件被自动删除
- 轮转不影响正在写入的日志

## Constraints

- inklog 通过 `inklog` feature 启用,非 default
- 保留 `tracing` crate 作为日志门面(inklog 内部使用 tracing)
- 日志级别从配置文件 `[logging].level` 读取,默认 `info`

## Out of Scope

- 不启用 inklog 数据库 sink(本轮仅 console + file)
- 不启用 inklog HTTP 健康检查端点(本轮不暴露 `/logs/health`)
- 不启用 inklog AES-256-GCM 文件加密(本轮仅明文文件,加密留下个 change)
