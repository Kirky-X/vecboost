# Spec — error-handling

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖此变更引入/修改的错误处理能力域需求。

## Requirements

### R-error-handling-001: VecboostError 类型重命名

`AppError` 枚举重命名为 `VecboostError`,所有 variant 名称保留,`#[error("...")]` 消息保留。`src/error.rs` 中保留 `pub type AppError = VecboostError;` 并标注 `#[deprecated(note = "Use VecboostError instead")]`。

**验收标准:**
- `grep -rn "AppError" src/` 输出仅剩 `error.rs` 中的 deprecated 别名
- `cargo doc --no-deps` 生成的文档中 `VecboostError` 为主要类型,`AppError` 标记 deprecated
- `cargo test --lib` 全部通过

### R-error-handling-002: IntoResponse impl 迁移

`impl IntoResponse for AppError` 改为 `impl IntoResponse for VecboostError`,HTTP 状态码映射保持不变(ConfigError→500,InvalidInput→400,AuthenticationError→401,RateLimitExceeded→429 等)。

**验收标准:**
- 单元测试 `test_app_error_to_response` 验证 16 个 variant 的状态码映射
- 响应体 JSON 格式保持 `{"error": "...", "code": <u16>}`

### R-error-handling-003: From impl 迁移

`impl From<std::io::Error> for AppError` / `impl From<candle_core::Error> for AppError` / `impl From<tokio::task::JoinError> for AppError` 全部迁移到 `VecboostError`。

**验收标准:**
- `?` 操作符在 `src/service/embedding.rs`、`src/model/`、`src/engine/` 中正常工作
- 无编译警告

## Constraints

- v0.2.0 是 breaking change,版本号必须 bump minor
- deprecated 别名保留到 v0.3.0 移除
- 错误消息 sanitization 逻辑(路径/token 脱敏)保留不变

## Out of Scope

- 不新增错误 variant(本轮仅重命名)
- 不重构错误层次(不引入子枚举)
- 不替换 thiserror(保留为依赖)
