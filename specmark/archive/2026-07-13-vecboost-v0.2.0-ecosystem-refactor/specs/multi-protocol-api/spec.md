# Spec — multi-protocol-api

> Delta spec for change `vecboost-v0.2.0-ecosystem-refactor`. 覆盖 sdforge 多协议接口能力域需求。

## Requirements

### R-multi-protocol-api-001: service_api 宏定义

`src/api/mod.rs` 用 `#[service_api(...)]` 宏定义 3 个接口:`embed`/`embed_batch`/`compute_similarity`。宏参数包含 `name`/`version`/`path`/`method`/`tool_name`/`description`,编译时根据 feature 生成 HTTP/MCP/CLI 代码。

**v0.2.0 实际实施状态(宏未落地,推迟到 v0.3.0)**:
- `src/api/mod.rs:33-66` 的 3 个函数均为普通 `async fn`,**未加任何 `#[service_api]` 或 `#[forge]` 宏注解**
- 函数签名接收 `service: &EmbeddingService` 而非 `kit: &Kit`
- HTTP 路由仍由 `src/routes/embedding.rs` 手写 Axum handler,非 sdforge 生成
- MCP 协议无任何实现
- CLI 由 `src/cli/mod.rs` 手写 clap `#[derive(Parser)]`,非 sdforge 生成
- `Cargo.toml` 已声明 `sdforge = { version = "0.4", optional = true }` 依赖 + `http`/`mcp`/`cli` feature 透传,但仅作为依赖声明保留
- 详见 `design.md` D5 决策 + "实施偏离记录"

**验收标准(目标,v0.3.0 实现):**
- 启用 `http` feature 时,生成 Axum 路由 `POST /api/v1/embed`/`POST /api/v1/embed_batch`/`POST /api/v1/similarity`
- 启用 `mcp` feature 时,生成 MCP server 暴露 `embed_text`/`embed_batch`/`compute_similarity` 三个 tool
- 启用 `cli` feature 时,生成 `vecboost` 二进制支持 `embed`/`batch`/`similarity` 子命令
- 宏注解参数错误时编译失败(如 `method = "INVALID"` 报错)

### R-multi-protocol-api-002: HTTP 协议

HTTP 协议基于 Axum 0.7,路由路径 `/api/v1/<endpoint>`。请求体 JSON,响应体 JSON。支持 OpenAPI 文档自动生成(utoipa)。

**v0.2.0 实际实施状态(部分实现)**:
- HTTP 路由由 `src/routes/embedding.rs` 手写 Axum handler(非 sdforge 生成)
- SwaggerUi 路径为 **`/api-docs`**(非 `/swagger-ui/`),OpenAPI JSON 路径为 `/api-docs/openapi.json`(见 `src/routes/impl_.rs:41`)
- **无 `/redoc/` 端点**(推迟到 v0.3.0)
- 3 个 HTTP 端点 `POST /api/v1/embed`/`POST /api/v1/embed_batch`/`POST /api/v1/similarity` 已实现

**验收标准(按实际代码):**
- `POST /api/v1/embed` with `{"text": "Hello"}` 返回 `{"embedding": [...], "dimension": 1024, "processing_time_ms": 15.5}`
- `POST /api/v1/embed_batch` with `{"texts": ["a", "b"]}` 返回 `{"embeddings": [[...], [...]], "count": 2}`
- `POST /api/v1/similarity` with `{"text1": "a", "text2": "b", "metric": "cosine"}` 返回 `{"score": 0.85}`
- `/api-docs` 显示 3 个端点的 Swagger UI 文档
- `/api-docs/openapi.json` 返回 OpenAPI 规范

**推迟到 v0.3.0:**
- `/swagger-ui/` 路径(或保留 `/api-docs` 并在文档中统一)
- `/redoc/` 端点
- OpenAPI 文档 `version` 字段当前为 `0.1.0`(`src/routes/mod.rs:29`),需在 v0.3.0 同步为 `0.2.0`

### R-multi-protocol-api-003: MCP 协议

MCP 协议基于 rmcp 2.1,通过 stdio 传输。3 个 tool 注册到 MCP server,客户端调用时返回 JSON 结果。

**v0.2.0 实际实施状态(未实现,推迟到 v0.3.0)**:
- 无 `rmcp` 依赖(`Cargo.toml` 未声明)
- 无 MCP server 代码(无 `src/mcp/` 目录,`src/api/` 中无 MCP 相关实现)
- `Cargo.toml` 的 `mcp = ["sdforge/mcp"]` feature 仅作为透传声明保留,无实际效果
- 详见 `design.md` D5 决策 + "实施偏离记录"

**验收标准(目标,v0.3.0 实现):**
- 启动 `vecboost --mcp` 进入 MCP server 模式,监听 stdio
- MCP 客户端调用 `embed_text` tool with `{"text": "Hello"}` 返回 `{"embedding": [...], "dimension": 1024}`
- MCP 客户端调用 `embed_batch` tool with `{"texts": ["a", "b"]}` 返回 `{"embeddings": [...]}`
- MCP 客户端调用 `compute_similarity` tool with `{"text1": "a", "text2": "b"}` 返回 `{"score": 0.85}`
- tool 描述包含 `name`/`description`/`inputSchema`(JSON Schema)

### R-multi-protocol-api-004: CLI 协议

CLI 协议基于 clap 4.6,`vecboost` 二进制支持子命令。每个子命令对应一个 API 调用,输出 JSON 到 stdout。

**v0.2.0 实际实施状态(手写 clap,非 sdforge 生成)**:
- `src/cli/mod.rs` 使用 `#[derive(Parser)]` + `CliCommand` 枚举手写 clap 命令解析
- 支持 3 个子命令:`embed --text`/`batch --input`/`similarity --text1 --text2`
- `Cargo.toml` 的 `cli = ["sdforge/cli", "dep:clap"]` feature 中,`sdforge/cli` 透传声明保留但未实际使用,仅 `dep:clap` 生效
- `src/lib.rs` 导出 `pub mod cli;` + `run_cli` 函数(`cli/mod.rs:67`)✓
- 详见 `design.md` D5 决策 + "实施偏离记录"

**验收标准(按实际代码):**
- `vecboost embed --text "Hello"` 输出 `{"embedding": [...], "dimension": 1024}` 到 stdout
- `vecboost batch --input texts.txt` 输出 `{"embeddings": [...]}` 到 stdout
- `vecboost similarity --text1 "a" --text2 "b"` 输出 `{"score": 0.85}` 到 stdout
- `vecboost --help` 显示 3 个子命令的帮助
- `vecboost embed --text "Hello" --json` 等价于默认(总是 JSON 输出)

### R-multi-protocol-api-005: Rust SDK 导出

`src/lib.rs` 导出 Rust SDK 公共 API,允许其他 crate 作为库引入。通过 feature 控制导出范围。

**v0.2.0 实际实施状态(部分实现)**:
- `use vecboost::EmbeddingService;` ✓
- `use vecboost::VecboostError;` ✓
- `use vecboost::EmbedRequest;` ✓
- `use vecboost::cli::run_cli;` ✓(启用 `cli` feature 时)
- **`use vecboost::http::run_server;` ❌**(无 `src/http/` 模块;HTTP server 启动逻辑当前在 `main.rs`,未封装为公共 API;推迟到 v0.3.0)

**验收标准(按实际代码):**
- `use vecboost::EmbeddingService;` 可在其他 crate 使用
- `use vecboost::VecboostError;` 导出错误类型
- `use vecboost::EmbedRequest;` 导出请求结构
- 启用 `cli` feature 时,`use vecboost::cli::run_cli;` 运行 CLI

**推迟到 v0.3.0:**
- 启用 `http` feature 时,`use vecboost::http::run_server;` 启动 HTTP server(需创建 `src/http/mod.rs` 封装)
- 不启用任何 feature 时,仅导出核心 service trait

## Constraints

- sdforge 通过 `http`/`mcp`/`cli` 三个独立 feature 控制,非 default 启用 mcp/cli
- **v0.2.0 实际 `default = ["http", "oxcache", "limiteron"]`**(原计划 `default = ["http", "config"]`,偏离原因见 `design.md` D9)
- 宏生成的代码零运行时开销(编译时协议选择)—— **v0.2.0 宏未实际使用,推迟到 v0.3.0**
- Rust SDK 通过 `src/lib.rs` `pub use` 导出,不暴露内部模块

## Out of Scope

- 不实现 gRPC 协议(已有 `grpc` feature,基于 tonic 直接实现,不走 sdforge)
- 不实现 WebSocket 协议(`websocket` feature 留待下个 change)
- 不实现 SSE 流式响应(`sse` feature 留待下个 change)
- 不实现 API 版本协商(本轮仅 v1,版本字段保留但忽略)
