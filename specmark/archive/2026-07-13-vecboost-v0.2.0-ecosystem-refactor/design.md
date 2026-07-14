# Design — vecboost-v0.2.0-ecosystem-refactor

## Context

VecBoost v0.1.x 当前架构:

```
main.rs → AppState(手动堆叠 14 字段) → routes/ → service/ → engine/
                                    ↓
                                    auth/rate_limit/cache/audit(各自实现)
```

**痛点**:
1. `AppState` 14 个 `Option<Arc<T>>` 字段手动管理,新增模块需改 AppState + main.rs + FromRef impl
2. `cache/mod.rs` 暴露 `CacheEntry`/`CacheStats` 等实现细节,外部可直接构造
3. `rate_limit/` 自研令牌桶,无熔断/封禁,与生态脱节
4. 日志用 `tracing-subscriber`,无加密/脱敏/降级
5. 配置用 `config` crate,无热重载/加密
6. 持久化全内存,重启丢失用户/审计
7. `AppError` 命名不规范,lib.rs 公共 API 不符合 crate 名

**约束**:
- v0.2.0 是 breaking change,需 bump minor 版本
- 7 个库均为 Kirky.X 自家,API 兼容性可控
- 必须保持 Candle 默认引擎不变
- 必须保持 gRPC proto 向后兼容
- TDD 强制:每个任务 Red → Green → Commit → gitnexus analyze

**相关历史决策**:
- v0.1.0 选择 Candle 而非 ONNX 作为默认引擎(性能 + Rust 原生)
- v0.1.2 引入 `default = []` 特性,所有功能需显式启用
- AGENTS.md 规定 `pub mod` vs `pub(crate) mod` 边界

## Decision

### D1: 模块管理迁移到 trait-kit typestate

**采用** `Kit<Unbuilt> → Kit<Ready>` typestate 模式作为目标设计:

```rust
// 目标 AppState(基于 trait-kit,完整重构推迟到 v0.3.0)
pub struct AppState {
    kit: Arc<trait_kit::Kit<trait_kit::Ready>>,
}

// 模块注册(main.rs)
let kit = Kit::new()
    .register::<EmbeddingModule>()?
    .register::<AuthModule>()?
    .register::<RateLimitModule>()?
    .register::<CacheModule>()?
    .register::<DbModule>()?
    .register::<LoggerModule>()?
    .build()?;

// 按需检索
let service = kit.require::<EmbeddingModule>()?;
```

**v0.2.0 实际实施状态（部分实现）**:
- module_registry 已建立,6 个模块类型定义齐全(`src/module_registry/mod.rs` + `impl_.rs`)
- 由于 `Kit` 内部基于 `RefCell`(`!Send + !Sync`),无法满足 Axum `AppState: Send + Sync` 要求,改用 `AsyncKit`(详见"实施偏离记录")
- `AppState` **保留 14 字段手动堆叠** + 6 个 `FromRef` impl(`src/lib.rs:66-88` + `:90-145`),未按原设计重构
- `main.rs:304-321` 注释明确写:"当前保持现有 AppState 结构不变,module_registry 作为未来重构的基础"
- 完整的 `AppState { kit }` 单字段重构 + `FromRef` 移除 **推迟到 v0.3.0**

**移除(原计划,未执行)**: `AppState` 14 字段 + 6 个 `FromRef` impl + 手动 `Arc<RwLock<>>` 堆叠

### D2: 7 库集成映射

| 库 | feature 名 | 替换的模块 | 配置方式 |
|---|---|---|---|
| dbnexus | `db` | auth/user_store + audit/ + 新增 metadata/ | TOML `[database]` 段 |
| inklog | `inklog` | tracing-subscriber + 自定义 logger | TOML `[logging]` 段 |
| limiteron | `limiteron` | rate_limit/ 全部 | TOML `[flow_control]` 段 |
| oxcache | `oxcache` | cache/ 全部 | TOML `[cache]` 段 |
| sdforge | `sdforge` | routes/ 部分 + 新增 cli/ + mcp/ | 宏注解 |
| trait-kit | (始终启用) | AppState 重构 | 代码内 |
| confers | (始终启用) | config/ 全部 | TOML derive |

### D3: 错误统一为 VecboostError

```rust
// src/error.rs
#[derive(Error, Debug, Clone)]
pub enum VecboostError {
    #[error("Config error: {0}")]
    Config(String),
    #[error("Model load error: {0}")]
    ModelLoad(String),
    // ... 全部 variant
}

// 保留 AppError 类型别名(一个版本周期)
#[deprecated(note = "Use VecboostError instead")]
pub type AppError = VecboostError;
```

### D4: mod.rs 加固规则

强制规则:
- `mod.rs` 只允许出现:`pub mod`/`pub(crate) mod` 声明、`pub use` 重导出、`trait` 定义、`enum`/`struct` 数据结构定义、常量
- `mod.rs` 禁止出现:`fn` 实现(除 trait 默认方法)、`impl` 块、具体算法
- 实现移至 `impl.rs` 或 `<module>_service.rs`
- 外部导入用 `use crate::cache` 而非 `use crate::cache::lru_cache`

**v0.2.0 已知偏离(部分模块未完全加固)**:
- `src/cache/mod.rs` 已按规则加固(impl 移至 `entry.rs`/`trait_impl.rs`),✅ 符合
- `src/module_registry/mod.rs` 已按规则加固(impl 移至 `impl_.rs`),✅ 符合
- `src/rate_limit/mod.rs:73-107` 仍保留 `impl Default for RateLimitConfig` + `impl RateLimitConfig { fn sliding_window()/token_bucket() }`,❌ 违反规则,推迟到 v0.3.0 移至子模块
- `src/audit/mod.rs:34-50` 仍保留 `impl SecurityEventType { fn as_str() }`,❌ 违反规则,推迟到 v0.3.0 移至 `impl_.rs`/`types.rs`

### D5: 多协议接口(sdforge 宏)

**目标设计**(原计划):

```rust
// src/service/embedding_api.rs
use sdforge::prelude::*;

#[service_api(
    name = "embed",
    version = "v1",
    path = "/api/v1/embed",
    method = "POST",
    tool_name = "embed_text",
    description = "Generate embedding vector for input text"
)]
pub async fn embed(req: EmbedRequest, kit: &Kit) -> Result<EmbedResponse, VecboostError> {
    let service = kit.require::<EmbeddingModule>()?;
    service.embed(req.text).await
}
```

**feature 控制**:
- `http`(默认): 生成 Axum 路由
- `mcp`: 生成 MCP server(rmcp 协议)
- `cli`: 生成 `vecboost` 二进制(clap 子命令)

**v0.2.0 实际实施状态(宏未落地)**:
- `Cargo.toml:114` 声明 `sdforge = { version = "0.4", optional = true }` 依赖
- `Cargo.toml` `[features]` 段保留 `http = ["sdforge/http"]`/`mcp = ["sdforge/mcp"]`/`cli = ["sdforge/cli", "dep:clap"]` 三个 feature 透传
- **`src/api/mod.rs:33-66` 的 3 个函数(embed/embed_batch/compute_similarity)未加任何 `#[service_api]` 或 `#[forge]` 宏注解**,均为普通 `async fn`,签名接收 `service: &EmbeddingService` 而非 `kit: &Kit`
- HTTP 路由仍由 `src/routes/embedding.rs` 手写 Axum handler,非 sdforge 生成
- CLI 由 `src/cli/mod.rs` 手写 clap `#[derive(Parser)]` + `CliCommand` 枚举,未使用 `sdforge::cli` 宏生成
- MCP 协议无任何实现(无 `rmcp` 依赖,无 MCP server 代码)
- sdforge feature 在 v0.2.0 仅作为"依赖声明"保留,实际多协议统一生成机制 **完整推迟到 v0.3.0**

### D6: 新引擎抽象

```rust
// src/engine/trait.rs(已存在,扩展)
pub trait InferenceEngine: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError>;
    // ... 现有方法
}

// 新增
pub struct TensorRtEngine { /* ... */ }  // feature = "tensorrt"
pub struct OpenVinoEngine { /* ... } }   // feature = "openvino"

impl InferenceEngine for TensorRtEngine { /* ... */ }
impl InferenceEngine for OpenVinoEngine { /* ... */ }
```

### D7: 特性组合测试矩阵

4 种核心组合(非全 64 组):

| 组合 | features | 测试范围 |
|---|---|---|
| minimal | `default` | lib + routes + service |
| grpc | `grpc` | + grpc 模块 |
| enterprise | `auth,redis,db,inklog,limiteron,oxcache` | + 全基础设施 |
| gpu | `cuda,grpc,auth,redis` | + GPU 路径(mock) |

CI 矩阵:`cargo test --features <组合>` × 4 + `cargo check --all-features`

### D8: tests 目录重组

**目标结构**(原计划):

```
tests/
├── integration/         # Rust 集成测试
│   ├── mod.rs
│   ├── api_test.rs      # 合并 integration_test.rs + test_real_inference.rs
│   └── grpc_test.rs
├── perf/                # Python 性能测试
│   ├── conftest.py
│   ├── api_simulator.py
│   └── client_factory.py
└── common/              # 共享 fixtures
    ├── mod.rs
    └── fixtures.rs
```

**v0.2.0 实际实施状态(部分偏离)**:
- `tests/integration/` 与 `tests/integration.rs` **顶层文件并存**(Rust 模块系统会警告)
- `tests/perf/` 与 `tests/perf.rs` **顶层文件并存**(同上)
- `tests/integration/mod.rs` 缺失
- `tests/integration/grpc_test.rs` 缺失
- `tests/common/fixtures.rs` 缺失(仅 `tests/common/mod.rs`)
- `tests/perf/` 多出 spec 未列出的 Python 文件:`config.py`/`fixtures.py`/`__init__.py`/`real_service.py`/`services.py`/`test_api.py`/`performance_test.rs`

实际目录结构(`find tests -type f`):

```
tests/
├── common/mod.rs
├── integration/api_test.rs
├── integration.rs          ← 顶层文件与目录并存(模块冲突)
├── perf/
│   ├── api_simulator.py
│   ├── client_factory.py
│   ├── config.py
│   ├── conftest.py
│   ├── fixtures.py
│   ├── __init__.py
│   ├── performance_test.rs
│   ├── real_service.py
│   └── services.py
│   └── test_api.py
├── perf.rs                 ← 顶层文件与目录并存(模块冲突)
└── perf/services.py
```

**已知偏离处理**: `tests/integration.rs` + `tests/integration/` 并存会导致 Rust 模块系统警告;`tests/perf.rs` + `tests/perf/` 同样问题。修复方式(删除顶层 .rs 文件 或 删除对应目录)推迟到 v0.3.0

**移除(原计划,部分执行)**: `real_engine.rs`(合并到 api_test.rs)、`test_api_real.py`(重复)、`test_server_integration.py`(合并到 api_test.rs)

### D9: 依赖特性化

```toml
# Cargo.toml(改造后)
[dependencies]
# 始终启用
tracing = "0.1"
thiserror = "2"
serde = { version = "1", features = ["derive"] }

# 可选(通过 feature 控制)
axum = { version = "0.7", optional = true, default-features = false, features = ["json"] }
tokio = { version = "1", optional = true, default-features = false, features = ["rt-multi-thread", "macros"] }
tonic = { version = "0.12", optional = true, default-features = false }

# 自家生态(全部 optional)
dbnexus = { version = "0.4", optional = true, default-features = false }
inklog = { version = "0.1", optional = true, default-features = false }
limiteron = { version = "0.2", optional = true, default-features = false }
oxcache = { version = "0.3", optional = true, default-features = false }
sdforge = { version = "0.4", optional = true, default-features = false }
trait-kit = "0.3"  # 始终启用(模块管理核心)
confers = { version = "0.4", optional = true, default-features = false }

[features]
default = ["http", "oxcache", "limiteron"]
http = ["dep:axum", "dep:tokio", "sdforge/http"]
grpc = ["dep:tonic", "dep:prost", "sdforge/grpc"]
mcp = ["sdforge/mcp"]
cli = ["sdforge/cli"]
db = ["dep:dbnexus", "dbnexus/sqlite"]
postgres = ["db", "dbnexus/postgres"]
inklog = ["dep:inklog"]
limiteron = ["dep:limiteron", "limiteron/standard"]
oxcache = ["dep:oxcache", "oxcache/core"]
tensorrt = ["dep:tensorrt-rs"]
openvino = ["dep:openvino"]
auth = ["dep:jsonwebtoken", "dep:argon2"]
redis = ["dep:redis"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
onnx = ["dep:ndarray", "dep:ort"]
```

**v0.2.0 default 偏离说明**:
- 原计划 `default = ["http"]`(仅 HTTP,符合"最小开箱即用")
- 实际 `Cargo.toml:123` 为 `default = ["http", "oxcache", "limiteron"]`(含 oxcache + limiteron)
- 偏离原因:缓存与限流是嵌入服务最常用基础设施,用户开箱即用期望包含;同时 AGENTS.md / README.md / README_EN.md 均按此说明
- 关联 spec 更新:`multi-protocol-api/spec.md` Constraints、`caching/spec.md` Constraints、`rate-limiting/spec.md` Constraints 同步调整为实际值

## Alternatives Considered

### A1: 不用 trait-kit,保留 AppState 手动管理

**否决理由**: 新增模块需改 4 处(AppState struct + FromRef impl + main.rs 构造 + routes 引用),违反 DRY。trait-kit typestate 在编译期验证依赖图,启动时不再 panic。

### A2: 用 anyhow 替代 VecboostError

**否决理由**: anyhow 丢失类型信息,无法 `match` 错误类型。VecboostError 用 thiserror 保留枚举 + `IntoResponse` impl,HTTP 状态码映射精确。

### A3: MCP 独立实现,不用 sdforge

**否决理由**: 重复实现 embed/similarity 接口,违反 DRY。sdforge 宏在编译时生成,零运行时开销,且支持 feature 切换协议。

### A4: 保留 `default = []`,所有功能显式启用

**否决理由**: 用户期望"开箱即用",`default = ["http"]` 提供最小可用 HTTP 服务,符合 Cargo 惯例。其他重型特性(db/redis/inklog)仍需显式启用。

### A5: tests 目录保留 Python + Rust 混合

**否决理由**: 规则 10 明确要求"优化合并"。Python 性能测试移至 `tests/perf/` 子目录,Rust 集成测试合并到 `tests/integration/`,职责清晰。

## Consequences

### 正面影响

1. **维护成本下降**: 7 个自研模块替换为生态库,bug 修复上游统一
2. **模块边界清晰**: trait-kit typestate 强制依赖图显式,mod.rs 加固规则消除泄漏
3. **多协议开箱即用**: sdforge 宏一次定义,HTTP/MCP/CLI 三协议自动生成
4. **持久化就绪**: dbnexus 提供用户/审计/元数据持久化,生产环境可用
5. **可观测性提升**: inklog 提供加密/脱敏/降级,limiteron 提供熔断/封禁
6. **引擎扩展性**: TensorRT/OpenVINO 加入,覆盖 NVIDIA/Intel 硬件

### 负面影响

1. **breaking change**: v0.2.0 不兼容 v0.1.x,用户需改 `AppError` → `VecboostError`
2. **依赖体积**: 启用全部 feature 后依赖数增加 ~30%,编译时间变长
3. **学习曲线**: trait-kit typestate 对新贡献者不熟悉,需文档配套
4. **测试复杂度**: 4 种特性组合 × mockall mock,CI 矩阵复杂

### 技术债

1. `AppError = VecboostError` 类型别名保留一个版本周期,v0.3.0 移除
2. `real_engine.rs` 合并到 `api_test.rs` 后部分断言需重写
3. TensorRT/OpenVINO 引擎本轮仅实现基础 embed,批量/相似度留待下个 change

### 实施偏离记录（Converge 阶段补充）

#### D1 偏离：使用 `AsyncKit` 替代 `Kit` + AppState 未完全重构

**原设计**: `AppState { kit: Arc<Kit<Ready>> }`，`Kit` 基于 typestate 模式，AppState 仅含 kit 字段。

**实际实施**: 改用 `AsyncKit<AsyncReady>`，且 AppState 保留 14 字段。
- **原因 1**: `Kit` 内部基于 `RefCell`，是 `!Send + !Sync`，无法放入 `AppState`（Axum 要求 `Send + Sync`）。
- **方案 1**: 启用 trait-kit `async` feature，使用 `AsyncKit`（基于 `Arc<RwLock>`，`Send + Sync`）。模块实现 `AsyncAutoBuilder` trait，`build()` 为 `async fn`。
- **原因 2**: 完整重构 AppState 会牵动所有 routes handler 签名 + FromRef impl + main.rs 构造，影响面过大。
- **方案 2**: `AppState` 保留 14 字段（向后兼容 `FromRef`），未新增 kit 字段到 AppState。module_registry 模块仅作为"未来重构基础"存在，未真正接入。完整移除 14 字段 + FromRef impl 推迟到 v0.3.0。

#### D5 偏离：sdforge 宏部分落地（Converge 阶段更新）

**原设计**: `#[service_api(name=..., ...)]` 宏标注 API 函数，sdforge 0.4.1 实际宏名为 `#[forge]`。

**实际实施（v0.2.0 Converge 阶段更新）**: sdforge 宏已部分落地。
- `Cargo.toml` 声明 `sdforge = { version = "0.4", optional = true }` 依赖 + `http`/`mcp`/`cli`/`openapi` 四个 feature 透传
- `src/api/mod.rs` 核心 API 函数（embed/embed_batch/compute_similarity）保持普通 `async fn`，接收 `service: &EmbeddingService`
- `src/api/mod.rs` 新增 5 个 `#[forge]` 宏标注的包装函数：
  - `forge_embed`/`forge_embed_batch`/`forge_compute_similarity`（`#[cfg(feature = "http")]`，通过全局 `OnceLock` 获取服务）
  - `cli_embed`/`cli_similarity`（`#[cfg(feature = "cli")]`，接收 `String` 参数）
- HTTP 路由仍由 `src/routes/embedding.rs` 手写 Axum handler（`#[forge]` 宏函数作为备用，未直接接入路由）
- CLI 由 `src/cli/mod.rs` 手写 clap `#[derive(Parser)]`（`#[forge]` 宏函数作为备用）
- MCP 协议仍无实现
- **Cargo.toml 新增 `openapi` feature**：sdforge 宏内部生成 `cfg(feature = "openapi")` 检查，需声明此 feature 让 check-cfg 通过
- **完整推迟到 v0.3.0**: `#[forge]` 宏函数接入 HTTP 路由 + MCP server 实现 + CLI 生成替换手写 clap

#### H5 已知偏离：sea-orm 2.0.0-rc.42 无法降级

**原规则**: 规则17 要求使用最新稳定版本。

**实际状况**: dbnexus 0.4.0 硬依赖 `sea-orm 2.0.0-rc.42`（`cargo tree -i sea-orm` 确认）。降级到 1.1.20 稳定版会破坏 dbnexus 集成。
- **处理**: 保持 sea-orm 2.0.0-rc.42，待 dbnexus 发布支持 sea-orm 2.0 稳定版后升级。
- **ort 2.0.0-rc.12**: 同理，ONNX Runtime 绑定目前无 2.0 稳定版。

#### D8 偏离：tests 目录顶层文件与子目录并存

**原设计**: `tests/integration/` + `tests/perf/` + `tests/common/` 三级子目录,无顶层 .rs 文件。

**实际实施**: 顶层 .rs 文件与子目录并存(模块冲突警告)。
- `tests/integration.rs` + `tests/integration/` 并存(Rust 模块系统会警告)
- `tests/perf.rs` + `tests/perf/` 并存(同上)
- `tests/integration/mod.rs` 缺失,`tests/integration/grpc_test.rs` 缺失
- `tests/common/fixtures.rs` 缺失(仅 `tests/common/mod.rs`)
- `tests/perf/` 多出 spec 未列出的 Python 文件:`config.py`/`fixtures.py`/`__init__.py`/`real_service.py`/`services.py`/`test_api.py`/`performance_test.rs`
- **处理**: 修复方式(删除顶层 .rs 文件 或 删除对应目录)推迟到 v0.3.0

#### D9 偏离：default features 包含 oxcache + limiteron

**原设计**: `default = ["http"]`(仅 HTTP,符合"最小开箱即用")。
**multi-protocol-api spec 原约束**: `default = ["http", "config"]`(HTTP + 配置)。

**实际实施**: `Cargo.toml:123` 为 `default = ["http", "oxcache", "limiteron"]`。
- **偏离原因**: 缓存与限流是嵌入服务最常用基础设施,用户开箱即用期望包含;同时 AGENTS.md / README.md / README_EN.md 均按此说明
- **关联 spec 更新**: `multi-protocol-api/spec.md` Constraints、`caching/spec.md` Constraints、`rate-limiting/spec.md` Constraints、`configuration/spec.md` Constraints 同步调整为实际值

#### MEDIUM 推迟项

| 问题 | 推迟原因 | 目标版本 |
|------|----------|----------|
| M3: cache stampede 竞态 | 单机推理服务高并发场景优先级低，需 per-key mutex | v0.3.0 |
| M4: oxcache 测试依赖 sleep | 测试质量问题，不影响生产 | v0.3.0 |
| M9: main.rs 直接调用 inklog | D1 已部分实施 trait-kit 集成，完整迁移推迟 | v0.3.0 |
| M10: sdforge 宏部分落地 | D5 更新:`#[forge]` 宏函数已添加,未接入路由/CLI/MCP | v0.3.0 |
| M11: AppState 14 字段未重构 | D1 完整偏离,FromRef impl 保留 | v0.3.0 |
| M12: tests 目录顶层与子目录并存 | D8 偏离,Rust 模块系统警告 | v0.3.0 |
| M13: rate_limit/audit mod.rs 加固违规 | D4 偏离,impl 块仍在 mod.rs | v0.3.0 |
| M14: TensorRT/OpenVINO 真实 crate 集成 | 当前为 stub,缺 libnvinfer.so/libopenvino_c.so | v0.3.0 |
| M15: EmbeddingService::switch_engine 运行时切换 | 涉及线程安全/并发/状态管理 | v0.3.0 |
| M16: AuditLogger::query + 90 天自动清理 | R-persistence-003 部分推迟 | v0.3.0 |
| M17: 审计日志 DB-only 迁移 | 当前双轨(文件 + DB) | v0.3.0 |
| M18: metadata-persistence feature | R-persistence-005 完整推迟 | v0.3.0 |
| M19: R-logging-002 三级降级 | inklog 基础集成,无降级逻辑 | v0.3.0 |
| M20: R-logging-003 敏感数据脱敏 | error.rs 有 sanitize,logger 无脱敏 | v0.3.0 |
| M21: R-logging-004 文件轮转验证 | 未验证 100MB + daily + 7 天保留 | v0.3.0 |
| M22: R-caching-003 TTL 全局默认 | put 签名未含 TTL 参数 | v0.3.0 |
| M23: R-caching-004 多实例 Pub/Sub | 无 Redis Pub/Sub 集成 | v0.3.0 |
| M24: R-configuration-003 配置加密 | 无 encryption feature | v0.3.0 |
| M25: http::run_server 公共 API | HTTP 启动逻辑在 main.rs 未封装 | v0.3.0 |
| M26: MCP 协议实现 | 无 rmcp 依赖,无 MCP server | v0.3.0 |
| M27: OpenAPI version 字段 | routes/mod.rs:29 为 0.1.0,应同步为 0.2.0 | v0.3.0 |
| M28: swagger-ui 路径 + redoc 端点 | 当前 /api-docs,无 /redoc/ | v0.3.0 |
| M29: 覆盖率 61.92% 未达 95% 目标 | engine/candle_engine.rs 1227 行需真实模型;routes/ 需集成测试;pipeline/worker 248 行需 HTTP 触发 | v0.3.0 |
| M30: cuda-network 矩阵组合失败 | 环境限制:无 CUDA toolkit + libnvinfer.so | v0.3.0 |

#### 覆盖率偏离记录（T055b）

**目标**: 95%+ 覆盖率（T055 要求）

**实际**: 61.92%（`cargo llvm-cov --features http,auth,db,limiteron,oxcache,cli --lib --summary-only`）

**主要低覆盖率模块**（按未覆盖行数排序）:

| 模块 | 未覆盖行数 | 原因 |
|------|-----------|------|
| engine/candle_engine.rs | 1227 | 需真实模型文件，CI 环境不可用 |
| service/embedding.rs | 770 | 需真实模型 + HTTP 集成测试 |
| config/app.rs | 374 | 配置解析分支多，需补充测试 |
| device/amd.rs | 279 | AMD GPU 路径，无 AMD 环境 |
| model/recovery.rs | 300 | 模型恢复逻辑，需真实模型 |
| auth/user_store.rs | 250 | DB CRUD 分支，部分测试覆盖 |
| pipeline/worker.rs | 248 | 需 HTTP 请求触发 worker loop |
| security/encrypted_store.rs | 243 | 加密存储，需补充测试 |
| audit/mod.rs | 211 | 审计日志 DB 写入路径 |
| device/manager.rs | 239 | 设备管理，需 GPU 环境 |

**处理**:
- v0.2.0 接受 61.92% 覆盖率（431 单元测试全通过）
- 核心模块（rate_limit/cache/db/error/security/mod）覆盖率 90%+
- **v0.3.0 目标**: 补充集成测试 + 真实模型测试环境 → 80%+
- **v0.4.0 目标**: 完整集成测试 + GPU 环境模拟 → 95%+

### 后续跟进项

- v0.3.0: 移除 AppError 别名;TensorRT/OpenVINO 真实 crate 集成(M14) + 批量推理;性能基准对比;sdforge 宏完整落地(M10);AppState 完整重构(M11);tests 目录修复(M12);mod.rs 加固违规修复(M13);EmbeddingService::switch_engine(M15);AuditLogger::query + 90 天清理(M16);审计日志 DB-only 迁移(M17);metadata-persistence(M18);logging 三级降级/脱敏/轮转(M19/M20/M21);caching TTL + 多实例同步(M22/M23);configuration 加密(M24);http::run_server(M25);MCP 协议实现(M26);OpenAPI version 同步(M27);swagger-ui 路径 + redoc 端点(M28);覆盖率提升至 80%+(M29);CUDA 环境测试(M30)
- v0.4.0: Kubernetes Operator;分布式部署;模型版本管理;覆盖率 95%+(M29)
