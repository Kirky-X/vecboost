# Tasks — vecboost-v0.2.0-ecosystem-refactor

> **执行规则**: 每个任务遵循 TDD 5 步循环(Red → Green → Commit → gitnexus analyze → Next),由独立 subagent 执行。每 Phase 完成后用 `diting` + `tiangang` 审查。失败重试遵循规则 20(L1→L2→L3→L4)。

## Phase 1: 基础准备

- [x] [T001] [P0] 升级 `Cargo.toml` 版本号到 `0.2.0`,更新 `src/lib.rs` 顶部的版权注释从 `2025` 改为 `2025-2026`,同步 `README.md`/`README_zh.md` 中的版本徽章为 `0.2.0`。验证:`cargo check` 通过
- [x] [T002] [P0] 重命名 `/home/dev/projects/vecboost/README.md` → `/home/dev/projects/vecboost/README_EN.md`,重命名 `/home/dev/projects/vecboost/README_zh.md` → `/home/dev/projects/vecboost/README.md`。更新 `README_EN.md` 顶部徽章 `Version-0.1.2` → `Version-0.2.0`;更新新 `README.md` 同步版本号
- [x] [T003] [P0] 统一所有 `.rs` 文件版权头为 `// Copyright (c) 2025 Kirky.X\n//\n// Licensed under the MIT License\n// See LICENSE file in the project root for full license information.`。扫描 `src/`、`tests/`、`examples/`、`build.rs` 所有 `.rs` 文件,修复 `Copyright (c) 2025 VecBoost` 等不一致格式。用脚本 `scripts/check_copyright.sh` 验证零违规
- [x] [T004] [P1] 改造 `Cargo.toml` 依赖为特性化:所有可选依赖改为 `optional = true, default-features = false`;版本号统一为 `x.x` 格式(如 `1.40` 而非 `1.40.0`);`[features]` 段重构为 `default = ["http"]` + 16 个独立 feature(http/grpc/mcp/cli/db/postgres/inklog/limiteron/oxcache/auth/redis/cuda/metal/onnx/tensorrt/openvino)。验证:`cargo check --no-default-features --features http` 通过
- [x] [T005] [P1] 创建 `scripts/check_copyright.sh` 脚本,扫描所有 `.rs` 文件检查版权头是否包含 `Copyright (c) 2025 Kirky.X` 和 `MIT License`,输出违规文件列表。在 `.pre-commit-config.yaml` 中追加此脚本作为 hook

## Phase 2: 错误统一与模块加固

- [x] [T006] [P0] 在 `src/error.rs` 中将 `AppError` 重命名为 `VecboostError`,所有 variant 保留。新增 `pub type AppError = VecboostError;` 并标注 `#[deprecated(note = "Use VecboostError instead")]`。修改 `impl IntoResponse for AppError` 为 `impl IntoResponse for VecboostError`。运行 `cargo test --lib` 验证现有测试通过(测试中改用 `VecboostError`)
- [x] [T007] [P0] 全局搜索替换 `AppError` → `VecboostError`(除 `error.rs` 中的 deprecated 别名):`src/auth/`、`src/cache/`、`src/config/`、`src/domain/`、`src/engine/`、`src/grpc/`、`src/metrics/`、`src/model/`、`src/pipeline/`、`src/rate_limit/`、`src/routes/`、`src/security/`、`src/service/`、`src/text/`、`src/utils/`、`src/main.rs`。用 `grep -rn "AppError" src/` 验证仅剩 deprecated 别名
- [x] [T008] [P1] 加固 `src/cache/mod.rs`:移除 `CacheEntry`/`CacheStats`/`CacheConfig`/`CacheStrategy` 的 `pub` 字段改为 `pub(crate)`;移除 `mod.rs` 中的 `impl CacheEntry`/`impl CacheStats`/`impl CacheGetOrInsert` 实现块,移至新建 `src/cache/entry.rs` 和 `src/cache/trait_impl.rs`。`mod.rs` 仅保留 `pub trait Cache` + 枚举/结构定义 + `pub use` 重导出
- [x] [T009] [P1] 加固 `src/rate_limit/mod.rs`:移除 `pub mod limiter/store/token_bucket` 改为 `pub(crate) mod`,仅通过 `pub use` 暴露 `RateLimiter`/`RateLimitConfig`/`RateLimitDimension`/`RateLimitStatus`。将 `TokenBucket`/`TokenBucketConfig` 等具体实现类型改为 `pub(crate)`
- [x] [T010] [P1] 加固其他 7 个 `mod.rs`:`src/auth/mod.rs`、`src/config/mod.rs`、`src/engine/mod.rs`、`src/metrics/mod.rs`、`src/pipeline/mod.rs`、`src/routes/mod.rs`、`src/service/mod.rs`。规则:`mod.rs` 只保留 `pub mod`/`pub(crate) mod` 声明、`pub use` 重导出、`trait`/`enum`/`struct` 定义,所有 `impl` 块和 `fn` 实现移至 `impl.rs` 或子模块
- [x] [T011] [P1] 修正跨模块导入:全局搜索 `use crate::cache::lru_cache::`、`use crate::rate_limit::token_bucket::`、`use crate::auth::jwt::` 等具体文件导入,改为 `use crate::cache::`、`use crate::rate_limit::`、`use crate::auth::` 模块级导入。验证:`cargo check --all-features` 通过

## Phase 3: 7 库集成(按依赖顺序)

### 3.1 trait-kit(模块管理核心)

- [x] [T012] [P0] 添加 `trait-kit = "0.3"` 到 `Cargo.toml`(始终启用,非 optional)。在 `src/lib.rs` 新增 `pub mod module_registry;`,创建 `src/module_registry/mod.rs` 定义 6 个模块:`EmbeddingModule`/`AuthModule`/`RateLimitModule`/`CacheModule`/`DbModule`/`LoggerModule`,每个实现 `trait_kit::prelude::ModuleMeta` + `AutoBuilder` trait。`mod.rs` 仅放 trait 定义和模块声明,实现放 `src/module_registry/impl.rs`
- [x] [T013] [P0] 编写 `src/module_registry/tests.rs` 单元测试:验证 `Kit::new().register::<EmbeddingModule>().build()` 返回 `Ok(Kit<Ready>)`;验证 `kit.require::<EmbeddingModule>()` 返回 `Arc<EmbeddingService>`;验证循环依赖检测(`ModuleA` 依赖 `ModuleB`,`ModuleB` 依赖 `ModuleA` 返回 `Err`)。测试应先失败(Red),然后实现 `AutoBuilder` 让测试通过(Green)
- [x] [T014] [P0] 重构 `src/lib.rs` 的 `AppState`:`pub struct AppState { pub kit: Arc<trait_kit::Kit<trait_kit::Ready>> }`,移除 14 个手动字段。更新 `src/main.rs` 用 `Kit::new().register::<...>().build()` 构造。更新所有 `FromRef` impl 为基于 `kit.require::<...>()` 的辅助方法。验证:`cargo test --lib` 全部通过

### 3.2 confers(配置管理)

- [x] [T015] [P0] 添加 `confers = { version = "0.4", optional = true, default-features = false, features = ["derive"] }` 到 `Cargo.toml`,`[features]` 新增 `config = ["dep:confers"]` 并加入 `default`。创建 `src/config/app_config.rs` 用 `#[derive(Config)]` 派生 `AppConfig`/`ServerConfig`/`ModelConfig`/`AuthConfig`。保留旧 `src/config/app.rs` 一个版本周期,内部转为 confers 加载
- [x] [T016] [P0] 编写 `src/config/tests.rs`:验证从 `config_minimal.toml` 加载 `AppConfig` 成功;验证环境变量 `VECBOOST_JWT_SECRET` 覆盖 TOML;验证热重载触发回调(`subscribe` API)。Red → Green → Commit
- [x] [T017] [P1] 更新 `config.toml` 和 `config_minimal.toml`:新增 `[database]`/`[logging]`/`[flow_control]`/`[cache]` 段,字段参考 7 库文档。新增 `config_full.toml` 示例文件展示全部特性

### 3.3 inklog(日志基础设施)

- [x] [T018] [P0] 添加 `inklog = { version = "0.1", optional = true, default-features = false, features = ["standard"] }` 到 `Cargo.toml`,`[features]` 新增 `inklog = ["dep:inklog"]`。创建 `src/logger/mod.rs` + `src/logger/impl.rs`,定义 `LoggerModule` 实现 `AutoBuilder`,`build()` 调用 `inklog::LoggerManager::builder().level("info").console(true).file("logs/vecboost.log").build().await`
- [x] [T019] [P0] 编写 `src/logger/tests.rs`:验证 `LoggerModule::build()` 返回 `Arc<LoggerManager>`;验证日志写入 `logs/test.log`;验证 `log::info!` 宏正常工作。Red → Green
- [x] [T020] [P1] 修改 `src/main.rs` 移除 `tracing_subscriber::fmt().init()`,改为 `let kit = kit.register::<LoggerModule>()?;`。保留 `tracing` crate 作为日志门面(inklog 内部使用 tracing)

### 3.4 oxcache(缓存)

- [x] [T021] [P0] 添加 `oxcache = { version = "0.3", optional = true, default-features = false, features = ["core"] }` 到 `Cargo.toml`,`[features]` 新增 `oxcache = ["dep:oxcache"]`。创建 `src/cache/oxcache_backend.rs` 实现 `pub(crate) struct OxCacheBackend` 包装 `oxcache::Cache`,实现 `crate::cache::Cache` trait
- [x] [T022] [P0] 编写 `src/cache/oxcache_tests.rs`:验证 `OxCacheBackend::put` + `get` 命中;验证 LRU 驱逐策略;验证 TTL 过期(用 `tokio::time::sleep`);验证 `clear` 清空。Red → Green
- [x] [T023] [P1] 删除 `src/cache/arc_cache.rs`、`src/cache/lfu_cache.rs`、`src/cache/lru_cache.rs`、`src/cache/kv_cache.rs`、`src/cache/bloom_filter.rs`、`src/cache/tiered_cache.rs` 6 个自研文件。更新 `src/cache/mod.rs` 移除对应 `pub(crate) mod` 声明,改为 `#[cfg(feature = "oxcache")] pub(crate) mod oxcache_backend;`。更新 `src/service/embedding.rs` 用 `OxCacheBackend` 替换 `LruCache`

### 3.5 limiteron(限流)

- [x] [T024] [P0] 添加 `limiteron = { version = "0.2", optional = true, default-features = false }` 到 `Cargo.toml`(不用 standard feature,核心限流器无需 feature),`[features]` 新增 `limiteron = ["dep:limiteron"]` 并加入 default。创建 `src/rate_limit/limiteron_adapter.rs` 实现 `pub struct LimiteronAdapter` 包装 `limiteron::TokenBucketLimiter`(按维度 key 管理),提供 check_rate_limit/get_status/get_remaining 兼容接口
- [x] [T025] [P0] 编写 `src/rate_limit/limiteron_adapter.rs` 内联测试:验证令牌桶限流(限额内允许/超限拒绝);验证多维度独立计数;验证全维度须通过;验证 get_status/get_remaining;验证 Global 隔离;验证令牌补充。8 个测试全绿
- [x] [T026] [P1] 删除 `src/rate_limit/limiter.rs`、`store.rs`、`token_bucket.rs`、`redis_store.rs` 4 个自研文件(-1064 行)。类型定义(RateLimitDimension/Config/Status/Algorithm)移到 `mod.rs`。更新 `lib.rs`/`main.rs`/`module_registry` 用 `LimiteronAdapter` 替换 `RateLimiter`。240 测试全绿

### 3.6 dbnexus(持久化)

- [x] [T027] [P0] 添加 `dbnexus = { version = "0.4", optional = true, default-features = false, features = ["sqlite", "runtime-tokio-rustls", "macros", "permission", "cache"] }` 到 `Cargo.toml`,`[features]` 新增 `db = ["dep:dbnexus"]`、`postgres = ["db", "dbnexus/postgres"]`。创建 `src/db/mod.rs` 定义 `DbPool` wrapper(包装 `dbnexus::DbPool`),提供 `DbPool::new(url)`/`get_session(role)`/`inner()` + `init_schema()` 建表函数(users + audit_logs)。在 `src/lib.rs` 新增 `#[cfg(feature = "db")] pub mod db;`
- [x] [T028] [P0] 编写 `src/db/mod.rs` 内联测试 7 个:验证 SQLite 内存模式 `sqlite::memory:` 连接成功;验证 admin 角色 get_session 成功;验证 user 角色 get_session 被拒绝(无权限配置);验证 init_schema 创建 users/audit_logs 表;验证 users 表 CRUD(insert/select/update/delete);验证审计日志写入 audit_logs 表;验证 pool clone 共享连接。7 测试全绿
- [ ] [T029] [P1] 重构 `src/auth/user_store.rs`:用 `dbnexus::Session` 替换内存 `HashMap`,实现 `UserStore::create`/`get`/`verify_password` 通过 `dbnexus::DbPool`。新增 `src/auth/migrations/001_users.sql` 建表语句(id/username/password_hash/role/created_at)。验证:`cargo test --features db --lib` 通过
- [ ] [T030] [P1] 重构 `src/audit/mod.rs`:用 `dbnexus::Session` 替换文件日志,实现 `AuditLogger::log` 通过 `dbnexus` 写入 `audit_logs` 表。新增 `src/audit/migrations/001_audit_logs.sql` 建表语句(id/user_id/action/resource/ip/timestamp)。验证:`cargo test --features db --lib` 通过

### 3.7 sdforge(多协议接口)

- [ ] [T031] [P0] 添加 `sdforge = { version = "0.4", optional = true, default-features = false }` 到 `Cargo.toml`,`[features]` 新增 `http = ["dep:axum", "dep:tokio", "sdforge/http"]`、`mcp = ["sdforge/mcp"]`、`cli = ["sdforge/cli"]`。创建 `src/api/mod.rs` 定义 3 个 `#[service_api]` 宏标注的函数:`embed`/`embed_batch`/`compute_similarity`
- [ ] [T032] [P0] 编写 `src/api/tests.rs`:验证 HTTP `POST /api/v1/embed` 返回 200 + embedding 向量;验证 HTTP `POST /api/v1/similarity` 返回相似度分数;验证请求体校验(空 text 返回 400)。Red → Green
- [ ] [T033] [P1] 创建 `src/cli/mod.rs` 用 `sdforge::cli` 生成 `vecboost` 二进制,支持子命令:`embed --text "Hello"`、`batch --input file.txt`、`similarity --text1 "a" --text2 "b"`。更新 `Cargo.toml` `[[bin]]` 段。验证:`cargo run --features cli -- embed --text "Hello"` 输出向量

## Phase 4: 新引擎支持

- [ ] [T034] [P1] 添加 `tensorrt-rs = { version = "0.5", optional = true, default-features = false }` 到 `Cargo.toml`,`[features]` 新增 `tensorrt = ["dep:tensorrt-rs"]`。创建 `src/engine/tensorrt_engine.rs` 实现 `pub(crate) struct TensorRtEngine` + `impl InferenceEngine for TensorRtEngine`。`mod.rs` 加 `#[cfg(feature = "tensorrt")] pub(crate) mod tensorrt_engine;`
- [ ] [T035] [P1] 编写 `src/engine/tensorrt_tests.rs`:验证 `TensorRtEngine::new(model_path)` 加载成功(mock);验证 `embed("Hello")` 返回 1024 维向量(mock);验证 batch embed 8 条文本返回 8 个向量。用 `mockall` mock tensorrt-rs API。Red → Green
- [ ] [T036] [P1] 添加 `openvino = { version = "0.7", optional = true, default-features = false }` 到 `Cargo.toml`,`[features]` 新增 `openvino = ["dep:openvino"]`。创建 `src/engine/openvino_engine.rs` 实现 `pub(crate) struct OpenVinoEngine` + `impl InferenceEngine for OpenVinoEngine`
- [ ] [T037] [P1] 编写 `src/engine/openvino_tests.rs`:验证 `OpenVinoEngine::new(model_path)` 加载成功(mock);验证 `embed("Hello")` 返回向量(mock);验证 CPU/GPU device 切换。用 `mockall` mock。Red → Green
- [ ] [T038] [P1] 扩展 `src/engine/factory.rs`(或新增)实现 `EngineFactory::create(engine_type: EngineType) -> Box<dyn InferenceEngine>`,支持 `Candle`/`Onnx`/`TensorRt`/`OpenVino` 4 种枚举。更新 `src/config/model.rs` `EngineType` 枚举新增 `TensorRt`/`OpenVino` variant

## Phase 5: 测试覆盖与质量

- [ ] [T039] [P0] 安装 `cargo-tarpaulin`:`cargo install cargo-tarpaulin`。运行 `cargo tarpaulin --features http,grpc --out Html --output-dir coverage/` 生成基准覆盖率报告。识别覆盖率 < 80% 的模块清单
- [ ] [T040] [P0] 为覆盖率 < 80% 的模块补充单元测试,目标 95%。重点模块:`src/auth/`(JWT 验证/密码哈希)、`src/security/`(加密/脱敏)、`src/pipeline/`(优先级队列)、`src/text/`(分块/分词)、`src/utils/`(相似度计算)。每模块至少 10 个测试用例,覆盖正常+边界+错误路径
- [ ] [T041] [P0] 创建 `.github/workflows/feature-matrix.yml` CI 矩阵:4 种组合(`default`/`grpc`/`auth,redis,db,inklog,limiteron,oxcache`/`cuda,grpc,auth,redis`),每种跑 `cargo test --features <组合>`。新增 `scripts/test-feature-matrix.sh` 本地脚本
- [ ] [T042] [P0] 跑 `cargo clippy --all-features --all-targets -- -D warnings` 修复全部告警。常见:unused_imports、clippy::needless_return、clippy::module_inception。验证:零告警
- [ ] [T043] [P1] 跑 `cargo fmt --all -- --check` 修复格式问题。在 `.pre-commit-config.yaml` 追加 `cargo fmt --check` hook

## Phase 6: 幽灵代码清理

- [ ] [T044] [P0] 运行 `npx gitnexus analyze --embeddings` 建立索引(若 `.gitnexus/` 不存在或 stale)。验证:`.gitnexus/meta.json` 中 `stats.embeddings` > 0
- [ ] [T045] [P0] 用 `gitnexus_impact` + `gitnexus_query` 分析未使用符号:对每个 `pub fn`/`pub struct`/`pub trait` 跑 `gitnexus_impact({target: "<symbol>", direction: "upstream"})`,d=0 (无调用者) 的为幽灵代码候选。手动验证每个候选(用户提示 gitnexus 有误报),确认真幽灵代码清单
- [ ] [T046] [P1] 删除确认真幽灵代码:更新 `src/lib.rs` 移除未使用的 `pub use`;移除未使用的 `pub fn`;将仅测试使用的 `pub` 改为 `pub(crate)`。每删除一个跑 `cargo test --all-features` 验证不破坏

## Phase 7: 工程规范与文档

- [ ] [T047] [P0] 重组 `tests/` 目录:新建 `tests/integration/`、`tests/perf/`、`tests/common/` 子目录。移动 `integration_test.rs` + `real_engine.rs` + `test_real_inference.rs` 到 `tests/integration/` 并合并为 `api_test.rs`。移动 `api_simulator.py` + `client_factory.py` + `test_api.py` + `test_api_real.py` + `real_service.py` + `services.py` + `conftest.py` + `fixtures.py` + `config.py` + `__init__.py` + `test_server_integration.py` 到 `tests/perf/`。删除 `test_api_real.py`(与 `test_api.py` 重复)
- [ ] [T048] [P0] 创建 `tests/integration/mod.rs` + `tests/common/mod.rs` 共享 fixtures。合并 `performance_test.rs` 到 `tests/perf/performance_test.rs`。验证:`cargo test --test integration` 通过
- [ ] [T049] [P1] 更新 `README.md`(中文,新主 README):版本号 0.2.0;新增"7 库生态"章节;新增"多协议接口"章节(HTTP/MCP/CLI);新增"新引擎支持"章节(TensorRT/OpenVINO);更新特性表;更新配置示例含 `[database]`/`[logging]`/`[flow_control]`/`[cache]` 段
- [ ] [T050] [P1] 更新 `README_EN.md`(英文,原 README):同步中文版所有变更。更新 `AGENTS.md`:新增 7 库依赖说明;新增 `VecboostError` 命名规范;更新 `pub mod` vs `pub(crate) mod` 边界;更新特性表
- [ ] [T051] [P1] 更新 `ARCHITECTURE.md`(若存在)或新建 `docs/ARCHITECTURE.md`:绘制新架构图(含 trait-kit Kit + 7 库);描述模块依赖图;描述多协议接口生成流程
- [ ] [T052] [P0] 派遣 subagent 跑 `diting` skill 进行全面代码审查:架构一致性、性能瓶颈、安全漏洞、过度工程。生成审查报告并修复所有 HIGH/CRITICAL 问题
- [ ] [T053] [P0] 派遣 subagent 跑 `tiangang` skill 进行 SAST 安全扫描:hardcoded secrets、SQL injection、unsafe eval、insecure deserialization。0 个 CRITICAL 才允许完成
- [ ] [T054] [P0] 派遣 subagent 验证代码与文档一致性:对照 `proposal.md` Scope 逐项检查;对照 `design.md` Decision 逐项检查;对照 `specs/` delta spec 逐项检查。生成不一致清单并修复

## Phase 8: 最终验证

- [ ] [T055] [P0] 跑 `cargo test --all-features -- --nocapture` 验证全部测试通过。覆盖率检查:`cargo tarpaulin --all-features --threshold 95` 必须 ≥ 95%
- [ ] [T056] [P0] 跑特性组合矩阵:`scripts/test-feature-matrix.sh` 验证 4 种组合全部通过。零告警
- [ ] [T057] [P0] 跑 `scripts/check_copyright.sh` 验证所有 `.rs` 文件版权头合规
- [ ] [T058] [P0] gitnexus 最终检测:`gitnexus_detect_changes({scope: "all"})` 确认变更范围与 spec 一致;`gitnexus_impact` 对所有新增 `pub` 符号跑影响分析,确认无 HIGH/CRITICAL 风险
- [ ] [T059] [P0] 生成最终变更报告:列出所有完成的任务、修改的文件数、新增的依赖、移除的自研代码行数、测试覆盖率提升数据。提交 `git commit -m "feat(vecboost): v0.2.0 ecosystem refactor with 7 library integration"`

## Phase 9: Convergence

<由 /specmark converge 追加>
