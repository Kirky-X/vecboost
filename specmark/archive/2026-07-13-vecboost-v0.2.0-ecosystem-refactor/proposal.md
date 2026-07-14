# vecboost-v0.2.0-ecosystem-refactor

## Motivation

VecBoost v0.1.x 当前自研了缓存(LRU/LFU/ARC/KV)、限流(令牌桶)、配置、日志等基础设施模块。这些模块功能完整但维护成本高、与企业级生态脱节。作者(Kirky.X)已在 crates.io 发布了 7 个企业级库(dbnexus/inklog/limiteron/oxcache/sdforge/trait-kit/confers),形成完整生态。本次变更将 VecBoost 从"自研一切"迁移到"集成自家生态",统一基础设施、降低维护成本、为多协议接口(HTTP+MCP+CLI)和扩展新引擎(TensorRT/OpenVINO)铺平道路。

同时,v0.1.x 存在若干技术债:错误类型命名不规范(AppError 应为 VecboostError)、`mod.rs` 暴露过多实现细节、tests 目录混合 Python/Rust 文件、版权头不统一、部分依赖未通过 feature 控制。本次变更一并清理。

## Scope

### 模块替换(7 库集成)

- **dbnexus 0.4** 替换所有持久化场景:用户存储(`src/auth/user_store.rs`)从内存迁移到 SQLite/PostgreSQL;审计日志(`src/audit/`)**v0.2.0 为双轨实现(文件 + DB 并存)**,完整 DB-only 迁移推迟到 v0.3.0;嵌入元数据表(`metadata-persistence` feature)推迟到 v0.3.0
- **inklog 0.1** 替换日志基础设施:替换 `tracing-subscriber` + 自定义 logger;启用 file+console sink;**AES-256-GCM 加密 + 三级降级 + 敏感数据脱敏 + 文件轮转等高级特性推迟到 v0.3.0**
- **limiteron 0.2** 替换 `src/rate_limit/`:迁移令牌桶到 limiteron 算法;移除自研 `token_bucket.rs`/`store.rs`;**`ban-manager` + `circuit-breaker` + `tower-middleware` feature 推迟到 v0.3.0**
- **oxcache 0.3** 替换 `src/cache/`:迁移 LRU/LFU/ARC/KV 到 oxcache backend;移除自研 7 个 cache 文件;**L2 Redis + 多实例 Pub/Sub 同步 + 全局 TTL 默认值推迟到 v0.3.0**
- **sdforge 0.4** 提供多协议接口:**v0.2.0 仅声明 feature 依赖,`#[service_api]`/`#[forge]` 宏未实际使用**,HTTP 路由手写、CLI 手写 clap、MCP 未实现;完整多协议生成推迟到 v0.3.0
- **trait-kit 0.3** 标准化模块接口:`Kit<Unbuilt> → Kit<Ready>` typestate 管理所有模块依赖;建立 `src/module_registry/`(6 个模块定义);**v0.2.0 仅建立 module_registry 模块,AppState 完整重构推迟到 v0.3.0**(详见 design.md D1 偏离记录);启用 `confers-macros` 集成
- **confers 0.4** 替换 `src/config/`:类型安全配置 + 热重载;迁移 `AppConfig`/`ModelConfig`/`ServerConfig` 到 confers derive;**配置加密存储(`encryption` feature)推迟到 v0.3.0**

### 错误统一与模块加固

- `AppError` → `VecboostError` 全局重命名(影响 lib.rs 公共 API,版本升 v0.2.0)
- 所有 `mod.rs` 仅保留接口(trait/枚举/数据结构),实现移至 `impl.rs` 或子模块
- 其他模块通过 `use crate::cache` 而非 `use crate::cache::lru_cache` 导入

### 多协议接口(规则 14)

- **Rust SDK**:`src/lib.rs` 导出公共 API,通过 feature 控制;**`http::run_server` 公共 API 推迟到 v0.3.0**(HTTP server 启动逻辑当前在 `main.rs`)
- **CLI**:**v0.2.0 为手写 clap 实现**(`src/cli/mod.rs` 用 `#[derive(Parser)]`),`sdforge::cli` 宏生成推迟到 v0.3.0
- **HTTP**:**v0.2.0 为手写 Axum 路由**(`src/routes/embedding.rs`),`#[service_api]`/`#[forge]` 宏生成推迟到 v0.3.0
- **MCP**:**v0.2.0 未实现**(无 `rmcp` 依赖),完整 MCP server 推迟到 v0.3.0

### 新引擎支持(规则 15)

- **TensorRT** 引擎(NVIDIA GPU,可选 feature `tensorrt`):**v0.2.0 为 stub 实现**(`tensorrt_engine.rs::new()` 始终返回 `InferenceError`),因 `tensorrt-sys` 缺 `libnvinfer.so` 编译失败,真实 crate 集成推迟到 v0.3.0
- **OpenVINO** 引擎(Intel CPU/GPU/VPU,可选 feature `openvino`):**v0.2.0 为 stub 实现**(`openvino_engine.rs::new()` 始终返回 `InferenceError`),因 `openvino-sys` 缺 `libopenvino_c.so` 编译失败,真实 crate 集成推迟到 v0.3.0
- **运行时引擎切换** `EmbeddingService::switch_engine()`:**v0.2.0 未实现**,推迟到 v0.3.0

### 质量与测试

- 测试覆盖率 ≥ 95%(规则 2);GPU 依赖模块用 `mockall` mock,CI 跑 CPU 路径
- 特性组合测试矩阵(规则 6):6 个核心特性 × 4 种组合 = 24 组(非全 64 组,按实际使用场景);**v0.2.0 实际 default = ["http", "oxcache", "limiteron"]**(见 design.md D9 偏离说明)
- 清理所有 clippy 告警(规则 5)
- diting 全面审查(规则 11):架构/性能/安全/简化

### 工程规范

- MIT 版权头统一(规则 8):所有 `.rs` 文件头为 `// Copyright (c) 2025 Kirky.X` + MIT 许可声明
- README 改名(规则 9):`README.md` → `README_EN.md`;`README_zh.md` → `README.md`
- tests 目录重组(规则 10):Python 性能测试移至 `tests/perf/`;Rust 集成测试合并到 `tests/integration/`;删除冗余 `real_*.rs`/`test_api_real.py`
- 依赖特性化(规则 16):所有可选依赖通过 `dep:` 语法控制;版本号用 `x` 而非 `x.x.x`(>=1.0 用 major 版本如 `1`,<1.0 保持 `0.x` 格式如 `0.3`)
- 文档同步(规则 21):代码变更同步更新 README/AGENTS.md/ARCHITECTURE.md

## Non-Goals

- **不重写推理引擎核心**:`src/engine/` + `src/service/` 的嵌入逻辑保持不变,仅替换基础设施
- **不引入新 ML 模型**:本轮聚焦 TensorRT/OpenVINO 引擎,不新增 BERT/MiniLM 等模型支持
- **不破坏 gRPC 接口**:gRPC proto 保持向后兼容,仅升级 tonic 版本
- **不做性能基准对比**:迁移前后性能对比放到下个 change(本轮聚焦正确性)
- **不引入 Kubernetes Operator**:部署清单仅更新依赖,不新增 CRD
- **不替换 Candle/ONNX 引擎**:Candle 保留为默认引擎,ONNX 保留为可选,新增 TensorRT/OpenVINO 为可选

## Clarifications

- **[scope]** Q: dbnexus 是 Sea-ORM 数据库抽象层,VecBoost 是嵌入向量推理服务。dbnexus 用于哪些场景?
  A: 全量替换所有持久化场景(用户存储/审计日志/嵌入元数据/限流计数器)

- **[breaking]** Q: AppError 改名 VecboostError 是 breaking change,版本号如何处理?
  A: 升级到 v0.2.0 + 全局重命名(遵循 SemVer)

- **[api]** Q: 第14项要求三种接口:Rust SDK + CLI + HTTP + MCP。MCP 接口如何实现?
  A: sdforge 宏生成 + feature 控制(`#[service_api]` 统一定义,编译时选协议)

- **[engine]** Q: 第15项要求增加其他模型/引擎支持。本轮优先加哪些?
  A: TensorRT(NVIDIA GPU) + OpenVINO(Intel)

- **[integration]** Q: 7 库集成策略?
  A: 日志用 inklog,限速用 limiteron,各库各司其职,而非 dbnexus 接管所有

## NEEDS CLARIFICATION

- **[testing]** GPU 依赖模块(device/cuda/metal)如何达到 95% 覆盖率 — CI 无 GPU 时无法跑真实路径 — 默认用 `mockall` 生成 mock,CPU 路径全覆盖,GPU 路径仅跑 API 签名测试

- **[features]** 特性组合测试矩阵的具体范围 — 6 特性 64 组合 CI 成本过高 — 默认测 4 种核心组合:`default` / `grpc` / `auth,redis` / `cuda,grpc,auth,redis`,其他组合仅跑 `cargo check`

- **[gitnexus]** gitnexus 索引是否已建 — 索引不存在时无法跑幽灵代码分析 — 默认先跑 `npx gitnexus analyze` 建索引,再分析
