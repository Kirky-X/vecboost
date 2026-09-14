# 更新日志

本文件记录 VecBoost 的全部重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [Unreleased]

_暂无变更。_

---

## [0.2.1] - 2026-09-06

### 新增

- **i18n 国际化系统**：完整 ICU+Fluent 兼容双语支持（中英），包含：
  - 轻量 FTL 解析器（`src/i18n/bundle.rs`），`include_str!` 编译期嵌入，114 个翻译键（errors.ftl 17 + messages.ftl 97）
  - 三级 locale 优先级：显式 locale > 请求级（`Accept-Language` 中间件 + tokio task_local）> 全局默认
  - `VecboostError::IntoResponse` 通过 `error_code()` + `tr_with_args()` 全翻译
  - `Display` trait 手动实现使用 `tr_with_args()`，消除 thiserror 英文前缀
  - P0（HTTP JSON 硬编码）+ P1（启动/OOM/metrics）+ P2（Display/engine）全量清零
- **Rerank 重排序**：HTTP/gRPC/CLI 三协议完整支持：
  - `InferenceEngine` trait 新增 `rerank()` / `rerank_batch()` / `supports_rerank()` 默认实现
  - `RerankService`（`src/service/rerank.rs`）+ `RerankConfig` 配置
  - `src/api/rerank.rs` forge/cli/grpc 端点，11 个 gRPC 方法增至含 rerank/rerank_batch
- **语义缓存**：`SemanticCache`（`src/cache/semantic_cache.rs`）三级查询：精确匹配 → trigram Jaccard 语义搜索 → 模型推理回填
- **BF16 精度推理**：`Precision::Bf16` 支持，Candle 引擎 BF16 推理路径
- **SIMD 向量化相似度**：`src/utils/vector.rs` 向量化余弦相似度计算
- **连续批处理调度**：`ContinuousBatchLoop`（`src/device/continuous_batch.rs`）持续收集批次并处理
- **GPU 内存分页**：`WeightPagingManager`（`src/device/memory_paging.rs`）LRU-K 策略
- **Matryoshka 自适应维度**：`information_retention_rate()` + `recommended_dimension()` 公共 API
- **加密配置模块**：`src/config/encryption.rs` AES-256-GCM 密钥管理
- **认证子系统重构**：拆分为 `src/auth/` 多文件接口化架构（middleware/types/csrf）
- **场景测试套件**：`tests/scenario/` 10 个 Python 场景测试 + `scripts/run-scenario-tests.sh`
- **Library 模式**：`schema` feature 独立、HTTP feature gate 分离，支持非 HTTP 库模式使用

### 变更

- **依赖升级**：h2 0.4.15→0.4.19（RUSTSEC-2026-0258 DoS 修复）、自研库迁移到本地路径
- **Rust 2024 迁移**：let-chain 语法、`#[allow(dead_code)]` 补充 reason
- **代码质量**：魔法值提取常量、集合预分配、JoinSet 统一管理、block_on 重构

### 修复

- **Tokenizer 中文分词**：非 macOS 平台加载实际模型词汇表（30K+ 词）、WordPiece 逐字符 UNK 回退、encode 保留实际字符
- **错误状态码映射**：`to_api_error` 正确映射 ValidationError→400、ModelLoadError→424、NotFound→404
- **认证白名单路径**：PUBLIC_PATHS 与 sdforge 路由前缀 `/api/1/` 对齐
- **限流中间件**：注入 RateLimitEnabled、无条件挂载、与 auth 解耦
- **Prometheus 指标**：添加 HTTP 指标记录中间件 + 路径归一化防 label 基数膨胀
- **gRPC 启动**：SdforgeLimiteronAdapter 超时保护 10s
- **优雅关闭**：保存 config watcher 句柄，关闭时 abort
- **HF Hub 镜像**：检测 HF_ENDPOINT 镜像端点 + ETag 不兼容警告
- **schema feature 编译**：feature gate 修复，`schema` 单独启用编译通过
- **杂项**：多字节字符安全切片、fallback 路径 repo_id 校验、CLI 示例参数格式

### 安全

- **h2 DoS 漏洞**：RUSTSEC-2026-0258 修复
- **vuln-0009**：HF Hub repo_id 格式校验
- **安全加固**：认证/限流/审计级联修复、Python 测试脚本安全扫描修复

---

## [0.2.0] - 2026-07-24

### 新增

- **schema feature 单独启用时编译失败**：`utoipa` 5.5.0 内部 `schema.rs` 引用 `crate::utoipa::Number`（被 `cfg(feature="macros")` 门控），不启用 `macros` feature 会导致 `schema` feature 单独编译失败。显式给 `utoipa` 添加 `macros` feature 修复此问题。
- **schema-only 模式下 http 依赖代码未 gate**：`src/error.rs` 中 17 个 `test_into_response_*` 测试、3 个 `sanitize_error_message` 边界测试、`use regex::Regex` 导入、`src/lib.rs` tests 模块的 http-only imports、`src/registry/tests.rs` 中 `PrometheusCollectorModule` 相关测试和辅助函数缺少 `#[cfg(feature = "http")]` gate，导致 `schema` 单独启用时编译失败。已全部补齐 gate。
- **config 测试环境变量竞争**：`test_confers_load_from_minimal_toml` 和 `test_confers_toml_overrides_defaults` 缺少 `ENV_LOCK`，与 `test_apply_security_env_overrides_*` 测试并行执行时会因 `VECBOOST_JWT_SECRET` 环境变量竞争而失败。已补齐 `ENV_LOCK` 和环境变量清理。
- **worker 测试 flaky**：`test_worker_loop_exits_immediately_when_not_running` 在 `current_workers() == 0` 后立即检查 `is_alive`，但 `worker_loop` 先 `decrement_worker_count` 再设置 `is_alive = false`，存在 TOCTOU 竞争。改为等待 `is_alive == false && current_workers() == 0` 同时满足。
- **bin target 在 schema-only 模式下编译失败**：`src/main.rs` 是 HTTP server 入口（`axum::serve`），整体依赖 `http` feature，但 `Cargo.toml` 未给 `[[bin]]` target 配置 `required-features`，导致 `cargo check --no-default-features --features schema`（含 bin）编译失败。已添加 `[[bin]] required-features = ["http"]`，与 `examples/Cargo.toml` 已有模式一致。

### 变更

- **utoipa features 显式化**：`utoipa` 依赖显式添加 `macros` feature（`ToSchema` derive 宏必需），符合规则25"依赖必须通过特性显式使用"。

## [0.2.0] - 2026-07-24

### 新增

- **sdforge 统一接口生成**：HTTP/gRPC/MCP/CLI 四种协议通过 `#[forge(...)]` 宏从 `src/api/embedding.rs` 单一源定义生成，消除手写协议代码。
- **gRPC 迁移到 sdforge**：手写 tonic gRPC 实现替换为 `#[forge(grpc_method = "vecboost.*")]` 宏封装，支持 9 个 gRPC 方法（embed/embed_batch/compute_similarity/embed_file/model_switch/get_current_model/get_model_info/list_models/health_check）。
- **gRPC 配置项**：新增 `grpc_max_connections`、`grpc_timeout_seconds`、`grpc_require_auth`、`grpc_allowed_roots` 配置，支持 JWT 认证和速率限制。
- **Bert/XlmRoberta 模型架构支持**：通过 `ModelArchitecture` 枚举自动检测模型类型。
- **Matryoshka 截断重归一化**：截断维度后自动调用 `normalize_l2`，保证余弦相似度正确。
- **vuln-0009 安全加固**：HF Hub `repo_id` 格式校验统一在 `src/utils/hf_hub.rs`（`is_valid_hf_repo_id` + `build_hf_repo`），覆盖所有远程下载入口。
- **协议无关 handler**：提取 `*_handler` 函数消除 gRPC/HTTP/CLI 间的代码重复。
- **批量大小校验**：`validate_batch_size` 使用 `EmbeddingConfig.max_batch_size` 限制。

### 变更

- **依赖升级**：hf-hub 0.4→1.0、prometheus 0.13→0.14、tokio 1.52→1.53、aes-gcm 0.10→0.11。
- **default feature 简化**：`default = ["http"]`（原 `["http", "oxcache", "limiteron"]`）。
- **必选依赖**：`confers`、`inklog`、`oxcache`、`limiteron`、`trait-kit` 改为始终启用，无需 feature 开启。
- **sdforge 本地依赖**：临时使用 `path = "../sdforge"`（v0.4.7，含 tokio 1.53 支持），待 sdforge 0.4.7 发布到 crates.io 后切回。

### 移除

- **手写 gRPC 实现**：删除 `src/grpc/`、`proto/` 目录。
- **手写 HTTP 路由**：删除 `src/routes/` 目录。
- **手写 CLI**：删除 `src/cli/` 目录。
- **CandleEngine.tensor_pool**：移除存在 mask 全零 bug 的张量池逻辑。
- **MemoryPoolManager.tensor_pool 死代码**：清理无消费者的张量池字段和方法。
- **tensorrt/openvino stub 特性**：移除未实现的引擎 stub。

### 修复

- **Matryoshka 截断未重归一化**：截断后向量非单位向量导致余弦相似度计算错误。
- **attention_mask 张量池 bug**：原张量池路径 TODO 未回填数据导致批量推理 mask 全零。
- **多字节字符切片 panic**：`sanitize_secret`、`sanitize_jwt_secret`、`mask_value`、`text_preview` 使用 `floor_char_boundary`/`ceil_char_boundary` 确保 UTF-8 安全切片。
- **fallback/onnx/recovery 路径缺少 repo_id 校验**：统一通过 `build_hf_repo` 处理。

### 安全

- **vuln-0009**：HF Hub `repo_id` 格式校验防止恶意配置注入和路径遍历攻击。
- **UTF-8 安全切片**：所有密钥脱敏和文本预览函数使用字符边界安全切片。
- **gRPC 路径校验**：`PathValidator` 从配置读取允许的根目录，默认拒绝 `/`、`/etc` 等敏感目录。

## [0.1.0] - 2025-12-15

VecBoost 初始发布。
