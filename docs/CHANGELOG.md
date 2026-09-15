# 📋 VecBoost 更新日志

本文件记录 VecBoost 的全部重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## 📋 目录

- [Unreleased](#unreleased)
- [0.2.1 - 2026-09-06](#021---2026-09-06)
- [0.2.0 - 2026-07-24](#020---2026-07-24)
- [0.1.0 - 2025-12-15](#010---2025-12-15)

---

## [Unreleased]

### 新增

- **推理正确性**:`PoolingMode::{Cls,Mean,Max,Auto}` 完整实现(Auto 按模型名推断),mean 为 attention-mask 加权平均;全平台统一 HuggingFace `tokenizers`,删除自研 WordPiece 与 250 词静默回退;vocab_size 从模型 config.json 推导;`/v1/embeddings` usage 为真实 token 计数(audit-remediation)
- **Swagger UI**:`/api-docs/openapi.json` + `/swagger-ui/`(sdforge 动态生成 OpenAPI)(audit-remediation-gaps)
- **真实就绪探测**:`/health?depth=full`(DB SELECT 1 + 引擎 dummy 推理 + 限流器健康)(audit-remediation-gaps)
- **时间窗动态拼批**:`[pipeline.worker] batch_wait_ms`(默认 5ms,`0` 还原排空式),窗口内聚合请求凑满 `max_batch_size` 提前发出(absorb-colibri-optimizations)
- **批内去重**:`/embed/batch` 相同文本批内只推理一次,结果按索引回填(字节等同);指标 `vecboost_inbatch_dedup_ratio`(absorb-colibri-optimizations)
- **物理核线程调优**:tokio/rayon 按物理核生效,SMT/E-core 检测,`VECBOOST_NO_THREAD_TUNE=1` 关闭;多 socket 启动时输出 numactl 建议(absorb-colibri-optimizations)
- **GGUF 量化路径**:`[model] quantized` + `--features quantized-gguf`(Q8_0/Q4_K);质量门余弦中位数 ≥0.98/≥0.95(`tests/quantized_parity.rs`,模型经 `VECBOOST_GGUF_MODEL`)(absorb-colibri-optimizations)
- **向量输出量化**:`[semantic_cache] comparison_mode = i8|binary`,Hadamard 旋转(kv_tq.h 同构)粗筛 + 原始向量精确复验(absorb-colibri-optimizations)
- **多模型 LFRU 驻留**:`[model] max_resident_models` / `resident_memory_budget_mb`;heat<<8|recent 评分、25%+4 迟滞、热度持久化 `data/model_heat.json`(absorb-colibri-optimizations)
- **缓存 WAL 落盘**:`[embedding] persist_path` / `persist_max_bytes`,崩溃安全两段追加,启动回放重建(absorb-colibri-optimizations)
- **硬件感知规划**:`[device] auto_plan = true` 启动探测 RAM/物理核/GPU/模型生成保守计划(显式配置优先);`scripts/autotune.py` 坐标下降实测调优(absorb-colibri-optimizations)
- **只读诊断**:`vecboost doctor`——config/tokenizer/缓存持久层/线程/GPU/模型完整性(safetensors 头、`__metadata__`、gguf 魔数、hidden_size 配对;按内容角色匹配),有 FAIL 退出码 1(absorb-colibri-optimizations)
- **启动预热**:`--warmup N` 合成文本预热;`scripts/warmup_corpus.py` 暖语料回放(去重保序)(absorb-colibri-optimizations)
- **新增指标**:`vecboost_batch_size`、`vecboost_batch_wait_seconds`、`vecboost_inbatch_dedup_ratio`、`vecboost_stage_seconds{stage=tokenize|inference|pool}`(absorb-colibri-optimizations)

### 变更(破坏性,升级必读)

- **安全默认值**:`host` 出厂默认 `0.0.0.0` → `127.0.0.1`;`auth.enabled=false` 时绑定非回环地址将拒绝启动(逃生阀 `VECBOOST_ALLOW_INSECURE=1` 会打 ERROR 告警);`use_gpu` 出厂默认 `true` → `false`
- **登录收敛**:仅 `default_admin_username`(默认 admin)可登录;启用认证时必须配置 `VECBOOST_ADMIN_PASSWORD`,否则拒绝启动;任意用户名 + 共享口令登录被拒绝
- **XFF 信任反转**:`trusted_proxies` 为空时忽略 `X-Forwarded-For`(原为无条件信任);反代部署需显式配置 `trusted_proxies`
- **RBAC 接线**:`/api/1/model/*` 与 `/embed/file` 要求 admin 角色
- **`/embed/file` 收敛**:必须显式配置 `[server] grpc_allowed_roots`(不再回退 cwd);单文件上限 10 MiB;`text_preview` 仅 admin
- **Token 生命周期**:`token_expiration_hours` 缺省 1 小时(原 garrison 默认 30 天)
- **CSRF 默认**:跟随 `auth.enabled`(`AuthConfig` 默认 csrf.enabled=true)
- **`--config` fail-fast**:显式指定且文件不存在 → 报错退出(原静默回退默认配置)
- **缓存与分词行为**:缓存键绑定模型标识(升级后首轮缓存全 miss);HF tokenizers 分词结果与旧自研实现存在差异(更正确),存量向量需重建
- **CLI**:未知子命令报错退出(码 2),不再静默启动 HTTP 服务器;`--help` 可用

#### 升级迁移指南（旧行为 → 新行为 → 迁移动作）

| 变更 | 旧行为 | 新行为 | 迁移动作 |
|------|--------|--------|----------|
| 绑定安全 | `auth.enabled=false` 可绑定 `0.0.0.0` | 非回环绑定 + 无认证 → **拒绝启动** | 本地开发保持 `127.0.0.1`；受信网络容器设 `VECBOOST_ALLOW_INSECURE=1`（打 ERROR 告警）；生产启用 auth |
| 管理员密码 | `VECBOOST_ADMIN_PASSWORD` 可选，缺失时任意凭据登录 | `auth.enabled=true` 且缺失 → **拒绝启动** | 设置 `VECBOOST_ADMIN_PASSWORD`（≥8 位） |
| 登录用户名 | 任意用户名 + admin 口令可登录 | 仅 `default_admin_username`（默认 `admin`）可登录 | 客户端固定使用 admin 用户名 |
| XFF 信任 | `trusted_proxies` 为空时无条件信任 `X-Forwarded-For` | 为空时**忽略 XFF**，使用直连地址 | 反代部署显式配置 `trusted_proxies = ["10.0.0.0/8"]` 等 |
| RBAC | `/model/*`、`/embed/file` 仅需登录 | 要求 **admin 角色** | 业务用户使用普通账号即可调 embed；模型管理用 admin |
| `/embed/file` | 默认允许根 = 进程 cwd，回传文本预览 | 必须显式配置 `grpc_allowed_roots`；单文件 ≤10 MiB；`text_preview` 仅 admin | 配置允许根；客户端不再依赖 preview |
| Token 有效期 | 缺省 30 天（garrison 默认） | 缺省 **1 小时** | 长会话场景显式配置 `token_expiration_hours` |
| CSRF | 默认关闭 | 跟随 `auth.enabled`（显式 `csrf.enabled=false` 仍生效） | 纯 Bearer API 无影响 |
| `use_gpu` | 出厂配置 `true`（无 GPU 构建下告警回退） | 出厂 `false`；请求 GPU 但 feature 缺失 → WARN + CPU 回退 | GPU 构建用户显式开启 |
| `--config` | 路径不存在 → 静默回退默认配置 | **报错退出**（码 2） | 排查路径拼写 |
| 缓存键 | `text:{原文}`（不含模型） | `emb:{model}:{xxh3_128}`；模型切换清缓存 | 升级后首轮缓存全 miss（一次性） |
| 分词器 | Linux 自研 WordPiece（异常时回退 250 词表） | 全平台 HuggingFace `tokenizers`；加载失败**报错** | 存量向量与重建向量不兼容，需重建索引 |
| CLI | 未知子命令静默启动 HTTP 服务器 | stderr 用法提示 + 退出码 2 | 脚本若依赖旧行为需调整；`--help` 可用 |

**多副本边界**：推理路径无状态，auth 关闭时可水平扩展（限流/缓存为进程内语义）。`auth.enabled=true` 时认证会话存进程内存（oxcache DAO），**仅限单副本**；会话外置需 garrison db 后端补齐（见 [FAQ](FAQ.md) 与 [SECURITY](SECURITY.md)）。

**热重载语义**：配置文件变更会进行校验并打日志，重启后生效（非运行时热切换）。Kubernetes 场景用 ConfigMap 滚动更新。

### 修复

- 安全:admin 密码缺失时任意凭据可得 admin(P0);`/model/switch` 路径白名单;请求路径 unwrap 消除;`SecretKey` 零化 + 环境变量 keystore 只读;测试环境变量竞态(共享 ENV_LOCK)
- 性能:队列入队唤醒替代 5s 退避轮询;worker 排空拼批;缓存键 xxh3+模型命名空间;`/search` 批量推理
- 文档:README/API_REFERENCE 与实现不一致处全量纠错;doc_consistency_check 纳入门禁
- 交付:Dockerfile 兼容 edition 2024;LICENSE 补齐;CI 覆盖率硬门禁 + Python 场景测试 nightly;clippy unwrap_used 门禁

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
- **utoipa features 显式化**：`utoipa` 依赖显式添加 `macros` feature（`ToSchema` derive 宏必需），符合规则25"依赖必须通过特性显式使用"。

### 移除

- **手写 gRPC 实现**：删除 `src/grpc/`、`proto/` 目录。
- **手写 HTTP 路由**：删除 `src/routes/` 目录。
- **手写 CLI**：删除 `src/cli/` 目录。
- **CandleEngine.tensor_pool**：移除存在 mask 全零 bug 的张量池逻辑。
- **MemoryPoolManager.tensor_pool 死代码**：清理无消费者的张量池字段和方法。
- **tensorrt/openvino stub 特性**：移除未实现的引擎 stub。

### 修复

- **schema feature 单独启用时编译失败**：`utoipa` 5.5.0 内部 `schema.rs` 引用 `crate::utoipa::Number`（被 `cfg(feature="macros")` 门控），不启用 `macros` feature 会导致 `schema` feature 单独编译失败。显式给 `utoipa` 添加 `macros` feature 修复此问题。
- **schema-only 模式下 http 依赖代码未 gate**：`src/error.rs` 中 17 个 `test_into_response_*` 测试、3 个 `sanitize_error_message` 边界测试、`use regex::Regex` 导入、`src/lib.rs` tests 模块的 http-only imports、`src/registry/tests.rs` 中 `PrometheusCollectorModule` 相关测试和辅助函数缺少 `#[cfg(feature = "http")]` gate，导致 `schema` 单独启用时编译失败。已全部补齐 gate。
- **config 测试环境变量竞争**：`test_confers_load_from_minimal_toml` 和 `test_confers_toml_overrides_defaults` 缺少 `ENV_LOCK`，与 `test_apply_security_env_overrides_*` 测试并行执行时会因 `VECBOOST_JWT_SECRET` 环境变量竞争而失败。已补齐 `ENV_LOCK` 和环境变量清理。
- **worker 测试 flaky**：`test_worker_loop_exits_immediately_when_not_running` 在 `current_workers() == 0` 后立即检查 `is_alive`，但 `worker_loop` 先 `decrement_worker_count` 再设置 `is_alive = false`，存在 TOCTOU 竞争。改为等待 `is_alive == false && current_workers() == 0` 同时满足。
- **bin target 在 schema-only 模式下编译失败**：`src/main.rs` 是 HTTP server 入口（`axum::serve`），整体依赖 `http` feature，但 `Cargo.toml` 未给 `[[bin]]` target 配置 `required-features`，导致 `cargo check --no-default-features --features schema`（含 bin）编译失败。已添加 `[[bin]] required-features = ["http"]`，与 `examples/Cargo.toml` 已有模式一致。
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
