# 📋 VecBoost 更新日志

本文件记录 VecBoost 的全部重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## 📋 目录

- [Unreleased](#unreleased)
- [0.3.0-rc.2 - 2026-10-15](#030-rc2---2026-10-15)
- [0.3.0-rc.1 - 2026-09-22](#unreleased)
- [0.2.1 - 2026-07-30](#021---2026-07-30)
- [0.2.0 - 2026-07-24](#020---2026-07-24)
- [0.1.2 - 2026-01-16](#012---2026-01-16)
- [0.1.1 - 2026-01-15](#011---2026-01-15)
- [0.1.0 - 2026-01-10](#010---2026-01-10)

---

<a id="unreleased"></a>

## [Unreleased]

### 变更

- **主推理引擎切换为 onnxruntime（ort），candle 降为 opt-in 回退后端（onnx-primary）**：
  - `default = ["http", "onnx"]`——默认构建即 ort 主引擎且依赖树不再含 candle（zip 解析风险随之出树；candle 三件套转 optional；zip 全量 yank 事件于 2026-10-16 被上游解除，candle 回指官方 0.11，应急 fork 已删除）
  - `EngineType` 枚举与全部 task 路由语义不变（wire/API 契约零变化）；未启用 `candle` feature 而配置 `engine_type="candle"` 时启动期显性拒绝（`startup-engine-candle-not-compiled`，禁止静默降级）
  - **迁移提示**：旧配置 `model_path` 指向 safetensors 目录（无 `engine_type` 字段）升级后启动报错——需显式 `engine_type = "candle"` 并以 `--features candle` 构建，或迁移至 onnx bundle
  - feature 语义调整：`cuda` 不再隐含 candle GPU 后端（需叠加 `candle-cuda`）；`metal`/`mkl`/`accelerate`/`quantized-gguf` 均隐含 `candle`；`transcribe` 仍经 `ort/cuda`（不受影响）
- **缺省 HTTP body 上限实际值 2MB→5MB**：修复隐藏硬顶后 `body_limit_mb`（缺省 5）真实生效（此前提取器层 2MB 先于配置拒绝）
- **审计日志 DB 模式新增文件双写**：db 构建下 `logs/audit.log`（轮转）与 `audit_logs` 表并存（此前仅写表，文件运维可观测面缺失）
- **并发同文请求合并（in-flight 去重）**：pipeline 入口对同文同 normalize 请求只推理一次，结果副本多播——同文并发逐位一致由「只算一次」结构性保证
- **Swagger UI 收敛为 `swagger-ui` feature（默认不编译，挂主服务端口，单二进制一次构建）**：`http`/`openapi` 不再携带 `dep:utoipa-swagger-ui` 与 `sdforge/docs`；交互页面与 `/api-docs/openapi.json` 仅在 `--features swagger-ui` 构建时挂载（与主服务同端口，一次 `cargo build` 产出完整二进制），`openapi` feature 保持为规范程序化导出（sdforge/openapi）。注：Cargo 锁文件为全 feature 并集解析——optional 依赖（解析层）始终存在，feature 隔离只裁剪编译；2026-10-17 实测 zip 3.0.0/8.6.0 已可正常解析（前一天全量 yank 已解除）。

### 修复

- **onnx embedding 真机路径三处兼容缺陷**（此前无真实资产测试，基准真机首跑暴露）：单条输入 rank-1→`[1, seq]` 二维（标准 BERT ONNX 图规格）；三输入图（optimum/transformers.js 导出）按 `session.inputs()` 探测补喂全零 `token_type_ids`；rank-2 输入对应 rank-3 输出索引口径对齐
- **onnx embedding pooling 硬编码 mean**（对 BGE 系静默改变语义且 `ModelConfig.pooling_mode` 配置无效）→ 构造期按模型名推断（与 candle 同一 `infer_pooling_mode` 单一事实源）+ Cls/Mean/Max 三模式支持；修复后同源双导出逐文本余弦 = 1.000000
- **onnx HF 拉取不兼容 `onnx/` 子目录 repo**（Xenova 风格直连必失败）→ 候选链 `model.onnx → onnx/model.onnx → model_quantized.onnx → onnx/model_quantized.onnx`
- **axum 提取器层 `DefaultBodyLimit` 2MB 硬顶**掩盖 `[server].body_limit_mb` 配置（>2MiB 静默失效，转录 48s 音频 413）→ 两层限制统一到同一配置口径
- **FT-03 并发同文位级不一致**：min_workers>1 下同文落多批、GEMM 批形状差异致位级漂移→ in-flight 去重结构性修复
- **R-auth-009c 限流审计文件面缺失**：db 构建走 DB-only 写入，场景锚定 `logs/audit.log` 恒空 → DB 双写修复
- **批处理 OOM 回退未消费真实配置（回退语义失真）**：`embed_chunks_concurrent` 的单块路径此前自带内联回退循环——以 `ModelConfig::default()` 调 `try_fallback_to_cpu`、绕开 `ModelManager` 门控，重试上限另有本地常量 1（与单文本路径的 2 不一致）。现统一委托 `common::handle_oom_fallback`（真实 `ModelConfig`+`ModelManager`、重试上限 2 次）；库模式无 manager 时按 `oom-no-fallback-available` 显性失败而非假装回退。连带删除失效的 `log-service-oom-error-detected-batch` / `log-service-successfully-fell-back-cpu-2` 两条 FTL 键
- **审计 flush 静默误报成功**：`AuditLogger::flush()` 原先 `let _ = rx.await`——writer 因连续写失败熔断退出（`ERROR_THRESHOLD=10`）后在途 Flush 无 ack，flush 照样返回 `Ok(())`，关停路径误判事件已落盘。现三段显性失败：send 失败（writer 已死）/ ack 超时（`FLUSH_ACK_TIMEOUT=5s`，writer 存活但失联）/ oneshot Err（处理中途退出），新增 `audit-flush-writer-unavailable`、`audit-flush-timeout` 两条 FTL 键（`registry/impl_.rs` 关停路径已消费该 Err，无需改动）
- 引擎/决策工厂测试隔离缺陷（2 处）：`test_create_decision_task_dispatches_to_candle_engine`、`test_process_decision_engine_panic_maps_to_inference_error` 断言依赖 i18n 初始化顺序 → 各自显式 `i18n::init()`

### 质量与测试

- **跨引擎对账基线**：embedding 精度（同源双导出逐文本余弦，阈 0.999）、决策精度（|Δlogit| ≤ 1e-4，既有 parity 闸门）、决策延迟观测（candle 455ms vs onnx 118ms/题，informational）
- **性能基准**（同机 CPU，criterion）：embedding 单条 onnx 2.89ms vs candle 71.99ms（~25×）、batch×32 27.65ms vs 176.32ms（~6.4×）；`embed_throughput_bench` 支持 `VECBOOST_BENCH_ENGINE` 引擎选择 + ID 含引擎名 + `expect` 防「快速失败假性能」
- **新增测试**：`onnx_embed_e2e`（真机 embedding 契约与单条-批量一致性）、`engine_backend_parity`（精度对账 + 性能观测）、场景 `test_transcribe.py`（HTTP 转录 e2e：真机 ASR/体量/防线 4 用例）、gRPC `vecboost.decide`/`vecboost.transcribe` 方法测试
- **测试基建修复**：场景 conftest 二进制路径动态推导（修 worktree 测错目标）、集成/场景配置模板显式 `engine_type`、`scripts/link-assets.sh`（worktree 资产链接）、`download_model --onnx/--translate`（embedding/翻译 bundle 资产获取）
- **批处理 OOM 回退契约测试**：`RecordingOomEngine` mock 钉住回退确实消费真实 `ModelConfig`（断言 `name != "default"`、`model_path` 为注入值），且无 manager 时不发起回退（`try_fallback_to_cpu` 零调用）
- **审计 flush 契约测试**：writer 不可用快速 Err / ack 后事件已落盘 / 熔断退出后 flush 快速失败且 break 前事件留存于轮转件

### 已知限制

- 翻译 golden 闸门（`translate_e2e`）需 en-zh + en-es 双 bundle（后者可经 `--translate Xenova/opus-mt-en-es` 获取）；en-zh 真机断言已由集成测试覆盖
- gRPC 转录请求受 sdforge Call 层 1MB 应用上限约束（大音频走 HTTP 面）；`samples` 数组协议对大音频（48s≈14MB JSON）不友好，流式/分片传输待协议演进
- candle 单条前向在其 CPU 实现下固有较慢（batch 摊销前 ~85ms@bge-small），已降为回退后端不投入优化

---

<a id="030-rc2---2026-10-15"></a>

## [0.3.0-rc.2] - 2026-10-15

### 新增

- **转录服务（transcribe 引擎与服务面并入主 crate）**：自研转录引擎（自写 fbank + conv/encoder + 28 层 KV-cache 贪心解码 + BPE 还原；自写 silero VAD 与 `--vad-only` 通路；VAD 段间并行 4 workers；全局线程池/EP 旋钮定档，全片 RTF 0.1535）；HTTP/gRPC 服务面接线（`/api/1/transcribe` 端点恒注册 + 装配门 400 + 背压 503、`[transcribe]` 配置段）；CLI 子命令（WAV 路径入参经 `read_wav_16k_mono` 转域请求）；`transcribe-poc` 收敛为薄壳 CLI（推理改调 `vecboost::transcribe` 公开门面，stdout 五行契约，PyO3 torch 后端对拍保留）
- **翻译推理补强**：译前 token 保护 + 译后回填（DTI 缩写/arXiv 标识符不再模型级丢失，占位符 ⟦Zn⟧ 两级回填）；600+ 字符长输入句级并行（4 worker，最差论文 155.7s→22.0s，7.1×，并行失败整请求显性报错禁 Partial 丢句）；M2M100/NLLB-200 多厂商家族；两遍解码（整段直解 + 退化按句救援）；EOS 抑制窗口 `min_new_tokens`（修复贪心提前截断）；出口后处理链（繁简归一化/符号修复/输出替换表）；`translate_ab` A/B 择优 harness；翻译/语言检测 Criterion 基准与基线数据
- **决策任务接入优先级队列**：Overloaded 背压与 `vecboost_decision_seconds` 指标补齐

### 修复

- **onnx 路径启动即崩（自 v0.2.1）**：ort feature 由 `alternative-backend` 改 `load-dynamic`——前者要求使用方自供 OrtApi 且全仓无调用点，首次构建 Session 即 panic abort；运行时经 `ORT_DYLIB_PATH` 指向 libonnxruntime ≥1.23（锁定 1.28.2），见 `3rdparty/onnxruntime/README.md`
- **ort rc.14 进程级 Environment 单例**：rc.14 每进程仅允许构建一个 `Environment`，`build_session` 改 `OnceLock` 进程级共享（并发竞争败者重查而非误报）——同进程构建第二个引擎即崩，由解锁的真实翻译 bundle 测试暴露
- **warm_up_cache 键方案漂移**：预热以裸文本为键写入、读路径查询 `emb:{model_id}:{xxh3}` 格式键，预热条目永不可读——抽 `cache_key_for` 统一构造并加回归测试钉死
- **token 缺省双轨漂移**：`DEFAULT_TOKEN_EXPIRATION_SECS=3600`（"缺省 1 小时"）实际不可达，改由 `DEFAULT_TOKEN_EXPIRATION_HOURS` 单一事实源派生（86400），文档/测试/示例统一 24 小时
- **process_search_batch 多 chunk 全局下标**：第二 chunk 向量永不落位且错误覆盖第一 chunk
- **RateLimitExceeded 直返路径统一 503**：429 归 auth 限流中间件，同一错误不再双映射

### 变更

- **自研依赖升级至 crates.io 最新 rc**：trait-kit 0.5.0-rc.9 / confers 0.6.0-rc.10 / inklog 0.3.0-rc.9 / oxcache 0.5.0-rc.8 / limiteron 0.3.0-rc.7 / dbnexus 0.6.0-rc.9 / garrison 0.9.0-rc.3 / sdforge 0.5.0-rc.8；升级适配 confers Zeroizing、garrison 发布版 error 新变体、sdforge UnifiedError/gRPC Status 契约与 utoipa 5.5/6.0 显式消歧
- **ort 2.0.0-rc.13 → rc.14**：`Session::builder(&env)` 显式环境、CUDA EP 迁至 `ort::ep::CUDA`、CUDA EP 门控依据 cuda→transcribe（ort/cuda 实际经 transcribe 特性引入）
- **重构与收敛**：超长函数拆分 10 批清零（启动编排/worker 状态机/推理热路径等）；公共 ONNX Session 构建层收编三处漂移（probe/precision 单一事实源）；single-flight 抽取共享模块（SemanticCache 与 OxCacheBackend 复用）；死代码清理与测试资产去重
- **依赖特性裁剪**：fluent-bundle/writeable `default-features = false`；utoipa-swagger-ui vendored（离线环境可编译）
- **GitHub actions 升级**：codeql-action v4.38.3、setup-buildx v4.4.1、metadata-action v6.2.0 等

### 构建与 CI

- Docker 工作流修复：Dockerfile 对齐仓库根 context（修复每次 push 均红）、HF_HOME 指向可写卷修复容器启动崩溃、Trivy SARIF 上传补 security-events:write、deploy 腿 GHCR/Docker Hub 拆分按凭据条件发布；scenario CI 补 protoc

### 质量与测试

- 行覆盖率达 **90.89%**（llvm-cov，CI 同特性面）；内联单元测试 2237 → 2277（本轮 +124：auth 四 handler 全分支覆盖、翻译真实 bundle 端到端、pipeline SLA 收割/取消传播、doctor/device/utils 边界与错误分支等）
- 文档一致性检查器泛化：CLI 白名单 5 → 全部 forge(cli=true) 名；env 消费面纳入 scripts/ 运维脚本（消除 memcap 旋钮 8 处误报）
- 发布前 SAST 整改：transcribe-poc 锁文件 rustls 对齐 0.23.46（4 条 MEDIUM）并同步 ort rc.14；`scripts/autotune.py`/`warmup_corpus.py` 补 http(s) scheme 守卫（bandit B310）；pyo3 0.22.6 已知漏洞（RUSTSEC-2026-0176 等）经评估**接受风险**——仅存在于非发布的 transcribe-poc 本地基线工具，升级需 pyo3 0.22→0.29 major 迁移，待其 /home/dev 解释器钉定解除后处理

### 评估归档

- trait-kit 统一 SDK 封装评估：技术可行但本次不实施（defer_with_proposal；P1 元数据诚实化 → P2 健康探测统一 → P3 装配收敛 → P4 资源契约，前置条件与风险见 [TRAIT-KIT-SDK-ASSESSMENT](TRAIT-KIT-SDK-ASSESSMENT.md)）

<a id="unreleased"></a>

## [0.3.0-rc.1] - 2026-09-22

### 修复（fix-audit-defects-r1：六主题审计 35 项发现全量闭环）

- **调度正确性**：优先级老化不再搁浅队首（等待超阈值的队首被兜底服务而非整级永久跳过）；新增 `dequeue_expired` 按请求 SLA 收割排队请求，杜绝"队首被跳过后永久搁浅+队列/响应通道泄漏"；批组装窗口内到达自身 timeout 的请求出批时完成 504 而非伪装限流
- **批量推理正确性**：candle fp32 批量池化按 `pooling_mode` 分派（此前硬编码 CLS，Mean/Max 模型批量结果错误，单/批余弦仅 ~0.5）；ONNX mean 池化除数不再随 hidden_size 放大；`max_seq_len==0` 显性报错替代静默零向量
- **引擎出口契约**：统一为"L2 归一化向量"——量化引擎补归一化、candle 单条路径补归一化、trait 文档修正（`normalize_l2` 幂等，服务层行为不变）
- **NaN/Inf 全链路防线**：`normalize_l2`/`cosine_similarity` 对非有限输入显性报错；引擎输出→缓存写入→响应出口逐层 `is_finite` 校验，坏向量不再入库或直达客户端
- **语义缓存**：内部键加入 model_id 且语义索引按模型隔离（跨模型不再互相污染）；模糊命中阈值独立为 `fuzzy_threshold`（默认 0.85，`similarity_threshold` 不再混用两种量纲）；模糊命中打点 `vecboost_semantic_cache_fuzzy_hits_total`；`comparison_mode` 标记废弃；`unload_model`/`switch_model` 均清语义缓存；切模型同步传播到 rerank（`replace_engine`）；rerank 缓存键哈希化消除拼接歧义并纳入模型名
- **缓存健壮性**：生产 TTL 默认 3600s（`OxCacheBackend::with_ttl`）；语义索引读路径 touch `last_access`（驱逐从 FIFO 修正为真 LRU）；WAL 回放按文件序号对齐 committed（前缀内损坏记录不再使尾部未提交记录复活）；per-key single-flight（并发同文本只推理一次）
- **向量计算**：同条件基准实测证伪"伪向量化"推断——原单累加器实现在基线构建下已被 LLVM 向量化（1024 维 353ns）；4 独立累加器改写反而 2.2× 劣化，故循环体保持原实现，`rotate` 尾部未旋转段一次性告警；PERFORMANCE.md 补 RUSTFLAGS 向量化说明与 fat-LTO 代码生成漂移观察
- **GPU 内存**：CUDA 池记账三修（Drop 归还计数 / free 失败不归还+error 日志 / clear 不再假清零）；内存上限真实执法（`[model] memory_limit_bytes` 配置时模型加载前检查、超限显性拒绝，加载后按目录大小记账）；显存查询失败按 0 可用+warn（不再伪装充足）；CUDA 版本改真实探测（NVML `sys_cuda_driver_version` / nvidia-smi banner 解析，替代伪造的 `Some(11)`）；删除未接线的 `tensor_pool.rs` 孤儿文件；分页预取完成路径补预算检查+回滚，换入驱逐改多 victim 循环
- **pipeline 韧性**：worker 推理 panic 隔离（批内请求统一补显性错误，worker 存活）；客户端断连取消传播（`CancellationRegistry`，出队丢弃已取消请求释放槽位）；服务端超时独立 `RequestTimeout` 错误码（HTTP 主路径 503+Retry-After=1，`VecboostError` 直返路径 504，不再伪装 400/429）；优雅停机排空队列（排队请求立即收到 shutting down 503）+ `drain_timeout_secs` 可配；扩缩容 monitor 生产启动（`WorkerManager::start`）+ 新指标 `vecboost_queue_wait_seconds`；移除无消费者的 `WorkerTask::ProcessRequest` 死分支
- **测试资产**：`model_snapshot_regression` SKIP 判定对齐权重存在性（与 lib 测试同口径）；pytorch bin 加载测试补前置检查；补齐 bge-small-en fp32 权重

### 修复（G1 评审残留闭环）

- **内存上限执法接线**：`AnyEngine` 的 `InferenceEngine` 实现补 `attach_memory_limit_controller` 转发（此前漏转发，服务层 switch_model 与启动期接线的控制器全部落在 trait 默认 no-op 上，Candle/ONNX 引擎内 Exceeded/Critical 自动 CPU 回退链路静默失效）
- **输入面收紧**：`/v1/embeddings` 与决策请求（name/options/state 字符串叶子）对 U+2028/U+2029（LINE/PARAGRAPH SEPARATOR）从接受改为 400——渲染级日志混淆面与 `\n` 同罪；错误回显同步将这两字符字面量化（回显单行化不依赖前置校验，防御纵深）。存量数据入库前请清洗（说明见 USER_GUIDE「输入字符防线」）
- **决策请求校验**：choice 型 option 拒绝空/纯空白串（与 name 同口径，消除按字符串匹配消费的歧义）
- **量化路由口径**：`switch_model` 的 `.gguf` 路径判定改复用 `should_use_quantized_engine`（大小写不敏感，与启动路径一致；`"MODEL.GGUF"` 此前会分叉到 safetensors 路径报错）
- **配置显性失败**：引擎级 `ModelConfig` 反序列化加 `deny_unknown_fields`（键名拼错如 `tsak` 显性报错，不再静默忽略后回落默认值）；TOML `[model]` 段的未知键仍为静默忽略，已在 USER_GUIDE 配置节标注

### 新增

- **翻译接口（add-translation-and-langdetect）**:`[translation]` 段 `[[translation.pairs]]` 多语言对配置（启动期急切构建、pair 上限 16、任一 bundle 缺失启动期 fail-fast），`/api/1/translate` 端点按 `(source_language, target_language)` 精确路由（未配置对 400 并回显已配置 pairs 全集），`source_language` 缺省经 whatlang auto 检测（不可靠 400 `translate-detect-failed`，检测语言回填 `detected_source_language`）；Marian/opus-mt seq2seq 引擎（`src/engine/onnx_translate.rs`，encoder+decoder_model_merged 双 Session 贪心解码，缓存分支禁用——Xenova merged 图导出缺陷实测，无缓存全量重算 O(L²) 有界）；token 预算双层校验（服务层预判 + 引擎 encode 后复检，超限显性拒绝）；E2E 黄金闸门 `tests/translate_e2e.rs`（20 句 golden 完全相等 + 贪心确定性 + 质量输出 `out/translate-quality.json`）；四协议面（HTTP/gRPC `vecboost.translate`/MCP `translate`/CLI）、驻留记账 Translate 臂（磁盘 × 2.0 保守系数，int8 无标定声明）与 USER_GUIDE/API_REFERENCE 文档同步
- **语言检测接口（add-translation-and-langdetect）**:`/api/1/detect` 端点与 `src/text/langdetect.rs` 检测核心（whatlang 0.18，70 变体 ISO 639-3→639-1 穷举映射、无等价回退 639-3 码）；`DetectService` µs 级纯 CPU 不走 spawn_blocking；不可判定返回 200 + `language: null` + `is_reliable: false`（显性表达）；`allow` 白名单收窄（非法码显性 400）；翻译 auto 检测与该端点共用同一函数（单一事实源）
- **项目定位变更**：从嵌入向量服务改为多功能推理服务（嵌入/重排/决策/翻译/语言检测）——Cargo `description`/`keywords`、双语 README 定位标语与特性列表、API_REFERENCE（翻译/检测小节 + gRPC 方法表）、USER_GUIDE（翻译配置/bundle 契约/限制节）同步改写
- **翻译覆盖闭环（可行性研究未覆盖项）**：中/日/韩码族（zh/zho/cmn/yue、ja/jpn、ko/kor）目标语译文 CJK 半角标点归一（`src/engine/onnx_translate.rs` 引擎层后处理，TemperatureCalibration 同层先例；`,;:?!` → 全角，数字两侧夹数字的千分位/时间上下文与句号不转换；金标经程序化差异核对恰好 4 句 zh 变化且仅标点，语义与 en-es 零变化）；`benches/translate_bench.rs` 基准（langdetect 短/10k 上限、en→zh 短/中/长、en→es 短、count_tokens；基线 `docs/benchmarks/translate_baseline_results.md`）；`cuda` feature 编译链真机验证（RTX 5080 + CUDA 13.0 nvcc，candle-kernels 实编译）与 cuda 门控单测实跑；三路评审 30 条意见处置台账入库 `docs/translation-review-ledger.md`（29 修复 1 显性保留，含逐条证据位置）

- **多 checkpoint 配置面（laya-p2p3-coverage P2-2）**:`[model.checkpoints.<name>]` 预设表——`switch_model` 按名命中即应用该 checkpoint 的决策序列预算（`head_max_len`/`max_len`，参数化 `DecisionParams` 替代编译期常量直传）与 model_path/task/engine_type/tokenizer_path（请求显式值优先、预设为缺省层，engine_type 未配置继承当前加载模型）；内置 `laya-multilingual` 预设（head_max_len 256 / max_len 256，文档 §2.1 多语言口径）零配置可切换，同名显式条目按字段级覆盖内置（未写的可选字段逐字段继承内置值，只定制 model_path 不丢多语言预算）；预设表经启动 fail-fast 校验（未知字段/缺失必填/参数为 0 或超硬上限——单字段 8192 或 head+state+4 固定特殊 token 组合超窗/表键与 name 漂移均拒绝启动，拒绝静默钳制）并注入 EmbeddingService 供 switch 查表；未配置时行为与现状完全等价
- **Candle 原生决策头（laya-p2p3-coverage P3）**:`task = "decision" + engine_type = "candle"` 走纯 Rust 前向（`src/engine/candle_decision.rs`，无 onnxruntime 依赖）——ModernBERT-large 骨干（candle-transformers，`encoder.`→`model.` 键重映射，transformers 5.0 无 bias/layer0 融合 attn_norm 权重实测兼容）+ 决策头（type_emb 按题广播 → 2 层 norm_first TransformerEncoderLayer（fused `in_proj_weight`，FFN ReLU）→ marker gather → option scorer → act/escalate 头）；预处理/后处理/温度校准与 onnx 管线同源（协议层拆分 `decision_protocol.rs` 共享，序列构造/collate/softmax/三类后处理单一实现）；官方 `convaiinnovations/laya` checkpoint 键映射以 safetensors header 实测 206 张量为准（缺键显性报错列全清单）；对齐闸门 `tests/candle_decision_parity.rs` 同题对照 `|Δlogit| ≤ 1e-4`（choice/score/noul 各一题，真 checkpoint + 真 bundle 实测通过；资产缺失整文件 SKIP）；`EngineFactory` 决策臂按 engine_type 分派后端（candle/onnx），`AnyEngine` 补 `supports_rerank` 变体转发（此前决策引擎的 false 覆盖经 AnyEngine 不可达）；checkpoint 获取步骤与限制见 USER_GUIDE「Candle 原生决策路径」。**迁移提示（对既有 task=decision 部署为行为变更）**：旧版决策任务不分派 engine_type、一律消费 ONNX bundle；升级后 `[model].engine_type` 缺省（candle）要求 `models/laya-pytorch/` 布局的 PyTorch checkpoint（启动期显性加载失败即命中此变更）——放置 checkpoint，或显式配置 `engine_type = "onnx"` 沿用既有 ONNX bundle。`memory_limit` 记账对 decision + candle 组合按 2× 磁盘放大（F16 checkpoint → F32 物化驻留口径，switch 投影执法与启动期记账同源），部署侧不再需要按 2× 磁盘预留余量
- **决策管线接入层（G3）**:`/api/1/decisions` 四协议端点（HTTP/MCP/CLI/gRPC `vecboost.decide`）——`DecisionService`（state/questions 校验链 + 域层 validate 复用 + supports_task 先门后调 + spawn_blocking 阻塞推理隔离）、`DecisionModule` 注册贯通 server/MCP/CLI 三处 kit 装配；`[model] engine_type` / `task` 配置驱动（消灭启动路径硬编码 Candle，未知值启动期显性报错）；switch 契约扩展 `task` 字段（请求值优先→继承当前→回落 embedding）并传播替换决策引擎；切模型二选一语义（`task=decision` 时 embed/rerank 端点 400，反之 decisions 端点 400）
- **Laya 决策管线（G2）**:`src/engine/decision.rs`——`[CLS] head [SEP] [MASK] option… [SEP] state [SEP]` 预处理（marker_pos/qtype 编码、head+options 192 token 预算前置校验、state 256 token 截断）、5 张量 ONNX 推理（input_ids/attention_mask/marker_pos/marker_mask bool/qtype）、per-cardinality 温度校准（`laya_config.json`，加载即钳制 [0.5,5]，缺桶回退 1.2，缺失 warn 不静默）、choice/score/noul 三型后处理（argmax+完整概率表 / 期望等级 Σi·p_i+分布 / P(true)）
- **factory task 分派臂**:`EngineFactory::create` 对 `task=decision` 一律路由决策管线（`AnyEngine::Decision` 生产构造点，embed/rerank 端点对决策模型以 UnsupportedTask 400 显性拒绝）
- **决策指标**:`vecboost_decision_seconds`(决策链路逐请求总时延直方图，`_count` 即调用计数);决策管线 take_stage_snapshot 恒 None（Stage 三值豁免，观测走独立指标）
- **laya_poc P0 数值对照 example**:`cargo run -p vecboost-examples --bin laya_poc --features onnx`——经 EngineFactory→AnyEngine→trait 全链调用，`LAYA_GOLDEN_JSON` 指向 Python onnxruntime 导出文件时逐值断言概率差 <1e-4，bundle 缺失打印下载指引后 SKIP
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
- **Token 生命周期**:`token_expiration_hours` 缺省 24 小时(原 garrison 默认 30 天)
- **CSRF 默认**:跟随 `auth.enabled`(`AuthConfig` 默认 csrf.enabled=false)
- **`--config` fail-fast**:显式指定且文件不存在 → 报错退出(原静默回退默认配置)
- **缓存与分词行为**:缓存键绑定模型标识(升级后首轮缓存全 miss);HF tokenizers 分词结果与旧自研实现存在差异(更正确),存量向量需重建
- **CLI**:未知子命令报错退出(码 2),不再静默启动 HTTP 服务器;`--help` 可用

#### 升级迁移指南（旧行为 → 新行为 → 迁移动作）

| 变更 | 旧行为 | 新行为 | 迁移动作 |
|------|--------|--------|----------|
| 绑定安全 | `auth.enabled=false` 可绑定 `0.0.0.0` | 非回环绑定 + 无认证 → **拒绝启动** | 本地开发保持 `127.0.0.1`；受信网络容器设 `VECBOOST_ALLOW_INSECURE=1`（打 ERROR 告警）；生产启用 auth |
| 管理员密码 | `VECBOOST_ADMIN_PASSWORD` 可选，缺失时任意凭据登录 | `auth.enabled=true` 且缺失 → **拒绝启动** | 设置 `VECBOOST_ADMIN_PASSWORD`（≥12 位） |
| 登录用户名 | 任意用户名 + admin 口令可登录 | 仅 `default_admin_username`（默认 `admin`）可登录 | 客户端固定使用 admin 用户名 |
| XFF 信任 | `trusted_proxies` 为空时无条件信任 `X-Forwarded-For` | 为空时**忽略 XFF**，使用直连地址 | 反代部署显式配置 `trusted_proxies = ["10.0.0.0/8"]` 等 |
| RBAC | `/model/*`、`/embed/file` 仅需登录 | 要求 **admin 角色** | 业务用户使用普通账号即可调 embed；模型管理用 admin |
| `/embed/file` | 默认允许根 = 进程 cwd，回传文本预览 | 必须显式配置 `grpc_allowed_roots`；单文件 ≤10 MiB；`text_preview` 仅 admin | 配置允许根；客户端不再依赖 preview |
| Token 有效期 | 缺省 30 天（garrison 默认） | 缺省 **24 小时** | 长会话场景显式配置 `token_expiration_hours` |
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

### 新增（translation-int8-optin：翻译权重 int8 量化 opt-in）

- **量化脚本**：`scripts/quantize-translate-bundles.py`（onnxruntime dynamic int8，图 I/O 契约生成即校验）；3 个配置 bundle 实测每对主集 -34%（encoder -75%，decoder 受 vocab fp32 残留限制），磁盘 6.4G→4.2G
- **选路旋钮**：`TRANSLATION_PREFER_INT8=1` 打开（缺省 fp32 优先，契约零变化）；`candidate_order` 纯函数四态单测
- **质量门禁**：`test_int8_quality_gate_m2m100`（strict 逐字节一致，实测两句全等）/`test_int8_quality_gate_opus`（容差档：前缀相似度≥0.85 + 尾部外延≤1.5×，实测两类量化噪声被精确含纳）——opus 默认不切，待 paper-translate-quality 评测

### 新增（wsl2-oom-hardening-r2：第二轮加固）

- **`.wslconfig` 内核阈值**：追加 `vm.min_free_kbytes=262144`（原 44MB → 256MB，OOM 决策前移；下次 `wsl --shutdown` 生效，原文件备份于 temp/）
- **通用进程内存上限**：`scripts/run-memcap.sh`（systemd --user scope，服务异常膨胀自爆保整机；MemoryMax/MemoryHigh/MemorySwapMax 三旋钮，cgroup 属性与超限击杀均已实测）
- **jemalloc decay 编译期烘焙**：`.cargo/config（旧名入库，config.toml 被 gitignore）[env] JEMALLOC_SYS_WITH_MALLOC_CONF` → decay 1000ms（大请求后 RSS 秒级回落）；三条注入路径排除证据链（运行时 mallctl 不存在 / `je_malloc_conf` 被 hidden visibility 否决）见 specmark design D3；`effective_boot_conf()` 启动期观测

### 新增（wsl2-oom-hardening：WSL2 OOM 事故六道防线）

- **Session 内存门禁**：`build_session` 在创建任何 ORT 分配前检查 MemAvailable（需求按模型文件 ×3/2 估算），不足即 fail-fast 显性报错（消息含 available/need/floor 与处置建议），全引擎 6 处调用点统一生效；旋钮 `SESSION_MEMORY_GUARD`（默认开）/`SESSION_MEMORY_FLOOR_MB`（默认 2048）
- **翻译并行 worker 惰性加载**：句级并行附加 Session 集从启动期预载改为首次长输入并行请求按需补建——idle 驻留从 4 pair×4 worker 全 FP32 的 ~30G+ 降为每 pair 一套；P7 时延契约不变（`TRANSLATION_PARALLEL_WORKERS` 语义不变）。根因与实测证据见 [MEMORY_SAFETY](MEMORY_SAFETY.md)
- **运维防线（免 root）**：`scripts/build-memcap.sh`（systemd --user scope 构建内存上限，cgroup `memory.max` 已验证内核级生效）；`scripts/mem-guard.sh` + `mem-guard-install.sh`（用户级 earlyoom，MemAvailable 阈值杀目标进程，正反向实测通过，已随安装脚本常驻）

### 变更（废弃报告退役与维护规范沉淀）

- **废弃报告移出跟踪**：根目录 `DEPRECATED.md` 与 `transcribe-poc/DEPRECATED.md` 经 `git rm` 删除（历史提交保留，不再跟踪）；沿革与裁定的事实源为 `specmark/archive/**`（只读）
- **新增 [🧹 维护规范](MAINTENANCE.md)**：沉淀仍有效的长期约束——转录消费面与禁止语义级双实现、`transcribe-own` CLI 契约（stdout 五行含 `TRANSCRIPT_CHARS`、`--vad-only` 仍写空 segments.txt、退出码 0/1/2、日志一律 stderr）、退役配置项“显性告警禁静默”纪律、废弃文档与勘误治理元规则
- **入链改指**：README 双语的“详见 DEPRECATED.md”改指 `transcribe-poc/CONTRACT.md` 与 [MAINTENANCE](MAINTENANCE.md)；根目录 `AGENTS.md`（按 `.gitignore` 约定不入库）收敛为指向仓库文档的指针，不复制规范正文（避免第二事实源）

### 变更（i18n-audit：全量国际化收口与默认英文加固）

- **默认语言约束可验证化**：`detect_locale()` 抽出纯函数 `detect_from(get_env, sys_locale)`（优先级链与早返回语义零变更），新增 5 个守卫单测：纯净环境必落 `en`、`C`/`POSIX`/不支持值在任一层级均不得产出 `zh`、`VECBOOST_LANG` 压过 POSIX 链与系统 locale、`en` 语言包零中文（含注释）且 `zh` 侧必须含中文
- **行为变更（协议面跟随 locale）**：`i18n::init()` 末尾新增 `sync_sdforge_i18n()`——将 sdforge 的 locale 对齐本仓 `current_locale()`，并把 FTL 的 `forge-*` 键注册进 sdforge 注册表（单一事实源）。此前 `VECBOOST_LANG` 管不到 CLI `--help`/MCP 描述，它们自行读 `SDFORGE_LANG`/系统 locale（双轨）；现统一为 `VECBOOST_LANG` 权威。9 个 `#[forge(cli = true)]` 已补 `i18n_key = "forge-cli-<name>"`
- **硬编码中文清零**：`factory.rs` 的 `task=decision` 未启用 onnx 错误改走 `factory-decision-no-onnx-feature`；`transcribe-poc` 补 `i18n::init()`（此前未初始化，任何 `tr()` 都会回退裸键名）并将其 stderr 诊断全部接 FTL；**stdout 五行契约逐字不动**（`CONTRACT.md` 未改）
- **用户可见文案全量国际化**：启动/装配/引擎/设备/缓存/审计/服务/doctor/CLI 共 300 余处硬编码英文字面量抽取为 FTL 键（en/zh 双侧同批），语言包由 240 键增至 577 键（`messages.ftl` 557 + `errors.ftl` 20）；`Tag: message` 形态的机器可 grep 前缀（`[thread-tune]`、`EmbeddingModule:`、`persist:`、`transcribe-own:`）与 `PASS/WARN/FAIL`、`CheckResult.name`、JSON 字段名、`error_code` 按红线保留英文
- **新增回归门禁**：`scripts/i18n_consistency_check.py`（R1 默认英文、R2 en/zh 键齐性、R3 `tr()`/`i18n_key` 引用有效、R4 生产码禁硬编码中文、R5 未国际化文案棘轮），已接入 `.pre-commit-config.yaml` / `scripts/pre-commit.sh i18n` / CI `Check` job；豁免记 `scripts/i18n_allowlist.json`（9 条规则覆盖 10 处站点，均带理由：开发者 panic 面、测试夹具、基准仪表），热路径封顶记 `scripts/i18n_baseline.json`（63 处，只降不升）
- **规范正文**：[MAINTENANCE](MAINTENANCE.md) 新增「🌐 i18n 约束」节（默认英文、协议面单一事实源、机器输出红线、门禁命令）；[CONTRIBUTING](CONTRIBUTING.md) 质量门禁加 i18n 一行
- **顺手修复（非 i18n，本地工具链 clippy 新增拦截）**：`api/transcribe.rs` 的 `use serde::Deserialize` 按 `cli + transcribe` 门控（否则无 transcribe 的 feature 组合报未用导入）；`engine/session.rs` 去掉 `split_whitespace` 前冗余的 `trim()`

---

## [0.2.1] - 2026-07-30

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

## [0.1.2] - 2026-01-16

- 新增 Matryoshka 维度约简功能
- 新增 OpenAI 兼容的 `/v1/embeddings` 端点
- 清理未使用的部署配置文件

## [0.1.1] - 2026-01-15

- 安全加固 phase 2 完成
- 改进错误处理并启用审计日志
- 测试支持真实推理（mock 回退）
- 文档增强（Mermaid 可视化）与 cargo publish 元数据

## [0.1.0] - 2026-01-10

VecBoost 初始发布。
