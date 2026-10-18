# VecBoost API 响应消息 — 中文

# ── 验证消息 ──
validate-text-length = 索引 { $index } 处的文本超过最大长度 { $max }（实际 { $got }）
validate-batch-size = 批处理大小 { $size } 超过最大限制 { $max }（配置项 embedding.max_batch_size）
validate-path-failed = 路径验证失败：{ $detail }
sensitive-dir-refused = 拒绝使用敏感目录 '{ $path }' 作为允许根目录；请在 [server] grpc_allowed_roots 中显式配置

# ── OpenAI 兼容端点 ──
openai-input-empty = 输入不能为空
openai-input-too-large = 输入数组过大（最多 { $max } 项；服务端 embedding.max_batch_size = { $effective }）
openai-model-not-found = 模型 '{ $model }' 不存在。可用模型：{ $available }

# ── 认证消息 ──
auth-admin-password-missing = 服务端未配置管理员密码，认证不可用；请设置 VECBOOST_ADMIN_PASSWORD 并重启
auth-admin-required = 该操作需要管理员权限
auth-credentials-missing = 缺少认证凭据
auth-password-empty = 密码不能为空
auth-invalid-credentials = 用户名或密码无效
auth-refresh-token-empty = 刷新令牌不能为空
auth-invalid-token = 令牌无效或已过期
logout-success = 登出成功。令牌已撤销。

# ── 模型管理 ──
model-switch-success = 模型切换成功
model-already-current = 已在使用该模型
model-load-failed = 加载模型 '{ $name }' 失败：{ $detail }
switch-path-rejected = 路径被拒绝：必须是允许根目录内已存在的路径（model_path 须为目录、tokenizer_path 须为文件）；详见服务端日志
switch-model-load-failed = 模型加载失败；详见服务端日志

# ── 文件嵌入 ──
file-empty = 文件为空
file-no-paragraphs = 文件中未找到段落
file-invalid-encoding = 无效的路径编码：路径包含非法 UTF-8 字符
embed-file-too-large = 文件超过 /embed/file 的 { $max } MiB 上限（实际 { $got } 字节）

# ── 内存不足 / 降级 ──
oom-no-fallback = 内存不足且已尝试降级回退

# ── 健康检查 ──
health-ok = 正常
health-check-failed = { $module }：健康检查失败：{ $detail }
health-engine-probe-failed = tokenizer/engine 流水线探测失败
health-limiter-failed = limiteron 健康检查失败

# ── Rerank 验证 ──
rerank-empty-docs = 文档列表不能为空
rerank-too-many-docs = 文档数量 { $count } 超过单次查询最大限制 { $max }
rerank-invalid-top-k = 指定 top_k 时必须大于 0
rerank-unsupported = 当前引擎不支持 rerank
rerank-query-too-long = 查询长度 { $length } 超过最大允许长度 { $max }

# ── Decision 验证 ──
decision-empty-state = state 不能为空（决策上下文 state 必填）
decision-empty-questions = questions 不能为空（至少一个问题）
decision-too-many-questions = questions 数量 { $count } 超过单次请求上限 { $max }
decision-unsupported = 当前引擎不支持 decision 决策推理
decision-inference-task-panicked = 决策推理任务失败（引擎任务 panic）；详见服务端日志

# ── Detect 验证 ──
detect-empty-text = 文本不能为空（语言检测 text 必填）
detect-too-long = 文本长度 { $length } 超过最大允许长度 { $max }
detect-invalid-allowlist = allowlist 语言码不支持：{ $code }（仅接受检测引擎已发布的语言码）

# ── Translate 验证与路由 ──
translate-empty-text = 文本不能为空（翻译 text 必填）
translate-too-long = 文本长度 { $length } 超过最大允许长度 { $max }
translate-unsupported = 当前引擎不支持 translate 翻译推理
translate-not-configured = 翻译服务未配置语言对（请在 [translation] 配置段添加 [[translation.pairs]]）
translate-unsupported-pair = 请求的语言对 { $requested } 未配置（已配置：{ $pairs }）
translate-detect-failed = 源语言检测不可靠，无法确定翻译路径（请显式指定 source_language）
translate-too-many-source-tokens = 源文本长度 { $count } 超过该语言对上限 { $max }
translation-inference-task-panicked = 翻译推理任务失败（引擎任务 panic）；详见服务端日志

# ── 目录错误 ──
dir-get-cwd-failed = 获取当前目录失败：{ $detail }

# ── 认证用户名校验 ──
auth-username-length = 用户名长度必须在 3 到 32 个字符之间
auth-username-start = 用户名必须以字母开头
auth-username-charset = 用户名只能包含字母、数字、下划线和连字符

# ── 文本 / 批处理校验 ──
validate-text-empty = 文本不能为空
validate-text-too-short = 文本过短：{ $got } 个字符（最少 { $min } 个）
validate-text-too-long = 文本过长：{ $got } 个字符（最多 { $max } 个）
validate-text-control-chars = 文本包含非法控制字符
validate-text-whitespace = 文本仅包含空白字符
validate-batch-empty = 批处理不能为空
validate-text-index-failed = 索引 { $index } 处文本校验失败：{ $detail }
validate-search-empty = 搜索文本列表不能为空
validate-top-k-exceeded = top_k { $got } 超过最大限制 { $max }

# ── 文件校验 ──
file-too-large = 文件大小 { $size } MB 超过最大允许 { $max } MB
file-access-failed = 无法访问文件 { $path }：{ $detail }
file-no-extension = 文件没有扩展名
file-extension-not-allowed = 不允许的文件扩展名 '{ $ext }'。允许的扩展名：{ $allowed }
file-open-failed = 无法打开文件：{ $detail }
file-read-failed = 无法读取文件：{ $detail }
file-binary-rejected = 文件包含非文本二进制数据

# ── 路径校验 ──
path-traversal-detected = 检测到路径穿越尝试：{ $detail }
path-invalid = 无效路径：{ $detail }
path-no-roots = 未配置允许的文件访问根目录
path-access-denied = 拒绝访问：路径 '{ $path }' 不在允许目录内。{ $detail }
path-not-file = 路径不是文件：{ $path }
path-not-dir = 路径不是目录：{ $path }

# ── 流水线 / 队列 ──
pipeline-channel-error = 响应通道错误
pipeline-timeout = 请求超时
queue-type-mismatch = 期望 Embed 请求但收到 Rerank 请求
queue-full-rejected = 队列已满，请求被拒绝
rerank-not-configured = Rerank 服务未配置
decision-not-configured = 决策服务未配置
pipeline-response-type-mismatch = Pipeline 返回了类型不匹配的响应
request-queue-sla-exceeded = 请求在队列中等待 { $waited } 秒，超过其 { $sla } 秒 SLA
inference-task-panicked = 推理任务已 panic
auth-disabled = 认证在运行时被禁用
auth-verify-failed = 密码验证失败
api-init-state-called = init_state 已被调用
api-init-state-missing = init_state 尚未调用

# ── 内存不足 / 引擎（P1） ──
oom-no-fallback-available = 内存不足且无可用降级方案
oom-fallback-failed = 内存不足错误且降级回退失败：{ $detail }
oom-max-attempts = 超出最大降级重试次数（{ $attempts }）。最后错误：{ $detail }
engine-semaphore-failed = 获取信号量失败：{ $detail }
engine-batch-timeout = 批处理分块处理超时（{ $secs }s）

# ── 审计日志 flush ──
audit-flush-writer-unavailable = 审计日志 writer 不可用，排队事件未能落盘
audit-flush-timeout = 等待审计日志 flush 确认超时

# ── 中间件错误响应 ──
rate-limit-exceeded = 请求频率超出限制
route-not-found = 请求的资源不存在

# ── Metrics 端点（P1） ──
metrics-limiter-unavailable = 速率限制器不可用
metrics-rate-limited = 超出速率限制
metrics-collector-missing = PrometheusCollector 未配置
metrics-encode-failed = 指标编码失败：{ $detail }

# ── 启动 / 配置（P1） ──
startup-db-pool = 创建数据库连接池失败：{ $detail }
startup-db-schema = 初始化数据库模式失败：{ $detail }
startup-jwt-length = JWT 密钥长度必须至少 32 个字符。当前长度：{ $got }
startup-jwt-missing = 启用认证时必须提供 JWT 密钥，请设置强 JWT 密钥
startup-admin-password-missing = 启用认证时必须配置管理员密码，请设置环境变量 VECBOOST_ADMIN_PASSWORD（否则任意凭据均可登录）
startup-auth-session-backend-without-auth = 会话后端 session_backend="database" 要求启用认证：请设置 [auth] enabled = true，或将 session_backend 改为 "memory"
startup-auth-session-backend-db-feature-missing = 会话后端 session_backend="database" 需要以 db 特性构建（如 --features "db,sqlite" 或 "db,postgres"），当前构建未启用 db
startup-rate-limit-redis-feature-missing = 限流后端 backend="redis" 需要以 distributed-redis 特性构建，当前构建未启用
startup-rate-limit-redis-url-missing = 限流后端 backend="redis" 必须配置 [rate_limit] redis_url
startup-cache-redis-feature-missing = 缓存后端 cache_backend="redis" 需要以 distributed-redis 特性构建，当前构建未启用
startup-cache-redis-url-missing = 缓存后端 cache_backend="redis" 必须配置 [embedding] redis_url
startup-cache-redis-persist-conflict = 缓存后端 cache_backend="redis" 与 persist_path（WAL）互斥：Redis 已提供持久化，WAL 仅适用于内存后端
startup-auth-session-backend-db-pool-unavailable = 会话后端 session_backend="database" 需要可用的数据库连接池：请检查 [database] 配置与连接池初始化
startup-auth-session-backend-auth-feature-missing = 会话后端 session_backend="database" 需要以 auth 特性构建（当前构建未启用 auth，无会话消费者），请改为 "memory" 或以 --features auth 构建
startup-logger = 初始化日志系统失败：{ $detail }
startup-config = 加载配置失败：{ $detail }
startup-encryption = 加密密钥验证失败：{ $detail }
startup-register-failed = 注册 { $module } 失败：{ $detail }
startup-grpc-bearer-failed = gRPC require_auth=true 但 BearerAuth 创建失败：{ $detail }。请设置 VECBOOST_JWT_SECRET（≥32 字符）或设置 [server] grpc_require_auth = false
startup-grpc-no-secret = gRPC require_auth=true 但 auth.jwt_secret 为 None。请设置 VECBOOST_JWT_SECRET 环境变量或设置 [server] grpc_require_auth = false
startup-grpc-auth-disabled = gRPC require_auth=true 但 auth.enabled=false。请启用 [auth] enabled = true 或设置 [server] grpc_require_auth = false
startup-grpc-no-feature = gRPC require_auth=true 但 vecboost 未启用 `auth` 功能。请启用 `auth` feature 或在配置中设置 [server] grpc_require_auth = false
config-jwt-empty = VECBOOST_JWT_SECRET 不能为空
config-jwt-length = VECBOOST_JWT_SECRET 长度必须至少 { $min } 个字符
config-password-empty = VECBOOST_ADMIN_PASSWORD 不能为空
config-password-length = VECBOOST_ADMIN_PASSWORD 长度必须至少 { $min } 个字符
config-encryption-missing = { $key } 环境变量未设置。生产部署必须配置此项
config-encryption-length = { $key } 必须恰好为 32 字节
cli-no-handler = 未注册 CLI 命令处理器：{ $name }
cli-failed = CLI 命令 '{ $name }' 执行失败：{ $detail }
engine-bert-config-required = Bert 模型需要 Bert 配置
engine-xlm-config-required = XLM-RoBERTa 模型需要 XLM-RoBERTa 配置

# ── CLI 调用错误（main.rs fail-fast 路径） ──
cli-unknown-subcommand = 错误：未知子命令 '{ $name }'
cli-available-subcommands = 可用子命令：{ $list }
cli-usage-hint = 运行 'vecboost --help' 查看用法。
cli-config-requires-path = 错误：--config 需要路径参数（--config <path>）
cli-config-not-found = 错误：配置文件未找到：{ $path }
cli-config-not-file-hint = --config 路径必须指向已存在的 TOML 文件。

# ── 运行时诊断 / 日志 ──
numa-advice = 检测到 NUMA：{ $sockets } 个 socket；建议使用 `numactl --interleave=all` 或 `--cpunodebind` 启动以均衡内存带宽（不做进程内绑定）
heat-read-failed = 模型热度表：无法读取 { $path }（{ $detail }），warmstart 跳过
heat-parse-failed = 模型热度表：{ $path } 解析失败（{ $detail }），弃用整个文件
heat-schema-mismatch = 模型热度表：schema 版本不匹配（文件 { $got }，期望 { $expected }），弃用
heat-entry-fingerprint-mismatch = 模型热度表：条目 '{ $name }' 指纹不匹配，弃用该条
doctor-tokenizer-vocab-missing = ；⚠️ 未按内容识别到 tokenizer 词表文件
transcribe-not-configured = 翻译服务未配置：请在 [transcribe] 配置段设置 enabled = true 与 models_dir

# ── Doctor 诊断（vecboost doctor 标准输出） ──
doctor-config-pass = 配置校验通过（model.task={ $task }）
doctor-config-fail = 配置校验失败：{ $detail }
doctor-model-dir-missing = 模型目录不存在：{ $path }（运行时将尝试 HF 下载）
doctor-tokenizer-json-missing = tokenizer.json 不存在：{ $path }
doctor-tokenizer-json-ok = tokenizer.json 加载成功：{ $path }
doctor-tokenizer-json-fail = tokenizer.json 加载失败：{ $detail }
doctor-cache-memory-only = 纯内存模式（未配置 persist_path）
doctor-cache-wal-writable = WAL 文件可写：{ $path }
doctor-cache-wal-unwritable = WAL 文件不可写：{ $path }：{ $detail }
doctor-cache-path-invalid = persist_path 无效：{ $path }
doctor-cache-parent-missing = 父目录不存在（启动写盘将失败，请先创建或修正配置）：{ $path }
doctor-cache-dir-writable = 目录可写（WAL 尚未创建，首次插入时建立）：{ $path }
doctor-cache-dir-unwritable = 目录不可写：{ $path }：{ $detail }
doctor-threads-detailed = 物理核 { $physical } / 逻辑核 { $logical }（调优按物理核生效）
doctor-threads-fallback = 物理核检测不可用，回退逻辑核 { $logical }（VECBOOST_NO_THREAD_TUNE 可显式控制）
doctor-threads-numa = ；检测到 { $sockets } 个 socket，建议 numactl --interleave=all 启动
doctor-gpu-cpu-mode = CPU 模式（use_gpu=false）
doctor-gpu-no-cuda = use_gpu=true 但未探测到 CUDA 设备
doctor-gpu-devices = CUDA 设备：{ $names }
doctor-gpu-no-cuda-feature = use_gpu=true 但二进制未编译 cuda feature（构建时启用 --features cuda）
doctor-models-none = 未找到本地模型目录（models/ 不存在且未配置 model_path），运行时将依赖 HF 下载
doctor-models-fail-count = { $count } 个模型目录存在完整性问题
doctor-dir-unreadable = 目录不可读
doctor-json-parse-fail = JSON 文件解析失败 { $path }：{ $detail }
doctor-json-unreadable = JSON 文件不可读 { $path }：{ $detail }
doctor-dims-no-hidden = 全部二维张量维度均不含 config.hidden_size({ $hidden })，权重与配置疑似不配对
doctor-no-weights = 未发现权重文件（.safetensors/.gguf）
doctor-weights-summary = 权重 { $weights } 个、gguf { $gguf } 个、tokenizer { $tokenizer }、config.hidden_size { $hidden }
doctor-tokenizer-recognized = 已识别
doctor-tokenizer-unrecognized = 未识别
doctor-weight-unreadable = 权重文件不可读 { $path }：{ $detail }
doctor-metadata-unreadable = 元数据不可读 { $path }：{ $detail }
doctor-file-too-small = 文件过小，不足 8 字节头：{ $path }
doctor-header-len-read-fail = 头长度读取失败 { $path }：{ $detail }
doctor-header-len-invalid = safetensors 头长度非法（{ $len }）：{ $path }
doctor-header-read-fail = 头读取失败 { $path }：{ $detail }
doctor-header-json-fail = safetensors 头 JSON 解析失败 { $path }：{ $detail }
doctor-tensor-not-object = 张量 { $tensor } 元数据不是对象：{ $path }
doctor-tensor-no-dtype = 张量 { $tensor } 缺 dtype：{ $path }
doctor-tensor-no-offsets = 张量 { $tensor } 缺 data_offsets：{ $path }
doctor-tensor-offsets-invalid = 张量 { $tensor } data_offsets 非法：{ $path }
doctor-tensor-offsets-reversed = 张量 { $tensor } data_offsets 逆序：{ $path }
doctor-file-truncated = 文件被截断（声明数据需要 { $need } 字节，实际 { $have }）：{ $path }
doctor-gguf-unreadable = gguf 文件不可读 { $path }：{ $detail }
doctor-gguf-too-small = gguf 文件过小 { $path }：{ $detail }
doctor-gguf-bad-magic = gguf 魔数非法（{ $got }，应为 GGUF）：{ $path }

# ── 启动 fail-fast / 线程调优横幅（main.rs 标准错误） ──
startup-engine-type-unknown = [model] engine_type 未知值 "{ $got }"{ $hint }，支持: candle, onnx
startup-engine-onnx-rebuild-hint = （engine_type="onnx" 需以 --features onnx 重新构建）
# onnx-primary：candle 降级 opt-in 回退后端后，engine_type="candle" 配置在未启用 feature `candle` 的构建中显性拒绝
startup-engine-candle-not-compiled = [model] engine_type="candle" 但本次构建未编译 candle 后端（onnx-primary 默认主引擎为 onnx）；请以 `--features candle` 重新构建，或移除 engine_type 配置以使用默认 onnx 引擎
engine-candle-cuda-unavailable = engine_type="candle" 配置了 device="cuda"，但本次构建/本机 CUDA 不可用：请以 `--features candle-cuda` 重新构建（CUDA 后端），并确认存在可用的 NVIDIA GPU；拒绝静默落到 CPU
startup-semantic-mode-invalid = [semantic_cache] comparison_mode 无效：{ $detail }
startup-translation-no-onnx = [translation] pairs 已配置但本次构建未启用 `onnx` feature；请以 --features onnx 重新构建
startup-translate-pair-invalid = [translation] pairs[{ $index }] ({ $src }-{ $tgt }) 参数非法：{ $detail }
startup-translate-pair-load-failed = [translation] pairs[{ $index }] ({ $src }-{ $tgt }) 加载失败：{ $detail }
startup-transcribe-session-open = 转录会话打开失败：{ $detail }
startup-thread-tune = [thread-tune] 物理核={ $physical } 逻辑核={ $logical } 生效线程数={ $effective }（显式配置={ $explicit }）
startup-thread-unknown = 未知(回退)
startup-thread-unset = 未设置
startup-rayon-skipped = [thread-tune] rayon 全局池已初始化，跳过：{ $detail }
semantic-cache-mode-unknown = 未知的 comparison_mode：'{ $got }'（可选 exact|i8|binary）

# ── 引擎路由 / checkpoint 配置错误（用户可见错误返回） ──
factory-translate-onnx-only = 翻译仅支持 onnx 引擎：请将 engine_type 配置为 onnx（task=translate 无 candle 实现；更改后需重启服务重新加载）
factory-transcribe-onnx-only = 转录仅支持 onnx 引擎：请将 engine_type 配置为 onnx（task=transcribe 无 candle 实现）
factory-translate-no-onnx-feature = 检测到 task=translate 模型配置，但本次构建未启用 `onnx` feature；请以 `--features onnx` 重新构建
config-checkpoint-engine-unknown = checkpoint { $name } engine_type 未知值 "{ $got }"{ $hint }，支持: candle, onnx
cache-persist-uncreatable = 无法创建 persist 文件 { $path }：{ $detail }（请检查 [embedding] persist_path 目录权限）
factory-transcribe-no-transcribe-feature = 检测到 task=transcribe 模型配置，但本次构建未启用 `transcribe` feature；请以 `--features onnx,transcribe` 重新构建
factory-quantized-no-feature = 检测到 GGUF 量化模型配置（.gguf + quantized=true），但本次构建未启用 `quantized-gguf` feature；请以 `--features quantized-gguf` 重新构建
factory-decision-no-onnx-feature = 检测到 task=decision 模型配置，但本次构建未启用 `onnx` feature；请以 `--features onnx` 重新构建

# ── GGUF 量化引擎错误（feature quantized-gguf；error-model-load/inference 详情） ──
quant-gguf-open-failed = 无法打开 GGUF 文件 { $path }：{ $detail }
quant-gguf-too-short = GGUF 文件过短 { $path }：{ $detail }
quant-gguf-magic-mismatch = GGUF 魔数校验失败 { $path }：期望 0x{ $expected }，实际 0x{ $got }
quant-safetensors-load-failed = safetensors 加载失败 { $path }：{ $detail }
quant-config-json-unreadable = config.json 不可读：{ $detail }
quant-config-json-parse-failed = config.json 解析失败：{ $detail }
quant-config-json-missing-key = config.json 缺少字段 { $key }
quant-tensor-no-mapping = safetensors 张量 { $name } 无 GGUF 映射（架构不支持或命名漂移）
quant-tensor-dtype-failed = 张量 { $name } 目标 dtype 与 F16 兜底均失败：{ $detail }
quant-tensor-quantize-failed = 张量 { $name } 量化失败：{ $detail }
quant-gguf-output-unwritable = GGUF 输出不可写 { $path }：{ $detail }
quant-gguf-write-failed = GGUF 写入失败：{ $detail }
quant-gguf-flush-failed = GGUF 输出 flush 失败：{ $detail }
quant-gguf-openfile-failed = GGUF 打开失败 { $path }：{ $detail }
quant-gguf-header-parse-failed = GGUF 头解析失败：{ $detail }
quant-gguf-arch-unsupported = GGUF 架构 { $arch } 不受支持（仅 bert；文件：{ $path }）
quant-gguf-meta-missing-key = GGUF 元数据缺少键 { $key }
quant-tensor-read-failed = GGUF 张量 { $name } 读取失败：{ $detail }
quant-tensor-dequant-failed = GGUF 张量 { $name } 反量化失败：{ $detail }
quant-tensor-no-candle-mapping = GGUF 张量 { $name } 无 candle 映射
quant-tensor-mapping-conflict = GGUF 张量 { $name } 映射冲突（{ $candle } 重复）
quant-gguf-missing-token-embd = GGUF 缺少 token_embd.weight
quant-token-embd-dim-failed = token_embd 维度读取失败：{ $detail }
quant-bert-assembly-failed = GGUF BERT 装配失败：{ $detail }
quant-tokenizer-load-failed = GGUF 同目录 tokenizer.json 加载失败（{ $path }）：{ $detail }
quant-empty-batch = 空 batch

# ── 运行日志：模块注册 / 指标端点 / doctor 横幅 ──
# 约定：`Tag: message` 形态的标识符前缀留在代码里（机器可 grep），只翻译 message。
log-registry-engine-ready = 模型引擎就绪
log-registry-service-ready = 服务就绪
log-registry-ratelimit-active = on_ready — limiteron 限流器已激活
log-registry-ratelimit-shutdown = on_shutdown — 限流器已关闭
log-registry-audit-flushing = 关机前刷写审计日志
log-registry-audit-flush-failed = 刷写审计日志失败：{ $detail }
log-registry-watcher-ready = 配置文件监视器已就绪
log-registry-watcher-shutdown = 正在关闭配置监视器
log-registry-watcher-shutdown-failed = 关闭监视器时出错：{ $detail }
log-metrics-collector-unregistered = PrometheusCollectorModule 未注册：{ $detail }
log-metrics-response-build-failed = 构建指标响应失败：{ $detail }
log-metrics-ip-allowlist-unregistered = IpWhitelistModule 未注册：{ $detail }
log-metrics-ratelimit-unavailable = RateLimitEnabled=true 但 RateLimitModule 不可用：{ $detail }，拒绝该请求
log-metrics-json-response-build-failed = 构建 JSON 错误响应失败：{ $detail }

# ── Doctor 报告外壳（标准输出；PASS/WARN/FAIL 标记保持英文 token）──
doctor-report-title = vecboost doctor — 只读诊断
doctor-summary = 共 { $total } 项检查：{ $pass } 通过、{ $warn } 告警、{ $fail } 失败

# ── 运行日志：服务启动 / 装配横幅（main.rs 标准输出+标准错误）──
log-main-building-module = 正在构建模块：{ $module }
log-main-module-built-slow = 模块 { $module } 构建耗时 { $seconds } 秒（超过 1 秒）
log-main-module-built = 模块 { $module } 构建耗时 { $ms } 毫秒
log-main-db-pool-init = 正在初始化数据库连接池 url={ $url }
log-main-db-retry = 数据库重试策略已启用（max_retries={ $retries }，指数退避）
log-main-db-warmup = dbnexus 连接池预热已启用：min_connections={ $min }，warmup_timeout={ $timeout } 秒，warmup_retries={ $retries }
log-main-db-ready = 数据库连接池已初始化并完成模式校验
log-main-db-metrics = dbnexus MetricsCollector 已创建
log-main-tokenizer-noop = [model].tokenizer_path 已配置但没有消费者匹配 engine_type={ $engine } task={ $task }：消费者仅为 onnx 本地 bundle 分支（engine_type=onnx + 本地目录）与 decision 管线（task=decision + 本地目录）；该值不生效
log-main-kvcache-enabled = KV Cache 已启用，容量：{ $size }
log-main-kvcache-wal = 嵌入缓存 WAL 持久化已启用：{ $path }（压缩阈值 { $bytes } 字节）
log-main-kvcache-disabled = KV Cache 已禁用
log-main-residency = 模型驻留：max_resident_models={ $max } budget_mb={ $budget }
log-main-semantic-mode-deprecated = [semantic_cache] comparison_mode={ $mode } 已废弃，本版本不产生任何效果（恒为 exact）
log-main-semantic-enabled = 语义缓存已启用：threshold={ $threshold } fuzzy_threshold={ $fuzzy } capacity={ $capacity }
log-main-translation-threads = translation：每会话 intra 线程预算 = { $budget }（{ $pairs } 个语言对）
log-main-engine-init = 正在初始化推理引擎（首次运行需下载模型，可能耗时较久）……
log-main-translation-ready = 翻译服务已初始化：配置 { $pairs } 个语言对
log-main-transcribe-ready = 转录服务已初始化：bundle={ $bundle }
log-main-mcp-start = 正在通过 stdio 启动 VecBoost MCP 服务
log-main-auth-enabled = Garrison 认证已启用（JWT + 会话 + 密码校验）
log-main-auth-session-backend = 认证会话后端：backend={ $backend }，table={ $table }
log-main-rate-limit-backend = 限流后端：backend={ $backend }，endpoint={ $endpoint }
log-main-cache-backend = 缓存后端：backend={ $backend }，endpoint={ $endpoint }
log-main-auth-disabled = 认证已禁用
log-main-csrf-enabled = CSRF 防护已启用（garrison）
log-main-csrf-disabled = CSRF 防护已禁用
log-main-plan-hardware = auto_plan：硬件方案 worker_threads={ $threads } max_batch_size={ $batch } batch_wait_ms={ $wait } quantized_recommended={ $quantized } bottleneck={ $bottleneck } rationale={ $rationale }
log-main-plan-applied = auto_plan：实际生效 max_batch_size={ $batch } batch_wait_ms={ $wait }（计划值 max_batch_size={ $planned_batch } batch_wait_ms={ $planned_wait }，建议 worker_threads={ $threads }）
log-main-pipeline-enabled = 请求流水线已启用，queue_size={ $size }
log-main-worker-start-failed = WorkerManager 启动失败：{ $detail }
log-main-pipeline-ready = 流水线组件初始化完成
log-main-pipeline-disabled = 请求流水线已禁用
log-main-gpu-no-backend = use_gpu=true 但本次构建未编译任何 GPU 后端（缺 'cuda' feature，或 macOS 上缺 'metal'）—— 回退到 CPU。如需 GPU 请以 --features cuda 重新构建。
log-main-insecure-warning = 安全风险：认证已关闭却绑定到非回环地址“{ $host }”（VECBOOST_ALLOW_INSECURE=1）。该网段上任何可达者都能通过 /embed/file 读取文件并管理模型。请启用认证或改绑 127.0.0.1。
log-main-insecure-refusal = 拒绝启动：认证已关闭（auth.enabled=false）却绑定到非回环地址“{ $host }”。请任选其一：(1) 将配置中的 host 设为 127.0.0.1；(2) 启用 [auth] 并设置 VECBOOST_JWT_SECRET 与 VECBOOST_ADMIN_PASSWORD；(3) 设置 VECBOOST_ALLOW_INSECURE=1 显式接受该风险。
log-main-jemalloc-decay = jemalloc decay：dirty={ $dirty }ms muzzy={ $muzzy }ms（boot conf）
log-main-jemalloc-probe-failed = jemalloc decay 探测失败：{ $detail }
cli-error-prefix = 错误：{ $detail }

# ── 运行日志：引擎 / 设备 / 缓存 / 服务 / CLI 诊断 ──
log-audit-failed-rotate-current-log = 轮转当前日志文件失败：{ $a1 }
log-audit-failed-rotate-log-file = 轮转日志文件失败：{ $a1 }
log-audit-failed-remove-old-log = 删除旧日志文件失败 { $a1 }：{ $a2 }
log-audit-failed-write-audit-log = 写入审计日志批次失败：{ $a1 }
log-audit-failed-reopen-log-file = 轮转后重新打开日志文件失败：{ $a1 }
log-audit-audit-event-lost-dead = 审计事件已丢失：死信写入失败：{ $e }（数据库错误：{ $db_error }）
log-audit-audit-event-dead-lettered = 审计事件已转入死信文件 { $a1 }
log-audit-failed-open-audit-log = 打开审计日志文件失败：{ $a1 }
log-audit-failed-create-log-directory = 创建日志目录失败：{ $a1 }
log-cache-persist-one-record-key = persist：{ $a1 } 有一条记录 key 非 UTF-8，已跳过
log-cache-persist-one-record-fingerprint = persist：{ $a1 } 有一条记录指纹不匹配，已跳过
log-cache-persist-one-record-failed = persist：{ $a1 } 有一条记录校验和失败，已跳过
log-cache-persist-version-mismatch-discarding = persist：{ $a1 } 版本 { $a2 } 不匹配，整体弃用
log-cache-persist-magic-mismatch-discarding = persist：{ $a1 } 魔数不匹配，整体弃用
log-cache-persist-unknown-record-type = persist：{ $a1 } 未知记录类型 { $a2 }，停止回放（尾部撕裂）
log-cache-persist-cannot-open-treating = persist：无法打开 { $a1 }（{ $a2 }），视为空缓存
log-cache-persist-truncate-clear-failed = persist：清空时截断失败（{ $a1 }）
log-cache-persist-compaction-failed-keeping = persist：压实失败（{ $a1 }），保留原文件
log-cache-persist-compaction-complete-records = persist：压实完成，共 { $a1 } 条记录
log-cache-persist-append-failed-this = persist：追加写入失败（{ $a1 }），本次插入仅留在内存
log-cache-persist-file-lock-poisoned = persist：文件锁已被污染，跳过本次刷写：{ $a1 }
log-cache-rejecting-non-finite-vector = 拒绝对键前缀 { $a1 } 写入非有限值向量
log-cache-persist-failed-write-header = persist：{ $a1 } 写文件头失败：{ $a2 }
log-engine-auto-pooling-unrecognized-model = 自动池化：无法识别模型族“{ $a1 }”，回退到 CLS。请显式设置 pooling_mode
log-engine-loaded-safetensors-model-weights = safetensors 模型权重加载完成
log-engine-loading-safetensors-model-weights = 正在从以下路径加载 safetensors 模型权重：{ $a1 }
log-engine-optimum-cli-export-onnx = optimum-cli export onnx --model { $a1 } --task feature-extraction ./model_safetensors
log-engine-pip-install-optimum = pip install optimum
log-engine-please-convert-model-safetensors = 请先将模型转换为 safetensors 格式：
log-engine-this-often-due-large = 这通常由模型文件过大或格式不兼容导致。
log-engine-failed-load-pytorch-weights = PyTorch 权重加载失败：{ $a1 }
log-engine-pytorch-weights-loaded-successfully = PyTorch 权重加载成功
log-engine-then-use-model-converted = 然后把 config/config.toml 里的 model_path 改为“./model_converted”
log-engine-python-c-from-transformers = python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{ $a1 }').save_pretrained('./model_safetensors')\"
log-engine-consider-converting-safetensors-format = 建议转换为 safetensors 格式以获得更好的加载性能：
log-engine-pytorch-model-file-large = PyTorch 模型文件较大（{ $a1 } GB），大文件可能加载异常。
log-engine-loading-pytorch-model-weights = 正在从以下路径加载 PyTorch 模型权重：{ $a1 }
log-engine-model-file-sha256-verification = 模型文件 SHA256 校验通过
log-engine-verifying-model-file-sha256 = 正在校验模型文件 SHA256 摘要……
log-engine-model-file-integrity-check = 模型文件完整性检查通过
log-engine-post-recovery-integrity-check = 修复后完整性检查通过
log-engine-re-running-integrity-check = 修复完成后重新执行完整性检查……
log-engine-successfully-recovered-all-corrupted = 所有损坏文件已修复完成
log-engine-attempting-automatic-recovery-corrupted = 正在尝试自动修复损坏文件……
log-engine-file-error = 文件：{ $a1 }，错误：{ $a2 }
log-engine-model-file-integrity-check-2 = 模型文件完整性检查失败！
log-engine-checking-model-file-integrity = 正在检查模型文件完整性……
log-engine-model-safetensors-unavailable-falling = model.safetensors 不可用，回退到 pytorch_model.bin；pickle 格式存在代码执行风险
log-engine-downloading-loading-model-from = 正在从 HuggingFace Hub 下载/加载模型：{ $a1 }
log-engine-loading-model-from-remote = 正在从远端 HuggingFace 仓库“{ $a1 }”加载模型且未设置 model_sha256 校验——建议补充校验值
log-engine-loading-model-from-local = 正在从本地路径加载模型：{ $a1 }
log-engine-using-fp32-precision = 使用 FP32 精度
log-engine-bf16-supported-non-cuda = 非 CUDA 设备不支持 BF16，回退到 FP32
log-engine-using-bf16-precision-brain = 使用 BF16 精度（Brain Float 16）
log-engine-fp16-supported-non-cuda = 非 CUDA 设备不支持 FP16，回退到 FP32
log-engine-using-fp16-precision = 使用 FP16 精度
log-engine-int8-quantization-yet-implemented = INT8 量化尚未实现，按 FP32 模式运行。use_quantization 开关当前无效果
log-engine-using-cpu = 使用 CPU
log-engine-candle-engine-does-natively = Candle 引擎不原生支持 AMD GPU，AMD 支持需启用 ROCm 的 Candle 构建
log-engine-using-metal-gpu = 使用 Metal GPU
log-engine-using-cuda-gpu = 使用 CUDA GPU
log-engine-successfully-fell-back-cpu = 已成功回退到 CPU
log-engine-fallback-cpu-supported-cpu = 回退到 CPU：{ $a1 } 在 CPU 上不受支持，改用 FP32
log-engine-fallback-detected-model-architecture = 回退：检测到模型结构：{ $a1 }
log-engine-attempting-fallback-from-gpu = 内存压力触发，正在尝试从 GPU 回退到 CPU
log-engine-token-id-exceeds-vocab = token ID { $a1 } 超出 vocab_size { $a2 }，已截断钳制
log-engine-memory-limit-critical-checking = 内存余量告急，正在检查内存压力以决定是否回退
log-engine-memory-limit-exceeded-attempting = 超出内存限制，正在尝试回退到 CPU
log-engine-memory-pressure-cleared-resetting = 内存压力已解除，重置回退标记——下次加载模型可重试 GPU
log-engine-pooling-mode-model = 池化方式：{ $a1 }（模型：{ $a2 }）
log-engine-detected-model-architecture = 检测到模型结构：{ $a1 }
log-engine-successfully-fell-back-cpu-2 = ONNX 引擎已成功回退到 CPU
log-engine-attempting-fallback-from-gpu-2 = ONNX 引擎正在尝试从 GPU 回退到 CPU
log-engine-memory-limit-critical-onnx = ONNX 引擎内存余量告急，正在检查内存压力以决定是否回退
log-engine-memory-limit-exceeded-onnx = ONNX 引擎超出内存限制，正在尝试回退到 CPU
log-engine-failed-create-tokio-runtime = 为内存检查创建 Tokio 运行时失败：{ $a1 }
log-engine-onnx-engine-initialized-hidden = ONNX 引擎初始化完成：hidden_size={ $a1 }，max_input_length={ $a2 }，vocab_size={ $a3 }
log-engine-using-precision = 使用 { $a1 } 精度
log-engine-fp16-supported-without-gpu = 无 GPU 加速时不支持 FP16，回退到 FP32
log-engine-using-fp16-precision-gpu = 使用 FP16 精度并启用 GPU 加速
log-engine-final-hidden-size-value = 最终 hidden_size 取值：{ $a1 }
log-engine-using-hidden-size-from = 采用配置文件中的 hidden_size：{ $a1 }
log-engine-loading-tokenizer = 正在加载 tokenizer……
log-engine-amd-gpu-detected-but = 检测到 AMD GPU，但本次构建未配置 ROCm 执行提供器。改用 CPU 执行提供器
log-engine-downloading-loading-onnx-model = 正在下载/加载 ONNX 模型文件……
log-engine-using-huggingface-hub-model = 使用 HuggingFace Hub 加载模型：{ $a1 }
log-engine-using-local-model-path = 使用本地模型路径：{ $a1 }
log-engine-successfully-fell-back-cpu-3 = 决策管线已成功回退到 CPU
log-engine-attempting-fallback-cpu-decision = 决策管线正在尝试回退到 CPU
log-engine-decision-pipeline-initialized-bundle = 决策管线初始化完成：bundle={ $a1 }，model={ $a2 }，precision={ $a3 }，calibration={ $a4 }
log-engine-verifying-decision-model-file = 正在校验决策模型文件 SHA256 摘要……
log-engine-msg-model = { $msg }（model={ $a1 }）
log-engine-memory-gate-passed-model = 内存门禁通过：model={ $a1 } need={ $need } available={ $available } floor={ $floor }
log-engine-session-memory-floor-mb = SESSION_MEMORY_FLOOR_MB={ $v } 不是合法整数，回退到默认值 { $DEFAULT }
log-engine-cuda-execution-provider-requires = CUDA 执行提供器需要 transcribe feature（含 ort/cuda），本次构建未启用，改用 CPU 执行提供器
log-engine-attempting-configure-cuda-execution = 正在尝试配置 CUDA 执行提供器
log-engine-transcribe-graphs = 转录三图：conv={ $conv }，encoder={ $encoder }，decoder={ $decoder }，precision={ $precision }（精度按图内容判定，不看文件名）
log-engine-gguf-pooling-inferred-as = 由标识“{ $pool_identity }”推断 GGUF 池化方式为 { $pooling }
log-engine-gguf-config-json-provides = GGUF 的 config.json 无可识别 hidden_act，回退到 Gelu（可能与原模型不一致）
log-engine-gguf-config-json-unknown = GGUF 的 config.json 中 hidden_act“{ $other }”无法识别，回退到 Gelu
log-engine-gguf-hidden-act-act = GGUF hidden_act“{ $act }”在 candle BERT 中无对应实现，回退到 Gelu
log-engine-gguf-tensor-other-this = GGUF 张量“{ $other }”：该仓库把它映射到 final-LN（output.LayerNorm），语义与常规不同，请自行核对
log-device-amd-gpu-initialization-complete = AMD GPU 初始化完成。发现 { $a1 } 个设备（ROCm：{ $a2 }，OpenCL：{ $a3 }）
log-cuda-free-failed = CUDA 显存释放失败：设备 { $device }（{ $bytes } 字节）：{ $e } —— 记账保留
log-cuda-free-on-drop-failed = Drop 释放 CUDA 显存失败：设备 { $device }（{ $bytes } 字节）：{ $e } —— 记账保留
log-device-found-opencl-compatible-amd = 发现 OpenCL 兼容的 AMD GPU：{ $a1 }，显存 { $a2 } 字节
log-device-found-rocm-compatible-amd = 发现 ROCm 兼容的 AMD GPU：{ $a1 }，显存 { $a2 } 字节
log-device-initializing-amd-gpu-device = 正在初始化 AMD GPU 设备管理器……
log-device-rocm-version = ROCm 版本：{ $a1 }
log-device-rocm-vram-query-failed = ROCm 显存查询失败，使用默认回退值：{ $a1 }GB
log-device-rocm-vram-rocm-smi = ROCm 显存（rocm-smi，设备 { $a1 }）：{ $a2 } MB
log-device-amd-driver-version-rocm = AMD 驱动版本（rocm-smi）：{ $a1 }
log-device-amd-driver-version-sysfs = AMD 驱动版本（sysfs）：{ $a1 }
log-device-gpu-memory-allocation-failed = GPU 显存分配失败：请求 { $a1 } 字节，可用 { $a2 } 字节
log-device-amd-gpu-opencl-using = AMD GPU（OpenCL）使用默认回退参数（{ $a1 }GB，compute { $a2}.{ $a3 }）
log-device-primary-cuda-device-set = 主 CUDA 设备已设为：{ $a1 }
log-device-detected-compatible-cuda-device = 检测到兼容的 CUDA 设备：{ $a1 }
log-device-failed-initialize-cuda-device = 通过 candle-core 初始化 CUDA 设备失败：{ $a1 }
log-device-detected-cuda-device-via = 通过 candle-core 检测到 CUDA 设备：{ $a1 }（{ $a2 } MB，compute { $a3}.{ $a4}）
log-device-nvidia-smi-query-failed = nvidia-smi 查询失败，使用默认回退设备参数（{ $a1 }GB，compute { $a2}.{ $a3 }）。建议安装 nvidia-smi 以获取准确信息
log-device-cuda-device-manager-initialized = CUDA 设备管理器初始化完成，共 { $a1 } 个设备
log-device-created-memory-manager-device = 已为设备 { $a1 } 创建内存管理器：{ $a2 } MB
log-device-cuda-detection-returned-error = CUDA 探测返回错误，按无可用设备处理：{ $a1 }
log-device-gpu-compute-mode = GPU 计算模式：{ $a1 }
log-device-gpu-compute-mode-default = GPU 计算模式：Default（多进程共享）
log-device-gpu-compute-mode-exclusive = GPU 计算模式：Exclusive（最优）
log-device-gpu-ecc-memory-disabled = GPU ECC 显存：已关闭（当前：{ $a1 }）
log-device-gpu-ecc-memory-enabled = GPU ECC 显存：已启用
log-device-cannot-read-gpu-current = 无法读取 GPU 当前频率，无法判断是否已锁频
log-device-gpu-current-clock-frequency = GPU 当前频率：{ $a1 }
log-device-transparent-hugepages-disabled = 透明大页：已关闭
log-device-transparent-hugepages-madvise-mode = 透明大页：madvise 模式（GPU 推荐）
log-device-transparent-hugepages-enabled-always = 透明大页：已启用（always）
log-device-gpu-persistence-mode-disabled = GPU 持久模式：已关闭（当前：{ $a1 }）
log-device-gpu-persistence-mode-enabled = GPU 持久模式：已启用
log-device-gpu-tuning-check-complete = GPU 调优检查完成，发现 { $a1 } 项可优化配置：
log-device-gpu-tuning-check-complete-2 = GPU 调优检查完成：全部配置已是最优
log-device-auto-fallback-disabled-skipping = 自动回退已关闭，跳过回退到 CPU
log-device-memory-pressure-detected-falling = 检测到内存压力，回退到 CPU
log-device-available-memory-query-failed = 设备 { $a1 } 的 available_memory 查询失败，按可用 0 上报
log-device-available-devices-using-cpu = 无可用设备，使用 CPU
log-device-available-gpu-falling-back = 无可用 GPU，回退到 CPU
log-device-refresh-devices-amd-device = refresh_devices：AMD 设备数不一致（内部：{ $a1 }，amd_manager：{ $a2}）。以 amd_manager 为准
log-device-failed-initialize-amd-device = 初始化 AMD 设备管理器失败：{ $a1 }
log-device-cuda-device-initialized = CUDA 设备已初始化：{ $a1 }
log-device-failed-initialize-cuda-device-2 = 初始化 CUDA 设备管理器失败：{ $a1 }
log-device-memory-limit-exceeded-triggering = 超出内存限制，触发回退到 CPU
log-device-memory-limit-set-bytes = 内存上限已设置为：{ $a1 } 字节
log-device-failed-create-tokio-runtime = 创建 Tokio 运行时失败：{ $a1 }
log-device-memory-usage-critical-bytes = 内存使用告急：{ $a1 } 字节（{ $a2 }%）
log-device-memory-usage-exceeded-limit = 内存使用超出上限：{ $a1 } 字节（上限 { $a2 } 字节）
log-device-attempted-deallocate-bytes-but = 尝试释放 { $a1 } 字节，但实际已分配仅 { $a2 } 字节
log-device-decreasing-batch-size-latency = 批大小下调至 { $a1 }（时延 { $a2 }ms，内存 { $a3 }%）
log-device-increasing-batch-size-latency = 批大小上调至 { $a1 }（时延 { $a2 }ms，内存 { $a3 }%）
log-device-insufficient-gpu-memory-model = 模型“{ $a1 }”显存不足，使用最小批大小 { $a2 }
log-device-registered-memory-requirements-model = 已登记模型内存需求：{ $a1 }
log-api-logout-token-revocation-failed = 登出时令牌吊销失败（尽力而为）：{ $a1 }
log-api-failed-revoke-old-refresh = 刷新时吊销旧 refresh token 失败（login_id“{ $a1 }”）：{ $a2 }
log-api-login-rejected-auth-enabled = 登录被拒：认证已启用但未配置管理员密码（缺少 VECBOOST_ADMIN_PASSWORD）
log-api-switch-path-validation-rejected = switch 路径校验拒绝 { $field }={ $path }：{ $e }
log-api-switch-model-failed-e = switch_model 失败：{ $e }
log-api-batch-rerank-individual-query = 批量 rerank：单条查询失败，跳过：{ $a1 }
log-auth-auth-endpoint-rate-limit = 认证端点触发限流（IP：{ $a1 }）
log-auth-ratelimitmodule-registered = RateLimitModule 未注册：{ $a1 }
log-auth-ipwhitelistmodule-registered = IpWhitelistModule 未注册：{ $a1 }
log-auth-ratelimitenabled-registered-rate-limiting = RateLimitEnabled 未注册，限流已关闭。部分端点可能因此失去限流保护
log-auth-trusted-proxies-empty-x = trusted_proxies 为空——忽略 X-Forwarded-For，按直连对端 IP 处理
log-config-encryption-key-env-must = { $ENCRYPTION_KEY_ENV } 必须正好 32 字节（XChaCha20-Poly1305），实际 { $a1 } 字节
log-config-encryption-key-env-set = { $ENCRYPTION_KEY_ENV } 未设置——敏感配置字段将以明文存储。生产环境请设置该变量
log-domain-queue-cleared-requests-discarded = 队列已清空，丢弃 { $a1 } 个请求
log-domain-shutdown-drain-queued-requests = 关机排空：已迁出 { $a1 } 个排队请求
log-domain-reaped-expired-requests-from = 已从队列回收 { $a1 } 个过期请求，queue_size={ $a2 }
log-model-reloading-onnx-model-from = 正在从以下路径重新加载 ONNX 模型：{ $a1 }
log-model-reloading-candle-model-from = 正在从以下路径重新加载 Candle 模型：{ $a1 }
log-model-model-integrity-check-failed = 模型完整性检查失败，开始修复 { $a1 } 个损坏文件
log-model-model-integrity-check-passed = 模型完整性检查通过，无需修复
log-model-all-recovered-files-verified = 全部修复后的文件校验通过
log-model-download-attempt-failed-file = 第 { $a1 } 次下载失败（文件 { $a2 }）：{ $a3 }
log-model-download-attempt-file = 第 { $a1 } 次下载，文件：{ $a2 }
log-model-recovery-attempt-failed-file = 第 { $a1 } 次修复失败（文件 { $a2 }）：{ $a3 }
log-model-recovery-attempt-file = 第 { $a1 } 次修复，文件：{ $a2 }
log-model-local-model-path-detected = 检测到本地模型路径但未提供 repo_id，无法修复 { $a1 }
log-model-backed-up-corrupted-file = 损坏文件已备份至：{ $a1 }
log-model-verification-after-recovery-failed = 修复后校验失败：{ $a1 }
log-model-verifying-recovered-files = 正在校验修复后的文件……
log-model-failed-recover-file = 文件修复失败：{ $a1 }
log-model-successfully-recovered-file = 文件修复成功：{ $a1 }
log-model-attempting-recover-corrupted-file = 正在尝试修复损坏文件：{ $a1 }
log-model-starting-recovery-model-corrupted = 开始修复模型“{ $a1 }”，检测到 { $a2 } 个损坏文件
log-monitor-failed-get-memory-info = 获取内存信息失败：{ $a1 }
log-security-failed-read-key-file = 读取密钥文件元数据以收紧权限失败：{ $a1 }
log-security-failed-set-restrictive-permissions = 为密钥文件设置严格权限失败：{ $a1 }
log-service-failed-fallback-cpu = 回退到 CPU 失败：{ $a1 }
log-service-successfully-fell-back-cpu = 已成功回退到 CPU，重试该操作
log-service-fallback-already-triggered-cannot = 回退已触发过，无法再次重试
log-service-oom-error-detected-attempting = 检测到 OOM 错误：{ $a1 }。正在尝试回退到 CPU（第 { $a2 }/{ $a3 } 次）
log-service-decision-inference-task-failed = 决策推理任务失败：{ $e }
log-service-estimate-model-bytes-cannot = estimate_model_bytes：无法读取模型目录 { $a1 }（{ $e }），按 0 字节记账
log-service-memory-limit-accounting-initial = 内存上限记账（初始）：模型 { $a1 } ≈ { $est } 字节
log-service-local-model-config-cleared = 已清除本地模型配置：{ $a1 }
log-service-model-unloaded-via-modelmanager = 模型 { $a1 } 已由 ModelManager 卸载
log-service-memory-limit-accounting-model = 内存上限记账：模型 { $a1 } ≈ { $new_est } 字节
log-service-unloading-previous-model = 正在卸载先前模型：{ $a1 }
log-service-model-switched-successfully = 模型已成功切换为 { $a1 }
log-service-switching-model-from = 正在把模型从 { $a1 } 切换为 { $a2 }
log-service-normalize-l2-failed-truncated = 被截断的向量 normalize_l2 失败：{ $a1 }
log-service-processed-paragraphs-2-ms = 已处理 { $a1 } 个段落，耗时 { $a2 }ms
log-service-processed-lines-2-ms = 已处理 { $a1 } 行，耗时 { $a2 }ms
log-service-file-streaming-failed = 文件流式处理失败：{ $a1 }
log-service-cache-warm-up-completed = 缓存预热完成：预加载 { $a1 }/{ $a2 } 条
log-service-warm-up-batch-failed = 预热批次失败，跳过 { $a1 } 条文本：{ $a2 }
log-service-warm-up-skip-non = 预热：跳过非有限值向量（“{ $a1 }……”）
log-service-dimension-mismatch-expected-got = 维度不匹配：期望 { $a1 }，实际 { $a2 }。模型“{ $a3 }”可能被错误配置
log-service-embedding-cache-wal-replay = 嵌入缓存 WAL 回放来源：{ $a1 }
log-service-model-heat-save-failed = 模型热度落盘失败：{ $e }
log-service-transcribe-segments-chars-ms = transcribe：{ $a1 } 段、{ $a2 } 字符、{ $a3 }ms
log-service-translation-inference-task-failed = 翻译推理任务失败：{ $e }
log-worker-inference-panicked = Worker { $worker_id } 推理 panic：{ $e }
log-util-list-tree-failed-repo = 仓库“{ $repo_id }”的 list_tree 失败（{ $e }）；按尺寸校验的短路不可用
log-util-hf-endpoint-detected-hf = 检测到 HF_ENDPOINT={ $a1 }——hf-hub 1.0.0 依赖 ETag 头，部分镜像不提供
log-util-vquant-rotate-tail-not-hadamard = rotate：末尾 { $dims } 维未做 Hadamard 旋转（该段量化质量下降）
log-util-vquant-quantize-i8-non-finite = quantize_i8：输入向量含非有限值（len={ $len }），已置零
cli-poc-transcribe-own-e = transcribe-own：{ $e }
cli-poc-transcribe-own-pipeline-done = transcribe-own：全管线完成，耗时 { $wall_s }s -> { $a1 }
cli-poc-transcribe-own-vad-only = transcribe-own：仅 VAD 完成，耗时 { $wall_s }s -> { $a1 }
cli-poc-transcribe-own-3-s = transcribe-own：[{ $a1 }s] worker { $w } 段 { $a2 }/{ $a3 } [{ $a4 }s-{ $a5 }s] { $a6 } 字符
cli-poc-transcribe-own-3-s-2 = transcribe-own：[{ $a1 }s] VAD 完成：{ $a2 } 段
cli-poc-transcribe-own-wav-loaded = transcribe-own：wav 已加载：{ $a1 } 采样点，{ $audio_seconds } 秒
cli-poc-transcribe-own-name-path = transcribe-own：{ $name } 路径含注解文本，已归一：{ $a1 } -> { $a2 }
cli-poc-transcribe-own-backend-py = transcribe-own：backend=py-torch（内嵌 CPython，GPU 1.7B bf16）
cli-poc-transcribe-own-key-retired = transcribe-own：{ $key } 已退役（实验路径已移除；结论记录在 out/ 汇总文档）
cli-poc-transcribe-own-decoding-segments = transcribe-own：正在解码 { $segments_len } 段，使用 { $workers } 个 worker（ort-intra={ $intra }）；将加载 { $workers } 套会话……

# ── CLI --help 协议面（sdforge 注册表；en 侧逐字对齐 #[forge(description)] 字面量）──
forge-cli-compute-similarity = 计算两段文本的余弦相似度
forge-cli-decisions = 就决策状态回答 choice/score/noul 三类问题
forge-cli-detect = 检测文本的自然语言（返回 ISO 639-1 编码，必要时回退 639-3）
forge-cli-embed = 为输入文本生成嵌入向量
forge-cli-embed-batch = 批量为多条文本生成嵌入向量
forge-cli-rerank = 按与查询的相关性重排文档
forge-cli-search = 按与查询的相似度对候选文本排序（1 对 N 检索）
forge-cli-transcribe = 将 16kHz 单声道 WAV 文件转录为文本（Qwen3-ASR；需启用 [transcribe]）
forge-cli-translate = 在已配置的语言对之间翻译文本（seq2seq）

# ── 预热诊断（main.rs 预热路径） ──
warmup-decision-pipeline-failed = warmup: 决策管线全链预热失败：{ $e }
warmup-mode-decide = decide×1（决策管线全链）
warmup-mode-embed = embed
warmup-incomplete = warmup: 预热未完全生效（mode={ $mode }，{ $failures } 次失败，{ $elapsed }）——首次请求可能更慢
warmup-complete = warmup: 预热推理完成（mode={ $mode }，{ $elapsed }）

# ── Candle 原生决策头：错误返回 + 加载期诊断 ──
engine-candle-decision-checkpoint-not-found = 未找到 Laya PyTorch checkpoint：{ $path }（获取步骤见 docs/USER_GUIDE.md「Candle 原生决策路径」；若已有 receptron/laya-onnx 的 ONNX bundle，可设置 engine_type = "onnx" 走 ort 管线沿用既有资产）
engine-candle-decision-encoder-config-read = 读取 encoder config { $path } 失败：{ $detail }（官方 repo convaiinnovations/laya 的 encoder/config.json，获取步骤见 docs/USER_GUIDE.md）
log-engine-verifying-candle-decision-checkpoint-sha256 = 正在校验 Candle 决策 checkpoint 的 SHA256 摘要……
log-engine-candle-decision-engine-initialized = Candle 决策引擎初始化完成：bundle={ $a1 }，hidden={ $a2 }，layers={ $a3 }，calibration_buckets={ $a4 }
log-engine-candle-decision-engine-using-cuda-gpu = Candle 决策引擎使用 CUDA GPU
log-engine-candle-decision-engine-using-metal-gpu = Candle 决策引擎使用 Metal GPU
log-engine-candle-decision-amd-fallback-cpu = Candle 引擎不原生支持 AMD GPU；Candle 决策引擎回退到 CPU

# ── ONNX 翻译引擎：解码路径 + 加载期诊断 ──
engine-translate-logits-nonempty = logits 非空
log-engine-tokenizer-shim-in-memory = tokenizer { $a1 }：{ $a2 }（内存内修正；tokenizers 0.23 解析契约）
log-engine-translate-generation-config-unreadable = 翻译 bundle { $a1 }：generation_config.json 不可读（{ $e }）；解码锚点回退用 tokenizer 特殊 token
log-engine-translate-generation-config-missing-ids = 翻译 bundle { $a1 }：generation_config.json 缺少 decoder_start_token_id/eos_token_id；解码锚点回退用 tokenizer 特殊 token
log-engine-translate-parallel-workers-lazy = 翻译并行 worker 数={ $a1 }（惰性；额外会话集在首次长输入并行请求时构建）
log-engine-translation-engine-initialized = 翻译引擎初始化完成：bundle={ $a1 }，encoder={ $a2 }，decoder={ $a3 }，precision={ $a4 }，max_source_tokens={ $a5 }，max_new_tokens={ $a6 }，decoder_start_token_id={ $a7 }，eos_token_id={ $a8 }，past_tensors={ $a9 }
log-engine-translate-whole-text-decode-degenerated = 翻译整段解码退化（已生成 { $a1 } 个 token，{ $a2 } 句）；改用逐句重试，输出预算 { $a3 }
log-engine-translate-rescue-sentence-count-exceeds = 救援句数 { $a1 } 超过上限 { $max }；其余句子已丢弃
log-engine-translate-rescue-budget-exhausted = 救援输出预算在生成 { $a1 } 个 token 后耗尽；其余句子已丢弃
log-engine-translate-parallel-session-lazy-loaded = 翻译并行 worker 会话集已惰性加载（{ $a1 }/{ $a2 } 套额外会话就绪）
log-engine-translate-parallel-sentence-count-exceeds = 并行句数 { $a1 } 超过上限 { $max }；其余句子已丢弃
log-engine-translate-protected-token-not-recovered = 翻译中受保护 token 未被还原（占位符 { $a1 }）：{ $a2 }
log-engine-translate-attempting-cpu-fallback = 翻译引擎正在尝试回退到 CPU
log-engine-translate-successfully-fell-back-cpu = 翻译引擎已成功回退到 CPU

# ── 温度校准表（决策管线）+ checkpoint 盘点（main.rs） ──
log-engine-calibration-file-missing-fallback-temperature = 校准文件缺失：{ $a1 }；概率将过度自信（模型卡：未校准 ECE 0.466 对已校准 0.081）；全部基数回退温度 { $temp }
log-engine-calibration-no-temperature-by-options = { $a1 } 缺少 temperature_by_options 字段；全部基数回退温度 { $temp }
log-main-model-checkpoints-resolved = 预设 checkpoint 解析完成：共 { $a1 } 个（内置 laya-multilingual + { $a2 } 个用户配置，其中 { $a3 } 个覆盖内置条目）

# ── transcribe-own CLI：用法 / 参数 / 环境旋钮报错（stderr） ──
cli-poc-transcribe-own-usage = 用法：transcribe-own <wav> <outDir> [--models <dir>] [--vad-only]
cli-poc-transcribe-own-models-requires-value = --models 需要一个取值
cli-poc-transcribe-own-unknown-argument = 未知参数：{ $a1 }
cli-poc-transcribe-own-env-not-valid-integer = { $key }={ $value } 不是合法整数
cli-poc-transcribe-own-env-out-of-range = { $key }={ $value } 超出范围 [{ $min },{ $max }]
cli-poc-transcribe-own-ep-invalid = TRANSCRIBE_EP={ $a1 } 无效（cpu|cuda）
