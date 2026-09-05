# VecBoost API 响应消息 — 中文

# ── 验证消息 ──
validate-text-length = 索引 { $index } 处的文本超过最大长度 { $max }（实际 { $got }）
validate-batch-size = 批处理大小 { $size } 超过最大限制 { $max }（配置项 embedding.max_batch_size）
validate-path-failed = 路径验证失败：{ $detail }
sensitive-dir-refused = 拒绝使用敏感目录 '{ $path }' 作为允许根目录；请在 [server] grpc_allowed_roots 中显式配置

# ── OpenAI 兼容端点 ──
openai-input-empty = 输入不能为空
openai-input-too-large = 输入数组过大（最多 { $max } 项）

# ── 认证消息 ──
auth-password-empty = 密码不能为空
auth-invalid-credentials = 用户名或密码无效
auth-refresh-token-empty = 刷新令牌不能为空
auth-invalid-token = 令牌无效或已过期
logout-success = 登出成功。令牌已撤销。

# ── 模型管理 ──
model-switch-success = 模型切换成功
model-already-current = 已在使用该模型
model-load-failed = 加载模型 '{ $name }' 失败：{ $detail }

# ── 文件嵌入 ──
file-empty = 文件为空
file-no-paragraphs = 文件中未找到段落
file-invalid-encoding = 无效的路径编码：路径包含非法 UTF-8 字符

# ── 内存不足 / 降级 ──
oom-no-fallback = 内存不足且已尝试降级回退

# ── 健康检查 ──
health-ok = 正常
health-check-failed = { $module }：健康检查失败：{ $detail }

# ── Rerank 验证 ──
rerank-empty-docs = 文档列表不能为空
rerank-too-many-docs = 文档数量 { $count } 超过单次查询最大限制 { $max }
rerank-invalid-top-k = 指定 top_k 时必须大于 0
rerank-unsupported = 当前引擎不支持 rerank
rerank-query-too-long = 查询长度 { $length } 超过最大允许长度 { $max }

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
