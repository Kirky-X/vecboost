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
