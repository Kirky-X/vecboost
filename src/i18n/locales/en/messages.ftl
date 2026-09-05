# VecBoost API Response Messages — English

# ── Validation messages ──
validate-text-length = Text at index { $index } exceeds max length { $max } (got { $got })
validate-batch-size = Batch size { $size } exceeds max { $max } (config embedding.max_batch_size)
validate-path-failed = Path validation failed: { $detail }
sensitive-dir-refused = Refusing to use sensitive directory '{ $path }' as allowed root; configure [server] grpc_allowed_roots explicitly

# ── OpenAI-compatible endpoint ──
openai-input-empty = Input cannot be empty
openai-input-too-large = Input array too large (max { $max } items)

# ── Authentication messages ──
auth-password-empty = Password must not be empty
auth-invalid-credentials = Invalid username or password
auth-refresh-token-empty = Refresh token must not be empty
auth-invalid-token = Invalid or expired token
logout-success = Logout successful. Token has been revoked.

# ── Model management ──
model-switch-success = Model switched successfully
model-already-current = Already using this model
model-load-failed = Failed to load model '{ $name }': { $detail }

# ── File embedding ──
file-empty = File is empty
file-no-paragraphs = No paragraphs found in file
file-invalid-encoding = Invalid path encoding: path contains invalid UTF-8

# ── OOM / fallback ──
oom-no-fallback = Out of memory and fallback already attempted

# ── Health check ──
health-ok = OK
health-check-failed = { $module }: health check failed: { $detail }

# ── Rerank validation ──
rerank-empty-docs = Documents list cannot be empty
rerank-too-many-docs = Documents count { $count } exceeds max documents per query { $max }
rerank-invalid-top-k = top_k must be greater than 0 when specified
rerank-unsupported = Current engine does not support rerank
rerank-query-too-long = Query length { $length } exceeds maximum allowed length { $max }

# ── Directory errors ──
dir-get-cwd-failed = Failed to get current directory: { $detail }
