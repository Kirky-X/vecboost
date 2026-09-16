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
auth-admin-password-missing = Server has no admin password configured; authentication is unavailable. Set VECBOOST_ADMIN_PASSWORD and restart
auth-admin-required = Admin privileges are required for this operation
auth-credentials-missing = Missing authentication credentials
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

# ── Auth username validation ──
auth-username-length = Username must be between 3 and 32 characters
auth-username-start = Username must start with a letter
auth-username-charset = Username may only contain letters, digits, underscores and hyphens

# ── Text / batch validation ──
validate-text-empty = Text cannot be empty
validate-text-too-short = Text too short: { $got } characters (minimum: { $min })
validate-text-too-long = Text too long: { $got } characters (maximum: { $max })
validate-text-whitespace = Text contains only whitespace
validate-batch-empty = Batch cannot be empty
validate-text-index-failed = Validation failed for text at index { $index }: { $detail }
validate-search-empty = Search texts list cannot be empty
validate-top-k-exceeded = top_k { $got } exceeds maximum { $max }

# ── File validation ──
file-too-large = File size { $size } MB exceeds maximum allowed size { $max } MB
file-access-failed = Cannot access file { $path }: { $detail }
file-no-extension = File has no extension
file-extension-not-allowed = File extension '{ $ext }' is not allowed. Allowed extensions: { $allowed }
file-open-failed = Cannot open file: { $detail }
file-read-failed = Cannot read file: { $detail }
file-binary-rejected = File contains non-text binary data

# ── Path validation ──
path-traversal-detected = Path traversal attempt detected: { $detail }
path-invalid = Invalid path: { $detail }
path-no-roots = No allowed root directories configured for file access
path-access-denied = Access denied: path '{ $path }' is not within allowed directories. { $detail }
path-not-file = Path is not a file: { $path }
path-not-dir = Path is not a directory: { $path }

# ── Pipeline / queue ──
pipeline-channel-error = Response channel error
pipeline-timeout = Request timeout
queue-type-mismatch = Expected Embed request but got Rerank
queue-full-rejected = Queue is full, request rejected
rerank-not-configured = Rerank service not configured
auth-disabled = Authentication is disabled at runtime
auth-verify-failed = Password verification failed
api-init-state-called = init_state already called
api-init-state-missing = init_state not called

# ── OOM / engine (P1) ──
oom-no-fallback-available = Out of memory and no fallback available
oom-fallback-failed = OOM error and fallback failed: { $detail }
oom-max-attempts = Max fallback attempts exceeded ({ $attempts }). Last error: { $detail }
engine-semaphore-failed = Failed to acquire semaphore: { $detail }
engine-batch-timeout = Batch chunk processing timed out after { $secs }s

# ── Middleware error responses ──
rate-limit-exceeded = Rate limit exceeded
route-not-found = The requested resource was not found

# ── Metrics endpoint (P1) ──
metrics-limiter-unavailable = Rate limiter unavailable
metrics-rate-limited = Rate limit exceeded
metrics-collector-missing = PrometheusCollector not configured
metrics-encode-failed = Failed to encode metrics: { $detail }

# ── Startup / config (P1) ──
startup-db-pool = Failed to create database pool: { $detail }
startup-db-schema = Failed to initialize database schema: { $detail }
startup-jwt-length = JWT secret must be at least 32 characters long for security. Current length: { $got }
startup-jwt-missing = JWT secret is required when authentication is enabled. Please provide a strong JWT secret
startup-admin-password-missing = Admin password is required when authentication is enabled. Set the VECBOOST_ADMIN_PASSWORD environment variable (otherwise any credentials would be able to log in)
startup-logger = Failed to initialize inklog logger: { $detail }
startup-config = Failed to load config: { $detail }
startup-encryption = Encryption key validation failed: { $detail }
startup-register-failed = Failed to register { $module }: { $detail }
startup-grpc-bearer-failed = gRPC require_auth=true but BearerAuth creation failed: { $detail }. Set VECBOOST_JWT_SECRET (>=32 chars) or set [server] grpc_require_auth = false for dev
startup-grpc-no-secret = gRPC require_auth=true but auth.jwt_secret is None. Set VECBOOST_JWT_SECRET env var or set [server] grpc_require_auth = false for dev
startup-grpc-auth-disabled = gRPC require_auth=true but auth.enabled=false. Enable [auth] enabled = true or set [server] grpc_require_auth = false
startup-grpc-no-feature = gRPC require_auth=true but vecboost `auth` feature is not enabled. Enable `auth` feature or set [server] grpc_require_auth = false in config
config-jwt-empty = VECBOOST_JWT_SECRET cannot be empty
config-jwt-length = VECBOOST_JWT_SECRET must be at least { $min } characters
config-password-empty = VECBOOST_ADMIN_PASSWORD cannot be empty
config-password-length = VECBOOST_ADMIN_PASSWORD must be at least { $min } characters
config-encryption-missing = { $key } environment variable is not set. Production deployments MUST configure it
config-encryption-length = { $key } must be exactly 32 bytes
cli-no-handler = No handler registered for CLI command: { $name }
cli-failed = CLI command '{ $name }' failed: { $detail }
engine-bert-config-required = Bert config is required for Bert model
engine-xlm-config-required = XLM-RoBERTa config is required
