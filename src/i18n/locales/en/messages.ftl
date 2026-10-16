# VecBoost API Response Messages — English

# ── Validation messages ──
validate-text-length = Text at index { $index } exceeds max length { $max } (got { $got })
validate-batch-size = Batch size { $size } exceeds max { $max } (config embedding.max_batch_size)
validate-path-failed = Path validation failed: { $detail }
sensitive-dir-refused = Refusing to use sensitive directory '{ $path }' as allowed root; configure [server] grpc_allowed_roots explicitly

# ── OpenAI-compatible endpoint ──
openai-input-empty = Input cannot be empty
openai-input-too-large = Input array too large (max { $max } items; server embedding.max_batch_size = { $effective })
openai-model-not-found = The model '{ $model }' does not exist. Available models: { $available }

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
switch-path-rejected = path rejected: must be an existing path inside the allowed roots (model_path must be a directory, tokenizer_path must be a file); see server logs
switch-model-load-failed = model load failed; see server logs

# ── File embedding ──
file-empty = File is empty
file-no-paragraphs = No paragraphs found in file
file-invalid-encoding = Invalid path encoding: path contains invalid UTF-8
embed-file-too-large = File exceeds the { $max } MiB limit for /embed/file (got { $got } bytes)

# ── OOM / fallback ──
oom-no-fallback = Out of memory and fallback already attempted

# ── Health check ──
health-ok = OK
health-check-failed = { $module }: health check failed: { $detail }
health-engine-probe-failed = tokenizer/engine pipeline probe failed
health-limiter-failed = limiteron health check failed

# ── Rerank validation ──
rerank-empty-docs = Documents list cannot be empty
rerank-too-many-docs = Documents count { $count } exceeds max documents per query { $max }
rerank-invalid-top-k = top_k must be greater than 0 when specified
rerank-unsupported = Current engine does not support rerank
rerank-query-too-long = Query length { $length } exceeds maximum allowed length { $max }

# ── Decision validation ──
decision-empty-state = state must not be empty (decision context state is required)
decision-empty-questions = questions must not be empty (at least one question required)
decision-too-many-questions = questions count { $count } exceeds max per request { $max }
decision-unsupported = Current engine does not support decision inference
decision-inference-task-panicked = decision inference task failed (engine task panicked); see server logs

# ── Detect validation ──
detect-empty-text = text must not be empty (language detection requires text)
detect-too-long = text length { $length } exceeds maximum allowed length { $max }
detect-invalid-allowlist = language code in allowlist is not supported: { $code } (only codes published by the detection engine)

# ── Translate validation & routing ──
translate-empty-text = text must not be empty (translation requires text)
translate-too-long = text length { $length } exceeds maximum allowed length { $max }
translate-unsupported = Current engine does not support translation inference
translate-not-configured = Translation service has no language pairs configured (add [[translation.pairs]] under [translation])
translate-unsupported-pair = Requested language pair { $requested } is not configured (configured: { $pairs })
translate-detect-failed = Source language detection is unreliable; specify source_language explicitly
translate-too-many-source-tokens = Source text length { $count } exceeds the pair limit { $max }
translation-inference-task-panicked = translation inference task failed (engine task panicked); see server logs

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
validate-text-control-chars = Text contains illegal control characters
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
decision-not-configured = Decision service not configured
pipeline-response-type-mismatch = Pipeline returned a mismatched response type
request-queue-sla-exceeded = request waited { $waited }s in queue, exceeding its { $sla }s SLA
inference-task-panicked = inference task panicked
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

# ── CLI invocation errors (main.rs fail-fast paths) ──
cli-unknown-subcommand = Error: unknown subcommand '{ $name }'
cli-available-subcommands = Available subcommands: { $list }
cli-usage-hint = Run 'vecboost --help' for usage.
cli-config-requires-path = Error: --config requires a path argument (--config <path>)
cli-config-not-found = Error: config file not found: { $path }
cli-config-not-file-hint = The --config path must point to an existing TOML file.

# ── Runtime diagnostics / logs ──
numa-advice = NUMA detected: { $sockets } sockets; consider starting with `numactl --interleave=all` or `--cpunodebind` to balance memory bandwidth (no in-process binding)
heat-read-failed = model heat: failed to read { $path } ({ $detail }); warmstart skipped
heat-parse-failed = model heat: failed to parse { $path } ({ $detail }); file discarded
heat-schema-mismatch = model heat: schema version mismatch (file { $got }, expected { $expected }); file discarded
heat-entry-fingerprint-mismatch = model heat: entry '{ $name }' fingerprint mismatch; entry discarded
doctor-tokenizer-vocab-missing = ; ⚠️ tokenizer vocab file not identified by content
transcribe-not-configured = Transcription service is not configured: set enabled = true and models_dir under [transcribe]

# ── Doctor CLI diagnostics (vecboost doctor stdout) ──
doctor-config-pass = config validated OK (model.task={ $task })
doctor-config-fail = config validation failed: { $detail }
doctor-model-dir-missing = model directory not found: { $path } (will attempt HF download at runtime)
doctor-tokenizer-json-missing = tokenizer.json not found: { $path }
doctor-tokenizer-json-ok = tokenizer.json loaded successfully: { $path }
doctor-tokenizer-json-fail = tokenizer.json failed to load: { $detail }
doctor-cache-memory-only = in-memory mode only (persist_path not configured)
doctor-cache-wal-writable = WAL file writable: { $path }
doctor-cache-wal-unwritable = WAL file not writable: { $path }: { $detail }
doctor-cache-path-invalid = invalid persist_path: { $path }
doctor-cache-parent-missing = parent directory does not exist (startup writes would fail; create it or fix config first): { $path }
doctor-cache-dir-writable = directory writable (WAL not yet created; established on first insert): { $path }
doctor-cache-dir-unwritable = directory not writable: { $path }: { $detail }
doctor-threads-detailed = physical cores { $physical } / logical cores { $logical } (tuning uses physical cores)
doctor-threads-fallback = physical core detection unavailable, falling back to logical cores { $logical } (VECBOOST_NO_THREAD_TUNE can override)
doctor-threads-numa = ; NUMA detected { $sockets } socket(s), consider numactl --interleave=all
doctor-gpu-cpu-mode = CPU mode (use_gpu=false)
doctor-gpu-no-cuda = use_gpu=true but no CUDA device detected
doctor-gpu-devices = CUDA devices: { $names }
doctor-gpu-no-cuda-feature = use_gpu=true but binary was built without the cuda feature (rebuild with --features cuda)
doctor-models-none = no local model directory found (models/ absent and model_path unset); runtime will rely on HF download
doctor-models-fail-count = { $count } model director(ies) have integrity problems
doctor-dir-unreadable = directory unreadable
doctor-json-parse-fail = JSON file failed to parse { $path }: { $detail }
doctor-json-unreadable = JSON file unreadable { $path }: { $detail }
doctor-dims-no-hidden = no 2-D tensor dimension matches config.hidden_size({ $hidden }); weights and config appear mismatched
doctor-no-weights = no weight files found (.safetensors/.gguf)
doctor-weights-summary = weights: { $weights }, gguf: { $gguf }, tokenizer: { $tokenizer }, config.hidden_size: { $hidden }
doctor-tokenizer-recognized = recognized
doctor-tokenizer-unrecognized = not recognized
doctor-weight-unreadable = weight file unreadable { $path }: { $detail }
doctor-metadata-unreadable = metadata unreadable { $path }: { $detail }
doctor-file-too-small = file too small, under 8-byte header: { $path }
doctor-header-len-read-fail = failed to read header length { $path }: { $detail }
doctor-header-len-invalid = illegal safetensors header length ({ $len }): { $path }
doctor-header-read-fail = failed to read header { $path }: { $detail }
doctor-header-json-fail = safetensors header JSON parse failed { $path }: { $detail }
doctor-tensor-not-object = tensor { $tensor } metadata is not an object: { $path }
doctor-tensor-no-dtype = tensor { $tensor } missing dtype: { $path }
doctor-tensor-no-offsets = tensor { $tensor } missing data_offsets: { $path }
doctor-tensor-offsets-invalid = tensor { $tensor } illegal data_offsets: { $path }
doctor-tensor-offsets-reversed = tensor { $tensor } data_offsets reversed: { $path }
doctor-file-truncated = file truncated (data needs { $need } bytes, have { $have }): { $path }
doctor-gguf-unreadable = gguf file unreadable { $path }: { $detail }
doctor-gguf-too-small = gguf file too small { $path }: { $detail }
doctor-gguf-bad-magic = invalid gguf magic (got { $got }, expected GGUF): { $path }

# ── Startup fail-fast / thread-tune banner (main.rs stderr) ──
startup-engine-type-unknown = [model] engine_type has unknown value "{ $got }"{ $hint }, supported: candle, onnx
startup-engine-onnx-rebuild-hint = (engine_type="onnx" requires rebuilding with --features onnx)
# onnx-primary: engine_type="candle" is explicitly rejected in builds without the `candle` feature (candle is an opt-in fallback backend)
startup-engine-candle-not-compiled = [model] engine_type="candle" but this build did not compile the candle backend (onnx-primary defaults to the onnx engine); rebuild with `--features candle`, or remove the engine_type config to use the default onnx engine
startup-semantic-mode-invalid = [semantic_cache] comparison_mode invalid: { $detail }
startup-translation-no-onnx = [translation] pairs configured but this build did not enable the `onnx` feature; rebuild with --features onnx
startup-translate-pair-invalid = [translation] pairs[{ $index }] ({ $src }-{ $tgt }) invalid parameters: { $detail }
startup-translate-pair-load-failed = [translation] pairs[{ $index }] ({ $src }-{ $tgt }) failed to load: { $detail }
startup-transcribe-session-open = transcribe session open: { $detail }
startup-thread-tune = [thread-tune] physical cores={ $physical } logical cores={ $logical } effective threads={ $effective } (explicit config={ $explicit })
startup-thread-unknown = unknown (fallback)
startup-thread-unset = unset
startup-rayon-skipped = [thread-tune] rayon global pool already initialized, skipping: { $detail }
semantic-cache-mode-unknown = unknown comparison_mode: '{ $got }' (choose exact|i8|binary)

# ── Engine routing / checkpoint config errors (user-visible error returns) ──
factory-translate-onnx-only = Translation only supports the onnx engine: set engine_type to onnx (task=translate has no candle implementation; restart the service to reload)
factory-transcribe-onnx-only = Transcription only supports the onnx engine: set engine_type to onnx (task=transcribe has no candle implementation)
factory-translate-no-onnx-feature = Detected a task=translate model config, but this build did not enable the `onnx` feature; rebuild with `--features onnx`
config-checkpoint-engine-unknown = checkpoint { $name } engine_type has unknown value "{ $got }"{ $hint }, supported: candle, onnx
cache-persist-uncreatable = cannot create persist file { $path }: { $detail } (check the [embedding] persist_path directory permissions)
factory-transcribe-no-transcribe-feature = Detected a task=transcribe model config, but this build did not enable the `transcribe` feature; rebuild with `--features onnx,transcribe`
factory-quantized-no-feature = Detected a GGUF quantized model config (.gguf + quantized=true), but this build did not enable the `quantized-gguf` feature; rebuild with `--features quantized-gguf`
factory-decision-no-onnx-feature = Detected a task=decision model config, but this build did not enable the `onnx` feature; rebuild with `--features onnx`

# ── Quantized GGUF engine errors (feature quantized-gguf; error-model-load/inference detail) ──
quant-gguf-open-failed = cannot open GGUF file { $path }: { $detail }
quant-gguf-too-short = GGUF file too short { $path }: { $detail }
quant-gguf-magic-mismatch = GGUF magic check failed { $path }: expected 0x{ $expected }, got 0x{ $got }
quant-safetensors-load-failed = safetensors load failed { $path }: { $detail }
quant-config-json-unreadable = config.json unreadable: { $detail }
quant-config-json-parse-failed = config.json parse failed: { $detail }
quant-config-json-missing-key = config.json missing field { $key }
quant-tensor-no-mapping = safetensors tensor { $name } has no GGUF mapping (unsupported architecture or name drift)
quant-tensor-dtype-failed = tensor { $name } failed both target dtype and F16 fallback: { $detail }
quant-tensor-quantize-failed = tensor { $name } quantization failed: { $detail }
quant-gguf-output-unwritable = GGUF output not writable { $path }: { $detail }
quant-gguf-write-failed = GGUF write failed: { $detail }
quant-gguf-flush-failed = GGUF output flush failed: { $detail }
quant-gguf-openfile-failed = failed to open GGUF file { $path }: { $detail }
quant-gguf-header-parse-failed = GGUF header parse failed: { $detail }
quant-gguf-arch-unsupported = GGUF architecture { $arch } not supported (only bert; file: { $path })
quant-gguf-meta-missing-key = GGUF metadata missing key { $key }
quant-tensor-read-failed = failed to read GGUF tensor { $name }: { $detail }
quant-tensor-dequant-failed = failed to dequantize GGUF tensor { $name }: { $detail }
quant-tensor-no-candle-mapping = GGUF tensor { $name } has no candle mapping
quant-tensor-mapping-conflict = GGUF tensor { $name } mapping conflict (duplicate { $candle })
quant-gguf-missing-token-embd = GGUF is missing token_embd.weight
quant-token-embd-dim-failed = failed to read token_embd dimension: { $detail }
quant-bert-assembly-failed = GGUF BERT assembly failed: { $detail }
quant-tokenizer-load-failed = failed to load tokenizer.json from the GGUF directory ({ $path }): { $detail }
quant-empty-batch = empty batch

# ── Runtime logs: module registry / metrics endpoint / doctor banner ──
# Convention: `Tag: message` identifier prefixes stay in code (machine-greppable);
# only the message part is localized here.
log-registry-engine-ready = model engine ready
log-registry-service-ready = service ready
log-registry-ratelimit-active = on_ready — limiteron rate limiter active
log-registry-ratelimit-shutdown = on_shutdown — rate limiter shut down
log-registry-audit-flushing = flushing audit log before shutdown
log-registry-audit-flush-failed = failed to flush audit log: { $detail }
log-registry-watcher-ready = config file watcher ready
log-registry-watcher-shutdown = shutting down config watcher
log-registry-watcher-shutdown-failed = error during watcher shutdown: { $detail }
log-metrics-collector-unregistered = PrometheusCollectorModule not registered: { $detail }
log-metrics-response-build-failed = Failed to build metrics response: { $detail }
log-metrics-ip-allowlist-unregistered = IpWhitelistModule not registered: { $detail }
log-metrics-ratelimit-unavailable = RateLimitModule not available while RateLimitEnabled=true: { $detail }. Denying request.
log-metrics-json-response-build-failed = Failed to build JSON error response: { $detail }

# ── Doctor report chrome (stdout; PASS/WARN/FAIL tokens stay untranslated) ──
doctor-report-title = vecboost doctor — read-only diagnostics
doctor-summary = { $total } check(s): { $pass } pass, { $warn } warn, { $fail } fail

# ── Runtime logs: server startup / assembly banners (main.rs stdout+stderr) ──
log-main-building-module = Building module: { $module }
log-main-module-built-slow = Module { $module } built in { $seconds }s (>1s)
log-main-module-built = Module { $module } built in { $ms }ms
log-main-db-pool-init = Initializing database pool with url={ $url }
log-main-db-retry = Database retry policy enabled (max_retries={ $retries }, exponential backoff)
log-main-db-warmup = dbnexus pool-warmup enabled: min_connections={ $min }, warmup_timeout={ $timeout }s, warmup_retries={ $retries }
log-main-db-ready = Database pool initialized and schema verified
log-main-db-metrics = dbnexus MetricsCollector created
log-main-tokenizer-noop = [model].tokenizer_path is configured but no consumer matches engine_type={ $engine } task={ $task }: consumers are the onnx local bundle branch (engine_type=onnx + local directory) and the decision pipeline (task=decision + local directory); the value is a no-op
log-main-kvcache-enabled = KV Cache enabled with size: { $size }
log-main-kvcache-wal = Embedding cache WAL persistence enabled: { $path } (compact threshold { $bytes } bytes)
log-main-kvcache-disabled = KV Cache disabled
log-main-residency = model residency: max_resident_models={ $max } budget_mb={ $budget }
log-main-semantic-mode-deprecated = [semantic_cache] comparison_mode={ $mode } is deprecated and has no effect in this version (always exact)
log-main-semantic-enabled = semantic cache enabled: threshold={ $threshold } fuzzy_threshold={ $fuzzy } capacity={ $capacity }
log-main-translation-threads = translation: per-session intra threads budget = { $budget } ({ $pairs } pairs)
log-main-engine-init = Initializing Inference Engine (this may take a while to download models)...
log-main-translation-ready = translation service initialized: { $pairs } pair(s) configured
log-main-transcribe-ready = transcribe service initialized: bundle={ $bundle }
log-main-mcp-start = Starting VecBoost MCP server over stdio
log-main-auth-enabled = Garrison authentication enabled (JWT + session + password verification)
log-main-auth-disabled = Authentication disabled
log-main-csrf-enabled = CSRF protection enabled (garrison)
log-main-csrf-disabled = CSRF protection disabled
log-main-plan-hardware = auto_plan: hardware plan worker_threads={ $threads } max_batch_size={ $batch } batch_wait_ms={ $wait } quantized_recommended={ $quantized } bottleneck={ $bottleneck } rationale={ $rationale }
log-main-plan-applied = auto_plan: applied max_batch_size={ $batch } batch_wait_ms={ $wait } (planned max_batch_size={ $planned_batch } batch_wait_ms={ $planned_wait } suggested worker_threads={ $threads })
log-main-pipeline-enabled = Request pipeline enabled with queue_size={ $size }
log-main-worker-start-failed = WorkerManager start failed: { $detail }
log-main-pipeline-ready = Pipeline components initialized successfully
log-main-pipeline-disabled = Request pipeline disabled
log-main-gpu-no-backend = use_gpu=true but no GPU backend is compiled in (missing 'cuda' feature, or 'metal' on macOS) — falling back to CPU. Rebuild with --features cuda for GPU.
log-main-insecure-warning = SECURITY RISK: authentication is disabled while binding to non-loopback address '{ $host }' (VECBOOST_ALLOW_INSECURE=1). Anyone reachable on this interface can read files via /embed/file and manage models. Enable auth or bind to 127.0.0.1.
log-main-insecure-refusal = Refusing to start: authentication is disabled (auth.enabled=false) while binding to non-loopback address '{ $host }'. Fix one of: (1) set host to 127.0.0.1 in the config, (2) enable [auth] with VECBOOST_JWT_SECRET and VECBOOST_ADMIN_PASSWORD, or (3) set VECBOOST_ALLOW_INSECURE=1 to accept the risk explicitly.
log-main-jemalloc-decay = jemalloc decay: dirty={ $dirty }ms muzzy={ $muzzy }ms (boot conf)
log-main-jemalloc-probe-failed = jemalloc decay probe failed: { $detail }
cli-error-prefix = Error: { $detail }

# ── Runtime logs: engine / device / cache / service / CLI diagnostics ──
log-audit-failed-rotate-current-log = Failed to rotate current log: { $a1 }
log-audit-failed-rotate-log-file = Failed to rotate log file: { $a1 }
log-audit-failed-remove-old-log = Failed to remove old log file { $a1 }: { $a2 }
log-audit-failed-write-audit-log = Failed to write audit log batch: { $a1 }
log-audit-failed-reopen-log-file = Failed to reopen log file after rotation: { $a1 }
log-audit-audit-event-lost-dead = Audit event lost: dead-letter write failed: { $e } (db error: { $db_error })
log-audit-audit-event-dead-lettered = Audit event dead-lettered to { $a1 }
log-audit-failed-open-audit-log = Failed to open audit log file: { $a1 }
log-audit-failed-create-log-directory = Failed to create log directory: { $a1 }
log-cache-persist-one-record-key = persist: { $a1 } one record key is not UTF-8, skipped
log-cache-persist-one-record-fingerprint = persist: { $a1 } one record fingerprint mismatch, skipped
log-cache-persist-one-record-failed = persist: { $a1 } one record failed checksum, skipped
log-cache-persist-version-mismatch-discarding = persist: { $a1 } version { $a2 } mismatch, discarding whole file
log-cache-persist-magic-mismatch-discarding = persist: { $a1 } magic mismatch, discarding whole file
log-cache-persist-unknown-record-type = persist: { $a1 } unknown record type { $a2 }, stopping replay (torn tail)
log-cache-persist-cannot-open-treating = persist: cannot open { $a1 } ({ $a2 }), treating as empty cache
log-cache-persist-truncate-clear-failed = persist: truncate on clear failed ({ $a1 })
log-cache-persist-compaction-failed-keeping = persist: compaction failed ({ $a1 }), keeping original file
log-cache-persist-compaction-complete-records = persist: compaction complete, { $a1 } records
log-cache-persist-append-failed-this = persist: append failed ({ $a1 }), this insert stays in-memory only
log-cache-persist-file-lock-poisoned = persist: file lock poisoned, skipping this flush: { $a1 }
log-cache-rejecting-non-finite-vector = rejecting non-finite vector put for key prefix { $a1 }
log-cache-persist-failed-write-header = persist: { $a1 } failed to write header: { $a2 }
log-engine-auto-pooling-unrecognized-model = Auto pooling: unrecognized model family '{ $a1 }', falling back to CLS. Set pooling_mode explicitly to silence this warning.
log-engine-loaded-safetensors-model-weights = Loaded safetensors model weights successfully
log-engine-loading-safetensors-model-weights = Loading safetensors model weights from: { $a1 }
log-engine-optimum-cli-export-onnx = optimum-cli export onnx --model { $a1 } --task feature-extraction ./model_safetensors
log-engine-pip-install-optimum = pip install optimum
log-engine-please-convert-model-safetensors = Please convert the model to safetensors format:
log-engine-this-often-due-large = This is often due to large model files or incompatible formats.
log-engine-failed-load-pytorch-weights = Failed to load PyTorch weights: { $a1 }
log-engine-pytorch-weights-loaded-successfully = PyTorch weights loaded successfully
log-engine-then-use-model-converted = Then use './model_converted' as the model_path in config/config.toml
log-engine-python-c-from-transformers = python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{ $a1 }').save_pretrained('./model_converted')\"
log-engine-consider-converting-safetensors-format = Consider converting to safetensors format for better performance:
log-engine-pytorch-model-file-large = PyTorch model file is large ({ $a1 } GB). Large pyTorch files may have loading issues.
log-engine-loading-pytorch-model-weights = Loading PyTorch model weights from: { $a1 }
log-engine-model-file-sha256-verification = Model file SHA256 verification passed
log-engine-verifying-model-file-sha256 = Verifying model file SHA256 hash...
log-engine-model-file-integrity-check = Model file integrity check passed
log-engine-post-recovery-integrity-check = Post-recovery integrity check passed
log-engine-re-running-integrity-check = Re-running integrity check after recovery...
log-engine-successfully-recovered-all-corrupted = Successfully recovered all corrupted files
log-engine-attempting-automatic-recovery-corrupted = Attempting automatic recovery of corrupted files...
log-engine-file-error = File: { $a1 }, Error: { $a2 }
log-engine-model-file-integrity-check-2 = Model file integrity check failed!
log-engine-checking-model-file-integrity = Checking model file integrity...
log-engine-model-safetensors-unavailable-falling = model.safetensors unavailable, falling back to pytorch_model.bin; pickle format carries code-execution risk — only load .bin models from trusted sources, prefer safetensors
log-engine-downloading-loading-model-from = Downloading/Loading model from HuggingFace Hub: { $a1 }
log-engine-loading-model-from-remote = Loading model from remote HuggingFace repo '{ $a1 }' without model_sha256 verification — recommend setting model_sha256 in config to prevent tampering and supply-chain attacks
log-engine-loading-model-from-local = Loading model from local path: { $a1 }
log-engine-using-fp32-precision = Using FP32 precision
log-engine-bf16-supported-non-cuda = BF16 not supported on non-CUDA devices, falling back to FP32
log-engine-using-bf16-precision-brain = Using BF16 precision (Brain Float 16)
log-engine-fp16-supported-non-cuda = FP16 not supported on non-CUDA devices, falling back to FP32
log-engine-using-fp16-precision = Using FP16 precision
log-engine-int8-quantization-yet-implemented = INT8 quantization is not yet implemented; running in FP32 mode. The use_quantization flag has no effect until INT8 support is added.
log-engine-using-cpu = Using CPU
log-engine-candle-engine-does-natively = Candle engine does not natively support AMD GPUs. AMD GPU support requires ROCm-enabled Candle build or ONNX Runtime. Falling back to CPU.
log-engine-using-metal-gpu = Using Metal GPU
log-engine-using-cuda-gpu = Using CUDA GPU
log-engine-successfully-fell-back-cpu = Successfully fell back to CPU
log-engine-fallback-cpu-supported-cpu = Fallback to CPU: { $a1 } not supported on CPU, using FP32
log-engine-fallback-detected-model-architecture = Fallback: Detected model architecture: { $a1 }
log-engine-attempting-fallback-from-gpu = Attempting fallback from GPU to CPU due to memory pressure
log-engine-token-id-exceeds-vocab = Token ID { $a1 } exceeds vocab_size { $a2 }, clamping
log-engine-memory-limit-critical-checking = Memory limit critical, checking memory pressure for fallback
log-engine-memory-limit-exceeded-attempting = Memory limit exceeded, attempting fallback to CPU
log-engine-memory-pressure-cleared-resetting = Memory pressure cleared, resetting fallback flag — GPU may be retried on next model load
log-engine-pooling-mode-model = Pooling mode: { $a1 } (model: { $a2 })
log-engine-detected-model-architecture = Detected model architecture: { $a1 }
log-engine-successfully-fell-back-cpu-2 = Successfully fell back to CPU for ONNX engine
log-engine-attempting-fallback-from-gpu-2 = Attempting fallback from GPU to CPU for ONNX engine
log-engine-memory-limit-critical-onnx = Memory limit critical for ONNX engine, checking memory pressure for fallback
log-engine-memory-limit-exceeded-onnx = Memory limit exceeded for ONNX engine, attempting fallback to CPU
log-engine-failed-create-tokio-runtime = Failed to create Tokio runtime for memory check: { $a1 }
log-engine-onnx-engine-initialized-hidden = ONNX Engine initialized: hidden_size={ $a1 }, max_input_length={ $a2 }, vocab_size={ $a3 }, precision={ $a4 }
log-engine-using-precision = Using { $a1 } precision
log-engine-fp16-supported-without-gpu = FP16 not supported without GPU acceleration, falling back to FP32
log-engine-using-fp16-precision-gpu = Using FP16 precision with GPU acceleration
log-engine-final-hidden-size-value = Final hidden_size value: { $a1 }
log-engine-using-hidden-size-from = Using hidden_size from configuration: { $a1 }
log-engine-loading-tokenizer = Loading tokenizer...
log-engine-amd-gpu-detected-but = AMD GPU detected but ROCm execution provider is not configured in this build. Using CPU execution provider.
log-engine-downloading-loading-onnx-model = Downloading/Loading ONNX model files...
log-engine-using-huggingface-hub-model = Using HuggingFace Hub for model: { $a1 }
log-engine-using-local-model-path = Using local model path: { $a1 }
log-engine-successfully-fell-back-cpu-3 = Successfully fell back to CPU for decision pipeline
log-engine-attempting-fallback-cpu-decision = Attempting fallback to CPU for decision pipeline
log-engine-decision-pipeline-initialized-bundle = Decision pipeline initialized: bundle={ $a1 }, model={ $a2 }, precision={ $a3 }, calibration_buckets={ $a4 }
log-engine-verifying-decision-model-file = Verifying decision model file SHA256 hash...
log-engine-msg-model = { $msg } (model={ $a1 })
log-engine-memory-gate-passed-model = memory gate passed: model={ $a1 } need={ $need } available={ $available } floor={ $floor }
log-engine-session-memory-floor-mb = SESSION_MEMORY_FLOOR_MB={ $v } is not a valid integer; falling back to default { $DEFAULT_MEMORY_FLOOR_MB }MB
log-engine-cuda-execution-provider-requires = CUDA execution provider requires the transcribe feature (bundles ort/cuda); using CPU execution provider
log-engine-attempting-configure-cuda-execution = Attempting to configure CUDA execution provider
log-engine-gguf-pooling-inferred-as = GGUF pooling inferred as { $pooling } from identity '{ $pool_identity }'
log-engine-gguf-config-json-provides = GGUF config.json provides no recognizable hidden_act, falling back to Gelu (making the previously hardcoded behavior explicit)
log-engine-gguf-config-json-unknown = GGUF config.json unknown hidden_act '{ $other }', falling back to Gelu
log-engine-gguf-hidden-act-act = GGUF hidden_act '{ $act }' has no candle BERT counterpart, falling back to Gelu
log-engine-gguf-tensor-other-this = GGUF tensor '{ $other }': this repo maps it to final-LN (output.LayerNorm), which differs from llama.cpp's attention-LN convention; for externally converted GGUF, rename to output_norm
log-device-amd-gpu-initialization-complete = AMD GPU initialization complete. Found { $a1 } device(s) (ROCm: { $a2 }, OpenCL: { $a3 })
log-cuda-free-failed = CUDA free FAILED for device { $device } ({ $bytes } bytes): { $e } — accounting kept
log-cuda-free-on-drop-failed = CUDA free on drop FAILED for device { $device } ({ $bytes } bytes): { $e } — accounting kept
log-device-found-opencl-compatible-amd = Found OpenCL-compatible AMD GPU: { $a1 } with { $a2 } bytes VRAM
log-device-found-rocm-compatible-amd = Found ROCm-compatible AMD GPU: { $a1 } with { $a2 } bytes VRAM
log-device-initializing-amd-gpu-device = Initializing AMD GPU device manager...
log-device-rocm-version = ROCm version: { $a1 }
log-device-rocm-vram-query-failed = ROCm VRAM query failed, using fallback default: { $a1 }GB
log-device-rocm-vram-rocm-smi = ROCm VRAM (rocm-smi, device { $a1 }): { $a2 } MB
log-device-amd-driver-version-rocm = AMD driver version (rocm-smi): { $a1 }
log-device-amd-driver-version-sysfs = AMD driver version (sysfs): { $a1 }
log-device-gpu-memory-allocation-failed = GPU memory allocation failed: requested { $a1 } bytes, available { $a2 } bytes
log-device-amd-gpu-opencl-using = AMD GPU (OpenCL) using fallback default parameters ({ $a1 }GB, compute { $a2 }.{ $a3 }). OpenCL cannot query real VRAM; use the ROCm path for accurate info
log-device-primary-cuda-device-set = Primary CUDA device set to: { $a1 }
log-device-detected-compatible-cuda-device = Detected compatible CUDA device: { $a1 }
log-device-failed-initialize-cuda-device = Failed to initialize CUDA device via candle-core: { $a1 }
log-device-detected-cuda-device-via = Detected CUDA device via candle-core: { $a1 } ({ $a2 } MB, compute { $a3 }.{ $a4 } )
log-device-nvidia-smi-query-failed = nvidia-smi query failed, using fallback default device parameters ({ $a1 }GB, compute { $a2 }.{ $a3 }). Install nvidia-smi for accurate info
log-device-cuda-device-manager-initialized = CUDA device manager initialized with { $a1 } device(s)
log-device-created-memory-manager-device = Created memory manager for device { $a1 }: { $a2 } MB
log-device-cuda-detection-returned-error = CUDA detection returned error, treating as no available devices: { $a1 }
log-device-gpu-compute-mode = GPU compute mode: { $a1 }
log-device-gpu-compute-mode-default = GPU compute mode: Default (shared by multiple processes)
log-device-gpu-compute-mode-exclusive = GPU compute mode: Exclusive (optimal)
log-device-gpu-ecc-memory-disabled = GPU ECC memory: disabled (current: { $a1 })
log-device-gpu-ecc-memory-enabled = GPU ECC memory: enabled
log-device-cannot-read-gpu-current = cannot read GPU current clock frequency, cannot determine whether it is locked
log-device-gpu-current-clock-frequency = GPU current clock frequency: { $a1 }
log-device-transparent-hugepages-disabled = transparent hugepages: disabled
log-device-transparent-hugepages-madvise-mode = transparent hugepages: madvise mode (recommended for GPU)
log-device-transparent-hugepages-enabled-always = transparent hugepages: enabled (always)
log-device-gpu-persistence-mode-disabled = GPU persistence mode: disabled (current: { $a1 })
log-device-gpu-persistence-mode-enabled = GPU persistence mode: enabled
log-device-gpu-tuning-check-complete = GPU tuning check complete, found { $a1 } optimizable settings:
log-device-gpu-tuning-check-complete-2 = GPU tuning check complete: all settings optimized
log-device-auto-fallback-disabled-skipping = Auto-fallback disabled, skipping fallback to CPU
log-device-memory-pressure-detected-falling = Memory pressure detected, falling back to CPU
log-device-available-memory-query-failed = available_memory query failed for device { $a1 }, reporting 0 available
log-device-available-devices-using-cpu = No available devices, using CPU
log-device-available-gpu-falling-back = No available GPU, falling back to CPU
log-device-refresh-devices-amd-device = refresh_devices: AMD device count mismatch (internal: { $a1 }, amd_manager: { $a2 }). Some devices may not be updated.
log-device-failed-initialize-amd-device = Failed to initialize AMD device manager: { $a1 }
log-device-cuda-device-initialized = CUDA device initialized: { $a1 }
log-device-failed-initialize-cuda-device-2 = Failed to initialize CUDA device manager: { $a1 }
log-device-memory-limit-exceeded-triggering = Memory limit exceeded, triggering fallback to CPU
log-device-memory-limit-set-bytes = Memory limit set to: { $a1 } bytes
log-device-failed-create-tokio-runtime = Failed to create Tokio runtime: { $a1 }
log-device-memory-usage-critical-bytes = Memory usage critical: { $a1 } bytes ({ $a2 }%)
log-device-memory-usage-exceeded-limit = Memory usage exceeded limit: { $a1 } bytes (limit: { $a2 } bytes)
log-device-attempted-deallocate-bytes-but = Attempted to deallocate { $a1 } bytes but only { $a2 } bytes allocated
log-device-decreasing-batch-size-latency = Decreasing batch size to { $a1 } (latency: { $a2 }ms, memory: { $a3 }%)
log-device-increasing-batch-size-latency = Increasing batch size to { $a1 } (latency: { $a2 }ms, memory: { $a3 }%)
log-device-insufficient-gpu-memory-model = Insufficient GPU memory for model '{ $a1 }', using minimum batch size { $a2 }
log-device-registered-memory-requirements-model = Registered memory requirements for model: { $a1 }
log-api-logout-token-revocation-failed = Logout token revocation failed (best-effort): { $a1 }
log-api-failed-revoke-old-refresh = Failed to revoke old refresh token during refresh for login_id '{ $a1 }': { $a2 }
log-api-login-rejected-auth-enabled = Login rejected: auth is enabled but no admin password is configured (missing VECBOOST_ADMIN_PASSWORD). This should have failed at startup.
log-api-switch-path-validation-rejected = switch path validation rejected { $field }={ $path }: { $e }
log-api-switch-model-failed-e = switch_model failed: { $e }
log-api-batch-rerank-individual-query = Batch rerank: individual query failed, skipping: { $a1 }
log-auth-auth-endpoint-rate-limit = Auth endpoint rate limit exceeded for IP: { $a1 }
log-auth-ratelimitmodule-registered = RateLimitModule not registered: { $a1 }
log-auth-ipwhitelistmodule-registered = IpWhitelistModule not registered: { $a1 }
log-auth-ratelimitenabled-registered-rate-limiting = RateLimitEnabled not registered, rate limiting is disabled. This may leave endpoints unprotected in production.
log-auth-trusted-proxies-empty-x = trusted_proxies is empty — X-Forwarded-For is ignored; the direct peer IP is used for rate limiting and audit. Behind a reverse proxy, configure trusted_proxies (e.g. [\"10.0.0.0/8\"]) to honor forwarded headers.
log-config-encryption-key-env-must = { $ENCRYPTION_KEY_ENV } must be exactly 32 bytes for XChaCha20-Poly1305, got { $a1 } bytes — encryption disabled
log-config-encryption-key-env-set = { $ENCRYPTION_KEY_ENV } not set — sensitive config fields stored as plaintext. Production deployments MUST set this to a 32-byte key (e.g. `openssl rand -hex 32`) and enable VECBOOST_REQUIRE_ENCRYPTION=1.
log-domain-queue-cleared-requests-discarded = Queue cleared, { $a1 } requests discarded
log-domain-shutdown-drain-queued-requests = Shutdown drain: { $a1 } queued requests evacuated
log-domain-reaped-expired-requests-from = Reaped { $a1 } expired requests from queue, queue_size={ $a2 }
log-model-reloading-onnx-model-from = Reloading ONNX model from: { $a1 }
log-model-reloading-candle-model-from = Reloading Candle model from: { $a1 }
log-model-model-integrity-check-failed = Model integrity check failed, initiating recovery for { $a1 } corrupted files
log-model-model-integrity-check-passed = Model integrity check passed, no recovery needed
log-model-all-recovered-files-verified = All recovered files verified successfully
log-model-download-attempt-failed-file = Download attempt { $a1 } failed for file { $a2 }: { $a3 }
log-model-download-attempt-file = Download attempt { $a1 } for file: { $a2 }
log-model-recovery-attempt-failed-file = Recovery attempt { $a1 } failed for file { $a2 }: { $a3 }
log-model-recovery-attempt-file = Recovery attempt { $a1 } for file: { $a2 }
log-model-local-model-path-detected = Local model path detected but no repo_id provided for recovery of { $a1 }
log-model-backed-up-corrupted-file = Backed up corrupted file to: { $a1 }
log-model-verification-after-recovery-failed = Verification after recovery failed: { $a1 }
log-model-verifying-recovered-files = Verifying recovered files...
log-model-failed-recover-file = Failed to recover file: { $a1 }
log-model-successfully-recovered-file = Successfully recovered file: { $a1 }
log-model-attempting-recover-corrupted-file = Attempting to recover corrupted file: { $a1 }
log-model-starting-recovery-model-corrupted = Starting recovery for model '{ $a1 }', { $a2 } corrupted files detected
log-monitor-failed-get-memory-info = Failed to get memory info: { $a1 }
log-security-failed-read-key-file = Failed to read key file metadata for permission hardening: { $a1 }
log-security-failed-set-restrictive-permissions = Failed to set restrictive permissions on key file: { $a1 }
log-service-failed-fallback-cpu = Failed to fallback to CPU: { $a1 }
log-service-successfully-fell-back-cpu = Successfully fell back to CPU, retrying operation
log-service-fallback-already-triggered-cannot = Fallback already triggered, cannot retry
log-service-oom-error-detected-attempting = OOM error detected: { $a1 }. Attempting fallback to CPU (attempt { $a2 }/{ $a3 })
log-service-decision-inference-task-failed = decision inference task failed: { $e }
log-service-estimate-model-bytes-cannot = estimate_model_bytes: cannot read model directory { $a1 } ({ $e }), accounting 0 bytes
log-service-memory-limit-accounting-initial = memory limit accounting (initial): model { $a1 } ≈ { $est } bytes
log-service-local-model-config-cleared = Local model config cleared for { $a1 }
log-service-model-unloaded-via-modelmanager = Model { $a1 } unloaded via ModelManager
log-service-memory-limit-accounting-model = memory limit accounting: model { $a1 } ≈ { $new_est } bytes
log-service-unloading-previous-model = Unloading previous model: { $a1 }
log-service-model-switched-successfully = Model switched successfully to { $a1 }
log-service-switching-model-from = Switching model from { $a1 } to { $a2 }
log-service-successfully-fell-back-cpu-2 = Successfully fell back to CPU, retrying batch chunk { $a1 }
log-service-oom-error-detected-batch = OOM error detected in batch chunk { $a1 }: { $a2 }. Attempting fallback to CPU (attempt { $a3 }/{ $a4 })
log-service-normalize-l2-failed-truncated = normalize_l2 failed for truncated embedding: { $a1 }
log-service-processed-paragraphs-2-ms = Processed { $a1 } paragraphs in { $a2 }ms
log-service-processed-lines-2-ms = Processed { $a1 } lines in { $a2 }ms
log-service-file-streaming-failed = File streaming failed: { $a1 }
log-service-cache-warm-up-completed = Cache warm-up completed: { $a1 }/{ $a2 } entries preloaded
log-service-warm-up-batch-failed = Warm-up batch failed, skipping { $a1 } texts: { $a2 }
log-service-warm-up-skip-non = warm-up: skip non-finite embedding for '{ $a1 }...'
log-service-dimension-mismatch-expected-got = Dimension mismatch: expected { $a1 }, got { $a2 }. Model '{ $a3 }' may have been configured incorrectly or the wrong model was loaded.
log-service-embedding-cache-wal-replay = Embedding cache WAL replay from { $a1 }
log-service-model-heat-save-failed = model heat save failed: { $e }
log-service-transcribe-segments-chars-ms = transcribe: { $a1 } segments, { $a2 } chars, { $a3 } ms
log-service-translation-inference-task-failed = translation inference task failed: { $e }
log-worker-inference-panicked = Worker { $worker_id } inference panicked: { $e }
log-util-list-tree-failed-repo = list_tree failed for '{ $repo_id }' ({ $e }); size-verified short-circuit unavailable, re-downloading all manifest files
log-util-hf-endpoint-detected-hf = HF_ENDPOINT={ $a1 } detected — hf-hub 1.0.0 requires ETag headers which some mirrors (e.g. hf-mirror.com) may not provide. If model download fails with 'missing ETag header', pre-download the model manually and set `model_path` in config to use local loading instead.
log-util-vquant-rotate-tail-not-hadamard = rotate: trailing { $dims } dims are not Hadamard-rotated (quantization quality degraded for this segment)
log-util-vquant-quantize-i8-non-finite = quantize_i8: non-finite input vector (len={ $len }), zeroed
cli-poc-transcribe-own-e = transcribe-own: { $e }
cli-poc-transcribe-own-pipeline-done = transcribe-own: pipeline done in { $wall_s }s -> { $a1 }
cli-poc-transcribe-own-vad-only = transcribe-own: vad-only done in { $wall_s }s -> { $a1 }
cli-poc-transcribe-own-3-s = transcribe-own: [{ $a1 }s] worker { $w } segment { $a2 }/{ $a3 } [{ $a4 }s-{ $a5 }s] { $a6 } chars
cli-poc-transcribe-own-3-s-2 = transcribe-own: [{ $a1 }s] vad done: { $a2 } segments
cli-poc-transcribe-own-wav-loaded = transcribe-own: wav loaded: { $a1 } samples, { $audio_seconds }s
cli-poc-transcribe-own-name-path = transcribe-own: { $name } path carries annotation text, normalized: { $a1 } -> { $a2 }
cli-poc-transcribe-own-backend-py = transcribe-own: backend=py-torch (embedded CPython, GPU 1.7B bf16)
cli-poc-transcribe-own-key-retired = transcribe-own: { $key } retired (experiment path removed; conclusions recorded in out/transcribe-poc/own-engine/summary.md) — ignored
cli-poc-transcribe-own-decoding-segments = transcribe-own: decoding { $segments_len } segments with { $workers } workers (ort-intra={ $intra }); loading { $workers } session sets...

# ── CLI --help surface (sdforge protocol registry; en mirrors the #[forge(description)] literal) ──
forge-cli-compute-similarity = Compute cosine similarity between two texts
forge-cli-decisions = Answer choice/score/noul questions about a decision state
forge-cli-detect = Detect the natural language of a text (ISO 639-1 code with 639-3 fallback)
forge-cli-embed = Generate embedding vector for input text
forge-cli-embed-batch = Generate embedding vectors for multiple texts in batch
forge-cli-rerank = Rerank documents by relevance to a query
forge-cli-search = Rank candidate texts by similarity to the query (1-to-N search)
forge-cli-transcribe = Transcribe a 16kHz mono WAV file to text (Qwen3-ASR; requires [transcribe] enabled)
forge-cli-translate = Translate text between configured language pairs (seq2seq)

# ── Warmup diagnostics (main.rs warmup paths) ──
warmup-decision-pipeline-failed = warmup: decision pipeline full-chain warmup failed: { $e }
warmup-mode-decide = decide×1 (decision pipeline full-chain)
warmup-mode-embed = embed
warmup-incomplete = warmup: incomplete (mode={ $mode }, { $failures } failures, { $elapsed }) — first request may be slower
warmup-complete = warmup: warmup inference complete (mode={ $mode }, { $elapsed })

# ── Candle native decision head: error returns + load-time diagnostics ──
engine-candle-decision-checkpoint-not-found = Laya PyTorch checkpoint not found: { $path } (acquisition steps are in docs/USER_GUIDE.md, "Candle native decision path"; if you already have a receptron/laya-onnx ONNX bundle, set engine_type = "onnx" to reuse the existing assets through the ort pipeline)
engine-candle-decision-encoder-config-read = failed to read encoder config { $path }: { $detail } (encoder/config.json of the official convaiinnovations/laya repo; acquisition steps are in docs/USER_GUIDE.md)
log-engine-verifying-candle-decision-checkpoint-sha256 = Verifying candle decision checkpoint SHA256 hash...
log-engine-candle-decision-engine-initialized = Candle decision engine initialized: bundle={ $a1 }, hidden={ $a2 }, layers={ $a3 }, calibration_buckets={ $a4 }
log-engine-candle-decision-engine-using-cuda-gpu = Candle decision engine using CUDA GPU
log-engine-candle-decision-engine-using-metal-gpu = Candle decision engine using Metal GPU
log-engine-candle-decision-amd-fallback-cpu = Candle engine does not natively support AMD GPUs; candle decision engine falling back to CPU

# ── ONNX translate engine: decode paths + load-time diagnostics ──
engine-translate-logits-nonempty = logits must not be empty
log-engine-tokenizer-shim-in-memory = tokenizer { $a1 }: { $a2 } (in-memory; tokenizers 0.23 parse contract)
log-engine-translate-generation-config-unreadable = translation bundle { $a1 }: generation_config.json unreadable ({ $e }); falling back to tokenizer special tokens for decode anchors
log-engine-translate-generation-config-missing-ids = translation bundle { $a1 }: generation_config.json missing decoder_start_token_id/eos_token_id; falling back to tokenizer special tokens for decode anchors
log-engine-translate-parallel-workers-lazy = translation parallel workers={ $a1 } (lazy; extra session sets build on first long-parallel request)
log-engine-translation-engine-initialized = Translation engine initialized: bundle={ $a1 }, encoder={ $a2 }, decoder={ $a3 }, precision={ $a4 }, max_source_tokens={ $a5 }, max_new_tokens={ $a6 }, decoder_start_token_id={ $a7 }, eos_token_id={ $a8 }, past_tensors={ $a9 }
log-engine-translate-whole-text-decode-degenerated = translation whole-text decode degenerated ({ $a1 } tokens generated, { $a2 } sentences); retrying per-sentence with output budget { $a3 }
log-engine-translate-rescue-sentence-count-exceeds = rescue sentence count { $a1 } exceeds cap { $max }; remaining sentences dropped
log-engine-translate-rescue-budget-exhausted = rescue output budget exhausted after { $a1 } tokens; remaining sentences dropped
log-engine-translate-parallel-session-lazy-loaded = translation parallel worker session set lazy-loaded ({ $a1 }/{ $a2 } extra sets ready)
log-engine-translate-parallel-sentence-count-exceeds = parallel sentence count { $a1 } exceeds cap { $max }; remaining sentences dropped
log-engine-translate-protected-token-not-recovered = protected token not recovered in translation (placeholder { $a1 }): { $a2 }
log-engine-translate-attempting-cpu-fallback = Attempting fallback to CPU for translation engine
log-engine-translate-successfully-fell-back-cpu = Successfully fell back to CPU for translation engine

# ── Calibration tables (decision pipeline) + checkpoint inventory (main.rs) ──
log-engine-calibration-file-missing-fallback-temperature = calibration file missing at { $a1 }: probabilities will be overconfident (uncalibrated ECE 0.466 vs 0.081 calibrated, model card); falling back to temperature { $temp } for all cardinalities
log-engine-calibration-no-temperature-by-options = { $a1 } has no temperature_by_options field; falling back to temperature { $temp } for all cardinalities
log-main-model-checkpoints-resolved = model checkpoints resolved: { $a1 } total (builtin laya-multilingual + { $a2 } user-configured, { $a3 } overriding the builtin)

# ── transcribe-own CLI: usage / argument / env knob errors (stderr) ──
cli-poc-transcribe-own-usage = usage: transcribe-own <wav> <outDir> [--models <dir>] [--vad-only]
cli-poc-transcribe-own-models-requires-value = --models requires a value
cli-poc-transcribe-own-unknown-argument = unknown argument: { $a1 }
cli-poc-transcribe-own-env-not-valid-integer = { $key }={ $value } is not a valid integer
cli-poc-transcribe-own-env-out-of-range = { $key }={ $value } out of range [{ $min },{ $max }]
cli-poc-transcribe-own-ep-invalid = TRANSCRIBE_EP={ $a1 } invalid (cpu|cuda)
