# VecBoost 未国际化字段审计报告

> 日期：2026-09-05
> 范围：`src/`（Rust 代码直返用户/客户端的字符串）+ `src/main.rs` 启动错误 + `src/api` forge 描述 + `config/` 默认配置
> 方法：3 路 subagent 并行扫描（错误与校验文案 / HTTP 响应文案 / 日志 CLI 文案，其中 HTTP 路取消后由人工用 `rg` 补查 `src/api/*.rs` 验证）+ 人工抽查取证
> 结论：**是，仍存在大量未国际化字段**。FTL 基线仅覆盖约 43 键，代码中直返用户的硬编码远多于此，且存在“有键但未使用”的漏网。
> 本文按 P0（直达客户端，必须修）→ P1（启动/运维可见，应该修）→ P2（文档/显示层）→ 明确不修清单排序，每项给出 `文件:行号 + 原文 + 状态 + 建议键名`，可按文档逐项处理。

---

## 0. 基线：现有 FTL 已覆盖（中英对齐，无需动）

### 0.1 `src/i18n/locales/en/errors.ftl` / `zh/errors.ftl`（17 键）

| 键 | en | zh |
|---|---|---|
| `error-config` | Config error: `{ $detail }` | 配置错误：`{ $detail }` |
| `error-model-load` | Model load error: `{ $detail }` | 模型加载错误：`{ $detail }` |
| `error-model-corrupted` | Model file corrupted: `{ $detail }` | 模型文件已损坏：`{ $detail }` |
| `error-model-integrity` | Model file integrity check failed: `{ $detail }` | 模型文件完整性校验失败：`{ $detail }` |
| `error-tokenization` | Tokenization error: `{ $detail }` | 分词错误：`{ $detail }` |
| `error-inference` | Inference error: `{ $detail }` | 推理错误：`{ $detail }` |
| `error-oom` | Out of memory error: `{ $detail }` | 内存不足错误：`{ $detail }` |
| `error-invalid-input` | Invalid input: `{ $detail }` | 输入无效：`{ $detail }` |
| `error-not-found` | Not found: `{ $detail }` | 未找到：`{ $detail }` |
| `error-model-not-loaded` | Model not loaded: `{ $detail }` | 模型未加载：`{ $detail }` |
| `error-authentication` | Authentication error: `{ $detail }` | 认证错误：`{ $detail }` |
| `error-security` | Security error: `{ $detail }` | 安全错误：`{ $detail }` |
| `error-io` | IO error: `{ $detail }` | IO 错误：`{ $detail }` |
| `error-validation` | Validation error: `{ $detail }` | 校验错误：`{ $detail }` |
| `error-rate-limit` | Rate limit exceeded: `{ $detail }` | 超出速率限制：`{ $detail }` |
| `error-database` | Database error: `{ $detail }` | 数据库错误：`{ $detail }` |
| `error-internal` | Internal error: `{ $detail }` | 内部错误：`{ $detail }` |

HTTP 通道已接通：`src/error.rs:249-286 IntoResponse` 用 `tr_with_args(error_code, detail)` 翻译，前缀已国际化（`detail` 本身第三方原文保留是预期的）。

### 0.2 `src/i18n/locales/en/messages.ftl` / `zh/messages.ftl`（26 键）

`validate-text-length` / `validate-batch-size` / `validate-path-failed` / `sensitive-dir-refused` / `openai-input-empty` / `openai-input-too-large` / `auth-password-empty` / `auth-invalid-credentials` / `auth-refresh-token-empty` / `auth-invalid-token` / `logout-success` / `model-switch-success` / `model-already-current` / `model-load-failed` / `file-empty` / `file-no-paragraphs` / `file-invalid-encoding` / `oom-no-fallback` / `health-ok` / `health-check-failed` / `rerank-empty-docs` / `rerank-too-many-docs` / `rerank-invalid-top-k` / `rerank-unsupported` / `rerank-query-too-long` / `dir-get-cwd-failed`。中英一一对应（行号 `en/messages.ftl:4-45`）。

### 0.3 正确范例（修复时仿此模式，不要自创）

`src/service/rerank.rs:99-144` 是全仓标杆：

```rust
// 带参数：
return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
    "rerank-query-too-long",
    crate::i18n::tr_args(&[
        ("length", &req.query.len().to_string()),
        ("max", &max_query_length.to_string()),
    ]),
)));
// 不带参数：
return Err(VecboostError::InvalidInput(
    crate::i18n::tr("rerank-empty-docs").to_string(),
));
```

同类正确点：`src/service/embedding.rs:551 tr("file-empty")`、`1104/1188 tr("model-already-current"/"model-switch-success")`、`1174 tr("model-load-failed")`、`src/api/embedding.rs:144,164,220,235,326,429 tr("validate-*"/"dir-get-cwd-failed"/"sensitive-dir-refused"/"health-check-failed")`、`633-646 tr("openai-input-*")`、`src/api/auth.rs:63,80,142,159,224 tr("auth-*"/"logout-success")`、`src/service/embedding.rs:911 tr("oom-no-fallback")`。

---

## 1. P0 — 直达客户端，必须首批修（400/401/429/500 JSON body）

### 1.1 `src/auth/types.rs` — 全仓仅有的中文泄漏（英文用户会看到中文）

`src/auth/types.rs:41-69`，经 `src/api/auth.rs:54-58 validate_username_format → ApiError::InvalidInput{field: username}` 原样直返 HTTP JSON：

| 文件:行 | 原文 | FTL | 建议键 |
|---|---|---|---|
| `src/auth/types.rs:44` | `"用户名长度必须在 3 到 32 个字符之间"` | 无 | `auth-username-length` |
| `src/auth/types.rs:55` | `"用户名必须以字母开头"` | 无 | `auth-username-start` |
| `src/auth/types.rs:64` | `"用户名只能包含字母、数字、下划线和连字符"` | 无 | `auth-username-charset` |

```rust
// 现状（3 处同构）：
return Err(VecboostError::ValidationError(
    "用户名长度必须在 3 到 32 个字符之间".to_string(),
));
```

### 1.2 `src/utils/validator/input.rs` — 约 20 处英文直返 `InvalidInput`

经 `to_api_error(): src/api/embedding.rs:78-84` 原样进 `ApiError::InvalidInput{message}` → HTTP JSON。注意同文件已有 `validate-text-length/validate-batch-size/openai-*` 走 FTL，前后不一致。

| 文件:行 | 原文 | FTL 状态 | 建议 |
|---|---|---|---|
| `input.rs:104` | `"Text cannot be empty"` | 无（语义近 `openai-input-empty` 但通道不同） | 新增 `validate-text-empty` |
| `input.rs:111` | `"Text too short: {} characters (minimum: {})"` | 无 | 新增 `validate-text-too-short`（`{ $got }/{ $min }`） |
| `input.rs:118` | `"Text too long: {} characters (maximum: {})"` | 部分：有 `validate-text-length`（按 `index/max/got`），此处无 `index` 且文案不同 | 复用 `validate-text-length` 或新增 `validate-text-too-long` |
| `input.rs:126` | `"Text contains only whitespace"` | 无 | 新增 `validate-text-whitespace` |
| `input.rs:142` | `"Batch cannot be empty"` | 无（近似 `rerank-empty-docs/openai-input-empty`） | 新增 `validate-batch-empty` |
| `input.rs:148` | `"Batch size {} exceeds maximum {}"` | 有 `validate-batch-size`（`Batch size {size} exceeds max {max} (config…)`），措辞/参数名不一致 | **复用 `validate-batch-size`**（改调用处传参） |
| `input.rs:157` | `"Validation failed for text at index {}: {}"` | 无（包装器） | 新增 `validate-text-index-failed`（`{ $index }/{ $detail }`，`detail` 保留内层原文） |
| `input.rs:176` | `"Search texts list cannot be empty"` | 有 `rerank-empty-docs` 近义，未复用 | 复用 `rerank-empty-docs` 或新增 `validate-search-empty` |
| `input.rs:182` | `"Search results count {} exceeds maximum {}"` | 有 `rerank-too-many-docs`，漏网 | **复用 `rerank-too-many-docs`** |
| `input.rs:191` | `"top_k must be at least 1"` | 有 `rerank-invalid-top-k`（`top_k must be greater than 0…`），文案分裂 | 统一到 `rerank-invalid-top-k` |
| `input.rs:196` | `"top_k {} exceeds maximum {}"` | 无上界键 | 新增 `validate-top-k-exceeded`（`{ $got }/{ $max }`） |
| `input.rs:206` | `"Validation failed for search text at index {}: {}"` | 无 | 复用 `validate-text-index-failed` |
| `input.rs:227` | `"File size {:.2} MB exceeds maximum allowed size {:.2} MB"` | 仅有 `file-empty/no-paragraphs/invalid-encoding` | 新增 `file-too-large`（`{ $size }/{ $max }`） |
| `input.rs:234` | `"Cannot access file {}: {}"` | 无 | 新增 `file-access-failed`（`{ $path }/{ $detail }`） |
| `input.rs:244` | `"File has no extension"` | 无 | 新增 `file-no-extension` |
| `input.rs:249` | `"File extension '.{}' is not allowed. Allowed extensions: {:?}"` | 无 | 新增 `file-extension-not-allowed`（`{ $ext }/{ $allowed }`；注意 `{:?}` Rust Debug 格式需重写为本地化列表，不要直译 `["txt", …]`） |
| `input.rs:261` | `"Cannot open file: {}"` | 无 | 新增 `file-open-failed`（`{ $detail }`） |
| `input.rs:268` | `"Cannot read file: {}"` | 无 | 新增 `file-read-failed`（`{ $detail }`） |
| `input.rs:299` | `"File contains non-text binary data"` | 无 | 新增 `file-binary-rejected` |

### 1.3 `src/utils/validator/path.rs` — 安全高敏路径（经 `validate-path-failed` 的 `detail` 透出）

| 文件:行 | 原文 | 建议键 |
|---|---|---|
| `path.rs:70` | `"Path traversal attempt detected: {}"` | `path-traversal-detected`（`{ $detail }`） |
| `path.rs:80` | `format!("Invalid path: {}", e)` | `path-invalid`（`{ $detail }`） |
| `path.rs:85` | `"No allowed root directories configured for file access"` | `path-no-roots` |
| `path.rs:96` | `"Access denied: path '{}' is not within allowed directories. Allowed roots: {:?}"` | `path-access-denied`（`{ $path }/{ $detail }`） |
| `path.rs:114` | `"Path is not a file: {}"` | `path-not-file`（`{ $path }`） |
| `path.rs:128` | `"Path is not a directory: {}"` | `path-not-dir`（`{ $path }`） |

### 1.4 流水线 / 调度（500/429 直达客户端）

| 文件:行 | 原文 | FTL | 建议键 |
|---|---|---|---|
| `src/pipeline/handler.rs:74` | `"Response channel error"`（500） | 无 | `pipeline-channel-error` |
| `src/pipeline/handler.rs:77` | `"Request timeout"`（语义 408 却用 `ValidationError`→400，顺手确认状态码） | 无 | `pipeline-timeout` |
| `src/pipeline/queue.rs:32` | `"Expected Embed request but got Rerank"`（内部断言，可低优） | 无 | `queue-type-mismatch` |
| `src/pipeline/queue.rs:89` | `"Queue is full, request rejected"`（429，用户高频可见；现有仅 `error-rate-limit` 前缀） | 无正文键 | `queue-full-rejected` |
| `src/pipeline/scheduler.rs:72` | `"Rerank service not configured"` | 有 `rerank-unsupported` 近义但未复用 | 复用 `rerank-unsupported` 或新增 `rerank-not-configured` |
| `src/api/init.rs:21,28` | `"init_state already called" / "init_state not called"`（经 `to_api_error → ApiError::Internal`） | 无 | `api-init-state-called / api-init-state-missing`（或一个带参 `api-init-state`） |

### 1.5 `src/api/auth.rs` 鉴权路径英文直返 / 透传

| 文件:行 | 原文 | 说明 | 建议键 |
|---|---|---|---|
| `src/api/auth.rs:46,133,208` | `kit_internal_error("auth disabled at runtime")` | 3 处重复 | `auth-disabled` |
| `src/api/auth.rs:73` | `kit_internal_error("password verification failed")` | 鉴权失败路径；`log::error!` 那行是日志可不修，`kit_internal_error` 这行必须修 | `auth-verify-failed` |
| `src/api/auth.rs:54-55` | `message: e.to_string()` | 把 1.1 的中文 `Display` 透传为 `message`；修 1.1 时此处自动解决，但要确认没有其他 `e.to_string()` 透传非 i18n 文案 | —（随 1.1 修） |
| `src/api/embedding.rs:102,123` | `message: other.to_string()` / `message: e.to_string()` | 把 `Display` 英文泄漏为 `message`；API 层绝不用 `to_string()` 做 `message`（见 §4.1） | —（改调用，不新增键） |

---

## 2. P1 — 启动阻断 / 运维可见（stderr / `/metrics` / OOM detail）

### 2.1 `src/service/common.rs` OOM（`log` 可不修，`Err` 必须修）

`src/service/common.rs:57-104`。`warn!` 4 行是运维日志可不修；4 个 `Err` 进 `error-oom` 的 `$detail`，用户可见必须修。且与已 i18n 的 `oom-no-fallback` / `service/embedding.rs:911 tr("oom-no-fallback")` 语义重复却各写一遍。

| 文件:行 | 原文 | 状态 | 建议 |
|---|---|---|---|
| `common.rs:67` | `"Out of memory and fallback already attempted"` | **有 `oom-no-fallback` 一字不差，漏网** | 改为 `tr("oom-no-fallback")`（一行即修） |
| `common.rs:89` | `"OOM error [{}] and fallback failed: {}"` | 无 | 新增 `oom-fallback-failed`（`{ $detail }` 透出 `e`） |
| `common.rs:97` | `"Out of memory and no fallback available"` | 无 | 新增 `oom-no-fallback-available` |
| `common.rs:102` | `"Max fallback attempts exceeded ({}). Last error: {}"` | 无 | 新增 `oom-max-attempts`（`{ $attempts }/{ $detail }`） |

同理 `src/service/embedding.rs:857 InferenceError("Failed to acquire semaphore: {}")` → `engine-semaphore-failed`；`:937-940 OutOfMemory("OOM error and fallback failed: {}")` → 复用 `oom-fallback-failed`；`:965-968 InferenceError("Batch chunk processing timed out after {}s")` → `engine-batch-timeout`（`{ $secs }`）。

### 2.2 `src/main.rs` — `anyhow!/bail!` 启动错误走 stderr，必须修

全部英文硬编码，无 `tr!`，进程非零退出，部署运维在终端直接看到（不是可选级别的 `log!`）。

| 文件:行 | 原文 | 建议键 |
|---|---|---|
| `main.rs:143` | `"Failed to create database pool: {}"` | `startup-db-pool`（`{ $detail }`） |
| `main.rs:146` | `"Failed to initialize database schema: {}"` | `startup-db-schema` |
| `main.rs:300-303` | `"JWT secret must be at least 32 characters long for security. Current length: {}"`（与 `config/app.rs:576` 重复，统一） | `startup-jwt-length`（`{ $got }`） |
| `main.rs:306-309` | `"JWT secret is required when authentication is enabled…"` | `startup-jwt-missing` |
| `main.rs:314,329,580` | `"Failed to create GarrisonDaoOxcache…/Failed to init GarrisonManager…/Failed to create PrometheusCollector…"` | `startup-dao / startup-garrison / startup-prometheus`（或一个带 `$module` 的 `startup-register-failed`，见下） |
| `main.rs:455,469,488` | `"Failed to initialize inklog logger…/Failed to load config via confers…/Encryption key validation failed: {}"` | `startup-logger / startup-config / startup-encryption` |
| `main.rs:919-924` | `"gRPC require_auth=true but BearerAuth creation failed: {}. Set VECBOOST_JWT_SECRET…"` | `startup-grpc-bearer-failed`（含操作指引，需双语） |
| `main.rs:928-932` | `"gRPC require_auth=true but auth.jwt_secret is None…"` | `startup-grpc-no-secret` |
| `main.rs:935-938` | `"gRPC require_auth=true but auth.enabled=false…"` | `startup-grpc-auth-disabled` |
| `main.rs:943-946` | `"gRPC require_auth=true but vecboost auth feature is not enabled…"` | `startup-grpc-no-feature` |
| `main.rs:219-266,602-640,657,678,697,710,821` | 约 17 处 `"Failed to register {Embedding,RateLimit,…}Module / Failed to build AsyncKit / Failed to require CsrfConfigModule…"` | 复用一个带 `$module` 参数的 `startup-register-failed`（P2 亦可，面向运维） |
| `main.rs:284,288` | `"No handler registered for CLI command: {}" / "CLI command '{}' failed: {:?}"`（CLI stderr 直接可见） | `cli-no-handler / cli-failed` |
| `main.rs:334` | `.expect("admin password hash must succeed")` | 不可达 panic，**不修** |

### 2.3 `src/config/app.rs` + `src/config/encryption.rs` — 启动校验阻断

| 文件:行 | 原文 | 建议键 |
|---|---|---|
| `config/app.rs:572` | `"VECBOOST_JWT_SECRET cannot be empty"` | `config-jwt-empty` |
| `config/app.rs:577` | `"VECBOOST_JWT_SECRET must be at least {} characters"` | `config-jwt-length`（与 `main.rs:300` 统一） |
| `config/app.rs:588` | `"VECBOOST_ADMIN_PASSWORD cannot be empty"` | `config-password-empty` |
| `config/app.rs:593` | `"…must be at least {} characters"`（password） | `config-password-length` |
| `config/encryption.rs:73-75` | `"{KEY} environment variable is not set. Production deployments MUST configure…"` | `config-encryption-missing` |
| `config/encryption.rs:79` | `"…must be exactly 32 bytes…"` | `config-encryption-length` |
| `config/encryption.rs:104-107,113-116` | `log::warn!("{KEY} not set…stored as plaintext…")` 非阻断 warn | **可不修**（日志面），但措辞与阻断文案统一 |

另 `src/config/app.rs:541,544,547` 独立 `enum ConfigError #[error("Configuration error/IO error/confers error: {0}")]` 从未走 `error_code + FTL`，建议复用 `error-config` 或新增 `config-load-failed`。

### 2.4 `src/metrics/endpoint.rs` — `/metrics` 纯文本 body 绕过 i18n 通道

`curl /metrics` 与 Prometheus 抓取直接可见：

| 文件:行 | 原文（状态码） | 建议 |
|---|---|---|
| `endpoint.rs:78` | `"Rate limiter unavailable"`（500） | 新增 `metrics-limiter-unavailable` |
| `endpoint.rs:94` | `"Rate limit exceeded"`（429，与 `error-rate-limit` 重复却未复用） | 复用 `error-rate-limit` 思想或新增 `metrics-rate-limited` |
| `endpoint.rs:106` | `"PrometheusCollector not configured"`（500） | 新增 `metrics-collector-missing` |
| `endpoint.rs:109` | `"Internal error"` | 复用 `error-internal` |
| `endpoint.rs:121` | `format!("Failed to encode metrics: {}", e)`（500） | 新增 `metrics-encode-failed`（`{ $detail }`） |
| `endpoint.rs:40,54,72,108` | `log::error!("…Module not registered/available…/Failed to build error response")` | 运维日志，**可不修** |

---

## 3. P1-续 — S2 类：`detail` 前缀英文（第三方 `e.to_string()` 保留原文，只翻前缀）

模式均为 `format!("<英文前缀>: {e}")`。仿 `validate-path-failed / dir-get-cwd-failed / model-load-failed` 的 `{ $detail }` 模式建族：

* `src/engine/candle_engine.rs`（约 83 处，抽样）：`:205 "Rejected model path due to path traversal…"、:229 "No model weights file found…"、:336 "Invalid configuration for architecture…"、:372 "Integrity check failed…"、:407 "Recovery failed…"、:459 "Failed to verify SHA256…"、:509 "Failed to load PyTorch weights…"、"XLM-RoBERTa config is required"、"Failed to get batch/token 0…" → `model-path-traversal-rejected / model-no-weights / model-invalid-arch / model-integrity-failed / model-recovery-failed / model-sha-mismatch / model-pytorch-load-failed` 等（有 `model-load-failed` 仅 `service/embedding.rs:1174` 用了，其余全硬编码）。
* `src/engine/onnx_engine.rs:62,72,85,158,162,397,425`：`"No ONNX model found…/Cannot determine HF cache path…/Tokenizer not found…/Failed to verify SHA256…/Empty sequence after mask/Failed to acquire fallback lock…"` → `model-*` + `engine-fallback-lock-failed`。
* `src/text/tokenizer.rs`（约 20 处）：`:258,319,365 "Cannot encode empty text / Cannot decode empty token ids / …at batch index {}"` → `tokenizer-empty-text / tokenizer-empty-ids / tokenizer-empty-batch`；`:264,373 "UTF-8 encoding validation failed at byte {}…"` → `tokenizer-utf8-failed`；`:205,234,279,327,337,392,479,486,530,538 "Failed to load tokenizer from model/file…/Failed to encode/decode…/Invalid token id…/has empty vocabulary/missing [UNK]"` → `tokenizer-load-model-failed / tokenizer-load-file-failed / tokenizer-encode-failed / tokenizer-decode-failed / tokenizer-invalid-id / tokenizer-batch-failed / tokenizer-json-read-failed / tokenizer-json-parse-failed / tokenizer-empty-vocab / tokenizer-missing-unk`；`:214,242 "max_length must be greater than 0, got {}"` → `tokenizer-invalid-max-length`。
* `src/utils/vector.rs:94,117,132,145 "Vector dimensions mismatch: {} vs {}"×4、:106 "cosine similarity is undefined for zero vectors"、:201 "cannot normalize near-zero vector…"` → `vector-dim-mismatch / vector-zero-undefined / vector-near-zero`。
* `src/utils/hash.rs:34,42,75,86 "Failed to open/read file…/Failed to get metadata…"` → `hash-io-failed`。
* `src/utils/hf_hub.rs:82,104 "Invalid HF repo ID…/HF hub initialization failed…"` → `model-invalid-repo-id / model-hf-init-failed`。
* `src/model/loader.rs:102,133 "Model not found at…"` → 有 `error-not-found` 前缀但 `detail` 键化为 `model-not-found-at`。
* `src/model/recovery.rs:131,143,183,197,229,243,285 "Failed to create backup dir/backup corrupted file…"` → `model-recovery-backup-failed` 等。
* `src/security/encrypted_store.rs`（约 27 处）：`:61,123,128,131,138,153,171,176,193,215,225,245,260,264 "Failed to check/open/read/create/write/flush key file / Invalid nonce/ciphertext / Decryption/Encryption failed / Failed to derive key / Invalid UTF-8/key data / Serialization failed"` → `security-key-file-failed / security-nonce-invalid / security-decrypt-failed…`；`src/security/mod.rs:67,72 + helpers.rs:20,26 + salt.rs:31,44 "Encryption key/key file path is required…/salt must be 16 bytes…/invalid salt hex…"` → `config-encryption-key-required / config-key-path-required / security-salt-invalid`。
* `src/device/memory_pool/tensor_pool.rs:146,153 "Batch size {} exceeds maximum {} / Sequence length {} exceeds maximum {}"` → 有 `validate-batch-size` 但场景不同，新增 `device-batch-exceeded / device-seq-exceeded`；`src/device/continuous_batch.rs:148 "Request {} timed out before processing"` → 复用 `pipeline-timeout` 加 `request_id` 参数。
* `src/library/mod.rs:137,140,146,159,175,197 "Failed to register/require EmbeddingModule/RerankModule…/Failed to build AsyncKit…"` → `api-kit-register-failed`（启动期，可低优）。
* `src/db/mod.rs:35,51,68,99,117 + audit/logger.rs:306,325 "Failed to create db pool/get session/create users/audit_logs table/get connection/insert audit log…"` → `db-pool-failed / db-session-failed…`（`error-database` 仅前缀）。
* `src/metrics/performance/mod.rs:41 "concurrent_requests must be greater than 0"` → `validate-concurrent-requests`。
* `src/registry/impl_.rs:188,251 detail:"rate limiter health check failed" / "cache disabled"` 作为 `health-check-failed` 的 `$detail` 原文塞入 → `health-ratelimit-failed / health-cache-disabled`，或保持小写技术标识但文档注明。

---

## 4. P2 — 显示 / 文档层

### 4.1 `src/error.rs` `Display`（17 处）

`src/error.rs:81-130 #[error("Config error: {0}")…]` 仍英文，与 `error-*` FTL 一字不差（仅参数名不同）。`IntoResponse` 已翻译所以 HTTP 头没问题，但：`Display` 经 `other.to_string()` 泄漏到 API（§1.5 两处），且测试锁死英文（`error.rs:510 assert == "Config error: test message"`、`639-653`）。二选一：(a) 保留英文供日志并加注释说明有意为之 + 杜绝 `to_string()` 做 `message`；（b) 让 `Display` 走 `tr`（注意 `tr` 前需 `init()`，未调用回退返回 key）。

### 4.2 `#[forge(description=)]` — 31 处全英文，无 FTL

经 `sdforge` 转为 `--help` / CLI JSON / MCP 工具描述，用户可见。注意 `#[forge]` 属性要求编译期常量，直调 `tr!` 不可行，需 `sdforge` 支持运行时 `description_key` 或文档层双语对照；先建 `cli-*/api-docs-*` 键。

| 文件 | 行 | 原文 |
|---|---|---|
| `src/api/embedding.rs` | 510,707,752 | `Generate embedding vector for input text` |
| `src/api/embedding.rs` | 523,718,763 | `Generate embedding vectors for multiple texts in batch` |
| `src/api/embedding.rs` | 536,729 | `Compute cosine similarity between two texts` |
| `src/api/embedding.rs` | 551,787 | `Embed text from a file with path validation` |
| `src/api/embedding.rs` | 565,842 | `Service health check` |
| `src/api/embedding.rs` | 578,798 | `Switch the currently loaded model` |
| `src/api/embedding.rs` | 591,809 | `Get information about the currently loaded model` |
| `src/api/embedding.rs` | 604,820 | `Get metadata about the currently loaded model` |
| `src/api/embedding.rs` | 617,831 | `List all available models` |
| `src/api/embedding.rs` | 631 | `OpenAI-compatible embeddings endpoint` |
| `src/api/embedding.rs` | 774 | `Compute similarity between two texts` |
| `src/api/rerank.rs` | 117,145,160 | `Rerank documents by relevance to a query` |
| `src/api/rerank.rs` | 130,171 | `Batch rerank multiple queries against document sets` |
| `src/api/auth.rs` | 35 | `User login with username and password` |
| `src/api/auth.rs` | 122 | `Refresh JWT token` |
| `src/api/auth.rs` | 197 | `Logout and revoke JWT token` |
| `src/api/auth.rs` | 234 | `Get current authenticated user info` |

### 4.3 `config/*.toml` 中文注释

值（`host/port/model_repo/use_gpu/batch_size/similarity_metric="cosine"/backend="memory"` 等）**必须不修**（机器契约）。`config.toml / config_full.toml` 全中文注释对英文部署者有门槛，但注释不进 FTL（confers 不解析注释），正确做法是提供 `config/config.en.toml` 或双语注释；`minimal.toml` 已是无注释语言中立范本。`full.toml:208` 示例密码有警示横幅，勿删。

---

## 5. 明确不修清单（避免误翻破坏兼容）

| 位置 | 原文 | 理由 |
|---|---|---|
| 全仓 `log::info!/warn!/error!/debug!`（如 `main.rs:83-1021`、`service/embedding.rs:293-1195`、`api/auth.rs:72,84,171,216`、`audit/logger.rs`、`auth/middleware.rs:83-304`） | 英文运维日志 | 机器可解析；含 5 条安全 warn（`Audit logging is DISABLED…/require_auth=false…insecure/trusted_proxies is empty…/No admin password…without password verification`）仍不修，译后反碍告警规则匹配 |
| `src/error.rs:32,37,41,45,49,53` | `[REDACTED_PATH] / [REDACTED_WINDOWS_PATH] / token [ID] / at position [REDACTED] / [INTERNAL_ERROR] / "..."` | 脱敏占位符，随 HTTP `{"error":…}` 可见但**必须不修**；翻译即破坏日志检索/告警/单测（`error.rs:341-633` 9 个 `assert!(contains!("[REDACTED…]"))`）。中文前缀 + 英文占位符混合属预期 |
| `src/api/auth.rs:98,185 token_type:"Bearer"`、`embedding.rs:679,688 object:"embedding"/"list"`、`field:"username/password/input/path"` | 协议常量 | OpenAI 兼容契约 |
| `examples/**` 100+ 处 `println!` 中文 + emoji | 演示代码不进货包 | FTL 是运行时依赖，示例保持零依赖可拷贝；`examples/README.md` 声明中文仅演示即可。例外 `examples/download_model.rs:35-200`（`print_usage/未知参数/下载完成…`）实为运维工具，一旦提升为 `src/bin` 即升 P1 |
| `src/main.rs:23,26` 等中文注释、`error.rs:67` char 边界注释、`.expect("sanitize pattern: …")` | 开发者可见 | 可顺手补英文双语，非必须 |
| `device/cuda.rs:489,516` 等 `Command::new("nvidia-smi"/"rocm-smi")`、`security/sanitize.rs:43 "[{} chars]"` | 子进程名 / 机器格式 | 不修 |
| `#[cfg(test)]` 假英文（`src/api/tests.rs:310,329,349 "bad input/model not found/cfg err"`，仅断 `status/code`） | 测试构造 | 无碍 |

---

## 6. 新增 FTL 键草案（按优先级，可直接粘贴扩写）

### 6.1 P0（`messages.ftl` 追加）

```ftl
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
```

```ftl
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
```

### 6.2 P1（`messages.ftl` 追加 + 启动键）

```ftl
# ── OOM / engine ──
oom-no-fallback-available = Out of memory and no fallback available
oom-fallback-failed = OOM error and fallback failed: { $detail }
oom-max-attempts = Max fallback attempts exceeded ({ $attempts }). Last error: { $detail }
engine-semaphore-failed = Failed to acquire semaphore: { $detail }
engine-batch-timeout = Batch chunk processing timed out after { $secs }s
# ── Metrics endpoint ──
metrics-limiter-unavailable = Rate limiter unavailable
metrics-rate-limited = Rate limit exceeded
metrics-collector-missing = PrometheusCollector not configured
metrics-encode-failed = Failed to encode metrics: { $detail }
# ── Startup / config ──
startup-db-pool = Failed to create database pool: { $detail }
startup-db-schema = Failed to initialize database schema: { $detail }
startup-jwt-length = JWT secret must be at least 32 characters long for security. Current length: { $got }
startup-jwt-missing = JWT secret is required when authentication is enabled. Please provide a strong JWT secret
startup-logger = Failed to initialize inklog logger: { $detail }
startup-config = Failed to load config via confers: { $detail }
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
```

（zh 镜像略，按 §0 表格体例翻译：OOM→内存不足…、startup→启动…失败、config→配置…、cli→CLI…；`{ $var }` 占位名保持与 en 一致。）

---

## 7. 按文档处理步骤（每组循环）

1. **加键**：先在 `src/i18n/locales/en/messages.ftl` + `zh/messages.ftl` 追加同名键（§6 草案），跑 `cargo test --lib i18n` 确认 `en/zh keys len` 一致（`bundle.rs` 有单测断言数量相等）。
2. **改调用**：仿 §0.3 把 `VecboostError::InvalidInput("…".to_string())` 改为 `tr / tr_with_args`；`format!("…: {e}")` 改为前缀键 + `{ $detail }` 透出 `e`（第三方原文保留）。
3. **改测试**：以下测试锁死英文断言，改完代码必须同步更新（否则必红）：
   * `src/error.rs:510,639-653`（`Display` 英文断言）
   * `src/utils/validator/input.rs:718,786`（`contains("Cannot access file"/"Cannot open file")`）及同文件 `834` 空白文本、`449-615` 各 `Expected InvalidInput` 处若断了具体文案
   * `src/utils/validator/path.rs:204,237,281,322,335,449`（`contains("No allowed root…/Access denied/Invalid path/Path traversal attempt")`）
   * `src/service/embedding.rs:2084-3517` 多处 `assert!(msg.contains("empty"))`（若改 `validate-text-empty` 文案需确认仍含 `empty` 或更新断言）
4. **验证**：`rg -n '"[A-Z][^"]{3,}"|用户名' src/utils/validator src/auth/types.rs src/pipeline/handler.rs src/pipeline/queue.rs src/pipeline/scheduler.rs src/service/common.rs src/metrics/endpoint.rs` 确认 P0 清零；`cargo test --lib` 全过。
5. **顺序**：§1（P0）→ §2（P1 启动/metrics/OOM）→ §3（S2 按 `tokenizer/vector/path/model/engine/security` 建族）→ §4（P2 显示层）。

## 附录：复核用命令

```bash
# P0 硬编码总览
rg -n '"Text |"Batch |"Search |"top_k |"File |"Cannot |"Validation failed|用户名' src/utils/validator src/auth
rg -n 'Response channel error|Request timeout|Queue is full|Rerank service not configured|auth disabled|password verification failed|init_state' src/pipeline src/api
# P1 启动/metrics/OOM
rg -n 'anyhow!|bail!' src/main.rs
rg -n 'Body::from\("' src/metrics/endpoint.rs
rg -n 'OutOfMemory\(|InferenceError\(format!' src/service/common.rs src/service/embedding.rs
# forge 描述
rg -n 'description\s*=\s*"' src/api
```
