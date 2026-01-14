# Tasks: Security Hardening V1

## Phase 1: Critical Issues (Day 1-2)

### 1.1 Environment Variable Support for JWT Secret
- [ ] 1.1.1 Add `VECBOOST_JWT_SECRET` environment variable loading in config
- [ ] 1.1.2 Add validation for minimum length (32 chars) and entropy
- [ ] 1.1.3 Update config.toml to remove hardcoded jwt_secret
- [ ] 1.1.4 Add tests for JWT secret loading
- [ ] 1.1.5 Update documentation

### 1.2 Environment Variable Support for Admin Password
- [ ] 1.2.1 Add `VECBOOST_ADMIN_PASSWORD` environment variable loading
- [ ] 1.2.2 Implement secure random password generator
- [ ] 1.2.3 Add password complexity validation
- [ ] 1.2.4 Update config.toml to remove hardcoded password
- [ ] 1.2.5 Add tests for password loading
- [ ] 1.2.6 Update documentation

### 1.3 Redis Token Blacklist Support
- [ ] 1.3.1 Create `TokenStore` trait in src/auth/
- [ ] 1.3.2 Implement `RedisTokenStore` for blacklist
- [ ] 1.3.3 Modify `JwtManager` to use `TokenStore` trait
- [ ] 1.3.4 Add configuration for Redis connection
- [ ] 1.3.5 Add fallback to memory store on Redis failure
- [ ] 1.3.6 Add tests for token blacklist operations

### 1.4 Redis CSRF Token Storage Support
- [ ] 1.4.1 Modify `CsrfTokenStore` to use configurable backend
- [ ] 1.4.2 Implement Redis backend for CSRF tokens
- [ ] 1.4.3 Add configuration for CSRF store type
- [ ] 1.4.4 Add tests for CSRF token storage

## Phase 2: High Priority Issues (Day 3-4)

### 2.1 Secure File Permissions
- [ ] 2.1.1 Update `EncryptedStore::create()` to set 0o600 permissions
- [ ] 2.1.2 Add verification of file permissions after creation
- [ ] 2.1.3 Add tests for file permission setting
- [ ] 2.1.4 Test on Unix-like systems

### 2.2 Configuration Logging Sanitization
- [ ] 2.2.1 Update `src/main.rs` to avoid logging sensitive config
- [ ] 2.2.2 Create safe config logging function
- [ ] 2.2.3 Add tests to verify sensitive data not logged
- [ ] 2.2.4 Review all logging statements for sensitive data

### 2.3 Audit Log Context Enhancement
- [ ] 2.3.1 Update `SecurityEvent` struct to include additional fields
- [ ] 2.3.2 Modify `AuditLogger` methods to accept context
- [ ] 2.3.3 Add user_agent, request_path, request_id to events
- [ ] 2.3.4 Update all audit log calls with context
- [ ] 2.3.5 Add tests for audit log context

### 2.4 Blocking Call Fixes
- [ ] 2.4.1 Identify all `futures::executor::block_on` calls in hot paths
- [ ] 2.4.2 Refactor to use `tokio::task::spawn_blocking`
- [ ] 2.4.3 Add tests to verify async behavior

### 2.5 Lock Optimization
- [ ] 2.5.1 Audit lock usage in `src/service/embedding.rs`
- [ ] 2.5.2 Combine multiple lock acquisitions where possible
- [ ] 2.5.3 Add benchmarks to verify improvement

## Phase 3: Medium Priority Issues (Week 1)

### 3.1 Replace Manual Clone Implementations
- [ ] 3.1.1 Replace manual `Clone` for `IntelDevice` with derive
- [ ] 3.1.2 Replace manual `Clone` for `AmdDevice` with derive
- [ ] 3.1.3 Verify all fields implement Clone

### 3.2 Extract Cache Stats Pattern
- [ ] 3.2.1 Create `CacheStats` trait
- [ ] 3.2.2 Refactor `LruCache`, `ArcCache`, `TieredCache` to use trait
- [ ] 3.2.3 Add tests for trait implementation

### 3.3 Remove unsafe from Environment Variable Access
- [ ] 3.3.1 Review `src/security/key_store.rs` unsafe usage
- [ ] 3.3.2 Replace with safe `std::env::set_var` or remove
- [ ] 3.3.3 Add tests to verify behavior

## Phase 4: Low Priority Issues (Week 2)

### 4.1 Batch Encoding Parallelization
- [ ] 4.1.1 Modify `CandleEngine::encode_batch` to use `FuturesUnordered`
- [ ] 4.1.2 Add benchmarks comparing sequential vs parallel
- [ ] 4.1.3 Add tests for batch encoding

### 4.2 ARC Cache Optimization
- [ ] 4.2.1 Replace `VecDeque` with `HashSet` in ARC cache
- [ ] 4.2.2 Add benchmarks for cache operations
- [ ] 4.2.3 Add tests for cache functionality

### 4.3 HTTPS Enforcement
- [ ] 4.3.1 Add `force_https` configuration option
- [ ] 4.3.2 Add middleware to redirect HTTP to HTTPS
- [ ] 4.3.3 Add tests for HTTPS enforcement

### 4.4 Account Lockout
- [ ] 4.4.1 Implement failed login counter
- [ ] 4.4.2 Add exponential backoff for locked accounts
- [ ] 4.4.3 Add tests for account lockout

### 4.5 Security Response Headers
- [ ] 4.5.1 Add middleware for security headers
- [ ] 4.5.2 Add headers: X-Content-Type-Options, X-Frame-Options
- [ ] 4.5.3 Add tests for response headers

## Validation

### Code Quality
- [ ] cargo check passes
- [ ] cargo clippy passes (no warnings)
- [ ] cargo fmt passes

### Testing
- [ ] All unit tests pass
- [ ] New tests added for each change
- [ ] Integration tests pass

### Security
- [ ] cargo audit passes
- [ ] No new vulnerabilities introduced

### Documentation
- [ ] README updated with environment variable requirements
- [ ] Migration guide created
