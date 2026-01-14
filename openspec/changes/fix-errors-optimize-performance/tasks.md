# Tasks: Fix Errors and Performance Optimization

## Phase 1: 错误处理修复

### Task 1.1: 修复 main.rs 中的 unwrap()
**Status**: pending  
**Priority**: High  
**Files**: `src/main.rs`  
**Validation**: `cargo check --lib` passes  

#### Steps
1. Read `src/main.rs` lines 140-145
2. Replace `.expect()` with proper error handling
3. Verify compilation

#### Details
- Replace `.expect("Failed to create default admin user")` with proper error mapping
- Replace `.expect("Failed to add default admin user")` with proper error mapping

---

### Task 1.2: 修复 auth 模块中的 unwrap()
**Status**: pending  
**Priority**: High  
**Files**: `src/auth/jwt.rs`, `src/auth/user_store.rs`  
**Validation**: `cargo check --lib` passes  

#### Steps
1. Scan for `.unwrap()` and `.expect()` in auth modules
2. Replace with proper error handling
3. Ensure error messages are descriptive

#### Details
- `src/auth/jwt.rs:98` - JWT secret validation
- `src/auth/user_store.rs` - multiple unwrap calls in password hashing

---

### Task 1.3: 修复 service 模块中的 unwrap()
**Status**: pending  
**Priority**: High  
**Files**: `src/service/embedding.rs`  
**Validation**: `cargo check --lib` passes  

#### Steps
1. Scan for `.unwrap()` in test code paths
2. Replace with proper error handling in production paths
3. Document any intentional unwrap in tests

#### Details
- Focus on lines 1184-1742 (test code)
- Ensure production paths have proper error handling

---

## Phase 2: 审计日志启用

### Task 2.1: 启用 auth_middleware 审计日志
**Status**: pending  
**Priority**: High  
**Files**: `src/auth/middleware.rs`  
**Validation**: Code compiles, audit logging works  

#### Steps
1. Uncomment audit logger state extraction
2. Uncomment unauthorized access logging
3. Test logging functionality

#### Details
- Lines 37-38: Uncomment `State(audit_logger)`
- Lines 54-55: Uncomment audit log calls
- Lines 74-76: Uncomment invalid token logging

---

### Task 2.2: 启用 permission_middleware 审计日志
**Status**: pending  
**Priority**: Medium  
**Files**: `src/auth/middleware.rs`  
**Validation**: Code compiles, permission denied logged  

#### Steps
1. Uncomment permission denied logging
2. Add user context to log entries
3. Verify log output

#### Details
- Lines 107: Uncomment `State(audit_logger)`
- Lines 119-124: Uncomment audit log calls

---

## Phase 3: 内存优化

### Task 3.1: 优化 JWT token 生成
**Status**: pending  
**Priority**: Medium  
**Files**: `src/auth/jwt.rs`  
**Validation**: Memory usage reduced, tests pass  

#### Steps
1. Analyze `.clone()` calls in `generate_token()`
2. Optimize to use references where possible
3. Benchmark before/after

#### Details
- Lines 177-185: Claims creation clones user fields
- Consider restructuring to reduce clones

---

### Task 3.2: 优化 embedding 服务
**Status**: pending  
**Priority**: Medium  
**Files**: `src/service/embedding.rs`  
**Validation**: Memory usage reduced, tests pass  

#### Steps
1. Analyze `.clone()` calls in process_text
2. Identify unnecessary clones in cache key creation
3. Optimize cache key generation

#### Details
- Line 323: `format!("text:{}", req.text)` - potential optimization
- Review all cache key generation patterns

---

## Phase 4: 代码清理

### Task 4.1: 替换 println! 为 tracing!
**Status**: pending  
**Priority**: Low  
**Files**: `src/text/aggregator.rs`  
**Validation**: `cargo check --lib` passes  

#### Steps
1. Replace `println!()` with `tracing::debug!()`
2. Verify logging output
3. Remove debug prints in production

#### Details
- Lines 217-218: Normalized output
- Lines 289-290: Weighted aggregation output

---

### Task 4.2: 清理未使用的导入
**Status**: pending  
**Priority**: Low  
**Files**: `src/auth/jwt.rs`, multiple files  
**Validation**: `cargo clippy --all-features` passes  

#### Steps
1. Run `cargo clippy --all-features`
2. Identify unused imports
3. Remove unused imports

#### Details
- `src/auth/jwt.rs:6` - `#![allow(unused)]`
- Check other files with similar patterns

---

## Phase 5: 验证和测试

### Task 5.1: 运行完整测试套件
**Status**: pending  
**Priority**: High  
**Files**: All modified files  
**Validation**: All tests pass  

#### Steps
1. Run `cargo test --all-features`
2. Fix any failing tests
3. Ensure test coverage maintained

---

### Task 5.2: 性能基准测试
**Status**: pending  
**Priority**: Medium  
**Files**: N/A  
**Validation**: Performance metrics collected  

#### Steps
1. Run performance tests before changes
2. Run performance tests after changes
3. Compare results

---

## Dependencies

- Task 1.1 must complete before Task 5.1
- Task 2.1 and 2.2 can run in parallel
- Task 3.1 and 3.2 can run in parallel
- Task 5.1 depends on all previous tasks

## Total Estimated Effort

- Phase 1: 4-6 hours
- Phase 2: 2-3 hours
- Phase 3: 6-8 hours
- Phase 4: 2-3 hours
- Phase 5: 2-4 hours

**Total: 16-24 hours**

---

**Created**: 2025-01-14  
**Status**: `draft`
