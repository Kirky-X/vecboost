# Change Proposal: Security Hardening V1

## Summary

修复代码审查中识别的 26 个安全和代码质量问题，重点解决 4 个 Critical 和 7 个 High 优先级问题，提升 VecBoost 的生产环境安全性。

## Why

当前代码存在多项安全风险：

1. **敏感凭据暴露**: config.toml 中明文存储 JWT 密钥和默认密码
2. **分布式失效**: 令牌黑名单和 CSRF 令牌使用内存存储，多实例部署时失效
3. **文件权限风险**: 密钥文件使用默认 Unix 权限 (0644)，可被其他用户读取
4. **运行时 panic**: 大量 `.unwrap()` 调用可能导致服务崩溃
5. **审计日志不完整**: 缺少关键上下文信息

这些问题在生产环境中可能导致：
- 未授权访问 (弱密码、明文凭据)
- 安全机制失效 (内存存储不支持分布式)
- 数据泄露 (密钥文件权限不当)
- 服务中断 (panic 导致的崩溃)

## What Changes

### Critical (必须立即修复)

| # | 问题 | 文件 | 修复内容 |
|---|------|------|---------|
| 1 | 明文密码在配置文件中 | `config.toml` | 改用环境变量 `VECBOOST_JWT_SECRET`, `VECBOOST_ADMIN_PASSWORD` |
| 2 | 默认密码强度不足 | `src/config/app.rs:412` | 使用随机生成的强密码 (16+ 字符) |
| 3 | 令牌黑名单内存存储 | `src/auth/jwt.rs:43` | 支持 Redis 持久化存储 |
| 4 | CSRF令牌不支持分布式 | `src/auth/csrf.rs:154` | 支持 Redis 存储 |

### High (本周修复)

| # | 问题 | 文件 | 修复内容 |
|---|------|------|---------|
| 5 | 密钥文件权限未设置 | `src/security/encrypted_store.rs:151` | 设置 `0o600` 权限 |
| 6 | 配置日志泄露敏感信息 | `src/main.rs:34` | 移除敏感字段的日志输出 |
| 7 | 审计日志缺少上下文 | `src/audit/mod.rs:256` | 添加 user_agent, request_path |
| 8 | 阻塞调用问题 | `src/engine/candle_engine.rs:745` | 使用 `spawn_blocking` |
| 9 | 多次获取读锁 | `src/service/embedding.rs` | 合并锁获取操作 |

### Medium (两周内修复)

| # | 问题 | 文件 |
|---|------|------|
| 10 | 手动实现 Clone | `src/device/intel.rs`, `amd.rs` |
| 11 | 重复代码模式 | 多缓存实现的 stats 方法 |
| 12 | AppState 结构庞大 | `src/lib.rs:48-63` |
| 13 | unsafe 块使用环境变量 | `src/security/key_store.rs:103` |

### Low (建议改进)

| # | 问题 | 文件 | 建议 |
|---|------|------|------|
| 14 | 批量编码可并行化 | `candle_engine.rs:706` | 使用 `FuturesUnordered` |
| 15 | ARC缓存 O(n) 查找 | `arc_cache.rs:61` | 改用 `HashSet` |
| 16 | 缺少 HTTPS 强制 | 配置 | 添加 `force_https` |
| 17 | 缺少账户锁定 | 认证模块 | 实现指数退避锁定 |
| 18 | 缺少安全响应头 | 中间件 | 添加安全响应头 |

## Impact

### Affected Specs

- `security/authentication` - JWT/CSRF 存储修改
- `security/audit` - 审计日志增强
- `config/app` - 默认配置修改

### Affected Code

- `src/auth/jwt.rs` - 令牌黑名单存储
- `src/auth/csrf.rs` - CSRF 令牌存储
- `src/config/app.rs` - 默认密码配置
- `src/security/encrypted_store.rs` - 文件权限
- `src/main.rs` - 配置日志
- `src/audit/mod.rs` - 审计日志上下文

### Breaking Changes

- **配置文件格式变化**: `jwt_secret` 和 `default_admin_password` 不再从配置文件读取
- **需要的环境变量**: `VECBOOST_JWT_SECRET`, `VECBOOST_ADMIN_PASSWORD`

## Dependencies

### External Dependencies

- Redis (可选): 用于分布式令牌存储

### Internal Dependencies

- `src/config/app.rs` - 配置模块
- `src/auth/` - 认证模块
- `src/security/` - 安全模块
- `src/audit/` - 审计模块

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 配置迁移复杂 | High | 提供迁移脚本和向后兼容模式 |
| Redis 不可用 | Medium | 保留内存存储作为回退 |
| 现有部署中断 | Medium | 提供渐进式迁移路径 |

## Success Criteria

- [ ] 所有 Critical 问题修复
- [ ] 所有 High 问题修复
- [ ] 代码编译通过，无新警告
- [ ] 单元测试通过
- [ ] 安全扫描通过 (cargo-audit)
- [ ] 文档更新

## Timeline

- **Phase 1 (Critical)**: 1-2 天
  - Day 1: 环境变量支持 + 默认密码加固
  - Day 2: Redis 令牌存储支持
- **Phase 2 (High)**: 3-4 天
  - Day 3: 文件权限 + 日志脱敏
  - Day 4: 审计日志上下文 + 阻塞调用修复
- **Phase 3 (Medium)**: 1 周
- **Phase 4 (Low)**: 2 周

---

**Change ID**: `security-hardening-v1`  
**Status**: `draft`  
**Created**: 2025-01-14  
**Author**: Sisyphus AI Agent
