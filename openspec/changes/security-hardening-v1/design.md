# Design: Security Hardening V1

## Context

VecBoost 项目在代码审查中被识别出多项安全和代码质量问题。本设计文档记录关键技术决策，确保安全加固的一致性和可维护性。

### 约束条件

1. **向后兼容性**: 尽可能保持 API 和配置格式的兼容性
2. **渐进式迁移**: 现有部署不应立即中断
3. **最小改动**: 只修复必要的安全问题，不引入新功能
4. **测试覆盖**: 所有修改必须有对应的测试

## Goals / Non-Goals

### Goals

- 消除 Critical 和 High 级别的安全漏洞
- 支持分布式部署场景
- 提升代码质量和可维护性
- 通过安全扫描

### Non-Goals

- 不重构整体架构
- 不添加新功能
- 不修改非安全问题相关的业务逻辑

## Decisions

### 1. 敏感配置环境变量化

**决策**: 敏感配置（JWT 密钥、管理员密码）必须从环境变量读取，配置文件只提供默认值或占位符。

**实现方案**:
```rust
// src/config/app.rs
fn load_jwt_secret(config: &mut Config) -> Result<(), AppError> {
    // 优先级: 环境变量 > 配置文件 > 错误
    if let Ok(secret) = env::var("VECBOOST_JWT_SECRET") {
        if secret.len() < 32 {
            return Err(AppError::ConfigError(
                "JWT secret must be at least 32 characters".to_string()
            ));
        }
        config.set("auth.jwt_secret", secret)?;
    } else if config.get_str("auth.jwt_secret").unwrap_or("").is_empty() {
        return Err(AppError::ConfigError(
            "JWT secret must be set via VECBOOST_JWT_SECRET environment variable".to_string()
        ));
    }
    Ok(())
}
```

**替代方案考虑**:
- AWS Secrets Manager / HashiCorp Vault - 复杂度高，暂不需要
- 加密配置文件 - 增加运维复杂度

### 2. 分布式令牌存储

**决策**: 令牌黑名单和 CSRF 令牌支持 Redis 存储，同时保留内存存储作为开发环境回退。

**架构**:
```
┌─────────────────────────────────────────────────┐
│              Auth Token Store                    │
├─────────────────────┬───────────────────────────┤
│  MemoryTokenStore   │   RedisTokenStore         │
│  (开发/测试)        │   (生产)                   │
│                     │                            │
│  - HashSet<String>  │   - SET with TTL          │
│  - 服务重启丢失     │   - 跨实例共享             │
└─────────────────────┴───────────────────────────┘
```

**实现**:
```rust
// src/auth/token_store.rs
trait TokenStore: Send + Sync {
    async fn add_to_blacklist(&self, jti: &str);
    async fn is_blacklisted(&self, jti: &str) -> bool;
}

#[cfg(feature = "redis")]
struct RedisTokenStore {
    client: redis::Client,
}

#[cfg(not(feature = "redis"))]
struct MemoryTokenStore {
    blacklist: RwLock<HashSet<String>>,
}
```

### 3. 文件权限加固

**决策**: 密钥文件必须设置 `0o600` 权限，只有所有者可读写。

**实现**:
```rust
// src/security/encrypted_store.rs
async fn create_secure_file(path: &Path) -> Result<File, AppError> {
    let mut file = File::create(path).await?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = file.metadata().await?.permissions();
        perms.set_mode(0o600);
        file.set_permissions(perms).await?;
    }

    Ok(file)
}
```

### 4. 日志脱敏

**决策**: 禁止在日志中输出敏感配置信息。

**实现**:
```rust
// src/main.rs
fn log_config_safely(config: &AppConfig) {
    tracing::info!("Configuration loaded successfully");
    tracing::debug!(
        "Server: host={}, port={}",
        config.server.host,
        config.server.port
    );
    // 敏感字段不记录
}
```

### 5. 审计日志增强

**决策**: 审计日志必须包含完整的上下文信息。

**上下文字段**:
```json
{
  "user_agent": "Mozilla/5.0...",
  "request_path": "/api/v1/embed",
  "geo_location": null,
  "timestamp": "2025-01-14T12:00:00Z",
  "request_id": "req-uuid",
  "session_id": "sess-uuid"
}
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 配置迁移出错 | 生产部署失败 | 提供详细的迁移指南和回滚脚本 |
| Redis 依赖 | 增加运维复杂度 | 提供纯内存回退模式 |
| 性能影响 | 额外的 Redis 延迟 | 使用连接池和本地缓存 |

## Migration Plan

### 步骤 1: 准备阶段
1. 设置环境变量:
   ```bash
   export VECBOOST_JWT_SECRET="your-32-char-min-secret"
   export VECBOOST_ADMIN_PASSWORD="your-secure-password"
   ```

### 步骤 2: 部署阶段
1. 更新配置文件，移除敏感值
2. 部署新版本
3. 验证功能正常

### 步骤 3: 可选阶段
1. 配置 Redis 连接
2. 启用分布式令牌存储

### 回滚计划
1. 恢复旧配置文件
2. 部署上一版本
3. 验证服务恢复

## Open Questions

1. **Q: 是否需要支持配置文件加密?**
   A: 暂不需要，优先使用环境变量

2. **Q: Redis 连接失败时如何处理?**
   A: 回退到内存存储，记录警告日志

3. **Q: 默认密码生成策略?**
   A: 使用 `rand` crate 生成 32 字符随机密码
