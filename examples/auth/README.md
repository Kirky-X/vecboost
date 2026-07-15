# Auth 示例

VecBoost 认证授权模块使用示例,涵盖 JWT token 生成/验证、CSRF 防护和 token 刷新流程。

## 前置条件

```bash
# 编译(需 auth + http feature)
cargo build --features auth,http
```

## 示例列表

| 示例 | 说明 | 功能 |
|------|------|------|
| `jwt_auth.rs` | JWT token 生成与验证 | `JwtManager` + `User` + `Claims` |
| `csrf.rs` | CSRF token 获取与使用 | `CsrfConfig` + `CsrfTokenStore` + `CsrfToken` |
| `refresh.rs` | Token 刷新流程 | `JwtManager::refresh_token` |

## 运行方式

```bash
# 方式一:通过 cargo run(需先在 Cargo.toml 注册 example)
cargo run -p vecboost-examples --bin jwt_auth --features auth,http
cargo run -p vecboost-examples --bin csrf --features auth,http
cargo run -p vecboost-examples --bin refresh --features auth,http

# 方式二:编译后直接运行
cargo build --release --features auth,http --example jwt_auth
./target/release/examples/jwt_auth
```

## 所需 Feature

- `auth` — 启用 JWT/CSRF 认证模块(jsonwebtoken + argon2 + aes-gcm)
- `http` — auth 模块内部依赖 axum/http 类型(CsrfConfig 等需要)

## API 参考

### JwtManager

```rust
JwtManager::new(secret: String) -> Result<Self, VecboostError>
    .with_expiration(hours: i64) -> Self
    .generate_token(&self, user: &User) -> Result<String, VecboostError>
    .validate_token(&self, token: &str) -> Result<Claims, VecboostError>  // async
    .refresh_token(&self, token: &str) -> Result<String, VecboostError>   // async
```

### CSRF

```rust
CsrfConfig::new(allowed_origins: Vec<String>) -> Self
    .with_token_validation(enabled: bool) -> Self
    .with_token_expiration(secs: u64) -> Self
    .is_origin_allowed(&self, origin: &str) -> bool

CsrfTokenStore::new() -> Self
    .store_token(&self, token: &str)        // async, 一次性使用
    .validate_token(&self, token: &str) -> bool  // async

CsrfToken::new(expires_in_secs: u64) -> Self
```

## 安全提示

- 示例中使用的 secret 仅用于演示,**生产环境必须使用高熵密钥**
- JWT secret 最小长度 32 字节,最小熵值 128 位
- CSRF token 为一次性使用,验证后自动失效(防重放攻击)
