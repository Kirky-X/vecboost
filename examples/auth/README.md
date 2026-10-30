# Auth 示例

VecBoost 认证授权模块使用示例（基于 garrison 框架），涵盖 JWT 登录/验证、CSRF 防护和 token 刷新流程。

## 前置条件

```bash
# 无需额外 feature：vecboost-examples 自身已依赖 vecboost 的 auth/http 能力
# （命令行加 --features auth,http 会被 cargo 拒绝——本 crate 不定义这两个 feature）
cargo build -p vecboost-examples --bin jwt_auth
```

## 示例列表

| 示例 | 说明 | 核心 API |
|------|------|----------|
| `jwt_auth.rs` | JWT 登录、token 验证与撤销 | `GarrisonManager::builder` + `GarrisonUtil::login_simple` |
| `csrf.rs` | CSRF 配置与 token 生成/校验 | `GarrisonCsrfConfig` + `generate_csrf_token` |
| `refresh.rs` | Token 刷新（创建新会话 + 撤销旧会话） | `GarrisonUtil::get_login_id_by_token` + `revoke_token` |

## 运行方式

```bash
# 方式一：通过 cargo run
cargo run -p vecboost-examples --bin jwt_auth
cargo run -p vecboost-examples --bin csrf
cargo run -p vecboost-examples --bin refresh

# 方式二：编译后直接运行（产物在 target/debug 或 target/release 下，非 examples/ 子目录）
cargo build -p vecboost-examples --release --bin jwt_auth
./target/release/jwt_auth
```

## 所需 Feature

`vecboost-examples` 不透传 `auth` / `http`，这三个示例无需命令行 feature 即可编译运行（凭证来自 garrison 的 `backend-kit` / `web-axum` 等由主 crate 依赖恒常启用）。

## API 参考

### GarrisonManager（全局单例初始化，garrison 0.9 builder 链）

```rust
GarrisonManager::builder()
    .dao(dao: Arc<dyn GarrisonDao>)                    // 数据层（GarrisonDaoOxcache = 内存）
    .config(config: Arc<GarrisonConfig>)               // 框架配置（timeout, jwt_secret 等）
    .interface(interface: Arc<dyn GarrisonInterface>)  // 业务适配（权限/角色映射）
    .build()                                           // 异步：注入全局单例 + 启动后台 task
    .await?;
```

### GarrisonUtil（静态工具方法）

```rust
GarrisonUtil::login_simple(login_id: &str) -> GarrisonResult<String>           // 登录，返回 token
GarrisonUtil::get_login_id_by_token(token: &str) -> GarrisonResult<Option<String>>  // token → login_id
GarrisonUtil::revoke_token(token: &str) -> GarrisonResult<()>                  // 撤销 token
```

### CSRF

```rust
CsrfConfig::default() -> Self                    // secure-by-default 默认配置
generate_csrf_token() -> GarrisonResult<String>  // 生成 token（OsRng + Base64）
validate_csrf_token(header: &str, cookie: &str) -> bool  // 常量时间校验
```

## 安全提示

- 示例中的 JWT secret 仅用于演示，**生产环境必须使用高熵密钥**（至少 32 字节）
- CSRF 采用 Double-Submit Cookie 模式，token 校验使用常量时间比较（防时序攻击）
- Token 刷新策略为「创建新会话 + 撤销旧会话」，旧 token 立即失效
