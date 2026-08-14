// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Auth 模块 — garrison 集成层。
//!
//! 本模块将 VecBoost 的认证鉴权完全委托给 garrison 框架：
//! - `config` — AuthConfig → GarrisonConfig 映射
//! - `interface` — VecBoostInterface（GarrisonInterface 实现）
//! - `middleware` — axum 中间件（外壳保留，内部委托 garrison）
//! - `types` — HTTP 请求/响应类型（serde 序列化，无业务逻辑）
pub mod config;
pub mod interface;
#[cfg(feature = "http")]
pub mod middleware;
pub mod types;

// Re-export garrison 核心类型
pub use garrison::prelude::{GarrisonConfig, GarrisonManager, GarrisonUtil};

// Re-export VecBoost 适配类型
pub use config::map_auth_config_to_garrison;
pub use interface::VecBoostInterface;

/// Garrison 初始化标记类型。
///
/// garrison 使用全局单例模式（`GarrisonManager::init()` 后通过 `GarrisonUtil` 静态方法访问），
/// 无需在 kit 中持有 `GarrisonManager` 实例。此 newtype 作为 AuthModule 的 Capability，
/// `Some` 表示 garrison 已初始化，`None` 表示 auth 未启用。
///
/// `admin_password_hash` 存储 admin 用户的 Argon2 密码哈希（启动时从 config 计算），
/// 供 `forge_login` 校验密码。`None` 表示未配置密码（仅受信任网络环境）。
/// `token_timeout_secs` 存储 garrison config 的 token 超时秒数，供 handler 读取。
#[derive(Clone, Debug)]
pub struct GarrisonHandle {
    /// admin 用户密码哈希（Argon2/Bcrypt PHC 格式），用于登录校验。
    pub admin_password_hash: Option<String>,
    /// Token 超时秒数（从 garrison config.timeout 读取）。
    pub token_timeout_secs: i64,
}

/// Re-export 保留的 HTTP 类型（API 契约不变）
pub use types::{AuthResponse, LoginRequest, RefreshTokenRequest, User, validate_username_format};

// Re-export garrison 密码哈希（替代手写 argon2 实现）
pub use garrison::account::credential::password::{Argon2Hasher, PasswordHasher};

// Re-export garrison CSRF 类型与工具函数
#[cfg(feature = "auth")]
pub use garrison::web::csrf::{
    CsrfConfig as GarrisonCsrfConfig, generate_csrf_token, validate_csrf_token,
};

// Re-export garrison CSRF 中间件（替代手写 csrf_origin/csrf/csrf_combined 中间件）
#[cfg(feature = "auth")]
pub use garrison::web::csrf::garrison_csrf_middleware;

// Re-export garrison task_local token 工具（供下游 handler 使用权限查询）
#[cfg(feature = "auth")]
pub use garrison::stp::with_current_token;

// Re-export garrison DAO（内存实现，供示例/测试使用）
#[cfg(feature = "auth")]
pub use garrison::dao::GarrisonDaoOxcache;

// Re-export middleware 函数
#[cfg(feature = "http")]
pub use middleware::{
    auth_middleware, auth_rate_limit_middleware, optional_auth_middleware,
    require_permission_middleware, require_role_middleware,
};

#[cfg(test)]
mod tests {
    use super::*;
    use garrison::account::credential::password::PasswordVerifier;

    #[test]
    fn argon2_hash_then_verify_correct_password() {
        let password = "secure_admin_pass";
        let hash = Argon2Hasher::default()
            .hash(password)
            .expect("hash must succeed");
        assert!(hash.starts_with("$argon2"));
        let ok = PasswordVerifier::verify(password, &hash).expect("verify must succeed");
        assert!(ok, "correct password must verify");
    }

    #[test]
    fn argon2_hash_then_verify_wrong_password() {
        let password = "correct_password";
        let hash = Argon2Hasher::default()
            .hash(password)
            .expect("hash must succeed");
        let ok = PasswordVerifier::verify("wrong_password", &hash).expect("verify must succeed");
        assert!(!ok, "wrong password must not verify");
    }

    #[test]
    fn garrison_handle_with_password_hash() {
        let hash = Argon2Hasher::default()
            .hash("test_pw")
            .expect("hash must succeed");
        let handle = GarrisonHandle {
            admin_password_hash: Some(hash.clone()),
            token_timeout_secs: 7200,
        };
        assert_eq!(handle.token_timeout_secs, 7200);
        assert!(handle.admin_password_hash.is_some());
        let ok = PasswordVerifier::verify("test_pw", handle.admin_password_hash.as_ref().unwrap())
            .expect("verify must succeed");
        assert!(ok);
    }

    #[test]
    fn garrison_handle_without_password_hash() {
        let handle = GarrisonHandle {
            admin_password_hash: None,
            token_timeout_secs: 3600,
        };
        assert!(handle.admin_password_hash.is_none());
        assert_eq!(handle.token_timeout_secs, 3600);
    }
}
