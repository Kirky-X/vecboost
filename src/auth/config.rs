// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! AuthConfig → GarrisonConfig 映射。
//!
//! 将 VecBoost 的 `AuthConfig`（面向用户的配置面）转换为 garrison 的
//! `GarrisonConfig`（框架内部配置）。VecBoost 保留自有配置字段不变
//! （向后兼容 config.toml），映射函数在初始化时一次性调用。

use crate::config::app::AuthConfig;
use garrison::prelude::GarrisonConfig;

/// 将 VecBoost `AuthConfig` 映射为 garrison `GarrisonConfig`。
///
/// 映射规则：
/// - `token_expiration_hours` → `timeout`（hours × 3600 秒）
/// - `jwt_secret` → `jwt_secret`（JWT 签名密钥）
/// - `token_style` 固定 `"jwt"`
/// - `throw_on_not_login` = `auth.enabled`
/// - `frontend_separation` = `true`（API 服务，从 Authorization Header 读取 token）
/// - `is_read_cookie` = `false`（API 服务不使用 Cookie）
/// - `is_read_header` = `true`（从 Authorization Header 读取 token）
pub fn map_auth_config_to_garrison(auth: &AuthConfig) -> GarrisonConfig {
    let mut config = GarrisonConfig::default_config();

    // 会话超时：VecBoost 用小时，garrison 用秒
    if let Some(hours) = auth.token_expiration_hours {
        config.timeout = hours * 3600;
    }

    // JWT 签名密钥
    if let Some(ref secret) = auth.jwt_secret {
        config.jwt_secret = secret.clone();
    }

    // Token 风格固定为 JWT
    config.token_style = "jwt".to_string();

    // auth.enabled 控制未登录时行为：true = 抛异常（401），false = 返回 false
    config.throw_on_not_login = auth.enabled;

    // API 服务模式：从 Header 读取 token，不使用 Cookie
    config.frontend_separation = true;
    config.is_read_cookie = false;
    config.is_read_header = true;

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::app::AuthConfig;

    fn make_auth_config() -> AuthConfig {
        AuthConfig {
            enabled: true,
            jwt_secret: Some("test-jwt-secret-at-least-32-chars!!".to_string()),
            token_expiration_hours: Some(24),
            default_admin_username: Some("admin".to_string()),
            default_admin_password: Some("SecurePass123!".to_string()),
            csrf: crate::config::app::CsrfConfig::default(),
            trusted_proxies: vec![],
        }
    }

    #[test]
    fn test_map_timeout_from_hours() {
        let auth = make_auth_config();
        let garrison = map_auth_config_to_garrison(&auth);
        assert_eq!(garrison.timeout, 24 * 3600);
    }

    #[test]
    fn test_map_timeout_default_when_none() {
        let mut auth = make_auth_config();
        auth.token_expiration_hours = None;
        let garrison = map_auth_config_to_garrison(&auth);
        // garrison default is 2592000 (30 days)
        assert_eq!(garrison.timeout, 2592000);
    }

    #[test]
    fn test_map_jwt_secret() {
        let auth = make_auth_config();
        let garrison = map_auth_config_to_garrison(&auth);
        assert_eq!(
            garrison.jwt_secret,
            "test-jwt-secret-at-least-32-chars!!"
        );
    }

    #[test]
    fn test_map_token_style_is_jwt() {
        let auth = make_auth_config();
        let garrison = map_auth_config_to_garrison(&auth);
        assert_eq!(garrison.token_style, "jwt");
    }

    #[test]
    fn test_map_throw_on_notlogin_matches_enabled() {
        let mut auth = make_auth_config();
        auth.enabled = true;
        let garrison = map_auth_config_to_garrison(&auth);
        assert!(garrison.throw_on_not_login);

        auth.enabled = false;
        let garrison = map_auth_config_to_garrison(&auth);
        assert!(!garrison.throw_on_not_login);
    }

    #[test]
    fn test_map_frontend_separation_enabled() {
        let auth = make_auth_config();
        let garrison = map_auth_config_to_garrison(&auth);
        assert!(garrison.frontend_separation);
        assert!(!garrison.is_read_cookie);
        assert!(garrison.is_read_header);
    }
}
