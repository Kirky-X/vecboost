// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! JWT Token 登录与验证示例 — 演示 garrison 初始化、登录获取 token、验证 token 有效性

use std::sync::Arc;
use vecboost::auth::{
    GarrisonDaoOxcache, GarrisonManager, GarrisonUtil, VecBoostInterface,
    map_auth_config_to_garrison,
};
use vecboost::config::app::AuthConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 JWT Token 登录与验证示例（garrison）");
    println!("========================================\n");

    // 1. 构建 AuthConfig（模拟 config.toml 配置）
    let auth_config = AuthConfig {
        enabled: true,
        jwt_secret: Some("demo-secret-at-least-32-characters-long!!".to_string()),
        token_expiration_hours: Some(24),
        default_admin_username: Some("admin".to_string()),
        default_admin_password: Some("SecurePass123!".to_string()),
        csrf: vecboost::config::app::CsrfConfig::default(),
        trusted_proxies: vec![],
    };

    // 2. 映射为 garrison 配置并初始化全局单例
    let garrison_config = map_auth_config_to_garrison(&auth_config);
    let dao = GarrisonDaoOxcache::new().await?;
    let interface = VecBoostInterface::new(
        auth_config.default_admin_username.clone().unwrap_or_else(|| "admin".to_string()),
    );

    GarrisonManager::init(Arc::new(dao), Arc::new(garrison_config), Arc::new(interface))?;
    println!("✅ Garrison 初始化成功");

    // 3. 用户登录（login_id = username）
    let login_id = "alice";
    let token = GarrisonUtil::login_simple(login_id).await?;
    println!("\n📝 用户 '{}' 登录成功", login_id);
    println!("\n🎫 获取的 JWT Token:");
    println!("  {}...", &token[..token.len().min(80)]);

    // 4. 验证 token 有效性（通过 token 反查 login_id）
    let resolved_id = GarrisonUtil::get_login_id_by_token(&token).await?;
    match resolved_id {
        Some(id) => println!("\n✅ Token 验证成功，对应 login_id: {}", id),
        None => println!("\n❌ Token 无效或已过期"),
    }

    // 5. 撤销 token（登出）
    GarrisonUtil::revoke_token(&token).await?;
    println!("\n🚪 Token 已撤销（登出）");

    // 6. 验证撤销后的 token 应失效
    let after_revoke = GarrisonUtil::get_login_id_by_token(&token).await?;
    println!(
        "\n🔍 撤销后验证: {}",
        if after_revoke.is_none() {
            "Token 已失效 ✅"
        } else {
            "Token 仍有效 ❌"
        }
    );

    println!("\n✅ JWT 示例完成");
    Ok(())
}
