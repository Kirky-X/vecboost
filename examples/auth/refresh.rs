// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Token 刷新流程示例 — 演示通过旧 token 获取新 token 并撤销旧 token 的完整刷新流程

use std::sync::Arc;
use vecboost::auth::{
    GarrisonDaoOxcache, GarrisonManager, GarrisonUtil, VecBoostInterface,
    map_auth_config_to_garrison,
};
use vecboost::config::app::AuthConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Token 刷新流程示例（garrison）");
    println!("==================================\n");

    // 1. 初始化 garrison
    let auth_config = AuthConfig {
        enabled: true,
        jwt_secret: Some("demo-secret-at-least-32-characters-long!!".to_string()),
        token_expiration_hours: Some(24),
        default_admin_username: Some("admin".to_string()),
        default_admin_password: Some("SecurePass123!".to_string()),
        csrf: vecboost::config::app::CsrfConfig::default(),
        trusted_proxies: vec![],
    };

    let garrison_config = map_auth_config_to_garrison(&auth_config);
    let dao = GarrisonDaoOxcache::new().await?;
    let interface = VecBoostInterface::new(
        auth_config
            .default_admin_username
            .clone()
            .unwrap_or_else(|| "admin".to_string()),
    );

    GarrisonManager::init(
        Arc::new(dao),
        Arc::new(garrison_config),
        Arc::new(interface),
    )?;
    println!("✅ Garrison 初始化成功");

    // 2. 用户登录获取原始 token
    let login_id = "bob";
    let original_token = GarrisonUtil::login_simple(login_id).await?;
    println!("\n🎫 原始 Token:");
    println!("  {}...", &original_token[..original_token.len().min(80)]);

    // 3. 刷新流程：从旧 token 反查 login_id，然后创建新会话
    println!("\n🔄 执行 Token 刷新...");
    let resolved_id = GarrisonUtil::get_login_id_by_token(&original_token)
        .await?
        .expect("Token should be valid");
    println!("  旧 token 对应 login_id: {}", resolved_id);

    let new_token = GarrisonUtil::login_simple(&resolved_id).await?;
    println!("✅ 新 Token 已生成");
    println!("  {}...", &new_token[..new_token.len().min(80)]);

    // 4. 撤销旧 token
    GarrisonUtil::revoke_token(&original_token).await?;
    println!("\n🚪 旧 Token 已撤销");

    // 5. 验证新旧 token 状态
    let old_valid = GarrisonUtil::get_login_id_by_token(&original_token).await?;
    let new_valid = GarrisonUtil::get_login_id_by_token(&new_token).await?;
    println!("\n📊 Token 状态:");
    println!(
        "  旧 token: {}",
        if old_valid.is_none() {
            "已失效 ✅"
        } else {
            "仍有效 ❌"
        }
    );
    println!(
        "  新 token: {}",
        if new_valid.is_some() {
            "有效 ✅"
        } else {
            "已失效 ❌"
        }
    );
    println!("  tokens 不同: {}", original_token != new_token);

    println!("\n💡 提示: garrison 的 token 刷新策略是「创建新会话 + 撤销旧会话」，");
    println!("   而非原地续期。这确保了旧 token 立即失效，提升安全性。");

    println!("\n✅ Token 刷新示例完成");
    Ok(())
}
