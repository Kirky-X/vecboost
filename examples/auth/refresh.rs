// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Token 刷新流程示例 — 演示 generate_token → refresh_token 的完整刷新流程

use vecboost::auth::{JwtManager, User};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Token 刷新流程示例");
    println!("========================\n");

    let secret = "test-secret-at-least-32-characters-long".to_string();
    let manager = JwtManager::new(secret)?.with_expiration(1);
    println!("✅ JwtManager 创建成功 (过期时间: 1 小时)");

    let user = User {
        username: "bob".to_string(),
        role: "user".to_string(),
        permissions: vec!["embedding:read".to_string()],
    };

    let original_token = manager.generate_token(&user)?;
    println!("\n🎫 原始 Token:");
    println!("  {}", &original_token[..original_token.len().min(80)]);

    println!("\n🔄 调用 refresh_token 刷新...");
    let refreshed_token = manager.refresh_token(&original_token).await?;
    println!("✅ 刷新成功");
    println!("\n🎫 新 Token:");
    println!("  {}", &refreshed_token[..refreshed_token.len().min(80)]);

    println!("\n📊 Token 对比:");
    println!("  原始 != 刷新: {}", original_token != refreshed_token);

    let claims = manager.validate_token(&refreshed_token).await?;
    println!("\n✅ 新 Token 验证成功:");
    println!("  username: {}", claims.username);
    println!("  role: {}", claims.role);
    println!("  jti: {}", claims.jti);

    println!("\n💡 提示: refresh_token 最多可刷新 5 次,超过后 token 将被加入黑名单");

    println!("\n✅ Token 刷新示例完成");
    Ok(())
}
