// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! JWT Token 生成与验证示例 — 演示 JwtManager 创建 token、验证 token 并打印 Claims

use vecboost::auth::{JwtManager, User};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 JWT Token 生成与验证示例");
    println!("============================\n");

    let secret = "test-secret-at-least-32-characters-long".to_string();
    let manager = JwtManager::new(secret)?.with_expiration(1);
    println!("✅ JwtManager 创建成功 (过期时间: 1 小时)");

    let user = User {
        username: "alice".to_string(),
        role: "user".to_string(),
        permissions: vec!["embedding:read".to_string(), "embedding:write".to_string()],
    };

    println!("\n📝 用户信息:");
    println!("  用户名: {}", user.username);
    println!("  角色: {}", user.role);
    println!("  权限: {:?}", user.permissions);

    let token = manager.generate_token(&user)?;
    println!("\n🎫 生成的 JWT Token:");
    println!("  {}", &token[..token.len().min(80)]);

    let claims = manager.validate_token(&token).await?;
    println!("\n✅ Token 验证成功! Claims:");
    println!("  sub: {}", claims.sub);
    println!("  username: {}", claims.username);
    println!("  role: {}", claims.role);
    println!("  permissions: {:?}", claims.permissions);
    println!("  iat: {}", claims.iat);
    println!("  exp: {}", claims.exp);
    println!("  jti: {}", claims.jti);

    println!("\n✅ JWT 示例完成");
    Ok(())
}
