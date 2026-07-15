// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CSRF Token 获取与使用示例 — 演示 CsrfConfig 配置、CsrfToken 生成与 CsrfTokenStore 验证

use vecboost::auth::{CsrfConfig, CsrfToken, CsrfTokenStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ CSRF Token 获取与使用示例");
    println!("==============================\n");

    let config = CsrfConfig::new(vec![
        "http://localhost:3000".to_string(),
        "https://example.com".to_string(),
    ])
    .with_token_validation(true)
    .with_token_expiration(3600);

    println!("📋 CSRF 配置:");
    println!("  允许的 Origins: http://localhost:3000, https://example.com");
    println!("  Token 验证: 启用");
    println!("  Token 过期时间: {} 秒", config.token_expiration_secs);

    println!("\n🔍 Origin 验证测试:");
    println!(
        "  http://localhost:3000 -> {}",
        config.is_origin_allowed("http://localhost:3000")
    );
    println!(
        "  https://example.com   -> {}",
        config.is_origin_allowed("https://example.com")
    );
    println!(
        "  https://evil.com      -> {}",
        config.is_origin_allowed("https://evil.com")
    );

    let store = CsrfTokenStore::new();
    let token = CsrfToken::new(3600);
    println!("\n🎫 生成的 CSRF Token:");
    println!("  value: {}...", &token.value[..16]);
    println!("  expires_at: {}", token.expires_at);
    println!("  is_expired: {}", token.is_expired());

    store.store_token(&token.value).await;
    println!(
        "\n💾 Token 已存储 (store count: {})",
        store.token_count().await
    );

    let valid = store.validate_token(&token.value).await;
    println!("✅ 首次验证结果: {}", valid);

    let replay = store.validate_token(&token.value).await;
    println!("🚫 重放攻击防护 (第二次验证): {}", replay);

    println!("\n✅ CSRF 示例完成");
    Ok(())
}
