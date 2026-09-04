// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 安全模块示例 — 演示密钥管理、盐值生成和敏感数据清理
//!
//! 涵盖：
//! - `KeyStore` / `EnvironmentKeyStore`: 环境变量密钥存储
//! - `SecretKey`: 密钥类型与掩码显示
//! - `SaltStore`: Argon2 盐值生成与序列化
//! - `sanitize_*`: 日志中敏感数据的脱敏处理
//!
//! 运行: cargo run -p vecboost-examples --bin security_demo

use vecboost::security::{
    KeyType, SaltStore, SecretKey, sanitize_jwt_secret, sanitize_password, sanitize_secret,
};

#[tokio::main]
async fn main() {
    println!("🔒 安全模块示例");
    println!("================\n");

    // ─── SecretKey 与掩码显示 ──────────────────────────────────────────
    println!("📝 SecretKey 密钥管理:");

    let jwt_key = SecretKey::jwt_secret("my_super_secret_jwt_key_2024");
    println!(
        "  JWT Secret: {} (掩码: {})",
        jwt_key.name,
        jwt_key.mask_value()
    );

    let api_key = SecretKey::api_key("huggingface", "hf_abcdefghijklmnopqrstuvwxyz");
    println!(
        "  API Key: {} (掩码: {})",
        api_key.name,
        api_key.mask_value()
    );

    let db_pass = SecretKey::database_password("p@ssw0rd!");
    println!(
        "  DB Password: {} (掩码: {})",
        db_pass.name,
        db_pass.mask_value()
    );

    let model_key = SecretKey::model_api_key("sk-1234567890abcdef");
    println!(
        "  Model API Key: {} (掩码: {})\n",
        model_key.name,
        model_key.mask_value()
    );

    // ─── EnvironmentKeyStore ────────────────────────────────────────────
    println!("📝 EnvironmentKeyStore 操作:");

    // 存储密钥到环境变量
    let store =
        vecboost::security::create_key_store(&vecboost::security::SecurityConfig::default())
            .await
            .unwrap();

    let key = SecretKey::api_key("demo_service", "demo_api_key_value_12345");
    store.set(&key).await.unwrap();
    println!("  已存储 API Key: {}", key.name);

    // 读取密钥
    if let Some(retrieved) = store.get(&KeyType::ApiKey, "demo_service").await.unwrap() {
        println!(
            "  读取成功: {} = {}",
            retrieved.name,
            retrieved.mask_value()
        );
    }

    // 检查存在性
    let exists = store
        .exists(&KeyType::ApiKey, "demo_service")
        .await
        .unwrap();
    println!("  存在性检查: {}", exists);

    // 列出同类型密钥
    let keys = store.list(&KeyType::ApiKey).await.unwrap();
    println!("  API Key 列表: {} 个", keys.len());

    // 删除密钥
    store
        .delete(&KeyType::ApiKey, "demo_service")
        .await
        .unwrap();
    println!("  已删除 API Key");

    let exists_after = store
        .exists(&KeyType::ApiKey, "demo_service")
        .await
        .unwrap();
    println!("  删除后存在性: {}\n", exists_after);

    // ─── SaltStore ──────────────────────────────────────────────────────
    println!("📝 SaltStore 盐值管理:");

    let salt = SaltStore::generate();
    println!("  生成盐值 (hex): {}", salt.to_hex());
    println!("  盐值长度: {} bytes", salt.as_bytes().len());

    // hex 往返序列化
    let hex = salt.to_hex();
    let restored = SaltStore::from_hex(&hex).unwrap();
    assert_eq!(salt.as_bytes(), restored.as_bytes());
    println!("  hex 往返验证: ✅ 一致\n");

    // ─── 敏感数据脱敏 ──────────────────────────────────────────────────
    println!("📝 敏感数据脱敏:");

    let secret = "my_super_secret_key_12345";
    println!(
        "  sanitize_secret(\"{}\"): {}",
        secret,
        sanitize_secret(secret)
    );

    let password = "password123";
    println!(
        "  sanitize_password(\"{}\"): {}",
        password,
        sanitize_password(password)
    );

    let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
    println!(
        "  sanitize_jwt_secret(\"{}\"): {}",
        jwt,
        sanitize_jwt_secret(jwt)
    );

    // CJK 安全脱敏（不会 panic）
    let cjk_secret = "密钥内容不能泄露abcdefgh";
    println!("  sanitize_secret(CJK): {}", sanitize_secret(cjk_secret));

    println!("\n✅ 安全模块示例完成");
}
