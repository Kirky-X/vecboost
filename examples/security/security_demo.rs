// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 安全模块示例 — 演示密钥管理、盐值生成和敏感数据清理
//!
//! 涵盖：
//! - `KeyStore` / `EnvironmentKeyStore`: 环境变量密钥存储（只读,set/delete 拒绝写入）
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

    // ─── EnvironmentKeyStore（只读安全设计） ───────────────────────────
    println!("📝 EnvironmentKeyStore 操作（只读）:");

    let store =
        vecboost::security::create_key_store(&vecboost::security::SecurityConfig::default())
            .await
            .unwrap();

    // set 被拒绝:环境变量是父进程注入源,写回会经 /proc/<pid>/environ 泄漏给同机读者与子进程
    let key = SecretKey::api_key("demo_service", "demo_api_key_value_12345");
    match store.set(&key).await {
        Ok(()) => println!("  set 成功(与只读设计矛盾,不应出现)"),
        Err(e) => println!("  set 拒绝(预期): {}", e.error_detail()),
    }

    // 读取:仅当父进程已注入 VECBOOST_API_KEY_* 时才有值
    let retrieved = store.get(&KeyType::ApiKey, "demo_service").await.unwrap();
    println!(
        "  读取 demo_service: {}",
        if retrieved.is_some() {
            "命中"
        } else {
            "未注入"
        }
    );

    let exists = store
        .exists(&KeyType::ApiKey, "demo_service")
        .await
        .unwrap();
    println!("  存在性检查: {}", exists);

    let keys = store.list(&KeyType::ApiKey).await.unwrap();
    println!("  API Key 列表: {} 个", keys.len());

    // delete 同样只读:运行时不可撤销父进程注入的环境变量
    match store.delete(&KeyType::ApiKey, "demo_service").await {
        Ok(()) => println!("  delete 成功(与只读设计矛盾,不应出现)"),
        Err(e) => println!("  delete 拒绝(预期): {}\n", e.error_detail()),
    }

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
    println!("  sanitize_secret: {}", sanitize_secret(secret));

    let password = "password123";
    println!("  sanitize_password: {}", sanitize_password(password));

    let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
    println!("  sanitize_jwt_secret: {}", sanitize_jwt_secret(jwt));

    // CJK 安全脱敏（不会 panic）
    let cjk_secret = "密钥内容不能泄露abcdefgh";
    println!("  sanitize_secret(CJK): {}", sanitize_secret(cjk_secret));

    println!("\n✅ 安全模块示例完成");
}
