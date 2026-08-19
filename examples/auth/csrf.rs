// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CSRF 防护配置与 token 校验示例 — 演示 garrison CsrfConfig 配置、
//! token 生成与常量时间校验。
//!
//! garrison 采用 Double-Submit Cookie 模式：
//! - 安全方法（GET/HEAD/OPTIONS）自动在响应中设置 CSRF Cookie
//! - 受保护方法（POST/PUT/PATCH/DELETE）校验 Header 与 Cookie 中的 token 一致性

use vecboost::auth::{GarrisonCsrfConfig, generate_csrf_token, validate_csrf_token};

fn main() {
    println!("🛡️ CSRF 防护配置与校验示例（garrison）");
    println!("========================================\n");

    // 1. 默认配置（secure-by-default：默认启用）
    let default_config = GarrisonCsrfConfig::default();
    println!("📋 默认 CSRF 配置:");
    println!("  enabled: {}", default_config.enabled);
    println!("  cookie_name: {}", default_config.cookie_name);
    println!("  header_name: {}", default_config.header_name);
    println!(
        "  protected_methods: {:?}",
        default_config.protected_methods
    );
    println!("  cookie_secure: {}", default_config.cookie_secure);
    println!("  cookie_domain: {:?}", default_config.cookie_domain);

    // 2. 自定义配置
    let custom_config = GarrisonCsrfConfig {
        enabled: true,
        excluded_paths: vec!["/api/webhook".to_string()],
        cookie_domain: Some("example.com".to_string()),
        ..Default::default()
    };
    println!("\n📋 自定义 CSRF 配置:");
    println!("  excluded_paths: {:?}", custom_config.excluded_paths);
    println!("  cookie_domain: {:?}", custom_config.cookie_domain);

    // 3. Token 生成（garrison 内部使用 OsRng + URL-safe Base64）
    let token = generate_csrf_token().expect("CSRF token generation should not fail");
    println!("\n🎫 生成的 CSRF Token:");
    println!("  value: {}...", &token[..16]);
    println!(
        "  length: {} chars (32 bytes base64url-no-pad)",
        token.len()
    );

    // 4. Token 校验（常量时间比较，防时序攻击）
    let same_result = validate_csrf_token(&token, &token);
    println!("\n✅ 相同 token 校验: {}", same_result);

    let other_token = generate_csrf_token().unwrap();
    let diff_result = validate_csrf_token(&token, &other_token);
    println!("🚫 不同 token 校验: {}", diff_result);

    // 5. 空 token 安全处理
    let empty_result = validate_csrf_token("", "");
    println!("🚫 空 token 校验: {}", empty_result);

    println!("\n💡 提示: 实际集成中，CSRF 中间件自动处理 Cookie/Header 的生成与校验。");
    println!(
        "   客户端流程: GET 获取 Cookie → POST 时从 Cookie 读取 token 放入 X-CSRF-Token Header"
    );

    println!("\n✅ CSRF 示例完成");
}
