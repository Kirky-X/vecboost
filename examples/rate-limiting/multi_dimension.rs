// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 全局 + IP + 用户多维度限流示例。
//!
//! 自定义小限额（global=10, ip=5, user=3），演示不同维度独立计数：
//! IP 维度耗尽后用户维度仍可请求。

use vecboost::rate_limit::{LimiteronAdapter, RateLimitSettings, RequestContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚦 多维度限流示例");
    println!("===================\n");

    let settings = RateLimitSettings {
        global_requests_per_minute: 10,
        ip_requests_per_minute: 5,
        user_requests_per_minute: 3,
        api_key_requests_per_minute: 100,
    };
    let adapter = LimiteronAdapter::new(settings).await;

    println!("配置: global=10, ip=5, user=3\n");

    // 1) 耗尽 IP 维度（限额 5）
    let ip_context = RequestContext {
        client_ip: Some("192.168.1.10".to_string()),
        path: "/api/v1/embed".to_string(),
        method: "POST".to_string(),
        ..Default::default()
    };

    println!("📍 模拟 IP 192.168.1.10 连续 6 次请求:");
    for i in 1..=6 {
        let allowed = adapter.check_rate_limit(&ip_context).await;
        println!("  第 {} 次: allowed={}", i, allowed);
    }
    println!("→ IP 维度已耗尽（第 6 次被拒绝）\n");

    // 2) 用户维度仍可用（不同 RequestContext 无 client_ip，仅检查 user_id）
    let user_context = RequestContext {
        user_id: Some("alice".to_string()),
        path: "/api/v1/embed".to_string(),
        method: "POST".to_string(),
        ..Default::default()
    };

    println!("👤 模拟用户 alice 连续 2 次请求（无 IP，仅 user 维度独立计数）:");
    for i in 1..=2 {
        let allowed = adapter.check_rate_limit(&user_context).await;
        println!("  第 {} 次: allowed={}", i, allowed);
    }

    // 3) 组合维度：IP 已耗尽，同时带 IP + User 应失败
    let combined_context = RequestContext {
        client_ip: Some("192.168.1.10".to_string()),
        user_id: Some("alice".to_string()),
        path: "/api/v1/embed".to_string(),
        method: "POST".to_string(),
        ..Default::default()
    };

    println!("\n🔍 组合维度检查（IP + User）:");
    let allowed = adapter.check_rate_limit(&combined_context).await;
    println!("  allowed={}  （IP 已耗尽，组合检查失败）", allowed);

    // 4) 切换新 IP 仍可用（IP 维度按 IP 独立计数）
    let new_ip_context = RequestContext {
        client_ip: Some("10.0.0.1".to_string()),
        path: "/api/v1/embed".to_string(),
        method: "POST".to_string(),
        ..Default::default()
    };

    let allowed = adapter.check_rate_limit(&new_ip_context).await;
    println!(
        "\n🌐 新 IP 10.0.0.1 请求: allowed={}",
        allowed
    );
    println!("→ 不同 IP 独立计数");

    println!("\n✅ 示例完成");
    Ok(())
}
