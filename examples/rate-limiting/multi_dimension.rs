// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 全局 + IP + 用户多维度限流示例。
//!
//! 自定义小限额（global=10, ip=5, user=3），演示不同维度独立计数：
//! IP 维度耗尽后用户维度仍可请求。

use vecboost::rate_limit::{LimiteronAdapter, RateLimitConfig, RateLimitDimension};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚦 多维度限流示例");
    println!("===================\n");

    let config = RateLimitConfig {
        global_requests_per_minute: 10,
        ip_requests_per_minute: 5,
        user_requests_per_minute: 3,
        api_key_requests_per_minute: 100,
        ..RateLimitConfig::default()
    };
    let adapter = LimiteronAdapter::new(config);

    let ip = RateLimitDimension::Ip("192.168.1.10".into());
    let user = RateLimitDimension::User("alice".into());

    println!("配置: global=10, ip=5, user=3\n");

    // 1) 耗尽 IP 维度（限额 5）
    println!("📍 模拟 IP 192.168.1.10 连续 6 次请求:");
    for i in 1..=6 {
        let allowed = adapter.check_rate_limit(vec![ip.clone()]).await;
        let status = adapter.get_status(ip.clone()).await;
        println!(
            "  第 {} 次: allowed={}  ip_remaining={}",
            i, allowed, status.remaining
        );
    }
    println!("→ IP 维度已耗尽（第 6 次被拒绝）\n");

    // 2) 用户维度仍可用
    println!("👤 模拟用户 alice 连续 2 次请求（IP 已耗尽，但 user 维度独立）:");
    for i in 1..=2 {
        let allowed = adapter.check_rate_limit(vec![user.clone()]).await;
        let status = adapter.get_status(user.clone()).await;
        println!(
            "  第 {} 次: allowed={}  user_remaining={}",
            i, allowed, status.remaining
        );
    }

    // 3) 组合维度：IP 已耗尽，组合检查应失败
    println!("\n🔍 组合维度检查（IP + User）:");
    let allowed = adapter
        .check_rate_limit(vec![ip.clone(), user.clone()])
        .await;
    println!("  allowed={}  （IP 已耗尽，组合检查失败）", allowed);

    // 4) 切换新 IP 仍可用（IP 维度按 IP 独立计数）
    let ip2 = RateLimitDimension::Ip("10.0.0.1".into());
    let allowed = adapter.check_rate_limit(vec![ip2.clone()]).await;
    let status = adapter.get_status(ip2).await;
    println!(
        "\n🌐 新 IP 10.0.0.1 请求: allowed={}  ip_remaining={}",
        allowed, status.remaining
    );
    println!("→ 不同 IP 独立计数");

    println!("\n✅ 示例完成");
    Ok(())
}
