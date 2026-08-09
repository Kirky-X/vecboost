// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron 限流配置示例。
//!
//! 使用默认配置创建 LimiteronAdapter，连续调用 check_rate_limit，
//! 观察 Governor stats 输出的统计信息。

use vecboost::rate_limit::{LimiteronAdapter, RequestContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚦 limiteron 限流示例（默认配置）");
    println!("==================================\n");

    let adapter = LimiteronAdapter::with_defaults().await;

    // 构建请求上下文（模拟来自同一 IP 的 HTTP POST 请求）
    let context = RequestContext {
        client_ip: Some("192.168.1.100".to_string()),
        user_id: Some("alice".to_string()),
        path: "/api/v1/embed".to_string(),
        method: "POST".to_string(),
        ..Default::default()
    };

    println!("初始健康状态: {}", adapter.is_healthy());

    println!("\n连续发起 5 次请求：\n");
    for i in 1..=5 {
        let allowed = adapter.check_rate_limit(&context).await;
        println!("  第 {} 次: allowed={}", i, allowed);
    }

    // 查看 Governor 统计快照
    let stats = adapter.stats().await;
    println!("\n📊 Governor 统计快照:");
    println!("  {:?}", stats);

    println!("\n💡 默认配置: global=1000/min, ip=100/min, user=200/min");
    println!("→ 5 次请求远低于限额，全部通过");
    println!("→ 超过限额后 check_rate_limit 将返回 false");

    println!("\n✅ 示例完成");
    Ok(())
}
