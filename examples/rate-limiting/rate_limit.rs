// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! limiteron 限流配置示例。
//!
//! 使用默认配置创建 LimiteronAdapter，连续调用 check_rate_limit，
//! 观察 get_status 返回的 remaining 递减。

use vecboost::rate_limit::{LimiteronAdapter, RateLimitDimension};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚦 limiteron 限流示例（默认配置）");
    println!("==================================\n");

    let adapter = LimiteronAdapter::with_default_config();

    let dim = RateLimitDimension::Global;
    let status0 = adapter.get_status(dim.clone()).await;
    println!("初始状态:");
    println!(
        "  维度={}  限额={}  剩余={}  算法={}",
        status0.dimension, status0.max_requests, status0.remaining, status0.algorithm
    );

    println!("\n连续发起 5 次请求：\n");
    for i in 1..=5 {
        let allowed = adapter.check_rate_limit(vec![dim.clone()]).await;
        let status = adapter.get_status(dim.clone()).await;
        println!(
            "  第 {} 次: allowed={}  remaining={}  current_count={}",
            i, allowed, status.remaining, status.current_count
        );
    }

    println!(
        "\n→ 默认全局限额 = {} 次/分钟，remaining 随请求递减",
        status0.max_requests
    );
    println!("→ 超过限额后 check_rate_limit 将返回 false");

    println!("\n✅ 示例完成");
    Ok(())
}
