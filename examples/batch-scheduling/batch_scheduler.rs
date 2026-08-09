// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 动态批量调度示例 — 演示 DynamicBatchScheduler 的自适应批量调整
//!
//! DynamicBatchScheduler 基于 P99 延迟和吞吐量自动调整批量大小，
//! 支持优先级请求、并发控制和性能统计。
//!
//! 运行: cargo run -p vecboost-examples --bin batch_scheduler

use std::time::Instant;

use vecboost::{BatchConfig, BatchPriority, BatchRequest, DynamicBatchScheduler};

#[tokio::main]
async fn main() {
    println!("📦 动态批量调度（DynamicBatchScheduler）示例");
    println!("=============================================\n");

    // 1. 创建调度器配置
    let config = BatchConfig {
        min_batch_size: 4,
        max_batch_size: 32,
        max_wait_time_ms: 50,
        max_concurrent_batches: 4,
        enable_dynamic_adjustment: true,
    };
    let scheduler = DynamicBatchScheduler::new(config);

    println!("✅ DynamicBatchScheduler 已创建");
    println!("  最小批量: 4");
    println!("  最大批量: 32");
    println!("  最大等待: 50ms");
    println!("  初始批量: {}\n", scheduler.current_batch_size().await);

    // 2. 提交多个请求
    println!("📝 提交 8 个请求...");
    for i in 0..8 {
        let priority = if i < 2 {
            BatchPriority::High
        } else if i < 5 {
            BatchPriority::Normal
        } else {
            BatchPriority::Low
        };
        let request = BatchRequest {
            request_id: format!("req-{}", i),
            data: vec![format!("text-{}", i)],
            priority,
            submitted_at: Instant::now(),
        };
        scheduler.submit_request(request).await.unwrap();
    }
    println!("  队列大小: {}\n", scheduler.queue_size().await);

    // 3. 等待超过 max_wait_time_ms 后获取批次
    println!("📝 等待 60ms 后收集批次...");
    tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;

    if let Some(batch) = scheduler.try_get_batch().await {
        println!("  批次大小: {}", batch.requests.len());
        // 模拟推理处理
        let latency = 20.0;
        scheduler
            .record_batch_completion(batch.requests.len(), latency)
            .await;
        println!("  已记录完成 (延迟: {:.1}ms)\n", latency);
    }

    // 4. 演示动态调整：记录多批高性能批次
    println!("📝 记录 15 批高性能批次 (batch=8, latency=20ms)...");
    for _ in 0..15 {
        scheduler.record_batch_completion(8, 20.0).await;
    }
    let new_size = scheduler.current_batch_size().await;
    println!("  调整后批量大小: {} (应增大)\n", new_size);

    // 5. 演示延迟过高时的缩减
    println!("📝 记录高延迟批次 (latency=150ms)...");
    for _ in 0..15 {
        scheduler.record_batch_completion(32, 150.0).await;
    }
    let reduced_size = scheduler.current_batch_size().await;
    println!("  调整后批量大小: {} (应减小)\n", reduced_size);

    // 6. 查看性能统计
    let stats = scheduler.get_performance_stats().await;
    println!("📊 性能统计:");
    println!("  当前批量大小: {}", stats.current_batch_size);
    println!("  队列大小: {}", stats.queue_size);
    println!("  活跃批次: {}", stats.active_batches);
    println!("  已处理批次: {}", stats.total_batches_processed);
    println!("  平均延迟: {:.1}ms", stats.avg_latency_ms);
    println!("  平均吞吐: {:.1} req/s", stats.avg_throughput_req_per_sec);

    // 7. 手动设置批量
    println!("\n📝 手动设置批量大小为 16...");
    scheduler.set_batch_size(16).await;
    println!("  当前批量: {}", scheduler.current_batch_size().await);

    // 8. 清空队列
    println!("\n📝 清空队列...");
    scheduler.clear_queue().await;
    println!("  队列大小: {}", scheduler.queue_size().await);

    println!("\n✅ 动态批量调度示例完成");
}
