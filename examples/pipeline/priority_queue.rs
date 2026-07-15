// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use vecboost::domain::EmbedRequest;
use vecboost::pipeline::{Priority, PriorityRequestQueue, QueuedRequest, RequestSource};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 优先级队列入队与出队示例");
    println!("============================\n");

    let queue = PriorityRequestQueue::new(100);

    let requests = [
        ("req-low", Priority::Low),
        ("req-critical", Priority::Critical),
        ("req-normal", Priority::Normal),
    ];

    for (id, priority) in &requests {
        let (tx, _rx) = oneshot::channel();
        let request = QueuedRequest {
            request_id: id.to_string(),
            embed_request: EmbedRequest {
                text: format!("text for {}", id),
                normalize: Some(true),
            },
            priority: *priority,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            source: RequestSource::http("127.0.0.1".to_string()),
            response_tx: tx,
        };
        queue.enqueue(request).await?;
        println!("  📥 入队: {} (priority={:?})", id, priority);
    }

    println!("\n队列大小: {}", queue.size());

    println!("\n📤 出队顺序 (应按 Critical → Normal → Low):");
    for _ in 0..3 {
        if let Some(req) = queue.dequeue().await {
            println!(
                "  📤 出队: {} (priority={:?})",
                req.request_id, req.priority
            );
        }
    }

    println!("\n出队后队列大小: {}", queue.size());
    println!("\n✅ 优先级队列示例完成");
    Ok(())
}
