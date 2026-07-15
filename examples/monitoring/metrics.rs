// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use prometheus::Encoder;
use std::time::Duration;
use vecboost::metrics::{InferenceCollector, PrometheusCollector};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Prometheus 指标暴露示例");
    println!("==========================\n");

    let collector = InferenceCollector::new();

    for i in 0..5 {
        collector
            .record_inference(
                "test-model",
                100,
                512,
                Duration::from_millis(50 + i * 10),
                1024 * 1024,
            )
            .await;
    }

    let summary = collector.get_summary().await;
    println!("InferenceCollector 统计:");
    println!("  总推理次数: {}", summary.total_inferences);
    println!("  成功推理: {}", summary.successful_inferences);
    println!("  平均延迟: {:.2} ms", summary.average_latency_ms);
    println!("  样本数: {}", summary.sample_count);

    let prom = PrometheusCollector::new()?;
    prom.record_http_request("POST", "/embed", 200);
    prom.record_http_request("POST", "/embed", 200);
    prom.record_http_request("POST", "/embed", 500);
    prom.update_active_connections("http", 10);
    prom.update_active_connections("grpc", 3);
    prom.record_cache_hit("embedding");
    prom.record_cache_hit("embedding");
    prom.record_cache_miss("embedding");

    let encoder = prometheus::TextEncoder::new();
    let families = prom.registry().gather();
    let mut buffer = Vec::new();
    encoder.encode(&families, &mut buffer)?;
    let output = String::from_utf8(buffer)?;

    println!("\nPrometheus 指标输出 (可被 Prometheus server 抓取):");
    println!("{}", output);

    println!("✅ Prometheus 指标示例完成");
    Ok(())
}
