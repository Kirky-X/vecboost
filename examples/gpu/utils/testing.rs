use std::time::Instant;

pub struct PerformanceMetrics {
    pub duration_ms: f64,
    pub throughput_tps: f64,
    pub avg_latency_ms: f64,
}

pub fn measure_time<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed().as_secs_f64() * 1000.0;
    (result, duration)
}

pub fn calculate_metrics(duration_ms: f64, num_items: usize) -> PerformanceMetrics {
    PerformanceMetrics {
        duration_ms,
        throughput_tps: num_items as f64 / (duration_ms / 1000.0),
        avg_latency_ms: duration_ms / num_items as f64,
    }
}

pub fn print_metrics(metrics: &PerformanceMetrics, operation: &str) {
    println!("\n{} 性能指标:", operation);
    println!("  总耗时: {:.2} ms", metrics.duration_ms);
    println!("  吞吐量: {:.2} items/s", metrics.throughput_tps);
    println!("  平均延迟: {:.2} ms", metrics.avg_latency_ms);
}

pub fn validate_embedding(embedding: &[f32], expected_dim: usize) -> bool {
    if embedding.len() != expected_dim {
        println!(
            "  ❌ 维度不匹配: 期望 {}, 实际 {}",
            expected_dim,
            embedding.len()
        );
        return false;
    }

    let norm: f32 = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 0.9 || norm > 1.1 {
        println!("  ❌ 向量未归一化: L2范数 = {:.4}", norm);
        return false;
    }

    println!(
        "  ✅ 向量验证通过 (维度: {}, L2范数: {:.4})",
        embedding.len(),
        norm
    );
    true
}

pub fn print_embedding_preview(embedding: &[f32], preview_len: usize) {
    let len = embedding.len().min(preview_len);
    print!("  向量前{}维: [", len);
    for (i, &val) in embedding.iter().take(len).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", val);
    }
    if embedding.len() > len {
        print!(" ...");
    }
    println!("]");
}
