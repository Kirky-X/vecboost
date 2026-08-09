// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! GPU 显存分页管理示例 — 演示 WeightPagingManager 的层换入换出与 LRU-K 驱逐
//!
//! WeightPagingManager 管理模型权重在 GPU/CPU 之间的分层调度：
//! - 热层常驻 GPU
//! - 冷层按需从 CPU 内存换入/换出
//! - 支持预取优化：根据当前推理层预测后续层，提前换入
//!
//! 运行: cargo run -p vecboost-examples --bin memory_paging

use vecboost::{PagingConfig, WeightPagingManager};

fn main() {
    println!("📦 GPU 显存分页管理（WeightPagingManager）示例");
    println!("================================================\n");

    // 1. 创建分页管理器：GPU 显存预算 1000 字节
    let config = PagingConfig {
        enabled: true,
        gpu_memory_budget_bytes: 1000,
        lru_k: 2,
        prefetch_depth: 2,
    };
    let mut manager = WeightPagingManager::new(&config);

    println!("✅ WeightPagingManager 已创建");
    println!("  GPU 显存预算: 1000 bytes");
    println!("  LRU-K: 2");
    println!("  预取深度: 2\n");

    // 2. 注册模型层（初始在 CPU 上）
    println!("📝 注册 5 个模型层...");
    manager.register_layer("embedding", 200);
    manager.register_layer("layer_0", 250);
    manager.register_layer("layer_1", 250);
    manager.register_layer("layer_2", 200);
    manager.register_layer("head", 100);

    for name in &["embedding", "layer_0", "layer_1", "layer_2", "head"] {
        println!("  {} on GPU: {}", name, manager.is_on_gpu(name));
    }
    println!();

    // 3. 换入热层到 GPU
    println!("📝 换入 embedding + layer_0 到 GPU...");
    manager.page_in("embedding").unwrap();
    manager.page_in("layer_0").unwrap();
    println!("  GPU 使用: {} / {} bytes", manager.gpu_usage(), manager.gpu_budget());
    println!("  embedding on GPU: {}", manager.is_on_gpu("embedding"));
    println!("  layer_0 on GPU: {}\n", manager.is_on_gpu("layer_0"));

    // 4. 记录访问（影响 LRU-K 驱逐优先级）
    println!("📝 模拟推理访问（记录 layer_0 频繁访问）...");
    for _ in 0..5 {
        manager.record_access("layer_0");
    }
    manager.record_access("embedding");
    println!("  layer_0 访问 5 次（热层）");
    println!("  embedding 访问 1 次\n");

    // 5. 显存不足时自动驱逐冷层
    println!("📝 尝试换入 layer_1 (250 bytes)...");
    println!("  当前 GPU 使用: {} bytes", manager.gpu_usage());
    manager.page_in("layer_1").unwrap();
    println!("  换入成功！GPU 使用: {} bytes", manager.gpu_usage());

    println!("\n📝 尝试换入 layer_2 (200 bytes) — 需要驱逐...");
    println!("  当前 GPU 使用: {} bytes (预算 1000)", manager.gpu_usage());
    manager.page_in("layer_2").unwrap();
    println!("  换入成功！GPU 使用: {} bytes", manager.gpu_usage());
    println!("  embedding on GPU: {} (可能被驱逐)", manager.is_on_gpu("embedding"));
    println!("  layer_0 on GPU: {} (频繁访问，保留)", manager.is_on_gpu("layer_0"));

    // 6. 查看驱逐候选
    if let Some(victim) = manager.evict_candidate() {
        println!("\n📊 当前驱逐候选: {}", victim);
    }

    // 7. 预取演示
    println!("\n📝 预取演示:");
    let layer_order = vec![
        "embedding".to_string(),
        "layer_0".to_string(),
        "layer_1".to_string(),
        "layer_2".to_string(),
        "head".to_string(),
    ];
    let prefetch_list = manager.get_prefetch_list("layer_0", &layer_order);
    println!("  当前层: layer_0");
    println!("  预取列表: {:?}", prefetch_list);
    manager.prefetch(&prefetch_list);
    println!("  已标记为预取（InTransfer 状态）");

    // 8. 手动换出
    println!("\n📝 手动换出 layer_1...");
    manager.page_out("layer_1").unwrap();
    println!("  layer_1 on GPU: {}", manager.is_on_gpu("layer_1"));

    // 9. 查看统计
    let stats = manager.stats();
    println!("\n📊 分页统计:");
    println!("  GPU 使用: {} bytes", stats.gpu_usage_bytes);
    println!("  CPU 备份: {} 层", stats.cpu_backup_count);
    println!("  换入次数: {}", stats.page_in_count);
    println!("  换出次数: {}", stats.page_out_count);

    println!("\n✅ GPU 显存分页管理示例完成");
}
