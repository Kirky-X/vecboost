// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 语义缓存示例 — 演示 SemanticCache 的三级查询策略
//!
//! 查询路径：精确匹配 → trigram 语义搜索 → 模型推理回填。
//! 核心价值：精确 miss 后、模型推理前，插入一层零开销的文本相似度检查。
//!
//! 运行: cargo run -p vecboost-examples --bin semantic_cache_demo

use vecboost::SemanticCache;

#[tokio::main]
async fn main() {
    println!("📦 语义缓存（SemanticCache）示例");
    println!("====================================\n");

    // 1. 创建语义缓存：相似度阈值 0.5，容量 100
    let cache = SemanticCache::with_capacity(0.5, 100);
    println!("✅ SemanticCache 已创建");
    println!("  启用: {}", cache.is_enabled());
    println!("  相似度阈值: 0.5");
    println!("  容量: 100\n");

    // 2. 模拟模型推理回调（实际场景中调用推理引擎）
    let compute = |text: &str| {
        let text = text.to_string();
        async move {
            println!("  🔧 [模型推理] 计算 \"{}\" 的嵌入向量...", text);
            // 模拟推理延迟
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            // 返回模拟向量
            Ok(vec![1.0, 2.0, 3.0])
        }
    };

    // 3. 首次查询 — 应 miss 并触发模型推理
    println!("📝 查询 1: \"机器学习是什么\"");
    let result = cache
        .get_or_compute("机器学习是什么", || compute("机器学习是什么"))
        .await
        .unwrap();
    println!("  结果: {:?}\n", result);

    // 4. 精确重复查询 — 应精确命中缓存
    println!("📝 查询 2: \"机器学习是什么\"（重复查询）");
    let result = cache
        .get_or_compute("机器学习是什么", || compute("机器学习是什么"))
        .await
        .unwrap();
    println!("  结果: {:?}\n", result);

    // 5. 近似文本查询 — 应语义命中（trigram Jaccard 相似度 > 0.5）
    println!("📝 查询 3: \"机器学习是什么呢\"（近似改写）");
    let result = cache
        .get_or_compute("机器学习是什么呢", || compute("机器学习是什么呢"))
        .await
        .unwrap();
    println!("  结果: {:?}\n", result);

    // 6. 完全不同文本 — 应 miss
    println!("📝 查询 4: \"今天天气怎么样\"（完全不同）");
    let result = cache
        .get_or_compute("今天天气怎么样", || compute("今天天气怎么样"))
        .await
        .unwrap();
    println!("  结果: {:?}\n", result);

    // 7. 查看统计信息
    let stats = cache.stats();
    println!("📊 缓存统计:");
    println!("  精确命中: {}", stats.exact_hits);
    println!("  语义命中: {}", stats.semantic_hits);
    println!("  未命中: {}", stats.misses);
    println!("  总条目数: {}", stats.total_entries);

    // 8. 演示 disabled 缓存
    println!("\n📝 禁用状态的 SemanticCache:");
    let disabled = SemanticCache::disabled();
    println!("  启用: {}", disabled.is_enabled());
    let result = disabled
        .get_or_compute("any text", || async { Ok(vec![42.0]) })
        .await
        .unwrap();
    println!("  直接计算结果: {:?}", result);

    println!("\n✅ 语义缓存示例完成");
}
