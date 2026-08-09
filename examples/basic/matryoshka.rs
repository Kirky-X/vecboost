// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Matryoshka 维度约简示例 — 演示任务自适应截断与信息保留率
//!
//! Matryoshka embedding 允许将高维向量截断到较低维度，同时保持语义信息。
//! 截断后必须重新 L2 归一化，以保证余弦相似度计算正确。
//!
//! 核心公共 API：
//! - `TaskType` / `recommended_dimension`: 根据下游任务推荐维度
//! - `truncate_vector`: 截断向量到目标维度
//! - `normalize_l2`: L2 归一化（截断后必须调用）
//! - `information_retention_rate`: 计算截断后的信息保留率
//!
//! 运行: cargo run -p vecboost-examples --bin matryoshka

use vecboost::{TaskType, information_retention_rate, recommended_dimension};
use vecboost::utils::{normalize_l2, truncate_vector};

fn main() {
    println!("🪆 Matryoshka 维度约简示例");
    println!("==========================\n");

    // 模拟一个 1024 维的 embedding（来自 Matryoshka 模型）
    let full_dimension = 1024;
    let embedding: Vec<f32> = (0..full_dimension)
        .map(|i| ((i as f32) * 0.01).sin() + 0.5)
        .collect();
    println!("📝 原始 embedding 维度: {}", embedding.len());

    // 1. 根据任务类型推荐维度
    println!("\n📊 任务自适应维度推荐:");
    let tasks = [
        TaskType::Retrieval,
        TaskType::SemanticSearch,
        TaskType::Clustering,
        TaskType::Classification,
    ];
    for task in &tasks {
        let dim = recommended_dimension(*task, full_dimension);
        println!("  {:?} → 推荐 {} 维", task, dim);
    }

    // 2. 截断到不同维度并计算信息保留率
    println!("\n📊 截断与信息保留率分析:");
    let target_dims = [128, 256, 512, 768];
    for &target in &target_dims {
        let truncated = truncate_vector(&embedding, target);
        let rate = information_retention_rate(&embedding, target);
        println!(
            "  {} 维: 信息保留率 = {:.2}%",
            truncated.len(),
            rate * 100.0
        );
    }

    // 3. 截断 + L2 归一化（关键步骤！）
    println!("\n📝 截断到 256 维 + L2 归一化:");
    let target_dim = recommended_dimension(TaskType::Clustering, full_dimension);
    let mut truncated = truncate_vector(&embedding, target_dim);
    println!("  截断后维度: {}", truncated.len());

    // 计算截断前的 L2 范数
    let original_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let truncated_norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  截断前 L2 范数: {:.4}", original_norm);
    println!("  截断后 L2 范数: {:.4}（需要重新归一化！）", truncated_norm);

    // L2 归一化 — 截断后必须执行，否则余弦相似度计算不正确
    normalize_l2(&mut truncated);
    let normalized_norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  归一化后 L2 范数: {:.4}", normalized_norm);

    // 4. 验证归一化后余弦相似度正确性
    println!("\n📝 验证余弦相似度:");
    let mut other = truncate_vector(
        &(0..full_dimension)
            .map(|i| ((i as f32) * 0.02).cos() + 0.3)
            .collect::<Vec<f32>>(),
        target_dim,
    );
    normalize_l2(&mut other);

    // 两个归一化向量的点积 = 余弦相似度
    let dot: f32 = truncated.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
    println!("  归一化后点积（= 余弦相似度）: {:.4}", dot);
    println!("  值域 [-1, 1]，越接近 1 越相似");

    // 5. 维度约简的存储节省
    println!("\n📊 存储节省对比:");
    let full_bytes = full_dimension * 4; // f32 = 4 bytes
    let reduced_bytes = target_dim * 4;
    println!(
        "  原始 {} 维: {} bytes/向量",
        full_dimension, full_bytes
    );
    println!(
        "  约简 {} 维: {} bytes/向量（节省 {}%）",
        target_dim,
        reduced_bytes,
        ((1.0 - reduced_bytes as f32 / full_bytes as f32) * 100.0) as u32
    );

    println!("\n✅ Matryoshka 维度约简示例完成");
}
