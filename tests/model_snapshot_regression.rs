// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 本地模型快照回归测试
//!
//! 对 models/ 下 4 个本地模型(缺席自动 SKIP)各取固定输入断言:
//! - 输出维度匹配 expected_dimension
//! - L2 范数 ≈ 1.0(归一化向量)
//! - 同输入两次调用结果逐位一致(确定性)

use std::path::PathBuf;
use vecboost::config::model::{DeviceType, EngineType, ModelConfig, Precision};
use vecboost::engine::{AnyEngine, InferenceEngine};
/// 模型快照测试参数: (模型名, 路径, 期望维度)
struct ModelSnapshot {
    name: &'static str,
    path: &'static str,
    expected_dim: usize,
}

const SNAPSHOTS: &[ModelSnapshot] = &[
    ModelSnapshot {
        name: "all-MiniLM-L6-v2",
        path: "models/all-MiniLM-L6-v2",
        expected_dim: 384,
    },
    ModelSnapshot {
        name: "BAAI-bge-small-en-v1.5",
        path: "models/BAAI-bge-small-en-v1.5",
        expected_dim: 384,
    },
    ModelSnapshot {
        name: "BAAI-bge-small-zh-v1.5",
        path: "models/BAAI-bge-small-zh-v1.5",
        expected_dim: 512,
    },
    ModelSnapshot {
        name: "multilingual-e5-small",
        path: "models/multilingual-e5-small",
        expected_dim: 384,
    },
];

fn model_config(snapshot: &ModelSnapshot) -> ModelConfig {
    ModelConfig {
        name: snapshot.name.to_string(),
        engine_type: EngineType::Candle,
        model_path: PathBuf::from(snapshot.path),
        tokenizer_path: None,
        device: DeviceType::Cpu,
        max_batch_size: 4,
        pooling_mode: None, // Auto
        expected_dimension: Some(snapshot.expected_dim),
        memory_limit_bytes: None,
        oom_fallback_enabled: true,
        model_sha256: None,
        quantized: false,
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn model_snapshot_regression() {
    for snap in SNAPSHOTS {
        let model_dir = PathBuf::from(snap.path);
        if !model_dir.join("config.json").exists() {
            eprintln!("SKIP {}: model files not present", snap.name);
            continue;
        }

        let config = model_config(snap);
        let engine = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32)
            .unwrap_or_else(|e| panic!("Failed to load {}: {}", snap.name, e));

        let input = "The quick brown fox jumps over the lazy dog.";

        let vec1 = engine
            .embed(input)
            .unwrap_or_else(|e| panic!("embed failed for {}: {}", snap.name, e));

        assert_eq!(
            vec1.len(),
            snap.expected_dim,
            "{}: dimension mismatch: got {} expected {}",
            snap.name,
            vec1.len(),
            snap.expected_dim
        );

        let norm = l2_norm(&vec1);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{}: L2 norm should be ~1.0, got {}",
            snap.name,
            norm
        );

        let vec2 = engine
            .embed(input)
            .unwrap_or_else(|e| panic!("embed failed for {}: {}", snap.name, e));

        assert_eq!(
            vec1.len(),
            vec2.len(),
            "{}: length mismatch between runs",
            snap.name
        );
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "{}: non-deterministic at index {}: {} vs {}",
                snap.name,
                i,
                a,
                b
            );
        }

        eprintln!(
            "PASS {}: dim={} norm={:.6} deterministic=ok",
            snap.name,
            vec1.len(),
            norm
        );
    }
}
