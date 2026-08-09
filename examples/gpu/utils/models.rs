// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub const DEFAULT_MODEL_PATHS: &[&str] = &[
    "/home/project/vecboost/models/bge-m3-onnx",
    "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    "./model",
];

pub fn get_test_texts() -> Vec<String> {
    vec![
        "这是一个测试文本，用于验证向量嵌入模型的性能。".to_string(),
        "VecBoost是一个高性能的向量嵌入服务框架。".to_string(),
        "GPU加速可以显著提升推理速度。".to_string(),
    ]
}

pub fn get_single_test_text() -> String {
    "这是一个测试文本，用于验证向量嵌入模型的性能。".to_string()
}
