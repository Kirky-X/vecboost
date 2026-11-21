// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

/// 原地 L2 归一化（零/近零向量静默跳过）。
///
/// 引擎出口契约要求归一化输出为单位向量（onnx 与 candle 两引擎共用）。
/// 范数 ≤ 1e-12 时静默跳过，向量保持原值；否则逐元素除以范数。
/// 决策 logits 绝不经过此函数（决策管线红线）。
pub(crate) fn l2_normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
