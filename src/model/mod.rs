// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// 模型层子树（loader/manager/recovery/heat）的消费者按 feature
// （onnx/candle/quantized-gguf）分叉；默认构建下部分分支无消费者，
// 逐项 cfg_attr 成本高于收益，故保留子树级豁免（子模块不再各自重复声明）。
#![allow(unused, reason = "model subtree is engine-feature gated")]

pub(crate) mod heat;
pub(crate) mod loader;
pub(crate) mod manager;
pub(crate) mod recovery;
