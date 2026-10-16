// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 集成测试入口
//!
//! 该文件是 `cargo test --test integration` 的入口点，
//! 声明集成测试用到的所有模块。

mod common;

#[path = "integration/api_test.rs"]
mod api_test;

// detect/translate 两模块的集成测试锚定 M1 safetensors 服务器主模型
//（candle 路径专属格式，无 onnx embedding bundle）：onnx-primary 下
// candle 转 opt-in，无 candle 的构建下两模块整体为空（整文件门先例
// tests/scenario_sdk.rs；translate 配对用例的 onnx opus-mt bundle 在
// candle 门内经既有 onnx 判据 SKIP，不冲突）
#[cfg(feature = "candle")]
#[path = "integration/detect_test.rs"]
mod detect_test;

#[cfg(feature = "candle")]
#[path = "integration/translate_test.rs"]
mod translate_test;
