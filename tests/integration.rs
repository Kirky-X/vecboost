// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 集成测试入口
//!
//! 该文件是 `cargo test --test integration` 的入口点，
//! 声明集成测试用到的所有模块。

mod common;

#[path = "integration/api_test.rs"]
mod api_test;
