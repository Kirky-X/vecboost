// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod candle_engine;

#[cfg(feature = "onnx")]
pub mod onnx_engine;

use crate::error::AppError;
use async_trait::async_trait;

/// 推理引擎抽象接口
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// 执行推理，返回未归一化的向量
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError>;

    /// 批量推理
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;
}
