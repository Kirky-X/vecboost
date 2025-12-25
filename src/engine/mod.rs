// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod candle_engine;

#[cfg(feature = "onnx")]
pub mod onnx_engine;

use crate::config::model::{EngineType, ModelConfig, Precision};
use crate::error::AppError;
use async_trait::async_trait;

/// 推理引擎抽象接口
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// 执行推理，返回未归一化的向量
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError>;

    /// 批量推理
    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;

    /// 获取当前精度设置
    fn precision(&self) -> Precision;

    /// 检查是否支持混合精度
    fn supports_mixed_precision(&self) -> bool;
}

pub enum AnyEngine {
    Candle(candle_engine::CandleEngine),
    #[cfg(feature = "onnx")]
    Onnx(onnx_engine::OnnxEngine),
}

impl AnyEngine {
    pub fn new(config: &ModelConfig, engine_type: EngineType, precision: Precision) -> Result<Self, AppError> {
        match engine_type {
            EngineType::Candle => Ok(AnyEngine::Candle(candle_engine::CandleEngine::new(config, precision)?)),
            #[cfg(feature = "onnx")]
            EngineType::Onnx => Ok(AnyEngine::Onnx(onnx_engine::OnnxEngine::new(config, precision)?)),
        }
    }
}

impl InferenceEngine for AnyEngine {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>, AppError> {
        match self {
            AnyEngine::Candle(engine) => engine.embed(text),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed(text),
        }
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        match self {
            AnyEngine::Candle(engine) => engine.embed_batch(texts),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed_batch(texts),
        }
    }

    fn precision(&self) -> Precision {
        match self {
            AnyEngine::Candle(engine) => engine.precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.precision(),
        }
    }

    fn supports_mixed_precision(&self) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.supports_mixed_precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.supports_mixed_precision(),
        }
    }
}
