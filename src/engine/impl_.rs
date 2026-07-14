// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! AnyEngine 的实现块

use super::{AnyEngine, InferenceEngine};
use crate::config::model::{EngineType, ModelConfig, Precision};
use crate::error::VecboostError;
use async_trait::async_trait;

impl AnyEngine {
    pub fn new(
        config: &ModelConfig,
        engine_type: EngineType,
        precision: Precision,
    ) -> Result<Self, VecboostError> {
        match engine_type {
            EngineType::Candle => Ok(AnyEngine::Candle(super::candle_engine::CandleEngine::new(
                config, precision,
            )?)),
            #[cfg(feature = "onnx")]
            EngineType::Onnx => Ok(AnyEngine::Onnx(super::onnx_engine::OnnxEngine::new(
                config, precision,
            )?)),
        }
    }
}

#[async_trait]
impl InferenceEngine for AnyEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.embed(text),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed(text),
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.embed_batch(texts),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed_batch(texts),
        }
    }

    fn precision(&self) -> &Precision {
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

    fn is_fallback_triggered(&self) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.is_fallback_triggered(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.is_fallback_triggered(),
        }
    }

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.try_fallback_to_cpu(config).await,
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.try_fallback_to_cpu(config).await,
        }
    }
}
