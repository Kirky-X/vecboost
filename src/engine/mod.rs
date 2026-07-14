// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

pub(crate) mod candle_engine;
pub mod factory;
pub use factory::EngineFactory;
pub(crate) mod impl_;

#[cfg(feature = "onnx")]
pub(crate) mod onnx_engine;

use crate::config::model::{ModelConfig, Precision};
use crate::error::VecboostError;
use async_trait::async_trait;

/// 推理引擎抽象接口
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// 执行推理，返回未归一化的向量
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError>;

    /// 批量推理
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError>;

    /// 获取当前精度设置
    fn precision(&self) -> &Precision;

    /// 检查是否支持混合精度
    fn supports_mixed_precision(&self) -> bool;

    /// 检查是否已触发降级
    fn is_fallback_triggered(&self) -> bool {
        false
    }

    /// 尝试降级到 CPU（在 OOM 时调用）
    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError>;
}

#[allow(clippy::large_enum_variant)]
pub enum AnyEngine {
    Candle(candle_engine::CandleEngine),
    #[cfg(feature = "onnx")]
    Onnx(onnx_engine::OnnxEngine),
}
