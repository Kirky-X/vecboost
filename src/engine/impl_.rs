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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig};
    use std::path::PathBuf;

    fn test_config_candle() -> ModelConfig {
        ModelConfig {
            name: "test-any-candle".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/nonexistent/model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(768),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        }
    }

    #[cfg(feature = "onnx")]
    fn test_config_onnx() -> ModelConfig {
        ModelConfig {
            name: "test-any-onnx".to_string(),
            engine_type: EngineType::Onnx,
            model_path: PathBuf::from("/nonexistent/onnx-model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        }
    }

    /// 验证 AnyEngine::new 在 Candle 引擎 + 不存在路径时返回错误
    #[test]
    fn test_any_engine_new_candle_missing_model() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError, got {:?}",
                e
            );
        }
    }

    /// 验证 AnyEngine::new 在 Candle 引擎 + 不存在的非目录路径时返回错误
    #[test]
    fn test_any_engine_new_candle_nonexistent_path() {
        let config = test_config_candle();
        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    /// 验证 AnyEngine::new 在不同 Precision 下都返回错误(覆盖精度路径)
    #[test]
    fn test_any_engine_new_candle_all_precisions_fail() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();

        let precisions = [Precision::Fp32, Precision::Fp16, Precision::Int8];
        for (idx, precision) in precisions.iter().enumerate() {
            let result = AnyEngine::new(&config, EngineType::Candle, precision.clone());
            assert!(
                result.is_err(),
                "AnyEngine::new with Candle should fail for precision at index {}",
                idx
            );
        }
    }

    /// 验证 AnyEngine::new 在 Onnx 引擎 + 空目录时返回错误
    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_missing_model() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                matches!(e, VecboostError::ModelLoadError(_)),
                "Expected ModelLoadError, got {:?}",
                e
            );
        }
    }

    /// 验证 AnyEngine::new 在 Onnx 引擎 + 不存在路径时返回错误
    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_nonexistent_path() {
        let config = test_config_onnx();
        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    /// 验证 `EngineType::Candle` 的 Display 实现
    #[test]
    fn test_engine_type_candle_display() {
        assert_eq!(EngineType::Candle.to_string(), "candle");
    }

    /// 验证 `EngineType::Onnx` 的 Display 实现
    #[cfg(feature = "onnx")]
    #[test]
    fn test_engine_type_onnx_display() {
        assert_eq!(EngineType::Onnx.to_string(), "onnx");
    }

    /// 验证 `Precision` 的 Display 实现
    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::Fp32.to_string(), "fp32");
        assert_eq!(Precision::Fp16.to_string(), "fp16");
        assert_eq!(Precision::Int8.to_string(), "int8");
    }
}
