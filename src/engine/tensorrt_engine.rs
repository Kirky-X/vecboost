// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::InferenceEngine;
use crate::config::model::{ModelConfig, Precision};
use crate::error::VecboostError;
use async_trait::async_trait;

const RUNTIME_UNAVAILABLE: &str = "TensorRT runtime not available";

/// TensorRT 推理引擎 (stub 实现)
///
/// 由于 TensorRT SDK 需要原生 CUDA 环境与 TensorRT 运行时库，
/// 此实现为 feature-gated stub。当 `tensorrt` feature 启用但
/// TensorRT 运行时不可用时，所有操作返回错误。
pub(crate) struct TensorRtEngine {
    precision: Precision,
    fallback_triggered: bool,
}

impl TensorRtEngine {
    /// 创建 TensorRT 引擎实例
    ///
    /// # Errors
    /// 当前为 stub 实现，始终返回 `VecboostError::InferenceError`，
    /// 提示 TensorRT 运行时不可用。
    pub fn new(_config: &ModelConfig, _precision: Precision) -> Result<Self, VecboostError> {
        Err(VecboostError::InferenceError(
            RUNTIME_UNAVAILABLE.to_string(),
        ))
    }

    /// 测试用构造函数，绕过运行时检查创建 stub 实例
    #[cfg(test)]
    fn new_for_test(precision: Precision) -> Self {
        Self {
            precision,
            fallback_triggered: false,
        }
    }
}

#[async_trait]
impl InferenceEngine for TensorRtEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Err(VecboostError::InferenceError(
            RUNTIME_UNAVAILABLE.to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Err(VecboostError::InferenceError(
            RUNTIME_UNAVAILABLE.to_string(),
        ))
    }

    fn precision(&self) -> &Precision {
        &self.precision
    }

    fn supports_mixed_precision(&self) -> bool {
        true
    }

    fn is_fallback_triggered(&self) -> bool {
        self.fallback_triggered
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Err(VecboostError::InferenceError(
            "TensorRT engine does not support CPU fallback".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig};
    use std::path::PathBuf;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-tensorrt".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/models/test"),
            tokenizer_path: None,
            device: DeviceType::Cuda,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        }
    }

    #[test]
    fn test_new_returns_runtime_unavailable_error() {
        let config = test_config();
        let result = TensorRtEngine::new(&config, Precision::Fp32);
        match result {
            Err(VecboostError::InferenceError(msg)) => {
                assert!(msg.contains(RUNTIME_UNAVAILABLE));
            }
            Err(other) => panic!("Expected InferenceError, got {:?}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_new_with_fp16_precision_returns_error() {
        let config = test_config();
        let result = TensorRtEngine::new(&config, Precision::Fp16);
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_returns_runtime_unavailable_error() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp32);
        let result = engine.embed("Hello");
        match result {
            Err(VecboostError::InferenceError(msg)) => {
                assert!(msg.contains(RUNTIME_UNAVAILABLE));
            }
            Err(other) => panic!("Expected InferenceError, got {:?}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_embed_batch_returns_error() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp32);
        let texts: Vec<String> = (0..8).map(|i| format!("text {}", i)).collect();
        let result = engine.embed_batch(&texts);
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_batch_empty_returns_error() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp32);
        let result = engine.embed_batch(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_precision_returns_configured_value() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp16);
        assert_eq!(*engine.precision(), Precision::Fp16);
    }

    #[test]
    fn test_supports_mixed_precision() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp32);
        assert!(engine.supports_mixed_precision());
    }

    #[test]
    fn test_is_fallback_triggered_default_false() {
        let engine = TensorRtEngine::new_for_test(Precision::Fp32);
        assert!(!engine.is_fallback_triggered());
    }

    #[tokio::test]
    async fn test_try_fallback_to_cpu_returns_error() {
        let mut engine = TensorRtEngine::new_for_test(Precision::Fp32);
        let config = test_config();
        let result = engine.try_fallback_to_cpu(&config).await;
        assert!(result.is_err());
    }
}
