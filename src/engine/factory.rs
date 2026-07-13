// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 引擎工厂：根据 `EngineType` 创建对应的推理引擎实例

use super::AnyEngine;
use crate::config::model::{EngineType, ModelConfig, Precision};
use crate::error::VecboostError;

/// 引擎工厂，根据配置创建推理引擎实例
pub struct EngineFactory;

impl EngineFactory {
    /// 根据引擎类型和模型配置创建引擎实例
    ///
    /// # Arguments
    /// * `engine_type` - 引擎类型枚举
    /// * `config` - 模型配置
    ///
    /// # Errors
    /// - `VecboostError::ConfigError`: 引擎类型对应的 feature 未启用
    /// - `VecboostError::InferenceError`: 引擎运行时不可用（stub 模式）
    /// - `VecboostError::ModelLoadError`: 模型加载失败
    pub fn create(
        engine_type: EngineType,
        config: &ModelConfig,
    ) -> Result<AnyEngine, VecboostError> {
        let precision = Precision::Fp32;
        AnyEngine::new(config, engine_type, precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, ModelConfig};
    use std::path::PathBuf;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "test-factory".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/nonexistent/model"),
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

    #[test]
    fn test_create_candle_returns_error_for_missing_model() {
        let config = test_config();
        let result = EngineFactory::create(EngineType::Candle, &config);
        // Candle 引擎会因模型路径不存在而返回错误
        assert!(result.is_err());
    }

    #[cfg(feature = "tensorrt")]
    #[test]
    fn test_create_tensorrt_returns_runtime_error() {
        let config = test_config();
        let result = EngineFactory::create(EngineType::TensorRt, &config);
        match result {
            Err(VecboostError::InferenceError(msg)) => {
                assert!(msg.contains("TensorRT runtime not available"));
            }
            Err(other) => panic!("Expected InferenceError, got {:?}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn test_create_openvino_returns_runtime_error() {
        let config = test_config();
        let result = EngineFactory::create(EngineType::OpenVino, &config);
        match result {
            Err(VecboostError::InferenceError(msg)) => {
                assert!(msg.contains("OpenVINO runtime not available"));
            }
            Err(other) => panic!("Expected InferenceError, got {:?}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_engine_type_display() {
        assert_eq!(EngineType::Candle.to_string(), "candle");
    }
}
