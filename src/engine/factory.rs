// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 引擎工厂：根据 `EngineType` 创建对应的推理引擎实例

use super::AnyEngine;
use crate::config::model::{EngineType, ModelConfig, Precision};
use crate::error::VecboostError;
use std::path::Path;

/// 量化路由决策（常编译）：`.gguf` 后缀 + `quantized=true` → 量化引擎。
/// 矩阵单测见 `quantized_engine`（feature 门内）与下方 factory 测试。
pub fn should_use_quantized_engine(model_path: &Path, quantized: bool) -> bool {
    if !quantized {
        return false;
    }
    model_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

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
        // loader 路由：.gguf + quantized=true 走量化引擎
        // （加载期反量化桥），否则维持 safetensors 路径。
        if should_use_quantized_engine(&config.model_path, config.quantized) {
            #[cfg(feature = "quantized-gguf")]
            {
                let engine =
                    super::quantized_engine::QuantizedCandleEngine::load(&config.model_path)?;
                return Ok(AnyEngine::Quantized(engine));
            }
            #[cfg(not(feature = "quantized-gguf"))]
            {
                return Err(VecboostError::ConfigError(
                    "检测到 GGUF 量化模型配置（.gguf + quantized=true），但本次构建未启用 \
                     `quantized-gguf` feature；请以 `--features quantized-gguf` 重新构建"
                        .to_string(),
                ));
            }
        }
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
            quantized: false,
        }
    }

    #[test]
    fn test_create_candle_returns_error_for_missing_model() {
        let config = test_config();
        let result = EngineFactory::create(EngineType::Candle, &config);
        // Candle 引擎会因模型路径不存在而返回错误
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_routing_matrix() {
        // (path, quantized, expected)
        let cases = [
            ("model.gguf", true, true),
            ("model.GGUF", true, true),
            ("model.safetensors", true, false),
            ("model.gguf", false, false),
            ("model", true, false),
            ("model.safetensors", false, false),
        ];
        for (p, q, expected) in cases {
            assert_eq!(
                should_use_quantized_engine(&PathBuf::from(p), q),
                expected,
                "路由矩阵: path={} quantized={}",
                p,
                q
            );
        }
    }

    #[test]
    fn test_create_gguf_without_feature_returns_config_error() {
        let mut config = test_config();
        config.model_path = PathBuf::from("/tmp/model-q8_0.gguf");
        config.quantized = true;
        let result = EngineFactory::create(EngineType::Candle, &config);
        assert!(result.is_err());
        #[cfg(not(feature = "quantized-gguf"))]
        match result {
            Err(VecboostError::ConfigError(msg)) => {
                assert!(
                    msg.contains("quantized-gguf"),
                    "应提示启用 feature: {}",
                    msg
                );
            }
            Err(other) => panic!("期望 ConfigError，实际 {:?}", other),
            Ok(_) => panic!("期望 GGUF 配置无 feature 时失败"),
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_create_onnx_engine_missing_model() {
        let mut config = test_config();
        config.engine_type = EngineType::Onnx;
        config.model_path = PathBuf::from("/nonexistent/onnx/model");
        let result = EngineFactory::create(EngineType::Onnx, &config);
        // ONNX 引擎会因模型路径不存在而返回错误
        assert!(
            result.is_err(),
            "ONNX engine should return error for missing model path"
        );
    }

    #[test]
    fn test_engine_type_display() {
        assert_eq!(EngineType::Candle.to_string(), "candle");
    }

    #[test]
    fn test_removed_engine_types_return_error() {
        let json = "\"tensorrt\"";
        let result: Result<EngineType, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "tensorrt should not deserialize after removal"
        );

        let json = "\"openvino\"";
        let result: Result<EngineType, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "openvino should not deserialize after removal"
        );
    }
}
