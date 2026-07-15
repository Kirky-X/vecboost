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

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_all_precisions_fail() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();

        let precisions = [Precision::Fp32, Precision::Fp16, Precision::Int8];
        for (idx, precision) in precisions.iter().enumerate() {
            let result = AnyEngine::new(&config, EngineType::Onnx, precision.clone());
            assert!(
                result.is_err(),
                "AnyEngine::new with Onnx should fail for precision at index {}",
                idx
            );
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_cuda_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Cuda;

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

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_amd_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Amd;

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_opencl_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::OpenCL;

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_with_sha256() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.model_sha256 = Some("abcdef1234567890".to_string());

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_with_sha256() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.model_sha256 = Some("abcdef1234567890".to_string());

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_type_candle_serde_roundtrip() {
        let json = serde_json::to_string(&EngineType::Candle).expect("serialize");
        assert_eq!(json, "\"candle\"");
        let deserialized: EngineType = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(deserialized, EngineType::Candle));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_engine_type_onnx_serde_roundtrip() {
        let json = serde_json::to_string(&EngineType::Onnx).expect("serialize");
        assert_eq!(json, "\"onnx\"");
        let deserialized: EngineType = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(deserialized, EngineType::Onnx));
    }

    #[test]
    fn test_precision_clone_and_eq() {
        let p1 = Precision::Fp32;
        let p2 = p1.clone();
        assert_eq!(p1, p2);
        assert_ne!(Precision::Fp32, Precision::Fp16);
        assert_ne!(Precision::Fp16, Precision::Int8);
    }

    #[test]
    fn test_device_type_cuda_and_metal_serde() {
        let cuda_json = serde_json::to_string(&DeviceType::Cuda).expect("serialize Cuda");
        assert_eq!(cuda_json, "\"cuda\"");
        let metal_json = serde_json::to_string(&DeviceType::Metal).expect("serialize Metal");
        assert_eq!(metal_json, "\"metal\"");
    }

    #[test]
    fn test_device_type_amd_serde_roundtrip() {
        let json = serde_json::to_string(&DeviceType::Amd).expect("serialize Amd");
        assert_eq!(json, "\"amd\"");
        let decoded: DeviceType = serde_json::from_str(&json).expect("deserialize Amd");
        assert_eq!(decoded, DeviceType::Amd);
    }

    #[test]
    fn test_device_type_opencl_serde_roundtrip() {
        let json = serde_json::to_string(&DeviceType::OpenCL).expect("serialize OpenCL");
        assert_eq!(json, "\"opencl\"");
        let decoded: DeviceType = serde_json::from_str(&json).expect("deserialize OpenCL");
        assert_eq!(decoded, DeviceType::OpenCL);
    }

    #[test]
    fn test_device_type_cpu_serde_roundtrip() {
        let json = serde_json::to_string(&DeviceType::Cpu).expect("serialize Cpu");
        assert_eq!(json, "\"cpu\"");
        let decoded: DeviceType = serde_json::from_str(&json).expect("deserialize Cpu");
        assert_eq!(decoded, DeviceType::Cpu);
    }

    #[test]
    fn test_device_type_clone_and_equality() {
        let cpu = DeviceType::Cpu;
        let cpu_clone = cpu.clone();
        assert_eq!(cpu, cpu_clone);
        assert_ne!(DeviceType::Cpu, DeviceType::Cuda);
        assert_ne!(DeviceType::Cuda, DeviceType::Metal);
        assert_ne!(DeviceType::Metal, DeviceType::Amd);
        assert_ne!(DeviceType::Amd, DeviceType::OpenCL);
        assert_ne!(DeviceType::OpenCL, DeviceType::Cpu);
    }

    #[test]
    fn test_pooling_mode_default_is_mean() {
        let mode = crate::config::model::PoolingMode::default();
        assert!(matches!(mode, crate::config::model::PoolingMode::Mean));
    }

    #[test]
    fn test_pooling_mode_clone_and_equality() {
        use crate::config::model::PoolingMode;
        let mean = PoolingMode::Mean;
        let max = PoolingMode::Max;
        let cls = PoolingMode::Cls;
        assert_eq!(mean.clone(), mean);
        assert_eq!(max.clone(), max);
        assert_eq!(cls.clone(), cls);
        assert_ne!(mean, max);
        assert_ne!(max, cls);
        assert_ne!(cls, mean);
    }

    #[test]
    fn test_pooling_mode_serde_roundtrip() {
        use crate::config::model::PoolingMode;
        let modes = [PoolingMode::Mean, PoolingMode::Max, PoolingMode::Cls];
        for mode in &modes {
            let json = serde_json::to_string(mode).expect("serialize PoolingMode");
            let decoded: PoolingMode =
                serde_json::from_str(&json).expect("deserialize PoolingMode");
            assert_eq!(*mode, decoded);
        }
    }

    #[test]
    fn test_engine_type_clone_and_equality() {
        let candle1 = EngineType::Candle;
        let candle2 = candle1.clone();
        assert_eq!(candle1, candle2);
    }

    #[test]
    fn test_engine_type_debug_format() {
        let format = format!("{:?}", EngineType::Candle);
        assert!(format.contains("Candle"));
    }

    #[test]
    fn test_precision_debug_format() {
        let format = format!("{:?}", Precision::Fp32);
        assert!(format.contains("Fp32"));
    }

    #[test]
    fn test_precision_serde_roundtrip() {
        let precisions = [Precision::Fp32, Precision::Fp16, Precision::Int8];
        for p in &precisions {
            let json = serde_json::to_string(p).expect("serialize Precision");
            let decoded: Precision = serde_json::from_str(&json).expect("deserialize Precision");
            assert_eq!(*p, decoded);
        }
    }

    #[test]
    fn test_inference_context_with_config_candle() {
        use crate::config::model::{DeviceType, InferenceContext, PoolingMode};
        let config = ModelConfig {
            name: "ctx-test".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/test/model"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 16,
            pooling_mode: Some(PoolingMode::Max),
            expected_dimension: Some(768),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        let ctx = InferenceContext::with_config(&config, Precision::Fp16);
        assert_eq!(ctx.model_name, "ctx-test");
        assert_eq!(ctx.engine_type, EngineType::Candle);
        assert_eq!(ctx.device, DeviceType::Cpu);
        assert_eq!(ctx.precision, Precision::Fp16);
        assert_eq!(ctx.batch_size, 16);
    }

    #[test]
    fn test_inference_context_default_values() {
        use crate::config::model::InferenceContext;
        let ctx = InferenceContext::default();
        assert_eq!(ctx.batch_size, 32);
        assert_eq!(ctx.max_sequence_length, 8192);
        assert_eq!(ctx.precision, Precision::Fp32);
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.max_batch_size, 32);
        assert!(config.oom_fallback_enabled);
        assert_eq!(config.expected_dimension, None);
    }

    #[test]
    fn test_model_repository_default() {
        use crate::config::model::ModelRepository;
        let repo = ModelRepository::default();
        assert_eq!(repo.models.len(), 1);
        assert_eq!(repo.models[0].name, "default");
    }

    #[test]
    fn test_any_engine_new_candle_cuda_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Cuda;

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_metal_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Metal;

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_amd_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Amd;

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_opencl_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::OpenCL;

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_factory_create_candle_missing_model() {
        let config = test_config_candle();
        let result = crate::engine::EngineFactory::create(EngineType::Candle, &config);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_engine_factory_create_onnx_missing_model() {
        let config = test_config_onnx();
        let result = crate::engine::EngineFactory::create(EngineType::Onnx, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_with_pooling_mode() {
        use crate::config::model::PoolingMode;
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.pooling_mode = Some(PoolingMode::Cls);

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_with_memory_limit() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.memory_limit_bytes = Some(1024 * 1024 * 1024);

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_with_tokenizer_path() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.tokenizer_path = Some(PathBuf::from("/nonexistent/tokenizer"));

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_any_engine_new_candle_oom_fallback_disabled() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_candle();
        config.model_path = temp_dir.path().to_path_buf();
        config.oom_fallback_enabled = false;

        let result = AnyEngine::new(&config, EngineType::Candle, Precision::Fp32);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_metal_device_fails() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.device = DeviceType::Metal;

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_with_large_batch_size() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();
        config.max_batch_size = 512;

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp32);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_int8_precision() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Int8);
        assert!(result.is_err());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_any_engine_new_onnx_fp16_precision() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let mut config = test_config_onnx();
        config.model_path = temp_dir.path().to_path_buf();

        let result = AnyEngine::new(&config, EngineType::Onnx, Precision::Fp16);
        assert!(result.is_err());
    }
}
