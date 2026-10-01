// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! AnyEngine 的实现块

use super::{AnyEngine, InferenceEngine};
use crate::config::model::{EngineType, ModelConfig, ModelTask, Precision};
use crate::domain::{DecisionRequest, DecisionResponse};
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
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.embed(text),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed(text),
            #[cfg(feature = "onnx")]
            // 决策引擎的 embed 覆盖返回 UnsupportedTask（不产向量）
            AnyEngine::Decision(engine) => engine.embed(text),
            AnyEngine::CandleDecision(engine) => engine.embed(text),
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.embed_batch(texts),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.embed_batch(texts),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.embed_batch(texts),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.embed_batch(texts),
            AnyEngine::CandleDecision(engine) => engine.embed_batch(texts),
        }
    }

    fn precision(&self) -> &Precision {
        match self {
            AnyEngine::Candle(engine) => engine.precision(),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.precision(),
            AnyEngine::CandleDecision(engine) => engine.precision(),
        }
    }

    fn supports_mixed_precision(&self) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.supports_mixed_precision(),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.supports_mixed_precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.supports_mixed_precision(),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.supports_mixed_precision(),
            AnyEngine::CandleDecision(engine) => engine.supports_mixed_precision(),
        }
    }

    fn is_fallback_triggered(&self) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.is_fallback_triggered(),
            #[cfg(feature = "quantized-gguf")]
            // QuantizedCandleEngine 走 trait 默认实现（恒 false，无回退状态）
            AnyEngine::Quantized(engine) => InferenceEngine::is_fallback_triggered(engine),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.is_fallback_triggered(),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.is_fallback_triggered(),
            AnyEngine::CandleDecision(engine) => engine.is_fallback_triggered(),
        }
    }

    fn count_tokens(&self, text: &str) -> Result<usize, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.count_tokens(text),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.count_tokens(text),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.count_tokens(text),
            #[cfg(feature = "onnx")]
            // 决策管线未覆盖 count_tokens，走 trait 默认（调用方回退 bytes/4）
            AnyEngine::Decision(engine) => InferenceEngine::count_tokens(engine, text),
            // candle 决策引擎同口径：未覆盖 count_tokens，走 trait 默认
            AnyEngine::CandleDecision(engine) => InferenceEngine::count_tokens(engine, text),
        }
    }

    fn take_stage_snapshot(&self) -> Option<super::StageSnapshot> {
        match self {
            // UFCS 显式走 trait 方法：CandleEngine 的同名固有方法返回
            // StageSnapshot（非 Option），方法解析时固有方法优先会遮蔽 trait 实现。
            AnyEngine::Candle(engine) => InferenceEngine::take_stage_snapshot(engine),
            #[cfg(feature = "quantized-gguf")]
            // 量化引擎暂无分阶段埋点，走 trait 默认 None
            AnyEngine::Quantized(engine) => InferenceEngine::take_stage_snapshot(engine),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.take_stage_snapshot(),
            #[cfg(feature = "onnx")]
            // 决策管线暂无分阶段埋点，走 trait 默认 None
            AnyEngine::Decision(engine) => InferenceEngine::take_stage_snapshot(engine),
            // candle 决策引擎同口径：无分阶段埋点，走 trait 默认 None
            AnyEngine::CandleDecision(engine) => InferenceEngine::take_stage_snapshot(engine),
        }
    }

    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.try_fallback_to_cpu(config).await,
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.try_fallback_to_cpu(config).await,
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.try_fallback_to_cpu(config).await,
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.try_fallback_to_cpu(config).await,
            AnyEngine::CandleDecision(engine) => engine.try_fallback_to_cpu(config).await,
        }
    }

    fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.decide(req),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.decide(req),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.decide(req),
            #[cfg(feature = "onnx")]
            // trait 方法（DecisionPipeline 固有方法名为 decision()，无同名
            // 遮蔽）：覆盖后经 trait 自然进决策管线，漏转发即恒 UnsupportedTask
            AnyEngine::Decision(engine) => engine.decide(req),
            // trait 方法（CandleDecisionEngine 固有方法名为 decide_impl()，
            // 无同名遮蔽）：覆盖后经 trait 自然进 candle 决策头
            AnyEngine::CandleDecision(engine) => engine.decide(req),
        }
    }

    /// 对齐/诊断出口按变体转发：仅决策两路有真实覆盖，其余引擎走 trait
    /// 默认 UnsupportedTask（与 decide 同款漏转发即恒失败的分布口径）
    fn decide_logits(&self, req: &DecisionRequest) -> Result<Vec<Vec<f32>>, VecboostError> {
        match self {
            AnyEngine::Candle(engine) => engine.decide_logits(req),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.decide_logits(req),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.decide_logits(req),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.decide_logits(req),
            AnyEngine::CandleDecision(engine) => engine.decide_logits(req),
        }
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.supports_task(task),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.supports_task(task),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.supports_task(task),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.supports_task(task),
            AnyEngine::CandleDecision(engine) => engine.supports_task(task),
        }
    }

    /// rerank 能力自报必须按变体转发：决策引擎（onnx/candle 两路）覆盖
    /// false（不产向量），漏转发会让 service/rerank 的能力检查误放行。
    /// embedding 引擎继承 trait 默认 true（bi-encoder rerank）。
    fn supports_rerank(&self) -> bool {
        match self {
            AnyEngine::Candle(engine) => engine.supports_rerank(),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(engine) => engine.supports_rerank(),
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.supports_rerank(),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(engine) => engine.supports_rerank(),
            AnyEngine::CandleDecision(engine) => engine.supports_rerank(),
        }
    }

    fn attach_memory_limit_controller(
        &mut self,
        controller: std::sync::Arc<crate::device::memory_limit::MemoryLimitController>,
    ) {
        match self {
            // Candle/Onnx 有真实覆盖（落引擎内 controller，供 Exceeded/Critical
            // 自动 CPU 回退执法）；漏转发会让服务层接线全部落在 trait 默认
            // no-op 上，引擎侧内存执法链路静默失效
            AnyEngine::Candle(engine) => engine.attach_memory_limit_controller(controller),
            #[cfg(feature = "quantized-gguf")]
            // 量化引擎无内存感知分支，UFCS 显式走 trait 默认 no-op
            //（与 is_fallback_triggered 的 Quantized 臂同口径）
            AnyEngine::Quantized(engine) => {
                InferenceEngine::attach_memory_limit_controller(engine, controller)
            }
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(engine) => engine.attach_memory_limit_controller(controller),
            #[cfg(feature = "onnx")]
            // 决策管线无内存感知分支，UFCS 显式走 trait 默认 no-op
            //（与 Quantized 臂同口径）
            AnyEngine::Decision(engine) => {
                InferenceEngine::attach_memory_limit_controller(engine, controller)
            }
            // candle 决策引擎同口径：无内存感知分支，UFCS 显式走 trait 默认 no-op
            AnyEngine::CandleDecision(engine) => {
                InferenceEngine::attach_memory_limit_controller(engine, controller)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, ModelTask};
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
            task: ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
            task: ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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
    fn test_pooling_mode_default_is_auto() {
        let mode = crate::config::model::PoolingMode::default();
        assert!(matches!(mode, crate::config::model::PoolingMode::Auto));
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
            task: ModelTask::Embedding,
            quantized: false,
            decision_params: None,
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

    /// 真实模型 AnyEngine（candle 变体）转发 decide/supports_task。
    /// 与 candle_engine::tests::require_real_model 同口径：权重缺失时 SKIP。
    fn real_model_available() -> bool {
        ["model.safetensors", "pytorch_model.bin"].iter().any(|w| {
            std::path::Path::new("models/BAAI-bge-small-en-v1.5")
                .join(w)
                .exists()
        })
    }

    #[test]
    fn test_any_engine_forwards_decide_and_supports_task() {
        if !real_model_available() {
            eprintln!("Skipping test: model weights not found at models/BAAI-bge-small-en-v1.5");
            return;
        }
        let config = ModelConfig {
            name: "bge-small-en-forward".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("models/BAAI-bge-small-en-v1.5"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        let engine =
            AnyEngine::new(&config, EngineType::Candle, Precision::Fp32).expect("load real model");

        assert!(engine.supports_task(ModelTask::Embedding));
        assert!(!engine.supports_task(ModelTask::Decision));

        let req: crate::domain::DecisionRequest =
            serde_json::from_str(r#"{"state":{},"questions":[]}"#).unwrap();
        let err = engine.decide(&req).unwrap_err();
        assert!(
            matches!(err, VecboostError::UnsupportedTask(_)),
            "AnyEngine::Candle 未覆盖 decide，转发必须落到 trait 默认 UnsupportedTask，got {:?}",
            err
        );
    }

    /// 真实模型 AnyEngine（candle 变体）转发 attach_memory_limit_controller：
    /// attach 后引擎内 controller 必须就位（get_memory_status 从 None 变
    /// Some(Ok)）。漏转发会让服务层两处真实接线（switch_model / 启动期
    /// init_memory_limit）全部落在 trait 默认 no-op 上，引擎侧内存执法
    /// 链路静默失效。权重缺失时 SKIP（与上一测试同口径）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_any_engine_forwards_attach_memory_limit_controller() {
        use crate::device::memory_limit::{MemoryLimitController, MemoryLimitStatus};
        use std::sync::Arc;
        if !real_model_available() {
            eprintln!("Skipping test: model weights not found at models/BAAI-bge-small-en-v1.5");
            return;
        }
        let config = ModelConfig {
            name: "bge-small-en-attach".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("models/BAAI-bge-small-en-v1.5"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(384),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
            task: ModelTask::Embedding,
            quantized: false,
            decision_params: None,
        };
        let mut engine =
            AnyEngine::new(&config, EngineType::Candle, Precision::Fp32).expect("load real model");

        let status_before = match &engine {
            AnyEngine::Candle(candle) => candle.get_memory_status().await,
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(_) => panic!("expected Candle variant"),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(_) => panic!("expected Candle variant"),
            AnyEngine::CandleDecision(_) => panic!("expected Candle variant"),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(_) => panic!("expected Candle variant"),
        };
        assert_eq!(status_before, None, "attach 前引擎内不得有 controller");

        engine.attach_memory_limit_controller(Arc::new(MemoryLimitController::new()));

        let status = match &engine {
            AnyEngine::Candle(candle) => candle.get_memory_status().await,
            #[cfg(feature = "onnx")]
            AnyEngine::Onnx(_) => panic!("expected Candle variant"),
            #[cfg(feature = "onnx")]
            AnyEngine::Decision(_) => panic!("expected Candle variant"),
            AnyEngine::CandleDecision(_) => panic!("expected Candle variant"),
            #[cfg(feature = "quantized-gguf")]
            AnyEngine::Quantized(_) => panic!("expected Candle variant"),
        };
        assert_eq!(
            status,
            Some(MemoryLimitStatus::Ok),
            "attach 必须经 AnyEngine 转发落进引擎内真实 controller"
        );
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
