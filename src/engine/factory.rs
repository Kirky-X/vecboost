// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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
                let any = AnyEngine::Quantized(engine);
                ensure_supports_task(&any, config)?;
                return Ok(any);
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
        let engine = AnyEngine::new(config, engine_type, precision)?;
        // fail-fast（评审 R5）：配置任务维度超出引擎能力时创建即失败，
        // 而非延迟到 decide 调用才以 4xx 暴露（decision 引擎落地后自动放行）
        ensure_supports_task(&engine, config)?;
        Ok(engine)
    }
}

/// fail-fast 校验：引擎不支持配置的任务维度时报 UnsupportedTask（显性失败，
/// 禁止静默加载后由调用时 4xx 兜底）。
///
/// 故意在 load 之后（而非 load 前按 config.task 静态短路）校验：当前所有
/// 引擎的 `supports_task` 恰为与实例无关的静态判定（仅 embedding），load 前
/// 短路可省一次误配置下的无效权重加载；但把它固化为工厂前置契约，会在
/// decision 引擎按实例能力 override 判定后被误拒——故保留 load 后以实例作答。
fn ensure_supports_task(engine: &AnyEngine, config: &ModelConfig) -> Result<(), VecboostError> {
    use super::InferenceEngine;
    if !engine.supports_task(config.task) {
        return Err(VecboostError::unsupported_task(format!(
            "engine does not support configured task {}",
            config.task
        )));
    }
    Ok(())
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
            task: crate::config::model::ModelTask::Embedding,
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

    /// fail-fast（评审 R5）：配置 task=decision 但引擎只支持 embedding 时，
    /// 工厂创建即报 UnsupportedTask，而非延迟到 decide 调用才以 4xx 暴露。
    /// 真实模型权重缺失时 SKIP（与 impl_.rs 同口径）。
    #[test]
    fn test_create_rejects_unsupported_task() {
        let has_weights = ["model.safetensors", "pytorch_model.bin"].iter().any(|w| {
            std::path::Path::new("models/BAAI-bge-small-en-v1.5")
                .join(w)
                .exists()
        });
        if !has_weights {
            eprintln!("Skipping test: model weights not found at models/BAAI-bge-small-en-v1.5");
            return;
        }
        let mut config = test_config();
        config.model_path = PathBuf::from("models/BAAI-bge-small-en-v1.5");
        config.task = crate::config::model::ModelTask::Decision;
        let result = EngineFactory::create(EngineType::Candle, &config);
        match result {
            Err(VecboostError::UnsupportedTask(msg)) => {
                assert!(
                    msg.contains("decision"),
                    "错误必须标明不支持的 task，msg={msg}"
                );
            }
            other => panic!("expected Err(UnsupportedTask), got {:?}", other.err()),
        }
    }

    /// task=embedding（默认）的既有加载路径不受 fail-fast 校验影响
    #[test]
    fn test_create_accepts_embedding_task() {
        let has_weights = ["model.safetensors", "pytorch_model.bin"].iter().any(|w| {
            std::path::Path::new("models/BAAI-bge-small-en-v1.5")
                .join(w)
                .exists()
        });
        if !has_weights {
            eprintln!("Skipping test: model weights not found at models/BAAI-bge-small-en-v1.5");
            return;
        }
        let mut config = test_config();
        config.model_path = PathBuf::from("models/BAAI-bge-small-en-v1.5");
        config.task = crate::config::model::ModelTask::Embedding;
        use crate::engine::InferenceEngine;
        let engine = EngineFactory::create(EngineType::Candle, &config)
            .expect("embedding task must load unchanged");
        assert!(engine.supports_task(crate::config::model::ModelTask::Embedding));
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
