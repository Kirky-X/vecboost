// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub engine_type: EngineType,
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub device: DeviceType,
    pub max_batch_size: usize,
    pub pooling_mode: Option<PoolingMode>,
    pub expected_dimension: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineType {
    Candle,
    Onnx,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingMode {
    Mean,
    Max,
    Cls,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("models/default"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRepository {
    pub models: Vec<ModelConfig>,
}

impl Default for ModelRepository {
    fn default() -> Self {
        Self {
            models: vec![ModelConfig::default()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.engine_type, EngineType::Candle);
        assert_eq!(config.device, DeviceType::Cpu);
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.expected_dimension, None);
    }

    #[test]
    fn test_model_config_with_dimension() {
        let config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Onnx,
            model_path: PathBuf::from("/models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("/models/bge-m3-tokenizer")),
            device: DeviceType::Cuda,
            max_batch_size: 64,
            pooling_mode: Some(PoolingMode::Mean),
            expected_dimension: Some(1024),
        };

        assert_eq!(config.name, "bge-m3");
        assert_eq!(config.expected_dimension, Some(1024));
    }

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Onnx,
            model_path: PathBuf::from("/models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("/models/bge-m3-tokenizer")),
            device: DeviceType::Cuda,
            max_batch_size: 64,
            pooling_mode: Some(PoolingMode::Mean),
            expected_dimension: Some(1024),
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: ModelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name, "bge-m3");
        assert_eq!(decoded.engine_type, EngineType::Onnx);
        assert_eq!(decoded.device, DeviceType::Cuda);
        assert_eq!(decoded.expected_dimension, Some(1024));
    }

    #[test]
    fn test_model_repository_default() {
        let repo = ModelRepository::default();
        assert_eq!(repo.models.len(), 1);
        assert_eq!(repo.models[0].name, "default");
    }

    #[test]
    fn test_engine_type_serialization() {
        let candle = EngineType::Candle;
        let onnx = EngineType::Onnx;

        let candle_json = serde_json::to_string(&candle).unwrap();
        let onnx_json = serde_json::to_string(&onnx).unwrap();

        assert_eq!(candle_json, "\"candle\"");
        assert_eq!(onnx_json, "\"onnx\"");
    }

    #[test]
    fn test_device_type_serialization() {
        let cpu = DeviceType::Cpu;
        let cuda = DeviceType::Cuda;
        let metal = DeviceType::Metal;

        let cpu_json = serde_json::to_string(&cpu).unwrap();
        let cuda_json = serde_json::to_string(&cuda).unwrap();
        let metal_json = serde_json::to_string(&metal).unwrap();

        assert_eq!(cpu_json, "\"cpu\"");
        assert_eq!(cuda_json, "\"cuda\"");
        assert_eq!(metal_json, "\"metal\"");
    }
}
