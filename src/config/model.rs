// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelType {
    Predefined(PredefinedModelType),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredefinedModelType {
    Bert,
    Roberta,
    M2Bert,
    SentenceBert,
}

impl<'de> Deserialize<'de> for PredefinedModelType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "bert" => Ok(PredefinedModelType::Bert),
            "roberta" => Ok(PredefinedModelType::Roberta),
            "m2-bert" | "m2bert" => Ok(PredefinedModelType::M2Bert),
            "sentence-bert" => Ok(PredefinedModelType::SentenceBert),
            _ => Err(serde::de::Error::unknown_variant(&s, &["bert", "roberta", "m2-bert", "sentence-bert"])),
        }
    }
}

impl Serialize for PredefinedModelType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            PredefinedModelType::Bert => serializer.serialize_str("bert"),
            PredefinedModelType::Roberta => serializer.serialize_str("roberta"),
            PredefinedModelType::M2Bert => serializer.serialize_str("m2-bert"),
            PredefinedModelType::SentenceBert => serializer.serialize_str("sentence-bert"),
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::Predefined(t) => write!(f, "{}", t.as_str()),
            ModelType::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl PredefinedModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PredefinedModelType::Bert => "bert",
            PredefinedModelType::Roberta => "roberta",
            PredefinedModelType::M2Bert => "m2-bert",
            PredefinedModelType::SentenceBert => "sentence-bert",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Fp32 => write!(f, "fp32"),
            Precision::Fp16 => write!(f, "fp16"),
            Precision::Int8 => write!(f, "int8"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceContext {
    pub model_name: String,
    pub model_type: ModelType,
    pub engine_type: EngineType,
    pub device: DeviceType,
    pub precision: Precision,
    pub batch_size: usize,
    pub max_sequence_length: usize,
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self {
            model_name: "default".to_string(),
            model_type: ModelType::Predefined(PredefinedModelType::M2Bert),
            engine_type: EngineType::Candle,
            device: DeviceType::Cpu,
            precision: Precision::Fp32,
            batch_size: 32,
            max_sequence_length: 8192,
        }
    }
}

impl InferenceContext {
    pub fn with_config(config: &ModelConfig, precision: Precision) -> Self {
        Self {
            model_name: config.name.clone(),
            model_type: ModelType::Predefined(PredefinedModelType::M2Bert),
            engine_type: config.engine_type.clone(),
            device: config.device.clone(),
            precision,
            batch_size: config.max_batch_size,
            max_sequence_length: 8192,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineType {
    Candle,
    #[cfg(feature = "onnx")]
    Onnx,
}

impl fmt::Display for EngineType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineType::Candle => write!(f, "candle"),
            #[cfg(feature = "onnx")]
            EngineType::Onnx => write!(f, "onnx"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
    #[serde(rename = "amd")]
    Amd,
    #[serde(rename = "opencl")]
    OpenCL,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingMode {
    Mean,
    Max,
    Cls,
}

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
    pub memory_limit_bytes: Option<u64>,
    pub oom_fallback_enabled: bool,
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
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
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
        assert_eq!(config.memory_limit_bytes, None);
        assert!(config.oom_fallback_enabled);
    }

    #[test]
    fn test_model_config_with_dimension() {
        let config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("/models/bge-m3-tokenizer")),
            device: DeviceType::Cuda,
            max_batch_size: 64,
            pooling_mode: Some(PoolingMode::Mean),
            expected_dimension: Some(1024),
            memory_limit_bytes: Some(8 * 1024 * 1024 * 1024),
            oom_fallback_enabled: true,
        };

        assert_eq!(config.name, "bge-m3");
        assert_eq!(config.expected_dimension, Some(1024));
        assert_eq!(config.memory_limit_bytes, Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_model_config_serialization() {
        let config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("/models/bge-m3-tokenizer")),
            device: DeviceType::Cuda,
            max_batch_size: 64,
            pooling_mode: Some(PoolingMode::Mean),
            expected_dimension: Some(1024),
            memory_limit_bytes: Some(8 * 1024 * 1024 * 1024),
            oom_fallback_enabled: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: ModelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name, "bge-m3");
        assert_eq!(decoded.engine_type, EngineType::Candle);
        assert_eq!(decoded.device, DeviceType::Cuda);
        assert_eq!(decoded.expected_dimension, Some(1024));
        assert_eq!(decoded.memory_limit_bytes, Some(8 * 1024 * 1024 * 1024));
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

        let candle_json = serde_json::to_string(&candle).unwrap();

        assert_eq!(candle_json, "\"candle\"");

        #[cfg(feature = "onnx")]
        {
            let onnx = EngineType::Onnx;
            let onnx_json = serde_json::to_string(&onnx).unwrap();
            assert_eq!(onnx_json, "\"onnx\"");
        }
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

    #[test]
    fn test_model_type_serialization() {
        let bert = ModelType::Predefined(PredefinedModelType::Bert);
        let roberta = ModelType::Predefined(PredefinedModelType::Roberta);
        let m2_bert = ModelType::Predefined(PredefinedModelType::M2Bert);
        let sentence_bert = ModelType::Predefined(PredefinedModelType::SentenceBert);
        let custom = ModelType::Custom("custom_model".to_string());

        assert_eq!(serde_json::to_string(&bert).unwrap(), "\"bert\"");
        assert_eq!(serde_json::to_string(&roberta).unwrap(), "\"roberta\"");
        assert_eq!(serde_json::to_string(&m2_bert).unwrap(), "\"m2-bert\"");
        assert_eq!(serde_json::to_string(&sentence_bert).unwrap(), "\"sentence-bert\"");
        assert_eq!(serde_json::to_string(&custom).unwrap(), "\"custom_model\"");
    }

    #[test]
    fn test_precision_serialization() {
        let fp32 = Precision::Fp32;
        let fp16 = Precision::Fp16;
        let int8 = Precision::Int8;

        assert_eq!(serde_json::to_string(&fp32).unwrap(), "\"fp32\"");
        assert_eq!(serde_json::to_string(&fp16).unwrap(), "\"fp16\"");
        assert_eq!(serde_json::to_string(&int8).unwrap(), "\"int8\"");
    }

    #[test]
    fn test_inference_context_default() {
        let context = InferenceContext::default();

        assert_eq!(context.model_name, "default");
        assert_eq!(context.model_type, ModelType::Predefined(PredefinedModelType::M2Bert));
        assert_eq!(context.engine_type, EngineType::Candle);
        assert_eq!(context.device, DeviceType::Cpu);
        assert_eq!(context.precision, Precision::Fp32);
        assert_eq!(context.batch_size, 32);
        assert_eq!(context.max_sequence_length, 8192);
    }

    #[test]
    fn test_inference_context_with_config() {
        let config = ModelConfig {
            name: "bge-m3".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from("/models/bge-m3"),
            tokenizer_path: Some(PathBuf::from("/models/bge-m3-tokenizer")),
            device: DeviceType::Cuda,
            max_batch_size: 64,
            pooling_mode: Some(PoolingMode::Mean),
            expected_dimension: Some(1024),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
        };

        let context = InferenceContext::with_config(&config, Precision::Fp16);

        assert_eq!(context.model_name, "bge-m3");
        assert_eq!(context.engine_type, EngineType::Candle);
        assert_eq!(context.device, DeviceType::Cuda);
        assert_eq!(context.precision, Precision::Fp16);
        assert_eq!(context.batch_size, 64);
    }

    #[test]
    fn test_inference_context_serialization() {
        let context = InferenceContext::default();
        let json = serde_json::to_string(&context).unwrap();
        let decoded: InferenceContext = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.model_name, "default");
        assert_eq!(decoded.model_type, ModelType::Predefined(PredefinedModelType::M2Bert));
        assert_eq!(decoded.precision, Precision::Fp32);
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(ModelType::Predefined(PredefinedModelType::Bert).to_string(), "bert");
        assert_eq!(ModelType::Predefined(PredefinedModelType::Roberta).to_string(), "roberta");
        assert_eq!(ModelType::Predefined(PredefinedModelType::M2Bert).to_string(), "m2-bert");
        assert_eq!(ModelType::Predefined(PredefinedModelType::SentenceBert).to_string(), "sentence-bert");
        assert_eq!(ModelType::Custom("custom".to_string()).to_string(), "custom");
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::Fp32.to_string(), "fp32");
        assert_eq!(Precision::Fp16.to_string(), "fp16");
        assert_eq!(Precision::Int8.to_string(), "int8");
    }
}
