// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelType {
    Predefined(PredefinedModelType),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &["bert", "roberta", "m2-bert", "sentence-bert"],
            )),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Int8,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Fp32 => write!(f, "fp32"),
            Precision::Fp16 => write!(f, "fp16"),
            Precision::Bf16 => write!(f, "bf16"),
            Precision::Int8 => write!(f, "int8"),
        }
    }
}

/// 模型任务维度。
///
/// 不设 Rerank 变体：rerank 是 embed 引擎的 trait 默认能力（`InferenceEngine::rerank`
/// 及其批量变体，bi-encoder 语义），无独立引擎类型可分派，任务维度上不构成独立 task。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ModelTask {
    /// 向量嵌入（既有部署语义，默认值）
    #[default]
    Embedding,
    /// 决策推理（多问题作答：choice/score/noul）
    Decision,
}

impl fmt::Display for ModelTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl ModelTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelTask::Embedding => "embedding",
            ModelTask::Decision => "decision",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceContext {
    pub model_name: String,
    pub model_type: ModelType,
    pub engine_type: EngineType,
    pub device: DeviceType,
    pub precision: Precision,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    /// 任务维度随上下文透传：decision 引擎复用本结构时不得静默缺 task。
    /// 旧 JSON 缺该字段回落 Embedding（#[serde(default)] 向后兼容）。
    #[serde(default)]
    pub task: ModelTask,
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
            task: ModelTask::Embedding,
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
            task: config.task,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum PoolingMode {
    /// Auto: 按模型名推断(minilm/e5/gte → Mean, bge → Cls, 其余 Cls+warn)。
    #[default]
    Auto,
    /// 平均池化(attention-mask 加权平均)。
    Mean,
    /// 最大池化(mask 内逐维最大)。
    Max,
    /// CLS token 池化(取 index 0)。
    Cls,
}

/// 键名拼错显性失败（如 "tsak" 报错而非被静默忽略后 task 回落默认值，
/// 模型以错误任务身份加载成功、错误延迟到 decide 调用才暴露）；
/// 序列化产物为对称全字段集合，回读不受影响。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    pub model_sha256: Option<String>,
    /// GGUF 量化模型开关：true 且 `model_path` 以 `.gguf` 结尾时
    /// 路由至 `QuantizedCandleEngine`；默认 false（safetensors 路径不变）。
    #[serde(default)]
    pub quantized: bool,
    /// 任务维度：旧配置缺省该字段时回落 Embedding（#[serde(default)] 向后兼容）。
    #[serde(default)]
    pub task: ModelTask,
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
            model_sha256: None,
            quantized: false,
            task: ModelTask::Embedding,
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
        assert!(
            !config.quantized,
            "quantized 默认为 false（safetensors 路径不变）"
        );
        assert_eq!(
            config.task,
            ModelTask::Embedding,
            "task 默认 Embedding（既有部署语义不变）"
        );
    }

    #[test]
    fn test_model_task_serde_lowercase_roundtrip() {
        assert_eq!(
            serde_json::to_string(&ModelTask::Embedding).unwrap(),
            "\"embedding\""
        );
        assert_eq!(
            serde_json::to_string(&ModelTask::Decision).unwrap(),
            "\"decision\""
        );

        let decoded: ModelTask = serde_json::from_str("\"embedding\"").unwrap();
        assert_eq!(decoded, ModelTask::Embedding);
        let decoded: ModelTask = serde_json::from_str("\"decision\"").unwrap();
        assert_eq!(decoded, ModelTask::Decision);
    }

    #[test]
    fn test_model_task_invalid_value_rejected() {
        let invalid: Result<ModelTask, _> = serde_json::from_str("\"rerank\"");
        assert!(invalid.is_err(), "rerank 不是任务维度（是 trait 默认能力）");
    }

    #[test]
    fn test_model_task_display_clone_eq() {
        assert_eq!(ModelTask::Embedding.to_string(), "embedding");
        assert_eq!(ModelTask::Decision.to_string(), "decision");
        assert_eq!(ModelTask::as_str(&ModelTask::Decision), "decision");
        assert_eq!(ModelTask::default(), ModelTask::Embedding);
        let copied = ModelTask::Decision;
        assert_eq!(copied, ModelTask::Decision);
        assert_ne!(ModelTask::Embedding, ModelTask::Decision);
        // Copy：supports_task(task) 以值传参，config 借用持有的场景不得被迫 clone
        fn assert_copy<T: Copy>() {}
        assert_copy::<ModelTask>();
        let a = ModelTask::Decision;
        let b = a;
        assert_eq!(a, b, "Copy 语义：move 后原值仍可用");
    }

    #[test]
    fn test_model_config_legacy_json_without_task_defaults_to_embedding() {
        // 零破坏核心断言：旧配置 JSON（无 task 字段）必须可解析且回落 Embedding
        let legacy = r#"{
            "name": "bge-m3",
            "engine_type": "candle",
            "model_path": "/models/bge-m3",
            "tokenizer_path": null,
            "device": "cpu",
            "max_batch_size": 32,
            "pooling_mode": null,
            "expected_dimension": null,
            "memory_limit_bytes": null,
            "oom_fallback_enabled": true,
            "model_sha256": null,
            "quantized": false
        }"#;
        let config: ModelConfig = serde_json::from_str(legacy).unwrap();
        assert_eq!(config.name, "bge-m3");
        assert_eq!(config.task, ModelTask::Embedding);
    }

    #[test]
    fn test_model_config_unknown_key_rejected() {
        // deny_unknown_fields：键名拼错（"tsak"）必须显性报错而非被静默忽略
        // 后 task 回落 Embedding（模型以错误任务身份加载成功、错误延迟到
        // decide 调用才暴露）
        let typo = r#"{
            "name": "bge-m3",
            "engine_type": "candle",
            "model_path": "/models/bge-m3",
            "tokenizer_path": null,
            "device": "cpu",
            "max_batch_size": 32,
            "pooling_mode": null,
            "expected_dimension": null,
            "memory_limit_bytes": null,
            "oom_fallback_enabled": true,
            "model_sha256": null,
            "quantized": false,
            "task": "embedding",
            "tsak": "decision"
        }"#;
        let result: Result<ModelConfig, _> = serde_json::from_str(typo);
        assert!(result.is_err(), "未知键 tsak 必须显性拒绝");
    }

    #[test]
    fn test_model_config_task_decision_roundtrip() {
        let config = ModelConfig {
            task: ModelTask::Decision,
            ..ModelConfig::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"task\":\"decision\""), "json={}", json);
        let decoded: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.task, ModelTask::Decision);
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
            model_sha256: None,
            task: ModelTask::Embedding,
            quantized: false,
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
            model_sha256: None,
            task: ModelTask::Embedding,
            quantized: false,
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
        assert_eq!(
            serde_json::to_string(&sentence_bert).unwrap(),
            "\"sentence-bert\""
        );
        assert_eq!(serde_json::to_string(&custom).unwrap(), "\"custom_model\"");
    }

    #[test]
    fn test_precision_serialization() {
        let fp32 = Precision::Fp32;
        let fp16 = Precision::Fp16;
        let bf16 = Precision::Bf16;
        let int8 = Precision::Int8;

        assert_eq!(serde_json::to_string(&fp32).unwrap(), "\"fp32\"");
        assert_eq!(serde_json::to_string(&fp16).unwrap(), "\"fp16\"");
        assert_eq!(serde_json::to_string(&bf16).unwrap(), "\"bf16\"");
        assert_eq!(serde_json::to_string(&int8).unwrap(), "\"int8\"");

        // Deserialization roundtrip
        let bf16_decoded: Precision = serde_json::from_str("\"bf16\"").unwrap();
        assert_eq!(bf16_decoded, Precision::Bf16);

        // Invalid precision string
        let invalid: Result<Precision, _> = serde_json::from_str("\"bf8\"");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_inference_context_default() {
        let context = InferenceContext::default();

        assert_eq!(context.model_name, "default");
        assert_eq!(
            context.model_type,
            ModelType::Predefined(PredefinedModelType::M2Bert)
        );
        assert_eq!(context.engine_type, EngineType::Candle);
        assert_eq!(context.device, DeviceType::Cpu);
        assert_eq!(context.precision, Precision::Fp32);
        assert_eq!(context.batch_size, 32);
        assert_eq!(context.max_sequence_length, 8192);
        assert_eq!(
            context.task,
            ModelTask::Embedding,
            "task 默认 Embedding（与 ModelConfig 默认同源）"
        );
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
            model_sha256: None,
            task: ModelTask::Embedding,
            quantized: false,
        };

        let context = InferenceContext::with_config(&config, Precision::Fp16);

        assert_eq!(context.model_name, "bge-m3");
        assert_eq!(context.engine_type, EngineType::Candle);
        assert_eq!(context.device, DeviceType::Cuda);
        assert_eq!(context.precision, Precision::Fp16);
        assert_eq!(context.batch_size, 64);
        // task 必须经 with_config 从 config 透传，不得静默缺省
        assert_eq!(context.task, config.task);
    }

    #[test]
    fn test_inference_context_serialization() {
        let context = InferenceContext::default();
        let json = serde_json::to_string(&context).unwrap();
        let decoded: InferenceContext = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.model_name, "default");
        assert_eq!(
            decoded.model_type,
            ModelType::Predefined(PredefinedModelType::M2Bert)
        );
        assert_eq!(decoded.precision, Precision::Fp32);
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(
            ModelType::Predefined(PredefinedModelType::Bert).to_string(),
            "bert"
        );
        assert_eq!(
            ModelType::Predefined(PredefinedModelType::Roberta).to_string(),
            "roberta"
        );
        assert_eq!(
            ModelType::Predefined(PredefinedModelType::M2Bert).to_string(),
            "m2-bert"
        );
        assert_eq!(
            ModelType::Predefined(PredefinedModelType::SentenceBert).to_string(),
            "sentence-bert"
        );
        assert_eq!(
            ModelType::Custom("custom".to_string()).to_string(),
            "custom"
        );
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(Precision::Fp32.to_string(), "fp32");
        assert_eq!(Precision::Fp16.to_string(), "fp16");
        assert_eq!(Precision::Bf16.to_string(), "bf16");
        assert_eq!(Precision::Int8.to_string(), "int8");
    }

    // -------------------------------------------------------------------------
    // Eq trait 验证（确保 InferenceContext 可派生 Eq）
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_type_eq() {
        let a = ModelType::Predefined(PredefinedModelType::Bert);
        let b = ModelType::Predefined(PredefinedModelType::Bert);
        let c = ModelType::Custom("test".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_precision_eq() {
        assert_eq!(Precision::Fp32, Precision::Fp32);
        assert_ne!(Precision::Fp32, Precision::Fp16);
        assert_ne!(Precision::Bf16, Precision::Int8);
    }

    #[test]
    fn test_engine_type_eq() {
        assert_eq!(EngineType::Candle, EngineType::Candle);
    }

    #[test]
    fn test_device_type_eq() {
        assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
        assert_ne!(DeviceType::Cpu, DeviceType::Cuda);
        assert_ne!(DeviceType::Cuda, DeviceType::Metal);
    }

    #[test]
    fn test_inference_context_eq() {
        let ctx = InferenceContext::default();
        let ctx2 = InferenceContext::default();
        assert_eq!(ctx, ctx2);
    }
}
