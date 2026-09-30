// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::path::PathBuf;

use crate::error::VecboostError;

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

/// head（qtype 前缀 + instructions）+ 全部 options（含各自 [MASK]）的 token
/// 预算默认值（英文 checkpoint 口径，文档 §2.1）。
/// `DecisionPipeline` 消费 [`DecisionParams`] 而非编译期常量直传；
/// 本常量是 [`DecisionParams::default`] 的唯一来源。
pub const HEAD_MAX_LEN: usize = 192;
/// state 部分 token 截断上限默认值（文档 §2.1：截断后约 256）——
/// [`DecisionParams::default`] 的来源，交叉引用同 [`HEAD_MAX_LEN`]
pub const STATE_MAX_TOKENS: usize = 256;
/// 决策序列预算硬上限（ModernBERT 位置编码窗宽，与 InferenceContext
/// max_sequence_length 同口径）：配置值超限解析期显性拒绝，禁止静默钳制
pub const MAX_DECISION_SEQ_LEN: usize = 8192;
/// 决策行手工拼接的固定特殊 token 数：`[CLS] + [SEP]×3`（head 后、state
/// 前、行尾各一；拼接结构见 `engine::decision::build_question_row`）。
/// 组合预算硬上限 = head_max_len + state_max_tokens + 本常量 ≤
/// [`MAX_DECISION_SEQ_LEN`]，单字段口径放行、组合超窗的组合在此挡下
/// （否则超窗错误延迟到请求期）
pub const DECISION_ROW_FIXED_TOKENS: usize = 4;

/// per-checkpoint 决策推理序列预算参数：决策预处理调用链消费它而非
/// 编译期常量直传。缺省为英文 checkpoint 口径（192/256），
/// `laya-multilingual` 等 checkpoint 经 `[model.checkpoints.<name>]`
/// 预设表覆盖。本类型定义于 config 层（配置面单一事实源）：预设表
/// 解析与决策管线（onnx feature 门内）共同消费，避免 config →
/// engine 的 feature 门依赖。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct DecisionParams {
    pub head_max_len: usize,
    pub state_max_tokens: usize,
}

impl Default for DecisionParams {
    fn default() -> Self {
        Self {
            head_max_len: HEAD_MAX_LEN,
            state_max_tokens: STATE_MAX_TOKENS,
        }
    }
}

impl DecisionParams {
    /// 校验构造：0 值与超 [`MAX_DECISION_SEQ_LEN`] 硬上限均显性报错
    /// （错误消息含字段名与实际值，拒绝静默钳制）
    pub fn new(head_max_len: usize, state_max_tokens: usize) -> Result<Self, VecboostError> {
        for (field, value) in [
            ("head_max_len", head_max_len),
            ("state_max_tokens", state_max_tokens),
        ] {
            if value == 0 {
                return Err(VecboostError::config_error(format!(
                    "{field} must be a positive integer, got 0"
                )));
            }
            if value > MAX_DECISION_SEQ_LEN {
                return Err(VecboostError::config_error(format!(
                    "{field} exceeds hard limit: got {value}, max is {MAX_DECISION_SEQ_LEN} \
                     (model positional-encoding window)"
                )));
            }
        }
        // 组合口径：两段独立合规不保证行总长不超窗——上界为
        // head_max_len + state_max_tokens + [CLS]/[SEP]x3 固定 token。
        // 两值已过单字段校验（各 ≤ 8192），加法无溢出。
        if head_max_len + state_max_tokens + DECISION_ROW_FIXED_TOKENS > MAX_DECISION_SEQ_LEN {
            return Err(VecboostError::config_error(format!(
                "combined sequence budget exceeds hard limit: head_max_len {head_max_len} + \
                 state_max_tokens {state_max_tokens} + {DECISION_ROW_FIXED_TOKENS} fixed \
                 markers > {MAX_DECISION_SEQ_LEN} (model positional-encoding window)"
            )));
        }
        Ok(Self {
            head_max_len,
            state_max_tokens,
        })
    }
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
    /// 决策推理序列预算（head_max_len/state_max_tokens，per-checkpoint）：
    /// None = 引擎缺省英文口径（192/256，与 [`HEAD_MAX_LEN`] 同源）；
    /// `Some` 仅对 task=decision 的加载路径有意义（`DecisionPipeline::load`
    /// 消费），其余任务下为 no-op。由 `[model.checkpoints.<name>]` 预设表
    /// 解析或 switch_model 命中预设时填充。
    #[serde(default)]
    pub decision_params: Option<DecisionParams>,
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
            decision_params: None,
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

/// `[model.checkpoints.<name>]` 预设表条目：一个可被 switch_model 按名切换的
/// checkpoint 预设。必填 name/model_path/task（serde 反序列化边界显性报错）；
/// engine_type 缺省继承当前加载模型（启动时即 `[model].engine_type`；同主段
/// `Option<String>` 口径，非法值校验期显性报错）；`head_max_len`/`max_len`
/// 缺省 = 引擎缺省英文口径（`DecisionParams::default`，192/256），配置时必须
/// 成对出现。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPreset {
    pub name: String,
    pub model_path: PathBuf,
    pub task: ModelTask,
    /// 缺省回落主段 engine_type；字符串口径与 `[model].engine_type` 一致
    /// （"candle"/"onnx"），非法值校验期显性报错
    #[serde(default)]
    pub engine_type: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    #[serde(default)]
    pub head_max_len: Option<usize>,
    /// state 序列预算（TOML 字段名 `max_len`，映射 `DecisionParams.state_max_tokens`）
    #[serde(default)]
    pub max_len: Option<usize>,
}

impl CheckpointPreset {
    /// 解析条目的决策序列预算：head_max_len/max_len 必须成对出现
    /// （部分配置语义不明，显性拒绝），值合法性委托
    /// [`DecisionParams::new`]（0/超硬上限/组合超窗报错）；
    /// 全缺省回落英文口径。
    pub fn resolved_decision_params(&self) -> Result<DecisionParams, VecboostError> {
        match (self.head_max_len, self.max_len) {
            (None, None) => Ok(DecisionParams::default()),
            (Some(head), Some(state)) => DecisionParams::new(head, state),
            (Some(_), None) => Err(VecboostError::config_error(format!(
                "checkpoint {} configures head_max_len without max_len; \
                 they must be set together",
                self.name
            ))),
            (None, Some(_)) => Err(VecboostError::config_error(format!(
                "checkpoint {} configures max_len without head_max_len; \
                 they must be set together",
                self.name
            ))),
        }
    }

    /// 条目级校验（启动 fail-fast 与测试共用）：name/model_path 非空、
    /// 决策参数合法、engine_type 取值合法（candle / onnx[onnx feature]）。
    pub fn validate(&self) -> Result<(), VecboostError> {
        if self.name.trim().is_empty() {
            return Err(VecboostError::config_error(
                "checkpoint name must not be empty".to_string(),
            ));
        }
        if self.model_path.as_os_str().is_empty() {
            return Err(VecboostError::config_error(format!(
                "checkpoint {} model_path must not be empty",
                self.name
            )));
        }
        self.resolved_decision_params()?;
        self.resolved_engine_type()?;
        Ok(())
    }

    /// engine_type 字符串 → 枚举解析（validate 与 switch 覆盖集构造共用的
    /// 单一事实源）：`None` = 未配置（switch 期继承当前加载模型），非法取值
    /// 显性报错（onnx 取值需 `--features onnx` 构建，错误消息含重建提示）。
    pub fn resolved_engine_type(&self) -> Result<Option<EngineType>, VecboostError> {
        match self.engine_type.as_deref() {
            None => Ok(None),
            Some("candle") => Ok(Some(EngineType::Candle)),
            #[cfg(feature = "onnx")]
            Some("onnx") => Ok(Some(EngineType::Onnx)),
            Some(other) => {
                let hint = if other == "onnx" {
                    "（engine_type=\"onnx\" 需以 --features onnx 重新构建）"
                } else {
                    ""
                };
                Err(VecboostError::config_error(format!(
                    "checkpoint {} engine_type 未知值 \"{other}\"{hint}，支持: candle, onnx",
                    self.name
                )))
            }
        }
    }
}

/// 预设表整体校验：TOML 键（`[model.checkpoints.<name>]` 的 `<name>`）必须与
/// 条目 name 字段一致（双写漂移即路由歧义：按名查找走键、条目内部用 name，
/// 显性拒绝静默分叉）；逐条目 [`CheckpointPreset::validate`]。
pub fn validate_checkpoint_map(
    checkpoints: &std::collections::BTreeMap<String, CheckpointPreset>,
) -> Result<(), VecboostError> {
    for (key, preset) in checkpoints {
        preset.validate()?;
        if *key != preset.name {
            return Err(VecboostError::config_error(format!(
                "checkpoint table key \"{key}\" does not match entry name \"{}\"; \
                 the two must be identical",
                preset.name
            )));
        }
    }
    Ok(())
}

/// 内置 `laya-multilingual` 预设的表键名（switch_model 按名命中）
pub const BUILTIN_MULTILINGUAL_CHECKPOINT: &str = "laya-multilingual";

impl CheckpointPreset {
    /// 内置 `laya-multilingual` 预设：head_max_len 256 / max_len 256（文档
    /// §2.1 多语言 checkpoint 口径）。model_path 为仓库相对惯例路径
    /// （bundle 资产由部署方放置），engine_type 缺省回落 `[model].engine_type`。
    /// 零配置可用性兜底：用户未写任何 TOML 也可 switch 到该名并拿到正确
    /// 参数；显式配置同名条目时以用户配置为准（[`merge_builtin_checkpoints`]）。
    pub fn builtin_laya_multilingual() -> Self {
        Self {
            name: BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(),
            model_path: PathBuf::from("models/laya-multilingual"),
            task: ModelTask::Decision,
            engine_type: None,
            tokenizer_path: None,
            head_max_len: Some(256),
            max_len: Some(256),
        }
    }
}

/// 预设表解析：内置 `laya-multilingual` 打底，用户显式配置的同名条目对内置
/// 做**字段级覆盖**——可选字段（engine_type/tokenizer_path/head_max_len/
/// max_len）为 `None` 时逐字段继承内置值（用户只定制 model_path 时内置
/// 256/256 多语言口径不被整条替换静默丢弃），必填字段（name/model_path/
/// task）serde 必填强制显式、天然以用户为准；其余条目整条插入。
/// 单向写 `head_max_len`/`max_len` 之一在内置名上**同样显性拒绝**（覆盖前
/// 先对用户原始条目跑 [`CheckpointPreset::resolved_decision_params`]）——
/// 否则字段级继承会把用户单向值与内置值拼成合法对，同一段 TOML 语法在内置
/// 名静默生效非对称预算、在非内置名报错。输出表即运行时唯一事实源：
/// main.rs fail-fast 校验 + kit.set_config 注入 + EmbeddingService switch
/// 查表共用同一 resolved 结果。
pub fn merge_builtin_checkpoints(
    configured: std::collections::BTreeMap<String, CheckpointPreset>,
) -> Result<std::collections::BTreeMap<String, CheckpointPreset>, VecboostError> {
    let mut merged = std::collections::BTreeMap::new();
    merged.insert(
        BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(),
        CheckpointPreset::builtin_laya_multilingual(),
    );
    for (key, preset) in configured {
        if key == BUILTIN_MULTILINGUAL_CHECKPOINT {
            preset.resolved_decision_params()?;
            let builtin = &merged[BUILTIN_MULTILINGUAL_CHECKPOINT];
            let overlay = CheckpointPreset {
                name: preset.name,
                model_path: preset.model_path,
                task: preset.task,
                engine_type: preset.engine_type.or_else(|| builtin.engine_type.clone()),
                tokenizer_path: preset
                    .tokenizer_path
                    .or_else(|| builtin.tokenizer_path.clone()),
                head_max_len: preset.head_max_len.or(builtin.head_max_len),
                max_len: preset.max_len.or(builtin.max_len),
            };
            merged.insert(key, overlay);
        } else {
            merged.insert(key, preset);
        }
    }
    Ok(merged)
}

/// switch_model 命中预设时的字段覆盖集：请求显式值优先，预设值为缺省层
/// （与 switch_model 既有「req 优先、缺省继承」模式对齐）。engine_type 为
/// `None` 表示预设未配置该字段（switch 期继承当前加载模型）。
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointSwitchOverride {
    pub model_path: PathBuf,
    pub engine_type: Option<EngineType>,
    pub tokenizer_path: Option<PathBuf>,
    pub task: ModelTask,
    pub decision_params: DecisionParams,
}

impl CheckpointPreset {
    /// 预设条目 → switch 覆盖集。决策参数经 [`Self::resolved_decision_params`]、
    /// engine_type 经 [`Self::resolved_engine_type`] 校验构造（启动 fail-fast
    /// 已挡非法值，此处防御性复检——switch 路径拿到的表理论恒合法，损坏即
    /// 显性报错而非静默缺省）。
    pub fn switch_override(&self) -> Result<CheckpointSwitchOverride, VecboostError> {
        Ok(CheckpointSwitchOverride {
            model_path: self.model_path.clone(),
            engine_type: self.resolved_engine_type()?,
            tokenizer_path: self.tokenizer_path.clone(),
            task: self.task,
            decision_params: self.resolved_decision_params()?,
        })
    }
}

/// 按名查预设（resolved 表，含内置 `laya-multilingual`）：命中返回覆盖集，
/// 未命中 None（switch 走既有自由切换路径，行为与现状等价）。
pub fn lookup_checkpoint_override(
    table: &std::collections::BTreeMap<String, CheckpointPreset>,
    name: &str,
) -> Result<Option<CheckpointSwitchOverride>, VecboostError> {
    table.get(name).map(|p| p.switch_override()).transpose()
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
            decision_params: None,
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
            decision_params: None,
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

    // ── [model.checkpoints.<name>] 预设表解析 ──

    fn full_preset_json() -> &'static str {
        r#"{
            "name": "laya-multilingual",
            "model_path": "models/laya-multilingual",
            "task": "decision",
            "engine_type": "onnx",
            "tokenizer_path": "models/laya-multilingual/tokenizer/tokenizer.json",
            "head_max_len": 256,
            "max_len": 256
        }"#
    }

    #[test]
    fn test_checkpoint_preset_parses_full_fields() {
        let preset: CheckpointPreset = serde_json::from_str(full_preset_json()).unwrap();
        assert_eq!(preset.name, "laya-multilingual");
        assert_eq!(preset.model_path, PathBuf::from("models/laya-multilingual"));
        assert_eq!(preset.task, ModelTask::Decision);
        assert_eq!(preset.engine_type.as_deref(), Some("onnx"));
        assert_eq!(
            preset.tokenizer_path,
            Some(PathBuf::from(
                "models/laya-multilingual/tokenizer/tokenizer.json"
            ))
        );
        assert_eq!(preset.head_max_len, Some(256));
        assert_eq!(preset.max_len, Some(256));
    }

    #[test]
    fn test_checkpoint_preset_optional_fields_default() {
        // 仅必填三字段可解析：engine_type 回落主段、参数回落引擎缺省口径
        let preset: CheckpointPreset = serde_json::from_str(
            r#"{"name": "laya", "model_path": "models/laya", "task": "decision"}"#,
        )
        .unwrap();
        assert_eq!(preset.engine_type, None);
        assert_eq!(preset.tokenizer_path, None);
        assert_eq!(preset.head_max_len, None);
        assert_eq!(preset.max_len, None);
        let params = preset.resolved_decision_params().expect("缺省参数合法");
        assert_eq!(
            params,
            DecisionParams::default(),
            "缺省回落英文口径 192/256"
        );
        assert_eq!(params.head_max_len, 192);
        assert_eq!(params.state_max_tokens, 256);
    }

    #[test]
    fn test_checkpoint_preset_missing_required_rejected() {
        for json in [
            r#"{"model_path": "m", "task": "decision"}"#,
            r#"{"name": "laya", "task": "decision"}"#,
            r#"{"name": "laya", "model_path": "m"}"#,
        ] {
            let result: Result<CheckpointPreset, _> = serde_json::from_str(json);
            assert!(result.is_err(), "缺失必填字段必须显性拒绝: {json}");
        }
    }

    #[test]
    fn test_checkpoint_preset_unknown_field_rejected() {
        let result: Result<CheckpointPreset, _> = serde_json::from_str(
            r#"{"name": "laya", "model_path": "m", "task": "decision", "head_max_lenght": 1}"#,
        );
        assert!(
            result.is_err(),
            "未知字段必须显性拒绝（deny_unknown_fields）"
        );
    }

    #[test]
    fn test_checkpoint_preset_invalid_task_rejected() {
        let result: Result<CheckpointPreset, _> =
            serde_json::from_str(r#"{"name": "laya", "model_path": "m", "task": "rerank"}"#);
        assert!(
            result.is_err(),
            "task 非法取值必须显性拒绝（无 Rerank 任务维度）"
        );
    }

    #[test]
    fn test_checkpoint_preset_invalid_engine_type_rejected() {
        let mut preset: CheckpointPreset = serde_json::from_str(
            r#"{"name": "laya", "model_path": "m", "task": "decision", "engine_type": "tensorrt"}"#,
        )
        .unwrap();
        let err = preset.validate().unwrap_err();
        assert!(
            matches!(err, VecboostError::ConfigError(_)),
            "engine_type 非法取值校验期显性报错，got {err:?}"
        );
        preset.engine_type = Some("onnx".to_string());
        if cfg!(feature = "onnx") {
            preset.validate().expect("onnx 取值在 onnx feature 下合法");
        } else {
            assert!(preset.validate().is_err());
        }
    }

    #[test]
    fn test_checkpoint_preset_resolved_engine_type() {
        // None = 未配置（switch 期继承当前加载模型），不产生引擎枚举值
        let mut preset: CheckpointPreset =
            serde_json::from_str(r#"{"name": "laya", "model_path": "m", "task": "decision"}"#)
                .unwrap();
        assert_eq!(
            preset.resolved_engine_type().expect("None 合法"),
            None,
            "未配置 = 继承当前，不预设引擎"
        );
        preset.engine_type = Some("candle".to_string());
        assert_eq!(
            preset.resolved_engine_type().expect("candle 合法"),
            Some(EngineType::Candle)
        );
        // EngineType::Onnx 变体本身 feature 门内（cfg! 运行时宏不门控编译，
        // 非 onnx 构建下引用即 E0599），两分支用 #[cfg] 编译期切换
        preset.engine_type = Some("onnx".to_string());
        #[cfg(feature = "onnx")]
        assert_eq!(
            preset.resolved_engine_type().expect("onnx 合法"),
            Some(EngineType::Onnx)
        );
        #[cfg(not(feature = "onnx"))]
        assert!(
            preset.resolved_engine_type().is_err(),
            "onnx 取值在非 onnx feature 下显性报错（含重建提示）"
        );
        preset.engine_type = Some("tensorrt".to_string());
        let err = preset.resolved_engine_type().unwrap_err();
        assert!(
            matches!(err, VecboostError::ConfigError(_)) && err.error_detail().contains("tensorrt"),
            "非法取值报错点名实际值，got {err:?}"
        );
    }

    #[test]
    fn test_checkpoint_preset_params_must_be_paired() {
        for (head, max) in [(Some(256), None), (None, Some(256))] {
            let preset = CheckpointPreset {
                name: "laya".to_string(),
                model_path: PathBuf::from("models/laya"),
                task: ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: head,
                max_len: max,
            };
            let err = preset.resolved_decision_params().unwrap_err();
            assert!(
                matches!(err, VecboostError::ConfigError(_)),
                "head_max_len/max_len 必须成对配置，got {err:?}"
            );
        }
    }

    #[test]
    fn test_checkpoint_preset_invalid_params_rejected_on_validate() {
        for (head, max) in [(0usize, 256usize), (192, 9000)] {
            let preset = CheckpointPreset {
                name: "laya".to_string(),
                model_path: PathBuf::from("models/laya"),
                task: ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: Some(head),
                max_len: Some(max),
            };
            let err = preset.validate().unwrap_err();
            assert!(
                matches!(err, VecboostError::ConfigError(_)),
                "head={head} max={max} 必须 validate 期显性拒绝（拒绝静默钳制），got {err:?}"
            );
        }
    }

    #[test]
    fn test_checkpoint_preset_resolve_multilingual_params() {
        let preset: CheckpointPreset = serde_json::from_str(full_preset_json()).unwrap();
        let params = preset
            .resolved_decision_params()
            .expect("multilingual 参数合法");
        assert_eq!(params.head_max_len, 256);
        assert_eq!(params.state_max_tokens, 256);
    }

    #[test]
    fn test_validate_checkpoint_map_key_mismatch_rejected() {
        let mut map = std::collections::BTreeMap::new();
        map.insert(
            "wrong-key".to_string(),
            CheckpointPreset {
                name: "laya".to_string(),
                model_path: PathBuf::from("models/laya"),
                task: ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: None,
                max_len: None,
            },
        );
        let err = validate_checkpoint_map(&map).unwrap_err();
        assert!(
            matches!(err, VecboostError::ConfigError(_))
                && err.error_detail().contains("wrong-key"),
            "TOML 键与条目 name 不一致必须显性拒绝，got {err:?}"
        );
    }

    #[test]
    fn test_validate_checkpoint_map_empty_ok() {
        let map = std::collections::BTreeMap::new();
        validate_checkpoint_map(&map).expect("空表 = 现状行为，必须通过");
    }

    // ── 内置 laya-multilingual 预设与 switch 覆盖集（R-multi-checkpoint-002）──

    #[test]
    fn test_builtin_laya_multilingual_profile() {
        let builtin = CheckpointPreset::builtin_laya_multilingual();
        assert_eq!(builtin.name, "laya-multilingual");
        assert_eq!(
            builtin.model_path,
            PathBuf::from("models/laya-multilingual")
        );
        assert_eq!(builtin.task, ModelTask::Decision);
        assert_eq!(
            builtin.engine_type, None,
            "engine_type 缺省继承当前加载模型（启动时即 [model] 主段）"
        );
        // 文档 §2.1 多语言 checkpoint 口径数值钉：漂移即协议变更，须显性确认
        assert_eq!(builtin.head_max_len, Some(256));
        assert_eq!(builtin.max_len, Some(256));
        builtin.validate().expect("内置预设必须过自身校验闸");
        let params = builtin.resolved_decision_params().expect("内置参数合法");
        assert_eq!(params.head_max_len, 256);
        assert_eq!(params.state_max_tokens, 256);
    }

    #[test]
    fn test_merge_builtin_checkpoints_defaults_and_override() {
        let merged = merge_builtin_checkpoints(Default::default()).expect("空配置 merge 恒通过");
        assert_eq!(merged.len(), 1, "零配置仅含内置预设（零配置可用性兜底）");
        assert!(merged.contains_key(BUILTIN_MULTILINGUAL_CHECKPOINT));

        // 用户同名显式配置覆盖内置（显式优先：可改 model_path/参数不绕内置）
        let user = CheckpointPreset {
            name: BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(),
            model_path: PathBuf::from("data/laya-multi"),
            task: ModelTask::Decision,
            engine_type: None,
            tokenizer_path: None,
            head_max_len: Some(384),
            max_len: Some(384),
        };
        let merged = merge_builtin_checkpoints(
            [(BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(), user)]
                .into_iter()
                .collect(),
        )
        .expect("成对覆盖内置必须通过");
        assert_eq!(merged.len(), 1, "同名覆盖不产生双条目");
        assert_eq!(
            merged[BUILTIN_MULTILINGUAL_CHECKPOINT].model_path,
            PathBuf::from("data/laya-multi")
        );
        assert_eq!(
            merged[BUILTIN_MULTILINGUAL_CHECKPOINT].head_max_len,
            Some(384)
        );
    }

    #[test]
    fn test_merge_builtin_checkpoints_partial_overlay_inherits_builtin_params() {
        // 最常见定制场景：同名条目只写必填三字段（换 model_path）——可选
        // 字段必须逐字段继承内置 256/256 多语言口径，而非整条替换后静默
        // 落回英文缺省 192/256 让 head 193-256 的合法请求报错
        let user = CheckpointPreset {
            name: BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(),
            model_path: PathBuf::from("data/laya-multi"),
            task: ModelTask::Decision,
            engine_type: None,
            tokenizer_path: None,
            head_max_len: None,
            max_len: None,
        };
        let merged = merge_builtin_checkpoints(
            [(BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(), user)]
                .into_iter()
                .collect(),
        )
        .expect("全缺省同名覆盖必须通过");
        let m = &merged[BUILTIN_MULTILINGUAL_CHECKPOINT];
        assert_eq!(
            m.model_path,
            PathBuf::from("data/laya-multi"),
            "必填字段以用户为准"
        );
        assert_eq!(
            m.head_max_len,
            Some(256),
            "未写的 head_max_len 继承内置多语言口径"
        );
        assert_eq!(m.max_len, Some(256), "未写的 max_len 继承内置多语言口径");
        let params = m.resolved_decision_params().expect("合并后参数合法");
        assert_eq!(
            (params.head_max_len, params.state_max_tokens),
            (256, 256),
            "合并结果的决策序列预算保持内置口径"
        );
    }

    #[test]
    fn test_merge_builtin_checkpoints_one_sided_params_rejected_like_non_builtin() {
        // 内置名不得绕过成对闸门：单向写 head_max_len/max_len 之一若在字段级
        // 覆盖后才校验，.or 继承会把用户单向值与内置 256 拼成合法对，同一 TOML
        // 语法在内置名静默生效非对称预算、在非内置名报错。覆盖前先对用户原始
        // 条目跑 resolved_decision_params，两种名字同语义显性拒绝。
        for (field, head, max) in [
            ("head_max_len", Some(384usize), None),
            ("max_len", None, Some(384usize)),
        ] {
            let user = CheckpointPreset {
                name: BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(),
                model_path: PathBuf::from("data/laya-multi"),
                task: ModelTask::Decision,
                engine_type: None,
                tokenizer_path: None,
                head_max_len: head,
                max_len: max,
            };
            let err = merge_builtin_checkpoints(
                [(BUILTIN_MULTILINGUAL_CHECKPOINT.to_string(), user)]
                    .into_iter()
                    .collect(),
            )
            .unwrap_err();
            assert!(
                matches!(err, VecboostError::ConfigError(_))
                    && err.error_detail().contains(BUILTIN_MULTILINGUAL_CHECKPOINT)
                    && err.error_detail().contains(field)
                    && err.error_detail().contains("must be set together"),
                "内置名单向写 {field} 必须与非内置名同语义显性拒绝，got {err:?}"
            );
        }
    }

    #[test]
    fn test_lookup_checkpoint_override_hits_builtin_profile() {
        let table = merge_builtin_checkpoints(Default::default()).expect("内置表 merge 恒通过");
        let hit = lookup_checkpoint_override(&table, BUILTIN_MULTILINGUAL_CHECKPOINT)
            .expect("内置预设参数合法")
            .expect("内置名必须命中");
        assert_eq!(
            hit.model_path,
            PathBuf::from("models/laya-multilingual"),
            "预设 model_path 作为 switch 缺省层"
        );
        assert_eq!(hit.task, ModelTask::Decision);
        assert_eq!(
            hit.engine_type, None,
            "内置未配 engine_type = switch 期继承当前加载模型"
        );
        assert_eq!(hit.tokenizer_path, None);
        assert_eq!(
            hit.decision_params,
            DecisionParams::new(256, 256).expect("256/256 合法"),
            "switch 命中后决策请求按该 checkpoint 的 DecisionParams 生效"
        );
    }

    #[test]
    fn test_lookup_checkpoint_override_miss_returns_none() {
        let table = merge_builtin_checkpoints(Default::default()).expect("内置表 merge 恒通过");
        let miss = lookup_checkpoint_override(&table, "not-a-checkpoint").expect("未命中不报错");
        assert!(
            miss.is_none(),
            "未命中 None = switch 走自由切换、与现状等价"
        );
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
            decision_params: None,
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
