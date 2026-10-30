// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 引擎工厂：根据 `EngineType` 创建对应的推理引擎实例

use super::AnyEngine;
use crate::config::model::{EngineType, ModelConfig, ModelTask, Precision};
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
        // task 路由（主维度）：task=decision 一律走决策引擎。engine_type
        // 在该任务下决定推理后端：candle → 原生决策头（无 onnxruntime 依赖，
        // 消费 convaiinnovations/laya PyTorch checkpoint；feature `candle`
        // opt-in，onnx-primary 默认不含）；onnx → ort 管线
        // （消费 receptron/laya-onnx bundle）。ensure_supports_task 因决策
        // 引擎的 supports_task(Decision)=true 覆盖自动放行。
        if config.task == ModelTask::Decision {
            if engine_type == EngineType::Candle {
                #[cfg(feature = "candle")]
                {
                    let engine = super::candle_decision::CandleDecisionEngine::load(config)?;
                    let any = AnyEngine::CandleDecision(engine);
                    ensure_supports_task(&any, config)?;
                    return Ok(any);
                }
                // candle 后端未编译时显性拒绝（禁止静默落入 onnx 管线——
                // engine_type 是用户显式选择，与 AnyEngine::new 同口径）
                #[cfg(not(feature = "candle"))]
                {
                    return Err(super::candle_not_compiled_error());
                }
            }
            #[cfg(feature = "onnx")]
            {
                let engine = super::decision::DecisionPipeline::load(config)?;
                let any = AnyEngine::Decision(engine);
                ensure_supports_task(&any, config)?;
                return Ok(any);
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-decision-no-onnx-feature",
                )));
            }
        }
        // task=translate 路由：seq2seq 翻译仅 onnx 实现（Marian/opus-mt
        // bundle，ort 双 Session 贪心解码）。candle 显性拒绝并给出换引擎
        // 提示；ensure_supports_task 因 supports_task(Translate)=true 覆盖
        // 自动放行。
        if config.task == ModelTask::Translate {
            if engine_type == EngineType::Candle {
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-translate-onnx-only",
                )));
            }
            #[cfg(feature = "onnx")]
            {
                let engine = super::onnx_translate::OnnxTranslationEngine::load(config)?;
                let any = AnyEngine::OnnxTranslate(engine);
                ensure_supports_task(&any, config)?;
                return Ok(any);
            }
            #[cfg(not(feature = "onnx"))]
            {
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-translate-no-onnx-feature",
                )));
            }
        }
        // task=transcribe 路由：Qwen3-ASR 三图仅 onnx 实现（需 feature `transcribe`）。
        // candle 显性拒绝并给出换引擎提示；feature 未启用 → ConfigError 提示重建。
        if config.task == ModelTask::Transcribe {
            if engine_type == EngineType::Candle {
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-transcribe-onnx-only",
                )));
            }
            #[cfg(feature = "transcribe")]
            {
                let engine = super::transcribe::engine::Qwen3AsrEngine::load(config)?;
                let any = AnyEngine::Transcribe(engine);
                ensure_supports_task(&any, config)?;
                return Ok(any);
            }
            #[cfg(not(feature = "transcribe"))]
            {
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-transcribe-no-transcribe-feature",
                )));
            }
        }
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
                return Err(VecboostError::ConfigError(crate::i18n::tr(
                    "factory-quantized-no-feature",
                )));
            }
        }
        let precision = Precision::Fp32;
        let engine = AnyEngine::new(config, engine_type, precision)?;
        // fail-fast：配置任务维度超出引擎能力时创建即失败，
        // 而非延迟到 decide 调用才以 4xx 暴露
        ensure_supports_task(&engine, config)?;
        Ok(engine)
    }
}

/// fail-fast 校验：引擎不支持配置的任务维度时报 UnsupportedTask（显性失败，
/// 禁止静默加载后由调用时 4xx 兜底）。
///
/// 故意在 load 之后（而非 load 前按 config.task 静态短路）校验：把它固化
/// 为工厂前置契约，未来新增 task 变体时按实例能力作答不会被误拒。
/// 当前 embedding 路径上 `supports_task` 恒真（task=Decision 已被分派臂
/// 截走），本防线作为契约保留。
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
            decision_params: None,
            translation_params: None,
            transcription_params: None,
        }
    }

    #[test]
    #[cfg(feature = "candle")]
    fn test_create_candle_returns_error_for_missing_model() {
        let config = test_config();
        let result = EngineFactory::create(EngineType::Candle, &config);
        // Candle 引擎会因模型路径不存在而返回错误
        assert!(result.is_err());
    }

    /// task=embedding（默认）的既有加载路径不受 fail-fast 校验影响
    #[cfg(feature = "candle")]
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

    /// task=decision + engine_type=onnx 走 ort 决策管线分派臂：空 bundle 时
    /// 报 ModelLoadError（bundle 探测失败）而非 UnsupportedTask——证明分派
    /// 到达 DecisionPipeline::load，而非落进 embedding 引擎的 fail-fast
    /// （AnyEngine::Decision 曾全库零构造点，全链在线闸门不可达）。
    /// 探测失败发生在 Session 构建之前，本测试不触发 ort。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_create_decision_task_dispatches_to_onnx_pipeline() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Decision;
        config.model_path = dir.path().to_path_buf();
        let result = EngineFactory::create(EngineType::Onnx, &config);
        match result {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("model.onnx"),
                    "分派臂必须报 bundle 探测错误，msg={msg}"
                );
            }
            Err(VecboostError::UnsupportedTask(msg)) => {
                panic!("分派臂接线后 task=decision 不得落入 embedding 引擎 fail-fast：{msg}")
            }
            other => panic!("期望 ModelLoadError，got {:?}", other.err()),
        }
    }

    /// task=translate + engine_type=onnx 走翻译引擎分派臂：空目录时报
    /// bundle 探测错误（ModelLoadError）而非 UnsupportedTask——钉住分派臂
    /// 接线（探针错误信息含 encoder 候选路径，探测发生在 Session 构建前，
    /// 本测试不触发 ort）。
    #[cfg(feature = "onnx")]
    #[test]
    fn test_create_translate_task_dispatches_to_onnx_engine() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Translate;
        config.model_path = dir.path().to_path_buf();
        let result = EngineFactory::create(EngineType::Onnx, &config);
        match result {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("onnx/encoder_model.onnx"),
                    "分派臂必须报翻译 bundle 探测错误，msg={msg}"
                );
            }
            Err(VecboostError::UnsupportedTask(msg)) => {
                panic!("分派臂接线后 task=translate 不得落入 embedding 引擎 fail-fast：{msg}")
            }
            other => panic!("期望 ModelLoadError，got {:?}", other.err()),
        }
    }

    /// task=translate + engine_type=candle 显性拒绝：翻译无 candle 实现，
    /// ConfigError 必须携带 feature/重建提示（禁止静默落入 embedding 引擎
    /// 后在调用期以 UnsupportedTask 暴露）。
    /// task=transcribe + engine_type=onnx 走转录引擎分派臂：空 bundle 时报
    /// ModelLoadError（Qwen3-ASR 三图候选加载失败）而非 UnsupportedTask——钉住
    /// 分派臂接线（同 decision/translate 先例）。
    #[cfg(feature = "transcribe")]
    #[test]
    fn test_create_transcribe_task_dispatches_to_onnx_engine() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Transcribe;
        config.model_path = dir.path().to_path_buf();
        let result = EngineFactory::create(EngineType::Onnx, &config);
        match result {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("qwen3-conv_frontend"),
                    "分派臂必须报转录 bundle 加载错误（含候选路径），msg={msg}"
                );
            }
            Err(VecboostError::UnsupportedTask(msg)) => {
                panic!("分派臂接线后 task=transcribe 不得落入 embedding 引擎 fail-fast：{msg}")
            }
            other => panic!("期望 ModelLoadError，got {:?}", other.err()),
        }
    }

    /// task=transcribe + engine_type=candle 显性拒绝：转录无 candle 实现，
    /// ConfigError 必须携带换引擎提示（禁止静默落入 embedding 引擎）。
    #[test]
    fn test_create_transcribe_task_rejects_candle_engine() {
        crate::i18n::init();
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Transcribe;
        config.model_path = PathBuf::from("models/qwen3-asr");
        let result = EngineFactory::create(EngineType::Candle, &config);
        match result {
            Err(VecboostError::ConfigError(msg)) => {
                assert!(
                    msg.contains(&crate::i18n::tr("factory-transcribe-onnx-only")),
                    "错误必须携带换引擎提示，msg={msg}"
                );
                assert!(
                    msg.contains("engine_type"),
                    "错误必须指明 engine_type 配置项，msg={msg}"
                );
            }
            other => panic!(
                "期望 ConfigError（转录不支持 candle），got {:?}",
                other.err()
            ),
        }
    }

    #[test]
    fn test_create_translate_task_rejects_candle_engine() {
        crate::i18n::init();
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Translate;
        config.model_path = PathBuf::from("models/opus-mt-en-zh");
        let result = EngineFactory::create(EngineType::Candle, &config);
        match result {
            Err(VecboostError::ConfigError(msg)) => {
                assert!(
                    msg.contains(&crate::i18n::tr("factory-translate-onnx-only")),
                    "错误必须携带换引擎提示，msg={msg}"
                );
                assert!(
                    msg.contains("engine_type"),
                    "错误必须指明 engine_type 配置项，msg={msg}"
                );
            }
            other => panic!(
                "期望 ConfigError（翻译不支持 candle），got {:?}",
                other.err()
            ),
        }
    }

    /// task=decision + engine_type=candle 走 candle 原生决策头分派臂（无
    /// onnx runtime 依赖）：空目录时报 candle 加载路径的目录探测错误而非
    /// UnsupportedTask——钉住 engine_type 在决策任务下的后端分派语义。
    #[cfg(feature = "candle")]
    #[test]
    fn test_create_decision_task_dispatches_to_candle_engine() {
        // 断言锚 i18n 渲染后的英文子串：显式 init 消除测试并行顺序依赖
        // （同文件 test_create_transcribe_task_rejects_candle_engine 先例）
        crate::i18n::init();
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Decision;
        config.model_path = dir.path().to_path_buf();
        let result = EngineFactory::create(EngineType::Candle, &config);
        match result {
            Err(VecboostError::ModelLoadError(msg)) => {
                assert!(
                    msg.contains("checkpoint not found"),
                    "分派臂必须报 candle 决策引擎资产探测错误，msg={msg}"
                );
            }
            Err(VecboostError::UnsupportedTask(msg)) => {
                panic!("candle 决策分派臂不得落入 embedding 引擎 fail-fast：{msg}")
            }
            other => panic!("期望 ModelLoadError，got {:?}", other.err()),
        }
    }

    /// 离线端到端：合成 checkpoint bundle（tiny 权重 + 官方 5.0 嵌套
    /// rope_parameters 格式 encoder config + WordLevel tokenizer）→
    /// EngineFactory 按 task=Decision + engine_type=candle 分派成功创建 →
    /// decide 三题型各一答案（qtype 0/1/2 wire 三分支），不产向量/不报
    /// rerank（诚实语义与 DecisionPipeline 同款）。
    #[cfg(feature = "candle")]
    #[test]
    fn test_create_candle_decision_engine_end_to_end() {
        use crate::domain::{DecisionAnswerBody, DecisionRequest};
        use crate::engine::InferenceEngine;
        let dir = tempfile::tempdir().expect("tempdir");
        let bundle = crate::engine::candle_decision::test_support::write_bundle(dir.path());
        let mut config = test_config();
        config.task = crate::config::model::ModelTask::Decision;
        config.model_path = bundle;
        let engine = EngineFactory::create(EngineType::Candle, &config)
            .expect("合成 bundle 必须分派成功创建 candle 决策引擎");
        assert!(engine.supports_task(crate::config::model::ModelTask::Decision));
        assert!(!engine.supports_task(crate::config::model::ModelTask::Embedding));
        let req: DecisionRequest = serde_json::from_str(
            r#"{"state":"hello","questions":[
                {"name":"destination","qtype":"choice","instructions":"pick one","options":["beach","mountain"]},
                {"name":"urgency","qtype":"score","instructions":"rate urgency"},
                {"name":"churn_risk","qtype":"noul","instructions":"churning?"}
            ]}"#,
        )
        .expect("guard request");
        let resp = engine.decide(&req).expect("decide");
        assert_eq!(resp.answers.len(), 3, "三题型各一答案");
        assert!(matches!(
            resp.answers[0].answer,
            DecisionAnswerBody::Choice { .. }
        ));
        assert!(matches!(
            resp.answers[1].answer,
            DecisionAnswerBody::Score { .. }
        ));
        assert!(matches!(
            resp.answers[2].answer,
            DecisionAnswerBody::Noul { .. }
        ));
        assert!(!engine.supports_rerank(), "决策引擎不得谎报 rerank 能力");
        match engine.embed("text") {
            Err(VecboostError::UnsupportedTask(_)) => {}
            other => panic!("决策引擎 embed 必须显性 UnsupportedTask，got {other:?}"),
        }
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
        crate::i18n::init();
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
