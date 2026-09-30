// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Decision 服务实现

use crate::config::model::{ModelConfig, ModelTask};
use crate::domain::{DecisionRequest, DecisionResponse};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

pub struct DecisionService {
    engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    /// 预构建装配随引擎克隆的模型元信息（main.rs 启动注入）；当前无读取方，
    /// 保留供引擎元信息消费接入，不作为行为开关
    #[allow(dead_code)]
    model_config: Option<ModelConfig>,
}

impl DecisionService {
    pub fn new(
        engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
        model_config: Option<ModelConfig>,
    ) -> Self {
        Self {
            engine,
            model_config,
        }
    }

    /// 替换底层引擎——EmbeddingService::switch_model 成功后由 API 层传播。
    /// 不替换则切模型后 decision 仍用旧引擎作答（审计 D27 同源缺陷：
    /// rerank 侧已修，决策服务接入时即内置传播）。
    pub fn replace_engine(&mut self, new_engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>>) {
        self.engine = new_engine;
    }

    /// 执行决策推理：对 state 回答一组 choice/score/noul 问题。
    ///
    /// 校验链（先门后调）：state/questions 非空与数量上限（本层，i18n
    /// 文案）→ [`DecisionRequest::validate`](crate::domain::DecisionRequest::validate)
    /// （choice 必须带非空 options、score/noul 不得携带、options 查重、
    /// 长度/总量预算与控制字符防线，域层单一实现）→ 引擎能力门
    /// `supports_task(Decision)`（false 报 UnsupportedTask，调用方按 400
    /// 语义换端点/模型）→ 经 trait `decide` 分发。
    ///
    /// decide 为 CPU 密集阻塞推理（单请求最多 32 题的 5 张量 batch，数十至
    /// 数百毫秒），按 [`InferenceEngine::decide`] 的调用方契约以
    /// `spawn_blocking` 移出 tokio worker 调度面——并发决策请求在引擎内
    /// Session 互斥锁上排队时，阻塞的是 blocking 池线程而非 worker，embed/
    /// 健康检查等端点不被饿死；闭包内 `blocking_read` 与 gate 读锁分离，
    /// gate 与推理之间引擎被 replace_engine 替换时本请求走新引擎（与
    /// switch 传播语义一致）。JoinError（引擎 panic）显性映射
    /// InferenceError，不吞错；wire 文案为固定脱敏前缀，panic 载荷原文
    /// 只入服务端日志（CWE-209）。
    pub async fn process_decision(
        &self,
        req: DecisionRequest,
        max_questions: usize,
    ) -> Result<DecisionResponse, VecboostError> {
        if req.state.is_null() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("decision-empty-state").to_string(),
            ));
        }
        if req.questions.is_empty() {
            return Err(VecboostError::InvalidInput(
                crate::i18n::tr("decision-empty-questions").to_string(),
            ));
        }
        if req.questions.len() > max_questions {
            return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
                "decision-too-many-questions",
                crate::i18n::tr_args(&[
                    ("count", &req.questions.len().to_string()),
                    ("max", &max_questions.to_string()),
                ]),
            )));
        }
        req.validate()?;

        // 热路径埋点：请求/问题计数（校验受理后计数，400 拒绝不计；
        // 全局 collector 未设置时零开销跳过——library 模式/单测环境）
        #[cfg(feature = "http")]
        if let Some(collector) = crate::metrics::prometheus_exporter::global_collector() {
            collector.inc_decision_requests();
            collector.add_decision_questions(req.questions.len());
        }

        let start = Instant::now();
        let engine = self.engine.read().await;
        if !engine.supports_task(ModelTask::Decision) {
            return Err(VecboostError::unsupported_task(
                crate::i18n::tr("decision-unsupported").to_string(),
            ));
        }
        drop(engine);

        let engine = self.engine.clone();
        let mut response = tokio::task::spawn_blocking(move || {
            let guard = engine.blocking_read();
            guard.decide(&req)
        })
        .await
        .map_err(|e| {
            // JoinError 的 Display 携带 panic 载荷原文（tokio
            // runtime/task/error.rs 的 Panic 分支），经 to_api_error 的 500
            // catch-all 会直达响应体（CWE-209 信息暴露面）：wire 侧只回
            // 可定位的固定文案，载荷与 task id 全量留在服务端日志
            log::error!("decision inference task failed: {e}");
            VecboostError::InferenceError(
                "decision inference task failed (engine task panicked); see server logs"
                    .to_string(),
            )
        })??;
        response.processing_time_ms = start.elapsed().as_millis();
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use crate::domain::{DecisionAnswer, DecisionAnswerBody, DecisionQuestion, QuestionType};
    use async_trait::async_trait;
    use std::collections::BTreeMap;

    /// 固定作答的 mock 决策引擎：按 qtype 构造确定性答案，覆盖自身 decide
    /// 验证 trait 分发契约（服务经 trait 调用而非引擎固有方法）。
    struct MockDecisionEngine {
        p_true: f32,
    }

    #[async_trait]
    impl InferenceEngine for MockDecisionEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 8])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        fn supports_task(&self, task: ModelTask) -> bool {
            matches!(task, ModelTask::Embedding | ModelTask::Decision)
        }

        fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
            let answers = req
                .questions
                .iter()
                .map(|q| DecisionAnswer {
                    question: q.name.clone(),
                    answer: match q.qtype {
                        QuestionType::Choice => DecisionAnswerBody::Choice {
                            index: 0,
                            option: q.options.first().cloned().unwrap_or_default(),
                            probabilities: q
                                .options
                                .iter()
                                .map(|o| (o.clone(), 1.0 / q.options.len().max(1) as f32))
                                .collect::<BTreeMap<_, _>>(),
                        },
                        QuestionType::Score => DecisionAnswerBody::Score {
                            expected: 3.0,
                            distribution: (1..=5)
                                .map(|i| (i.to_string(), 0.2))
                                .collect::<BTreeMap<_, _>>(),
                        },
                        QuestionType::Noul => DecisionAnswerBody::Noul {
                            p_true: self.p_true,
                        },
                    },
                })
                .collect();
            Ok(DecisionResponse {
                answers,
                processing_time_ms: 0,
            })
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// 不支持 decision 任务的引擎（trait 默认 supports_task 仅 Embedding）
    struct EmbeddingOnlyEngine;

    #[async_trait]
    impl InferenceEngine for EmbeddingOnlyEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 8])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_service(p_true: f32) -> DecisionService {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockDecisionEngine { p_true }));
        DecisionService::new(engine, None)
    }

    fn choice_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "destination".to_string(),
            qtype: QuestionType::Choice,
            instructions: "pick one".to_string(),
            options: vec!["beach".to_string(), "mountain".to_string()],
        }
    }

    fn score_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "budget".to_string(),
            qtype: QuestionType::Score,
            instructions: "rate 1-5".to_string(),
            options: vec![],
        }
    }

    fn noul_q() -> DecisionQuestion {
        DecisionQuestion {
            name: "confident".to_string(),
            qtype: QuestionType::Noul,
            instructions: "state your p(true)".to_string(),
            options: vec![],
        }
    }

    fn mixed_request() -> DecisionRequest {
        DecisionRequest {
            state: serde_json::json!({"topic": "vacation"}),
            questions: vec![choice_q(), score_q(), noul_q()],
        }
    }

    // -------------------------------------------------------------------------
    // 全题型全链（choice / score / noul 逐题型断言）
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_decision_answers_all_three_qtypes() {
        crate::i18n::init();
        let service = make_service(0.7);
        let resp = service.process_decision(mixed_request(), 32).await.unwrap();

        assert_eq!(resp.answers.len(), 3, "三题型必须逐一作答");
        assert_eq!(resp.answers[0].question, "destination");
        match &resp.answers[0].answer {
            DecisionAnswerBody::Choice {
                index,
                option,
                probabilities,
            } => {
                assert_eq!(*index, 0);
                assert_eq!(option, "beach");
                assert_eq!(probabilities.len(), 2, "choice 必须回各选项概率");
                let sum: f32 = probabilities.values().sum();
                assert!((sum - 1.0).abs() < 1e-5, "概率表应归一，sum={sum}");
            }
            other => panic!("第一题应为 choice，got {other:?}"),
        }
        match &resp.answers[1].answer {
            DecisionAnswerBody::Score {
                expected,
                distribution,
            } => {
                assert!((expected - 3.0).abs() < 1e-5, "score 必须回期望等级");
                assert_eq!(distribution.len(), 5, "score 必须回完整分布");
                let sum: f32 = distribution.values().sum();
                assert!((sum - 1.0).abs() < 1e-5, "分布应归一，sum={sum}");
            }
            other => panic!("第二题应为 score，got {other:?}"),
        }
        match &resp.answers[2].answer {
            DecisionAnswerBody::Noul { p_true } => {
                assert!((p_true - 0.7).abs() < 1e-5, "noul 必须回 P(true)");
            }
            other => panic!("第三题应为 noul，got {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // 校验失败分支（InvalidInput）
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_decision_rejects_null_state() {
        crate::i18n::init();
        let service = make_service(0.7);
        let mut req = mixed_request();
        req.state = serde_json::Value::Null;
        let err = service.process_decision(req, 32).await.unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("state"), "got: {msg}");
            }
            other => panic!("Expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_decision_rejects_empty_questions() {
        crate::i18n::init();
        let service = make_service(0.7);
        let req = DecisionRequest {
            state: serde_json::json!({}),
            questions: vec![],
        };
        let err = service.process_decision(req, 32).await.unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("questions"), "got: {msg}");
            }
            other => panic!("Expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_decision_rejects_too_many_questions() {
        crate::i18n::init();
        let service = make_service(0.7);
        let mut req = mixed_request();
        req.questions = (0..33)
            .map(|i| DecisionQuestion {
                name: format!("q{i}"),
                qtype: QuestionType::Noul,
                instructions: "x".to_string(),
                options: vec![],
            })
            .collect();
        let err = service.process_decision(req, 32).await.unwrap_err();
        match err {
            VecboostError::InvalidInput(msg) => {
                assert!(msg.to_lowercase().contains("questions"), "got: {msg}");
            }
            other => panic!("Expected InvalidInput, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_decision_respects_custom_max_questions() {
        let service = make_service(0.7);
        // max_questions 参数生效：2 问 > 上限 1 必须拒绝
        let err = service
            .process_decision(mixed_request(), 1)
            .await
            .unwrap_err();
        assert!(matches!(err, VecboostError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn test_process_decision_rejects_choice_without_options() {
        crate::i18n::init();
        let service = make_service(0.7);
        let mut req = mixed_request();
        req.questions[0].options.clear();
        let err = service.process_decision(req, 32).await.unwrap_err();
        // 域层 validate 以 ValidationError 显性返回（to_api_error 同映射 400）
        assert!(
            matches!(err, VecboostError::ValidationError(_)),
            "choice 题空 options 必须拒绝（域层一致性经服务强制），got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_process_decision_rejects_duplicate_options() {
        crate::i18n::init();
        let service = make_service(0.7);
        let mut req = mixed_request();
        req.questions[0].options[1] = "beach".to_string();
        let err = service.process_decision(req, 32).await.unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                assert!(msg.contains("duplicate option"), "got: {msg}");
            }
            other => panic!("Expected ValidationError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_process_decision_rejects_score_with_options() {
        crate::i18n::init();
        let service = make_service(0.7);
        let mut req = mixed_request();
        req.questions[1].options = vec!["1".to_string()];
        let err = service.process_decision(req, 32).await.unwrap_err();
        assert!(
            matches!(err, VecboostError::ValidationError(_)),
            "score 题携带 options 必须拒绝（noul 同款），got {err:?}"
        );
    }

    // -------------------------------------------------------------------------
    // 引擎能力门（先门后调）
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_process_decision_unsupported_engine_returns_unsupported_task() {
        crate::i18n::init();
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(EmbeddingOnlyEngine));
        let service = DecisionService::new(engine, None);
        let err = service
            .process_decision(mixed_request(), 32)
            .await
            .unwrap_err();
        match err {
            VecboostError::UnsupportedTask(msg) => {
                assert!(msg.contains("decision"), "got: {msg}");
            }
            other => panic!("Expected UnsupportedTask, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // 切模型传播（D27 同源缺陷回归钉，仿 rerank.rs replace 传播钉）
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_replace_engine_switches_decision_source() {
        let engine_a: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockDecisionEngine { p_true: 0.1 }));
        let mut service = DecisionService::new(engine_a, None);
        let req = DecisionRequest {
            state: serde_json::json!({}),
            questions: vec![noul_q()],
        };
        let r1 = service.process_decision(req.clone(), 32).await.unwrap();
        let DecisionAnswerBody::Noul { p_true } = &r1.answers[0].answer else {
            panic!("应为 noul 答案");
        };
        assert!((p_true - 0.1).abs() < 1e-5);

        let engine_b: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockDecisionEngine { p_true: 0.9 }));
        service.replace_engine(engine_b);
        let r2 = service.process_decision(req, 32).await.unwrap();
        let DecisionAnswerBody::Noul { p_true } = &r2.answers[0].answer else {
            panic!("应为 noul 答案");
        };
        assert!(
            (p_true - 0.9).abs() < 1e-5,
            "replace_engine 后决策必须走新引擎，p_true={p_true}"
        );
    }

    /// 引擎 decide panic 经 spawn_blocking JoinError 必须显性映射
    /// InferenceError（规则 11：不吞错、不落默认值），且 wire 文案不得
    /// 携带 panic 载荷原文（CWE-209：载荷经 to_api_error 500 catch-all
    /// 直达响应体，全量细节只允许入服务端日志）
    #[tokio::test]
    async fn test_process_decision_engine_panic_maps_to_inference_error() {
        struct PanickingDecisionEngine;

        #[async_trait]
        impl InferenceEngine for PanickingDecisionEngine {
            fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
                Ok(vec![0.0; 8])
            }

            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
                Ok(texts.iter().map(|_| vec![0.0; 8]).collect())
            }

            fn precision(&self) -> &Precision {
                &Precision::Fp32
            }

            fn supports_mixed_precision(&self) -> bool {
                false
            }

            fn supports_task(&self, task: ModelTask) -> bool {
                matches!(task, ModelTask::Embedding | ModelTask::Decision)
            }

            fn decide(&self, _req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
                panic!("engine exploded mid-inference");
            }

            async fn try_fallback_to_cpu(
                &mut self,
                _config: &ModelConfig,
            ) -> Result<(), VecboostError> {
                Ok(())
            }
        }

        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(PanickingDecisionEngine));
        let service = DecisionService::new(engine, None);
        let err = service
            .process_decision(mixed_request(), 32)
            .await
            .unwrap_err();
        match err {
            VecboostError::InferenceError(msg) => {
                assert!(
                    msg.contains("decision inference task"),
                    "JoinError 映射文案必须可定位，got: {msg}"
                );
                assert!(
                    !msg.contains("engine exploded mid-inference"),
                    "wire 文案不得携带 panic 载荷原文（CWE-209），got: {msg}"
                );
            }
            other => panic!("Expected InferenceError, got: {other:?}"),
        }
    }
}
