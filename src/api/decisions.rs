// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Decision forge handlers — HTTP/CLI/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/embedding.rs` 相同模式：协议无关的 `decisions_handler`
//! 包含业务逻辑，`forge_*` / `cli_*` / `grpc_*` 仅为薄包装。
//!
//! # 路由偏差说明（文档 4.3 写作 /v1/decisions）
//! sdforge 按 `version = 1` 统一注入 `/api/{version}` 前缀，实际路由
//! `/api/1/decisions`——与既有 embed/rerank 端点同一前缀约定，不单独引入
//! `/v1` 前缀（文档 4.3 的 `/v1/decisions` 为规格期草写）。

use crate::domain::{DecisionRequest, DecisionResponse};
use crate::error::VecboostError;
use crate::service::decision::DecisionService;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::registry::DecisionModule;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use std::sync::Arc;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use tokio::sync::RwLock;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions
// =============================================================================

pub async fn decide(
    svc: &DecisionService,
    req: DecisionRequest,
    max_questions: usize,
) -> Result<DecisionResponse, VecboostError> {
    svc.process_decision(req, max_questions).await
}

// =============================================================================
// Protocol-agnostic handlers
// =============================================================================

/// Load decision service from the global kit.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn load_decision_service() -> Result<Arc<RwLock<DecisionService>>, ApiError> {
    let st = state().map_err(to_api_error)?;
    st.kit
        .require::<DecisionModule>()
        .map_err(kit_internal_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn decisions_handler(req: DecisionRequest) -> Result<DecisionResponse, ApiError> {
    let svc = load_decision_service().await?;
    let guard = svc.read().await;
    decide(&guard, req, crate::domain::decision::MAX_QUESTIONS)
        .await
        .map_err(to_api_error)
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "decisions",
    version = 1,
    path = "/decisions",
    method = "POST",
    tool_name = "decisions",
    description = "Answer choice/score/noul questions about a decision state"
)]
pub async fn forge_decisions(req: DecisionRequest) -> Result<DecisionResponse, ApiError> {
    decisions_handler(req).await
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "decisions",
    version = 1,
    cli = true,
    description = "Answer choice/score/noul questions about a decision state"
)]
pub async fn cli_decisions(req: DecisionRequest) -> Result<DecisionResponse, ApiError> {
    decisions_handler(req).await
}

// =============================================================================
// gRPC forge handlers
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "decisions",
    version = 1,
    grpc_method = "vecboost.decide",
    description = "Answer choice/score/noul questions about a decision state"
)]
pub async fn grpc_decisions(req: DecisionRequest) -> Result<DecisionResponse, ApiError> {
    decisions_handler(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, ModelTask, Precision};
    use crate::domain::{DecisionAnswerBody, DecisionQuestion, QuestionType};
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;
    use std::collections::BTreeMap;

    /// 固定作答的 mock 决策引擎（与 service 层 mock 同构，三题型确定性作答）
    struct MockDecisionEngine;

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
                .map(|q| crate::domain::DecisionAnswer {
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
                        QuestionType::Noul => DecisionAnswerBody::Noul { p_true: 0.7 },
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

    fn make_svc() -> DecisionService {
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(MockDecisionEngine));
        DecisionService::new(engine, None)
    }

    fn mixed_request() -> DecisionRequest {
        DecisionRequest {
            state: serde_json::json!({"topic": "vacation"}),
            questions: vec![
                DecisionQuestion {
                    name: "destination".to_string(),
                    qtype: QuestionType::Choice,
                    instructions: "pick one".to_string(),
                    options: vec!["beach".to_string(), "mountain".to_string()],
                },
                DecisionQuestion {
                    name: "budget".to_string(),
                    qtype: QuestionType::Score,
                    instructions: "rate 1-5".to_string(),
                    options: vec![],
                },
                DecisionQuestion {
                    name: "confident".to_string(),
                    qtype: QuestionType::Noul,
                    instructions: "state your p(true)".to_string(),
                    options: vec![],
                },
            ],
        }
    }

    #[tokio::test]
    async fn test_sdk_decide_delegates_to_service() {
        let svc = make_svc();
        let result = decide(&svc, mixed_request(), 32).await.unwrap();
        assert_eq!(result.answers.len(), 3);
    }

    #[tokio::test]
    async fn test_sdk_decide_propagates_validation_error() {
        let svc = make_svc();
        let req = DecisionRequest {
            state: serde_json::json!({}),
            questions: vec![],
        };
        let result = decide(&svc, req, 32).await;
        assert!(result.is_err(), "空 questions 必须经服务校验拒绝");
    }

    #[tokio::test]
    async fn test_sdk_decide_wire_shape_all_three_qtypes() {
        // 三题型 wire 形态钉：internally tagged（{"type":"choice"|...}）
        let svc = make_svc();
        let resp = decide(&svc, mixed_request(), 32).await.unwrap();
        let json = serde_json::to_value(&resp).unwrap();
        let answers = json["answers"].as_array().unwrap();
        assert_eq!(answers[0]["answer"]["type"], "choice");
        assert_eq!(answers[0]["answer"]["option"], "beach");
        assert_eq!(answers[1]["answer"]["type"], "score");
        assert!(answers[1]["answer"]["expected"].is_number());
        assert_eq!(answers[2]["answer"]["type"], "noul");
        assert!(answers[2]["answer"]["p_true"].is_number());
    }
}
