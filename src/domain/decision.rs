// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Decision 领域类型域：决策推理请求/响应契约。
//!
//! 本组只定形状、serde 表示与输入校验；不含业务逻辑。serde 约定：
//! 请求侧 `qtype`/`task` 为 lowercase 枚举，答案体为 internally tagged
//! （`{"type":"choice",...}`）以钉死对外契约。

use crate::error::VecboostError;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
#[cfg(feature = "schema")]
use utoipa::ToSchema;

/// Decision 请求输入上限（与 embed 的 `MAX_TEXT_LENGTH` 同口径防线）：
/// prompt 构造按 questions×options 展开，无上限时 token 成本可被恶意放大。
/// handler 落地时必须强制走 [`DecisionRequest::validate`]，
/// 不得仅依赖 HTTP 层 body size limit。
pub const MAX_QUESTIONS: usize = 32;
pub const MAX_OPTIONS_PER_QUESTION: usize = 64;
/// 单条 name/instructions/option 的字符长度上限（同 embed 文本口径）
pub const MAX_DECISION_FIELD_LENGTH: usize = crate::utils::constants::MAX_TEXT_LENGTH;
/// state JSON 序列化字节上限
pub const MAX_STATE_BYTES: usize = 64 * 1024;

/// 决策问题类型：choice（选项作答）/ score（分值分布）/ noul（真值概率）
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DecisionQuestion {
    pub name: String,
    pub qtype: QuestionType,
    pub instructions: String,
    /// choice 型必填非空选项集；score/noul 型必须缺省或为空
    /// （跨字段一致性由 [`DecisionRequest::validate`] 强制）
    #[serde(default)]
    pub options: Vec<String>,
}

impl FromStr for DecisionQuestion {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DecisionRequest {
    /// 决策上下文（自由 JSON，由调用方与模型约定语义）
    pub state: serde_json::Value,
    pub questions: Vec<DecisionQuestion>,
}

impl FromStr for DecisionRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl DecisionRequest {
    /// 输入校验：数量/长度/字节上限 + qtype 与 options 的跨字段一致性
    /// （serde 无法表达跨字段约束，必须显式代码）。失败以
    /// `ValidationError` 显式返回，禁止静默截断或忽略。
    pub fn validate(&self) -> Result<(), VecboostError> {
        if self.questions.is_empty() {
            return Err(VecboostError::validation_error(
                "decision request must contain at least one question".to_string(),
            ));
        }
        if self.questions.len() > MAX_QUESTIONS {
            return Err(VecboostError::validation_error(format!(
                "too many questions: {} > {MAX_QUESTIONS}",
                self.questions.len()
            )));
        }
        let state_bytes = serde_json::to_vec(&self.state).map_err(|e| {
            VecboostError::validation_error(format!("state is not serializable: {e}"))
        })?;
        if state_bytes.len() > MAX_STATE_BYTES {
            return Err(VecboostError::validation_error(format!(
                "state too large: {} bytes > {MAX_STATE_BYTES}",
                state_bytes.len()
            )));
        }
        for q in &self.questions {
            if q.name.len() > MAX_DECISION_FIELD_LENGTH {
                return Err(VecboostError::validation_error(format!(
                    "question name too long: {} chars > {MAX_DECISION_FIELD_LENGTH}",
                    q.name.len()
                )));
            }
            if q.instructions.len() > MAX_DECISION_FIELD_LENGTH {
                return Err(VecboostError::validation_error(format!(
                    "question {} instructions too long: {} chars > {MAX_DECISION_FIELD_LENGTH}",
                    q.name,
                    q.instructions.len()
                )));
            }
            match q.qtype {
                QuestionType::Choice if q.options.is_empty() => {
                    return Err(VecboostError::validation_error(format!(
                        "choice question {} requires non-empty options",
                        q.name
                    )));
                }
                QuestionType::Score | QuestionType::Noul if !q.options.is_empty() => {
                    return Err(VecboostError::validation_error(format!(
                        "question {} of type {:?} must not carry options",
                        q.name, q.qtype
                    )));
                }
                _ => {}
            }
            if q.options.len() > MAX_OPTIONS_PER_QUESTION {
                return Err(VecboostError::validation_error(format!(
                    "question {} has too many options: {} > {MAX_OPTIONS_PER_QUESTION}",
                    q.name,
                    q.options.len()
                )));
            }
            if let Some(oversized) = q
                .options
                .iter()
                .find(|o| o.len() > MAX_DECISION_FIELD_LENGTH)
            {
                return Err(VecboostError::validation_error(format!(
                    "question {} has an option too long: {} chars > {MAX_DECISION_FIELD_LENGTH}",
                    q.name,
                    oversized.len()
                )));
            }
        }
        Ok(())
    }
}

/// 单个问题的答案体，按 qtype 三型。
///
/// serde 表示显式钉死为 internally tagged（`{"type":"choice",...}` 形态）——
/// 默认 externally tagged 会产出 `{"Choice":{...}}`，破坏对外契约。
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum DecisionAnswerBody {
    Choice {
        /// 命中选项下标。不变量 `index < options.len()`（options 为请求侧同名问题
        /// 的选项集）：实现 decide 的引擎在返回前必须校验；消费方跨信任边界
        /// 使用时仍应自行做范围断言，禁止裸 `options[index]`。
        index: usize,
        option: String,
        probabilities: std::collections::BTreeMap<String, f32>,
    },
    Score {
        expected: f32,
        distribution: std::collections::BTreeMap<String, f32>,
    },
    Noul {
        p_true: f32,
    },
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DecisionAnswer {
    pub question: String,
    pub answer: DecisionAnswerBody,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DecisionResponse {
    pub answers: Vec<DecisionAnswer>,
    pub processing_time_ms: u128,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // -------------------------------------------------------------------------
    // QuestionType
    // -------------------------------------------------------------------------

    #[test]
    fn test_question_type_invalid_rejected() {
        // enum 无 catch-all：非法 qtype 必须在反序列化边界被拒（对称于
        // config::model 的 test_model_task_invalid_value_rejected）
        let invalid: Result<QuestionType, _> = serde_json::from_str("\"rank\"");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_decision_request_from_str() {
        let req: DecisionRequest = r#"{
            "state": {"topic": "vacation"},
            "questions": [
                {
                    "name": "destination",
                    "qtype": "choice",
                    "instructions": "pick one",
                    "options": ["beach", "mountain"]
                },
                {"name": "budget", "qtype": "score", "instructions": "rate 1-5"}
            ]
        }"#
        .parse()
        .unwrap();
        assert_eq!(req.questions.len(), 2);
        assert_eq!(req.questions[0].qtype, QuestionType::Choice);
        assert_eq!(
            req.questions[0].options,
            vec!["beach".to_string(), "mountain".to_string()]
        );
        // options 缺省必须回落空 vec（score/noul 型无选项）
        assert!(req.questions[1].options.is_empty());
        assert_eq!(req.questions[1].qtype, QuestionType::Score);
    }

    #[test]
    fn test_decision_question_from_str() {
        let q: DecisionQuestion = r#"{
            "name": "confidence",
            "qtype": "noul",
            "instructions": "state your p(true)"
        }"#
        .parse()
        .unwrap();
        assert_eq!(q.name, "confidence");
        assert_eq!(q.qtype, QuestionType::Noul);
        assert!(q.options.is_empty());
    }

    // -------------------------------------------------------------------------
    // DecisionRequest::validate —— 输入上限 + qtype/options 跨字段一致性
    // -------------------------------------------------------------------------

    fn choice_question() -> DecisionQuestion {
        DecisionQuestion {
            name: "destination".to_string(),
            qtype: QuestionType::Choice,
            instructions: "pick one".to_string(),
            options: vec!["beach".to_string(), "mountain".to_string()],
        }
    }

    fn valid_request() -> DecisionRequest {
        DecisionRequest {
            state: serde_json::json!({"topic": "vacation"}),
            questions: vec![choice_question()],
        }
    }

    #[test]
    fn test_validate_accepts_valid_request() {
        valid_request().validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_empty_questions() {
        let req = DecisionRequest {
            state: serde_json::json!({}),
            questions: vec![],
        };
        assert!(req.validate().is_err(), "空问题集无意义，必须显式拒绝");
    }

    #[test]
    fn test_validate_rejects_too_many_questions() {
        let mut req = valid_request();
        req.questions = (0..MAX_QUESTIONS + 1)
            .map(|i| DecisionQuestion {
                name: format!("q{i}"),
                qtype: QuestionType::Noul,
                instructions: "x".to_string(),
                options: vec![],
            })
            .collect();
        let err = req.validate().unwrap_err();
        // 断言 error_detail（原始消息）而非 Display——后者依赖 i18n 初始化
        assert!(
            err.error_detail().contains("questions"),
            "err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_too_many_options() {
        let mut req = valid_request();
        req.questions[0].options = (0..MAX_OPTIONS_PER_QUESTION + 1)
            .map(|i| format!("opt{i}"))
            .collect();
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_oversized_field() {
        let mut req = valid_request();
        req.questions[0].instructions = "x".repeat(MAX_DECISION_FIELD_LENGTH + 1);
        assert!(req.validate().is_err());

        let mut req = valid_request();
        req.questions[0].name = "x".repeat(MAX_DECISION_FIELD_LENGTH + 1);
        assert!(req.validate().is_err());

        let mut req = valid_request();
        req.questions[0].options[0] = "x".repeat(MAX_DECISION_FIELD_LENGTH + 1);
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_oversized_state() {
        let mut req = valid_request();
        req.state = serde_json::json!({ "blob": "x".repeat(MAX_STATE_BYTES + 1) });
        assert!(req.validate().is_err(), "state 超字节上限必须拒绝");
    }

    #[test]
    fn test_validate_rejects_choice_with_empty_options() {
        let mut req = valid_request();
        req.questions[0].options.clear();
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("destination"),
            "错误必须携带问题名定位，err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_non_choice_with_options() {
        let mut req = valid_request();
        req.questions[0].qtype = QuestionType::Score;
        assert!(req.validate().is_err(), "score 型携带 options 必须拒绝");

        let mut req = valid_request();
        req.questions[0].qtype = QuestionType::Noul;
        assert!(req.validate().is_err(), "noul 型携带 options 必须拒绝");
    }

    // -------------------------------------------------------------------------
    // DecisionAnswerBody wire 形态（internally tagged 契约）
    // -------------------------------------------------------------------------

    #[test]
    fn test_decision_answer_body_noul_wire_shape() {
        let body = DecisionAnswerBody::Noul { p_true: 0.5 };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["type"], "noul");
        assert_eq!(v["p_true"], serde_json::json!(0.5));
    }

    #[test]
    fn test_decision_answer_body_choice_wire_shape() {
        let body = DecisionAnswerBody::Choice {
            index: 1,
            option: "mountain".to_string(),
            probabilities: BTreeMap::from([
                ("beach".to_string(), 0.25),
                ("mountain".to_string(), 0.75),
            ]),
        };
        let v = serde_json::to_value(&body).unwrap();
        // internally tagged：tag 必须是 "type" 字段而非 externally tagged 形态
        assert_eq!(v["type"], "choice");
        assert_eq!(v["index"], 1);
        assert_eq!(v["option"], "mountain");
        assert_eq!(v["probabilities"]["mountain"], serde_json::json!(0.75));
    }

    #[test]
    fn test_decision_answer_body_score_wire_shape() {
        let body = DecisionAnswerBody::Score {
            expected: 3.5,
            distribution: BTreeMap::from([("3".to_string(), 0.5), ("4".to_string(), 0.5)]),
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["type"], "score");
        assert_eq!(v["expected"], serde_json::json!(3.5));
        assert_eq!(v["distribution"]["4"], serde_json::json!(0.5));
    }

    #[test]
    fn test_decision_response_serialize() {
        let resp = DecisionResponse {
            answers: vec![DecisionAnswer {
                question: "destination".to_string(),
                answer: DecisionAnswerBody::Noul { p_true: 0.9 },
            }],
            processing_time_ms: 7,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"answers\""), "json={}", json);
        assert!(json.contains("\"processing_time_ms\":7"), "json={}", json);
    }
}
