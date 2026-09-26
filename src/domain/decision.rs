// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Decision 领域类型域：决策推理请求/响应契约。
//!
//! 本组只定形状、serde 表示与输入校验；不含业务逻辑。serde 约定：
//! 请求侧 `qtype`/`task` 为 lowercase 枚举，答案体为 internally tagged
//! （`{"type":"choice",...}`）以钉死对外契约。

use crate::error::VecboostError;
use crate::utils::validator::input::{echo, has_disallowed_control_chars};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
#[cfg(feature = "schema")]
use utoipa::ToSchema;

/// 递归检查 state 所有字符串叶子的控制字符。serde_json 只拒绝裸控制字节，
/// `\uXXXX` 转义形式可解析并还原为真实控制字符（评审 R1），
/// 故 state 与自由文本字段适用同一防线。
fn state_has_control_chars(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(s) => has_disallowed_control_chars(s),
        serde_json::Value::Array(items) => items.iter().any(state_has_control_chars),
        serde_json::Value::Object(map) => map.values().any(state_has_control_chars),
        _ => false,
    }
}

/// Decision 请求输入上限（与 embed 的 `MAX_TEXT_LENGTH` 同源默认值防线）：
/// prompt 构造按 questions×options 展开，无上限时 token 成本可被恶意放大。
/// handler 落地时必须强制走 [`DecisionRequest::validate`]，
/// 不得仅依赖 HTTP 层 body size limit。
pub const MAX_QUESTIONS: usize = 32;
pub const MAX_OPTIONS_PER_QUESTION: usize = 64;
/// 单条 name/instructions/option 的字符长度上限（口径与 embed 校验一致，
/// 按 `chars().count()` 计而非字节数；默认值同源 `constants::MAX_TEXT_LENGTH`）
pub const MAX_DECISION_FIELD_LENGTH: usize = crate::utils::constants::MAX_TEXT_LENGTH;
/// state JSON 序列化字节上限
pub const MAX_STATE_BYTES: usize = 64 * 1024;
/// prompt 展开总量预算（字节）：sum(name + instructions + options)。
/// 单维度上限的合法乘积（32 问 × 64 选项 × 10K 字符）理论可达 ~20MB prompt，
/// 该跨字段总量防线将其压至 128KB（数万 token 量级）。
/// 字节口径仅为粗防线：tokenizer 精确 token 预算与 head 截断由 prompt
/// 构造方（decide 引擎落地组）负责，本层不重复实现——落地入口与输入契约见
/// [`crate::engine::InferenceEngine::decide`]，CJK 输入 128KB 可达数万
/// token，落地时遗漏精确预算可超模型窗口。
pub const MAX_TOTAL_PROMPT_BYTES: usize = 128 * 1024;

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
    /// 红线同 [`DecisionRequest::from_str`]：handler 不得把 serde 错误原样透传客户端。
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
    /// # 红线（handler 落地时）
    /// `Err = serde_json::Error` 携带内部结构名与字节偏移，handler 必须
    /// 映射为通用 `VecboostError::validation_error` 文案，禁止原样透传客户端。
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl DecisionRequest {
    /// 输入校验：数量/长度/字节上限 + qtype 与 options 的跨字段一致性 +
    /// prompt 展开总量预算 + 控制字符拒绝（与 embed 校验同款防线，
    /// `\t` `\n` `\r` 豁免）。state 的字符串叶子同样递归过控制字符防线：
    /// serde_json 只拒绝裸控制字节，`\uXXXX` 转义可还原真实控制字符。
    /// serde 无法表达跨字段约束，必须显式代码。失败以
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
        // 封顶计数序列化：命中 MAX_STATE_BYTES 即停，不完整展开 state
        let mut counter = CappedWriter {
            written: 0,
            cap: MAX_STATE_BYTES,
        };
        if let Err(e) = serde_json::to_writer(&mut counter, &self.state) {
            return Err(VecboostError::validation_error(
                if counter.written > MAX_STATE_BYTES {
                    format!("state too large: > {MAX_STATE_BYTES} bytes")
                } else {
                    format!("state is not serializable: {e}")
                },
            ));
        }
        if state_has_control_chars(&self.state) {
            return Err(VecboostError::validation_error(
                "state contains control characters in string values".to_string(),
            ));
        }
        let mut total_prompt_bytes: usize = 0;
        let mut seen_names = std::collections::HashSet::new();
        for (question_index, q) in self.questions.iter().enumerate() {
            // answer 按 question 名回显：空/空白 name 使消费方无法对应问题，
            // 重复 name 使按名索引有歧义（协议按 questions 数组下标对应，
            // name 仅作诊断与路由提示）
            if q.name.trim().is_empty() {
                return Err(VecboostError::validation_error(
                    "question name must not be empty or blank".to_string(),
                ));
            }
            let name_chars = q.name.chars().count();
            if name_chars > MAX_DECISION_FIELD_LENGTH {
                return Err(VecboostError::validation_error(format!(
                    "question name too long: {name_chars} chars > {MAX_DECISION_FIELD_LENGTH}"
                )));
            }
            if has_disallowed_control_chars(&q.name) {
                // 不回显原文：控制字符直入 wire/日志即日志注入面（评审 R2）
                return Err(VecboostError::validation_error(format!(
                    "question at index {question_index} name contains control characters"
                )));
            }
            if !seen_names.insert(q.name.as_str()) {
                return Err(VecboostError::validation_error(format!(
                    "duplicate question name: {}",
                    echo(&q.name)
                )));
            }
            let instructions_chars = q.instructions.chars().count();
            if instructions_chars > MAX_DECISION_FIELD_LENGTH {
                return Err(VecboostError::validation_error(format!(
                    "question {} instructions too long: {instructions_chars} chars > \
                     {MAX_DECISION_FIELD_LENGTH}",
                    echo(&q.name)
                )));
            }
            if has_disallowed_control_chars(&q.instructions) {
                return Err(VecboostError::validation_error(format!(
                    "question at index {question_index} instructions contain control characters"
                )));
            }
            match q.qtype {
                QuestionType::Choice if q.options.is_empty() => {
                    return Err(VecboostError::validation_error(format!(
                        "choice question {} requires non-empty options",
                        echo(&q.name)
                    )));
                }
                QuestionType::Score | QuestionType::Noul if !q.options.is_empty() => {
                    return Err(VecboostError::validation_error(format!(
                        "question {} of type {:?} must not carry options",
                        echo(&q.name),
                        q.qtype
                    )));
                }
                _ => {}
            }
            if q.options.len() > MAX_OPTIONS_PER_QUESTION {
                return Err(VecboostError::validation_error(format!(
                    "question {} has too many options: {} > {MAX_OPTIONS_PER_QUESTION}",
                    echo(&q.name),
                    q.options.len()
                )));
            }
            let mut options_bytes: usize = 0;
            let mut seen_options = std::collections::HashSet::new();
            for option in &q.options {
                // 与 name 同口径：option 按字符串匹配消费，空/纯空白 option
                // 使匹配结果有歧义（answer 的 index 权威，但空白串本身无意义）
                if option.trim().is_empty() {
                    return Err(VecboostError::validation_error(format!(
                        "question {} has an empty or blank option",
                        echo(&q.name)
                    )));
                }
                let option_chars = option.chars().count();
                if option_chars > MAX_DECISION_FIELD_LENGTH {
                    return Err(VecboostError::validation_error(format!(
                        "question {} has an option too long: {option_chars} chars > \
                         {MAX_DECISION_FIELD_LENGTH}",
                        echo(&q.name)
                    )));
                }
                if has_disallowed_control_chars(option) {
                    return Err(VecboostError::validation_error(format!(
                        "question at index {question_index} has an option containing control characters"
                    )));
                }
                // 重复 option 使概率表键合并/索引匹配歧义——校验层显性拒绝，
                // 避免合法请求白跑一次推理后才在引擎层被拒
                if !seen_options.insert(option.as_str()) {
                    return Err(VecboostError::validation_error(format!(
                        "question {} has duplicate option {}",
                        echo(&q.name),
                        echo(option)
                    )));
                }
                options_bytes += option.len();
            }
            // prompt 展开总量（字节）：单维度合法的乘积在此被跨字段预算截停
            total_prompt_bytes += q.name.len() + q.instructions.len() + options_bytes;
            if total_prompt_bytes > MAX_TOTAL_PROMPT_BYTES {
                return Err(VecboostError::validation_error(format!(
                    "total prompt budget exceeded: {total_prompt_bytes} bytes > \
                     {MAX_TOTAL_PROMPT_BYTES}"
                )));
            }
        }
        Ok(())
    }
}

/// 封顶计数 writer：累计字节数超过 cap 即返回错误，
/// 使 state 大小校验在命中上限时停止而非完整序列化（评审 R10）
struct CappedWriter {
    written: usize,
    cap: usize,
}

impl std::io::Write for CappedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written = self.written.saturating_add(buf.len());
        if self.written > self.cap {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "state exceeds budget",
            ));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
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
    fn test_validate_rejects_total_prompt_over_budget() {
        // 乘积防线（评审 R1）：每个问题单独看均合法（noul 无 options、
        // instructions 远小于单字段上限），但 questions×字段总量超预算必须拒绝
        let mut req = valid_request();
        req.questions.clear();
        for i in 0..MAX_QUESTIONS {
            req.questions.push(DecisionQuestion {
                name: format!("q{i}"),
                qtype: QuestionType::Noul,
                instructions: "x".repeat(5000),
                options: vec![],
            });
        }
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("total"),
            "总量超预算错误必须标明 total，err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_at_total_budget_boundary_passes() {
        // 32 问 × 4000 字节 instructions（每问均低于单字段上限），
        // 总量 ~128KB 恰在预算内应通过；与超预算测试构成边界两侧
        let mut req = valid_request();
        req.questions.clear();
        for i in 0..MAX_QUESTIONS {
            req.questions.push(DecisionQuestion {
                name: format!("q{i}"),
                qtype: QuestionType::Noul,
                instructions: "x".repeat(4000),
                options: vec![],
            });
        }
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_char_count_not_bytes() {
        // 口径与 embed 校验一致按字符数计（评审 R2/R8）：多字节字符
        // 不因字节数提前触发上限（10000 个 3 字节字符 = 30000 字节）
        let mut req = valid_request();
        req.questions[0].instructions = "水".repeat(MAX_DECISION_FIELD_LENGTH);
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_control_chars() {
        // 与 embed 侧 validate_text_content 同款防线（评审 R3）：
        // NUL/C0 拒绝，\t \n \r 豁免
        let mut req = valid_request();
        req.questions[0].name.push('\u{0}');
        assert!(req.validate().is_err(), "name 含 NUL 必须拒绝");

        let mut req = valid_request();
        req.questions[0].instructions.push('\u{1}');
        assert!(req.validate().is_err(), "instructions 含 C0 必须拒绝");

        let mut req = valid_request();
        req.questions[0].options[0].push('\u{0}');
        assert!(req.validate().is_err(), "options 含 NUL 必须拒绝");
    }

    #[test]
    fn test_validate_allows_tab_newline_carriage_return() {
        let mut req = valid_request();
        req.questions[0].instructions = "line1\nline2\ttabbed\rend".to_string();
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_rejects_blank_name() {
        // answer 按 question 名回显，空/纯空白 name 使消费方无法对应问题（评审 R9）
        let mut req = valid_request();
        req.questions[0].name = "   ".to_string();
        assert!(req.validate().is_err(), "纯空白 name 必须拒绝");

        let mut req = valid_request();
        req.questions[0].name = String::new();
        assert!(req.validate().is_err(), "空 name 必须拒绝");
    }

    #[test]
    fn test_validate_rejects_duplicate_names() {
        // 重复 name 产生两个同名 answer，按名索引有歧义（评审 R9）
        let mut req = valid_request();
        req.questions.push(choice_question());
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("destination"),
            "重复 name 错误必须携带问题名，err={}",
            err.error_detail()
        );
    }

    #[test]
    fn test_validate_rejects_state_control_chars_via_escape() {
        // serde_json 只拒裸控制字节：\u0000 转义可解析并还原真实控制字符
        // （评审 R1），state 字符串叶子必须同样过控制字符防线
        let mut req = valid_request();
        req.state = serde_json::from_str(r#"{"note":"line1\u0000line2"}"#).unwrap();
        assert!(
            req.validate().is_err(),
            "state 字符串叶子含控制字符必须拒绝"
        );
    }

    #[test]
    fn test_validate_state_control_chars_check_recurses() {
        // 嵌套 object/array 内的字符串叶子同样覆盖
        let mut req = valid_request();
        req.state = serde_json::json!({
            "outer": {"arr": ["ok", 1, true, {"deep": "bad\u{0}char"}]}
        });
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_clean_state() {
        // 合法 state（含数字/布尔/嵌套）不受控制字符检查影响
        let mut req = valid_request();
        req.state = serde_json::json!({"topic": "vacation", "budget": [1, 2.5, true]});
        req.validate().unwrap();
    }

    #[test]
    fn test_validate_error_echo_single_line() {
        // echo 必须字面量化豁免集字符（\t\n\r）：换行 name 不得把伪造日志行
        // 带入错误 detail（日志注入面，评审 R1 四轮）
        let mut req = valid_request();
        let evil_name = "evil\nFAKE LOG LINE".to_string();
        req.questions[0].name = evil_name.clone();
        req.questions.push(DecisionQuestion {
            name: evil_name,
            qtype: QuestionType::Noul,
            instructions: "x".to_string(),
            options: vec![],
        });
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(
            !detail.contains('\n'),
            "回显必须单行化，detail={:?}",
            detail
        );
        assert!(
            detail.contains("\\n"),
            "换行应字面量化，detail={:?}",
            detail
        );
    }

    #[test]
    fn test_validate_error_echo_truncated() {
        // 错误 detail 回显截断（评审 R2）：10K 字符 name 不得被完整回显
        let long_name = "n".repeat(MAX_DECISION_FIELD_LENGTH);
        let mut req = valid_request();
        req.questions[0].name = long_name.clone();
        req.questions.push(DecisionQuestion {
            name: long_name,
            qtype: QuestionType::Noul,
            instructions: "x".to_string(),
            options: vec![],
        });
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().len() < 200,
            "错误回显必须截断，len={}",
            err.error_detail().len()
        );
    }

    #[test]
    fn test_validate_control_char_error_does_not_echo_raw() {
        // 控制字符臂不得回显原文（控制字符直入日志 = 日志注入面，评审 R2）：
        // detail 必须不含任何控制字符（含构造输入里的 NUL）
        let mut req = valid_request();
        req.questions[0].name = "ab_cd".repeat(100).replace('_', "\u{0}");
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(
            !detail.chars().any(|c| c.is_control()),
            "控制字符错误不得回显原文，detail={:?}",
            detail
        );
        assert!(!detail.contains("ab_cd"), "原文不得出现在 detail");
    }

    #[test]
    fn test_validate_rejects_duplicate_options() {
        // 重复 option 使概率表键合并/索引匹配歧义：校验层前置拒绝（引擎层
        // choice_answer 保留防御性查重），错误在推理前显性返回
        let mut req = valid_request();
        req.questions[0].options[1] = "beach".to_string();
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(
            detail.contains("duplicate option"),
            "重复 option 必须显性拒绝，err={detail}"
        );
        assert!(
            detail.contains("destination"),
            "错误必须携带问题名定位，err={detail}"
        );
    }

    #[test]
    fn test_validate_duplicate_option_error_echo_single_line() {
        // 重复 option 错误回显必须单行化+截断（echo 防线），换行 option
        // 不得把伪造日志行带入 detail
        let mut req = valid_request();
        req.questions[0].options[0] = "evil\nFAKE LOG LINE".repeat(20);
        req.questions[0].options[1] = "evil\nFAKE LOG LINE".repeat(20);
        let err = req.validate().unwrap_err();
        let detail = err.error_detail();
        assert!(!detail.contains('\n'), "回显必须单行化，detail={detail:?}");
        assert!(
            detail.len() < 300,
            "重复 option 原文（40×17 字符）必须被 echo 截断，len={}",
            detail.len()
        );
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
    fn test_validate_rejects_blank_option() {
        // 与 name 同口径：choice 型 option 按字符串匹配消费，
        // 空/纯空白 option 的匹配结果有歧义
        let mut req = valid_request();
        req.questions[0].options[0] = "   ".to_string();
        let err = req.validate().unwrap_err();
        assert!(
            err.error_detail().contains("blank option"),
            "纯空白 option 必须拒绝且错误指明 option，err={}",
            err.error_detail()
        );

        let mut req = valid_request();
        req.questions[0].options[0] = String::new();
        assert!(req.validate().is_err(), "空 option 必须拒绝");
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
