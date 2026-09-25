// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

pub mod openai_embedding;
pub mod scheduling;

use crate::config::model::{DeviceType, PoolingMode};
use crate::utils::AggregationMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
#[cfg(feature = "schema")]
use utoipa::ToSchema;

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct EmbedRequest {
    pub text: String,
    pub normalize: Option<bool>,
}

impl FromStr for EmbedRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub processing_time_ms: u128,
    /// 信息保留率（仅在 Matryoshka 截断时填充）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub information_retention_rate: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SimilarityRequest {
    pub source: String,
    pub target: String,
    /// 相似度度量：cosine（默认）/ euclidean / dot_product / manhattan
    #[serde(default)]
    pub metric: Option<String>,
}

impl FromStr for SimilarityRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SimilarityResponse {
    pub score: f32,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SearchRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub top_k: Option<usize>,
}

impl FromStr for SearchRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl FromStr for UnloadModelRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub index: usize,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ParagraphEmbedding {
    pub embedding: Vec<f32>,
    pub position: usize,
    pub text_preview: String,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum EmbeddingOutput {
    Single(EmbedResponse),
    Paragraphs(Vec<ParagraphEmbedding>),
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct FileProcessingStats {
    pub total_chunks: usize,
    pub successful_chunks: usize,
    pub failed_chunks: usize,
    pub processing_time_ms: u128,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct FileEmbedRequest {
    pub path: String,
    pub mode: Option<AggregationMode>,
}

impl FromStr for FileEmbedRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct FileEmbedResponse {
    pub mode: AggregationMode,
    pub stats: FileProcessingStats,
    pub embedding: Option<Vec<f32>>,
    pub paragraphs: Option<Vec<ParagraphEmbedding>>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchEmbedRequest {
    pub texts: Vec<String>,
    pub mode: Option<AggregationMode>,
    pub normalize: Option<bool>,
}

impl FromStr for BatchEmbedRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchEmbedResponse {
    pub embeddings: Vec<BatchEmbeddingResult>,
    pub dimension: usize,
    pub processing_time_ms: u128,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchEmbeddingResult {
    pub text_preview: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelSwitchRequest {
    pub model_name: String,
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub device: Option<DeviceType>,
    pub max_batch_size: Option<usize>,
    pub pooling_mode: Option<PoolingMode>,
    pub expected_dimension: Option<usize>,
    pub memory_limit_bytes: Option<u64>,
    pub oom_fallback_enabled: Option<bool>,
}

impl FromStr for ModelSwitchRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelSwitchResponse {
    pub previous_model: Option<String>,
    pub current_model: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelInfo {
    pub name: String,
    pub engine_type: String,
    pub dimension: Option<usize>,
    pub is_loaded: bool,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub engine_type: String,
    pub dimension: Option<usize>,
    pub max_input_length: usize,
    pub is_loaded: bool,
    pub loaded_at: Option<String>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
    pub total_count: usize,
}

/// 卸载模型请求（/api/1/model/unload）
#[derive(Debug, Deserialize, Clone)]
pub struct UnloadModelRequest {
    pub model_name: String,
}

/// 卸载模型响应
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct UnloadModelResponse {
    pub model_name: String,
    pub unloaded: bool,
}

// =============================================================================
// Rerank 领域类型
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

impl FromStr for RerankRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
    pub processing_time_ms: u128,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchRerankRequest {
    pub queries: Vec<RerankRequest>,
}

impl FromStr for BatchRerankRequest {
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchRerankResponse {
    pub responses: Vec<RerankResponse>,
    /// 与请求 queries 按下标一一对应的状态位（容错语义可视化）：
    /// 单个 query 失败不产生响应（responses 仅含成功项），但在此处可见
    /// 失败原因，调用方据此把响应对位回请求。
    pub statuses: Vec<BatchRerankQueryStatus>,
}

/// 批量重排单条 query 的处理状态（审计建议：消除"静默跳过"不可观测性）
#[derive(Debug, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct BatchRerankQueryStatus {
    /// 对应请求 queries 的下标
    pub index: usize,
    /// 该 query 是否成功产出响应
    pub ok: bool,
    /// 失败原因（ok=true 时省略）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// 服务响应枚举 — pipeline 调度器统一返回类型
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum ServiceResponse {
    Embed(EmbedResponse),
    Rerank(RerankResponse),
}

// =============================================================================
// Decision 领域类型
// =============================================================================

/// 决策问题类型（文档 §2.2 编码：choice=0 / score=1 / noul=2）
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    /// 下游引擎/日志使用的稳定整数编码。
    pub fn as_code(&self) -> i64 {
        match self {
            QuestionType::Choice => 0,
            QuestionType::Score => 1,
            QuestionType::Noul => 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct DecisionQuestion {
    pub name: String,
    pub qtype: QuestionType,
    pub instructions: String,
    /// choice 型必填选项集；score/noul 型可缺省（serde default 回落空 vec）
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

/// 单个问题的答案体，按 qtype 三型。
///
/// serde 表示显式钉死为 internally tagged（`{"type":"choice",...}` 形态）——
/// 默认 externally tagged 会产出 `{"Choice":{...}}`，破坏对外契约。
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum DecisionAnswerBody {
    Choice {
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
    use serde_json;
    use std::collections::BTreeMap;

    #[test]
    fn test_embed_request_from_str() {
        let req: Result<EmbedRequest, _> = r#"{"text":"hello"}"#.parse();
        assert!(req.is_ok());
        assert_eq!(req.unwrap().text, "hello");
    }

    #[test]
    fn test_embed_request_with_normalize() {
        let req: EmbedRequest = serde_json::from_str(r#"{"text":"hi","normalize":true}"#).unwrap();
        assert_eq!(req.normalize, Some(true));
    }

    #[test]
    fn test_embed_response_serialize() {
        let resp = EmbedResponse {
            embedding: vec![1.0, 2.0],
            dimension: 2,
            processing_time_ms: 10,
            information_retention_rate: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("information_retention_rate"));
    }

    #[test]
    fn test_embed_response_with_retention_rate() {
        let resp = EmbedResponse {
            embedding: vec![1.0],
            dimension: 1,
            processing_time_ms: 5,
            information_retention_rate: Some(0.95),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("information_retention_rate"));
    }

    #[test]
    fn test_similarity_request_from_str() {
        let req: Result<SimilarityRequest, _> = r#"{"source":"a","target":"b"}"#.parse();
        assert!(req.is_ok());
    }

    #[test]
    fn test_similarity_response_serialize() {
        let resp = SimilarityResponse { score: 0.95 };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_search_request_deserialize() {
        let req: SearchRequest =
            serde_json::from_str(r#"{"query":"test","texts":["a","b"],"top_k":5}"#).unwrap();
        assert_eq!(req.top_k, Some(5));
    }

    #[test]
    fn test_search_response_serialize() {
        let resp = SearchResponse {
            results: vec![SearchResult {
                text: "a".into(),
                score: 0.9,
                index: 0,
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("0.9"));
    }

    #[test]
    fn test_file_embed_request_from_str() {
        let req: Result<FileEmbedRequest, _> = r#"{"path":"/tmp/test.txt"}"#.parse();
        assert!(req.is_ok());
    }

    #[test]
    fn test_batch_embed_request_from_str() {
        let req: Result<BatchEmbedRequest, _> = r#"{"texts":["a","b"]}"#.parse();
        assert!(req.is_ok());
    }

    #[test]
    fn test_batch_embed_response_serialize() {
        let resp = BatchEmbedResponse {
            embeddings: vec![BatchEmbeddingResult {
                text_preview: "hello".into(),
                embedding: vec![1.0],
            }],
            dimension: 1,
            processing_time_ms: 10,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("text_preview"));
    }

    #[test]
    fn test_model_switch_request_from_str() {
        let req: Result<ModelSwitchRequest, _> = r#"{"model_name":"test"}"#.parse();
        assert!(req.is_ok());
    }

    #[test]
    fn test_model_switch_response_serialize() {
        let resp = ModelSwitchResponse {
            previous_model: Some("old".into()),
            current_model: "new".into(),
            success: true,
            message: "ok".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("true"));
    }

    #[test]
    fn test_model_info_serialize() {
        let info = ModelInfo {
            name: "test".into(),
            engine_type: "candle".into(),
            dimension: Some(768),
            is_loaded: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("true"));
    }

    #[test]
    fn test_model_metadata_serialize() {
        let meta = ModelMetadata {
            name: "m".into(),
            version: "1.0".into(),
            engine_type: "candle".into(),
            dimension: Some(384),
            max_input_length: 512,
            is_loaded: false,
            loaded_at: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("1.0"));
    }

    #[test]
    fn test_model_list_response_serialize() {
        let resp = ModelListResponse {
            models: vec![],
            total_count: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("0"));
    }

    #[test]
    fn test_rerank_request_from_str() {
        let req: Result<RerankRequest, _> = r#"{"query":"q","documents":["d1"],"top_k":3}"#.parse();
        assert!(req.is_ok());
        assert_eq!(req.unwrap().top_k, Some(3));
    }

    #[test]
    fn test_rerank_result_serialize() {
        let r = RerankResult {
            index: 0,
            score: 0.8,
            document: Some("doc".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("doc"));
    }

    #[test]
    fn test_rerank_result_skip_document() {
        let r = RerankResult {
            index: 1,
            score: 0.5,
            document: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("document"));
    }

    #[test]
    fn test_rerank_response_serialize() {
        let resp = RerankResponse {
            results: vec![],
            processing_time_ms: 42,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("42"));
    }

    #[test]
    fn test_batch_rerank_request_from_str() {
        let req: Result<BatchRerankRequest, _> =
            r#"{"queries":[{"query":"q","documents":["d"]}]}"#.parse();
        assert!(req.is_ok());
    }

    #[test]
    fn test_batch_rerank_response_serialize() {
        let resp = BatchRerankResponse {
            responses: vec![],
            statuses: vec![BatchRerankQueryStatus {
                index: 0,
                ok: false,
                error: Some("empty query".to_string()),
            }],
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("responses"));
        assert!(json.contains("statuses"));
        // 失败状态必须携带 error;成功状态的 error 字段省略
        assert!(json.contains("error"));
        let ok_only = BatchRerankResponse {
            responses: vec![],
            statuses: vec![BatchRerankQueryStatus {
                index: 0,
                ok: true,
                error: None,
            }],
        };
        let ok_json = serde_json::to_string(&ok_only).unwrap();
        assert!(
            !ok_json.contains("\"error\""),
            "成功状态不应序列化 error 字段"
        );
    }

    #[test]
    fn test_service_response_embed_variant() {
        let resp = ServiceResponse::Embed(EmbedResponse {
            embedding: vec![1.0],
            dimension: 1,
            processing_time_ms: 1,
            information_retention_rate: None,
        });
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Embed"));
    }

    #[test]
    fn test_service_response_rerank_variant() {
        let resp = ServiceResponse::Rerank(RerankResponse {
            results: vec![],
            processing_time_ms: 0,
        });
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Rerank"));
    }

    #[test]
    fn test_paragraph_embedding_serialize() {
        let pe = ParagraphEmbedding {
            embedding: vec![0.1, 0.2],
            position: 3,
            text_preview: "hello world".into(),
        };
        let json = serde_json::to_string(&pe).unwrap();
        assert!(json.contains("3"));
    }

    #[test]
    fn test_embedding_output_single() {
        let out = EmbeddingOutput::Single(EmbedResponse {
            embedding: vec![],
            dimension: 0,
            processing_time_ms: 0,
            information_retention_rate: None,
        });
        let json = serde_json::to_string(&out).unwrap();
        assert!(json.contains("Single"));
    }

    #[test]
    fn test_embedding_output_paragraphs() {
        let out = EmbeddingOutput::Paragraphs(vec![]);
        let json = serde_json::to_string(&out).unwrap();
        assert!(json.contains("Paragraphs"));
    }

    #[test]
    fn test_file_processing_stats_serialize() {
        let stats = FileProcessingStats {
            total_chunks: 10,
            successful_chunks: 8,
            failed_chunks: 2,
            processing_time_ms: 100,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("2"));
    }

    #[test]
    fn test_file_embed_response_serialize() {
        let resp = FileEmbedResponse {
            mode: crate::utils::AggregationMode::SlidingWindow,
            stats: FileProcessingStats {
                total_chunks: 1,
                successful_chunks: 1,
                failed_chunks: 0,
                processing_time_ms: 5,
            },
            embedding: Some(vec![1.0]),
            paragraphs: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("embedding"));
    }

    // -------------------------------------------------------------------------
    // Decision 领域类型
    // -------------------------------------------------------------------------

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

    #[test]
    fn test_question_type_as_code() {
        // 文档 §2.2 编码：choice=0 / score=1 / noul=2
        assert_eq!(QuestionType::Choice.as_code(), 0);
        assert_eq!(QuestionType::Score.as_code(), 1);
        assert_eq!(QuestionType::Noul.as_code(), 2);
    }

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
