// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

pub mod openai_embedding;

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
}

/// 服务响应枚举 — pipeline 调度器统一返回类型
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum ServiceResponse {
    Embed(EmbedResponse),
    Rerank(RerankResponse),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

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
        let resp = BatchRerankResponse { responses: vec![] };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("responses"));
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
}
