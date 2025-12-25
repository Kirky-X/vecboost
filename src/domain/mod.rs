// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::utils::AggregationMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub dimension: usize,
}

#[derive(Debug, Deserialize)]
pub struct SimilarityRequest {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Serialize)]
pub struct SimilarityResponse {
    pub score: f32,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub texts: Vec<String>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct ParagraphEmbedding {
    pub embedding: Vec<f32>,
    pub position: usize,
    pub text_preview: String,
}

#[derive(Debug, Serialize)]
pub enum EmbeddingOutput {
    Single(EmbedResponse),
    Paragraphs(Vec<ParagraphEmbedding>),
}

#[derive(Debug, Serialize)]
pub struct FileProcessingStats {
    pub lines_processed: usize,
    pub paragraphs_processed: usize,
    pub processing_time_ms: u128,
    pub memory_peak_mb: usize,
}

#[derive(Debug, Deserialize)]
pub struct FileEmbedRequest {
    pub path: String,
    pub mode: Option<AggregationMode>,
}

#[derive(Debug, Serialize)]
pub struct FileEmbedResponse {
    pub mode: AggregationMode,
    pub stats: FileProcessingStats,
    pub embedding: Option<Vec<f32>>,
    pub paragraphs: Option<Vec<ParagraphEmbedding>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchEmbedRequest {
    pub texts: Vec<String>,
    pub mode: Option<AggregationMode>,
}

#[derive(Debug, Serialize)]
pub struct BatchEmbedResponse {
    pub embeddings: Vec<BatchEmbeddingResult>,
    pub dimension: usize,
    pub processing_time_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct BatchEmbeddingResult {
    pub text_preview: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelSwitchRequest {
    pub model_name: String,
}

#[derive(Debug, Serialize)]
pub struct ModelSwitchResponse {
    pub previous_model: Option<String>,
    pub current_model: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub engine_type: String,
    pub dimension: Option<usize>,
    pub is_loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
    pub total_count: usize,
}
