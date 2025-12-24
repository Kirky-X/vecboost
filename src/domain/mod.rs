// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

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
