// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::utils::AggregationMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ChunkRequest {
    pub text: String,
    pub chunk_size: Option<usize>,
    pub overlap_ratio: Option<f32>,
    pub mode: Option<AggregationMode>,
}

#[derive(Debug, Serialize)]
pub struct ChunkResponse {
    pub chunks: Vec<String>,
    pub chunk_count: usize,
}

#[derive(Debug, Serialize)]
pub struct ChunkResult {
    pub chunks: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub aggregated_embedding: Option<Vec<f32>>,
    pub chunk_count: usize,
}
