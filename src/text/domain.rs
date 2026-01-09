// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(test)]
use crate::utils::AggregationMode;
#[cfg(test)]
use serde::{Deserialize, Serialize};

#[cfg(test)]
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChunkRequest {
    pub text: String,
    pub chunk_size: Option<usize>,
    pub overlap_ratio: Option<f32>,
    pub mode: Option<AggregationMode>,
}

#[cfg(test)]
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ChunkResponse {
    pub chunks: Vec<String>,
    pub chunk_count: usize,
}

#[cfg(test)]
#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ChunkResult {
    pub chunks: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub aggregated_embedding: Option<Vec<f32>>,
    pub chunk_count: usize,
}
