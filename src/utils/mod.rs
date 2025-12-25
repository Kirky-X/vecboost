// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod constants;
pub mod validator;
pub mod vector;

pub use constants::{
    DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_RATIO, DEFAULT_TOP_K, MAX_BATCH_SIZE,
    MAX_CONCURRENT_REQUESTS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, MAX_SEARCH_RESULTS,
    MAX_TEXT_LENGTH, MAX_TOP_K, MIN_CHUNK_SIZE_RATIO, MIN_TEXT_LENGTH,
};
pub use validator::{FileValidator, InputValidator, TextValidator, ValidationConfig};
pub use vector::{
    calculate_similarity, cosine_similarity, dot_product, euclidean_distance, manhattan_distance,
    normalize_l2, AggregationMode, SimilarityMetric,
};
