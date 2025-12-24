// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod constants;
pub mod validator;
pub mod vector;

pub use constants::{
    DEFAULT_TOP_K, MAX_BATCH_SIZE, MAX_CONCURRENT_REQUESTS, MAX_SEARCH_RESULTS, MAX_TEXT_LENGTH,
    MAX_TOP_K, MIN_TEXT_LENGTH,
};
pub use validator::{InputValidator, TextValidator, ValidationConfig};
pub use vector::{
    calculate_similarity, cosine_similarity, dot_product, euclidean_distance, l2_normalize,
    manhattan_distance, normalize_l2, AggregationMode, SimilarityMetric,
};
