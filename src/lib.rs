// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

// SDK Public API
pub use config::AppConfig;
pub use domain::{BatchEmbedRequest, EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse, SimilarityMetric};
pub use engine::candle_engine::CandleEngine;
pub use model::config::{DeviceType, EngineType, ModelConfig, PoolingMode};
pub use service::embedding::EmbeddingService;

// Internal modules (re-exported for testing)
mod config;
mod domain;
mod engine;
mod model;
mod service;
