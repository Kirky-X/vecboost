// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod config;
pub mod device;
pub mod domain;
pub mod engine;
pub mod error;
pub mod metrics;
pub mod model;
pub mod monitor;
pub mod service;
pub mod text;
pub mod utils;

pub use config::app::{AppConfig, ModelConfig as AppModelConfig, ServerConfig};
pub use config::model::{DeviceType, EngineType, ModelConfig, PoolingMode};
pub use domain::{EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse};
pub use engine::candle_engine::CandleEngine;
pub use service::embedding::EmbeddingService;
pub use utils::SimilarityMetric;
