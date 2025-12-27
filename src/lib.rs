// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

pub mod auth;
pub mod cache;
pub mod config;
pub mod device;
pub mod domain;
pub mod engine;
pub mod error;
pub mod grpc;
pub mod metrics;
pub mod model;
pub mod monitor;
pub mod security;
pub mod service;
pub mod text;
pub mod utils;

pub use config::app::{AppConfig, AuthConfig, ServerConfig};
pub use config::model::ModelConfig;
pub use domain::{EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse};
pub use service::embedding::EmbeddingService;
pub use utils::SimilarityMetric;
