// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

unsafe impl Send for AppError {}
unsafe impl Sync for AppError {}

#[derive(Error, Debug, Clone)]
pub enum AppError {
    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
}

impl AppError {
    pub fn config_error(message: String) -> Self {
        AppError::ConfigError(message)
    }

    pub fn model_load_error(message: String) -> Self {
        AppError::ModelLoadError(message)
    }

    pub fn tokenization_error(message: String) -> Self {
        AppError::TokenizationError(message)
    }

    pub fn inference_error(message: String) -> Self {
        AppError::InferenceError(message)
    }

    pub fn invalid_input(message: String) -> Self {
        AppError::InvalidInput(message)
    }

    pub fn not_found(message: String) -> Self {
        AppError::NotFound(message)
    }

    pub fn model_not_loaded(message: String) -> Self {
        AppError::ModelNotLoaded(message)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match self {
            AppError::ConfigError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ModelLoadError(_) => StatusCode::FAILED_DEPENDENCY,
            AppError::TokenizationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            AppError::InferenceError(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::ModelNotLoaded(_) => StatusCode::FAILED_DEPENDENCY,
        };

        let body = Json(json!({
            "error": self.to_string(),
            "code": status.as_u16()
        }));

        (status, body).into_response()
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::inference_error(e.to_string())
    }
}

impl From<candle_core::Error> for AppError {
    fn from(e: candle_core::Error) -> Self {
        AppError::inference_error(e.to_string())
    }
}
