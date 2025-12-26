// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use regex::Regex;
use serde_json::json;
use thiserror::Error;

const MAX_ERROR_MESSAGE_LENGTH: usize = 200;

static SANITIZE_PATTERNS: std::sync::OnceLock<Vec<(Regex, &'static str)>> =
    std::sync::OnceLock::new();

fn get_sanitize_patterns() -> &'static Vec<(Regex, &'static str)> {
    SANITIZE_PATTERNS.get_or_init(|| {
        vec![
            (
                Regex::new(r#"/[a-zA-Z0-9/_.-]+/[a-zA-Z0-9/_.-]+\.\w+"#).unwrap(),
                "[REDACTED_PATH]",
            ),
            (
                Regex::new(r#"C:\\[a-zA-Z0-9_\\]+\.\w+"#).unwrap(),
                "[REDACTED_WINDOWS_PATH]",
            ),
            (Regex::new(r#"token \d+"#).unwrap(), "token [ID]"),
            (
                Regex::new(r#"at position \d+"#).unwrap(),
                "at position [REDACTED]",
            ),
            (Regex::new(r#"\.unwrap\(\)"#).unwrap(), "[INTERNAL_ERROR]"),
            (
                Regex::new(r#"expect\([^)]+\)"#).unwrap(),
                "[INTERNAL_ERROR]",
            ),
        ]
    })
}

fn sanitize_error_message(msg: &str) -> String {
    let mut sanitized = msg.to_string();
    for (pattern, replacement) in get_sanitize_patterns() {
        sanitized = pattern.replace_all(&sanitized, *replacement).to_string();
    }

    if sanitized.len() > MAX_ERROR_MESSAGE_LENGTH {
        sanitized.truncate(MAX_ERROR_MESSAGE_LENGTH);
        sanitized.push_str("...");
    }

    sanitized
}

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

        let sanitized_message = sanitize_error_message(&self.to_string());

        let body = Json(json!({
            "error": sanitized_message,
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

impl From<tokio::task::JoinError> for AppError {
    fn from(e: tokio::task::JoinError) -> Self {
        AppError::inference_error(e.to_string())
    }
}
