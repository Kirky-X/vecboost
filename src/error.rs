// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
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
pub enum VecboostError {
    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("Model load error: {0}")]
    ModelLoadError(String),

    #[error("Model file corrupted: {0}")]
    ModelFileCorrupted(String),

    #[error("Model file integrity check failed: {0}")]
    ModelIntegrityError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Out of memory error: {0}")]
    OutOfMemory(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Security error: {0}")]
    SecurityError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl VecboostError {
    pub fn config_error(message: String) -> Self {
        VecboostError::ConfigError(message)
    }

    pub fn model_load_error(message: String) -> Self {
        VecboostError::ModelLoadError(message)
    }

    pub fn model_file_corrupted(message: String) -> Self {
        VecboostError::ModelFileCorrupted(message)
    }

    pub fn model_integrity_error(message: String) -> Self {
        VecboostError::ModelIntegrityError(message)
    }

    pub fn tokenization_error(message: String) -> Self {
        VecboostError::TokenizationError(message)
    }

    pub fn inference_error(message: String) -> Self {
        VecboostError::InferenceError(message)
    }

    pub fn invalid_input(message: String) -> Self {
        VecboostError::InvalidInput(message)
    }

    pub fn not_found(message: String) -> Self {
        VecboostError::NotFound(message)
    }

    pub fn model_not_loaded(message: String) -> Self {
        VecboostError::ModelNotLoaded(message)
    }

    pub fn authentication_error(message: String) -> Self {
        VecboostError::AuthenticationError(message)
    }

    pub fn security_error(message: String) -> Self {
        VecboostError::SecurityError(message)
    }

    pub fn io_error(message: String) -> Self {
        VecboostError::IoError(message)
    }

    pub fn validation_error(message: String) -> Self {
        VecboostError::ValidationError(message)
    }
}

impl IntoResponse for VecboostError {
    fn into_response(self) -> Response {
        let status = match self {
            VecboostError::ConfigError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VecboostError::ModelLoadError(_) => StatusCode::FAILED_DEPENDENCY,
            VecboostError::ModelFileCorrupted(_) => StatusCode::FAILED_DEPENDENCY,
            VecboostError::ModelIntegrityError(_) => StatusCode::FAILED_DEPENDENCY,
            VecboostError::TokenizationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            VecboostError::InferenceError(_) => StatusCode::SERVICE_UNAVAILABLE,
            VecboostError::OutOfMemory(_) => StatusCode::INSUFFICIENT_STORAGE,
            VecboostError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            VecboostError::NotFound(_) => StatusCode::NOT_FOUND,
            VecboostError::ModelNotLoaded(_) => StatusCode::FAILED_DEPENDENCY,
            VecboostError::AuthenticationError(_) => StatusCode::UNAUTHORIZED,
            VecboostError::SecurityError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VecboostError::IoError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VecboostError::ValidationError(_) => StatusCode::BAD_REQUEST,
            VecboostError::RateLimitExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
            VecboostError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        let sanitized_message = sanitize_error_message(&self.to_string());

        let body = Json(json!({
            "error": sanitized_message,
            "code": status.as_u16()
        }));

        (status, body).into_response()
    }
}

impl From<std::io::Error> for VecboostError {
    fn from(e: std::io::Error) -> Self {
        VecboostError::inference_error(e.to_string())
    }
}

impl From<candle_core::Error> for VecboostError {
    fn from(e: candle_core::Error) -> Self {
        VecboostError::inference_error(e.to_string())
    }
}

impl From<tokio::task::JoinError> for VecboostError {
    fn from(e: tokio::task::JoinError) -> Self {
        VecboostError::inference_error(e.to_string())
    }
}

/// Deprecated alias for backward compatibility.
/// Use [`VecboostError`] instead.
#[deprecated(note = "Use VecboostError instead")]
pub type AppError = VecboostError;
