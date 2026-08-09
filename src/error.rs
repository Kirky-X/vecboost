// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#[cfg(feature = "http")]
use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
#[cfg(feature = "http")]
use regex::Regex;
#[cfg(feature = "http")]
use serde_json::json;
use thiserror::Error;

#[cfg(feature = "http")]
const MAX_ERROR_MESSAGE_LENGTH: usize = 200;

#[cfg(feature = "http")]
static SANITIZE_PATTERNS: std::sync::OnceLock<Vec<(Regex, &'static str)>> =
    std::sync::OnceLock::new();

#[cfg(feature = "http")]
fn get_sanitize_patterns() -> &'static Vec<(Regex, &'static str)> {
    SANITIZE_PATTERNS.get_or_init(|| {
        vec![
            (
                Regex::new(r#"/[a-zA-Z0-9/_.-]+/[a-zA-Z0-9/_.-]+\.\w+"#)
                    .expect("sanitize pattern: valid Unix path regex"),
                "[REDACTED_PATH]",
            ),
            (
                Regex::new(r#"C:\\[a-zA-Z0-9_\\]+\.\w+"#)
                    .expect("sanitize pattern: valid Windows path regex"),
                "[REDACTED_WINDOWS_PATH]",
            ),
            (
                Regex::new(r#"token \d+"#).expect("sanitize pattern: valid token regex"),
                "token [ID]",
            ),
            (
                Regex::new(r#"at position \d+"#)
                    .expect("sanitize pattern: valid position regex"),
                "at position [REDACTED]",
            ),
            (
                Regex::new(r#"\.unwrap\(\)"#)
                    .expect("sanitize pattern: valid unwrap regex"),
                "[INTERNAL_ERROR]",
            ),
            (
                Regex::new(r#"expect\([^)]+\)"#)
                    .expect("sanitize pattern: valid expect regex"),
                "[INTERNAL_ERROR]",
            ),
        ]
    })
}

#[cfg(feature = "http")]
fn sanitize_error_message(msg: &str) -> String {
    let mut sanitized = msg.to_string();
    for (pattern, replacement) in get_sanitize_patterns() {
        sanitized = pattern.replace_all(&sanitized, *replacement).to_string();
    }

    if sanitized.len() > MAX_ERROR_MESSAGE_LENGTH {
        // 确保在有效的 char 边界处截断，避免 UTF-8 多字节字符被截断导致 panic
        let mut truncate_at = MAX_ERROR_MESSAGE_LENGTH;
        while truncate_at > 0 && !sanitized.is_char_boundary(truncate_at) {
            truncate_at -= 1;
        }
        sanitized.truncate(truncate_at);
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

    #[error("Database error: {0}")]
    DatabaseError(String),

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

    pub fn database_error(message: String) -> Self {
        VecboostError::DatabaseError(message)
    }

    pub fn rate_limit_exceeded(message: String) -> Self {
        VecboostError::RateLimitExceeded(message)
    }

    pub fn out_of_memory(message: String) -> Self {
        VecboostError::OutOfMemory(message)
    }

    pub fn internal_error(message: String) -> Self {
        VecboostError::InternalError(message)
    }
}

#[cfg(feature = "http")]
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
            VecboostError::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
        VecboostError::IoError(e.to_string())
    }
}

#[cfg(feature = "db")]
impl From<sea_orm::DbErr> for VecboostError {
    fn from(e: sea_orm::DbErr) -> Self {
        VecboostError::DatabaseError(e.to_string())
    }
}

#[cfg(feature = "auth")]
impl From<garrison::error::GarrisonError> for VecboostError {
    fn from(e: garrison::error::GarrisonError) -> Self {
        use garrison::error::GarrisonError as GE;
        match e {
            GE::NotLogin(msg)
            | GE::NotPermission(msg)
            | GE::NotRole(msg)
            | GE::InvalidToken(msg)
            | GE::TokenRevoked(msg)
            | GE::ExpiredToken(msg)
            | GE::Session(msg) => VecboostError::AuthenticationError(msg),
            GE::Config(msg) => VecboostError::ConfigError(msg),
            GE::Dao(msg) | GE::Internal(msg) | GE::Annotation(msg) | GE::Context(msg) => {
                VecboostError::InternalError(msg)
            }
            GE::Exception(ex) => VecboostError::AuthenticationError(ex.to_string()),
            _ => VecboostError::InternalError(e.to_string()),
        }
    }
}

impl From<candle_core::Error> for VecboostError {
    fn from(e: candle_core::Error) -> Self {
        VecboostError::InferenceError(e.to_string())
    }
}

impl From<tokio::task::JoinError> for VecboostError {
    fn from(e: tokio::task::JoinError) -> Self {
        VecboostError::InferenceError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_unix_path() {
        let msg = "Failed to load /home/user/model/file.safetensors";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("[REDACTED_PATH]"));
        assert!(!sanitized.contains("/home/user/model/file.safetensors"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_windows_path() {
        let msg = r#"Failed to load C:\Users\admin\config.json"#;
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("[REDACTED_WINDOWS_PATH]"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_token_id() {
        let msg = "Invalid token 12345";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("token [ID]"));
        assert!(!sanitized.contains("12345"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_position() {
        let msg = "JSON parse error at position 42";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("at position [REDACTED]"));
        assert!(!sanitized.contains("42"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_unwrap() {
        let msg = "Error in value.unwrap() at line 42";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("[INTERNAL_ERROR]"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_expect() {
        let msg = "called Result::expect(hello) on an Err value";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("[INTERNAL_ERROR]"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_truncation() {
        let long_msg = "x".repeat(300);
        let sanitized = sanitize_error_message(&long_msg);
        assert!(sanitized.len() <= MAX_ERROR_MESSAGE_LENGTH + 3);
        assert!(sanitized.ends_with("..."));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_short_message() {
        let msg = "Simple error";
        let sanitized = sanitize_error_message(msg);
        assert_eq!(sanitized, "Simple error");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_empty() {
        let sanitized = sanitize_error_message("");
        assert_eq!(sanitized, "");
    }


    // -------------------------------------------------------------------------
    // IntoResponse 测试
    // -------------------------------------------------------------------------
    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_config_error() {
        let err = VecboostError::ConfigError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_invalid_input() {
        let err = VecboostError::InvalidInput("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_not_found() {
        let err = VecboostError::NotFound("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_authentication_error() {
        let err = VecboostError::AuthenticationError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_rate_limit_exceeded() {
        let err = VecboostError::RateLimitExceeded("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_model_load_error() {
        let err = VecboostError::ModelLoadError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_inference_error() {
        let err = VecboostError::InferenceError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_validation_error() {
        let err = VecboostError::ValidationError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_tokenization_error() {
        let err = VecboostError::TokenizationError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_out_of_memory() {
        let err = VecboostError::OutOfMemory("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INSUFFICIENT_STORAGE);
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let vecboost_err: VecboostError = io_err.into();
        match vecboost_err {
            VecboostError::IoError(msg) => assert!(msg.contains("file not found")),
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = VecboostError::ConfigError("test message".to_string());
        assert_eq!(format!("{}", err), "Config error: test message");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_model_file_corrupted() {
        let err = VecboostError::ModelFileCorrupted("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_model_integrity_error() {
        let err = VecboostError::ModelIntegrityError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_model_not_loaded() {
        let err = VecboostError::ModelNotLoaded("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_security_error() {
        let err = VecboostError::SecurityError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_io_error_variant() {
        let err = VecboostError::IoError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_database_error() {
        let err = VecboostError::DatabaseError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_into_response_internal_error() {
        let err = VecboostError::InternalError("test".to_string());
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_from_candle_error() {
        let candle_err = candle_core::Error::Msg("candle failure".to_string());
        let vecboost_err: VecboostError = candle_err.into();
        match vecboost_err {
            VecboostError::InferenceError(msg) => {
                assert!(msg.contains("candle failure"), "got: {}", msg)
            }
            _ => panic!("Expected InferenceError"),
        }
    }

    #[tokio::test]
    async fn test_from_join_error() {
        let handle = tokio::spawn(async {
            panic!("test panic");
        });
        let join_err = handle.await.unwrap_err();
        let vecboost_err: VecboostError = join_err.into();
        match vecboost_err {
            VecboostError::InferenceError(_) => {}
            other => panic!("Expected InferenceError, got {:?}", other),
        }
    }

    #[cfg(feature = "db")]
    #[test]
    fn test_from_db_error() {
        let db_err = sea_orm::DbErr::RecordNotFound("not found".to_string());
        let vecboost_err: VecboostError = db_err.into();
        match vecboost_err {
            VecboostError::DatabaseError(msg) => {
                assert!(msg.contains("not found"), "got: {}", msg)
            }
            _ => panic!("Expected DatabaseError"),
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_multiple_patterns() {
        let msg = "Error at position 42 in value.unwrap() for token 12345 at /home/user/model/file.safetensors";
        let sanitized = sanitize_error_message(msg);
        assert!(sanitized.contains("at position [REDACTED]"));
        assert!(sanitized.contains("[INTERNAL_ERROR]"));
        assert!(sanitized.contains("token [ID]"));
        assert!(sanitized.contains("[REDACTED_PATH]"));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_exact_boundary() {
        let msg = "x".repeat(MAX_ERROR_MESSAGE_LENGTH);
        let sanitized = sanitize_error_message(&msg);
        assert_eq!(sanitized.len(), MAX_ERROR_MESSAGE_LENGTH);
        assert!(!sanitized.ends_with("..."));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sanitize_error_message_one_char_over_boundary() {
        let msg = "x".repeat(MAX_ERROR_MESSAGE_LENGTH + 1);
        let sanitized = sanitize_error_message(&msg);
        assert!(sanitized.ends_with("..."));
        assert!(sanitized.len() <= MAX_ERROR_MESSAGE_LENGTH + 3);
    }

    #[test]
    fn test_all_error_variants_display() {
        assert_eq!(
            format!("{}", VecboostError::OutOfMemory("oom".to_string())),
            "Out of memory error: oom"
        );
        assert_eq!(
            format!("{}", VecboostError::DatabaseError("db".to_string())),
            "Database error: db"
        );
        assert_eq!(
            format!("{}", VecboostError::InternalError("int".to_string())),
            "Internal error: int"
        );
        assert_eq!(
            format!("{}", VecboostError::RateLimitExceeded("rl".to_string())),
            "Rate limit exceeded: rl"
        );
    }
}
