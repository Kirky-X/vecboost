// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! OpenAI Compatible Embedding API Types
//!
//! This module contains request and response structures that conform to the
//! OpenAI Embeddings API specification.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// OpenAI-compatible embedding request.
///
/// This structure matches the OpenAI Embeddings API request format.
/// See: https://platform.openai.com/docs/api-reference/embeddings
#[derive(Debug, Deserialize, ToSchema)]
#[schema(example = json!({
    "input": "The food was delicious and the waiter was friendly.",
    "model": "text-embedding-ada-002",
    "encoding_format": "float",
    "dimensions": 1024
}))]
pub struct OpenAIEmbedRequest {
    /// The input text to embed, either as a string or an array of strings.
    /// Maximum array size is 2048 elements.
    #[schema(value_type = String, example = "The food was delicious and the waiter was friendly.")]
    pub input: OpenAIInput,

    /// The ID of the model to use for embedding.
    /// Must be a model that supports embeddings.
    #[schema(example = "text-embedding-ada-002")]
    pub model: String,

    /// The format to return the embeddings in.
    /// Can be "float" or "base64".
    /// Defaults to "float".
    #[schema(example = "float")]
    #[serde(default)]
    pub encoding_format: Option<String>,

    /// The number of dimensions the resulting output embeddings should have.
    /// Only supported by text-embedding-3 and later models.
    #[schema(example = 1024)]
    #[serde(default)]
    pub dimensions: Option<usize>,

    /// A unique identifier representing your end-user.
    /// Helps OpenAI to monitor and detect abuse.
    #[serde(default)]
    pub user: Option<String>,
}

/// Input type for embedding request - can be a single string or array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenAIInput {
    /// Single text input
    Single(String),
    /// Array of text inputs (max 2048 elements)
    Multiple(Vec<String>),
}

impl OpenAIInput {
    /// Returns true if this is a single text input
    pub fn is_single(&self) -> bool {
        matches!(self, OpenAIInput::Single(_))
    }

    /// Returns true if this is a batch input
    pub fn is_multiple(&self) -> bool {
        matches!(self, OpenAIInput::Multiple(_))
    }

    /// Returns the text if single, or None if batch
    pub fn as_single(&self) -> Option<&str> {
        match self {
            OpenAIInput::Single(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the texts if batch, or None if single
    pub fn as_multiple(&self) -> Option<&Vec<String>> {
        match self {
            OpenAIInput::Multiple(v) => Some(v),
            _ => None,
        }
    }

    /// Converts to a vector of strings
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            OpenAIInput::Single(s) => vec![s.clone()],
            OpenAIInput::Multiple(v) => v.clone(),
        }
    }

    /// Returns the number of input texts
    pub fn len(&self) -> usize {
        match self {
            OpenAIInput::Single(_) => 1,
            OpenAIInput::Multiple(v) => v.len(),
        }
    }

    /// Returns true if there are no inputs
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// OpenAI-compatible embedding response.
///
/// This structure matches the OpenAI Embeddings API response format.
#[derive(Debug, Serialize, ToSchema)]
#[schema(example = json!({
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.0023064255, -0.009327292, -0.0028842222],
            "index": 0
        }
    ],
    "model": "text-embedding-ada-002",
    "usage": {
        "prompt_tokens": 8,
        "total_tokens": 8
    }
}))]
pub struct OpenAIEmbedResponse {
    /// The object type, always "list"
    pub object: String,

    /// The list of embedding results
    pub data: Vec<EmbeddingObject>,

    /// The model used for embedding
    pub model: String,

    /// Usage information
    pub usage: Usage,
}

/// An embedding object returned by the API.
#[derive(Debug, Serialize, ToSchema)]
pub struct EmbeddingObject {
    /// The object type, always "embedding"
    pub object: String,

    /// The embedding vector
    pub embedding: Vec<f32>,

    /// The index of this embedding in the response
    pub index: usize,
}

/// Usage statistics for the request.
#[derive(Debug, Serialize, ToSchema)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Total number of tokens (same as prompt_tokens for embeddings)
    pub total_tokens: u32,
}

/// OpenAI-style error response.
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAIError {
    pub error: OpenAIErrorDetail,
}

/// Details of an API error.
#[derive(Debug, Serialize, ToSchema)]
pub struct OpenAIErrorDetail {
    /// Human-readable error message
    pub message: String,

    /// Error type (e.g., "invalid_request_error", "authentication_error")
    #[serde(rename = "type")]
    pub error_type: String,

    /// The parameter that caused the error (if applicable)
    #[serde(default)]
    pub param: Option<String>,

    /// Error code for programmatic handling
    #[serde(default)]
    pub code: Option<String>,
}

/// Encoding format for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    /// Standard float array
    Float,
    /// Base64 encoded bytes
    Base64,
}

impl EncodingFormat {
    /// Parse encoding format from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "float" => Some(EncodingFormat::Float),
            "base64" => Some(EncodingFormat::Base64),
            _ => None,
        }
    }

    /// Returns true if this is base64 encoding
    pub fn is_base64(&self) -> bool {
        matches!(self, EncodingFormat::Base64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_input_single() {
        let input = OpenAIInput::Single("hello".to_string());
        assert!(input.is_single());
        assert!(!input.is_multiple());
        assert_eq!(input.as_single(), Some("hello"));
        assert!(input.as_multiple().is_none());
        assert_eq!(input.len(), 1);
        assert!(!input.is_empty());
    }

    #[test]
    fn test_openai_input_multiple() {
        let input = OpenAIInput::Multiple(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert!(!input.is_single());
        assert!(input.is_multiple());
        assert!(input.as_single().is_none());
        assert_eq!(input.as_multiple().map(|v| v.len()), Some(3));
        assert_eq!(input.len(), 3);
        assert!(!input.is_empty());
    }

    #[test]
    fn test_openai_input_empty_multiple() {
        let input = OpenAIInput::Multiple(vec![]);
        assert!(input.is_multiple());
        assert_eq!(input.len(), 0);
        assert!(input.is_empty());
    }

    #[test]
    fn test_openai_input_to_vec_single() {
        let input = OpenAIInput::Single("world".to_string());
        let v = input.to_vec();
        assert_eq!(v, vec!["world".to_string()]);
    }

    #[test]
    fn test_openai_input_to_vec_multiple() {
        let input = OpenAIInput::Multiple(vec!["x".to_string(), "y".to_string()]);
        let v = input.to_vec();
        assert_eq!(v, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn test_openai_input_deserialize_single() {
        let json = "\"single text\"";
        let input: OpenAIInput = serde_json::from_str(json).unwrap();
        assert!(input.is_single());
        assert_eq!(input.as_single(), Some("single text"));
    }

    #[test]
    fn test_openai_input_deserialize_multiple() {
        let json = "[\"text1\", \"text2\"]";
        let input: OpenAIInput = serde_json::from_str(json).unwrap();
        assert!(input.is_multiple());
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn test_encoding_format_parse_float() {
        assert_eq!(EncodingFormat::parse("float"), Some(EncodingFormat::Float));
        assert_eq!(EncodingFormat::parse("FLOAT"), Some(EncodingFormat::Float));
        assert_eq!(EncodingFormat::parse("Float"), Some(EncodingFormat::Float));
    }

    #[test]
    fn test_encoding_format_parse_base64() {
        assert_eq!(
            EncodingFormat::parse("base64"),
            Some(EncodingFormat::Base64)
        );
        assert_eq!(
            EncodingFormat::parse("BASE64"),
            Some(EncodingFormat::Base64)
        );
    }

    #[test]
    fn test_encoding_format_parse_invalid() {
        assert_eq!(EncodingFormat::parse("binary"), None);
        assert_eq!(EncodingFormat::parse(""), None);
    }

    #[test]
    fn test_encoding_format_is_base64() {
        assert!(EncodingFormat::Base64.is_base64());
        assert!(!EncodingFormat::Float.is_base64());
    }

    #[test]
    fn test_openai_embed_request_deserialize() {
        let json = r#"{
            "input": "hello world",
            "model": "text-embedding-ada-002"
        }"#;
        let req: OpenAIEmbedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-ada-002");
        assert!(req.input.is_single());
        assert_eq!(req.encoding_format, None);
        assert_eq!(req.dimensions, None);
        assert_eq!(req.user, None);
    }

    #[test]
    fn test_openai_embed_request_deserialize_with_optional() {
        let json = r#"{
            "input": ["text1", "text2"],
            "model": "text-embedding-3-small",
            "encoding_format": "base64",
            "dimensions": 512,
            "user": "user-123"
        }"#;
        let req: OpenAIEmbedRequest = serde_json::from_str(json).unwrap();
        assert!(req.input.is_multiple());
        assert_eq!(req.encoding_format, Some("base64".to_string()));
        assert_eq!(req.dimensions, Some(512));
        assert_eq!(req.user, Some("user-123".to_string()));
    }
}
