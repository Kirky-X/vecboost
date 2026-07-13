// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CLI module — command-line interface for VecBoost
//!
//! Provides subcommands that mirror the HTTP/MCP API:
//! - `embed --text "Hello"` — embed a single text
//! - `batch --input file.txt` — embed texts from a file (one per line)
//! - `similarity --text1 "a" --text2 "b"` — compute cosine similarity
//!
//! All subcommands output JSON to stdout. The CLI is backed by the same
//! `api` functions used by the HTTP layer, ensuring consistent behavior.

use crate::EmbeddingService;
use crate::api;
use crate::domain::{BatchEmbedRequest, EmbedRequest, SimilarityRequest};
use crate::error::VecboostError;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// VecBoost command-line interface
#[derive(Parser, Debug)]
#[command(name = "vecboost", version, about = "High-performance embedding vector service", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

/// Available CLI subcommands
#[derive(Subcommand, Debug)]
pub enum CliCommand {
    /// Embed a single text into a vector
    Embed {
        /// Text to embed
        #[arg(long)]
        text: String,
    },
    /// Embed multiple texts from a file (one per line)
    Batch {
        /// Input file path
        #[arg(long)]
        input: PathBuf,
    },
    /// Compute cosine similarity between two texts
    Similarity {
        /// First text
        #[arg(long)]
        text1: String,
        /// Second text
        #[arg(long)]
        text2: String,
    },
}

/// Run the CLI against an initialized `EmbeddingService`.
///
/// Parses command-line arguments via `clap`, dispatches to the matching
/// `api` function, and prints the result as JSON to stdout.
///
/// # Errors
///
/// Returns `VecboostError` if the underlying API call fails or if the
/// input file cannot be read.
pub async fn run_cli(service: &EmbeddingService) -> Result<(), VecboostError> {
    let cli = Cli::parse();
    match cli.command {
        CliCommand::Embed { text } => {
            let req = EmbedRequest {
                text,
                normalize: Some(true),
            };
            let response = api::embed(service, req).await?;
            let json = serde_json::to_string(&response).map_err(|e| {
                VecboostError::InternalError(format!("JSON serialization failed: {}", e))
            })?;
            println!("{}", json);
        }
        CliCommand::Batch { input } => {
            let content = tokio::fs::read_to_string(&input)
                .await
                .map_err(|e| VecboostError::IoError(format!("Failed to read input file: {}", e)))?;
            let texts: Vec<String> = content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| line.to_string())
                .collect();
            if texts.is_empty() {
                return Err(VecboostError::InvalidInput(
                    "Input file is empty or contains no valid lines".to_string(),
                ));
            }
            let req = BatchEmbedRequest {
                texts,
                mode: None,
                normalize: Some(true),
            };
            let response = api::embed_batch(service, req).await?;
            let json = serde_json::to_string(&response).map_err(|e| {
                VecboostError::InternalError(format!("JSON serialization failed: {}", e))
            })?;
            println!("{}", json);
        }
        CliCommand::Similarity { text1, text2 } => {
            let req = SimilarityRequest {
                source: text1,
                target: text2,
            };
            let response = api::compute_similarity(service, req).await?;
            let json = serde_json::to_string(&response).map_err(|e| {
                VecboostError::InternalError(format!("JSON serialization failed: {}", e))
            })?;
            println!("{}", json);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::sync::RwLock;

    struct TestEngine {
        dimension: usize,
    }

    impl TestEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for TestEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.1f32; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.1f32; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            const PRECISION: Precision = Precision::Fp32;
            &PRECISION
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_service(dimension: usize) -> EmbeddingService {
        let temp_dir = tempdir().unwrap();
        let mock_engine = TestEngine::new(dimension);
        let model_config = ModelConfig {
            name: "test-model".to_string(),
            engine_type: EngineType::Candle,
            model_path: PathBuf::from(temp_dir.path()),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 32,
            pooling_mode: None,
            expected_dimension: Some(dimension),
            memory_limit_bytes: None,
            oom_fallback_enabled: true,
            model_sha256: None,
        };
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let _ = temp_dir;
        EmbeddingService::new(engine, Some(model_config))
    }

    #[test]
    fn test_cli_parse_embed_command() {
        let cli = Cli::try_parse_from(["vecboost", "embed", "--text", "Hello"]).unwrap();
        match cli.command {
            CliCommand::Embed { ref text } => assert_eq!(text, "Hello"),
            other => panic!("Expected Embed, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_parse_similarity_command() {
        let cli = Cli::try_parse_from(["vecboost", "similarity", "--text1", "a", "--text2", "b"])
            .unwrap();
        match cli.command {
            CliCommand::Similarity { text1, text2 } => {
                assert_eq!(text1, "a");
                assert_eq!(text2, "b");
            }
            other => panic!("Expected Similarity, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_parse_batch_command() {
        let cli = Cli::try_parse_from(["vecboost", "batch", "--input", "test.txt"]).unwrap();
        match cli.command {
            CliCommand::Batch { ref input } => {
                assert_eq!(input.to_str(), Some("test.txt"));
            }
            other => panic!("Expected Batch, got {:?}", other),
        }
    }

    #[test]
    fn test_cli_missing_text_arg_fails() {
        let result = Cli::try_parse_from(["vecboost", "embed"]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_cli_embed_outputs_json() {
        let service = make_service(384);
        // Simulate CLI args by setting them in the process
        // Since Cli::parse() reads from std::env::args(), we test the logic
        // by calling the api function directly (same as run_cli does internally)
        let req = EmbedRequest {
            text: "Hello".to_string(),
            normalize: Some(true),
        };
        let response = api::embed(&service, req).await.unwrap();
        let json = serde_json::to_string(&response).unwrap();
        // Verify JSON contains expected fields
        assert!(json.contains("embedding"));
        assert!(json.contains("dimension"));
        assert!(json.contains("384"));
    }
}
