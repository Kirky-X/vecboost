// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
#[cfg(test)]
use crate::text::domain::ChunkResult;
use crate::text::domain::{ChunkRequest, ChunkResponse};
use crate::utils::AggregationMode;
use crate::utils::constants::{DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_RATIO, MIN_CHUNK_SIZE_RATIO};

#[cfg(target_os = "macos")]
use tokenizers::Tokenizer;

#[cfg(not(target_os = "macos"))]
use crate::text::Tokenizer;

#[derive(Debug, Clone)]
pub struct TextChunker {
    tokenizer: Tokenizer,
    chunk_size: usize,
    overlap_size: usize,
    min_chunk_size: usize,
}

impl TextChunker {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self::with_config(tokenizer, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_RATIO)
    }

    pub fn with_config(tokenizer: Tokenizer, chunk_size: usize, overlap_ratio: f32) -> Self {
        let overlap_size = (chunk_size as f32 * overlap_ratio) as usize;
        let min_chunk_size = chunk_size / MIN_CHUNK_SIZE_RATIO;

        Self {
            tokenizer,
            chunk_size,
            overlap_size,
            min_chunk_size,
        }
    }

    pub fn chunk(&self, text: &str) -> Result<Vec<String>, AppError> {
        self.chunk_with_mode(text, AggregationMode::SlidingWindow)
    }

    pub fn chunk_with_mode(
        &self,
        text: &str,
        mode: AggregationMode,
    ) -> Result<Vec<String>, AppError> {
        match mode {
            AggregationMode::SlidingWindow => self.sliding_window_chunk(text),
            AggregationMode::Paragraph => self.paragraph_chunk(text),
            AggregationMode::Paragraphs => self.paragraph_chunk(text),
            AggregationMode::FixedSize => self.fixed_size_chunk(text),
            AggregationMode::Average
            | AggregationMode::MaxPooling
            | AggregationMode::MinPooling
            | AggregationMode::Document => self.sliding_window_chunk(text),
        }
    }

    fn sliding_window_chunk(&self, text: &str) -> Result<Vec<String>, AppError> {
        let tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| AppError::TokenizationError(e.to_string()))?;

        let token_ids = tokens.get_ids();
        let num_tokens = token_ids.len();

        if num_tokens <= self.chunk_size {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < num_tokens {
            let end = (start + self.chunk_size).min(num_tokens);
            let chunk_tokens = &token_ids[start..end];

            let chunk_text = self
                .tokenizer
                .decode(chunk_tokens, true)
                .map_err(|e| AppError::TokenizationError(e.to_string()))?;

            if !chunk_text.is_empty() {
                chunks.push(chunk_text);
            }

            if end >= num_tokens {
                break;
            }

            let step = self.chunk_size - self.overlap_size;
            if step == 0 {
                break;
            }
            start += step;
        }

        Ok(chunks)
    }

    fn paragraph_chunk(&self, text: &str) -> Result<Vec<String>, AppError> {
        let paragraphs: Vec<String> = text
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .map(|p| p.replace('\n', " "))
            .collect();

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_token_count = 0;

        for paragraph in paragraphs {
            let para_tokens = self
                .tokenizer
                .encode(paragraph.as_str(), false)
                .map_err(|e| AppError::TokenizationError(e.to_string()))?;
            let para_token_count = para_tokens.get_ids().len();

            if para_token_count > self.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                    current_token_count = 0;
                }
                let sub_chunks = self.sliding_window_chunk(&paragraph)?;
                chunks.extend(sub_chunks);
            } else if current_token_count + para_token_count <= self.chunk_size {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&paragraph);
                current_token_count += para_token_count;
            } else {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                }
                current_chunk = paragraph;
                current_token_count = para_token_count;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        if chunks.is_empty() {
            chunks.push(text.to_string());
        }

        Ok(chunks)
    }

    fn fixed_size_chunk(&self, text: &str) -> Result<Vec<String>, AppError> {
        let chars: Vec<char> = text.chars().collect();
        let char_count = chars.len();
        let chunk_chars = (self.chunk_size * 4).min(char_count);

        if char_count <= chunk_chars {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < char_count {
            let end = (start + chunk_chars).min(char_count);
            let chunk: String = chars[start..end].iter().collect();
            let trimmed = chunk.trim();

            if !trimmed.is_empty() && trimmed.len() >= self.min_chunk_size * 4 {
                chunks.push(trimmed.to_string());
            }

            if end >= char_count {
                break;
            }

            let step = chunk_chars - self.overlap_size * 4;
            if step == 0 {
                break;
            }
            start += step;
        }

        if chunks.is_empty() {
            chunks.push(text.to_string());
        }

        Ok(chunks)
    }

    pub fn chunk_request(&self, request: &ChunkRequest) -> Result<ChunkResponse, AppError> {
        let mode = request.mode.unwrap_or(AggregationMode::SlidingWindow);
        let chunks = self.chunk_with_mode(&request.text, mode)?;

        Ok(ChunkResponse {
            chunks: chunks.clone(),
            chunk_count: chunks.len(),
        })
    }

    pub fn get_chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn get_overlap_size(&self) -> usize {
        self.overlap_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_default_values() {
        assert_eq!(DEFAULT_CHUNK_SIZE, 512);
        assert_eq!(DEFAULT_OVERLAP_RATIO, 0.2);
        assert_eq!(MIN_CHUNK_SIZE_RATIO, 4);
    }

    #[test]
    fn test_chunker_config_calculation() {
        let chunk_size = 256;
        let overlap_ratio = 0.1;
        let expected_overlap_size = (chunk_size as f32 * overlap_ratio) as usize;
        let expected_min_chunk_size = chunk_size / MIN_CHUNK_SIZE_RATIO;

        assert_eq!(expected_overlap_size, 25);
        assert_eq!(expected_min_chunk_size, 64);
    }

    #[test]
    fn test_chunk_request_default_mode() {
        let request = ChunkRequest {
            text: "test text".to_string(),
            chunk_size: None,
            overlap_ratio: None,
            mode: None,
        };
        assert_eq!(request.mode, None);
        assert_eq!(request.chunk_size, None);
    }

    #[test]
    fn test_chunk_response_creation() {
        let chunks = vec!["chunk1".to_string(), "chunk2".to_string()];
        let response = ChunkResponse {
            chunks: chunks.clone(),
            chunk_count: chunks.len(),
        };
        assert_eq!(response.chunks, chunks);
        assert_eq!(response.chunk_count, 2);
    }

    #[test]
    fn test_chunk_result_creation() {
        let chunks = vec!["chunk1".to_string()];
        let embeddings = Some(vec![vec![1.0, 2.0, 3.0]]);
        let aggregated = Some(vec![1.0, 2.0, 3.0]);

        let result = ChunkResult {
            chunks: chunks.clone(),
            embeddings,
            aggregated_embedding: aggregated,
            chunk_count: 1,
        };

        assert_eq!(result.chunks, chunks);
        assert_eq!(result.chunk_count, 1);
        assert!(result.embeddings.is_some());
        assert!(result.aggregated_embedding.is_some());
    }
}
