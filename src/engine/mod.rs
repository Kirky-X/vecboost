// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

pub(crate) mod candle_engine;
pub mod factory;
pub use factory::EngineFactory;
pub(crate) mod impl_;

#[cfg(feature = "onnx")]
pub(crate) mod onnx_engine;

use crate::config::model::{ModelConfig, Precision};
use crate::error::VecboostError;
use async_trait::async_trait;

/// 推理引擎抽象接口
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// 执行推理，返回未归一化的向量
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError>;

    /// 批量推理
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError>;

    /// 获取当前精度设置
    fn precision(&self) -> &Precision;

    /// 检查是否支持混合精度
    fn supports_mixed_precision(&self) -> bool;

    /// 检查是否已触发降级
    fn is_fallback_triggered(&self) -> bool {
        false
    }

    /// 对 (query, document) 对进行重排序评分
    ///
    /// 默认实现：bi-encoder (embed_batch + cosine + sigmoid)。
    /// 任何实现了 `embed_batch` 的引擎自动获得 rerank 能力。
    fn rerank(&self, query: &str, document: &str) -> Result<f32, VecboostError> {
        let texts = vec![query.to_string(), document.to_string()];
        let embeddings = self.embed_batch(&texts)?;
        let similarity = crate::utils::vector::cosine_similarity(&embeddings[0], &embeddings[1])?;
        Ok(1.0 / (1.0 + (-similarity).exp()))
    }

    /// 批量重排序：query 只 embed 1 次，documents 批量 embed 1 次
    ///
    /// 默认实现：2 次 forward pass（而非 N 次 rerank = 2N 次）。
    fn rerank_batch(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, VecboostError> {
        let mut texts = Vec::with_capacity(1 + documents.len());
        texts.push(query.to_string());
        texts.extend(documents.iter().cloned());

        let embeddings = self.embed_batch(&texts)?;
        let query_emb = &embeddings[0];

        embeddings[1..]
            .iter()
            .map(|doc_emb| {
                let similarity = crate::utils::vector::cosine_similarity(query_emb, doc_emb)?;
                Ok(1.0 / (1.0 + (-similarity).exp()))
            })
            .collect()
    }

    /// 检查引擎是否支持重排序
    ///
    /// 默认返回 true — bi-encoder rerank 对任何 embedding 引擎都可用。
    fn supports_rerank(&self) -> bool {
        true
    }

    /// 统计文本的 token 数(用于 API usage.prompt_tokens)。
    /// 默认实现: bytes/4 估算;真实引擎应覆盖为 tokenizer 精确计数。
    fn count_tokens(&self, _text: &str) -> Result<usize, VecboostError> {
        Ok(0) // 调用方回退到 bytes/4
    }

    /// 尝试降级到 CPU（在 OOM 时调用）
    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError>;
}

#[allow(clippy::large_enum_variant)]
pub enum AnyEngine {
    Candle(candle_engine::CandleEngine),
    #[cfg(feature = "onnx")]
    Onnx(onnx_engine::OnnxEngine),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{DeviceType, EngineType, Precision};
    use async_trait::async_trait;

    /// Mock engine that returns deterministic embeddings for testing default trait methods.
    struct MockEngine {
        dimension: usize,
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![1.0; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![1.0; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
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

    /// Mock engine whose embed_batch returns orthogonal vectors so cosine ≈ 0.
    struct OrthogonalEngine {
        dimension: usize,
    }

    #[async_trait]
    impl InferenceEngine for OrthogonalEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            // Return distinct unit-like vectors: first text = [1,0,0,...], second = [0,1,0,...], etc.
            let mut result = Vec::new();
            for (i, _) in texts.iter().enumerate() {
                let mut v = vec![0.0f32; self.dimension];
                if i < self.dimension {
                    v[i] = 1.0;
                }
                result.push(v);
            }
            Ok(result)
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
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

    fn mock_config() -> ModelConfig {
        ModelConfig {
            name: "mock".to_string(),
            engine_type: EngineType::Candle,
            model_path: std::path::PathBuf::from("/tmp/mock"),
            tokenizer_path: None,
            device: DeviceType::Cpu,
            max_batch_size: 1,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: false,
            model_sha256: None,
        }
    }

    // -- is_fallback_triggered default --
    #[test]
    fn test_default_is_fallback_triggered_returns_false() {
        let engine = MockEngine { dimension: 4 };
        assert!(!engine.is_fallback_triggered());
    }

    // -- supports_rerank default --
    #[test]
    fn test_default_supports_rerank_returns_true() {
        let engine = MockEngine { dimension: 4 };
        assert!(engine.supports_rerank());
    }

    // -- rerank default (identical vectors → cosine=1 → sigmoid(1)≈0.731) --
    #[test]
    fn test_default_rerank_identical_embeddings() {
        let engine = MockEngine { dimension: 4 };
        let score = engine.rerank("query", "doc").unwrap();
        // cosine_similarity([1,1,1,1], [1,1,1,1]) = 1.0
        // sigmoid(1.0) = 1/(1+e^{-1}) ≈ 0.7311
        assert!((score - 0.7311).abs() < 0.01, "score={}", score);
    }

    // -- rerank default (orthogonal vectors → cosine≈0 → sigmoid(0)=0.5) --
    #[test]
    fn test_default_rerank_orthogonal_embeddings() {
        let engine = OrthogonalEngine { dimension: 4 };
        let score = engine.rerank("query", "doc").unwrap();
        // cosine_similarity([1,0,0,0], [0,1,0,0]) = 0.0
        // sigmoid(0.0) = 0.5
        assert!((score - 0.5).abs() < 0.01, "score={}", score);
    }

    // -- rerank_batch default (identical vectors) --
    #[test]
    fn test_default_rerank_batch_identical() {
        let engine = MockEngine { dimension: 4 };
        let docs = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];
        let scores = engine.rerank_batch("query", &docs).unwrap();
        assert_eq!(scores.len(), 3);
        for &s in &scores {
            assert!((s - 0.7311).abs() < 0.01, "score={}", s);
        }
    }

    // -- rerank_batch default (orthogonal vectors) --
    #[test]
    fn test_default_rerank_batch_orthogonal() {
        let engine = OrthogonalEngine { dimension: 4 };
        let docs = vec!["doc1".to_string(), "doc2".to_string()];
        let scores = engine.rerank_batch("query", &docs).unwrap();
        assert_eq!(scores.len(), 2);
        for &s in &scores {
            // cosine(query=[1,0,0,0], doc=[0,1,0,0] or [0,0,1,0]) = 0 → sigmoid(0) = 0.5
            assert!((s - 0.5).abs() < 0.01, "score={}", s);
        }
    }

    // -- rerank_batch with empty documents --
    #[test]
    fn test_default_rerank_batch_empty_docs() {
        let engine = MockEngine { dimension: 4 };
        let scores = engine.rerank_batch("query", &[]).unwrap();
        assert!(scores.is_empty());
    }

    // -- try_fallback_to_cpu --
    #[tokio::test]
    async fn test_mock_try_fallback_to_cpu() {
        let mut engine = MockEngine { dimension: 4 };
        let config = mock_config();
        assert!(engine.try_fallback_to_cpu(&config).await.is_ok());
    }

    // -- Direct embed() call --
    #[test]
    fn test_mock_embed_returns_deterministic_vector() {
        let engine = MockEngine { dimension: 8 };
        let vec = engine.embed("hello").unwrap();
        assert_eq!(vec.len(), 8);
        assert!(vec.iter().all(|&v| v == 1.0));
    }

    // -- Direct embed_batch() call --
    #[test]
    fn test_mock_embed_batch_returns_vectors() {
        let engine = MockEngine { dimension: 4 };
        let texts = vec!["a".to_string(), "b".to_string()];
        let vecs = engine.embed_batch(&texts).unwrap();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 4);
    }

    // -- precision() accessor --
    #[test]
    fn test_mock_precision_is_fp32() {
        let engine = MockEngine { dimension: 4 };
        assert_eq!(*engine.precision(), Precision::Fp32);
    }

    // -- supports_mixed_precision() --
    #[test]
    fn test_mock_supports_mixed_precision_false() {
        let engine = MockEngine { dimension: 4 };
        assert!(!engine.supports_mixed_precision());
    }

    // -- OrthogonalEngine direct calls --
    #[test]
    fn test_orthogonal_embed() {
        let engine = OrthogonalEngine { dimension: 4 };
        let vec = engine.embed("test").unwrap();
        assert_eq!(vec, vec![0.0; 4]);
    }

    #[test]
    fn test_orthogonal_precision() {
        let engine = OrthogonalEngine { dimension: 4 };
        assert_eq!(*engine.precision(), Precision::Fp32);
    }

    #[test]
    fn test_orthogonal_supports_mixed_precision() {
        let engine = OrthogonalEngine { dimension: 4 };
        assert!(!engine.supports_mixed_precision());
    }

    #[tokio::test]
    async fn test_orthogonal_try_fallback_to_cpu() {
        let mut engine = OrthogonalEngine { dimension: 4 };
        let config = mock_config();
        assert!(engine.try_fallback_to_cpu(&config).await.is_ok());
    }
}
