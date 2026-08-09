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
        let similarity =
            crate::utils::vector::cosine_similarity(&embeddings[0], &embeddings[1])?;
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
                let similarity =
                    crate::utils::vector::cosine_similarity(query_emb, doc_emb)?;
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

    /// 尝试降级到 CPU（在 OOM 时调用）
    async fn try_fallback_to_cpu(&mut self, config: &ModelConfig) -> Result<(), VecboostError>;
}

#[allow(clippy::large_enum_variant)]
pub enum AnyEngine {
    Candle(candle_engine::CandleEngine),
    #[cfg(feature = "onnx")]
    Onnx(onnx_engine::OnnxEngine),
}
