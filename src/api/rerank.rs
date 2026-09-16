// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Rerank forge handlers — HTTP/CLI/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/embedding.rs` 相同模式：协议无关的 `*_handler` 函数包含
//! 业务逻辑，`forge_*` / `cli_*` / `grpc_*` 仅为薄包装。

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
use crate::domain::{
    BatchRerankQueryStatus, BatchRerankRequest, BatchRerankResponse, RerankRequest, RerankResponse,
};
use crate::error::VecboostError;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::registry::RerankModule;
use crate::service::rerank::RerankService;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use std::sync::Arc;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use tokio::sync::RwLock;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions
// =============================================================================

pub async fn rerank(
    svc: &RerankService,
    req: RerankRequest,
    max_documents: usize,
    max_query_length: usize,
) -> Result<RerankResponse, VecboostError> {
    svc.process_rerank(req, max_documents, max_query_length)
        .await
}

pub async fn rerank_batch(
    svc: &RerankService,
    req: BatchRerankRequest,
    max_documents: usize,
    max_query_length: usize,
) -> Result<BatchRerankResponse, VecboostError> {
    let mut responses = Vec::with_capacity(req.queries.len());
    let mut statuses = Vec::with_capacity(req.queries.len());
    for (index, q) in req.queries.into_iter().enumerate() {
        match svc.process_rerank(q, max_documents, max_query_length).await {
            Ok(resp) => {
                responses.push(resp);
                statuses.push(BatchRerankQueryStatus {
                    index,
                    ok: true,
                    error: None,
                });
            }
            Err(e) => {
                log::warn!("Batch rerank: individual query failed, skipping: {}", e);
                // 单个 query 失败不影响其他 query 的处理；失败经 statuses
                // 可视化（审计建议），调用方可将响应对位回请求
                statuses.push(BatchRerankQueryStatus {
                    index,
                    ok: false,
                    error: Some(e.to_string()),
                });
            }
        }
    }
    Ok(BatchRerankResponse {
        responses,
        statuses,
    })
}

// =============================================================================
// Protocol-agnostic handlers
// =============================================================================

/// Load rerank service and limits from the global kit.
///
/// Returns the `Arc<RwLock<RerankService>>` capability together with the
/// configured `max_documents_per_query` and `max_query_length` limits so
/// callers only need to acquire a read-guard and dispatch.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn load_rerank_service() -> Result<(Arc<RwLock<RerankService>>, usize, usize), ApiError> {
    let st = state().map_err(to_api_error)?;
    let rerank_config = st
        .kit
        .config::<crate::config::app::RerankConfig>()
        .unwrap_or_default();
    let svc = st
        .kit
        .require::<RerankModule>()
        .map_err(kit_internal_error)?;
    Ok((
        svc,
        rerank_config.max_documents_per_query,
        rerank_config.max_query_length,
    ))
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn rerank_handler(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    let (svc, max_documents, max_query_length) = load_rerank_service().await?;
    let guard = svc.read().await;
    rerank(&guard, req, max_documents, max_query_length)
        .await
        .map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn rerank_batch_handler(req: BatchRerankRequest) -> Result<BatchRerankResponse, ApiError> {
    let (svc, max_documents, max_query_length) = load_rerank_service().await?;
    let guard = svc.read().await;
    rerank_batch(&guard, req, max_documents, max_query_length)
        .await
        .map_err(to_api_error)
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "rerank",
    version = 1,
    path = "/rerank",
    method = "POST",
    tool_name = "rerank",
    description = "Rerank documents by relevance to a query"
)]
pub async fn forge_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "rerank_batch",
    version = 1,
    path = "/rerank/batch",
    method = "POST",
    tool_name = "rerank_batch",
    description = "Batch rerank multiple queries against document sets"
)]
pub async fn forge_rerank_batch(req: BatchRerankRequest) -> Result<BatchRerankResponse, ApiError> {
    rerank_batch_handler(req).await
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "rerank",
    version = 1,
    cli = true,
    description = "Rerank documents by relevance to a query"
)]
pub async fn cli_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

// =============================================================================
// gRPC forge handlers
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "rerank",
    version = 1,
    grpc_method = "vecboost.rerank",
    description = "Rerank documents by relevance to a query"
)]
pub async fn grpc_rerank(req: RerankRequest) -> Result<RerankResponse, ApiError> {
    rerank_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "rerank_batch",
    version = 1,
    grpc_method = "vecboost.rerank_batch",
    description = "Batch rerank multiple queries against document sets"
)]
pub async fn grpc_rerank_batch(req: BatchRerankRequest) -> Result<BatchRerankResponse, ApiError> {
    rerank_batch_handler(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::{ModelConfig, Precision};
    use crate::engine::InferenceEngine;
    use async_trait::async_trait;

    struct MockRerankEngine;

    #[async_trait]
    impl InferenceEngine for MockRerankEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.0; 128])
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
        }
        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }
        fn supports_mixed_precision(&self) -> bool {
            false
        }
        fn rerank(&self, _query: &str, document: &str) -> Result<f32, VecboostError> {
            Ok((document.len() as f32) / 100.0)
        }
        fn rerank_batch(
            &self,
            query: &str,
            documents: &[String],
        ) -> Result<Vec<f32>, VecboostError> {
            documents
                .iter()
                .map(|doc| self.rerank(query, doc))
                .collect()
        }
        fn supports_rerank(&self) -> bool {
            true
        }
        async fn try_fallback_to_cpu(
            &mut self,
            _config: &ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_svc() -> RerankService {
        let engine: std::sync::Arc<tokio::sync::RwLock<dyn InferenceEngine + Send + Sync>> =
            std::sync::Arc::new(tokio::sync::RwLock::new(MockRerankEngine));
        RerankService::new(engine, None)
    }

    #[tokio::test]
    async fn test_sdk_rerank_delegates_to_service() {
        let svc = make_svc();
        let req = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc one".to_string(), "doc two".to_string()],
            top_k: None,
            return_documents: None,
        };
        let result = rerank(&svc, req, 100, 8192).await.unwrap();
        assert_eq!(result.results.len(), 2);
    }

    #[tokio::test]
    async fn test_sdk_rerank_batch_processes_multiple_queries() {
        let svc = make_svc();
        let req = BatchRerankRequest {
            queries: vec![
                RerankRequest {
                    query: "query 1".to_string(),
                    documents: vec!["doc a".to_string()],
                    top_k: None,
                    return_documents: None,
                },
                RerankRequest {
                    query: "query 2".to_string(),
                    documents: vec!["doc b".to_string(), "doc c".to_string()],
                    top_k: None,
                    return_documents: None,
                },
            ],
        };
        let result = rerank_batch(&svc, req, 100, 8192).await.unwrap();
        assert_eq!(result.responses.len(), 2);
        assert_eq!(result.statuses.len(), 2);
        assert!(result.statuses.iter().all(|s| s.ok && s.error.is_none()));
    }

    #[tokio::test]
    async fn test_sdk_rerank_batch_empty_queries() {
        let svc = make_svc();
        let req = BatchRerankRequest { queries: vec![] };
        let result = rerank_batch(&svc, req, 100, 8192).await.unwrap();
        assert!(result.responses.is_empty());
        assert!(result.statuses.is_empty());
    }

    #[tokio::test]
    async fn test_sdk_rerank_batch_partial_failure_statuses() {
        // 容错可视化：失败 query 不产生响应，但在 statuses 中可见
        let svc = make_svc();
        let req = BatchRerankRequest {
            queries: vec![
                RerankRequest {
                    query: "valid".to_string(),
                    documents: vec!["doc a".to_string()],
                    top_k: None,
                    return_documents: None,
                },
                RerankRequest {
                    query: String::new(),
                    documents: vec!["doc b".to_string()],
                    top_k: None,
                    return_documents: None,
                },
            ],
        };
        let result = rerank_batch(&svc, req, 100, 8192).await.unwrap();
        assert_eq!(result.responses.len(), 1, "失败 query 不产生响应");
        assert_eq!(result.statuses.len(), 2, "statuses 与请求一一对位");
        assert!(result.statuses[0].ok);
        assert!(!result.statuses[1].ok);
        assert!(result.statuses[1].error.is_some());
        assert_eq!(result.statuses[1].index, 1);
    }

    #[tokio::test]
    async fn test_sdk_rerank_empty_docs_returns_error() {
        let svc = make_svc();
        let req = RerankRequest {
            query: "test".to_string(),
            documents: vec![],
            top_k: None,
            return_documents: None,
        };
        let result = rerank(&svc, req, 100, 8192).await;
        assert!(result.is_err());
    }
}
