// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;

use crate::error::VecboostError;
use crate::grpc::embedding_service::VecboostEmbeddingService;
use crate::grpc::embedding_service::embedding_service_server::EmbeddingServiceServer;
use crate::service::embedding::EmbeddingService;

pub struct GrpcServer {
    addr: SocketAddr,
    service: Arc<RwLock<EmbeddingService>>,
}

impl GrpcServer {
    pub fn new(addr: SocketAddr, service: Arc<RwLock<EmbeddingService>>) -> Self {
        Self { addr, service }
    }

    pub async fn run(self) -> Result<(), VecboostError> {
        let embedding_service = VecboostEmbeddingService::new(self.service.clone());

        tracing::info!("Starting gRPC server on {}", self.addr);

        Server::builder()
            .add_service(EmbeddingServiceServer::new(embedding_service))
            .serve(self.addr)
            .await
            .map_err(|e| VecboostError::io_error(format!("gRPC server error: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use crate::engine::InferenceEngine;
    use crate::error::VecboostError;
    use crate::service::embedding::EmbeddingService;
    use async_trait::async_trait;
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// Mock engine that returns deterministic embeddings for testing.
    struct MockEngine {
        dimension: usize,
    }

    impl MockEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    /// Build an `Arc<RwLock<EmbeddingService>>` backed by a 384-dim `MockEngine`.
    fn make_service() -> Arc<RwLock<EmbeddingService>> {
        let mock_engine = MockEngine::new(384);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        Arc::new(RwLock::new(EmbeddingService::new(engine, None)))
    }

    /// Find a free TCP port by binding to port 0 and reading the assigned port.
    fn find_free_port() -> SocketAddr {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind for port probe");
        listener.local_addr().expect("failed to get local addr")
    }

    #[test]
    fn test_server_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        // Server should be created without panic; fields are private so we
        // only verify construction succeeds. Dropping the server is safe.
        drop(server);
    }

    #[tokio::test]
    async fn test_server_start_shutdown() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        // Spawn the server in a background task.
        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        // Give the server a moment to bind and start serving.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Verify the server is listening by opening a TCP connection.
        let connect_result = tokio::net::TcpStream::connect(addr).await;
        assert!(
            connect_result.is_ok(),
            "gRPC server should be listening on {}",
            addr
        );
        drop(connect_result);

        // Shut down the server by aborting the spawned task. Dropping the
        // tonic serve future closes the underlying listener.
        handle.abort();
        let join_result = handle.await;
        assert!(join_result.is_err(), "task should be cancelled after abort");

        // Give the OS a brief moment to release the listening socket.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify the server is no longer accepting connections.
        let reconnect_result = tokio::net::TcpStream::connect(addr).await;
        assert!(
            reconnect_result.is_err(),
            "gRPC server should no longer be listening on {} after shutdown",
            addr
        );
    }

    #[tokio::test]
    async fn test_server_run_bind_error() {
        // Occupy a TCP port so tonic's bind to the same addr fails.
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind for port probe");
        let addr = listener
            .local_addr()
            .expect("failed to get local addr for occupied port");

        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let result = server.run().await;
        assert!(
            result.is_err(),
            "run() should fail when the port is already occupied"
        );
        match result.unwrap_err() {
            VecboostError::IoError(msg) => {
                assert!(
                    msg.contains("gRPC server error"),
                    "expected 'gRPC server error' in message, got: {}",
                    msg
                );
            }
            other => panic!("expected IoError, got {:?}", other),
        }

        drop(listener);
    }

    #[test]
    fn test_server_new_with_ipv6_addr() {
        let addr: SocketAddr = "[::1]:0".parse().unwrap();
        let service = make_service();
        let server = GrpcServer::new(addr, service);
        drop(server);
    }

    #[test]
    fn test_server_new_with_loopback_addr() {
        let addr: SocketAddr = "127.0.0.1:50051".parse().unwrap();
        let service = make_service();
        let server = GrpcServer::new(addr, service);
        drop(server);
    }

    #[test]
    fn test_server_new_with_different_dimensions() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mock_engine = MockEngine::new(768);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let server = GrpcServer::new(addr, service);
        drop(server);
    }

    #[test]
    fn test_server_new_with_multiple_instances() {
        let addr1: SocketAddr = "127.0.0.1:50051".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:50052".parse().unwrap();
        let service1 = make_service();
        let service2 = make_service();
        let server1 = GrpcServer::new(addr1, service1);
        let server2 = GrpcServer::new(addr2, service2);
        drop(server1);
        drop(server2);
    }

    #[tokio::test]
    async fn test_server_run_returns_io_error_on_connect_to_used_port() {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind for port probe");
        let addr = listener
            .local_addr()
            .expect("failed to get local addr for occupied port");

        let mock_engine = MockEngine::new(128);
        let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> =
            Arc::new(RwLock::new(mock_engine));
        let service = Arc::new(RwLock::new(EmbeddingService::new(engine, None)));
        let server = GrpcServer::new(addr, service);

        let result = server.run().await;
        assert!(result.is_err());
        let err_msg = match result.unwrap_err() {
            VecboostError::IoError(msg) => msg,
            other => panic!("expected IoError, got {:?}", other),
        };
        assert!(err_msg.contains("gRPC server error"));

        drop(listener);
    }

    /// Verify MockEngine::embed returns deterministic embeddings.
    #[test]
    fn test_mock_engine_embed_returns_expected_vector() {
        let engine = MockEngine::new(128);
        let result = engine.embed("hello").expect("embed should succeed");
        assert_eq!(result.len(), 128);
        assert!(result.iter().all(|&v| v == 0.5));
    }

    /// Verify MockEngine::embed_batch returns one vector per input text.
    #[test]
    fn test_mock_engine_embed_batch_returns_correct_count() {
        let engine = MockEngine::new(64);
        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert_eq!(result.len(), 3);
        for v in &result {
            assert_eq!(v.len(), 64);
        }
    }

    /// Verify MockEngine::embed_batch with empty slice returns empty vec.
    #[test]
    fn test_mock_engine_embed_batch_empty_input() {
        let engine = MockEngine::new(32);
        let texts: Vec<String> = vec![];
        let result = engine
            .embed_batch(&texts)
            .expect("embed_batch should succeed");
        assert!(result.is_empty());
    }

    /// Verify MockEngine::precision returns Fp32.
    #[test]
    fn test_mock_engine_precision() {
        let engine = MockEngine::new(8);
        assert_eq!(engine.precision(), &Precision::Fp32);
    }

    /// Verify MockEngine::supports_mixed_precision returns false.
    #[test]
    fn test_mock_engine_supports_mixed_precision() {
        let engine = MockEngine::new(8);
        assert!(!engine.supports_mixed_precision());
    }

    /// Verify MockEngine::try_fallback_to_cpu returns Ok.
    #[tokio::test]
    async fn test_mock_engine_try_fallback_to_cpu() {
        let mut engine = MockEngine::new(8);
        let config = crate::config::model::ModelConfig::default();
        let result = engine.try_fallback_to_cpu(&config).await;
        assert!(result.is_ok());
    }

    /// Verify make_service produces a service backed by a 384-dim engine.
    #[tokio::test]
    async fn test_make_service_embeds_correctly() {
        let service = make_service();
        let guard = service.read().await;
        let result = guard
            .process_text(
                crate::domain::EmbedRequest {
                    text: "test".to_string(),
                    normalize: Some(true),
                },
                None,
            )
            .await;
        assert!(result.is_ok(), "process_text should succeed");
        let response = result.unwrap();
        assert_eq!(response.dimension, 384);
    }

    /// Verify find_free_port returns an address with port 0 assigned.
    #[test]
    fn test_find_free_port_returns_loopback() {
        let addr = find_free_port();
        assert!(addr.is_ipv4());
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
    }

    /// Verify the gRPC server can handle an actual embed request end-to-end.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_handles_embed_request() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        // Poll-connect until the server is ready (max ~2s).
        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening on {}", addr);

        use crate::grpc::embedding_service::EmbedRequest as GrpcEmbedRequest;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcEmbedRequest {
            text: "hello world".to_string(),
            normalize: Some(true),
        });
        let response = client
            .embed(request)
            .await
            .expect("embed RPC should succeed");
        let inner = response.into_inner();
        assert_eq!(inner.dimension, 384);
        assert_eq!(inner.embedding.len(), 384);

        handle.abort();
    }

    /// Verify the gRPC server rejects empty text with InvalidArgument.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_embed_rejects_empty_text() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening");

        use crate::grpc::embedding_service::EmbedRequest as GrpcEmbedRequest;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcEmbedRequest {
            text: "".to_string(),
            normalize: Some(true),
        });
        let result = client.embed(request).await;
        assert!(result.is_err(), "empty text should return an error");
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);

        handle.abort();
    }

    /// Verify the gRPC server handles batch embed requests.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_handles_embed_batch_request() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening");

        use crate::grpc::embedding_service::BatchEmbedRequest as GrpcBatchEmbedRequest;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcBatchEmbedRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
            normalize: Some(true),
        });
        let response = client
            .embed_batch(request)
            .await
            .expect("embed_batch RPC should succeed");
        let inner = response.into_inner();
        assert_eq!(inner.total_count, 2);
        assert_eq!(inner.embeddings.len(), 2);
        for emb in &inner.embeddings {
            assert_eq!(emb.dimension, 384);
            assert_eq!(emb.embedding.len(), 384);
        }

        handle.abort();
    }

    /// Verify the gRPC server handles compute_similarity requests.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_handles_compute_similarity() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening");

        use crate::grpc::embedding_service::SimilarityRequest as GrpcSimilarityRequest;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcSimilarityRequest {
            vector1: vec![1.0, 0.0, 0.0],
            vector2: vec![1.0, 0.0, 0.0],
            metric: "cosine".to_string(),
        });
        let response = client
            .compute_similarity(request)
            .await
            .expect("compute_similarity RPC should succeed");
        let inner = response.into_inner();
        assert!(
            (inner.score - 1.0).abs() < 1e-5,
            "cosine of identical vectors should be 1.0"
        );
        assert_eq!(inner.metric, "cosine");

        handle.abort();
    }

    /// Verify the gRPC server rejects similarity with mismatched dimensions.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_similarity_rejects_mismatched_dims() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening");

        use crate::grpc::embedding_service::SimilarityRequest as GrpcSimilarityRequest;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcSimilarityRequest {
            vector1: vec![1.0, 0.0],
            vector2: vec![1.0, 0.0, 0.0],
            metric: "cosine".to_string(),
        });
        let result = client.compute_similarity(request).await;
        assert!(
            result.is_err(),
            "mismatched dimensions should return an error"
        );
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);

        handle.abort();
    }

    /// Verify the gRPC server handles health check requests.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_server_run_handles_health_check() {
        let addr = find_free_port();
        let service = make_service();
        let server = GrpcServer::new(addr, service);

        let handle = tokio::spawn(async move {
            let _ = server.run().await;
        });

        let mut connected = false;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                connected = true;
                break;
            }
        }
        assert!(connected, "server should be listening");

        use crate::grpc::embedding_service::Empty as GrpcEmpty;
        use crate::grpc::embedding_service::embedding_service_client::EmbeddingServiceClient;

        let endpoint = format!("http://{}", addr);
        let mut client = EmbeddingServiceClient::connect(endpoint)
            .await
            .expect("client should connect");

        let request = tonic::Request::new(GrpcEmpty {});
        let response = client
            .health_check(request)
            .await
            .expect("health_check RPC should succeed");
        assert_eq!(response.into_inner().status, "OK");

        handle.abort();
    }
}
