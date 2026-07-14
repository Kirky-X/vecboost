// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;

use super::embedding_service::VecboostEmbeddingService;
use super::embedding_service::embedding_service_server::EmbeddingServiceServer;
use crate::error::VecboostError;
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
}
