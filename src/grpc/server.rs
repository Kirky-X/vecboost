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
