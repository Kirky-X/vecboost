use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;

use super::embedding_service::VecboostEmbeddingService;
use super::embedding_service::embedding_service_server::EmbeddingServiceServer;
use crate::error::AppError;
use crate::service::embedding::EmbeddingService;

pub struct GrpcServer {
    addr: SocketAddr,
    service: Arc<RwLock<EmbeddingService>>,
}

impl GrpcServer {
    pub fn new(addr: SocketAddr, service: Arc<RwLock<EmbeddingService>>) -> Self {
        Self { addr, service }
    }

    pub async fn run(self) -> Result<(), AppError> {
        let embedding_service = VecboostEmbeddingService::new(self.service.clone());

        tracing::info!("Starting gRPC server on {}", self.addr);

        Server::builder()
            .add_service(EmbeddingServiceServer::new(embedding_service))
            .serve(self.addr)
            .await
            .map_err(|e| AppError::io_error(format!("gRPC server error: {}", e)))?;

        Ok(())
    }
}
