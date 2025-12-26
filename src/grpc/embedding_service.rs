use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::domain::{
    BatchEmbedRequest as DomainBatchEmbedRequest, EmbedRequest as DomainEmbedRequest,
    ModelSwitchRequest as DomainModelSwitchRequest,
};
use crate::service::embedding::EmbeddingService;
use crate::utils::AggregationMode;

tonic::include_proto!("vecboost");

pub struct VecboostEmbeddingService {
    service: Arc<RwLock<EmbeddingService>>,
}

impl VecboostEmbeddingService {
    pub fn new(service: Arc<RwLock<EmbeddingService>>) -> Self {
        Self { service }
    }
}

#[tonic::async_trait]
impl embedding_service_server::EmbeddingService for VecboostEmbeddingService {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();

        let domain_req = DomainEmbedRequest {
            text: req.text,
            normalize: Some(req.normalize.unwrap_or(true)),
        };

        let service_guard = self.service.read().await;
        let domain_res = service_guard
            .process_text(domain_req)
            .await
            .map_err(|e| Status::internal(format!("Embedding failed: {}", e)))?;

        let res = EmbedResponse {
            embedding: domain_res.embedding,
            dimension: domain_res.dimension as i64,
            processing_time_ms: domain_res.processing_time_ms as f64,
        };

        Ok(Response::new(res))
    }

    async fn embed_batch(
        &self,
        request: Request<BatchEmbedRequest>,
    ) -> Result<Response<BatchEmbedResponse>, Status> {
        let req = request.into_inner();

        let domain_req = DomainBatchEmbedRequest {
            texts: req.texts,
            mode: None,
            normalize: req.normalize,
        };

        let service_guard = self.service.read().await;
        let domain_res = service_guard
            .process_batch(domain_req)
            .await
            .map_err(|e| Status::internal(format!("Batch embedding failed: {}", e)))?;

        let total_count = domain_res.embeddings.len();
        let embeddings: Vec<EmbedResponse> = domain_res
            .embeddings
            .into_iter()
            .map(|e| EmbedResponse {
                embedding: e.embedding,
                dimension: domain_res.dimension as i64,
                processing_time_ms: 0.0,
            })
            .collect();

        let res = BatchEmbedResponse {
            embeddings,
            total_count: total_count as i64,
            processing_time_ms: domain_res.processing_time_ms as f64,
        };

        Ok(Response::new(res))
    }

    async fn compute_similarity(
        &self,
        request: Request<SimilarityRequest>,
    ) -> Result<Response<SimilarityResponse>, Status> {
        let req = request.into_inner();

        if req.vector1.is_empty() || req.vector2.is_empty() {
            return Err(Status::invalid_argument("Vectors cannot be empty"));
        }

        if req.vector1.len() != req.vector2.len() {
            return Err(Status::invalid_argument(
                "Vectors must have the same dimension",
            ));
        }

        let metric = if req.metric.is_empty() {
            "cosine".to_string()
        } else {
            req.metric.clone()
        };

        let metric_clone = metric.clone();

        let score = tokio::task::spawn_blocking(move || {
            let mut v1 = req.vector1;
            let mut v2 = req.vector2;

            let result = match metric_clone.as_str() {
                "cosine" => {
                    crate::utils::normalize_l2(&mut v1);
                    crate::utils::normalize_l2(&mut v2);
                    crate::utils::cosine_similarity(&v1, &v2)
                }
                "euclidean" => crate::utils::euclidean_distance(&v1, &v2),
                "manhattan" => crate::utils::manhattan_distance(&v1, &v2),
                "dot" => crate::utils::dot_product(&v1, &v2),
                _ => crate::utils::cosine_similarity(&v1, &v2),
            };

            result.map_err(|e| format!("Computation error: {}", e))
        })
        .await
        .map_err(|e| Status::internal(format!("Similarity computation failed: {}", e)))?
        .map_err(Status::internal)?;

        let res = SimilarityResponse {
            score: score as f64,
            metric,
        };

        Ok(Response::new(res))
    }

    async fn embed_file(
        &self,
        request: Request<FileEmbedRequest>,
    ) -> Result<Response<FileEmbedResponse>, Status> {
        let req = request.into_inner();

        let mode = match req.mode.as_deref() {
            Some("document") => AggregationMode::Document,
            Some("paragraph") => AggregationMode::Paragraph,
            Some("average") => AggregationMode::Average,
            _ => AggregationMode::Document,
        };

        let path = PathBuf::from(&req.path);

        let service_guard = self.service.read().await;
        let stats = service_guard
            .get_processing_stats(&path)
            .map_err(|e| Status::invalid_argument(format!("Failed to get file stats: {}", e)))?;

        let output = service_guard
            .embed_file(&path, mode)
            .await
            .map_err(|e| Status::internal(format!("File embedding failed: {}", e)))?;

        let (embedding, paragraphs) = match output {
            crate::domain::EmbeddingOutput::Single(response) => {
                (Some(response.embedding), Vec::new())
            }
            crate::domain::EmbeddingOutput::Paragraphs(paragraphs) => {
                let paragraph_embeddings: Vec<ParagraphEmbedding> = paragraphs
                    .into_iter()
                    .map(|p| ParagraphEmbedding {
                        index: p.position as i32,
                        text: p.text_preview,
                        embedding: p.embedding,
                    })
                    .collect();
                (None, paragraph_embeddings)
            }
        };

        let res = FileEmbedResponse {
            mode: format!("{:?}", mode),
            stats: Some(FileStats {
                total_lines: stats.total_chunks as i64,
                total_chars: 0,
                total_paragraphs: 0,
                processed_chunks: stats.successful_chunks as i64,
                processing_time_ms: stats.processing_time_ms as f64,
            }),
            embedding: embedding.unwrap_or_default(),
            paragraphs,
        };

        Ok(Response::new(res))
    }

    async fn model_switch(
        &self,
        request: Request<ModelSwitchRequest>,
    ) -> Result<Response<ModelSwitchResponse>, Status> {
        let req = request.into_inner();

        let device_type = req.device_type.as_deref().map(|dt| match dt {
            "cpu" => crate::config::model::DeviceType::Cpu,
            "cuda" => crate::config::model::DeviceType::Cuda,
            "metal" => crate::config::model::DeviceType::Metal,
            "rocm" => crate::config::model::DeviceType::Amd,
            "amd" => crate::config::model::DeviceType::Amd,
            _ => crate::config::model::DeviceType::Cpu,
        });

        let domain_req = DomainModelSwitchRequest {
            model_name: req.model_name,
            model_path: None,
            tokenizer_path: None,
            device: device_type,
            max_batch_size: None,
            pooling_mode: None,
            expected_dimension: None,
            memory_limit_bytes: None,
            oom_fallback_enabled: None,
        };

        let mut service_guard = self.service.write().await;
        let domain_res = service_guard
            .switch_model(domain_req)
            .await
            .map_err(|e| Status::internal(format!("Model switch failed: {}", e)))?;

        let res = ModelSwitchResponse {
            success: domain_res.success,
            message: domain_res.message,
            model_info: Some(ModelInfo {
                name: domain_res.current_model,
                engine_type: "unknown".to_string(),
                device_type: "cpu".to_string(),
                dimension: 0,
                precision: "fp32".to_string(),
                max_batch_size: 0,
                cache_enabled: false,
                cache_size: 0,
            }),
        };

        Ok(Response::new(res))
    }

    async fn get_current_model(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ModelInfo>, Status> {
        let service_guard = self.service.read().await;
        let info = service_guard
            .get_model_info()
            .ok_or_else(|| Status::not_found("No model loaded"))?;

        let res = ModelInfo {
            name: info.name,
            engine_type: info.engine_type,
            device_type: "cpu".to_string(),
            dimension: info.dimension.unwrap_or(0) as i64,
            precision: "fp32".to_string(),
            max_batch_size: 0,
            cache_enabled: info.is_loaded,
            cache_size: 0,
        };

        Ok(Response::new(res))
    }

    async fn get_model_info(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ModelMetadata>, Status> {
        let service_guard = self.service.read().await;
        let metadata = service_guard
            .get_model_metadata()
            .ok_or_else(|| Status::not_found("No model loaded"))?;

        let res = ModelMetadata {
            model_name: metadata.name,
            version: metadata.version,
            architecture: metadata.engine_type,
            max_position_embeddings: metadata.max_input_length as i64,
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            intermediate_size: 0,
            supported_devices: vec![],
            supported_precisions: vec![],
        };

        Ok(Response::new(res))
    }

    async fn list_models(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ModelListResponse>, Status> {
        let service_guard = self.service.read().await;
        let result = service_guard.list_available_models();

        let current_model = service_guard.get_model_info().map(|info| info.name);

        let models: Vec<ModelMetadata> = result
            .models
            .into_iter()
            .map(|m| ModelMetadata {
                model_name: m.name,
                version: "1.0".to_string(),
                architecture: m.engine_type,
                max_position_embeddings: 0,
                vocab_size: 0,
                hidden_size: 0,
                num_hidden_layers: 0,
                num_attention_heads: 0,
                intermediate_size: 0,
                supported_devices: vec![],
                supported_precisions: vec![],
            })
            .collect();

        let res = ModelListResponse {
            models,
            current_model: current_model.unwrap_or_default(),
        };

        Ok(Response::new(res))
    }

    async fn health_check(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<HealthResponse>, Status> {
        let service_guard = self.service.read().await;
        let model_loaded = service_guard.get_model_info().map(|info| info.name);

        let uptime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let res = HealthResponse {
            status: "OK".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime: format!("{}s", uptime),
            model_loaded,
        };

        Ok(Response::new(res))
    }
}
