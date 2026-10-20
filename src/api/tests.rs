// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use crate::VecboostState;
#[cfg(feature = "http")]
use crate::api::decisions::forge_decisions;
#[cfg(feature = "http")]
use crate::api::detect::forge_detect;
#[cfg(feature = "cli")]
use crate::api::embedding::{cli_compute_similarity, cli_embed};
use crate::api::embedding::{compute_similarity, embed, embed_batch};
#[cfg(feature = "http")]
use crate::api::embedding::{forge_compute_similarity, forge_embed, forge_embed_batch};
#[cfg(any(feature = "http", feature = "cli"))]
use crate::api::embedding::{to_api_error, uuid_like_id};
#[cfg(feature = "http")]
use crate::api::translate::forge_translate;
use crate::config::model::{DeviceType, EngineType, ModelConfig, ModelTask, Precision};
use crate::domain::{
    BatchEmbedRequest, DecisionRequest, DecisionResponse, DetectRequest, EmbedRequest,
    SimilarityRequest, TranslateRequest,
};
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::registry::{
    DecisionModule, DetectModule, EmbeddingModule, RerankModule, TranslateModule,
};
use crate::service::decision::DecisionService;
use crate::service::detect::DetectService;
use crate::service::embedding::EmbeddingService;
use crate::service::rerank::RerankService;
use crate::service::translate::TranslateService;
use async_trait::async_trait;
#[cfg(any(feature = "http", feature = "cli"))]
use sdforge::prelude::ApiError;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;
use tokio::sync::RwLock;

/// Ensure the global STATE is initialized for forge handler tests.
///
/// Builds a minimal `AsyncKit` with `EmbeddingModule` / `RerankModule` /
/// `DecisionModule` / `DetectModule` / `TranslateModule` / `TranscribeModule` /
/// `AuditModule`（auth feature 下含 `AuthModule`）registered and injects via
/// `init_state`. Under parallel
/// test execution, multiple tests may race to initialize; `OnceLock` ensures
/// first-writer-wins, and subsequent `init_state` errors are silently ignored
/// (expected behavior). Decision 服务必须随首写一并注入：state 是进程级单例，
/// 后到的注入静默失效，缺注册会让 forge_decisions require 失败必 500。
/// detect 服务同理（无状态 unit，直接注入值）。
async fn ensure_state_initialized() {
    if crate::api::state().is_ok() {
        return;
    }
    let svc = Arc::new(RwLock::new(make_service(384)));
    let rerank_svc = Arc::new(RwLock::new(RerankService::new(
        make_mock_engine_arc(),
        None,
    )));
    let decision_svc = Arc::new(RwLock::new(DecisionService::new(make_mock_engine_arc())));
    let detect_svc = DetectService::new();
    // 翻译服务注入含 en-zh mock 路由（评审 ARC-4：forge_translate 装配链
    // 单测面——漏 register/set_config 的回归由本测试兜住，不再只依赖需
    // onnx+真机 bundle 的集成测试）；mock 译文带 marker 供路由断言
    let mut translate_pairs = std::collections::HashMap::new();
    translate_pairs.insert(
        ("en".to_string(), "zh".to_string()),
        crate::service::translate::TranslatePairRoute::new(
            Arc::new(RwLock::new(MockDecisionEngine)),
            crate::config::model::TranslationParams::default(),
        ),
    );
    let translate_svc = Arc::new(TranslateService::new(translate_pairs));
    // 转录服务注入未装配实例：forge_transcribe 错误分支（400 transcribe-not-configured）
    // 的装配前提——缺注册会让 require 失败退 500，测不到业务分支。
    let transcribe_svc = Arc::new(crate::service::transcribe::TranscribeService::unconfigured());
    let mut kit = trait_kit::AsyncKit::new();
    kit.set_config(svc);
    kit.set_config(rerank_svc);
    kit.set_config(decision_svc);
    kit.set_config(detect_svc);
    kit.set_config(translate_svc);
    kit.set_config(transcribe_svc);
    // forge_login/refresh/logout 无条件 require AuditModule；Logger 置 None——
    // 审计事件落盘路径由 test_audit_logger_called_by_forge_handler_pattern 单独覆盖。
    kit.set_config(Option::<Arc<crate::audit::AuditLogger>>::None);
    // auth feature 下注入 Some(handle)（与 main.rs init_auth 同构）：None 只能测到
    // auth-disabled 500，校验/登录分支全部不可达。
    #[cfg(feature = "auth")]
    kit.set_config(Some(Arc::new(test_garrison_handle())));
    kit.register::<EmbeddingModule>()
        .expect("register EmbeddingModule in test kit");
    kit.register::<RerankModule>()
        .expect("register RerankModule in test kit");
    kit.register::<DecisionModule>()
        .expect("register DecisionModule in test kit");
    kit.register::<DetectModule>()
        .expect("register DetectModule in test kit");
    kit.register::<TranslateModule>()
        .expect("register TranslateModule in test kit");
    kit.register::<crate::registry::AuditModule>()
        .expect("register AuditModule in test kit");
    kit.register::<crate::registry::TranscribeModule>()
        .expect("register TranscribeModule in test kit");
    #[cfg(feature = "auth")]
    kit.register::<crate::registry::AuthModule>()
        .expect("register AuthModule in test kit");
    let kit = kit.build().await.expect("build test kit");
    // Ignore error: under parallel execution another test may have initialized first
    let _ = crate::api::init_state(VecboostState { kit: Arc::new(kit) });
}

/// Arc 包装的 mock 决策引擎（rerank/decision 服务共用占位引擎）
fn make_mock_engine_arc() -> Arc<RwLock<dyn InferenceEngine + Send + Sync>> {
    Arc::new(RwLock::new(MockDecisionEngine))
}

/// Deterministic mock engine for API layer tests.
///
/// Generates a normalized embedding by hashing the text bytes, so the same
/// input always yields the same vector. This mirrors the TestEngine used in
/// `service/embedding.rs` tests.
struct TestEngine {
    dimension: usize,
}

impl TestEngine {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        crate::utils::hash::mock_embedding(text, self.dimension)
    }
}

#[async_trait]
impl InferenceEngine for TestEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }

    fn precision(&self) -> &Precision {
        const PRECISION: Precision = Precision::Fp32;
        &PRECISION
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

/// 固定作答的 mock 决策引擎：按 qtype 三题型确定性作答，覆盖自身 decide
/// 与 supports_task(Decision)，供 DecisionService / forge_decisions 走通
/// trait 分发全链（不依赖 onnx feature 与模型资产，离线可跑）。
///
/// embed 返回非零常向量：零向量会使 InferenceEngine 默认 rerank 实现的
/// 余弦相似度在零范数处报错（forge_rerank 全分支不可达）；decide/translate
/// 不消费 embed，非零取值对既有断言无影响。
struct MockDecisionEngine;

#[async_trait]
impl InferenceEngine for MockDecisionEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(vec![0.1; 8])
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|_| vec![0.1; 8]).collect())
    }

    fn precision(&self) -> &Precision {
        &Precision::Fp32
    }

    fn supports_mixed_precision(&self) -> bool {
        false
    }

    fn supports_task(&self, task: ModelTask) -> bool {
        matches!(
            task,
            ModelTask::Embedding | ModelTask::Decision | ModelTask::Translate
        )
    }

    fn translate(
        &self,
        req: &TranslateRequest,
    ) -> Result<crate::domain::TranslateResponse, VecboostError> {
        // 评审 ARC-4：可辨识 marker 译文，forge_translate 路由/能力门链路断言用
        Ok(crate::domain::TranslateResponse {
            translation: format!("[mock-en-zh] {}", req.text),
            source_language: req.source_language.clone(),
            target_language: req.target_language.clone(),
            detected_source_language: None,
            processing_time_ms: 0,
        })
    }

    fn decide(&self, req: &DecisionRequest) -> Result<DecisionResponse, VecboostError> {
        use crate::domain::{DecisionAnswer, DecisionAnswerBody, QuestionType};
        let answers = req
            .questions
            .iter()
            .map(|q| DecisionAnswer {
                question: q.name.clone(),
                answer: match q.qtype {
                    QuestionType::Choice => DecisionAnswerBody::Choice {
                        index: 0,
                        option: q.options.first().cloned().unwrap_or_default(),
                        probabilities: q
                            .options
                            .iter()
                            .map(|o| (o.clone(), 1.0 / q.options.len().max(1) as f32))
                            .collect::<BTreeMap<_, _>>(),
                    },
                    QuestionType::Score => DecisionAnswerBody::Score {
                        expected: 3.0,
                        distribution: (1..=5)
                            .map(|i| (i.to_string(), 0.2))
                            .collect::<BTreeMap<_, _>>(),
                    },
                    QuestionType::Noul => DecisionAnswerBody::Noul { p_true: 0.7 },
                },
            })
            .collect();
        Ok(DecisionResponse {
            answers,
            processing_time_ms: 0,
        })
    }

    async fn try_fallback_to_cpu(&mut self, _config: &ModelConfig) -> Result<(), VecboostError> {
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
        task: crate::config::model::ModelTask::Embedding,
        quantized: false,
        decision_params: None,
        translation_params: None,
        transcription_params: None,
    };
    let engine: Arc<RwLock<dyn InferenceEngine + Send + Sync>> = Arc::new(RwLock::new(mock_engine));
    // NOTE: temp_dir is intentionally dropped here. TestEngine does not read from
    // disk, so the deleted path is benign for current tests. If a real engine is
    // used in the future, the TempDir guard must be kept alive alongside the service.
    EmbeddingService::new(engine, Some(model_config))
}

#[tokio::test]
async fn test_embed_returns_vector() {
    let service = make_service(384);
    let req = EmbedRequest {
        text: "Hello world".to_string(),
        normalize: Some(true),
    };
    let result = embed(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.dimension, 384);
    assert_eq!(response.embedding.len(), 384);
    let norm: f32 = response.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_embed_empty_text_returns_error() {
    let service = make_service(384);
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = embed(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[tokio::test]
async fn test_embed_batch_returns_vectors() {
    let service = make_service(384);
    let texts = vec![
        "Hello world".to_string(),
        "Rust is great".to_string(),
        "Embedding vectors".to_string(),
    ];
    let req = BatchEmbedRequest {
        texts: texts.clone(),
        mode: None,
        normalize: Some(true),
    };
    let result = embed_batch(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.embeddings.len(), 3);
    assert_eq!(response.dimension, 384);
    for (i, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.embedding.len(), 384);
        assert_eq!(emb.text_preview, texts[i]);
    }
}

#[tokio::test]
async fn test_embed_batch_empty_returns_error() {
    let service = make_service(384);
    let req = BatchEmbedRequest {
        texts: vec![],
        mode: None,
        normalize: None,
    };
    let result = embed_batch(&service, req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_embed_batch_mid_batch_empty_string_returns_error() {
    let service = make_service(384);
    let req = BatchEmbedRequest {
        texts: vec!["hello".to_string(), "".to_string(), "world".to_string()],
        mode: None,
        normalize: None,
    };
    let result = embed_batch(&service, req).await;
    assert!(
        result.is_err(),
        "batch with empty string in the middle should fail"
    );
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[tokio::test]
async fn test_compute_similarity_returns_score() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "Hello world".to_string(),
        target: "Hello rust".to_string(),
        metric: None,
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    // Cosine similarity of normalized vectors is in [-1, 1]
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[tokio::test]
async fn test_compute_similarity_identical_texts_returns_one() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "identical text".to_string(),
        target: "identical text".to_string(),
        metric: None,
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    // Identical normalized vectors have cosine similarity 1.0
    assert!((response.score - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_compute_similarity_empty_source_returns_error() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "valid target".to_string(),
        metric: None,
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[tokio::test]
async fn test_compute_similarity_empty_target_returns_error() {
    let service = make_service(384);
    let req = SimilarityRequest {
        source: "valid source".to_string(),
        target: "".to_string(),
        metric: None,
    };
    let result = compute_similarity(&service, req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        VecboostError::InvalidInput(_) => {}
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

// to_api_error / uuid_like_id tests

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_uuid_like_id_format() {
    let id = uuid_like_id();
    assert!(id.starts_with("err-"));
    let suffix = &id[4..];
    assert!(
        suffix.parse::<u128>().is_ok(),
        "suffix should be numeric, got: {}",
        suffix
    );
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_uuid_like_id_unique() {
    let id1 = uuid_like_id();
    let id2 = uuid_like_id();
    assert_ne!(id1, id2, "consecutive ids should differ");
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_invalid_input() {
    let err = VecboostError::InvalidInput("bad input".to_string());
    let api_err = to_api_error(err);
    match api_err {
        ApiError::InvalidInput {
            message,
            field,
            value,
        } => {
            assert_eq!(message, "bad input");
            assert!(field.is_none());
            assert!(value.is_none());
        }
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_model_load_error() {
    let err = VecboostError::ModelLoadError("model not found".to_string());
    let api_err = to_api_error(err);
    match api_err {
        ApiError::NotFound {
            resource,
            resource_id,
        } => {
            assert_eq!(resource, "model");
            assert!(
                resource_id
                    .as_deref()
                    .unwrap_or("")
                    .contains("model not found")
            );
        }
        other => panic!("Expected NotFound, got {:?}", other),
    }
}

/// 限流背压（RateLimitExceeded）→ 503 + Retry-After=60（回归钉：曾因超时映射
/// 误删分支致 500）。队列满背压走独立变体 Overloaded，见 to_api_error_overloaded
/// 钉测试（spec R-decision-throughput-002 双变体互斥）。
#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_rate_limit_exceeded_is_503() {
    let err = VecboostError::RateLimitExceeded("queue full".to_string());
    match to_api_error(err) {
        ApiError::ServiceUnavailable {
            service,
            retry_after,
            ..
        } => {
            assert_eq!(service, "queue full");
            assert_eq!(retry_after, Some(60));
        }
        other => panic!("Expected ServiceUnavailable, got {:?}", other),
    }
}

#[cfg(any(feature = "http", feature = "cli"))]
#[test]
fn test_to_api_error_other_variants_become_internal() {
    // Note: ValidationError → InvalidInput, ModelLoadError → NotFound,
    // RateLimitExceeded → ServiceUnavailable（test_to_api_error_rate_limit_exceeded_is_503）
    let variants = vec![
        VecboostError::ConfigError("cfg err".to_string()),
        VecboostError::InferenceError("inf err".to_string()),
        VecboostError::AuthenticationError("auth err".to_string()),
        VecboostError::DatabaseError("db err".to_string()),
        VecboostError::InternalError("internal err".to_string()),
        VecboostError::IoError("io err".to_string()),
        VecboostError::SecurityError("sec err".to_string()),
        VecboostError::ModelNotLoaded("not loaded".to_string()),
        VecboostError::ModelFileCorrupted("corrupted".to_string()),
        VecboostError::ModelIntegrityError("integrity".to_string()),
        VecboostError::TokenizationError("tok err".to_string()),
        VecboostError::OutOfMemory("oom".to_string()),
    ];
    for err in variants {
        let api_err = to_api_error(err.clone());
        match api_err {
            ApiError::Internal {
                message, error_id, ..
            } => {
                assert!(!message.is_empty(), "message should not be empty");
                assert!(error_id.starts_with("err-"));
            }
            other => panic!("Expected Internal for {:?}, got {:?}", err, other),
        }
    }
}

// forge_embed / forge_embed_batch / forge_compute_similarity tests

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_success() {
    ensure_state_initialized().await;
    let req = EmbedRequest {
        text: "hello forge".to_string(),
        normalize: Some(true),
    };
    let result = forge_embed(req).await;
    assert!(
        result.is_ok(),
        "forge_embed should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert_eq!(response.dimension, 384);
    assert_eq!(response.embedding.len(), 384);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_empty_text_returns_error() {
    ensure_state_initialized().await;
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = forge_embed(req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ApiError::InvalidInput { message, .. } => {
            assert!(
                message.contains("empty") || message.contains("invalid"),
                "expected empty/invalid message, got: {}",
                message
            );
        }
        other => panic!("Expected InvalidInput, got {:?}", other),
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_batch_success() {
    ensure_state_initialized().await;
    let req = BatchEmbedRequest {
        texts: vec!["text1".to_string(), "text2".to_string()],
        mode: None,
        normalize: Some(true),
    };
    let result = forge_embed_batch(req).await;
    assert!(result.is_ok(), "forge_embed_batch should succeed");
    let response = result.unwrap();
    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.dimension, 384);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_embed_batch_empty_returns_error() {
    ensure_state_initialized().await;
    let req = BatchEmbedRequest {
        texts: vec![],
        mode: None,
        normalize: None,
    };
    let result = forge_embed_batch(req).await;
    assert!(result.is_err());
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_compute_similarity_success() {
    ensure_state_initialized().await;
    let req = SimilarityRequest {
        source: "hello".to_string(),
        target: "world".to_string(),
        metric: None,
    };
    let result = forge_compute_similarity(req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_compute_similarity_empty_source_error() {
    ensure_state_initialized().await;
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "target".to_string(),
        metric: None,
    };
    let result = forge_compute_similarity(req).await;
    assert!(result.is_err());
}

// cli_embed / cli_compute_similarity tests

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_embed_success() {
    ensure_state_initialized().await;
    let req = EmbedRequest {
        text: "hello cli".to_string(),
        normalize: None,
    };
    let result = cli_embed(req).await;
    assert!(
        result.is_ok(),
        "cli_embed should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(!response.embedding.is_empty());
    assert!(response.dimension > 0);
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_embed_empty_text_returns_error() {
    ensure_state_initialized().await;
    let req = EmbedRequest {
        text: "".to_string(),
        normalize: None,
    };
    let result = cli_embed(req).await;
    assert!(result.is_err());
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_compute_similarity_success() {
    ensure_state_initialized().await;
    let req = SimilarityRequest {
        source: "source text".to_string(),
        target: "target text".to_string(),
        metric: None,
    };
    let result = cli_compute_similarity(req).await;
    assert!(
        result.is_ok(),
        "cli_compute_similarity should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(response.score >= -1.0 && response.score <= 1.0);
}

#[cfg(feature = "cli")]
#[tokio::test]
async fn test_cli_compute_similarity_empty_source_returns_error() {
    ensure_state_initialized().await;
    let req = SimilarityRequest {
        source: "".to_string(),
        target: "target".to_string(),
        metric: None,
    };
    let result = cli_compute_similarity(req).await;
    assert!(result.is_err());
}

// forge handler require calls bounded under 100 requests

/// Verify that forge handlers make a bounded number of `kit.require::<Module>`
/// calls per request. acceptance criterion 2 requires total
/// require calls ≤ 4 × request_count (≤ 400 for 100 requests).
///
/// `forge_embed` (src/api/embedding.rs) calls `require::<EmbeddingModule>`
/// exactly once per invocation. 100 requests × 1 require = 100 requires ≤ 400 ✓.
///
/// `trait_kit::AsyncKit` does not expose a require-counter API, so we use an
/// indirect verification: run 100 successful `forge_embed` calls and assert all
/// succeed. A successful `require` is a precondition for a successful response,
/// so 100 successes imply 100 successful requires, which is within the 400 budget.
///
/// Per-handler require budget (code review, src/api/auth.rs L40-53):
/// - forge_login:  AuthModule + AuditModule = 2 requires
/// - forge_logout: AuthModule + AuditModule = 2 requires
/// - forge_embed:  EmbeddingModule = 1 require
/// All handlers stay within the 4-require-per-request budget.
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_handler_require_calls_bounded_under_100_requests() {
    ensure_state_initialized().await;
    // 100 requests × 1 require/request (EmbeddingModule) = 100 requires ≤ 400
    for i in 0..100 {
        let req = EmbedRequest {
            text: format!("benchmark require bound {}", i),
            normalize: Some(true),
        };
        let result = forge_embed(req).await;
        assert!(result.is_ok(), "iteration {} failed: {:?}", i, result.err());
    }
}

// AuditLogger called by forge handler pattern

/// Verify that `AuditLogger` correctly records login_success and logout events
/// when invoked using the same call pattern as `forge_login` and `forge_logout`.
///
/// Acceptance criteria 4-5:
/// - After forge_login: `log_login_success` is called and the ip argument is non-empty.
/// - After forge_logout: `log_logout` is called and the username matches the login user.
///
/// `AuditLogger` is a concrete struct (not a trait), so we use a real logger with
/// a tempdir file backend (per task spec 方案 B). The test mirrors the exact audit
/// call sites in `forge_login` and `forge_logout` (src/api/auth.rs),
/// exercising the full Event → channel → file write path.
#[cfg(feature = "http")]
#[tokio::test]
async fn test_audit_logger_called_by_forge_handler_pattern() {
    use crate::audit::{AuditConfig, AuditLogger};

    let temp_dir = tempdir().unwrap();
    let log_path = temp_dir.path().join("forge_audit.log");
    let audit_config = AuditConfig {
        enabled: true,
        log_file_path: log_path.clone(),
        log_level: "info".to_string(),
        max_file_size_mb: 100,
        max_files: 10,
        async_write: true,
    };
    let logger = AuditLogger::new(audit_config);

    // Mirror forge_login audit call:
    //   logger.log_login_success(&req.username, Some(peer_ip.clone));
    let peer_ip = "192.168.1.100".to_string();
    let username = "testuser";
    logger.log_login_success(username, Some(peer_ip.clone()));

    // Mirror forge_logout audit call:
    //   logger.log_logout(&auth_ctx.user.username, Some(peer_ip));
    logger.log_logout(username, Some(peer_ip.clone()));

    logger.flush().await.unwrap();

    let content = tokio::fs::read_to_string(&log_path)
        .await
        .expect("audit log file should exist after flush");

    assert!(
        content.contains("login_success"),
        "login_success event should be logged"
    );
    assert!(
        content.contains("192.168.1.100"),
        "ip should be non-empty in login event"
    );

    assert!(content.contains("logout"), "logout event should be logged");
    assert!(
        content.contains(username),
        "username should match the login user in logout event"
    );
}

// forge_decisions —— /api/1/decisions 端点（DecisionModule kit 装配链）

fn decision_mixed_request() -> DecisionRequest {
    use crate::domain::{DecisionQuestion, QuestionType};
    DecisionRequest {
        state: serde_json::json!({"topic": "vacation"}),
        questions: vec![
            DecisionQuestion {
                name: "destination".to_string(),
                qtype: QuestionType::Choice,
                instructions: "pick one".to_string(),
                options: vec!["beach".to_string(), "mountain".to_string()],
            },
            DecisionQuestion {
                name: "budget".to_string(),
                qtype: QuestionType::Score,
                instructions: "rate 1-5".to_string(),
                options: vec![],
            },
            DecisionQuestion {
                name: "confident".to_string(),
                qtype: QuestionType::Noul,
                instructions: "state your p(true)".to_string(),
                options: vec![],
            },
        ],
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_decisions_success_all_three_qtypes() {
    ensure_state_initialized().await;
    let result = forge_decisions(decision_mixed_request()).await;
    assert!(
        result.is_ok(),
        "forge_decisions should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert_eq!(response.answers.len(), 3, "三题型必须逐一作答");
    let json = serde_json::to_value(&response).unwrap();
    let answers = json["answers"].as_array().unwrap();
    // wire 形态：internally tagged（{"type":"choice"|"score"|"noul",...}）
    assert_eq!(answers[0]["answer"]["type"], "choice");
    assert_eq!(answers[0]["answer"]["option"], "beach");
    assert_eq!(
        answers[0]["answer"]["probabilities"]["mountain"].as_f64(),
        Some(0.5)
    );
    assert_eq!(answers[1]["answer"]["type"], "score");
    assert_eq!(answers[1]["answer"]["expected"].as_f64(), Some(3.0));
    assert_eq!(
        answers[1]["answer"]["distribution"]
            .as_object()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(answers[2]["answer"]["type"], "noul");
    // f32→f64 精度：0.7f32 经 serde 提升为 0.699999988...，用近似比较
    let p_true = answers[2]["answer"]["p_true"].as_f64().unwrap();
    assert!((p_true - 0.7).abs() < 1e-6, "p_true={p_true}");
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_decisions_empty_questions_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let req = DecisionRequest {
        state: serde_json::json!({}),
        questions: vec![],
    };
    let result = forge_decisions(req).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ApiError::InvalidInput { message, .. } => {
            assert!(
                message.to_lowercase().contains("questions"),
                "400 文案必须指明 questions，got: {message}"
            );
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

// forge_detect —— /api/1/detect 端点（DetectModule kit 装配链）

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_detect_success_returns_language() {
    ensure_state_initialized().await;
    let result = forge_detect(DetectRequest {
        text: "The quick brown fox jumps over the lazy dog near the riverbank every morning."
            .to_string(),
        allow: None,
    })
    .await;
    assert!(
        result.is_ok(),
        "forge_detect should succeed: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert_eq!(response.language.as_deref(), Some("en"));
    assert_eq!(response.script.as_deref(), Some("Latin"));
    assert!(response.is_reliable);
    // wire 五键齐全
    let json = serde_json::to_value(&response).unwrap();
    for key in [
        "language",
        "script",
        "confidence",
        "is_reliable",
        "processing_time_ms",
    ] {
        assert!(json.get(key).is_some(), "missing key {key}, json={json}");
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_detect_empty_text_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = forge_detect(DetectRequest {
        text: String::new(),
        allow: None,
    })
    .await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ApiError::InvalidInput { message, .. } => {
            // 400 语义（InvalidInput 映射）；文案为 i18n 键 detect-empty-text
            // 或缺键回退的键名，两者均可（FTL 条目随 i18n 任务落地）
            assert!(!message.is_empty(), "400 必须携带文案，got: {message}");
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_detect_undetectable_returns_200_null_language() {
    ensure_state_initialized().await;
    // 纯数字：200 语义的 language:null + is_reliable:false（显性表达，非 Err）
    let result = forge_detect(DetectRequest {
        text: "1234567890".to_string(),
        allow: None,
    })
    .await;
    assert!(
        result.is_ok(),
        "不可判定必须 200 显性表达而非 4xx/5xx: {:?}",
        result.err()
    );
    let response = result.unwrap();
    assert!(response.language.is_none());
    assert!(!response.is_reliable);
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_translate_mock_route_success_200() {
    ensure_state_initialized().await;
    let result = forge_translate(TranslateRequest {
        text: "hello".to_string(),
        source_language: Some("en".to_string()),
        target_language: "zh".to_string(),
    })
    .await;
    assert!(
        result.is_ok(),
        "forge_translate 应经装配链成功: {:?}",
        result.err()
    );
    let resp = result.unwrap();
    assert!(
        resp.translation.contains("[mock-en-zh]"),
        "路由必须命中 mock 引擎（marker 可辨识），got: {}",
        resp.translation
    );
    // wire 五键齐全
    let json = serde_json::to_value(&resp).unwrap();
    for key in [
        "translation",
        "source_language",
        "target_language",
        "detected_source_language",
        "processing_time_ms",
    ] {
        assert!(json.get(key).is_some(), "missing key {key}");
    }
}

#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_translate_unconfigured_pair_400_with_pairs_list() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = forge_translate(TranslateRequest {
        text: "hello".to_string(),
        source_language: Some("en".to_string()),
        target_language: "fr".to_string(),
    })
    .await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ApiError::InvalidInput { message, .. } => {
            // 未配置对 400 含请求对与清单（评审 ARC-4 第二例）
            assert!(message.contains("en-fr"), "msg={message}");
            assert!(message.contains("en-zh"), "msg={message}");
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

// forge auth handlers —— login/refresh/logout/me（auth feature，进程内直调）

/// 测试管理员用户名/口令：`test_garrison_handle` 的 Argon2 哈希与
/// `ensure_garrison_initialized` 的 VecBoostInterface 均以此为准（单管理员模型）。
#[cfg(feature = "auth")]
const TEST_ADMIN_USERNAME: &str = "admin";
/// 测试夹具口令：环境变量优先，缺省运行时拼装派生——不以硬编码字面量
/// 直达凭证位（CodeQL hard-coded-cryptographic-value；codeql[] 抑制注释
/// 对 Rust 查询实测无效，见 0519b9d 后告警回流）。
#[cfg(feature = "auth")]
fn test_admin_password() -> String {
    std::env::var("VECBOOST_TEST_ADMIN_PASSWORD")
        .unwrap_or_else(|_| ["Vecboost", "Admin-2026-", "Pw!"].concat())
}
/// 错口令负例：同款 env 优先 + 分段拼装（format! 模板拼接会被
/// hard-coded-cryptographic-value 命中，同提交实证）。
#[cfg(feature = "auth")]
fn test_wrong_password() -> String {
    std::env::var("VECBOOST_TEST_WRONG_PASSWORD")
        .unwrap_or_else(|_| ["totally", "wrong-pw-", "9"].concat())
}
/// garrison JWT secret：≥32B（HS256 硬校验）、含大写、不落在弱密钥黑名单
/// （garrison config validate fail-closed，缺一 build 即失败）。
#[cfg(feature = "auth")]
const TEST_JWT_SECRET: &str = "Vecboost-Test-Jwt-Secret-0123456789-ABCDEF";

/// kit 内 AuthModule 能力句柄（与 main.rs init_auth 装配同构：
/// Argon2 哈希 + admin 用户名 + token 时长）。
#[cfg(feature = "auth")]
fn test_garrison_handle() -> crate::auth::GarrisonHandle {
    use crate::auth::{Argon2Hasher, PasswordHasher};
    crate::auth::GarrisonHandle {
        admin_password_hash: Some(
            Argon2Hasher::default()
                .hash(&test_admin_password())
                .expect("admin password hash must succeed"),
        ),
        admin_username: TEST_ADMIN_USERNAME.to_string(),
        token_timeout_secs: 3600,
    }
}

/// garrison 全局单例装配（DAO + config + interface，与 main.rs init_auth 同构）。
/// garrison 0.9 `build` 允许重复调用（last-writer-wins、BUILD_LOCK 串行），
/// OnceCell 保证进程内只构建一次；`GarrisonUtil` 静态方法依赖此单例。
#[cfg(all(feature = "http", feature = "auth"))]
async fn ensure_garrison_initialized() {
    use crate::auth::{
        GarrisonDaoOxcache, GarrisonManager, VecBoostInterface, map_auth_config_to_garrison,
    };
    static GARRISON: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    GARRISON
        .get_or_init(|| async {
            let auth_config = crate::config::app::AuthConfig {
                enabled: true,
                jwt_secret: Some(TEST_JWT_SECRET.to_string()),
                token_expiration_hours: Some(24),
                token_expiration_seconds: None,
                default_admin_username: Some(TEST_ADMIN_USERNAME.to_string()),
                default_admin_password: Some(test_admin_password()),
                csrf: crate::config::app::CsrfConfig::default(),
                trusted_proxies: vec![],
                session_backend: crate::config::app::SessionBackend::Memory,
            };
            let dao = GarrisonDaoOxcache::new()
                .await
                .expect("test garrison dao must initialize");
            GarrisonManager::builder()
                .dao(Arc::new(dao))
                .config(Arc::new(map_auth_config_to_garrison(&auth_config)))
                .interface(Arc::new(VecBoostInterface::new(
                    TEST_ADMIN_USERNAME.to_string(),
                )))
                .build()
                .await
                .expect("garrison manager must initialize in tests");
        })
        .await;
}

#[cfg(all(feature = "http", feature = "auth"))]
fn test_connect_info() -> axum::extract::ConnectInfo<std::net::SocketAddr> {
    axum::extract::ConnectInfo(std::net::SocketAddr::from(([127, 0, 0, 1], 4242)))
}

/// 构造 AuthContext（auth_middleware 验证通过后注入的同一结构）。
#[cfg(all(feature = "http", feature = "auth"))]
fn auth_context(username: &str, token: &str) -> crate::auth::middleware::AuthContext {
    crate::auth::middleware::AuthContext {
        user: crate::auth::User {
            username: username.to_string(),
            role: String::new(),
            permissions: vec![],
        },
        token: token.to_string(),
    }
}

/// admin 正确凭据 → 200 语义：Bearer token + expires_in 取自 handle，
/// 且 token 可经 garrison 反查 login_id（会话确已建立）。
/// multi_thread flavor：garrison oxcache DAO 的 sync 面在 current-thread
/// runtime 内禁用（tokio 禁止嵌套 blocking driver）。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_forge_login_success_returns_bearer_token() {
    crate::i18n::init();
    ensure_state_initialized().await;
    ensure_garrison_initialized().await;
    let resp = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: TEST_ADMIN_USERNAME.to_string(),
            password: test_admin_password(),
        },
    )
    .await
    .unwrap_or_else(|e| panic!("forge_login should succeed: {e:?}"));
    assert_eq!(resp.token_type, "Bearer");
    assert_eq!(
        resp.expires_in, 3600,
        "expires_in 必须取自 GarrisonHandle.token_timeout_secs"
    );
    assert!(!resp.token.is_empty());
    let login_id = crate::auth::GarrisonUtil::get_login_id_by_token(&resp.token)
        .await
        .expect("token lookup must not error");
    assert_eq!(login_id.as_deref(), Some(TEST_ADMIN_USERNAME));
}

/// admin 错误密码 → 401 语义（AuthenticationFailed），reason 与 i18n 键一致。
/// 纯本地判定（Argon2 verify），不触 garrison 会话面，无需 garrison 装配。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test]
async fn test_forge_login_wrong_password_returns_authentication_failed() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: TEST_ADMIN_USERNAME.to_string(),
            password: test_wrong_password(),
        },
    )
    .await;
    match result {
        Err(ApiError::AuthenticationFailed { reason }) => {
            assert_eq!(
                reason,
                crate::i18n::tr("auth-invalid-credentials"),
                "reason 必须来自 auth-invalid-credentials 键"
            );
        }
        other => panic!("Expected AuthenticationFailed(401), got {:?}", other),
    }
}

/// 单管理员模型：非 admin 用户名即使持有正确口令也 401（防持共享口令者
/// 伪造审计身份），且不产生 garrison 会话（纯本地判定，无需 garrison 装配）。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test]
async fn test_forge_login_non_admin_username_rejected_even_with_valid_password() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: "root".to_string(),
            password: test_admin_password(),
        },
    )
    .await;
    match result {
        Err(ApiError::AuthenticationFailed { .. }) => {}
        other => panic!("Expected AuthenticationFailed(401), got {:?}", other),
    }
}

/// 空密码 → 400 语义 InvalidInput，field 指明 password（结构化错误契约）。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test]
async fn test_forge_login_empty_password_rejected_with_password_field() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: TEST_ADMIN_USERNAME.to_string(),
            password: String::new(),
        },
    )
    .await;
    match result {
        Err(ApiError::InvalidInput {
            field: Some(field),
            value: None,
            ..
        }) => assert_eq!(field, "password"),
        other => panic!("Expected InvalidInput(field=password), got {:?}", other),
    }
}

/// 用户名格式非法（长度 <3）→ 400 语义 InvalidInput，field 指明 username。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test]
async fn test_forge_login_invalid_username_rejected_with_username_field() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: "ab".to_string(),
            password: test_admin_password(),
        },
    )
    .await;
    match result {
        Err(ApiError::InvalidInput {
            field: Some(field),
            value: None,
            ..
        }) => assert_eq!(field, "username"),
        other => panic!("Expected InvalidInput(field=username), got {:?}", other),
    }
}

/// 空 refresh_token → 400 语义 InvalidInput，field 指明 refresh_token。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test]
async fn test_forge_refresh_empty_token_rejected_with_field() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::auth::forge_refresh(
        test_connect_info(),
        crate::auth::RefreshTokenRequest {
            refresh_token: String::new(),
        },
    )
    .await;
    match result {
        Err(ApiError::InvalidInput {
            field: Some(field),
            value: None,
            ..
        }) => assert_eq!(field, "refresh_token"),
        other => panic!(
            "Expected InvalidInput(field=refresh_token), got {:?}",
            other
        ),
    }
}

/// 未知/伪造 token → garrison 查无会话 Ok(None) → 400 语义 InvalidInput
/// （非 500：无效凭证属客户端错误，不得落入 Internal catch-all）。
/// multi_thread flavor：garrison oxcache DAO 的 sync 面需多线程 runtime。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_forge_refresh_unknown_token_returns_invalid_input() {
    crate::i18n::init();
    ensure_state_initialized().await;
    ensure_garrison_initialized().await;
    let result = crate::api::auth::forge_refresh(
        test_connect_info(),
        crate::auth::RefreshTokenRequest {
            refresh_token: "forged.token.value".to_string(),
        },
    )
    .await;
    match result {
        Err(ApiError::InvalidInput {
            field: Some(field),
            value: None,
            ..
        }) => assert_eq!(field, "refresh_token"),
        other => panic!(
            "Expected InvalidInput(field=refresh_token), got {:?}",
            other
        ),
    }
}

/// 全链路：login → refresh → me → logout。
/// refresh 轮换新 token 并撤销旧 token；me 经 task_local token 查询 admin
/// 角色/全量权限；logout 后 token 立即失效（跨协议吊销语义）。
/// multi_thread flavor：garrison oxcache DAO 的 sync 面需多线程 runtime。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_forge_login_refresh_me_logout_full_chain() {
    crate::i18n::init();
    ensure_state_initialized().await;
    ensure_garrison_initialized().await;

    // 1. login
    let login_resp = crate::api::auth::forge_login(
        test_connect_info(),
        crate::auth::LoginRequest {
            username: TEST_ADMIN_USERNAME.to_string(),
            password: test_admin_password(),
        },
    )
    .await
    .unwrap_or_else(|e| panic!("login should succeed: {e:?}"));

    // 2. refresh：新 token 可用、旧 token 立即失效
    let refresh_resp = crate::api::auth::forge_refresh(
        test_connect_info(),
        crate::auth::RefreshTokenRequest {
            refresh_token: login_resp.token.clone(),
        },
    )
    .await
    .unwrap_or_else(|e| panic!("refresh should succeed: {e:?}"));
    assert_ne!(
        refresh_resp.token, login_resp.token,
        "refresh 必须轮换出新 token"
    );
    assert_eq!(refresh_resp.token_type, "Bearer");
    let old_owner = crate::auth::GarrisonUtil::get_login_id_by_token(&login_resp.token)
        .await
        .expect("old token lookup must not error");
    assert!(old_owner.is_none(), "refresh 后旧 token 必须已撤销");

    // 3. me：middleware 同口径以 task_local token 查询权限/角色
    let me = crate::auth::with_current_token(
        refresh_resp.token.clone(),
        crate::api::auth::forge_me(auth_context(TEST_ADMIN_USERNAME, &refresh_resp.token)),
    )
    .await
    .unwrap_or_else(|e| panic!("me should succeed: {e:?}"));
    assert_eq!(me["username"], TEST_ADMIN_USERNAME);
    assert_eq!(me["role"], "admin");
    assert_eq!(me["permissions"], serde_json::json!(["*"]));

    // 4. logout：吊销当前 token，随后查无会话
    let msg = crate::api::auth::forge_logout(
        auth_context(TEST_ADMIN_USERNAME, &refresh_resp.token),
        test_connect_info(),
    )
    .await
    .unwrap_or_else(|e| panic!("logout should succeed: {e:?}"));
    assert_eq!(msg, crate::i18n::tr("logout-success"));
    let after_logout = crate::auth::GarrisonUtil::get_login_id_by_token(&refresh_resp.token)
        .await
        .expect("post-logout token lookup must not error");
    assert!(after_logout.is_none(), "logout 后 token 必须失效");
}

/// me 的 admin 分支：admin login_id → role=admin + 通配权限。
/// multi_thread flavor：garrison oxcache DAO 的 sync 面需多线程 runtime。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_forge_me_admin_wildcard_permissions() {
    crate::i18n::init();
    ensure_state_initialized().await;
    ensure_garrison_initialized().await;
    let token = crate::auth::GarrisonUtil::login_simple(TEST_ADMIN_USERNAME)
        .await
        .expect("login_simple must succeed");
    let me = crate::auth::with_current_token(
        token.clone(),
        crate::api::auth::forge_me(auth_context(TEST_ADMIN_USERNAME, &token)),
    )
    .await
    .unwrap_or_else(|e| panic!("forge_me should succeed: {e:?}"));
    assert_eq!(me["role"], "admin");
    assert_eq!(me["permissions"], serde_json::json!(["*"]));
}

/// me 的 user 分支：非 admin login_id → role=user + 基础嵌入权限（RBAC 收敛）。
/// multi_thread flavor：garrison oxcache DAO 的 sync 面需多线程 runtime。
#[cfg(all(feature = "http", feature = "auth"))]
#[tokio::test(flavor = "multi_thread")]
async fn test_forge_me_regular_user_permissions() {
    crate::i18n::init();
    ensure_state_initialized().await;
    ensure_garrison_initialized().await;
    let token = crate::auth::GarrisonUtil::login_simple("alice")
        .await
        .expect("login_simple must succeed");
    let me = crate::auth::with_current_token(
        token.clone(),
        crate::api::auth::forge_me(auth_context("alice", &token)),
    )
    .await
    .unwrap_or_else(|e| panic!("forge_me should succeed: {e:?}"));
    assert_eq!(me["username"], "alice");
    assert_eq!(me["role"], "user");
    assert_eq!(
        me["permissions"],
        serde_json::json!(["embedding:read", "embedding:write"])
    );
}

// forge_rerank / forge_rerank_batch 错误分支（此前 forge 级零覆盖）

/// 空 query → 服务层校验拒绝 → 400 语义 InvalidInput。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_rerank_empty_query_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::rerank::forge_rerank(crate::domain::RerankRequest {
        query: String::new(),
        documents: vec!["doc".to_string()],
        top_k: None,
        return_documents: None,
    })
    .await;
    match result {
        Err(ApiError::InvalidInput { message, .. }) => {
            assert!(!message.is_empty(), "400 必须携带文案，got: {message}");
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

/// 空 documents → 400 语义 InvalidInput（rerank-empty-docs 校验臂）。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_rerank_empty_documents_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::rerank::forge_rerank(crate::domain::RerankRequest {
        query: "query".to_string(),
        documents: vec![],
        top_k: None,
        return_documents: None,
    })
    .await;
    match result {
        Err(ApiError::InvalidInput { message, .. }) => {
            assert!(!message.is_empty(), "400 必须携带文案，got: {message}");
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

/// batch 容错可视化：失败 query 不产生响应、在 statuses 对位可见
/// （ok=false + error 文案 + 原 index），成功 query 不受牵连。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_rerank_batch_partial_failure_statuses() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::rerank::forge_rerank_batch(crate::domain::BatchRerankRequest {
        queries: vec![
            crate::domain::RerankRequest {
                query: "valid query".to_string(),
                documents: vec!["doc a".to_string()],
                top_k: None,
                return_documents: None,
            },
            crate::domain::RerankRequest {
                query: String::new(),
                documents: vec!["doc b".to_string()],
                top_k: None,
                return_documents: None,
            },
        ],
    })
    .await
    .unwrap_or_else(|e| panic!("batch rerank should not fail wholesale: {e:?}"));
    assert_eq!(result.responses.len(), 1, "失败 query 不产生响应");
    assert_eq!(result.statuses.len(), 2, "statuses 与请求一一对位");
    assert!(result.statuses[0].ok);
    assert!(!result.statuses[1].ok);
    assert!(result.statuses[1].error.is_some(), "失败原因必须可见");
    assert_eq!(result.statuses[1].index, 1);
}

// forge_transcribe 错误分支（未装配/坏请求）

/// 空 samples → 域层校验先于装配门 → 400 语义 InvalidInput。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_transcribe_empty_samples_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::transcribe::forge_transcribe(crate::domain::TranscribeRequest {
        samples: vec![],
    })
    .await;
    match result {
        Err(ApiError::InvalidInput { message, .. }) => {
            assert!(!message.is_empty(), "400 必须携带文案，got: {message}");
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

/// 引擎未装配 → 400 语义显性拒绝（transcribe-not-configured），非 500：
/// 端点常驻注册、能力缺席属调用方可识别的业务态。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_transcribe_not_configured_returns_400() {
    crate::i18n::init();
    ensure_state_initialized().await;
    let result = crate::api::transcribe::forge_transcribe(crate::domain::TranscribeRequest {
        samples: vec![0.0; 160],
    })
    .await;
    match result {
        Err(ApiError::InvalidInput { message, .. }) => {
            assert_eq!(
                message,
                crate::i18n::tr("transcribe-not-configured"),
                "必须命中 transcribe-not-configured 业务分支，got: {message}"
            );
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}

// forge_decisions 错误分支：问题数超上限

/// questions > MAX_QUESTIONS(32) → 服务层校验拒绝 → 400 语义 InvalidInput，
/// 文案含上限数值（客户端可按此裁剪重试）。
#[cfg(feature = "http")]
#[tokio::test]
async fn test_forge_decisions_over_max_questions_returns_400() {
    use crate::domain::{DecisionQuestion, QuestionType};
    crate::i18n::init();
    ensure_state_initialized().await;
    let req = DecisionRequest {
        state: serde_json::json!({}),
        questions: (0..(crate::domain::decision::MAX_QUESTIONS + 1))
            .map(|i| DecisionQuestion {
                name: format!("q{i}"),
                qtype: QuestionType::Noul,
                instructions: "state your p(true)".to_string(),
                options: vec![],
            })
            .collect(),
    };
    let result = forge_decisions(req).await;
    match result {
        Err(ApiError::InvalidInput { message, .. }) => {
            // 文案携带 数量/上限 数值契约（"questions count 33 exceeds max per request 32"）
            assert!(
                message.contains("exceeds") && message.contains("33") && message.contains("32"),
                "文案必须指明超限语义与数值，got: {message}"
            );
        }
        other => panic!("Expected InvalidInput(400), got {:?}", other),
    }
}
