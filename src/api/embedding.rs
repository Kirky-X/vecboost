// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Embedding forge handlers — HTTP/MCP/CLI/gRPC protocol-agnostic.
//!
//! All handlers access `EmbeddingService` via
//! `state()?.kit.require::<EmbeddingModule>()` (returns `Arc<RwLock<EmbeddingService>>`).
//!
//! # Architecture
//!
//! Protocol-specific `forge_*` / `cli_*` / `grpc_*` handlers are thin wrappers
//! that only attach `#[forge(...)]` macros; the actual business logic lives in
//! protocol-agnostic `*_handler` functions below. This eliminates ~96 lines of
//! duplicated state-acquire/validate/dispatch code across the three protocols.

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
#[cfg(feature = "http")]
use crate::domain::openai_embedding::{
    EmbeddingData, EmbeddingObject, OpenAIEmbedRequest, OpenAIEmbedResponse, Usage,
};
use crate::domain::{
    BatchEmbedRequest, BatchEmbedResponse, EmbedRequest, EmbedResponse, SearchRequest,
    SearchResponse, SimilarityRequest, SimilarityResponse,
};
#[cfg(any(feature = "http", feature = "grpc"))]
use crate::domain::{
    EmbeddingOutput, FileEmbedRequest, FileEmbedResponse, ModelInfo, ModelListResponse,
    ModelMetadata, ModelSwitchRequest, ModelSwitchResponse, UnloadModelRequest,
    UnloadModelResponse,
};
use crate::error::VecboostError;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::registry::EmbeddingModule;
#[cfg(any(feature = "http", feature = "grpc"))]
use crate::registry::{CacheModule, RateLimitModule, RerankModule};
#[cfg(any(feature = "http", feature = "grpc"))]
use crate::utils::{AggregationMode, PathValidator};
#[cfg(any(feature = "http", feature = "grpc"))]
use std::path::PathBuf;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions — used by examples and external callers
// =============================================================================

pub async fn embed(
    svc: &crate::service::embedding::EmbeddingService,
    req: EmbedRequest,
) -> Result<EmbedResponse, VecboostError> {
    svc.process_text(req, None).await
}

pub async fn embed_batch(
    svc: &crate::service::embedding::EmbeddingService,
    req: BatchEmbedRequest,
) -> Result<BatchEmbedResponse, VecboostError> {
    svc.process_batch(req, None).await
}

pub async fn compute_similarity(
    svc: &crate::service::embedding::EmbeddingService,
    req: SimilarityRequest,
) -> Result<SimilarityResponse, VecboostError> {
    svc.process_similarity(req).await
}

/// 1对N 语义检索：给定查询文本，在候选文本列表中按相似度排序
pub async fn search(
    svc: &crate::service::embedding::EmbeddingService,
    req: SearchRequest,
) -> Result<SearchResponse, VecboostError> {
    svc.process_search(req).await
}

// =============================================================================
// Error conversion helpers
// =============================================================================

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn to_api_error(e: VecboostError) -> ApiError {
    match e {
        VecboostError::InvalidInput(msg) => ApiError::InvalidInput {
            message: msg,
            field: None,
            value: None,
        },
        VecboostError::ValidationError(msg) => ApiError::InvalidInput {
            message: msg,
            field: None,
            value: None,
        },
        VecboostError::ModelLoadError(msg) => ApiError::NotFound {
            resource: "model".to_string(),
            resource_id: Some(msg),
        },
        VecboostError::NotFound(msg) => ApiError::NotFound {
            resource: "resource".to_string(),
            resource_id: Some(msg),
        },
        VecboostError::RateLimitExceeded(msg) => ApiError::ServiceUnavailable {
            service: msg,
            retry_after: Some(60),
            source: None,
        },
        other => ApiError::Internal {
            message: other.error_detail().to_string(),
            error_id: uuid_like_id(),
            source: None,
            context: None,
        },
    }
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("err-{}", nanos)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
pub(crate) fn kit_internal_error(e: impl std::fmt::Display) -> ApiError {
    ApiError::Internal {
        message: e.to_string(),
        error_id: uuid_like_id(),
        source: None,
        context: None,
    }
}

// =============================================================================
// Validation helpers
// =============================================================================

/// Validate that no text exceeds the maximum allowed byte length.
///
/// Returns `ValidationError` on the first offending text, including index,
/// limit, and actual length for diagnostics. Byte length (`str::len`) is used
/// to match tokenizer input boundaries and prevent resource exhaustion.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn validate_text_length(texts: &[String], max: usize) -> Result<(), VecboostError> {
    for (idx, text) in texts.iter().enumerate() {
        if text.len() > max {
            return Err(VecboostError::ValidationError(crate::i18n::tr_with_args(
                "validate-text-length",
                crate::i18n::tr_args(&[
                    ("index", &idx.to_string()),
                    ("max", &max.to_string()),
                    ("got", &text.len().to_string()),
                ]),
            )));
        }
    }
    Ok(())
}

/// Validate that batch size does not exceed the configured maximum, preventing
/// resource exhaustion via oversized batch requests.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn validate_batch_size(texts_len: usize, max: usize) -> Result<(), VecboostError> {
    if texts_len > max {
        return Err(VecboostError::ValidationError(crate::i18n::tr_with_args(
            "validate-batch-size",
            crate::i18n::tr_args(&[("size", &texts_len.to_string()), ("max", &max.to_string())]),
        )));
    }
    Ok(())
}

/// Retrieve `max_text_length` from kit config, falling back to the default
/// (`EmbeddingConfig::default().max_text_length` = 8192) when config is absent.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn max_text_length_from_kit(kit: &trait_kit::AsyncKit<trait_kit::AsyncReady>) -> usize {
    kit.config::<crate::config::app::EmbeddingConfig>()
        .unwrap_or_default()
        .max_text_length
}

/// Retrieve `max_batch_size` from kit config, falling back to the default
/// (`EmbeddingConfig::default().max_batch_size` = 64) when config is absent.
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
fn max_batch_size_from_kit(kit: &trait_kit::AsyncKit<trait_kit::AsyncReady>) -> usize {
    kit.config::<crate::config::app::EmbeddingConfig>()
        .unwrap_or_default()
        .max_batch_size
}

/// Build a `PathValidator` from `[server] grpc_allowed_roots` config.
///
/// 允许根必须显式配置。移除旧的 cwd 隐式回退 —— 该回退使 `/embed/file` 成为
/// 以进程工作目录为根的任意文件读取原语(含 config/、源码、日志)。未配置时
/// 返回 400,错误信息指明配置项。
#[cfg(any(feature = "http", feature = "grpc"))]
fn build_path_validator() -> Result<PathValidator, ApiError> {
    const CONFIG_HINT: &str = "[server] grpc_allowed_roots is required for /embed/file; add explicit allowed roots          to config and restart";

    let st = state().map_err(to_api_error)?;
    let server_cfg = st
        .kit
        .config::<crate::config::app::ServerConfig>()
        .unwrap_or_default();

    let Some(roots) = &server_cfg.grpc_allowed_roots else {
        return Err(ApiError::InvalidInput {
            message: CONFIG_HINT.to_string(),
            field: Some("path".to_string()),
            value: None,
        });
    };
    if roots.is_empty() {
        return Err(ApiError::InvalidInput {
            message: CONFIG_HINT.to_string(),
            field: Some("path".to_string()),
            value: None,
        });
    }

    let mut validator = PathValidator::new();
    for root in roots {
        validator = validator.add_allowed_root(root);
    }
    Ok(validator)
}

/// `/embed/file` 单文件大小上限:10 MiB（防大文件 DoS 与内容外泄放大）。
#[cfg(any(feature = "http", feature = "grpc"))]
const FILE_EMBED_MAX_BYTES: u64 = 10 * 1024 * 1024;

/// 校验 /embed/file 目标文件大小,超限返回错误文案(纯函数,可单测)。
#[cfg(any(feature = "http", feature = "grpc"))]
fn check_file_embed_size(len: u64) -> Result<(), String> {
    if len > FILE_EMBED_MAX_BYTES {
        Err(format!(
            "File exceeds the {} MiB limit for /embed/file (got {} bytes)",
            FILE_EMBED_MAX_BYTES / (1024 * 1024),
            len
        ))
    } else {
        Ok(())
    }
}

// =============================================================================
// Protocol-agnostic business-logic handlers
//
// Each `*_handler` performs state acquisition, input validation, and service
// dispatch. Protocol-specific `forge_*` / `cli_*` / `grpc_*` functions below
// delegate to these helpers, keeping each protocol's surface to a single line
// of delegation plus the `#[forge(...)]` macro registration.
// =============================================================================

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn embed_handler(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    validate_text_length(
        std::slice::from_ref(&req.text),
        max_text_length_from_kit(&st.kit),
    )
    .map_err(to_api_error)?;

    // Pipeline 启用时，通过流水线处理请求
    #[cfg(feature = "http")]
    {
        let pipeline_enabled = st
            .kit
            .config::<crate::registry::PipelineEnabled>()
            .map(|c| c.0)
            .unwrap_or(false);
        if pipeline_enabled {
            let result =
                crate::pipeline::handle_pipeline_request(st.clone(), req, "api".to_string())
                    .await
                    .map_err(to_api_error)?;
            return Ok(result.0);
        }
    }

    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn embed_batch_handler(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    validate_batch_size(req.texts.len(), max_batch_size_from_kit(&st.kit)).map_err(to_api_error)?;
    validate_text_length(&req.texts, max_text_length_from_kit(&st.kit)).map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    embed_batch(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn compute_similarity_handler(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    compute_similarity(&guard, req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "grpc", feature = "cli"))]
async fn search_handler(req: SearchRequest) -> Result<SearchResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    // 改用 process_search_batch，候选向量先查缓存
    guard
        .process_search_batch(&req.query, &req.texts, req.top_k)
        .await
        .map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn unload_model_handler(req: UnloadModelRequest) -> Result<UnloadModelResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let mut guard = svc.write().await;
    // 服务器主路径不构造 ModelManager —— 此时卸载能力不可用，明确 404 而非谎报成功
    if !guard.has_model_manager() {
        return Err(ApiError::NotFound {
            resource: "model manager".to_string(),
            resource_id: None,
        });
    }
    guard
        .unload_model(&req.model_name)
        .await
        .map_err(to_api_error)?;
    Ok(UnloadModelResponse {
        model_name: req.model_name,
        unloaded: true,
    })
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn embed_file_handler(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    let mode = req.mode.unwrap_or(AggregationMode::Document);
    let path = PathBuf::from(&req.path);

    let validator = build_path_validator()?;
    let validated_path = validator
        .validate_file(&path)
        .map_err(|e| ApiError::InvalidInput {
            message: crate::i18n::tr_with_args(
                "validate-path-failed",
                crate::i18n::tr_args(&[("detail", &e.to_string())]),
            ),
            field: Some("path".to_string()),
            value: Some(serde_json::Value::String(req.path.clone())),
        })?;

    // 文件大小上限
    if let Ok(meta) = std::fs::metadata(&validated_path)
        && let Err(msg) = check_file_embed_size(meta.len())
    {
        return Err(ApiError::InvalidInput {
            message: msg,
            field: Some("path".to_string()),
            value: Some(serde_json::Value::String(req.path.clone())),
        });
    }

    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    let stats = guard
        .get_processing_stats(&validated_path)
        .map_err(to_api_error)?;
    let output = guard
        .embed_file(&validated_path, mode)
        .await
        .map_err(to_api_error)?;
    drop(guard);

    Ok(match output {
        EmbeddingOutput::Single(response) => FileEmbedResponse {
            mode,
            stats,
            embedding: Some(response.embedding),
            paragraphs: None,
        },
        EmbeddingOutput::Paragraphs(paragraphs) => {
            // text_preview 会回传文件原文,仅 admin 可见;auth 关闭时
            // (启动闸门已限制回环绑定)保留本地开发可用性。
            let may_preview = requester_may_preview(&st).await;
            let paragraphs = if may_preview {
                paragraphs
            } else {
                paragraphs
                    .into_iter()
                    .map(|mut p| {
                        p.text_preview = String::new();
                        p
                    })
                    .collect()
            };
            FileEmbedResponse {
                mode,
                stats,
                embedding: None,
                paragraphs: Some(paragraphs),
            }
        }
    })
}

/// 判定当前请求者是否可获取文件内容回传(text_preview):
/// auth 未启用 → 允许(非回环绑定已被启动闸门封堵);启用时 → 仅 admin。
#[cfg(all(any(feature = "http", feature = "grpc"), feature = "auth"))]
async fn requester_may_preview(st: &crate::VecboostState) -> bool {
    let auth_enabled = matches!(st.kit.require::<crate::registry::AuthModule>(), Ok(Some(_)));
    if !auth_enabled {
        return true;
    }
    crate::auth::middleware::current_token_is_admin_pub().await
}

/// 无 auth feature:预览权限不生效(无认证体系)。
#[cfg(all(any(feature = "http", feature = "grpc"), not(feature = "auth")))]
async fn requester_may_preview(st: &crate::VecboostState) -> bool {
    let _ = st;
    true
}

/// 构造 `/model/switch` 本地路径白名单校验器。
///
/// 允许根优先取 `[server] grpc_allowed_roots`;未配置时回落到进程工作目录下的
/// `models/`(项目默认模型目录)。显式提供 `model_path`/`tokenizer_path` 的
/// 模型切换必须落在允许根内,防止认证用户加载任意本地目录(超大文件 OOM/
/// 解析器攻击面/敏感路径探测)。
#[cfg(any(feature = "http", feature = "grpc", feature = "cli"))]
fn model_path_validator(configured_roots: Option<&[String]>) -> PathValidator {
    let mut validator = PathValidator::new();
    match configured_roots {
        Some(roots) if !roots.is_empty() => {
            validator = validator.add_allowed_roots(roots);
        }
        _ => {
            validator = validator.add_allowed_root("models");
        }
    }
    validator
}

#[cfg(any(feature = "http", feature = "grpc", feature = "cli"))]
async fn model_switch_handler(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    let st = state().map_err(to_api_error)?;

    // 显式本地路径必须落在白名单根内(HF repo-id 切换不受影响)
    if req.model_path.is_some() || req.tokenizer_path.is_some() {
        let server_cfg = st
            .kit
            .config::<crate::config::app::ServerConfig>()
            .unwrap_or_default();
        let validator = model_path_validator(server_cfg.grpc_allowed_roots.as_deref());
        for path in req.model_path.iter().chain(req.tokenizer_path.iter()) {
            validator.validate_directory(path).map_err(to_api_error)?;
        }
    }

    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let mut guard = svc.write().await;
    guard.switch_model(req).await.map_err(to_api_error)
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn get_current_model_handler() -> Result<ModelInfo, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    guard.get_model_info().ok_or_else(|| ApiError::NotFound {
        resource: "model".to_string(),
        resource_id: None,
    })
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn get_model_info_handler() -> Result<ModelMetadata, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    guard
        .get_model_metadata()
        .ok_or_else(|| ApiError::NotFound {
            resource: "model_metadata".to_string(),
            resource_id: None,
        })
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn list_models_handler() -> Result<ModelListResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    Ok(guard.list_available_models())
}

/// Unified health-check response — queries module health via trait-kit health checks.
///
/// Returns `{"status": "OK"}` when all modules are healthy, or
/// `ApiError::ServiceUnavailable` when any module reports unhealthy.
#[cfg(any(feature = "http", feature = "grpc"))]
async fn health_handler() -> Result<serde_json::Value, ApiError> {
    let st = state().map_err(to_api_error)?;
    let mut unhealthy_modules = Vec::new();

    // Query registered health checks
    match st.kit.health_check::<EmbeddingModule>() {
        Ok(status) if !status.is_healthy() => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[
                    ("module", "embedding"),
                    ("detail", &format!("{:?}", status)),
                ]),
            ));
        }
        Err(e) => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "embedding"), ("detail", &e.to_string())]),
            ));
        }
        Ok(_) => {}
    }
    match st.kit.health_check::<RerankModule>() {
        Ok(status) if !status.is_healthy() => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "rerank"), ("detail", &format!("{:?}", status))]),
            ));
        }
        Err(e) => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "rerank"), ("detail", &e.to_string())]),
            ));
        }
        Ok(_) => {}
    }
    match st.kit.health_check::<RateLimitModule>() {
        Ok(status) if !status.is_healthy() => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[
                    ("module", "rate_limit"),
                    ("detail", &format!("{:?}", status)),
                ]),
            ));
        }
        Err(e) => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "rate_limit"), ("detail", &e.to_string())]),
            ));
        }
        Ok(_) => {}
    }
    match st.kit.health_check::<CacheModule>() {
        Ok(status) if !status.is_healthy() => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "cache"), ("detail", &format!("{:?}", status))]),
            ));
        }
        Err(e) => {
            unhealthy_modules.push(crate::i18n::tr_with_args(
                "health-check-failed",
                crate::i18n::tr_args(&[("module", "cache"), ("detail", &e.to_string())]),
            ));
        }
        Ok(_) => {}
    }

    if unhealthy_modules.is_empty() {
        Ok(serde_json::json!({ "status": crate::i18n::tr("health-ok") }))
    } else {
        Err(ApiError::ServiceUnavailable {
            service: unhealthy_modules.join(", "),
            retry_after: Some(5),
            source: None,
        })
    }
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "embed",
    version = 1,
    path = "/embed",
    method = "POST",
    tool_name = "embed_text",
    description = "Generate embedding vector for input text"
)]
pub async fn forge_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "embed_batch",
    version = 1,
    path = "/embed/batch",
    method = "POST",
    tool_name = "embed_batch",
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn forge_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "compute_similarity",
    version = 1,
    path = "/similarity",
    method = "POST",
    tool_name = "compute_similarity",
    description = "Compute cosine similarity between two texts"
)]
pub async fn forge_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    compute_similarity_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "search",
    version = 1,
    path = "/search",
    method = "POST",
    tool_name = "search",
    description = "1-to-N semantic search: rank candidate texts by similarity to the query"
)]
pub async fn forge_search(req: SearchRequest) -> Result<SearchResponse, ApiError> {
    search_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "file_embed",
    version = 1,
    path = "/embed/file",
    method = "POST",
    tool_name = "file_embed",
    description = "Embed text from a file with path validation"
)]
pub async fn forge_file_embed(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    embed_file_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "health",
    version = 1,
    path = "/health",
    method = "GET",
    no_prefix = true,
    tool_name = "health",
    description = "Service health check"
)]
pub async fn forge_health() -> Result<serde_json::Value, ApiError> {
    health_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "model_switch",
    version = 1,
    path = "/model/switch",
    method = "POST",
    tool_name = "model_switch",
    description = "Switch the currently loaded model"
)]
pub async fn forge_model_switch(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    model_switch_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "get_current_model",
    version = 1,
    path = "/model/current",
    method = "GET",
    tool_name = "get_current_model",
    description = "Get information about the currently loaded model"
)]
pub async fn forge_get_current_model() -> Result<ModelInfo, ApiError> {
    get_current_model_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "get_model_info",
    version = 1,
    path = "/model/info",
    method = "GET",
    tool_name = "get_model_info",
    description = "Get metadata about the currently loaded model"
)]
pub async fn forge_get_model_info() -> Result<ModelMetadata, ApiError> {
    get_model_info_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "list_models",
    version = 1,
    path = "/models",
    method = "GET",
    tool_name = "list_models",
    description = "List all available models"
)]
pub async fn forge_list_models() -> Result<ModelListResponse, ApiError> {
    list_models_handler().await
}

#[cfg(feature = "http")]
#[forge(
    name = "model_unload",
    version = 1,
    path = "/model/unload",
    method = "POST",
    tool_name = "model_unload",
    description = "Unload a model from the model manager cache"
)]
pub async fn forge_unload_model(req: UnloadModelRequest) -> Result<UnloadModelResponse, ApiError> {
    unload_model_handler(req).await
}

#[cfg(feature = "http")]
#[forge(
    name = "openai_embed",
    version = 1,
    path = "/v1/embeddings",
    method = "POST",
    no_prefix = true,
    tool_name = "openai_embed",
    description = "OpenAI-compatible embeddings endpoint"
)]
pub async fn forge_openai_embed(req: OpenAIEmbedRequest) -> Result<OpenAIEmbedResponse, ApiError> {
    if req.input.is_empty() {
        return Err(ApiError::InvalidInput {
            message: crate::i18n::tr("openai-input-empty"),
            field: Some("input".to_string()),
            value: None,
        });
    }
    if req.input.len() > 2048 {
        return Err(ApiError::InvalidInput {
            message: crate::i18n::tr_with_args(
                "openai-input-too-large",
                crate::i18n::tr_args(&[("max", "2048")]),
            ),
            field: Some("input".to_string()),
            value: None,
        });
    }

    let st = state().map_err(to_api_error)?;
    validate_batch_size(req.input.len(), max_batch_size_from_kit(&st.kit)).map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;

    let texts = req.input.to_vec();
    validate_text_length(&texts, max_text_length_from_kit(&st.kit)).map_err(to_api_error)?;
    // 预先计算 total_chars 和 token 计数，避免后续 move texts 到 batch_req 后再访问
    let total_chars: usize = texts.iter().map(|s| s.len()).sum();
    // 使用 tokenizer 真实计数替代 bytes/4 估算
    let real_token_count: usize = texts
        .iter()
        .filter_map(|t| guard.count_tokens(t).ok())
        .sum();
    let batch_req = BatchEmbedRequest {
        texts,
        mode: None,
        normalize: Some(true),
    };
    let batch_response = guard
        .process_batch(batch_req, req.dimensions)
        .await
        .map_err(to_api_error)?;

    // DEFECT-OPENAI-001 修复：实现 OpenAI 规范的 encoding_format=base64
    // （小端 f32 字节流的 base64 编码），旧实现直接忽略该参数。
    let as_base64 = req.encoding_format.as_deref() == Some("base64");
    let embedding_objects: Vec<EmbeddingObject> = batch_response
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            let embedding = if as_base64 {
                #[cfg(feature = "http")]
                {
                    use base64::Engine as _;
                    let bytes: Vec<u8> = result
                        .embedding
                        .iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect();
                    EmbeddingData::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
                #[cfg(not(feature = "http"))]
                {
                    let _ = as_base64;
                    EmbeddingData::Floats(result.embedding)
                }
            } else {
                EmbeddingData::Floats(result.embedding)
            };
            EmbeddingObject {
                object: "embedding".to_string(),
                embedding,
                index: idx,
            }
        })
        .collect();

    // 优先使用 tokenizer 真实计数,回退到 bytes/4
    let prompt_tokens = if real_token_count > 0 {
        real_token_count as u32
    } else {
        (total_chars / 4) as u32
    };

    Ok(OpenAIEmbedResponse {
        object: "list".to_string(),
        data: embedding_objects,
        model: req.model.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "embed",
    version = 1,
    cli = true,
    description = "Generate embedding vector for input text"
)]
pub async fn cli_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}

#[cfg(feature = "cli")]
#[forge(
    name = "embed_batch",
    version = 1,
    cli = true,
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn cli_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

#[cfg(feature = "cli")]
#[forge(
    name = "compute_similarity",
    version = 1,
    cli = true,
    description = "Compute cosine similarity between two texts"
)]
pub async fn cli_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    compute_similarity_handler(req).await
}

// =============================================================================
// gRPC forge handlers — sdforge unified `Call` protocol
//
// Each function is registered via `#[forge(grpc_method = "...")]` and invoked
// through sdforge's `SdForgeService/Call` RPC with the corresponding method
// name. Request payloads are JSON-serialized domain types passed via
// `CallRequest.data`; responses are JSON-serialized domain types returned in
// `CallResponse.data`.
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed",
    version = 1,
    grpc_method = "vecboost.embed",
    description = "Generate embedding vector for input text"
)]
pub async fn grpc_embed(req: EmbedRequest) -> Result<EmbedResponse, ApiError> {
    embed_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed_batch",
    version = 1,
    grpc_method = "vecboost.embed_batch",
    description = "Generate embedding vectors for multiple texts in batch"
)]
pub async fn grpc_embed_batch(req: BatchEmbedRequest) -> Result<BatchEmbedResponse, ApiError> {
    embed_batch_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_compute_similarity",
    version = 1,
    grpc_method = "vecboost.compute_similarity",
    description = "Compute similarity between two texts"
)]
pub async fn grpc_compute_similarity(
    req: SimilarityRequest,
) -> Result<SimilarityResponse, ApiError> {
    compute_similarity_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_embed_file",
    version = 1,
    grpc_method = "vecboost.embed_file",
    description = "Embed text from a file with path validation"
)]
pub async fn grpc_embed_file(req: FileEmbedRequest) -> Result<FileEmbedResponse, ApiError> {
    embed_file_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_model_switch",
    version = 1,
    grpc_method = "vecboost.model_switch",
    description = "Switch the currently loaded model"
)]
pub async fn grpc_model_switch(req: ModelSwitchRequest) -> Result<ModelSwitchResponse, ApiError> {
    model_switch_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_get_current_model",
    version = 1,
    grpc_method = "vecboost.get_current_model",
    description = "Get information about the currently loaded model"
)]
pub async fn grpc_get_current_model() -> Result<ModelInfo, ApiError> {
    get_current_model_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_get_model_info",
    version = 1,
    grpc_method = "vecboost.get_model_info",
    description = "Get metadata about the currently loaded model"
)]
pub async fn grpc_get_model_info() -> Result<ModelMetadata, ApiError> {
    get_model_info_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_list_models",
    version = 1,
    grpc_method = "vecboost.list_models",
    description = "List all available models"
)]
pub async fn grpc_list_models() -> Result<ModelListResponse, ApiError> {
    list_models_handler().await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_search",
    version = 1,
    grpc_method = "vecboost.search",
    description = "1-to-N semantic search over candidate texts"
)]
pub async fn grpc_search(req: SearchRequest) -> Result<SearchResponse, ApiError> {
    search_handler(req).await
}

#[cfg(feature = "cli")]
#[forge(
    name = "search",
    version = 1,
    cli = true,
    description = "Rank candidate texts by similarity to the query (1-to-N search)"
)]
pub async fn cli_search(req: SearchRequest) -> Result<SearchResponse, ApiError> {
    search_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_model_unload",
    version = 1,
    grpc_method = "vecboost.model_unload",
    description = "Unload a model from the model manager cache"
)]
pub async fn grpc_unload_model(req: UnloadModelRequest) -> Result<UnloadModelResponse, ApiError> {
    unload_model_handler(req).await
}

#[cfg(feature = "grpc")]
#[forge(
    name = "vecboost_health_check",
    version = 1,
    grpc_method = "vecboost.health_check",
    description = "Service health check"
)]
pub async fn grpc_health_check() -> Result<serde_json::Value, ApiError> {
    health_handler().await
}

#[cfg(test)]
mod tests {
    /// 文件大小上限(10 MiB 内通过,超限报错并给出限值)
    #[test]
    fn check_file_embed_size_enforces_limit() {
        assert!(check_file_embed_size(0).is_ok());
        assert!(check_file_embed_size(10 * 1024 * 1024).is_ok());
        let err = check_file_embed_size(10 * 1024 * 1024 + 1).unwrap_err();
        assert!(err.contains("10 MiB"));
    }

    /// model_path 白名单 —— 配置根内通过,越界拒绝,未配置回落 models/
    #[test]
    fn model_path_validator_enforces_allowed_roots() {
        let base = std::env::temp_dir().join(format!("vb_model_paths_{}", std::process::id()));
        let inside = base.join("my-model");
        std::fs::create_dir_all(&inside).expect("create dirs");

        let roots = vec![base.to_string_lossy().to_string()];
        let validator = model_path_validator(Some(&roots));
        assert!(validator.validate_directory(&inside).is_ok());

        let outside = std::env::temp_dir().join(format!("vb_outside_{}", std::process::id()));
        std::fs::create_dir_all(&outside).expect("create dirs");
        assert!(validator.validate_directory(&outside).is_err());

        // 未配置 grpc_allowed_roots → 默认根为 models/
        let default_validator = model_path_validator(None);
        let default_root = std::path::Path::new("models").canonicalize().ok();
        if let Some(root) = default_root {
            assert!(default_validator.validate_directory(&root).is_ok());
        }

        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    use super::*;

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_text_length_under_limit_passes() {
        let texts = vec!["short".to_string(), "also short".to_string()];
        assert!(validate_text_length(&texts, 100).is_ok());
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_text_length_at_boundary_passes() {
        let text = "a".repeat(8192);
        let texts = vec![text];
        assert!(validate_text_length(&texts, 8192).is_ok());
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_text_length_exceeds_limit_returns_error() {
        crate::i18n::init();
        let texts = vec!["ok".to_string(), "x".repeat(101)];
        let err = validate_text_length(&texts, 100).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                // i18n translated message contains the parameter values
                assert!(msg.contains("1"), "error should mention index 1: {msg}");
                assert!(msg.contains("100"), "error should mention limit: {msg}");
                assert!(
                    msg.contains("101"),
                    "error should mention actual length: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_text_length_empty_slice_passes() {
        let texts: Vec<String> = vec![];
        assert!(validate_text_length(&texts, 100).is_ok());
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_text_length_first_offending_text_reported() {
        crate::i18n::init();
        let texts = vec!["x".repeat(51), "x".repeat(52)];
        let err = validate_text_length(&texts, 50).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                // Message should contain the offending index "0"
                assert!(
                    msg.contains("0"),
                    "should report first offending index: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_under_limit_passes() {
        assert!(validate_batch_size(10, 64).is_ok());
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_exceeds_limit_returns_error() {
        crate::i18n::init();
        let err = validate_batch_size(100, 64).unwrap_err();
        match err {
            VecboostError::ValidationError(msg) => {
                assert!(
                    msg.contains("100"),
                    "error should mention actual size: {msg}"
                );
                assert!(msg.contains("64"), "error should mention limit: {msg}");
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    #[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
    #[test]
    fn test_validate_batch_size_at_boundary_passes() {
        assert!(validate_batch_size(64, 64).is_ok());
    }
}
