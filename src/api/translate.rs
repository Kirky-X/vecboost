// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Translate forge handlers — HTTP/CLI/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/detect.rs` 相同模式：协议无关的 `translate_handler`
//! 包含业务逻辑，`forge_*` / `cli_*` / `grpc_*` 仅为薄包装。
//!
//! # 直连处理（无 pipeline 分流）
//! 翻译不入 pipeline 队列、与 pipeline_enabled 配置无关（design.md
//! Non-Goals）：路由表按 (src,tgt) 多引擎分派，pipeline 的单引擎
//! ServiceRequest 语义不适配多语言对路由；服务层已含 spawn_blocking
//! 阻塞隔离，二次调度只会叠加延迟。

use crate::domain::{TranslateRequest, TranslateResponse};
use crate::error::VecboostError;
use crate::service::translate::TranslateService;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::registry::TranslateModule;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions
// =============================================================================

pub async fn translate(
    svc: &TranslateService,
    req: TranslateRequest,
) -> Result<TranslateResponse, VecboostError> {
    svc.process_translate(req).await
}

// =============================================================================
// Protocol-agnostic handlers
// =============================================================================

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn translate_handler(req: TranslateRequest) -> Result<TranslateResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<TranslateModule>()
        .map_err(kit_internal_error)?;
    translate(&svc, req).await.map_err(to_api_error)
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "translate",
    version = 1,
    path = "/translate",
    method = "POST",
    tool_name = "translate",
    description = "Translate text between configured language pairs (seq2seq)"
)]
pub async fn forge_translate(req: TranslateRequest) -> Result<TranslateResponse, ApiError> {
    translate_handler(req).await
}

// =============================================================================
// CLI forge handlers
// =============================================================================

#[cfg(feature = "cli")]
#[forge(
    name = "translate",
    version = 1,
    cli = true,
    description = "Translate text between configured language pairs (seq2seq)",
    i18n_key = "forge-cli-translate"
)]
pub async fn cli_translate(req: TranslateRequest) -> Result<TranslateResponse, ApiError> {
    translate_handler(req).await
}

// =============================================================================
// gRPC forge handlers
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "translate",
    version = 1,
    grpc_method = "vecboost.translate",
    description = "Translate text between configured language pairs (seq2seq)"
)]
pub async fn grpc_translate(req: TranslateRequest) -> Result<TranslateResponse, ApiError> {
    translate_handler(req).await
}
