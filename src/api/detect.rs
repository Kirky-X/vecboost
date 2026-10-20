// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Detect forge handlers — HTTP/CLI/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/decisions.rs` 相同模式：协议无关的 `detect_handler`
//! 包含业务逻辑，`forge_*` / `cli_*` / `grpc_*` 仅为薄包装。
//!
//! # 直连处理（无 pipeline 分流）
//! 检测为 µs 级纯 CPU 统计，经优先级队列调度的收益为负（入队开销远超
//! 检测本身），与 pipeline_enabled 配置无关（design.md Non-Goals）；
//! 其余端点的 pipeline_enabled 分支此处不适用。

use crate::domain::{DetectRequest, DetectResponse};
use crate::error::VecboostError;
use crate::service::detect::DetectService;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::api::init::state;
#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use crate::registry::DetectModule;

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
use sdforge::prelude::*;

// 消歧：sdforge prelude 亦再导出其 utoipa(6.0)，显式导入钉住本 crate 的 5.5
#[cfg(all(feature = "openapi", feature = "http"))]
use ::utoipa;

// Public SDK functions

pub async fn detect(
    svc: &DetectService,
    req: DetectRequest,
) -> Result<DetectResponse, VecboostError> {
    svc.process_detect(req).await
}

// Protocol-agnostic handlers

#[cfg(any(feature = "http", feature = "cli", feature = "grpc"))]
async fn detect_handler(req: DetectRequest) -> Result<DetectResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<DetectModule>()
        .map_err(kit_internal_error)?;
    detect(&svc, req).await.map_err(to_api_error)
}

// HTTP forge handlers

#[cfg(feature = "http")]
#[forge(
    name = "detect",
    version = "v1",
    path = "/detect",
    method = "POST",
    tool_name = "detect",
    description = "Detect the natural language of a text (ISO 639-1 code with 639-3 fallback)"
)]
pub async fn forge_detect(req: DetectRequest) -> Result<DetectResponse, ApiError> {
    detect_handler(req).await
}

// CLI forge handlers

#[cfg(feature = "cli")]
#[forge(
    name = "detect",
    version = "v1",
    cli = true,
    description = "Detect the natural language of a text (ISO 639-1 code with 639-3 fallback)",
    i18n_key = "forge-cli-detect"
)]
pub async fn cli_detect(req: DetectRequest) -> Result<DetectResponse, ApiError> {
    detect_handler(req).await
}

// gRPC forge handlers

#[cfg(feature = "grpc")]
#[forge(
    name = "detect",
    version = "v1",
    grpc_method = "vecboost.detect",
    description = "Detect the natural language of a text (ISO 639-1 code with 639-3 fallback)"
)]
pub async fn grpc_detect(req: DetectRequest) -> Result<DetectResponse, ApiError> {
    detect_handler(req).await
}
