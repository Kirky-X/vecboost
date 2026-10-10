// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Transcribe forge handlers — HTTP/gRPC protocol-agnostic.
//!
//! 遵循 `src/api/translate.rs` 相同模式：协议无关的 `transcribe` SDK 函数 +
//! `forge_*` / `grpc_*` 薄包装。
//!
//! **不提供 CLI 子命令**：域契约 `TranscribeRequest.samples` 是 f32 样本
//! 数组，CLI 逐样本传参无实用形态；音频文件入参路径由 `transcribe-poc`
//! 薄壳 CLI（读 wav → samples → 本端点）承接。
//!
//! # 直连处理（无 pipeline 分流）
//! 转录不入 pipeline 队列（单引擎阻塞推理，服务层已含 spawn_blocking 隔离，
//! 与 translate 同口径）。

use crate::domain::{TranscribeRequest, TranscribeResponse};
use crate::error::VecboostError;
use crate::service::transcribe::TranscribeService;

#[cfg(any(feature = "http", feature = "grpc"))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(feature = "http", feature = "grpc"))]
use crate::api::init::state;
#[cfg(any(feature = "http", feature = "grpc"))]
use crate::registry::TranscribeModule;

#[cfg(any(feature = "http", feature = "grpc"))]
use sdforge::prelude::*;

// =============================================================================
// Public SDK functions
// =============================================================================

pub async fn transcribe(
    svc: &TranscribeService,
    req: TranscribeRequest,
) -> Result<TranscribeResponse, VecboostError> {
    svc.process_transcribe(req).await
}

// =============================================================================
// Protocol-agnostic handlers
// =============================================================================

#[cfg(any(feature = "http", feature = "grpc"))]
async fn transcribe_handler(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<TranscribeModule>()
        .map_err(kit_internal_error)?;
    transcribe(&svc, req).await.map_err(to_api_error)
}

// =============================================================================
// HTTP forge handlers
// =============================================================================

#[cfg(feature = "http")]
#[forge(
    name = "transcribe",
    version = 1,
    path = "/transcribe",
    method = "POST",
    tool_name = "transcribe",
    description = "Transcribe 16kHz mono f32 speech samples (Qwen3-ASR: VAD + per-segment decode)"
)]
pub async fn forge_transcribe(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    transcribe_handler(req).await
}

// =============================================================================
// gRPC forge handlers
// =============================================================================

#[cfg(feature = "grpc")]
#[forge(
    name = "transcribe",
    version = 1,
    grpc_method = "vecboost.transcribe",
    description = "Transcribe 16kHz mono f32 speech samples (Qwen3-ASR)"
)]
pub async fn grpc_transcribe(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    transcribe_handler(req).await
}
