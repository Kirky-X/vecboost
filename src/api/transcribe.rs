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
// `Deserialize` 仅 CLI 入参结构（cli + transcribe）消费；其余 feature 组合下不得引入未用导入。
#[cfg(all(feature = "cli", feature = "transcribe"))]
use serde::Deserialize;

#[cfg(any(
    feature = "http",
    feature = "grpc",
    all(feature = "cli", feature = "transcribe")
))]
use crate::api::embedding::{kit_internal_error, to_api_error};
#[cfg(any(
    feature = "http",
    feature = "grpc",
    all(feature = "cli", feature = "transcribe")
))]
use crate::api::init::state;
#[cfg(any(
    feature = "http",
    feature = "grpc",
    all(feature = "cli", feature = "transcribe")
))]
use crate::registry::TranscribeModule;

#[cfg(any(feature = "http", feature = "grpc"))]
use sdforge::prelude::*;

// 消歧：sdforge prelude 亦再导出其 utoipa(6.0)，显式导入钉住本 crate 的 5.5
// （http forge 的 #[utoipa::path] 与 CLI 入参 ToSchema 派生均按此名解析）
#[cfg(any(
    all(feature = "openapi", feature = "http"),
    all(feature = "schema", feature = "cli", feature = "transcribe")
))]
use ::utoipa;

// Public SDK functions

pub async fn transcribe(
    svc: &TranscribeService,
    req: TranscribeRequest,
) -> Result<TranscribeResponse, VecboostError> {
    svc.process_transcribe(req).await
}

// Protocol-agnostic handlers

#[cfg(any(feature = "http", feature = "grpc"))]
async fn transcribe_handler(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<TranscribeModule>()
        .map_err(kit_internal_error)?;
    transcribe(&svc, req).await.map_err(to_api_error)
}

// HTTP forge handlers

#[cfg(feature = "http")]
#[forge(
    name = "transcribe",
    version = "v1",
    path = "/transcribe",
    method = "POST",
    tool_name = "transcribe",
    description = "Transcribe 16kHz mono f32 speech samples (Qwen3-ASR: VAD + per-segment decode)"
)]
pub async fn forge_transcribe(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    transcribe_handler(req).await
}

// gRPC forge handlers

#[cfg(feature = "grpc")]
#[forge(
    name = "transcribe",
    version = "v1",
    grpc_method = "vecboost.transcribe",
    description = "Transcribe 16kHz mono f32 speech samples (Qwen3-ASR)"
)]
pub async fn grpc_transcribe(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    transcribe_handler(req).await
}

// CLI forge handler（音频文件入参）

/// CLI 入参：WAV 文件路径（域契图为 samples 数组，CLI 逐样本传参无实用形态，
/// 故 CLI 面收敛为单文件路径——读 wav → samples → 域请求）。
#[cfg(all(feature = "cli", feature = "transcribe"))]
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct TranscribeWavRequest {
    pub wav_path: String,
}

#[cfg(all(feature = "cli", feature = "transcribe"))]
impl std::str::FromStr for TranscribeWavRequest {
    /// # 红线（cli/grpc forge 落地时）
    /// `Err = serde_json::Error` 携带内部结构名与字节偏移，handler 必须
    /// 映射为通用 `VecboostError::validation_error` 文案（translate 同款）。
    type Err = serde_json::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[cfg(all(feature = "cli", feature = "transcribe"))]
#[forge(
    name = "transcribe",
    version = "v1",
    cli = true,
    description = "Transcribe a 16kHz mono WAV file to text (Qwen3-ASR; requires [transcribe] enabled)",
    i18n_key = "forge-cli-transcribe"
)]
pub async fn cli_transcribe(req: TranscribeWavRequest) -> Result<TranscribeResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<TranscribeModule>()
        .map_err(kit_internal_error)?;
    let samples = crate::transcribe::read_wav_16k_mono(std::path::Path::new(&req.wav_path))
        .map_err(to_api_error)?;
    transcribe(&svc, TranscribeRequest { samples })
        .await
        .map_err(to_api_error)
}
