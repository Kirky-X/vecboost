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

/// 转录错误 → wire 映射（脱敏，同 `switch_error_to_api_error` 口径）：
/// 推理/加载期错误携带内部张量 shape/长度与 bundle 路径，`to_api_error`
/// 的 ModelLoadError 臂会把原文整体塞进 NotFound.resource_id 直达响应体
/// （CWE-209/497，先例见 api/embedding.rs switch 路径修复）。原文全量入
/// 服务端日志；UnsupportedTask 与输入类错误直传——客户端可纠正的 4xx
/// 语义，不得被 catch-all 洗成 404。
#[cfg(any(feature = "http", feature = "grpc"))]
fn transcribe_error_to_api_error(e: VecboostError) -> ApiError {
    match e {
        VecboostError::UnsupportedTask(_)
        | VecboostError::InvalidInput(_)
        | VecboostError::ValidationError(_) => to_api_error(e),
        e => {
            log::error!(
                "{}",
                crate::i18n::tr_with_args(
                    "log-api-transcribe-failed-e",
                    crate::i18n::tr_args(&[("e", &(e).to_string())])
                )
            );
            ApiError::NotFound {
                resource: "model".to_string(),
                resource_id: Some(crate::i18n::tr("transcribe-model-load-failed").to_string()),
            }
        }
    }
}

#[cfg(any(feature = "http", feature = "grpc"))]
async fn transcribe_handler(req: TranscribeRequest) -> Result<TranscribeResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    let svc = st
        .kit
        .require::<TranscribeModule>()
        .map_err(kit_internal_error)?;
    transcribe(&svc, req)
        .await
        .map_err(transcribe_error_to_api_error)
}

// HTTP forge handlers

#[cfg(feature = "http")]
#[forge(
    name = "transcribe",
    version = "v1",
    path = "/transcribe",
    method = "POST",
    tool_name = "transcribe",
    description = "Transcribe 16kHz mono f32 speech samples (whisper/Qwen3-ASR backend dispatch: VAD + per-segment decode)"
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
    description = "Transcribe 16kHz mono f32 speech samples (whisper/Qwen3-ASR backend dispatch)"
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
    description = "Transcribe a 16kHz mono WAV file to text (whisper/Qwen3-ASR backend dispatch; requires [transcribe] enabled)",
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
        .map_err(transcribe_error_to_api_error)
}

#[cfg(all(test, any(feature = "http", feature = "grpc")))]
mod tests {
    use super::*;

    /// 推理/加载期错误脱敏：内部 shape/路径不得直达 wire
    /// （CWE-209/497，switch 路径同款红线；e2e 审查发现的脱敏缺口回归钉）。
    #[test]
    fn inference_and_load_errors_are_desensitized_on_wire() {
        for e in [
            VecboostError::ModelLoadError(
                "unexpected encoder output shape [1,1500,1280], expected [1,seq,1280]".to_string(),
            ),
            VecboostError::InferenceError(
                "ort run failed: logits len 1920000 != 1*1500".to_string(),
            ),
        ] {
            let api = transcribe_error_to_api_error(e);
            let msg = match &api {
                ApiError::NotFound {
                    resource,
                    resource_id: Some(m),
                    ..
                } => {
                    assert_eq!(resource, "model");
                    m.clone()
                }
                other => panic!("expected NotFound(model), got {other:?}"),
            };
            assert!(
                msg.contains("server logs")
                    || msg.contains("\u{670d}\u{52a1}\u{7aef}\u{65e5}\u{5fd7}"),
                "must be generic message, got {msg}"
            );
            assert!(
                !msg.contains("1500"),
                "internal detail must not leak, got {msg}"
            );
        }
    }

    /// 输入类错误直传 4xx：调用方可纠正语义不得被 catch-all 洗成 404。
    #[test]
    fn input_errors_pass_through() {
        let api = transcribe_error_to_api_error(VecboostError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
        assert!(
            matches!(api, ApiError::InvalidInput { .. }),
            "InvalidInput must pass through, got {api:?}"
        );
    }
}
