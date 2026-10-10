// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Transcribe 服务实现：门面会话句柄 + spawn_blocking 推理隔离。
//!
//! 与 translate 同构的「端点恒注册、能力按装配」语义：`enabled=false` 或
//! `transcribe` feature 未启用时 `session=None`，请求以 400
//! transcribe-not-configured 显性拒绝（禁 404 禁 500 禁 panic）。并发背压
//! 同 translate（permits=1：会话内 asr Mutex 串行化，等待者 503 Overloaded）。
//!
//! 推理链经公开门面 `vecboost::transcribe`（VAD 分段 → 逐段解码 → 段序
//! 拼接），与 `transcribe-poc` 薄壳共用同一入口（单一事实源）。

#[cfg(feature = "transcribe")]
use crate::domain::TranscribeSegment;
use crate::domain::{TranscribeRequest, TranscribeResponse};
use crate::error::VecboostError;
use crate::i18n;
#[cfg(feature = "transcribe")]
use std::sync::Arc;
#[cfg(feature = "transcribe")]
use std::time::Instant;
#[cfg(feature = "transcribe")]
use tokio::sync::Mutex;
#[cfg(feature = "transcribe")]
use tokio::sync::Semaphore;

/// 转录服务：单门面会话（Qwen3-ASR 三图 + VAD）。
///
/// 服务构造后不可变（翻译多引擎同款口径）：会话在启动期急切装配，
/// `replace_engine` 入口不开放——switch_model 的主模型为 embedding 任务时
/// 注入会静默废掉转录能力。
pub struct TranscribeService {
    /// 门面会话句柄；`None` = 未装配（feature 未启用或 enabled=false）
    #[cfg(feature = "transcribe")]
    session: Option<Arc<Mutex<crate::transcribe::TranscribeSession>>>,
    /// 并发背压许可（permits=1：会话内串行推理；transcribe-only 消费）
    #[cfg(feature = "transcribe")]
    semaphore: Arc<Semaphore>,
    /// 配置态（日志/诊断）
    configured: bool,
}

impl TranscribeService {
    /// 未装配构造（feature 未启用 / `[transcribe] enabled=false`）。
    pub fn unconfigured() -> Self {
        Self {
            #[cfg(feature = "transcribe")]
            session: None,
            #[cfg(feature = "transcribe")]
            semaphore: Arc::new(Semaphore::new(1)),
            configured: false,
        }
    }

    /// 已装配构造（启动期经门面 `open` 创建会话后注入）。
    #[cfg(feature = "transcribe")]
    pub fn new(session: crate::transcribe::TranscribeSession) -> Self {
        Self {
            session: Some(Arc::new(Mutex::new(session))),
            semaphore: Arc::new(Semaphore::new(1)),
            configured: true,
        }
    }

    /// 是否已装配会话（启动日志/健康视图）。
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// 转录本体：域层校验 → 装配门 → 背压 → spawn_blocking 推理
    /// （VAD 分段 → 逐段解码 → 段序拼接，全部在 blocking 线程内）。
    pub async fn process_transcribe(
        &self,
        req: TranscribeRequest,
    ) -> Result<TranscribeResponse, VecboostError> {
        // 域层校验（非空/有限值/长度上限）先于装配门——坏请求恒 400
        req.validate()?;

        #[cfg(feature = "transcribe")]
        {
            let session = match &self.session {
                Some(s) => s.clone(),
                None => {
                    return Err(VecboostError::InvalidInput(
                        i18n::tr("transcribe-not-configured").to_string(),
                    ));
                }
            };

            // 背压：会话串行推理，排队请求显性 503（与 translate 同口径）
            let _permit = self.semaphore.clone().try_acquire_owned().map_err(|_| {
                VecboostError::Overloaded(
                    "transcribe session is busy with another request; retry shortly".to_string(),
                )
            })?;

            let started = Instant::now();
            let resp = tokio::task::spawn_blocking(move || {
                let session = session.blocking_lock();
                let segments = session.detect_segments(&req.samples)?;
                let mut out: Vec<TranscribeSegment> = Vec::with_capacity(segments.len());
                for seg in &segments {
                    let start = seg.start_sample.max(0) as usize;
                    let end = (seg.end_sample.max(0) as usize).min(req.samples.len());
                    if end <= start {
                        continue;
                    }
                    let text = session.transcribe_segment(&req.samples[start..end])?;
                    if text.is_empty() {
                        continue;
                    }
                    out.push(TranscribeSegment {
                        start_sec: seg.start_sec,
                        end_sec: seg.end_sec,
                        text,
                    });
                }
                Ok::<_, VecboostError>(TranscribeResponse::from_segments(out))
            })
            .await
            .map_err(|e| VecboostError::InferenceError(format!("transcribe task join: {e}")))??;

            log::info!(
                "transcribe: {} segments, {} chars, {} ms",
                resp.segments.len(),
                resp.text.chars().count(),
                started.elapsed().as_millis(),
            );
            Ok(resp)
        }
        #[cfg(not(feature = "transcribe"))]
        {
            let _ = &req;
            Err(VecboostError::InvalidInput(
                i18n::tr("transcribe-not-configured").to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unconfigured_service_rejects_explicitly() {
        let svc = TranscribeService::unconfigured();
        assert!(!svc.is_configured());
        let req = TranscribeRequest {
            samples: vec![0.0; 16000],
        };
        let err = svc
            .process_transcribe(req)
            .await
            .expect_err("未装配必须显性拒绝");
        assert!(matches!(err, VecboostError::InvalidInput(_)), "err={err:?}");
    }

    #[tokio::test]
    async fn invalid_samples_rejected_before_gate() {
        let svc = TranscribeService::unconfigured();
        // 空 samples：域层校验先于装配门——坏请求恒 ValidationError
        let req = TranscribeRequest { samples: vec![] };
        let err = svc
            .process_transcribe(req)
            .await
            .expect_err("空输入必须 ValidationError");
        assert!(
            matches!(
                err,
                VecboostError::ValidationError(_) | VecboostError::InvalidInput(_)
            ),
            "err={err:?}"
        );
    }
}
