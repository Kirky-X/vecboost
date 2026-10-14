// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 服务层公共工具函数（OOM 降级等）

use crate::config::model::ModelConfig;
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::i18n;
use crate::model::manager::ModelManager;
use log::warn;
use std::sync::Arc;
use tokio::sync::RwLock;

const MAX_FALLBACK_ATTEMPTS: usize = 2;

/// 判断错误是否为 OOM（内存溢出）错误
pub fn is_oom_error(error: &VecboostError) -> bool {
    match error {
        VecboostError::InferenceError(msg) | VecboostError::OutOfMemory(msg) => {
            let lower_msg = msg.to_lowercase();
            lower_msg.contains("out of memory")
                || lower_msg.contains("cuda out of memory")
                || lower_msg.contains("gpu out of memory")
                || lower_msg.contains("memory allocation failed")
                || lower_msg.contains("failed to allocate")
                || lower_msg.contains("not enough memory")
                || (lower_msg.contains("alloc")
                    && (lower_msg.contains("fail")
                        || lower_msg.contains("error")
                        || lower_msg.contains("unable")))
        }
        _ => false,
    }
}

/// 通用 OOM 降级处理器：检测 OOM 错误后尝试回退到 CPU 并重试
pub async fn handle_oom_fallback<F, Fut, T>(
    engine: &Arc<RwLock<dyn InferenceEngine + Send + Sync>>,
    model_config: &Option<ModelConfig>,
    model_manager: &Option<Arc<ModelManager>>,
    operation: F,
) -> Result<T, VecboostError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, VecboostError>>,
{
    let mut attempts = 0;

    loop {
        attempts += 1;

        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) if is_oom_error(&error) && attempts <= MAX_FALLBACK_ATTEMPTS => {
                warn!(
                    "{}",
                    crate::i18n::tr_with_args(
                        "log-service-oom-error-detected-attempting",
                        crate::i18n::tr_args(&[
                            ("a1", &(error).to_string()),
                            ("a2", &(attempts).to_string()),
                            ("a3", &(MAX_FALLBACK_ATTEMPTS).to_string())
                        ])
                    )
                );

                let engine_read = engine.read().await;

                if engine_read.is_fallback_triggered() {
                    warn!(
                        "{}",
                        crate::i18n::tr("log-service-fallback-already-triggered-cannot")
                    );
                    return Err(VecboostError::OutOfMemory(i18n::tr("oom-no-fallback")));
                }

                drop(engine_read);

                if let Some(config) = model_config
                    && let Some(manager) = model_manager
                    && let Some(_model) = manager.get(&config.name).await
                {
                    let mut engine_guard = engine.write().await;
                    let config_clone = config.clone();
                    let fallback_result = engine_guard.try_fallback_to_cpu(&config_clone).await;

                    match fallback_result {
                        Ok(()) => {
                            warn!(
                                "{}",
                                crate::i18n::tr("log-service-successfully-fell-back-cpu")
                            );
                            continue;
                        }
                        Err(e) => {
                            warn!(
                                "{}",
                                crate::i18n::tr_with_args(
                                    "log-service-failed-fallback-cpu",
                                    crate::i18n::tr_args(&[("a1", &(e).to_string())])
                                )
                            );
                            return Err(VecboostError::OutOfMemory(i18n::tr_with_args(
                                "oom-fallback-failed",
                                i18n::tr_args(&[("detail", &e.to_string())]),
                            )));
                        }
                    }
                }

                return Err(VecboostError::OutOfMemory(i18n::tr(
                    "oom-no-fallback-available",
                )));
            }
            Err(error) if is_oom_error(&error) => {
                return Err(VecboostError::OutOfMemory(i18n::tr_with_args(
                    "oom-max-attempts",
                    i18n::tr_args(&[
                        ("attempts", &MAX_FALLBACK_ATTEMPTS.to_string()),
                        ("detail", &error.to_string()),
                    ]),
                )));
            }
            Err(error) => return Err(error),
        }
    }
}
