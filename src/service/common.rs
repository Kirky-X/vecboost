// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 服务层公共工具函数（OOM 降级等）

use crate::config::model::ModelConfig;
use crate::engine::InferenceEngine;
use crate::error::VecboostError;
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
                    "OOM error detected: {}. Attempting fallback to CPU (attempt {}/{})",
                    error, attempts, MAX_FALLBACK_ATTEMPTS
                );

                let engine_read = engine.read().await;

                if engine_read.is_fallback_triggered() {
                    warn!("Fallback already triggered, cannot retry");
                    return Err(VecboostError::OutOfMemory(
                        "Out of memory and fallback already attempted".to_string(),
                    ));
                }

                drop(engine_read);

                if let Some(config) = model_config {
                    if let Some(manager) = model_manager {
                        let loaded_model = manager.get(&config.name).await;

                        if let Some(_model) = loaded_model {
                            let mut engine_guard = engine.write().await;
                            let config_clone = config.clone();
                            let fallback_result =
                                engine_guard.try_fallback_to_cpu(&config_clone).await;

                            match fallback_result {
                                Ok(()) => {
                                    warn!("Successfully fell back to CPU, retrying operation");
                                    continue;
                                }
                                Err(e) => {
                                    warn!("Failed to fallback to CPU: {}", e);
                                    return Err(VecboostError::OutOfMemory(format!(
                                        "OOM error [{}] and fallback failed: {}",
                                        error, e
                                    )));
                                }
                            }
                        }
                    }
                }

                return Err(VecboostError::OutOfMemory(
                    "Out of memory and no fallback available".to_string(),
                ));
            }
            Err(error) if is_oom_error(&error) => {
                return Err(VecboostError::OutOfMemory(format!(
                    "Max fallback attempts exceeded ({}). Last error: {}",
                    MAX_FALLBACK_ATTEMPTS, error
                )));
            }
            Err(error) => return Err(error),
        }
    }
}
