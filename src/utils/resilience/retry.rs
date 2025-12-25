// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use std::future::Future;
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponent_base: f64,
    pub include_error_type: bool,
    pub retryable_errors: Vec<&'static str>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            exponent_base: 2.0,
            include_error_type: true,
            retryable_errors: vec![
                "Tokenization error",
                "Inference error",
                "timeout",
                "temporary",
                "busy",
            ],
        }
    }
}

impl RetryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_initial_delay(mut self, delay_ms: u64) -> Self {
        self.initial_delay_ms = delay_ms;
        self
    }

    pub fn with_max_delay(mut self, delay_ms: u64) -> Self {
        self.max_delay_ms = delay_ms;
        self
    }

    pub fn with_exponent_base(mut self, base: f64) -> Self {
        self.exponent_base = base;
        self
    }
}

pub trait Retryable<T> {
    type Future: Future<Output = Result<T, AppError>>;
    fn call(&self) -> Self::Future;
}

impl<F, T, Fut> Retryable<T> for F
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, AppError>>,
{
    type Future = Fut;
    fn call(&self) -> Self::Future {
        (self)()
    }
}

pub async fn with_retry<T, R: Retryable<T>>(
    retryable: &R,
    config: Option<RetryConfig>,
) -> Result<T, AppError> {
    let config = config.unwrap_or_default();
    let mut last_error: Option<AppError> = None;
    let start_time = Instant::now();

    for attempt in 0..=config.max_retries {
        match retryable.call().await {
            Ok(result) => {
                if attempt > 0 {
                    tracing::info!(
                        "Operation succeeded after {} retries in {:.2}ms",
                        attempt,
                        start_time.elapsed().as_millis() as f64
                    );
                }
                return Ok(result);
            }
            Err(e) => {
                last_error = Some(e.clone());

                if !is_retryable_error(&e, &config) {
                    tracing::warn!(
                        "Non-retryable error on attempt {}: {}",
                        attempt + 1,
                        e.to_string()
                    );
                    return Err(e);
                }

                if attempt < config.max_retries {
                    let delay = calculate_delay(attempt, &config);
                    tracing::warn!(
                        "Retryable error on attempt {}: {}. Retrying in {:.2}ms...",
                        attempt + 1,
                        e.to_string(),
                        delay.as_millis() as f64
                    );
                    sleep(delay).await;
                } else {
                    tracing::error!(
                        "Operation failed after {} attempts in {:.2}ms: {}",
                        attempt + 1,
                        start_time.elapsed().as_millis() as f64,
                        e.to_string()
                    );
                    return Err(e);
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        AppError::inference_error("Unknown retry error".to_string())
    }))
}

fn is_retryable_error(error: &AppError, config: &RetryConfig) -> bool {
    let error_msg = error.to_string();

    for pattern in &config.retryable_errors {
        if error_msg.contains(pattern) {
            return true;
        }
    }

    matches!(
        error,
        AppError::InferenceError(_) | AppError::TokenizationError(_)
    )
}

fn calculate_delay(attempt: u32, config: &RetryConfig) -> Duration {
    let delay_ms = (config.initial_delay_ms as f64)
        .mul_add(config.exponent_base.powi(attempt as i32), 0.0)
        .min(config.max_delay_ms as f64);

    let jitter_ms = (delay_ms * 0.1 * (Instant::now().elapsed().subsec_nanos() as f64 / u32::MAX as f64)) as u64;
    let total_ms = (delay_ms as u64).saturating_add(jitter_ms);

    Duration::from_millis(total_ms)
}
