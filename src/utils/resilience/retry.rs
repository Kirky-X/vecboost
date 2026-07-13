// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
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
    type Future: Future<Output = Result<T, VecboostError>>;
    fn call(&self) -> Self::Future;
}

impl<F, T, Fut> Retryable<T> for F
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, VecboostError>>,
{
    type Future = Fut;
    fn call(&self) -> Self::Future {
        (self)()
    }
}

pub async fn with_retry<T, R: Retryable<T>>(
    retryable: &R,
    config: Option<RetryConfig>,
) -> Result<T, VecboostError> {
    let config = config.unwrap_or_default();
    let mut last_error: Option<VecboostError> = None;
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

    Err(last_error
        .unwrap_or_else(|| VecboostError::inference_error("Unknown retry error".to_string())))
}

fn is_retryable_error(error: &VecboostError, config: &RetryConfig) -> bool {
    let error_msg = error.to_string();

    for pattern in &config.retryable_errors {
        if error_msg.contains(pattern) {
            return true;
        }
    }

    matches!(
        error,
        VecboostError::InferenceError(_) | VecboostError::TokenizationError(_)
    )
}

fn calculate_delay(attempt: u32, config: &RetryConfig) -> Duration {
    let delay_ms = (config.initial_delay_ms as f64)
        .mul_add(config.exponent_base.powi(attempt as i32), 0.0)
        .min(config.max_delay_ms as f64);

    let jitter_ms =
        (delay_ms * 0.1 * (Instant::now().elapsed().subsec_nanos() as f64 / u32::MAX as f64))
            as u64;
    let total_ms = (delay_ms as u64).saturating_add(jitter_ms);

    Duration::from_millis(total_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 5000);
        assert_eq!(config.exponent_base, 2.0);
        assert!(config.include_error_type);
        assert!(!config.retryable_errors.is_empty());
    }

    #[test]
    fn test_retry_config_new_equals_default() {
        let new_config = RetryConfig::new();
        let default_config = RetryConfig::default();
        assert_eq!(new_config.max_retries, default_config.max_retries);
        assert_eq!(new_config.initial_delay_ms, default_config.initial_delay_ms);
    }

    #[test]
    fn test_retry_config_builders() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_initial_delay(50)
            .with_max_delay(2000)
            .with_exponent_base(3.0);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay_ms, 50);
        assert_eq!(config.max_delay_ms, 2000);
        assert_eq!(config.exponent_base, 3.0);
    }

    #[tokio::test]
    async fn test_with_retry_succeeds_immediately() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let closure = move || {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok::<i32, VecboostError>(42)
            }
        };

        let result = with_retry(&closure, None).await.unwrap();
        assert_eq!(result, 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_succeeds_after_retries() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let config = RetryConfig::new().with_max_retries(3).with_initial_delay(1);
        let closure = move || {
            let count = count_clone.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(VecboostError::inference_error(
                        "Inference error: temporary".to_string(),
                    ))
                } else {
                    Ok(99)
                }
            }
        };

        let result = with_retry(&closure, Some(config)).await.unwrap();
        assert_eq!(result, 99);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_non_retryable_error() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let closure = move || {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(VecboostError::ConfigError("config issue".to_string()))
            }
        };

        let result = with_retry(&closure, None).await;
        assert!(result.is_err());
        // Should not retry non-retryable errors
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_exhausts_retries() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let config = RetryConfig::new().with_max_retries(2).with_initial_delay(1);
        let closure = move || {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(VecboostError::inference_error(
                    "Inference error".to_string(),
                ))
            }
        };

        let result = with_retry(&closure, Some(config)).await;
        assert!(result.is_err());
        // Initial attempt + 2 retries = 3 calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_zero_retries() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count_clone = call_count.clone();
        let config = RetryConfig::new().with_max_retries(0).with_initial_delay(1);
        let closure = move || {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, _>(VecboostError::inference_error(
                    "Inference error".to_string(),
                ))
            }
        };

        let result = with_retry(&closure, Some(config)).await;
        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_is_retryable_error_inference() {
        let config = RetryConfig::default();
        let err = VecboostError::inference_error("Inference error".to_string());
        assert!(is_retryable_error(&err, &config));
    }

    #[test]
    fn test_is_retryable_error_tokenization() {
        let config = RetryConfig::default();
        let err = VecboostError::TokenizationError("Tokenization error".to_string());
        assert!(is_retryable_error(&err, &config));
    }

    #[test]
    fn test_is_retryable_error_config_not_retryable() {
        let config = RetryConfig::default();
        let err = VecboostError::ConfigError("config issue".to_string());
        assert!(!is_retryable_error(&err, &config));
    }

    #[test]
    fn test_is_retryable_error_pattern_match() {
        let config = RetryConfig::default();
        // "timeout" is in default retryable_errors
        let err = VecboostError::InternalError("request timeout".to_string());
        assert!(is_retryable_error(&err, &config));
    }

    #[test]
    fn test_calculate_delay_within_bounds() {
        let config = RetryConfig::new()
            .with_initial_delay(100)
            .with_max_delay(1000)
            .with_exponent_base(2.0);

        let delay0 = calculate_delay(0, &config);
        let delay1 = calculate_delay(1, &config);
        let delay2 = calculate_delay(2, &config);

        // Base delay (without jitter) for attempt 0 should be 100ms
        assert!(delay0.as_millis() >= 100);
        // Base delay for attempt 1 should be 200ms
        assert!(delay1.as_millis() >= 200);
        // Base delay for attempt 2 should be 400ms
        assert!(delay2.as_millis() >= 400);
        // All delays should respect max_delay (with jitter)
        assert!(delay2.as_millis() <= 1100);
    }

    #[test]
    fn test_calculate_delay_respects_max() {
        let config = RetryConfig::new()
            .with_initial_delay(100)
            .with_max_delay(500)
            .with_exponent_base(10.0);

        // Even with large exponent, delay should not exceed max significantly
        let delay = calculate_delay(5, &config);
        assert!(delay.as_millis() <= 600); // max + 10% jitter
    }
}
