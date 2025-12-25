// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_ms: u64,
    pub volume_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_ms: 30000,
            volume_threshold: 10,
        }
    }
}

impl CircuitBreakerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

#[derive(Debug)]
pub struct CircuitBreaker {
    name: String,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
    total_calls: Arc<AtomicUsize>,
    total_failures: Arc<AtomicUsize>,
    total_successes: Arc<AtomicUsize>,
}

impl CircuitBreaker {
    pub fn new(name: &str, config: Option<CircuitBreakerConfig>) -> Self {
        Self {
            name: name.to_string(),
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            config: config.unwrap_or_default(),
            total_calls: Arc::new(AtomicUsize::new(0)),
            total_failures: Arc::new(AtomicUsize::new(0)),
            total_successes: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
        E: std::fmt::Display + From<String>,
    {
        let current_state = self.state.read().await.clone();

        if current_state == CircuitState::Open {
            if self.should_attempt_reset().await {
                *self.state.write().await = CircuitState::HalfOpen;
                debug!("Circuit breaker '{}' moved to HalfOpen state", self.name);
            } else {
                return Err(format!(
                    "Circuit breaker '{}' is open. Retry after timeout.",
                    self.name
                )
                .into());
            }
        }

        let start = Instant::now();
        self.total_calls.fetch_add(1, Ordering::SeqCst);

        match operation.await {
            Ok(result) => {
                self.on_success().await;
                let elapsed = start.elapsed();
                debug!(
                    "Circuit breaker '{}' call succeeded in {:.2}ms",
                    self.name,
                    elapsed.as_millis() as f64
                );
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                let elapsed = start.elapsed();
                warn!(
                    "Circuit breaker '{}' call failed in {:.2}ms: {}",
                    self.name,
                    elapsed.as_millis() as f64,
                    e
                );
                Err(e)
            }
        }
    }

    async fn on_success(&self) {
        self.success_count.fetch_add(1, Ordering::SeqCst);
        self.total_successes.fetch_add(1, Ordering::SeqCst);

        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitState::HalfOpen => {
                let success_count = self.success_count.load(Ordering::SeqCst);
                if success_count >= self.config.success_threshold as usize {
                    self.reset().await;
                    info!("Circuit breaker '{}' closed after successful half-open checks", self.name);
                }
            }
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            CircuitState::Open => {}
        }
    }

    async fn on_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::SeqCst);
        self.total_failures.fetch_add(1, Ordering::SeqCst);
        *self.last_failure_time.write().await = Some(Instant::now());

        let failure_count = self.failure_count.load(Ordering::SeqCst);
        let total_calls = self.total_calls.load(Ordering::SeqCst);

        let should_open = failure_count >= self.config.failure_threshold as usize
            || (total_calls >= self.config.volume_threshold as usize
                && failure_count as f64 / total_calls as f64 > 0.5);

        if should_open {
            *self.state.write().await = CircuitState::Open;
            self.failure_count.store(0, Ordering::SeqCst);
            self.success_count.store(0, Ordering::SeqCst);
            warn!(
                "Circuit breaker '{}' opened after {} failures",
                self.name, failure_count
            );
        }
    }

    async fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = *self.last_failure_time.read().await {
            let timeout = Duration::from_millis(self.config.timeout_ms);
            Instant::now().duration_since(last_failure) >= timeout
        } else {
            true
        }
    }

    async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        *self.last_failure_time.write().await = None;
    }

    pub async fn state(&self) -> CircuitState {
        self.state.read().await.clone()
    }

    pub fn stats(&self) -> CircuitBreakerStats {
        let state = self.state.blocking_read().clone();
        CircuitBreakerStats {
            name: self.name.clone(),
            state,
            total_calls: self.total_calls.load(Ordering::SeqCst),
            total_successes: self.total_successes.load(Ordering::SeqCst),
            total_failures: self.total_failures.load(Ordering::SeqCst),
            failure_rate: if self.total_calls.load(Ordering::SeqCst) > 0 {
                self.total_failures.load(Ordering::SeqCst) as f64
                    / self.total_calls.load(Ordering::SeqCst) as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub name: String,
    pub state: CircuitState,
    pub total_calls: usize,
    pub total_successes: usize,
    pub total_failures: usize,
    pub failure_rate: f64,
}

impl CircuitBreakerStats {
    pub fn is_open(&self) -> bool {
        self.state == CircuitState::Open
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_calls > 0 {
            self.total_successes as f64 / self.total_calls as f64
        } else {
            0.0
        }
    }
}
