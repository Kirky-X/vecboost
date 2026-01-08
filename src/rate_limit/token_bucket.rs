// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 令牌桶限流算法实现

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info};

fn current_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// 令牌桶配置
#[derive(Debug, Clone)]
pub struct TokenBucketConfig {
    pub capacity: u64,
    pub refill_rate: u64,
    pub allow_burst: bool,
    pub min_refill_interval_ms: u64,
}

impl Default for TokenBucketConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            refill_rate: 10,
            allow_burst: true,
            min_refill_interval_ms: 100,
        }
    }
}

impl TokenBucketConfig {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            refill_rate,
            allow_burst: true,
            min_refill_interval_ms: 100,
        }
    }

    pub fn strict(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity,
            refill_rate,
            allow_burst: false,
            min_refill_interval_ms: 100,
        }
    }

    pub fn from_requests_per_minute(requests_per_minute: u64) -> Self {
        let capacity = requests_per_minute;
        let refill_rate = (requests_per_minute / 60).max(1);
        Self::new(capacity.max(1), refill_rate)
    }
}

/// 令牌桶状态
#[derive(Debug, Clone, Default)]
pub struct TokenBucketState {
    pub tokens: u64,
    pub last_refill: u64,
    pub total_requests: u64,
    pub denied_requests: u64,
}

/// 令牌桶
pub struct TokenBucket {
    config: TokenBucketConfig,
    tokens: AtomicU64,
    last_refill: AtomicU64,
    total_requests: AtomicU64,
    denied_requests: AtomicU64,
    name: String,
}

impl TokenBucket {
    pub fn new(name: &str, config: TokenBucketConfig) -> Self {
        let epoch = current_epoch_secs();
        Self {
            config: config.clone(),
            tokens: AtomicU64::new(config.capacity),
            last_refill: AtomicU64::new(epoch),
            total_requests: AtomicU64::new(0),
            denied_requests: AtomicU64::new(0),
            name: name.to_string(),
        }
    }

    pub async fn try_acquire(&self, tokens_needed: u64) -> (bool, u64, u64) {
        let current_time = current_epoch_secs();

        loop {
            let last_refill = self.last_refill.load(Ordering::SeqCst);
            let elapsed_secs = current_time.saturating_sub(last_refill);
            let intervals = elapsed_secs * 1000 / self.config.min_refill_interval_ms as u64;

            let mut current_tokens = self.tokens.load(Ordering::SeqCst);

            if intervals > 0 {
                let tokens_to_add = intervals * self.config.refill_rate;
                current_tokens = std::cmp::min(
                    current_tokens.saturating_add(tokens_to_add),
                    self.config.capacity,
                );
            }

            if current_tokens >= tokens_needed {
                match self.tokens.compare_exchange(
                    current_tokens,
                    current_tokens.saturating_sub(tokens_needed),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        self.last_refill.store(current_time, Ordering::SeqCst);
                        self.total_requests.fetch_add(1, Ordering::SeqCst);
                        return (true, current_tokens.saturating_sub(tokens_needed), 0);
                    }
                    Err(_) => continue,
                }
            } else if self.config.allow_burst {
                match self.tokens.compare_exchange(
                    current_tokens,
                    0,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        self.last_refill.store(current_time, Ordering::SeqCst);
                        self.total_requests.fetch_add(1, Ordering::SeqCst);
                        return (true, 0, 0);
                    }
                    Err(_) => continue,
                }
            } else {
                self.denied_requests.fetch_add(1, Ordering::SeqCst);
                let wait_time = self.calculate_wait_time(tokens_needed - current_tokens);
                return (false, current_tokens, wait_time);
            }
        }
    }

    fn calculate_wait_time(&self, needed_tokens: u64) -> u64 {
        if self.config.refill_rate == 0 {
            return u64::MAX;
        }
        let wait_secs = needed_tokens.saturating_div(self.config.refill_rate);
        let wait_ms = wait_secs * 1000;
        let remainder = needed_tokens % self.config.refill_rate;
        let remainder_ms = if remainder > 0 {
            self.config.min_refill_interval_ms
        } else {
            0
        };
        wait_ms + remainder_ms
    }

    pub async fn allows(&self, tokens: u64) -> bool {
        self.try_acquire(tokens).await.0
    }

    pub async fn state(&self) -> TokenBucketState {
        TokenBucketState {
            tokens: self.tokens.load(Ordering::SeqCst),
            last_refill: self.last_refill.load(Ordering::SeqCst),
            total_requests: self.total_requests.load(Ordering::SeqCst),
            denied_requests: self.denied_requests.load(Ordering::SeqCst),
        }
    }

    pub async fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::SeqCst)
    }

    pub async fn reset(&self) {
        self.tokens.store(self.config.capacity, Ordering::SeqCst);
        self.last_refill
            .store(current_epoch_secs(), Ordering::SeqCst);
        self.total_requests.store(0, Ordering::SeqCst);
        self.denied_requests.store(0, Ordering::SeqCst);
        info!("Token bucket '{}' reset", self.name);
    }
}

#[async_trait::async_trait]
pub trait TokenBucketStore: Send + Sync {
    async fn get_bucket(&self, key: &str) -> Arc<TokenBucket>;
    async fn remove_bucket(&self, key: &str);
    async fn cleanup_expired(&self, max_idle_secs: u64);
    async fn get_all_states(&self) -> HashMap<String, TokenBucketState>;
}

pub struct TokenBucketStoreImpl {
    buckets: Arc<AsyncMutex<HashMap<String, (Arc<TokenBucket>, Instant)>>>,
    default_config: TokenBucketConfig,
}

impl TokenBucketStoreImpl {
    pub fn new(default_config: TokenBucketConfig) -> Self {
        Self {
            buckets: Arc::new(AsyncMutex::new(HashMap::new())),
            default_config,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(TokenBucketConfig::default())
    }
}

#[async_trait::async_trait]
impl TokenBucketStore for TokenBucketStoreImpl {
    async fn get_bucket(&self, key: &str) -> Arc<TokenBucket> {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();

        if let Some((bucket, _)) = buckets.get(key) {
            return bucket.clone();
        }

        let bucket = Arc::new(TokenBucket::new(key, self.default_config.clone()));
        buckets.insert(key.to_string(), (bucket.clone(), now));
        info!("Created new token bucket for key: {}", key);
        bucket
    }

    async fn remove_bucket(&self, key: &str) {
        let mut buckets = self.buckets.lock().await;
        buckets.remove(key);
        debug!("Removed token bucket for key: {}", key);
    }

    async fn cleanup_expired(&self, max_idle_secs: u64) {
        let mut buckets = self.buckets.lock().await;
        let before = buckets.len();
        let now = Instant::now();

        buckets.retain(|_, (_, last_used)| {
            now.duration_since(*last_used) < Duration::from_secs(max_idle_secs)
        });

        let after = buckets.len();
        if before != after {
            info!("Cleaned up {} expired token buckets", before - after);
        }
    }

    async fn get_all_states(&self) -> HashMap<String, TokenBucketState> {
        let buckets = self.buckets.lock().await;
        let mut states = HashMap::new();
        for (key, (bucket, _)) in buckets.iter() {
            states.insert(key.clone(), bucket.state().await);
        }
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let config = TokenBucketConfig::strict(10, 2);
        let bucket = TokenBucket::new("test", config);

        assert_eq!(bucket.tokens().await, 10);

        let (success, tokens, _) = bucket.try_acquire(5).await;
        assert!(success);
        assert_eq!(tokens, 5);

        let (success, tokens, _) = bucket.try_acquire(5).await;
        assert!(success);
        assert_eq!(tokens, 0);

        let (success, tokens, _) = bucket.try_acquire(1).await;
        assert!(!success);
        assert_eq!(tokens, 0);
    }

    #[tokio::test]
    async fn test_token_bucket_with_burst() {
        let config = TokenBucketConfig::new(10, 2);
        let bucket = TokenBucket::new("test_burst", config);

        assert!(bucket.allows(10).await);
        assert_eq!(bucket.tokens().await, 0);

        assert!(bucket.allows(1).await);
        assert_eq!(bucket.tokens().await, 0);
    }

    #[tokio::test]
    async fn test_token_bucket_store() {
        let store = TokenBucketStoreImpl::with_default_config();

        let bucket1 = store.get_bucket("user:123").await;
        let bucket2 = store.get_bucket("user:123").await;
        assert!(Arc::ptr_eq(&bucket1, &bucket2));

        let bucket3 = store.get_bucket("user:456").await;
        assert!(!Arc::ptr_eq(&bucket1, &bucket3));
    }
}
