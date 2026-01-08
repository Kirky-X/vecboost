// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 限流存储模块
//!
//! 提供限流数据的存储接口和实现

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// 限流存储接口
#[async_trait::async_trait]
pub trait RateLimitStore: Send + Sync {
    /// 检查并增加计数器
    async fn check_and_increment(&self, key: &str, window_secs: u64, max_requests: u64) -> bool;

    /// 重置计数器
    async fn reset(&self, key: &str);

    /// 获取当前计数
    async fn get_count(&self, key: &str) -> u64;
}

/// 内存限流存储
pub struct MemoryRateLimitStore {
    counters: Arc<AsyncMutex<HashMap<String, (u64, u64)>>>, // (count, window_start)
}

impl MemoryRateLimitStore {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(AsyncMutex::new(HashMap::new())),
        }
    }

    /// 清理过期的计数器
    pub async fn cleanup_expired(&self, window_secs: u64) {
        let mut counters = self.counters.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        counters.retain(|_, window_start| (now - window_start.1) < window_secs);
    }
}

#[async_trait::async_trait]
impl RateLimitStore for MemoryRateLimitStore {
    async fn check_and_increment(&self, key: &str, window_secs: u64, max_requests: u64) -> bool {
        let mut counters = self.counters.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let (count, window_start) = counters.entry(key.to_string()).or_insert((0, now));

        // 检查窗口是否过期
        if now - *window_start >= window_secs {
            *count = 0;
            *window_start = now;
        }

        // 检查是否超过限制
        if *count >= max_requests {
            false
        } else {
            *count += 1;
            true
        }
    }

    async fn reset(&self, key: &str) {
        let mut counters = self.counters.lock().await;
        counters.remove(key);
    }

    async fn get_count(&self, key: &str) -> u64 {
        let counters = self.counters.lock().await;
        counters.get(key).map(|(count, _)| *count).unwrap_or(0)
    }
}

impl Default for MemoryRateLimitStore {
    fn default() -> Self {
        Self::new()
    }
}
