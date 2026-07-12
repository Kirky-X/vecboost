// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! 限流存储模块
//!
//! 提供限流数据的存储接口和实现，支持滑动窗口算法

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// 滑动窗口计数器结构
#[derive(Debug, Clone)]
struct SlidingWindowCounter {
    /// 当前窗口的请求计数
    current_window_count: u64,
    /// 上一个窗口的请求计数
    previous_window_count: u64,
    /// 当前窗口的开始时间（秒）
    current_window_start: u64,
}

impl SlidingWindowCounter {
    fn new(now: u64) -> Self {
        Self {
            current_window_count: 0,
            previous_window_count: 0,
            current_window_start: now,
        }
    }

    /// 获取加权后的请求数（滑动窗口核心算法）
    fn get_weighted_count(&self, now: u64, window_secs: u64) -> f64 {
        let elapsed = (now - self.current_window_start) as f64;
        let window_weight = 1.0 - (elapsed / window_secs as f64);

        // 加权计数 = 上一窗口计数 * 权重 + 当前窗口计数
        self.previous_window_count as f64 * window_weight + self.current_window_count as f64
    }

    /// 更新窗口（如果需要）
    fn update_window(&mut self, now: u64, window_secs: u64) {
        let elapsed = now - self.current_window_start;

        if elapsed >= window_secs {
            // 窗口已过期，移动窗口
            let windows_passed = elapsed / window_secs;
            if windows_passed == 1 {
                // 只过了一个窗口，保留上一窗口的数据
                self.previous_window_count = self.current_window_count;
            } else {
                // 过了多个窗口，清空历史
                self.previous_window_count = 0;
            }

            self.current_window_count = 0;
            self.current_window_start = self.current_window_start + windows_passed * window_secs;
        }
    }

    /// 增加计数
    fn increment(&mut self) {
        self.current_window_count += 1;
    }
}

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

/// 内存限流存储（滑动窗口实现）
pub struct MemoryRateLimitStore {
    counters: Arc<AsyncMutex<HashMap<String, SlidingWindowCounter>>>,
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

        // 移除超过 2 个窗口时间的计数器
        counters.retain(|_, counter| (now - counter.current_window_start) < window_secs * 2);
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

        let counter = counters
            .entry(key.to_string())
            .or_insert_with(|| SlidingWindowCounter::new(now));

        // 更新窗口
        counter.update_window(now, window_secs);

        // 获取加权后的请求数（滑动窗口核心）
        let weighted_count = counter.get_weighted_count(now, window_secs);

        // 检查是否超过限制
        if weighted_count >= max_requests as f64 {
            false
        } else {
            counter.increment();
            true
        }
    }

    async fn reset(&self, key: &str) {
        let mut counters = self.counters.lock().await;
        counters.remove(key);
    }

    async fn get_count(&self, key: &str) -> u64 {
        let counters = self.counters.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        counters
            .get(key)
            .map(|counter| counter.get_weighted_count(now, 60).round() as u64)
            .unwrap_or(0)
    }
}

impl Default for MemoryRateLimitStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sliding_window_basic() {
        let store = MemoryRateLimitStore::new();
        let window_secs = 60; // 1 分钟
        let max_requests = 10;

        // 前 10 个请求应该都允许
        for i in 0..10 {
            assert!(
                store
                    .check_and_increment("test_key", window_secs, max_requests)
                    .await
            );
        }

        // 第 11 个请求应该被拒绝
        assert!(
            !store
                .check_and_increment("test_key", window_secs, max_requests)
                .await
        );
    }

    #[tokio::test]
    async fn test_sliding_window_weighted_count() {
        let store = MemoryRateLimitStore::new();
        let window_secs = 2; // 2 秒窗口（用于快速测试）
        let max_requests = 5;

        // 发送 3 个请求
        for _ in 0..3 {
            assert!(
                store
                    .check_and_increment("weighted_test", window_secs, max_requests)
                    .await
            );
        }

        // 等待半个窗口时间（1 秒）
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // 由于滑动窗口，加权计数应该约为 3 * 0.5 = 1.5
        // 所以还能接受新的请求
        assert!(
            store
                .check_and_increment("weighted_test", window_secs, max_requests)
                .await
        );
    }

    #[tokio::test]
    async fn test_sliding_window_cleanup() {
        let store = MemoryRateLimitStore::new();
        let window_secs = 1; // 1 秒窗口
        let max_requests = 5;

        // 发送一些请求
        for _ in 0..3 {
            assert!(
                store
                    .check_and_increment("cleanup_test", window_secs, max_requests)
                    .await
            );
        }

        // 等待超过 2 个窗口时间
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        // 清理过期计数器
        store.cleanup_expired(window_secs).await;

        // 计数器应该已被清理，可以重新开始计数
        assert!(
            store
                .check_and_increment("cleanup_test", window_secs, max_requests)
                .await
        );
    }
}
