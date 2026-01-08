// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! Redis 限流存储模块
//!
//! 支持分布式环境下的限流，使用 Redis 作为后端存储
//! 仅在启用 `redis` feature 时编译

#[cfg(feature = "redis")]
use redis::aio::ConnectionManager;

#[cfg(feature = "redis")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "redis")]
use std::sync::Arc;

#[cfg(feature = "redis")]
use tokio::sync::Mutex;

/// Redis 连接配置
#[cfg(feature = "redis")]
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub prefix: String,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            prefix: "vecboost".to_string(),
        }
    }
}

impl RedisConfig {
    pub fn new(url: String) -> Self {
        Self {
            url,
            prefix: "vecboost".to_string(),
        }
    }

    pub fn with_prefix(url: String, prefix: String) -> Self {
        Self { url, prefix }
    }
}

/// Redis 计数器值
#[cfg(feature = "redis")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CounterValue {
    count: u64,
    window_start: u64,
}

/// Redis 限流存储
#[cfg(feature = "redis")]
pub struct RedisRateLimitStore {
    manager: Arc<Mutex<ConnectionManager>>,
    config: RedisConfig,
}

#[cfg(feature = "redis")]
impl RedisRateLimitStore {
    pub async fn new(config: RedisConfig) -> Result<Self, String> {
        let client = redis::Client::open(config.url.clone())
            .map_err(|e| format!("Failed to create Redis client: {}", e))?;
        let manager = client
            .get_connection_manager()
            .await
            .map_err(|e| format!("Failed to create connection manager: {}", e))?;
        Ok(Self {
            manager: Arc::new(Mutex::new(manager)),
            config,
        })
    }

    fn make_key(&self, key: &str) -> String {
        format!("{}:{}", self.config.prefix, key)
    }
}

#[cfg(feature = "redis")]
#[async_trait::async_trait]
impl super::RateLimitStore for RedisRateLimitStore {
    async fn check_and_increment(&self, key: &str, window_secs: u64, max_requests: u64) -> bool {
        let full_key = self.make_key(key);
        let mut conn = self.manager.lock().await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let json: Option<String> = redis::cmd("GET")
            .arg(&full_key)
            .query_async(&mut *conn)
            .await
            .ok()
            .flatten();

        let mut should_allow = true;

        if let Some(json) = json {
            if let Ok(value) = serde_json::from_str::<CounterValue>(&json) {
                if now - value.window_start >= window_secs {
                    should_allow = true;
                } else if value.count >= max_requests {
                    should_allow = false;
                } else {
                    let new_value = CounterValue {
                        count: value.count + 1,
                        window_start: value.window_start,
                    };
                    let json_str = serde_json::to_string(&new_value).unwrap_or_default();
                    let _ = redis::cmd("SET")
                        .arg(&full_key)
                        .arg(&json_str)
                        .query_async::<()>(&mut *conn)
                        .await;
                    should_allow = true;
                }
            }
        }

        if should_allow {
            let new_value = CounterValue {
                count: 1,
                window_start: now,
            };
            let json_str = serde_json::to_string(&new_value).unwrap_or_default();
            let _ = redis::cmd("SETEX")
                .arg(&full_key)
                .arg(window_secs)
                .arg(&json_str)
                .query_async::<()>(&mut *conn)
                .await;
        }

        should_allow
    }

    async fn reset(&self, key: &str) {
        let full_key = self.make_key(key);
        let mut conn = self.manager.lock().await;
        let _: () = redis::cmd("DEL")
            .arg(&full_key)
            .query_async(&mut *conn)
            .await
            .unwrap_or_default();
    }

    async fn get_count(&self, key: &str) -> u64 {
        let full_key = self.make_key(key);
        let mut conn = self.manager.lock().await;
        let json: Option<String> = redis::cmd("GET")
            .arg(&full_key)
            .query_async(&mut *conn)
            .await
            .ok()
            .flatten();
        if let Some(json) = json {
            if let Ok(value) = serde_json::from_str::<CounterValue>(&json) {
                return value.count;
            }
        }
        0
    }
}

#[cfg(feature = "redis")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.prefix, "vecboost");
    }

    #[test]
    fn test_redis_config_new() {
        let config = RedisConfig::new("redis://localhost:6379".to_string());
        assert_eq!(config.url, "redis://localhost:6379");
    }
}
