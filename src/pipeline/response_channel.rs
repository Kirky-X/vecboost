// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, oneshot};
use tracing::{debug, warn};

use crate::domain::EmbedResponse;
use crate::error::AppError;

/// 待处理的响应
pub struct PendingResponse {
    /// 响应发送器
    pub tx: oneshot::Sender<Result<EmbedResponse, AppError>>,
    /// 提交时间
    pub submitted_at: Instant,
    /// 超时时间
    pub timeout: Duration,
}

/// 响应通道
pub struct ResponseChannel {
    /// 待处理的响应
    pending: Arc<RwLock<HashMap<String, PendingResponse>>>,
    /// 默认超时时间
    default_timeout: Duration,
}

impl ResponseChannel {
    /// 创建新的响应通道（使用默认超时 30 秒）
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }

    /// 创建新的响应通道（使用指定的超时时间）
    pub fn with_timeout(timeout: Duration) -> Self {
        debug!("Creating ResponseChannel with timeout={:?}", timeout);
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            default_timeout: timeout,
        }
    }

    /// 获取默认超时时间
    pub fn get_default_timeout(&self) -> Duration {
        self.default_timeout
    }

    /// 注册响应
    pub async fn register(
        &self,
        request_id: String,
    ) -> oneshot::Receiver<Result<EmbedResponse, AppError>> {
        let (tx, rx) = oneshot::channel();

        let pending = PendingResponse {
            tx,
            submitted_at: Instant::now(),
            timeout: self.default_timeout,
        };

        self.pending
            .write()
            .await
            .insert(request_id.clone(), pending);

        debug!("Registered response for request {}", request_id);

        rx
    }

    /// 完成响应
    pub async fn complete(&self, request_id: String, response: Result<EmbedResponse, AppError>) {
        let mut pending = self.pending.write().await;

        if let Some(pending_response) = pending.remove(&request_id) {
            let elapsed = pending_response.submitted_at.elapsed();

            match pending_response.tx.send(response) {
                Ok(_) => {
                    debug!(
                        "Response sent for request {}, elapsed: {:?}",
                        request_id, elapsed
                    );
                }
                Err(_) => {
                    warn!(
                        "Failed to send response for request {}, receiver dropped",
                        request_id
                    );
                }
            }
        } else {
            warn!("No pending response found for request {}", request_id);
        }
    }

    /// 清理超时的响应
    pub async fn cleanup_expired(&self) {
        let mut pending = self.pending.write().await;

        let now = Instant::now();
        let mut expired = Vec::new();

        for (request_id, pending_response) in pending.iter() {
            if now.duration_since(pending_response.submitted_at) > pending_response.timeout {
                expired.push(request_id.clone());
            }
        }

        for request_id in &expired {
            pending.remove(request_id);
            warn!("Removed expired response for request {}", request_id);
        }

        if !expired.is_empty() {
            debug!("Cleaned up {} expired responses", expired.len());
        }
    }

    /// 获取待处理的响应数量
    pub async fn pending_count(&self) -> usize {
        self.pending.read().await.len()
    }

    /// 清空所有待处理的响应
    pub async fn clear(&self) {
        let mut pending = self.pending.write().await;
        let count = pending.len();
        pending.clear();
        warn!("Cleared {} pending responses", count);
    }
}

impl Default for ResponseChannel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_channel_creation() {
        let channel = ResponseChannel::new();
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_register_complete() {
        let channel = ResponseChannel::new();

        let rx = channel.register("test-1".to_string()).await;
        assert_eq!(channel.pending_count().await, 1);

        let response = Ok(EmbedResponse {
            embedding: vec![0.0; 768],
            dimension: 768,
            processing_time_ms: 100,
        });

        channel.complete("test-1".to_string(), response).await;

        let result = rx.await;
        assert!(result.is_ok());
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_clear() {
        let channel = ResponseChannel::new();

        // 注册多个响应
        for i in 0..10 {
            channel.register(format!("test-{}", i)).await;
        }

        assert_eq!(channel.pending_count().await, 10);

        channel.clear().await;

        assert_eq!(channel.pending_count().await, 0);
    }
}
