// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, oneshot};
use tracing::{debug, warn};

use crate::domain::EmbedResponse;
use crate::error::VecboostError;

/// 待处理的响应
pub struct PendingResponse {
    /// 响应发送器
    pub tx: oneshot::Sender<Result<EmbedResponse, VecboostError>>,
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
    ) -> oneshot::Receiver<Result<EmbedResponse, VecboostError>> {
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
    pub async fn complete(
        &self,
        request_id: String,
        response: Result<EmbedResponse, VecboostError>,
    ) {
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

    #[tokio::test]
    async fn test_with_timeout_and_get_default_timeout() {
        let channel = ResponseChannel::with_timeout(Duration::from_secs(10));
        assert_eq!(channel.get_default_timeout(), Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_default_uses_30_secs_timeout() {
        let channel = ResponseChannel::default();
        assert_eq!(channel.get_default_timeout(), Duration::from_secs(30));
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_complete_with_error_response_propagates() {
        let channel = ResponseChannel::new();
        let rx = channel.register("err-1".to_string()).await;

        let error_response: Result<EmbedResponse, VecboostError> =
            Err(VecboostError::InferenceError("boom".to_string()));

        channel.complete("err-1".to_string(), error_response).await;

        let result = rx.await;
        assert!(result.is_ok(), "receiver should not be dropped");
        match result.unwrap() {
            Err(VecboostError::InferenceError(msg)) => {
                assert!(msg.contains("boom"));
            }
            other => panic!("expected InferenceError, got {:?}", other),
        }
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_complete_unknown_request_id_is_noop() {
        let channel = ResponseChannel::new();
        channel
            .complete(
                "unknown".to_string(),
                Ok(EmbedResponse {
                    embedding: vec![0.0; 4],
                    dimension: 4,
                    processing_time_ms: 1,
                }),
            )
            .await;
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_complete_after_receiver_dropped_logs_warning() {
        let channel = ResponseChannel::new();
        {
            let _rx = channel.register("drop-1".to_string()).await;
        }
        assert_eq!(channel.pending_count().await, 1);

        channel
            .complete(
                "drop-1".to_string(),
                Ok(EmbedResponse {
                    embedding: vec![0.0; 4],
                    dimension: 4,
                    processing_time_ms: 1,
                }),
            )
            .await;
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_cleanup_expired_removes_timed_out_entries() {
        let channel = ResponseChannel::with_timeout(Duration::from_millis(10));

        channel.register("expired-1".to_string()).await;
        channel.register("expired-2".to_string()).await;
        assert_eq!(channel.pending_count().await, 2);

        tokio::time::sleep(Duration::from_millis(50)).await;

        channel.cleanup_expired().await;
        assert_eq!(
            channel.pending_count().await,
            0,
            "expired entries must be cleaned up"
        );
    }

    #[tokio::test]
    async fn test_cleanup_expired_keeps_unexpired_entries() {
        let channel = ResponseChannel::with_timeout(Duration::from_secs(60));

        channel.register("alive-1".to_string()).await;
        channel.register("alive-2".to_string()).await;
        assert_eq!(channel.pending_count().await, 2);

        channel.cleanup_expired().await;
        assert_eq!(
            channel.pending_count().await,
            2,
            "unexpired entries must be kept"
        );
    }

    #[tokio::test]
    async fn test_cleanup_expired_on_empty_channel_is_noop() {
        let channel = ResponseChannel::new();
        assert_eq!(channel.pending_count().await, 0);
        channel.cleanup_expired().await;
        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_concurrent_register_and_complete() {
        let channel = std::sync::Arc::new(ResponseChannel::new());
        let mut handles = Vec::new();

        for i in 0..20 {
            let channel_clone = std::sync::Arc::clone(&channel);
            let request_id = format!("conc-{}", i);
            handles.push(tokio::spawn(async move {
                let rx = channel_clone.register(request_id.clone()).await;
                let response = Ok(EmbedResponse {
                    embedding: vec![0.0; 8],
                    dimension: 8,
                    processing_time_ms: 1,
                });
                channel_clone.complete(request_id, response).await;
                rx.await
            }));
        }

        for handle in handles {
            let result = handle.await.expect("task must not panic");
            assert!(result.is_ok(), "receiver should receive Ok response");
            assert!(result.unwrap().is_ok());
        }

        assert_eq!(channel.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_register_replaces_same_request_id() {
        let channel = ResponseChannel::new();

        let _rx1 = channel.register("dup-1".to_string()).await;
        assert_eq!(channel.pending_count().await, 1);

        let rx2 = channel.register("dup-1".to_string()).await;
        assert_eq!(channel.pending_count().await, 1);

        channel
            .complete(
                "dup-1".to_string(),
                Ok(EmbedResponse {
                    embedding: vec![1.0; 4],
                    dimension: 4,
                    processing_time_ms: 0,
                }),
            )
            .await;

        let result = rx2.await;
        assert!(result.is_ok());
        let response = result.unwrap().unwrap();
        assert_eq!(response.embedding, vec![1.0; 4]);
    }

    #[tokio::test]
    async fn test_complete_returns_no_pending_after_clear() {
        let channel = ResponseChannel::new();
        let _rx = channel.register("cleared-1".to_string()).await;
        assert_eq!(channel.pending_count().await, 1);

        channel.clear().await;
        assert_eq!(channel.pending_count().await, 0);

        channel
            .complete(
                "cleared-1".to_string(),
                Ok(EmbedResponse {
                    embedding: vec![],
                    dimension: 0,
                    processing_time_ms: 0,
                }),
            )
            .await;
    }
}
