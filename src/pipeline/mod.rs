// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 请求流水线模块
//!
//! 实现请求流水线处理，包括优先级队列、动态 Worker 管理等

mod config;
mod priority;
mod queue;
mod response_channel;
mod scheduler;
mod worker;

pub use config::{PipelineConfig, PriorityConfig, QueueConfig, WorkerConfig};
pub use priority::{Priority, PriorityCalculator, RequestSource};
pub use queue::{PriorityRequestQueue, QueuedRequest};
pub use response_channel::ResponseChannel;
pub use scheduler::PipelineScheduler;
pub use worker::WorkerManager;

use crate::AppState;
use crate::domain::EmbedRequest;
use crate::error::AppError;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// 处理流水线请求
pub async fn handle_pipeline_request(
    state: AppState,
    req: EmbedRequest,
    ip: String,
) -> Result<axum::Json<crate::domain::EmbedResponse>, AppError> {
    // 生成请求 ID
    let request_id = Uuid::new_v4().to_string();

    // 创建响应通道
    let response_rx = state.response_channel.register(request_id.clone()).await;

    // 构建队列请求
    let priority = state
        .priority_calculator
        .calculate(crate::pipeline::priority::PriorityInput {
            base_priority: crate::pipeline::priority::Priority::Normal,
            time_until_timeout: Duration::from_secs(30),
            user_tier: None,
            source: crate::pipeline::priority::RequestSource::http(ip.clone()),
            queue_length: state.pipeline_queue.size(),
        });

    let (tx, _) = oneshot::channel();

    let queued_request = crate::pipeline::queue::QueuedRequest {
        request_id: request_id.clone(),
        embed_request: req,
        priority,
        submitted_at: std::time::Instant::now(),
        timeout: Duration::from_secs(30),
        source: crate::pipeline::priority::RequestSource::http(ip),
        response_tx: tx,
    };

    // 提交到流水线队列
    state.pipeline_queue.enqueue(queued_request).await?;

    // 等待响应
    match tokio::time::timeout(Duration::from_secs(30), response_rx).await {
        Ok(Ok(Ok(response))) => Ok(axum::Json(response)),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(_)) => Err(AppError::InternalError(
            "Response channel error".to_string(),
        )),
        Err(_) => Err(AppError::ValidationError("Request timeout".to_string())),
    }
}
