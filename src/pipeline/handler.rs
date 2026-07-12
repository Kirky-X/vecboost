// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! 流水线请求处理函数

use crate::AppState;
use crate::domain::EmbedRequest;
use crate::error::VecboostError;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// 处理流水线请求
pub async fn handle_pipeline_request(
    state: AppState,
    req: EmbedRequest,
    ip: String,
) -> Result<axum::Json<crate::domain::EmbedResponse>, VecboostError> {
    // 生成请求 ID
    let request_id = Uuid::new_v4().to_string();

    // 创建响应通道
    let response_rx = state.response_channel.register(request_id.clone()).await;

    // 构建队列请求
    let priority = state
        .priority_calculator
        .calculate(crate::pipeline::PriorityInput {
            base_priority: crate::pipeline::Priority::Normal,
            time_until_timeout: Duration::from_secs(30),
            user_tier: None,
            source: crate::pipeline::RequestSource::http(ip.clone()),
            queue_length: state.pipeline_queue.size(),
        });

    let (tx, _) = oneshot::channel();

    let queued_request = crate::pipeline::QueuedRequest {
        request_id: request_id.clone(),
        embed_request: req,
        priority,
        submitted_at: std::time::Instant::now(),
        timeout: Duration::from_secs(30),
        source: crate::pipeline::RequestSource::http(ip),
        response_tx: tx,
    };

    // 提交到流水线队列
    state.pipeline_queue.enqueue(queued_request).await?;

    // 等待响应
    match tokio::time::timeout(Duration::from_secs(30), response_rx).await {
        Ok(Ok(Ok(response))) => Ok(axum::Json(response)),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(_)) => Err(VecboostError::InternalError(
            "Response channel error".to_string(),
        )),
        Err(_) => Err(VecboostError::ValidationError(
            "Request timeout".to_string(),
        )),
    }
}
