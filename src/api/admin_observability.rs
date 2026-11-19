// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Admin 可观测端点（`http` feature；路径受 admin 角色护栏）。
//!
//! - `GET /api/v1/admin/inference/replay/{request_id}`：从 TaskEventRing 回放事件；
//! - `POST /api/v1/admin/cache/explain`：缓存命中过程干跑解释。
//!
//! 两类端点暴露运维敏感面（缓存归属/相似度），经 admin 前缀护栏限定角色。

use crate::VecboostState;
use crate::api::embedding::{kit_internal_error, to_api_error};
use crate::api::init::state;
#[cfg(feature = "http")]
use sdforge::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct CacheExplainRequest {
    pub model_id: String,
    pub text: String,
    /// 调用方标识（缺省为 anonymous）。
    #[serde(default = "anonymous")]
    pub caller: String,
}

fn anonymous() -> String {
    "anonymous".to_string()
}

/// 回放命中响应体：200 时事件必存在（未命中以 404 表达，不再用
/// `found:false` 的 200 伪装成功——验收 FAIL 修复 2）。
#[derive(Debug, Serialize)]
pub struct ReplayResponse {
    pub event: crate::audit::TaskExecutionEvent,
}

#[cfg(feature = "http")]
#[forge(
    name = "inference_replay",
    version = "v1",
    path = "/admin/inference/replay/{request_id}",
    method = "GET",
    tool_name = "inference_replay",
    description = "Replay a task execution audit event by request id (admin only)"
)]
pub async fn forge_inference_replay(
    #[param(kind = "path")] request_id: String,
) -> Result<ReplayResponse, ApiError> {
    let st = state().map_err(to_api_error)?;
    replay_handler(&st, &request_id)
}

#[cfg(feature = "http")]
#[forge(
    name = "cache_explain",
    version = "v1",
    path = "/admin/cache/explain",
    method = "POST",
    tool_name = "cache_explain",
    description = "Explain semantic cache hit path (admin only)"
)]
pub async fn forge_cache_explain(
    req: CacheExplainRequest,
) -> Result<crate::cache::CacheExplain, ApiError> {
    let st = state().map_err(to_api_error)?;
    cache_explain_handler(&st, req).await
}

fn replay_handler(st: &VecboostState, request_id: &str) -> Result<ReplayResponse, ApiError> {
    let audit = crate::audit::task_audit::from_kit(&st.kit);
    match audit.and_then(|a| a.ring.get_by_request_id(request_id)) {
        Some(event) => Ok(ReplayResponse { event }),
        // 未命中（含审计上下文未装配）：404 NotFound，与项目错误机制同口径
        //（unload_model_handler 的 ApiError::NotFound 先例）。
        None => Err(ApiError::NotFound {
            resource: "task execution event".to_string(),
            resource_id: Some(request_id.to_string()),
        }),
    }
}

async fn cache_explain_handler(
    st: &VecboostState,
    req: CacheExplainRequest,
) -> Result<crate::cache::CacheExplain, ApiError> {
    let svc = st
        .kit
        .require::<crate::registry::EmbeddingModule>()
        .map_err(kit_internal_error)?;
    let guard = svc.read().await;
    let cache = guard.semantic_cache().cloned();
    drop(guard);
    if let Some(cache) = cache {
        Ok(cache.explain(&req.model_id, &req.caller, &req.text).await)
    } else {
        // 未装配语义缓存：全未命中的解释（L3 would compute）。
        Ok(crate::cache::CacheExplain {
            l1_exact_hit: false,
            l2_semantic_hit: None,
            l3_would_compute: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_explain_request_defaults_caller_anonymous() {
        let req: CacheExplainRequest =
            serde_json::from_str(r#"{"model_id":"m","text":"t"}"#).unwrap();
        assert_eq!(req.caller, "anonymous");
    }

    /// 回放命中响应形状：200 体直接携带事件（found 字段已随 404 语义移除）。
    #[test]
    fn test_replay_response_hit_shape() {
        let event = crate::audit::TaskExecutionEvent {
            task_type: crate::config::model::ModelTask::Decision,
            model_id: "m".to_string(),
            model_sha256_prefix: None,
            input_fingerprint: "fp".to_string(),
            output_confidence: None,
            low_confidence_flagged: false,
            latency_ms: 1.0,
            request_id: "req-1".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        };
        let json = serde_json::to_value(&ReplayResponse { event }).unwrap();
        assert_eq!(json["event"]["request_id"], "req-1");
        assert_eq!(json["event"]["model_id"], "m");
        assert!(json.get("found").is_none(), "found 字段必须移除");
    }

    /// 回放未命中：404 NotFound（此前恒 200 + found:false，验收 FAIL 修复 2）。
    /// 独立构建极简 kit（不触进程级单例 state，防与并行测试竞态）。
    #[tokio::test]
    async fn test_replay_handler_miss_returns_not_found() {
        let kit = trait_kit::AsyncKit::new();
        let kit = kit.build().await.expect("bare kit must build");
        let st = crate::VecboostState::new(std::sync::Arc::new(kit));
        let result = replay_handler(&st, "definitely-missing-request-id");
        match result {
            Err(ApiError::NotFound {
                resource,
                resource_id,
            }) => {
                assert_eq!(resource, "task execution event");
                assert_eq!(
                    resource_id.as_deref(),
                    Some("definitely-missing-request-id")
                );
            }
            other => panic!("Expected NotFound 404, got {other:?}"),
        }
    }
}
