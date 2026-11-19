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

#[derive(Debug, Serialize)]
pub struct ReplayResponse {
    pub found: bool,
    pub event: Option<crate::audit::TaskExecutionEvent>,
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
    Ok(replay_handler(&st, &request_id))
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

fn replay_handler(st: &VecboostState, request_id: &str) -> ReplayResponse {
    let audit = crate::audit::task_audit::from_kit(&st.kit);
    let event = audit.and_then(|a| a.ring.get_by_request_id(request_id));
    ReplayResponse {
        found: event.is_some(),
        event,
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

    #[test]
    fn test_replay_response_not_found_shape() {
        let resp = ReplayResponse {
            found: false,
            event: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["found"], false);
        assert!(json["event"].is_null());
    }
}
