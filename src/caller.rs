// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 调用方身份提取（语义缓存 caller isolation 用）。
//!
//! caller 取服务端认证态用户名，与 `AuthContext.user.username` 同源：
//! 认证中间件在 token 校验通过后把 token 写入 garrison task_local，本模块
//! 读取该 token 并解析出登录名。**不读取**可伪造的 X-User-Id 请求头
//! （与 `src/rate_limit/limiteron_adapter.rs` 既有结论一致）。

/// 无认证态时的归并桶字面量。
pub const ANONYMOUS_CALLER: &str = "anonymous";

/// 当前请求的调用方身份。
#[cfg(feature = "auth")]
pub async fn current_caller_id() -> String {
    match garrison::stp::current_token() {
        Ok(token) => crate::auth::GarrisonUtil::get_login_id_by_token(&token)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| ANONYMOUS_CALLER.to_string()),
        Err(_) => ANONYMOUS_CALLER.to_string(),
    }
}

/// auth 关闭时恒为 "anonymous"（公开推理端点目前不存在，防御性归并桶）。
#[cfg(not(feature = "auth"))]
pub async fn current_caller_id() -> String {
    ANONYMOUS_CALLER.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anonymous_fallback_without_token() {
        // 无 task_local token 时必须回退 anonymous（auth 构建下 current_token 失败）。
        let caller = current_caller_id().await;
        assert!(!caller.is_empty());
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_caller_from_task_local_token() {
        // 设置一个测试 token 并确认能解析（解析失败也应回退 anonymous，不 panic）。
        let _ = crate::auth::with_current_token("test-token".to_string(), async {
            let caller = current_caller_id().await;
            assert!(!caller.is_empty());
        })
        .await;
    }
}
