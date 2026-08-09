// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::VecboostState;
use crate::audit::AuditLogger;
use crate::auth::{GarrisonUtil, User};
use crate::config::app::AuthConfig;
use axum::{
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

#[derive(Clone)]
pub struct AuthContext {
    pub user: User,
    pub token: String,
}

const PUBLIC_PATHS: &[&str] = &["/health", "/api/v1/auth/login", "/api/v1/auth/refresh"];

/// Extract client IP respecting the X-Forwarded-For trust boundary.
///
/// Trust logic:
/// - `trusted_proxies` non-empty: `X-Forwarded-For` / `X-Real-IP` are honored only
///   when `connect_info` peer IP matches a `trusted_proxies` CIDR entry (reuses
///   `crate::rate_limit::is_ip_whitelisted`). Prevents spoofing by clients outside
///   the trust boundary.
/// - `trusted_proxies` empty: XFF honored unconditionally (legacy v0.3.0–v0.3.2
///   behavior, kept for backward compatibility).
/// - XFF absent or invalid: fall back to `connect_info` peer IP; if `connect_info`
///   is also unavailable, returns `None`.
fn extract_client_ip(
    headers: &HeaderMap,
    connect_info: Option<SocketAddr>,
    trusted_proxies: &[String],
) -> Option<IpAddr> {
    let peer_ip = connect_info.map(|sa| sa.ip());

    let xff_trusted = if trusted_proxies.is_empty() {
        // Legacy behavior: trust XFF unconditionally when no boundary is configured.
        true
    } else {
        match peer_ip {
            Some(ip) => crate::rate_limit::is_ip_whitelisted(&ip.to_string(), trusted_proxies),
            None => false,
        }
    };

    if xff_trusted
        && let Some(xff_ip) = headers
            .get("x-forwarded-for")
            .or_else(|| headers.get("x-real-ip"))
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.trim().parse().ok())
    {
        return Some(xff_ip);
    }

    peer_ip
}

pub async fn auth_middleware(
    State(audit_logger): State<Option<Arc<AuditLogger>>>,
    State(auth_config): State<AuthConfig>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = request.uri().path();

    if PUBLIC_PATHS.contains(&path) {
        return Ok(next.run(request).await);
    }

    // 从 Authorization 头获取 token
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|h| h.to_str().ok());

    let connect_info = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0);
    let ip = extract_client_ip(
        request.headers(),
        connect_info,
        &auth_config.trusted_proxies,
    );

    let token = match auth_header {
        Some(header) if header.starts_with("Bearer ") => header[7..].to_string(),
        _ => {
            if let Some(ref logger) = audit_logger {
                logger.log_unauthorized_access(ip.map(|i| i.to_string()), path);
            }
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // 通过 garrison 验证 token 并获取 login_id
    match GarrisonUtil::get_login_id_by_token(&token).await {
        Ok(Some(login_id)) => {
            let user = User {
                username: login_id.clone(),
                role: String::new(), // garrison 通过 interface 查询角色
                permissions: vec![],
            };
            let mut request = request;
            request.extensions_mut().insert(AuthContext {
                user,
                token: token.clone(),
            });
            // 设置 task_local token，使下游 handler 可使用 GarrisonUtil::has_permission/has_role
            Ok(garrison::stp::with_current_token(token, next.run(request)).await)
        }
        _ => {
            if let Some(ref logger) = audit_logger {
                logger.log_unauthorized_access(ip.map(|i| i.to_string()), path);
            }
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

pub async fn optional_auth_middleware(
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Response {
    if let Some(auth_header) = headers.get("authorization")
        && let Ok(auth_str) = auth_header.to_str()
        && let Some(token) = auth_str.strip_prefix("Bearer ")
        && let Ok(Some(login_id)) = GarrisonUtil::get_login_id_by_token(token).await
    {
        let user = User {
            username: login_id,
            role: String::new(),
            permissions: vec![],
        };
        request.extensions_mut().insert(AuthContext {
            user,
            token: token.to_string(),
        });
        // 设置 task_local token，使下游 handler 可使用 GarrisonUtil 权限查询
        return garrison::stp::with_current_token(token.to_string(), next.run(request)).await;
    }

    next.run(request).await
}

/// 权限校验中间件 — 委托 garrison `GarrisonUtil::has_permission()`。
///
/// 需要 `auth_middleware` 先设置 task_local token（已通过 `with_current_token` 完成）。
pub async fn require_permission_middleware(
    permission: &'static str,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 确保已认证
    let _auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    match GarrisonUtil::has_permission(permission).await {
        Ok(true) => Ok(next.run(request).await),
        _ => Err(StatusCode::FORBIDDEN),
    }
}

/// 角色校验中间件 — 委托 garrison `GarrisonUtil::has_role()`。
///
/// 需要 `auth_middleware` 先设置 task_local token（已通过 `with_current_token` 完成）。
pub async fn require_role_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    // 确保已认证
    let _auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    match GarrisonUtil::has_role("admin").await {
        Ok(true) => Ok(next.run(request).await),
        _ => Err(StatusCode::FORBIDDEN),
    }
}

// ============================================================================
// Auth Endpoint Rate Limiting Middleware
// ============================================================================

/// Auth 端点速率限制中间件(vuln-0006 修复)
///
/// 应用到 `/api/v1/auth/login`、`/api/v1/auth/refresh`、`/api/v1/auth/logout`
/// 和 `/api/v1/auth/me` 等认证端点,防止暴力破解和 token 枚举攻击。
///
/// 限流维度:`Global` + `Ip`(用户尚未认证时不使用 `User` 维度)。
/// 白名单内的 IP 跳过限流。限流未启用时直接放行。
pub async fn auth_rate_limit_middleware(
    State(state): State<VecboostState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 检查限流是否启用
    let rate_limit_enabled = state
        .kit
        .config::<crate::module_registry::RateLimitEnabled>()
        .map(|c| c.0)
        .unwrap_or(false);

    if !rate_limit_enabled {
        return Ok(next.run(request).await);
    }

    // 获取 IP 白名单
    let ip_whitelist = state
        .kit
        .require::<crate::module_registry::IpWhitelistModule>()
        .map_err(|e| {
            log::error!("IpWhitelistModule not registered: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // 从 ConnectInfo 获取客户端 IP(与 embedding handler 一致)
    let ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // 白名单内的 IP 不限流
    if crate::rate_limit::is_ip_whitelisted(&ip, &ip_whitelist) {
        return Ok(next.run(request).await);
    }

    // 检查 Global + Ip 维度限流
    let rate_limiter = state
        .kit
        .require::<crate::module_registry::RateLimitModule>()
        .map_err(|e| {
            log::error!("RateLimitModule not registered: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let allowed = rate_limiter
        .check_rate_limit(vec![
            crate::rate_limit::Dimension::Global,
            crate::rate_limit::Dimension::Ip(ip.clone()),
        ])
        .await;

    // 记录限流决策指标
    if let Ok(prom_collector) = state.kit.require::<crate::module_registry::PrometheusCollectorModule>() {
        if let Some(prom) = prom_collector.as_ref() {
            if allowed {
                prom.record_rate_limit_allowed("ip");
            } else {
                prom.record_rate_limit_denied("ip");
            }
        }
    }

    if !allowed {
        log::warn!("Auth endpoint rate limit exceeded for IP: {}", ip);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    Ok(next.run(request).await)
}


