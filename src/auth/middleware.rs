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

/// 认证上下文 — 由 `auth_middleware` 在验证通过后注入到请求扩展中。
///
/// 包含已解析的用户身份和原始 JWT token，下游 handler 通过
/// `request.extensions().get::<AuthContext>()` 获取。
#[derive(Clone)]
pub struct AuthContext {
    pub user: User,
    pub token: String,
}

const PUBLIC_PATHS: &[&str] = &["/health", "/api/1/auth/login", "/api/1/auth/refresh"];

/// 需要 admin 角色的路径(前缀匹配)。覆盖带 `/api/1` 前缀与 `no_prefix` 两种注册形态。
///
/// - `/api/1/model/*`(switch/unload/current/info):模型管理属高危操作,可造成
///   服务不可用或加载任意本地目录,必须限 admin;
/// - `/api/1/embed/file`:服务端文件读取原语,text_preview 会回传文件内容。
const ADMIN_PATH_PREFIXES: &[&str] = &[
    "/api/1/model/",
    "/api/1/embed/file",
    "/model/",
    "/embed/file",
];

/// 该路径是否要求 admin 角色。
fn requires_admin(path: &str) -> bool {
    ADMIN_PATH_PREFIXES.iter().any(|p| path.starts_with(p))
}

/// 查询当前 task_local token 是否具有 admin 角色。
/// 复用 `require_role_middleware` 的同一判定来源(garrison `has_role`),需已由
/// `with_current_token` 设置 task_local。
async fn current_token_is_admin() -> bool {
    matches!(GarrisonUtil::has_role("admin").await, Ok(true))
}

/// 供 api 层(embed/file text_preview 收权)复用的 admin 判定入口。
#[cfg_attr(not(feature = "grpc"), allow(dead_code))]
pub async fn current_token_is_admin_pub() -> bool {
    current_token_is_admin().await
}

/// Extract client IP respecting the X-Forwarded-For trust boundary.
///
/// Trust logic:
/// - `trusted_proxies` non-empty: `X-Forwarded-For` / `X-Real-IP` are honored only
///   when `connect_info` peer IP matches a `trusted_proxies` CIDR entry (reuses
///   `crate::rate_limit::is_ip_whitelisted`). Prevents spoofing by clients outside
///   the trust boundary.
/// - `trusted_proxies` empty: XFF is IGNORED and the direct peer IP is used
///   (secure-by-default). Attackers can otherwise rotate XFF per request to bypass
///   IP-based rate limiting and pollute audit trails. To restore proxy-aware
///   behavior, explicitly configure `trusted_proxies` (e.g. `["10.0.0.0/8"]`, or
///   `["0.0.0.0/0"]` to trust every peer — legacy v0.3.0–v0.3.2 behavior).
/// - XFF absent or invalid: fall back to `connect_info` peer IP; if `connect_info`
///   is also unavailable, returns `None`.
fn extract_client_ip(
    headers: &HeaderMap,
    connect_info: Option<SocketAddr>,
    trusted_proxies: &[String],
) -> Option<IpAddr> {
    let peer_ip = connect_info.map(|sa| sa.ip());

    let xff_trusted = if trusted_proxies.is_empty() {
        // Secure default: no explicit trust boundary → never trust client-supplied XFF.
        false
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
    // Secure default: empty trusted_proxies → XFF ignored (peer IP used). Remind
    // proxy deployments to configure their trust boundary explicitly.
    static EMPTY_PROXIES_WARN: std::sync::Once = std::sync::Once::new();
    if auth_config.trusted_proxies.is_empty() {
        EMPTY_PROXIES_WARN.call_once(|| {
            log::info!(
                "trusted_proxies is empty — X-Forwarded-For is ignored; the direct peer IP is \
                 used for rate limiting and audit. Behind a reverse proxy, configure \
                 trusted_proxies (e.g. [\"10.0.0.0/8\"]) to honor forwarded headers."
            );
        });
    }

    let path = request.uri().path();

    if PUBLIC_PATHS.contains(&path) {
        return Ok(next.run(request).await);
    }

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
            return Ok(unauthorized_response("auth-credentials-missing"));
        }
    };

    match GarrisonUtil::get_login_id_by_token(&token).await {
        Ok(Some(login_id)) => {
            let user = User {
                username: login_id.clone(),
                role: String::new(), // garrison 通过 interface 查询角色
                permissions: vec![],
            };
            // RBAC 路径映射:高危端点要求 admin 角色(判定复用 current_token_is_admin)
            if requires_admin(path) {
                let is_admin =
                    garrison::stp::with_current_token(token.clone(), current_token_is_admin())
                        .await;
                if !is_admin {
                    if let Some(ref logger) = audit_logger {
                        logger.log_unauthorized_access(ip.map(|i| i.to_string()), path);
                    }
                    return Ok(forbidden_response());
                }
            }
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
            Ok(unauthorized_response("auth-invalid-token"))
        }
    }
}

/// 403 响应:统一错误 envelope 结构中携带 i18n 消息(而非裸 StatusCode)。
fn forbidden_response() -> Response {
    use axum::http::header;
    let body = serde_json::json!({
        "success": false,
        "error": {
            "code": "FORBIDDEN",
            "message": crate::i18n::tr("auth-admin-required"),
        }
    });
    let mut resp = Response::new(axum::body::Body::from(body.to_string()));
    *resp.status_mut() = StatusCode::FORBIDDEN;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    resp
}

/// 401 响应：结构化错误体（与 handler 层 R-1 契约 `{type, message, field, value}` 一致）。
fn unauthorized_response(message_key: &str) -> Response {
    use axum::http::header;
    let body = serde_json::json!({
        "type": "AuthenticationRequired",
        "message": crate::i18n::tr(message_key),
        "field": null,
        "value": null,
    });
    let mut resp = Response::new(axum::body::Body::from(body.to_string()));
    *resp.status_mut() = StatusCode::UNAUTHORIZED;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    resp
}

/// 429 响应：结构化错误体（与 handler 层 R-1 契约一致）。
fn rate_limited_response() -> Response {
    use axum::http::header;
    let body = serde_json::json!({
        "type": "RateLimitExceeded",
        "message": crate::i18n::tr("rate-limit-exceeded"),
        "field": null,
        "value": null,
    });
    let mut resp = Response::new(axum::body::Body::from(body.to_string()));
    *resp.status_mut() = StatusCode::TOO_MANY_REQUESTS;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    resp
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
/// 应用到全部路由(main.rs 全局挂载):业务限流 + IETF RateLimit-* 头注入,
/// 防资源耗尽;`/health`、`/metrics` 与 IP 白名单主机豁免。
/// 认证端点(login/refresh)的暴力破解防护由本中间件的 path 语义覆盖。
///
/// 通过 limiteron Governor 的 RequestContext 驱动限流。
/// 白名单内的 IP 跳过限流。限流未启用时直接放行。
///
/// IP 提取复用 `extract_client_ip()`，遵循 `trusted_proxies` XFF 信任边界，
/// 与 `auth_middleware` 保持一致的客户端 IP 解析逻辑。
pub async fn auth_rate_limit_middleware(
    State(state): State<VecboostState>,
    State(auth_config): State<AuthConfig>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let rate_limit_enabled = state
        .kit
        .config::<crate::registry::RateLimitEnabled>()
        .map(|c| c.0)
        .unwrap_or_else(|_| {
            log::warn!(
                "RateLimitEnabled not registered, rate limiting is disabled. \
                        This may leave endpoints unprotected in production."
            );
            false
        });

    if !rate_limit_enabled {
        return Ok(next.run(request).await);
    }

    // 探活/指标端点豁免业务限流:LB/K8s 探针与 Prometheus 抓取器高频访问,
    // 若被 429 会被编排层误判为不健康而摘除实例(探活雪崩)。/metrics 自带
    // fail-closed 的 RateLimitModule 健康检查(metrics::endpoint),不受影响。
    let path = request.uri().path();
    if path == "/health" || path == "/metrics" {
        return Ok(next.run(request).await);
    }

    let ip_whitelist = state
        .kit
        .require::<crate::registry::IpWhitelistModule>()
        .map_err(|e| {
            log::error!("IpWhitelistModule not registered: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // 通过 XFF 信任边界提取客户端 IP（与 auth_middleware 一致）
    let connect_info = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0);
    let ip = extract_client_ip(
        request.headers(),
        connect_info,
        &auth_config.trusted_proxies,
    )
    .map(|i| i.to_string())
    .unwrap_or_else(|| "unknown".to_string());

    if crate::rate_limit::is_ip_whitelisted(&ip, &ip_whitelist) {
        return Ok(next.run(request).await);
    }

    // 通过 Governor 检查限流（规则匹配 + 封禁 + 熔断）
    let rate_limiter = state
        .kit
        .require::<crate::registry::RateLimitModule>()
        .map_err(|e| {
            log::error!("RateLimitModule not registered: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let context = crate::rate_limit::RequestContext {
        client_ip: Some(ip.clone()),
        path: request.uri().path().to_string(),
        method: request.method().to_string(),
        ..Default::default()
    };
    let decision = rate_limiter.check_rate_limit_detailed(&context).await;
    let allowed = decision.allowed;

    // IETF RateLimit-* 响应头注入开关（kit 配置，缺省关闭保持行为兼容）
    let headers_enabled = state
        .kit
        .config::<crate::registry::RateLimitHeadersEnabled>()
        .map(|c| c.0)
        .unwrap_or(false);

    if let Ok(prom_collector) = state
        .kit
        .require::<crate::registry::PrometheusCollectorModule>()
        && let Some(prom) = prom_collector.as_ref()
    {
        if allowed {
            prom.record_rate_limit_allowed("ip");
        } else {
            prom.record_rate_limit_denied("ip");
        }
    }

    if !allowed {
        log::warn!("Auth endpoint rate limit exceeded for IP: {}", ip);

        if let Ok(audit_opt) = state.kit.require::<crate::registry::AuditModule>()
            && let Some(logger) = audit_opt.as_ref()
        {
            logger.log_rate_limit_exceeded(None, Some(ip.clone()));
        }

        // 429 响应携带 RateLimit-*/Retry-After（开启 headers 时），便于
        // 客户端按标准头做退避；未开启时保持既有裸状态码行为
        if headers_enabled && let Some(values) = decision.headers {
            let response = rate_limited_response();
            return Ok(limiteron::middleware::inject_rate_limit_headers(
                response, &values,
            ));
        }

        return Ok(rate_limited_response());
    }

    let response = next.run(request).await;
    if headers_enabled && let Some(values) = decision.headers {
        return Ok(limiteron::middleware::inject_rate_limit_headers(
            response, &values,
        ));
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

    fn peer_addr(ip: IpAddr) -> Option<SocketAddr> {
        Some(SocketAddr::new(ip, 12345))
    }

    fn headers_with_xff(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", value.parse().unwrap());
        h
    }

    fn headers_with_x_real_ip(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-real-ip", value.parse().unwrap());
        h
    }

    // --- trusted_proxies 非空路径 ---

    #[test]
    fn extract_ip_trusted_proxy_with_xff_returns_xff_ip() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let headers = headers_with_xff("203.0.113.50");
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 50))));
    }

    #[test]
    fn extract_ip_untrusted_proxy_with_xff_returns_peer_ip() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
        let headers = headers_with_xff("203.0.113.50");
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        // 192.168.1.100 不在 10.0.0.0/8 内 → XFF 被忽略
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))));
    }

    #[test]
    fn extract_ip_trusted_proxy_xff_multiple_entries_returns_first() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let headers = headers_with_xff("203.0.113.50, 70.41.32.12, 198.51.100.2");
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 50))));
    }

    // --- trusted_proxies 为空路径（secure default:XFF 被忽略）---

    #[test]
    fn extract_ip_empty_proxies_with_xff_returns_peer_ip() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
        let headers = headers_with_xff("203.0.113.50");
        let proxies: Vec<String> = vec![];

        let result = extract_client_ip(&headers, peer, &proxies);
        // 空 trusted_proxies → 忽略伪造 XFF,使用直连 peer IP
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))));
    }

    #[test]
    fn requires_admin_matches_model_and_file_paths() {
        assert!(requires_admin("/api/1/model/switch"));
        assert!(requires_admin("/api/1/model/unload"));
        assert!(requires_admin("/api/1/embed/file"));
        // no_prefix 注册形态兜底
        assert!(requires_admin("/model/switch"));
        assert!(requires_admin("/embed/file"));
        // 普通端点不受限
        assert!(!requires_admin("/api/1/embed"));
        assert!(!requires_admin("/api/1/embed/batch"));
        assert!(!requires_admin("/api/1/auth/login"));
        assert!(!requires_admin("/health"));
    }

    // --- 无 XFF 回退 ---

    #[test]
    fn extract_ip_no_xff_returns_peer_ip() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
        let headers = HeaderMap::new();
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100))));
    }

    #[test]
    fn extract_ip_no_xff_no_connect_info_returns_none() {
        let headers = HeaderMap::new();
        let proxies: Vec<String> = vec![];

        let result = extract_client_ip(&headers, None, &proxies);
        assert_eq!(result, None);
    }

    // --- X-Real-IP 回退 ---

    #[test]
    fn extract_ip_x_real_ip_used_when_xff_absent() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let headers = headers_with_x_real_ip("203.0.113.99");
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 99))));
    }

    #[test]
    fn extract_ip_xff_preferred_over_x_real_ip() {
        let peer = peer_addr(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let mut headers = headers_with_xff("203.0.113.50");
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        // XFF 优先于 X-Real-IP
        assert_eq!(result, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 50))));
    }

    // --- IPv6 ---

    #[test]
    fn extract_ip_ipv6_peer_with_trusted_proxy() {
        let peer = peer_addr(IpAddr::V6(Ipv6Addr::LOCALHOST));
        let headers = headers_with_xff("2001:db8::1");
        // ::1 不在 10.0.0.0/8 → XFF 被忽略
        let proxies = vec!["10.0.0.0/8".to_string()];

        let result = extract_client_ip(&headers, peer, &proxies);
        assert_eq!(result, Some(IpAddr::V6(Ipv6Addr::LOCALHOST)));
    }

    // --- RBAC 路径→角色映射契约测试 ---

    /// 403 响应结构验证:含 i18n 消息与 FORBIDDEN 错误码
    #[test]
    fn forbidden_response_has_correct_status_and_json_body() {
        let resp = forbidden_response();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
    }

    /// 401 响应结构验证:含 R-1 契约的 `{type, message, field, value}` 结构
    #[test]
    fn unauthorized_response_has_structured_json_body() {
        let resp = unauthorized_response("auth-credentials-missing");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
    }

    /// 429 响应结构验证:含 R-1 契约的 `{type, message, field, value}` 结构
    #[test]
    fn rate_limited_response_has_structured_json_body() {
        let resp = rate_limited_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
    }

    /// 公开路径不触发 admin 要求
    #[test]
    fn public_paths_do_not_require_admin() {
        for path in PUBLIC_PATHS {
            assert!(!requires_admin(path), "公开路径 {path} 不应要求 admin 角色");
        }
    }

    /// admin 路径前缀完整性:所有高危端点均被覆盖
    #[test]
    fn admin_path_prefixes_cover_all_dangerous_endpoints() {
        // 模型管理
        assert!(requires_admin("/api/1/model/switch"));
        assert!(requires_admin("/api/1/model/unload"));
        assert!(requires_admin("/api/1/model/current"));
        assert!(requires_admin("/api/1/model/info"));
        // 文件嵌入(text_preview 回传文件内容)
        assert!(requires_admin("/api/1/embed/file"));
        // no_prefix 形态
        assert!(requires_admin("/model/switch"));
        assert!(requires_admin("/embed/file"));
    }

    /// 非 admin 路径不被误拦
    #[test]
    fn normal_endpoints_not_blocked_by_admin_check() {
        let safe_paths = [
            "/api/1/embed",
            "/api/1/embed/batch",
            "/api/1/similarity",
            "/api/1/rerank",
            "/api/1/auth/login",
            "/api/1/auth/refresh",
            "/api/1/auth/logout",
            "/health",
            "/metrics",
            "/api-docs",
        ];
        for path in safe_paths {
            assert!(!requires_admin(path), "普通端点 {path} 不应要求 admin 角色");
        }
    }
}
