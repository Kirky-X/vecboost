// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! ReDoc 交互文档端点（feature `swagger-ui`）。
//!
//! 与 Swagger UI 共用同一份 OpenAPI 规范（`/api-docs/openapi.json`），
//! 仅渲染器不同。页面 HTML 内嵌（`include_str!`），渲染脚本走 CDN 并锁定
//! 精确版本 + SRI 完整性校验（防 CDN/npm 供应链篡改）——Swagger UI 仍为
//! vendored 离线可用，ReDoc 页面在浏览器无外网时不可用
//! （见 `docs/API_REFERENCE.md`）。

use axum::Router;
use axum::response::Html;
use axum::routing::get;

/// ReDoc standalone 脚本地址（页面注入；锁定精确版本，不追浮动标签）。
/// 升级版本时必须同步重算 [`REDOC_SCRIPT_INTEGRITY`]。
pub const REDOC_SCRIPT_URL: &str =
    "https://cdn.jsdelivr.net/npm/redoc@2.5.4/bundles/redoc.standalone.js";

/// `REDOC_SCRIPT_URL` 内容的 subresource integrity 摘要
/// （sha384，`openssl dgst -sha384 -binary | openssl base64`）。
pub const REDOC_SCRIPT_INTEGRITY: &str =
    "sha384-w447zOpYfw/1Tv/5AK9NfHTlQIqE3RVR6KY62jCyy9zNDgO64cMwGGP1Fj0zJVf5";

/// 构建 `/redoc` 路由（由 `build_http_router` 在 `swagger-ui` 构建下挂载）。
pub fn redoc_router() -> Router {
    Router::new().route("/redoc", get(redoc_page))
}

async fn redoc_page() -> Html<String> {
    Html(
        include_str!("redoc.html")
            .replace("__REDOC_SCRIPT_URL__", REDOC_SCRIPT_URL)
            .replace("__REDOC_SCRIPT_INTEGRITY__", REDOC_SCRIPT_INTEGRITY),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, header};
    use tower::ServiceExt;

    #[tokio::test]
    async fn redoc_route_serves_html_with_spec_url() {
        let resp = redoc_router()
            .oneshot(
                Request::builder()
                    .uri("/redoc")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        assert!(
            content_type.contains("text/html"),
            "unexpected content-type: {content_type}"
        );

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let html = String::from_utf8(body.to_vec()).unwrap();
        assert!(html.contains("spec-url=\"/api-docs/openapi.json\""));
        assert!(html.contains("redoc.standalone.js"));
        assert!(html.contains("redoc@2.5.4"), "CDN version must be pinned");
        assert!(
            html.contains("integrity=\"sha384-"),
            "SRI integrity attribute must be present"
        );
        assert!(
            html.contains("crossorigin=\"anonymous\""),
            "crossorigin attribute must be present for SRI"
        );
        assert!(
            !html.contains("__REDOC_SCRIPT_URL__") && !html.contains("__REDOC_SCRIPT_INTEGRITY__"),
            "placeholders must be replaced"
        );
    }
}
