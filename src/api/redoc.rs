// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! ReDoc 交互文档端点（feature `swagger-ui`）。
//!
//! 与 Swagger UI 共用同一份 OpenAPI 规范（`/api-docs/openapi.json`），
//! 仅渲染器不同。页面 HTML 内嵌（`include_str!`），渲染脚本走 CDN ——
//! Swagger UI 仍为 vendored 离线可用，ReDoc 页面在浏览器无外网时不可用
//! （见 `docs/API_REFERENCE.md`）。

use axum::Router;
use axum::response::Html;
use axum::routing::get;

/// ReDoc standalone 脚本地址（页面注入；测试同时钉住该常量与 HTML 占位符）。
pub const REDOC_SCRIPT_URL: &str =
    "https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js";

/// 构建 `/redoc` 路由（由 `build_http_router` 在 `swagger-ui` 构建下挂载）。
pub fn redoc_router() -> Router {
    Router::new().route("/redoc", get(redoc_page))
}

async fn redoc_page() -> Html<String> {
    Html(include_str!("redoc.html").replace("__REDOC_SCRIPT_URL__", REDOC_SCRIPT_URL))
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
        assert!(
            !html.contains("__REDOC_SCRIPT_URL__"),
            "script url placeholder must be replaced"
        );
    }
}
