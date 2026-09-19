// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Extractor 拒绝响应规范化中间件（审计建议）。
//!
//! axum 原生 `Json` extractor 的拒绝（缺字段/类型错/非法 JSON/缺 Content-Type）
//! 产生 text/plain 错误体，与 handler 层结构化 `{type, message, field, value}`
//! 错误契约不一致。本中间件仅对可识别的 axum 拒绝文案做形状改写：
//! 状态码保持不变（400/415/422），body 改为结构化 JSON；无法识别的响应原样透传。

#[cfg(feature = "http")]
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    middleware::Next,
    response::Response,
};

/// axum JsonRejection 文案前缀 → 结构化改写候选
#[cfg(feature = "http")]
const REJECTION_PREFIXES: [&str; 3] = [
    "Failed to deserialize the JSON body into the target type:",
    "Failed to parse the request body as JSON:",
    "Expected request with `Content-Type",
];

/// 从 axum 拒绝文案中提取字段名（尽力而为，提取失败返回 None）。
#[cfg(feature = "http")]
fn extract_field(detail: &str) -> Option<String> {
    // "missing field `text` at line ..." → text
    if let Some(pos) = detail.find("missing field `") {
        let rest = &detail[pos + "missing field `".len()..];
        if let Some(end) = rest.find('`') {
            return Some(rest[..end].to_string());
        }
    }
    // "text: invalid type: integer `123`, expected a string" → text
    if let Some(pos) = detail.find(": invalid type") {
        let head = detail[..pos].trim();
        let head = head.trim_start_matches('`').trim_end_matches('`');
        // serde 路径可能带点（a.b），仅取末段作为字段名
        let field = head.rsplit('.').next().unwrap_or(head);
        if !field.is_empty()
            && field
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Some(field.to_string());
        }
    }
    None
}

/// 将 axum 拒绝文案改写为结构化错误 JSON；非可识别文案返回 None（透传）。
#[cfg(feature = "http")]
fn normalize_rejection_body(text: &str) -> Option<serde_json::Value> {
    let matched = REJECTION_PREFIXES
        .iter()
        .find(|p| text.trim_start().starts_with(**p))?;
    let detail = text.trim_start()[matched.len()..].trim();
    let message = if detail.is_empty() {
        matched.to_string()
    } else {
        detail.to_string()
    };
    let field = extract_field(&message);
    Some(serde_json::json!({
        "type": "InvalidInput",
        "message": message,
        "field": field,
        "value": serde_json::Value::Null,
    }))
}

/// 中间件：规范化 extractor 拒绝的错误体（仅 400/415/422 + text/plain + 可识别文案）。
#[cfg(feature = "http")]
pub async fn rejection_normalizer(req: Request, next: Next) -> Response {
    let res = next.run(req).await;
    let candidate = matches!(
        res.status(),
        StatusCode::BAD_REQUEST
            | StatusCode::UNSUPPORTED_MEDIA_TYPE
            | StatusCode::UNPROCESSABLE_ENTITY
    ) && res
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.starts_with("text/plain"))
        .unwrap_or(false);
    if !candidate {
        return res;
    }

    let (mut parts, body) = res.into_parts();
    let bytes = match axum::body::to_bytes(body, 16 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            // body 读取失败（已消费/超限）：无法规范化，回退空 4xx 响应
            return Response::builder()
                .status(parts.status)
                .body(Body::empty())
                .unwrap_or_else(|_| Response::new(Body::empty()));
        }
    };
    let text = String::from_utf8_lossy(&bytes);
    match normalize_rejection_body(&text) {
        Some(payload) => {
            let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
            parts.headers.insert(
                CONTENT_TYPE,
                "application/json"
                    .parse()
                    .unwrap_or(HeaderValue::from_static("application/json")),
            );
            Response::from_parts(parts, Body::from(json))
        }
        None => Response::from_parts(parts, Body::from(bytes)),
    }
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use super::*;

    #[test]
    fn missing_field_yields_field_name() {
        let out = normalize_rejection_body(
            "Failed to deserialize the JSON body into the target type: missing field `text` at line 1 column 40",
        )
        .unwrap();
        assert_eq!(out["type"], "InvalidInput");
        assert_eq!(out["field"], "text");
        assert!(out["message"].as_str().unwrap().contains("missing field"));
    }

    #[test]
    fn invalid_type_yields_field_name() {
        let out = normalize_rejection_body(
            "Failed to deserialize the JSON body into the target type: text: invalid type: integer `12345`, expected a string at line 1 column 34",
        )
        .unwrap();
        assert_eq!(out["field"], "text");
    }

    #[test]
    fn syntax_error_keeps_message_without_field() {
        let out = normalize_rejection_body(
            "Failed to parse the request body as JSON: key must be a string at line 1 column 2",
        )
        .unwrap();
        assert_eq!(out["type"], "InvalidInput");
        assert!(out["field"].is_null());
        assert_eq!(out["value"], serde_json::Value::Null);
    }

    #[test]
    fn content_type_rejection_normalized() {
        let out = normalize_rejection_body(
            "Expected request with `Content-Type: application/json` but the request was missing it",
        )
        .unwrap();
        assert_eq!(out["type"], "InvalidInput");
    }

    #[test]
    fn unrecognized_text_passes_through() {
        assert!(normalize_rejection_body("some other error").is_none());
    }

    #[test]
    fn serde_path_dotted_field_takes_last_segment() {
        let out = normalize_rejection_body(
            "Failed to deserialize the JSON body into the target type: queries[1].query: invalid type: null, expected a string",
        )
        .unwrap();
        assert_eq!(out["field"], "query");
    }
}
