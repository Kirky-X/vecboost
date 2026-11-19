// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 推理输入安全校验 + 拒绝审计的统一挂接点。
//!
//! API 层（embed / embed_batch / rerank / decisions）与 pipeline 入队门共用
//! 本模块的 [`guard_text`]：从 kit 读取 `[input_security]` 开关与审计 logger，
//! 执行校验，拒绝时写安全审计事件（仅原文 sha256 指纹，不落原文）。
//! 校验逻辑单一事实源在 [`validate_text_security`]，本模块只做装配与审计。

use crate::config::app::InputSecurityConfig;
use crate::error::VecboostError;
use crate::utils::validator::input::{
    detect_suspicious_patterns, has_disallowed_format_chars, validate_text_security,
};
use std::sync::Arc;

/// 从 kit 读取 `[input_security]` 配置；未注入时回退全 false（默认零变更）。
pub(crate) fn input_security_from_kit<S>(kit: &trait_kit::AsyncKit<S>) -> InputSecurityConfig {
    kit.config::<InputSecurityConfig>().unwrap_or_default()
}

/// 记录一条输入拒绝审计事件（供长度等前置校验路径复用）。
/// 仅落 sha256 指纹，不落原文。
pub(crate) fn audit_text_rejection<S>(
    kit: &trait_kit::AsyncKit<S>,
    endpoint: &str,
    caller: Option<&str>,
    reason: &str,
    text: &str,
) {
    if let Ok(Some(logger)) = kit.config::<Option<Arc<crate::audit::AuditLogger>>>() {
        let fingerprint = crate::utils::hash::sha256_hex(text.as_bytes());
        logger.log_input_rejected(endpoint, caller, reason, &fingerprint);
    }
}

/// 校验单条文本并在拒绝时写审计事件。
///
/// - `endpoint`：路由路径（如 `/api/v1/embed`），作为审计事件的端点字段；
/// - `caller`：调用方用户名（无法取得时传 None）；
/// - `text`：原始输入，仅用于计算 sha256 指纹，不入审计日志。
///
/// 返回原始 [`VecboostError::InvalidInput`]，由调用方映射为 400。
pub(crate) fn guard_text<S>(
    kit: &trait_kit::AsyncKit<S>,
    endpoint: &str,
    caller: Option<&str>,
    text: &str,
) -> Result<(), VecboostError> {
    let cfg = input_security_from_kit(kit);
    let result = validate_text_security(
        text,
        cfg.reject_format_chars,
        cfg.detect_suspicious_patterns,
    );
    if result.is_err() {
        let fingerprint = crate::utils::hash::sha256_hex(text.as_bytes());
        if let Ok(Some(logger)) = kit.config::<Option<Arc<crate::audit::AuditLogger>>>() {
            // 分类判定按 validate_text_security 检查顺序，与拒绝原因一致。
            if cfg.detect_suspicious_patterns
                && let Some(pattern) = detect_suspicious_patterns(text)
                && !has_disallowed_format_chars(text)
            {
                logger.log_adversarial_input(endpoint, caller, &pattern, &fingerprint);
            } else if cfg.reject_format_chars && has_disallowed_format_chars(text) {
                logger.log_input_rejected(endpoint, caller, "format_chars", &fingerprint);
            } else {
                logger.log_input_rejected(endpoint, caller, "control_chars", &fingerprint);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::app::InputSecurityConfig;

    fn build_kit(
        cfg: InputSecurityConfig,
        logger: Option<Arc<crate::audit::AuditLogger>>,
    ) -> trait_kit::AsyncKit {
        let kit = trait_kit::AsyncKit::new();
        kit.set_config(cfg);
        kit.set_config(logger);
        // config() 直接读 set_config 注入的 configs 表，无需 build/注册模块。
        kit
    }

    #[test]
    fn test_input_security_defaults_when_absent() {
        let kit = build_kit(InputSecurityConfig::default(), None);
        let cfg = input_security_from_kit(&kit);
        assert!(!cfg.reject_format_chars);
        assert!(!cfg.detect_suspicious_patterns);
    }

    #[test]
    fn test_guard_text_allows_normal_input_by_default() {
        let kit = build_kit(InputSecurityConfig::default(), None);
        assert!(guard_text(&kit, "/api/v1/embed", None, "normal text").is_ok());
    }

    #[test]
    fn test_guard_text_rejects_format_chars_when_enabled() {
        crate::i18n::init();
        let kit = build_kit(
            InputSecurityConfig {
                reject_format_chars: true,
                ..Default::default()
            },
            None,
        );
        assert!(guard_text(&kit, "/api/v1/embed", None, "a\u{200b}b").is_err());
    }

    #[tokio::test]
    async fn test_guard_text_logs_audit_without_raw_input() {
        crate::i18n::init();
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("guard_audit.log");
        let config = crate::audit::AuditConfig {
            enabled: true,
            log_file_path: log_path.clone(),
            ..Default::default()
        };
        let logger = Arc::new(crate::audit::AuditLogger::new(config));
        let kit = build_kit(
            InputSecurityConfig {
                reject_format_chars: true,
                ..Default::default()
            },
            Some(logger.clone()),
        );
        let raw = "secret-raw-payload\u{200b}";
        assert!(guard_text(&kit, "/api/v1/embed", None, raw).is_err());
        logger.flush().await.unwrap();
        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(content.contains("input_validation_rejected"));
        assert!(content.contains("format_chars"));
        assert!(!content.contains("secret-raw-payload"));
    }
}
