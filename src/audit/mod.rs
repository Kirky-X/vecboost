// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use time::OffsetDateTime;

pub mod logger;
pub use logger::AuditLogger;
pub mod task_audit;
pub use task_audit::TaskAudit;
pub mod task_event;
pub use task_event::TaskExecutionEvent;
pub mod types;

/// Security event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    LoginSuccess,
    LoginFailed,
    Logout,
    PermissionDenied,
    UserCreated,
    UserUpdated,
    UserDeleted,
    TokenRefresh,
    UnauthorizedAccess,
    RateLimitExceeded,
    ConfigChanged,
    /// 输入安全校验拒绝（Cf 格式字符 / 长度 / 控制字符）。
    InputValidationRejected,
    /// 可疑输入模式命中（同字符长填充 / 格式字符密度异常）。
    AdversarialInputDetected,
}

/// Security event log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    pub event_type: String,
    pub user: Option<String>,
    pub ip: Option<String>,
    pub request_id: Option<String>,
    pub user_agent: Option<String>,
    pub details: serde_json::Value,
    pub success: bool,
}

impl SecurityEvent {
    pub fn new(
        event_type: SecurityEventType,
        user: Option<String>,
        ip: Option<String>,
        details: serde_json::Value,
        success: bool,
    ) -> Self {
        Self {
            timestamp: OffsetDateTime::now_utc(),
            event_type: event_type.as_str().to_string(),
            user,
            ip,
            request_id: None,
            user_agent: None,
            details,
            success,
        }
    }
}

/// Audit log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    /// 任务执行审计开关（[task_audit] enabled）；false 时 log_task_execution no-op。
    pub task_audit_enabled: bool,
    pub log_file_path: PathBuf,
    pub log_level: String,
    pub max_file_size_mb: usize,
    pub max_files: usize,
    pub async_write: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_event_creation() {
        let event = SecurityEvent::new(
            SecurityEventType::LoginSuccess,
            Some("test_user".to_string()),
            Some("127.0.0.1".to_string()),
            serde_json::json!({"test": "data"}),
            true,
        );

        assert_eq!(event.event_type, "login_success");
        assert_eq!(event.user, Some("test_user".to_string()));
        assert_eq!(event.ip, Some("127.0.0.1".to_string()));
        assert!(event.success);
    }

    #[test]
    fn test_security_event_type_as_str() {
        assert_eq!(SecurityEventType::LoginSuccess.as_str(), "login_success");
        assert_eq!(SecurityEventType::LoginFailed.as_str(), "login_failed");
        assert_eq!(SecurityEventType::Logout.as_str(), "logout");
        assert_eq!(
            SecurityEventType::PermissionDenied.as_str(),
            "permission_denied"
        );
    }

    #[test]
    fn test_security_event_type_all_variants_as_str() {
        assert_eq!(SecurityEventType::UserCreated.as_str(), "user_created");
        assert_eq!(SecurityEventType::UserUpdated.as_str(), "user_updated");
        assert_eq!(SecurityEventType::UserDeleted.as_str(), "user_deleted");
        assert_eq!(SecurityEventType::TokenRefresh.as_str(), "token_refresh");
        assert_eq!(
            SecurityEventType::UnauthorizedAccess.as_str(),
            "unauthorized_access"
        );
        assert_eq!(
            SecurityEventType::RateLimitExceeded.as_str(),
            "rate_limit_exceeded"
        );
        assert_eq!(SecurityEventType::ConfigChanged.as_str(), "config_changed");
        assert_eq!(
            SecurityEventType::InputValidationRejected.as_str(),
            "input_validation_rejected"
        );
        assert_eq!(
            SecurityEventType::AdversarialInputDetected.as_str(),
            "adversarial_input_detected"
        );
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert!(config.enabled);
        assert_eq!(config.log_file_path, PathBuf::from("logs/audit.log"));
        assert_eq!(config.log_level, "info");
        assert_eq!(config.max_file_size_mb, 100);
        assert_eq!(config.max_files, 10);
        assert!(config.async_write);
    }

    #[test]
    fn test_security_event_new_with_none_fields() {
        let event = SecurityEvent::new(
            SecurityEventType::UnauthorizedAccess,
            None,
            None,
            serde_json::json!({"path": "/admin"}),
            false,
        );
        assert_eq!(event.event_type, "unauthorized_access");
        assert!(event.user.is_none());
        assert!(event.ip.is_none());
        assert!(!event.success);
        assert!(event.request_id.is_none());
        assert!(event.user_agent.is_none());
    }
}
