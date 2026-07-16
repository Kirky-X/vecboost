// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod logger;
pub use logger::AuditLogger;

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
}

impl SecurityEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SecurityEventType::LoginSuccess => "login_success",
            SecurityEventType::LoginFailed => "login_failed",
            SecurityEventType::Logout => "logout",
            SecurityEventType::PermissionDenied => "permission_denied",
            SecurityEventType::UserCreated => "user_created",
            SecurityEventType::UserUpdated => "user_updated",
            SecurityEventType::UserDeleted => "user_deleted",
            SecurityEventType::TokenRefresh => "token_refresh",
            SecurityEventType::UnauthorizedAccess => "unauthorized_access",
            SecurityEventType::RateLimitExceeded => "rate_limit_exceeded",
            SecurityEventType::ConfigChanged => "config_changed",
        }
    }
}

/// Security event log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub timestamp: DateTime<Utc>,
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
            timestamp: Utc::now(),
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
    pub log_file_path: PathBuf,
    pub log_level: String,
    pub max_file_size_mb: usize,
    pub max_files: usize,
    pub async_write: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_file_path: PathBuf::from("logs/audit.log"),
            log_level: "info".to_string(),
            max_file_size_mb: 100,
            max_files: 10,
            async_write: true,
        }
    }
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
