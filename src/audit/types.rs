// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! `impl` blocks for audit data types (rule 25: keep mod.rs interface-only).

use std::path::PathBuf;

use super::{AuditConfig, SecurityEventType};

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
