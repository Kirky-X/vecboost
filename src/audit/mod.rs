// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

#[cfg(feature = "db")]
use crate::error::VecboostError;
#[cfg(feature = "db")]
use sea_orm::{ConnectionTrait, DatabaseBackend, Statement, Value};

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

/// Audit logger for security events
pub struct AuditLogger {
    config: AuditConfig,
    sender: Option<mpsc::Sender<SecurityEvent>>,
    _handle: Option<tokio::task::JoinHandle<()>>,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditConfig) -> Self {
        if !config.enabled {
            return Self {
                config,
                sender: None,
                _handle: None,
            };
        }

        let (sender, mut receiver) = mpsc::channel::<SecurityEvent>(1000);
        let log_file_path = config.log_file_path.clone();
        let max_file_size = config.max_file_size_mb * 1024 * 1024;
        let max_files = config.max_files;

        // Spawn background task for async logging
        let handle = tokio::spawn(async move {
            let mut current_size = 0u64;
            let mut consecutive_errors = 0u32;
            const MAX_RETRIES: u32 = 3;
            const ERROR_THRESHOLD: u32 = 10;

            // Ensure log directory exists
            #[allow(clippy::collapsible_if)]
            if let Some(parent) = log_file_path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    eprintln!("Failed to create log directory: {}", e);
                    return;
                }
            }

            #[allow(clippy::while_let_loop)]
            loop {
                match receiver.recv().await {
                    Some(event) => {
                        // Check if we should rotate the log
                        if let Ok(metadata) = tokio::fs::metadata(&log_file_path).await {
                            current_size = metadata.len();
                        }

                        if current_size > max_file_size as u64 {
                            Self::rotate_logs(&log_file_path, max_files).await;
                            current_size = 0;
                        }

                        // Write log entry with retry mechanism
                        if let Ok(log_line) = serde_json::to_string(&event) {
                            let mut success = false;

                            for attempt in 1..=MAX_RETRIES {
                                match Self::write_log_entry(&log_file_path, &log_line).await {
                                    Ok(_) => {
                                        consecutive_errors = 0;
                                        success = true;
                                        break;
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Failed to write audit log (attempt {}): {}",
                                            attempt, e
                                        );
                                        if attempt < MAX_RETRIES {
                                            tokio::time::sleep(tokio::time::Duration::from_millis(
                                                100 * attempt as u64,
                                            ))
                                            .await;
                                        }
                                    }
                                }
                            }

                            if !success {
                                consecutive_errors += 1;
                                eprintln!(
                                    "Failed to write audit log after {} attempts, falling back to stderr",
                                    MAX_RETRIES
                                );

                                // Fallback to stderr
                                eprintln!("[AUDIT FALLBACK] {}", log_line);

                                // Check if we should disable logging due to persistent errors
                                if consecutive_errors >= ERROR_THRESHOLD {
                                    eprintln!(
                                        "Audit logging disabled due to persistent errors ({} consecutive failures)",
                                        consecutive_errors
                                    );
                                    break;
                                }
                            }
                        }
                    }
                    None => {
                        // Channel closed, exit the loop
                        break;
                    }
                }
            }
        });

        Self {
            config,
            sender: Some(sender),
            _handle: Some(handle),
        }
    }

    /// Create a new audit logger with database backend
    #[cfg(feature = "db")]
    pub fn new_with_db(config: AuditConfig, db_pool: crate::db::DbPool) -> Self {
        if !config.enabled {
            return Self {
                config,
                sender: None,
                _handle: None,
            };
        }

        let (sender, mut receiver) = mpsc::channel::<SecurityEvent>(1000);
        let pool = db_pool.clone();

        let handle = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                if let Err(e) = Self::write_to_db(&pool, &event).await {
                    eprintln!("Failed to write audit log to db: {}", e);
                }
            }
        });

        Self {
            config,
            sender: Some(sender),
            _handle: Some(handle),
        }
    }

    /// Write audit event to database
    #[cfg(feature = "db")]
    async fn write_to_db(
        pool: &crate::db::DbPool,
        event: &SecurityEvent,
    ) -> Result<(), VecboostError> {
        let session = pool.get_session("admin").await?;
        let conn = session
            .connection()
            .map_err(|e| VecboostError::InternalError(format!("Failed to get connection: {e}")))?;
        let details_json =
            serde_json::to_string(&event.details).unwrap_or_else(|_| "{}".to_string());
        let timestamp = event.timestamp.format("%Y-%m-%d %H:%M:%S").to_string();
        let stmt = Statement::from_sql_and_values(
            DatabaseBackend::Sqlite,
            "INSERT INTO audit_logs (event_type, user, ip, request_id, user_agent, details, success, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                Value::String(Some(event.event_type.clone())),
                Value::String(event.user.clone()),
                Value::String(event.ip.clone()),
                Value::String(event.request_id.clone()),
                Value::String(event.user_agent.clone()),
                Value::String(Some(details_json)),
                Value::Int(Some(if event.success { 1 } else { 0 })),
                Value::String(Some(timestamp)),
            ],
        );
        conn.execute_raw(stmt).await.map_err(|e| {
            VecboostError::InternalError(format!("Failed to insert audit log: {e}"))
        })?;
        Ok(())
    }

    /// Write a log entry to the file
    async fn write_log_entry(
        log_file_path: &PathBuf,
        log_line: &str,
    ) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file_path)
            .await?;

        file.write_all((log_line.to_owned() + "\n").as_bytes())
            .await?;
        file.flush().await?;
        Ok(())
    }

    /// Log a security event
    pub fn log(&self, event: SecurityEvent) {
        if !self.config.enabled {
            return;
        }

        if let Some(sender) = &self.sender {
            match sender.try_send(event) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    eprintln!("Audit log channel full, event dropped");
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    eprintln!("Audit log channel closed, event dropped");
                }
            }
        }
    }

    /// Log a login success event
    pub fn log_login_success(&self, username: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::LoginSuccess,
            Some(username.to_string()),
            ip,
            serde_json::json!({}),
            true,
        );
        self.log(event);
    }

    /// Log a login failed event
    pub fn log_login_failed(&self, username: &str, ip: Option<String>, reason: &str) {
        let event = SecurityEvent::new(
            SecurityEventType::LoginFailed,
            Some(username.to_string()),
            ip,
            serde_json::json!({ "reason": reason }),
            false,
        );
        self.log(event);
    }

    /// Log a logout event
    pub fn log_logout(&self, username: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::Logout,
            Some(username.to_string()),
            ip,
            serde_json::json!({}),
            true,
        );
        self.log(event);
    }

    /// Log a permission denied event
    pub fn log_permission_denied(&self, username: &str, ip: Option<String>, resource: &str) {
        let event = SecurityEvent::new(
            SecurityEventType::PermissionDenied,
            Some(username.to_string()),
            ip,
            serde_json::json!({ "resource": resource }),
            false,
        );
        self.log(event);
    }

    /// Log a user created event
    pub fn log_user_created(&self, username: &str, creator: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::UserCreated,
            Some(creator.to_string()),
            ip,
            serde_json::json!({ "target_user": username }),
            true,
        );
        self.log(event);
    }

    /// Log a user updated event
    pub fn log_user_updated(&self, username: &str, updater: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::UserUpdated,
            Some(updater.to_string()),
            ip,
            serde_json::json!({ "target_user": username }),
            true,
        );
        self.log(event);
    }

    /// Log a user deleted event
    pub fn log_user_deleted(&self, username: &str, deleter: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::UserDeleted,
            Some(deleter.to_string()),
            ip,
            serde_json::json!({ "target_user": username }),
            true,
        );
        self.log(event);
    }

    /// Log a token refresh event
    pub fn log_token_refresh(&self, username: &str, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::TokenRefresh,
            Some(username.to_string()),
            ip,
            serde_json::json!({}),
            true,
        );
        self.log(event);
    }

    /// Log an unauthorized access event
    pub fn log_unauthorized_access(&self, ip: Option<String>, path: &str) {
        let event = SecurityEvent::new(
            SecurityEventType::UnauthorizedAccess,
            None,
            ip,
            serde_json::json!({ "path": path }),
            false,
        );
        self.log(event);
    }

    /// Log a rate limit exceeded event
    pub fn log_rate_limit_exceeded(&self, username: Option<String>, ip: Option<String>) {
        let event = SecurityEvent::new(
            SecurityEventType::RateLimitExceeded,
            username,
            ip,
            serde_json::json!({}),
            false,
        );
        self.log(event);
    }

    /// Rotate log files
    #[allow(clippy::collapsible_if)]
    async fn rotate_logs(log_file_path: &PathBuf, max_files: usize) {
        // Delete the oldest log if we have too many
        if max_files > 0 {
            let oldest_log = format!("{}.{}", log_file_path.display(), max_files);
            if let Err(e) = tokio::fs::remove_file(&oldest_log).await {
                if !e.kind().eq(&std::io::ErrorKind::NotFound) {
                    eprintln!("Failed to remove old log file {}: {}", oldest_log, e);
                }
            }
        }

        // Rotate existing logs
        for i in (1..max_files).rev() {
            let old_file = format!("{}.{}", log_file_path.display(), i);
            let new_file = format!("{}.{}", log_file_path.display(), i + 1);

            if let Err(e) = tokio::fs::rename(&old_file, &new_file).await {
                if !e.kind().eq(&std::io::ErrorKind::NotFound) {
                    eprintln!("Failed to rotate log file: {}", e);
                }
            }
        }

        // Move current log to .1
        let rotated_log = format!("{}.1", log_file_path.display());
        if let Err(e) = tokio::fs::rename(log_file_path, &rotated_log).await {
            eprintln!("Failed to rotate current log: {}", e);
        }
    }

    /// Check if audit logging is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_audit_logger_creation() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("test_audit.log");

        let config = AuditConfig {
            enabled: true,
            log_file_path: log_path.clone(),
            ..Default::default()
        };

        let logger = AuditLogger::new(config);
        assert!(logger.is_enabled());

        // Log an event
        logger.log_login_success("test_user", Some("127.0.0.1".to_string()));

        // Give the async task time to write
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify the log file was created and contains the event
        let log_content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(log_content.contains("login_success"));
        assert!(log_content.contains("test_user"));
        assert!(log_content.contains("127.0.0.1"));
    }

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

    fn make_logger(temp_dir: &TempDir, filename: &str) -> (AuditLogger, PathBuf) {
        let log_path = temp_dir.path().join(filename);
        let config = AuditConfig {
            enabled: true,
            log_file_path: log_path.clone(),
            ..Default::default()
        };
        (AuditLogger::new(config), log_path)
    }

    async fn read_log_content(log_path: &PathBuf) -> String {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        tokio::fs::read_to_string(log_path)
            .await
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn test_log_login_failed_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "login_failed.log");
        logger.log_login_failed("baduser", Some("10.0.0.1".to_string()), "wrong password");
        let content = read_log_content(&log_path).await;
        assert!(content.contains("login_failed"));
        assert!(content.contains("baduser"));
        assert!(content.contains("10.0.0.1"));
        assert!(content.contains("wrong password"));
        assert!(content.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn test_log_logout_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "logout.log");
        logger.log_logout("user1", Some("10.0.0.2".to_string()));
        let content = read_log_content(&log_path).await;
        assert!(content.contains("logout"));
        assert!(content.contains("user1"));
        assert!(content.contains("\"success\":true"));
    }

    #[tokio::test]
    async fn test_log_permission_denied_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "perm_denied.log");
        logger.log_permission_denied("user2", Some("10.0.0.3".to_string()), "/admin/users");
        let content = read_log_content(&log_path).await;
        assert!(content.contains("permission_denied"));
        assert!(content.contains("user2"));
        assert!(content.contains("/admin/users"));
        assert!(content.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn test_log_user_created_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "user_created.log");
        logger.log_user_created("newuser", "admin", Some("10.0.0.4".to_string()));
        let content = read_log_content(&log_path).await;
        assert!(content.contains("user_created"));
        assert!(content.contains("admin"));
        assert!(content.contains("newuser"));
        assert!(content.contains("\"success\":true"));
    }

    #[tokio::test]
    async fn test_log_user_updated_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "user_updated.log");
        logger.log_user_updated("target_user", "admin", None);
        let content = read_log_content(&log_path).await;
        assert!(content.contains("user_updated"));
        assert!(content.contains("target_user"));
    }

    #[tokio::test]
    async fn test_log_user_deleted_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "user_deleted.log");
        logger.log_user_deleted("deleted_user", "admin", None);
        let content = read_log_content(&log_path).await;
        assert!(content.contains("user_deleted"));
        assert!(content.contains("deleted_user"));
    }

    #[tokio::test]
    async fn test_log_token_refresh_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "token_refresh.log");
        logger.log_token_refresh("user3", Some("10.0.0.5".to_string()));
        let content = read_log_content(&log_path).await;
        assert!(content.contains("token_refresh"));
        assert!(content.contains("user3"));
        assert!(content.contains("\"success\":true"));
    }

    #[tokio::test]
    async fn test_log_unauthorized_access_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "unauthorized.log");
        logger.log_unauthorized_access(Some("10.0.0.6".to_string()), "/api/v1/embed");
        let content = read_log_content(&log_path).await;
        assert!(content.contains("unauthorized_access"));
        assert!(content.contains("/api/v1/embed"));
        assert!(content.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn test_log_rate_limit_exceeded_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "rate_limit.log");
        logger.log_rate_limit_exceeded(Some("user4".to_string()), Some("10.0.0.7".to_string()));
        let content = read_log_content(&log_path).await;
        assert!(content.contains("rate_limit_exceeded"));
        assert!(content.contains("user4"));
    }

    #[tokio::test]
    async fn test_log_rate_limit_exceeded_no_user() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "rate_limit_no_user.log");
        logger.log_rate_limit_exceeded(None, Some("10.0.0.8".to_string()));
        let content = read_log_content(&log_path).await;
        assert!(content.contains("rate_limit_exceeded"));
        assert!(content.contains("10.0.0.8"));
    }

    #[tokio::test]
    async fn test_disabled_logger_does_not_write() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("disabled.log");
        let config = AuditConfig {
            enabled: false,
            log_file_path: log_path.clone(),
            ..Default::default()
        };
        let logger = AuditLogger::new(config);
        assert!(!logger.is_enabled());
        logger.log_login_success("user", None);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        let exists = tokio::fs::try_exists(&log_path).await.unwrap_or(false);
        assert!(!exists, "disabled logger should not create log file");
    }

    #[tokio::test]
    async fn test_log_method_noop_when_disabled() {
        let config = AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let logger = AuditLogger::new(config);
        let event = SecurityEvent::new(
            SecurityEventType::LoginSuccess,
            Some("user".to_string()),
            None,
            serde_json::json!({}),
            true,
        );
        logger.log(event);
    }

    #[tokio::test]
    async fn test_rotate_logs_removes_oldest_and_shifts() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("rotate.log");
        let max_files = 3;
        tokio::fs::write(&log_path, "current".to_string())
            .await
            .unwrap();
        for i in 1..=max_files {
            let rotated = format!("{}.{}", log_path.display(), i);
            tokio::fs::write(&rotated, format!("file_{}", i))
                .await
                .unwrap();
        }
        AuditLogger::rotate_logs(&log_path, max_files).await;

        // rotate_logs: delete .{max_files}, shift .{i} -> .{i+1} for i in
        // (1..max_files).rev(), then move current -> .1
        // Final state: .1=current, .2=file_1, .3=file_2 (oldest slot reused)
        let new_3 = format!("{}.3", log_path.display());
        let content_3 = tokio::fs::read_to_string(&new_3).await.unwrap();
        assert_eq!(content_3, "file_2");

        let new_2 = format!("{}.2", log_path.display());
        let content_2 = tokio::fs::read_to_string(&new_2).await.unwrap();
        assert_eq!(content_2, "file_1");

        let new_1 = format!("{}.1", log_path.display());
        let content_1 = tokio::fs::read_to_string(&new_1).await.unwrap();
        assert_eq!(content_1, "current");

        let original_exists = tokio::fs::try_exists(&log_path).await.unwrap_or(false);
        assert!(!original_exists, "original log should be moved to .1");

        let beyond = format!("{}.4", log_path.display());
        let beyond_exists = tokio::fs::try_exists(&beyond).await.unwrap_or(false);
        assert!(!beyond_exists, "no file should exist beyond max_files");
    }

    #[tokio::test]
    async fn test_rotate_logs_with_max_files_zero() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("no_rotate.log");
        tokio::fs::write(&log_path, "data".to_string())
            .await
            .unwrap();
        AuditLogger::rotate_logs(&log_path, 0).await;

        // max_files=0 skips the oldest removal and the shift loop, but the
        // final rename of the current log to .1 is unconditional
        let rotated = format!("{}.1", log_path.display());
        let rotated_content = tokio::fs::read_to_string(&rotated).await.unwrap();
        assert_eq!(rotated_content, "data");

        let original_exists = tokio::fs::try_exists(&log_path).await.unwrap_or(false);
        assert!(!original_exists, "original log should be moved to .1");

        let beyond = format!("{}.2", log_path.display());
        let beyond_exists = tokio::fs::try_exists(&beyond).await.unwrap_or(false);
        assert!(
            !beyond_exists,
            "no .2 file should be created when max_files=0"
        );
    }

    #[tokio::test]
    async fn test_rotate_logs_nonexistent_source() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("nonexistent.log");
        AuditLogger::rotate_logs(&log_path, 3).await;
        let exists = tokio::fs::try_exists(&log_path).await.unwrap_or(false);
        assert!(!exists);
    }

    #[tokio::test]
    async fn test_write_log_entry_creates_and_appends() {
        let temp_dir = TempDir::new().unwrap();
        let log_path = temp_dir.path().join("write_entry.log");
        AuditLogger::write_log_entry(&log_path, "first line")
            .await
            .unwrap();
        AuditLogger::write_log_entry(&log_path, "second line")
            .await
            .unwrap();
        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(content.contains("first line"));
        assert!(content.contains("second line"));
        assert_eq!(content.matches('\n').count(), 2);
    }

    #[tokio::test]
    async fn test_log_all_methods_with_none_ip() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "all_none.log");
        logger.log_login_success("u", None);
        logger.log_login_failed("u", None, "reason");
        logger.log_logout("u", None);
        logger.log_permission_denied("u", None, "res");
        logger.log_user_created("u", "creator", None);
        logger.log_user_updated("u", "updater", None);
        logger.log_user_deleted("u", "deleter", None);
        logger.log_token_refresh("u", None);
        logger.log_unauthorized_access(None, "/path");
        logger.log_rate_limit_exceeded(None, None);
        let content = read_log_content(&log_path).await;
        assert!(content.contains("login_success"));
        assert!(content.contains("login_failed"));
        assert!(content.contains("logout"));
        assert!(content.contains("permission_denied"));
        assert!(content.contains("user_created"));
        assert!(content.contains("user_updated"));
        assert!(content.contains("user_deleted"));
        assert!(content.contains("token_refresh"));
        assert!(content.contains("unauthorized_access"));
        assert!(content.contains("rate_limit_exceeded"));
    }

    #[tokio::test]
    async fn test_log_channel_closed_does_not_panic() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, _log_path) = make_logger(&temp_dir, "closed.log");
        drop(logger);
    }
}

#[cfg(all(test, feature = "db"))]
mod db_tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_logger_with_db() {
        let pool = crate::db::DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create db pool");
        crate::db::init_schema(&pool)
            .await
            .expect("Failed to init schema");

        let config = AuditConfig::default();
        let logger = AuditLogger::new_with_db(config, pool.clone());

        logger.log_login_success("db_test_user", Some("127.0.0.1".to_string()));

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let session = pool.get_session("admin").await.unwrap();
        let result = session
            .execute_raw("SELECT event_type, user, ip FROM audit_logs")
            .await
            .expect("Failed to query audit logs");

        assert!(result.rows_affected() >= 1, "Audit log should have entries");
    }

    #[tokio::test]
    async fn test_audit_logger_db_disabled() {
        let config = AuditConfig {
            enabled: false,
            ..Default::default()
        };
        let pool = crate::db::DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create db pool");

        let logger = AuditLogger::new_with_db(config, pool);
        assert!(!logger.is_enabled());
    }
}
