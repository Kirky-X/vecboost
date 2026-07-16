// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::path::PathBuf;
#[cfg(feature = "db")]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{Duration, MissedTickBehavior};

use crate::error::VecboostError;
#[cfg(feature = "db")]
use sea_orm::{ConnectionTrait, DatabaseBackend, Statement, Value};

use super::{AuditConfig, SecurityEvent, SecurityEventType};

/// Writer task commands: enqueue event or request synchronous flush ack.
enum LoggerCommand {
    Event(SecurityEvent),
    Flush(oneshot::Sender<()>),
}

/// Audit logger — async batched writes via background task.
///
/// Events are pushed to an unbounded mpsc channel; a background writer task
/// holds a persistent `tokio::fs::File` handle and flushes when either 100
/// entries are buffered or 1s has elapsed since the last flush. File size is
/// tracked in-memory via `AtomicU64` (no per-event `fs::metadata` call).
pub struct AuditLogger {
    config: AuditConfig,
    sender: Option<mpsc::UnboundedSender<LoggerCommand>>,
    _handle: Option<JoinHandle<()>>,
}

impl AuditLogger {
    /// Create a new file-backed audit logger.
    pub fn new(config: AuditConfig) -> Self {
        if !config.enabled {
            return Self {
                config,
                sender: None,
                _handle: None,
            };
        }

        let (sender, receiver) = mpsc::unbounded_channel::<LoggerCommand>();
        let log_file_path = config.log_file_path.clone();
        let max_file_size = (config.max_file_size_mb * 1024 * 1024) as u64;
        let max_files = config.max_files;

        let handle = tokio::spawn(async move {
            Self::run_file_writer(receiver, log_file_path, max_file_size, max_files).await;
        });

        Self {
            config,
            sender: Some(sender),
            _handle: Some(handle),
        }
    }

    /// Create a new audit logger with database backend.
    #[cfg(feature = "db")]
    pub fn new_with_db(config: AuditConfig, db_pool: crate::db::DbPool) -> Self {
        if !config.enabled {
            return Self {
                config,
                sender: None,
                _handle: None,
            };
        }

        let (sender, receiver) = mpsc::unbounded_channel::<LoggerCommand>();
        let pool = Arc::new(db_pool);

        let handle = tokio::spawn(async move {
            Self::run_db_writer(receiver, pool).await;
        });

        Self {
            config,
            sender: Some(sender),
            _handle: Some(handle),
        }
    }

    /// File-mode writer task: persistent File handle + batched flush.
    async fn run_file_writer(
        mut receiver: mpsc::UnboundedReceiver<LoggerCommand>,
        log_file_path: PathBuf,
        max_file_size: u64,
        max_files: usize,
    ) {
        if let Some(parent) = log_file_path.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            eprintln!("Failed to create log directory: {}", e);
            return;
        }

        let mut file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to open audit log file: {}", e);
                return;
            }
        };

        let current_size = AtomicU64::new(
            tokio::fs::metadata(&log_file_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0),
        );

        let mut buffer: Vec<String> = Vec::with_capacity(100);
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        let mut consecutive_errors: u32 = 0;
        const ERROR_THRESHOLD: u32 = 10;

        loop {
            tokio::select! {
                cmd = receiver.recv() => {
                    match cmd {
                        Some(LoggerCommand::Event(event)) => {
                            if let Ok(line) = serde_json::to_string(&event) {
                                current_size.fetch_add(line.len() as u64 + 1, Ordering::Relaxed);
                                buffer.push(line);
                                if buffer.len() >= 100 {
                                    consecutive_errors = Self::flush_buffer(
                                        &mut file,
                                        &mut buffer,
                                        &log_file_path,
                                        max_file_size,
                                        max_files,
                                        &current_size,
                                        consecutive_errors,
                                    ).await;
                                    if consecutive_errors >= ERROR_THRESHOLD { break; }
                                }
                            }
                        }
                        Some(LoggerCommand::Flush(ack)) => {
                            // Drain pending events before flushing
                            while let Ok(cmd) = receiver.try_recv() {
                                if let LoggerCommand::Event(event) = cmd
                                    && let Ok(line) = serde_json::to_string(&event)
                                {
                                    current_size.fetch_add(line.len() as u64 + 1, Ordering::Relaxed);
                                    buffer.push(line);
                                }
                                // Drop nested Flush acks — caller's flush is in flight
                            }
                            consecutive_errors = Self::flush_buffer(
                                &mut file,
                                &mut buffer,
                                &log_file_path,
                                max_file_size,
                                max_files,
                                &current_size,
                                consecutive_errors,
                            ).await;
                            let _ = ack.send(());
                            if consecutive_errors >= ERROR_THRESHOLD { break; }
                        }
                        None => {
                            // Channel closed: final flush and exit
                            Self::flush_buffer(
                                &mut file,
                                &mut buffer,
                                &log_file_path,
                                max_file_size,
                                max_files,
                                &current_size,
                                consecutive_errors,
                            ).await;
                            break;
                        }
                    }
                }
                _ = interval.tick() => {
                    if !buffer.is_empty() {
                        consecutive_errors = Self::flush_buffer(
                            &mut file,
                            &mut buffer,
                            &log_file_path,
                            max_file_size,
                            max_files,
                            &current_size,
                            consecutive_errors,
                        ).await;
                        if consecutive_errors >= ERROR_THRESHOLD { break; }
                    }
                }
            }
        }
    }

    /// DB-mode writer task: per-event INSERT (no batching — single-stmt tx cost).
    #[cfg(feature = "db")]
    async fn run_db_writer(
        mut receiver: mpsc::UnboundedReceiver<LoggerCommand>,
        pool: Arc<crate::db::DbPool>,
    ) {
        while let Some(cmd) = receiver.recv().await {
            match cmd {
                LoggerCommand::Event(event) => {
                    if let Err(e) = Self::write_to_db(&pool, &event).await {
                        eprintln!("Failed to write audit log to db: {}", e);
                    }
                }
                LoggerCommand::Flush(ack) => {
                    // Drain pending events
                    while let Ok(cmd) = receiver.try_recv() {
                        if let LoggerCommand::Event(event) = cmd
                            && let Err(e) = Self::write_to_db(&pool, &event).await
                        {
                            eprintln!("Failed to write audit log to db: {}", e);
                        }
                    }
                    let _ = ack.send(());
                }
            }
        }
    }

    /// Flush the in-memory buffer to the persistent file handle. Handles rotation
    /// (close → rename → reopen) when `current_size` exceeds `max_file_size`.
    async fn flush_buffer(
        file: &mut tokio::fs::File,
        buffer: &mut Vec<String>,
        log_file_path: &PathBuf,
        max_file_size: u64,
        max_files: usize,
        current_size: &AtomicU64,
        consecutive_errors: u32,
    ) -> u32 {
        if buffer.is_empty() {
            return consecutive_errors;
        }

        let size = current_size.load(Ordering::Relaxed);
        let needs_rotation = size > max_file_size;

        if needs_rotation {
            let _ = file.flush().await;
            Self::rotate_logs(log_file_path, max_files).await;
            current_size.store(0, Ordering::Relaxed);
            *file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file_path)
                .await
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to reopen log file after rotation: {}", e);
                    buffer.clear();
                    return consecutive_errors + 1;
                }
            };
        }

        let mut data = String::new();
        for line in buffer.iter() {
            data.push_str(line);
            data.push('\n');
        }
        match file.write_all(data.as_bytes()).await {
            Ok(_) => {
                let _ = file.flush().await;
                if needs_rotation {
                    current_size.store(data.len() as u64, Ordering::Relaxed);
                }
                buffer.clear();
                0
            }
            Err(e) => {
                eprintln!("Failed to write audit log batch: {}", e);
                consecutive_errors + 1
            }
        }
    }

    /// Write audit event to database.
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

    /// Log a security event (non-blocking).
    pub fn log(&self, event: SecurityEvent) {
        if !self.config.enabled {
            return;
        }
        if let Some(sender) = &self.sender {
            let _ = sender.send(LoggerCommand::Event(event));
        }
    }

    /// Synchronously wait for all queued events to be flushed to disk/db.
    pub async fn flush(&self) -> Result<(), VecboostError> {
        if let Some(sender) = &self.sender {
            let (tx, rx) = oneshot::channel();
            let _ = sender.send(LoggerCommand::Flush(tx));
            let _ = rx.await;
        }
        Ok(())
    }

    /// Log a login success event.
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

    /// Log a login failed event.
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

    /// Log a logout event.
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

    /// Log a permission denied event.
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

    /// Log a user created event.
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

    /// Log a user updated event.
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

    /// Log a user deleted event.
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

    /// Log a token refresh event.
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

    /// Log an unauthorized access event.
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

    /// Log a rate limit exceeded event.
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

    /// Rotate log files.
    #[allow(clippy::collapsible_if)]
    async fn rotate_logs(log_file_path: &PathBuf, max_files: usize) {
        if max_files > 0 {
            let oldest_log = format!("{}.{}", log_file_path.display(), max_files);
            if let Err(e) = tokio::fs::remove_file(&oldest_log).await {
                if !e.kind().eq(&std::io::ErrorKind::NotFound) {
                    eprintln!("Failed to remove old log file {}: {}", oldest_log, e);
                }
            }
        }

        for i in (1..max_files).rev() {
            let old_file = format!("{}.{}", log_file_path.display(), i);
            let new_file = format!("{}.{}", log_file_path.display(), i + 1);

            if let Err(e) = tokio::fs::rename(&old_file, &new_file).await {
                if !e.kind().eq(&std::io::ErrorKind::NotFound) {
                    eprintln!("Failed to rotate log file: {}", e);
                }
            }
        }

        let rotated_log = format!("{}.1", log_file_path.display());
        if let Err(e) = tokio::fs::rename(log_file_path, &rotated_log).await {
            eprintln!("Failed to rotate current log: {}", e);
        }
    }

    /// Check if audit logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditConfig, SecurityEvent, SecurityEventType};
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

        logger.log_login_success("test_user", Some("127.0.0.1".to_string()));
        logger.flush().await.unwrap();

        let log_content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(log_content.contains("login_success"));
        assert!(log_content.contains("test_user"));
        assert!(log_content.contains("127.0.0.1"));
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
        tokio::fs::read_to_string(log_path)
            .await
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn test_log_login_failed_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "login_failed.log");
        logger.log_login_failed("baduser", Some("10.0.0.1".to_string()), "wrong password");
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
        let content = read_log_content(&log_path).await;
        assert!(content.contains("user_updated"));
        assert!(content.contains("target_user"));
    }

    #[tokio::test]
    async fn test_log_user_deleted_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "user_deleted.log");
        logger.log_user_deleted("deleted_user", "admin", None);
        logger.flush().await.unwrap();
        let content = read_log_content(&log_path).await;
        assert!(content.contains("user_deleted"));
        assert!(content.contains("deleted_user"));
    }

    #[tokio::test]
    async fn test_log_token_refresh_writes_to_file() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "token_refresh.log");
        logger.log_token_refresh("user3", Some("10.0.0.5".to_string()));
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
        let content = read_log_content(&log_path).await;
        assert!(content.contains("rate_limit_exceeded"));
        assert!(content.contains("user4"));
    }

    #[tokio::test]
    async fn test_log_rate_limit_exceeded_no_user() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "rate_limit_no_user.log");
        logger.log_rate_limit_exceeded(None, Some("10.0.0.8".to_string()));
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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
        logger.flush().await.unwrap();
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

    /// Batch flush triggers when buffer reaches 100 entries (no explicit flush call).
    #[tokio::test]
    async fn test_batch_flush_at_100_entries() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "batch.log");
        for i in 0..150 {
            logger.log_login_success(&format!("user{}", i), None);
        }
        // Give writer task time to process the batch flush trigger
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        logger.flush().await.unwrap();
        let content = read_log_content(&log_path).await;
        // All 150 entries should be present
        assert_eq!(content.matches("login_success").count(), 150);
    }

    /// Interval-based flush triggers after 1s even with small buffer.
    #[tokio::test]
    async fn test_interval_flush_within_1s() {
        let temp_dir = TempDir::new().unwrap();
        let (logger, log_path) = make_logger(&temp_dir, "interval.log");
        logger.log_login_success("solo_user", None);
        // Wait for interval tick (>1s)
        tokio::time::sleep(tokio::time::Duration::from_millis(1200)).await;
        let content = read_log_content(&log_path).await;
        assert!(
            content.contains("solo_user"),
            "interval flush should write within 1s"
        );
    }
}

#[cfg(all(test, feature = "db"))]
mod db_tests {
    use super::*;
    use crate::audit::AuditConfig;

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
        logger.flush().await.unwrap();

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
