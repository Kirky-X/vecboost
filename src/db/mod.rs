// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 数据库模块 — 基于 dbnexus 提供持久化能力。
//!
//! 提供 [`DbPool`] wrapper、schema 初始化和 CRUD 辅助函数。
//! 仅在 `db` feature 启用时可用。

use crate::error::VecboostError;
use dbnexus::{DbPool as NexusDbPool, Session};

/// VecBoost 数据库连接池 wrapper
///
/// 包装 `dbnexus::DbPool`,提供与 VecBoost 错误类型集成的接口。
#[derive(Clone)]
pub struct DbPool {
    inner: NexusDbPool,
}

impl DbPool {
    /// 创建新的数据库连接池
    ///
    /// # Arguments
    ///
    /// * `url` - 数据库连接 URL(如 `sqlite::memory:` 或 `sqlite:data.db`)
    ///
    /// # Errors
    ///
    /// 如果连接失败,返回 `VecboostError::InternalError`
    pub async fn new(url: &str) -> Result<Self, VecboostError> {
        let pool = NexusDbPool::new(url)
            .await
            .map_err(|e| VecboostError::InternalError(format!("Failed to create db pool: {e}")))?;
        Ok(Self { inner: pool })
    }

    /// 获取数据库会话
    ///
    /// # Arguments
    ///
    /// * `role` - 用户角色(如 `admin`、`user`)
    ///
    /// # Errors
    ///
    /// 如果角色未定义或获取连接失败,返回 `VecboostError::InternalError`
    pub async fn get_session(&self, role: &str) -> Result<Session, VecboostError> {
        self.inner
            .get_session(role)
            .await
            .map_err(|e| VecboostError::InternalError(format!("Failed to get session: {e}")))
    }

    /// 获取内部 dbnexus 连接池引用(高级用法)
    pub fn inner(&self) -> &NexusDbPool {
        &self.inner
    }
}

/// 初始化数据库 schema(建表)
///
/// 创建 `users` 和 `audit_logs` 表(如果不存在)。
///
/// # Errors
///
/// 如果建表失败,返回 `VecboostError::InternalError`
pub async fn init_schema(pool: &DbPool) -> Result<(), VecboostError> {
    let session = pool.get_session("admin").await?;

    session
        .execute_raw_ddl(
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                permissions TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )
        .await
        .map_err(|e| VecboostError::InternalError(format!("Failed to create users table: {e}")))?;

    session
        .execute_raw_ddl(
            "CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user TEXT,
                ip TEXT,
                request_id TEXT,
                user_agent TEXT,
                details TEXT NOT NULL DEFAULT '{}',
                success INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )
        .await
        .map_err(|e| {
            VecboostError::InternalError(format!("Failed to create audit_logs table: {e}"))
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_pool_memory_sqlite() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");
        let status = pool.inner().status();
        assert_eq!(status.total, 0, "New pool should have 0 connections");
    }

    #[tokio::test]
    async fn test_get_session_admin_role() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        let session = pool
            .get_session("admin")
            .await
            .expect("Failed to get admin session");
        assert_eq!(session.role(), "admin");
    }

    #[tokio::test]
    async fn test_get_session_user_role_rejected() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        let result = pool.get_session("user").await;
        assert!(
            result.is_err(),
            "user role should be rejected without explicit permission config"
        );
    }

    #[tokio::test]
    async fn test_init_schema_creates_tables() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        init_schema(&pool).await.expect("Failed to init schema");

        let session = pool
            .get_session("admin")
            .await
            .expect("Failed to get session");

        // 验证 users 表存在
        let result = session
            .execute_raw("SELECT COUNT(*) FROM users")
            .await
            .expect("Failed to query users table");
        assert!(result.rows_affected() == 0 || result.rows_affected() == 1);

        // 验证 audit_logs 表存在
        let result = session
            .execute_raw("SELECT COUNT(*) FROM audit_logs")
            .await
            .expect("Failed to query audit_logs table");
        assert!(result.rows_affected() == 0 || result.rows_affected() == 1);
    }

    #[tokio::test]
    async fn test_users_crud() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        init_schema(&pool).await.expect("Failed to init schema");
        let session = pool
            .get_session("admin")
            .await
            .expect("Failed to get session");

        // Insert
        session
            .execute_raw(
                "INSERT INTO users (username, password_hash, role) VALUES ('testuser', 'hash123', 'admin')",
            )
            .await
            .expect("Failed to insert user");

        // Select (通过查询验证插入成功)
        let select_result = session
            .execute_raw("SELECT username FROM users WHERE username = 'testuser'")
            .await
            .expect("Failed to select user");
        assert_eq!(select_result.rows_affected(), 1);

        // Update
        session
            .execute_raw("UPDATE users SET role = 'user' WHERE username = 'testuser'")
            .await
            .expect("Failed to update user");

        // Delete
        let delete_result = session
            .execute_raw("DELETE FROM users WHERE username = 'testuser'")
            .await
            .expect("Failed to delete user");
        assert_eq!(delete_result.rows_affected(), 1);
    }

    #[tokio::test]
    async fn test_audit_logs_insert() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        init_schema(&pool).await.expect("Failed to init schema");
        let session = pool
            .get_session("admin")
            .await
            .expect("Failed to get session");

        session
            .execute_raw(
                "INSERT INTO audit_logs (event_type, user, ip, details, success) \
                 VALUES ('login_success', 'admin', '127.0.0.1', '{}', 1)",
            )
            .await
            .expect("Failed to insert audit log");

        let result = session
            .execute_raw("SELECT COUNT(*) FROM audit_logs WHERE event_type = 'login_success'")
            .await
            .expect("Failed to query audit_logs");
        assert_eq!(result.rows_affected(), 1);
    }

    #[tokio::test]
    async fn test_pool_clone_shares_connections() {
        let pool = DbPool::new("sqlite::memory:")
            .await
            .expect("Failed to create pool");
        let cloned = pool.clone();

        // 验证 clone 后两者都能获取 session(NexusDbPool 内部 Arc 共享)
        let session1 = pool
            .get_session("admin")
            .await
            .expect("Failed to get session from original");
        let session2 = cloned
            .get_session("admin")
            .await
            .expect("Failed to get session from clone");
        assert_eq!(session1.role(), session2.role());
    }
}
