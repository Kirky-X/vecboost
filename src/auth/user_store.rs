// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 用户存储模块
//!
//! 当 `db` feature 启用时,基于 dbnexus 提供持久化存储;
//! 否则使用内存 `HashMap` 提供临时存储。
//! 所有公共方法均为 `async`,以统一两种后端的 API。

use crate::auth::User;
use crate::error::VecboostError;
use argon2::{PasswordHash, PasswordHasher, PasswordVerifier};
use regex::Regex;
#[cfg(feature = "db")]
use sea_orm::{ConnectionTrait, DatabaseBackend, Statement, Value};
use serde::{Deserialize, Serialize};
#[cfg(not(feature = "db"))]
use std::collections::HashMap;
use std::sync::Arc;
use utoipa::ToSchema;
use zeroize::Zeroize;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StoredUser {
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateUserRequest {
    pub username: String,
    pub password: String,
    pub role: Option<String>,
    pub permissions: Option<Vec<String>>,
}

pub struct UpdateUserRequest {
    pub username: String,
    pub role: Option<String>,
    pub permissions: Option<Vec<String>>,
}

/// 用户存储,根据 feature 切换后端
pub struct UserStore {
    #[cfg(feature = "db")]
    pool: Arc<crate::db::DbPool>,
    #[cfg(not(feature = "db"))]
    users: Arc<tokio::sync::RwLock<HashMap<String, StoredUser>>>,
}

impl UserStore {
    /// 创建新的用户存储
    ///
    /// 当 `db` feature 启用时,需要传入 `DbPool`;否则使用内存后端。
    #[cfg(feature = "db")]
    pub fn new(pool: Arc<crate::db::DbPool>) -> Self {
        Self { pool }
    }

    #[cfg(not(feature = "db"))]
    pub fn new() -> Self {
        Self {
            users: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    /// 添加用户
    ///
    /// # Errors
    ///
    /// 如果用户名格式无效或用户已存在,返回 `VecboostError`
    pub async fn add_user(&self, user: StoredUser) -> Result<(), VecboostError> {
        validate_username_format(&user.username)?;

        #[cfg(feature = "db")]
        {
            let session =
                self.pool.get_session("admin").await.map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get session: {e}"))
                })?;
            let conn = session.connection().map_err(|e| {
                VecboostError::InternalError(format!("Failed to get connection: {e}"))
            })?;
            let permissions_json = serde_json::to_string(&user.permissions).map_err(|e| {
                VecboostError::InternalError(format!("Failed to serialize permissions: {e}"))
            })?;
            let result = conn
                .execute_raw(Statement::from_sql_and_values(
                    DatabaseBackend::Sqlite,
                    "INSERT INTO users (username, password_hash, role, permissions) VALUES (?, ?, ?, ?)",
                    [
                        Value::String(Some(user.username.clone())),
                        Value::String(Some(user.password_hash.clone())),
                        Value::String(Some(user.role.clone())),
                        Value::String(Some(permissions_json)),
                    ],
                ))
                .await
                .map_err(|e| VecboostError::InternalError(format!("Failed to insert user: {e}")))?;
            let _ = result;
            Ok(())
        }

        #[cfg(not(feature = "db"))]
        {
            let mut users = self.users.write().await;
            if users.contains_key(&user.username) {
                return Err(VecboostError::AuthenticationError(format!(
                    "User {} already exists",
                    user.username
                )));
            }
            users.insert(user.username.clone(), user);
            Ok(())
        }
    }

    /// 获取用户
    ///
    /// # Errors
    ///
    /// 如果查询失败,返回 `VecboostError`
    pub async fn get_user(&self, username: &str) -> Result<Option<StoredUser>, VecboostError> {
        #[cfg(feature = "db")]
        {
            let session =
                self.pool.get_session("admin").await.map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get session: {e}"))
                })?;
            let conn = session.connection().map_err(|e| {
                VecboostError::InternalError(format!("Failed to get connection: {e}"))
            })?;
            let stmt = Statement::from_sql_and_values(
                DatabaseBackend::Sqlite,
                "SELECT username, password_hash, role, permissions FROM users WHERE username = ?",
                [Value::String(Some(username.to_string()))],
            );
            let rows = conn
                .query_all_raw(stmt)
                .await
                .map_err(|e| VecboostError::InternalError(format!("Failed to query user: {e}")))?;
            if rows.is_empty() {
                return Ok(None);
            }
            let row = &rows[0];
            let stored_username: String = row.try_get("", "username").map_err(|e| {
                VecboostError::InternalError(format!("Failed to get username: {e}"))
            })?;
            let password_hash: String = row.try_get("", "password_hash").map_err(|e| {
                VecboostError::InternalError(format!("Failed to get password_hash: {e}"))
            })?;
            let role: String = row
                .try_get("", "role")
                .map_err(|e| VecboostError::InternalError(format!("Failed to get role: {e}")))?;
            let permissions_json: String = row.try_get("", "permissions").map_err(|e| {
                VecboostError::InternalError(format!("Failed to get permissions: {e}"))
            })?;
            let permissions: Vec<String> =
                serde_json::from_str(&permissions_json).map_err(|e| {
                    VecboostError::InternalError(format!("Failed to parse permissions: {e}"))
                })?;
            Ok(Some(StoredUser {
                username: stored_username,
                password_hash,
                role,
                permissions,
            }))
        }

        #[cfg(not(feature = "db"))]
        {
            let users = self.users.read().await;
            Ok(users.get(username).cloned())
        }
    }

    /// 验证密码
    ///
    /// # Errors
    ///
    /// 如果用户不存在、密码哈希格式无效或密码不匹配,返回 `VecboostError`
    pub async fn verify_password(
        &self,
        username: &str,
        password: &str,
    ) -> Result<User, VecboostError> {
        let stored_user = self
            .get_user(username)
            .await?
            .ok_or_else(|| VecboostError::AuthenticationError("User not found".to_string()))?;

        let parsed_hash = PasswordHash::new(&stored_user.password_hash).map_err(|e| {
            VecboostError::AuthenticationError(format!("Invalid password hash format: {e}"))
        })?;

        argon2::Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .map_err(|_| VecboostError::AuthenticationError("Invalid password".to_string()))?;

        Ok(User {
            username: stored_user.username,
            role: stored_user.role,
            permissions: stored_user.permissions,
        })
    }

    /// 列出所有用户名
    ///
    /// # Errors
    ///
    /// 如果查询失败,返回 `VecboostError`
    pub async fn list_users(&self) -> Result<Vec<String>, VecboostError> {
        #[cfg(feature = "db")]
        {
            let session =
                self.pool.get_session("admin").await.map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get session: {e}"))
                })?;
            let conn = session.connection().map_err(|e| {
                VecboostError::InternalError(format!("Failed to get connection: {e}"))
            })?;
            let stmt =
                Statement::from_string(DatabaseBackend::Sqlite, "SELECT username FROM users");
            let rows = conn
                .query_all_raw(stmt)
                .await
                .map_err(|e| VecboostError::InternalError(format!("Failed to list users: {e}")))?;
            let mut usernames = Vec::with_capacity(rows.len());
            for row in &rows {
                let username: String = row.try_get("", "username").map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get username: {e}"))
                })?;
                usernames.push(username);
            }
            Ok(usernames)
        }

        #[cfg(not(feature = "db"))]
        {
            let users = self.users.read().await;
            Ok(users.keys().cloned().collect())
        }
    }

    /// 删除用户
    ///
    /// # Errors
    ///
    /// 如果删除失败,返回 `VecboostError`
    pub async fn remove_user(&self, username: &str) -> Result<bool, VecboostError> {
        #[cfg(feature = "db")]
        {
            let session =
                self.pool.get_session("admin").await.map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get session: {e}"))
                })?;
            let conn = session.connection().map_err(|e| {
                VecboostError::InternalError(format!("Failed to get connection: {e}"))
            })?;
            let result = conn
                .execute_raw(Statement::from_sql_and_values(
                    DatabaseBackend::Sqlite,
                    "DELETE FROM users WHERE username = ?",
                    [Value::String(Some(username.to_string()))],
                ))
                .await
                .map_err(|e| VecboostError::InternalError(format!("Failed to delete user: {e}")))?;
            Ok(result.rows_affected() > 0)
        }

        #[cfg(not(feature = "db"))]
        {
            let mut users = self.users.write().await;
            Ok(users.remove(username).is_some())
        }
    }

    /// 更新用户
    ///
    /// # Errors
    ///
    /// 如果用户不存在或更新失败,返回 `VecboostError`
    pub async fn update_user(
        &self,
        username: &str,
        request: UpdateUserRequest,
    ) -> Result<(), VecboostError> {
        #[cfg(feature = "db")]
        {
            let session =
                self.pool.get_session("admin").await.map_err(|e| {
                    VecboostError::InternalError(format!("Failed to get session: {e}"))
                })?;
            let conn = session.connection().map_err(|e| {
                VecboostError::InternalError(format!("Failed to get connection: {e}"))
            })?;

            // 构建动态 UPDATE 语句
            let mut sets: Vec<&str> = Vec::new();
            let mut values: Vec<Value> = Vec::new();
            if let Some(role) = request.role {
                sets.push("role = ?");
                values.push(Value::String(Some(role)));
            }
            if let Some(permissions) = request.permissions {
                let permissions_json = serde_json::to_string(&permissions).map_err(|e| {
                    VecboostError::InternalError(format!("Failed to serialize permissions: {e}"))
                })?;
                sets.push("permissions = ?");
                values.push(Value::String(Some(permissions_json)));
            }
            if sets.is_empty() {
                return Ok(());
            }
            values.push(Value::String(Some(username.to_string())));
            let sql = format!("UPDATE users SET {} WHERE username = ?", sets.join(", "));
            let result = conn
                .execute_raw(Statement::from_sql_and_values(
                    DatabaseBackend::Sqlite,
                    sql,
                    values,
                ))
                .await
                .map_err(|e| VecboostError::InternalError(format!("Failed to update user: {e}")))?;
            if result.rows_affected() == 0 {
                return Err(VecboostError::AuthenticationError(format!(
                    "User {username} not found"
                )));
            }
            Ok(())
        }

        #[cfg(not(feature = "db"))]
        {
            let mut users = self.users.write().await;
            let stored_user = users.get_mut(username).ok_or_else(|| {
                VecboostError::AuthenticationError(format!("User {username} not found"))
            })?;
            if let Some(role) = request.role {
                stored_user.role = role;
            }
            if let Some(permissions) = request.permissions {
                stored_user.permissions = permissions;
            }
            Ok(())
        }
    }
}

#[cfg(not(feature = "db"))]
impl Default for UserStore {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hash_password(password: &str) -> Result<String, VecboostError> {
    let argon2 = argon2::Argon2::default();

    let password_hash = argon2
        .hash_password(password.as_bytes())
        .map_err(|e| VecboostError::AuthenticationError(format!("Failed to hash password: {e}")))?;

    let mut password_vec = password.as_bytes().to_vec();
    password_vec.zeroize();

    Ok(password_hash.to_string())
}

pub fn create_default_admin_user(
    username: &str,
    password: &str,
) -> Result<StoredUser, VecboostError> {
    let password_hash = hash_password(password)?;

    Ok(StoredUser {
        username: username.to_string(),
        password_hash,
        role: "admin".to_string(),
        permissions: vec![
            "embedding:read".to_string(),
            "embedding:write".to_string(),
            "model:read".to_string(),
            "model:write".to_string(),
            "model:switch".to_string(),
            "user:read".to_string(),
            "user:write".to_string(),
            "user:delete".to_string(),
        ],
    })
}

/// 验证密码复杂度
///
/// 要求:
/// - 至少 12 个字符
/// - 包含大写字母
/// - 包含小写字母
/// - 包含数字
/// - 包含特殊字符
/// - 不包含常见弱密码模式
pub fn validate_password_complexity(password: &str) -> Result<(), VecboostError> {
    if password.len() < 12 {
        return Err(VecboostError::ValidationError(
            "密码长度必须至少为 12 个字符".to_string(),
        ));
    }

    if !password.chars().any(|c| c.is_uppercase()) {
        return Err(VecboostError::ValidationError(
            "密码必须包含至少一个大写字母".to_string(),
        ));
    }

    if !password.chars().any(|c| c.is_lowercase()) {
        return Err(VecboostError::ValidationError(
            "密码必须包含至少一个小写字母".to_string(),
        ));
    }

    if !password.chars().any(|c| c.is_ascii_digit()) {
        return Err(VecboostError::ValidationError(
            "密码必须包含至少一个数字".to_string(),
        ));
    }

    let special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`";
    if !password.chars().any(|c| special_chars.contains(c)) {
        return Err(VecboostError::ValidationError(
            "密码必须包含至少一个特殊字符 (!@#$%^&*()_+-=[]{}|;:,.<>?/~`)".to_string(),
        ));
    }

    let weak_patterns = vec![
        "password",
        "Password",
        "PASSWORD",
        "123456",
        "12345678",
        "123456789",
        "qwerty",
        "QWERTY",
        "abc123",
        "ABC123",
        "admin",
        "Admin",
        "ADMIN",
        "letmein",
        "LetMeIn",
        "welcome",
        "Welcome",
        "monkey",
        "Monkey",
        "dragon",
        "Dragon",
        "master",
        "Master",
        "hello",
        "Hello",
        "football",
        "Football",
        "iloveyou",
        "ILoveYou",
    ];

    let lower_password = password.to_lowercase();
    for pattern in weak_patterns {
        if lower_password.contains(&pattern.to_lowercase()) {
            return Err(VecboostError::ValidationError(format!(
                "密码不能包含常见弱密码模式: {pattern}"
            )));
        }
    }

    for i in 0..password.len().saturating_sub(4) {
        let chars: Vec<char> = password.chars().collect();
        let slice = &chars[i..i + 5];

        let is_consecutive_digits = slice.windows(2).all(|w| {
            w[0].is_ascii_digit() && w[1].is_ascii_digit() && (w[1] as i32 - w[0] as i32).abs() == 1
        });

        let is_consecutive_letters = slice.windows(2).all(|w| {
            w[0].is_ascii_alphabetic()
                && w[1].is_ascii_alphabetic()
                && (w[1] as i32 - w[0] as i32).abs() == 1
        });

        if is_consecutive_digits || is_consecutive_letters {
            return Err(VecboostError::ValidationError(
                "密码不能包含连续的字符序列（如 12345 或 abcde）".to_string(),
            ));
        }
    }

    for i in 0..password.len().saturating_sub(4) {
        let chars: Vec<char> = password.chars().collect();
        let slice = &chars[i..i + 5];

        let is_repeated = slice.windows(2).all(|w| w[0] == w[1]);
        if is_repeated {
            return Err(VecboostError::ValidationError(
                "密码不能包含重复的字符序列（如 aaaaa 或 11111）".to_string(),
            ));
        }
    }

    Ok(())
}

/// 验证用户名格式
///
/// 要求:
/// - 长度 3-32 个字符
/// - 只允许字母、数字、下划线和连字符
/// - 必须以字母开头
pub fn validate_username_format(username: &str) -> Result<(), VecboostError> {
    if username.len() < 3 || username.len() > 32 {
        return Err(VecboostError::ValidationError(
            "用户名长度必须在 3 到 32 个字符之间".to_string(),
        ));
    }

    if !username
        .chars()
        .next()
        .map(|c| c.is_ascii_alphabetic())
        .unwrap_or(false)
    {
        return Err(VecboostError::ValidationError(
            "用户名必须以字母开头".to_string(),
        ));
    }

    let username_regex = Regex::new(r"^[a-zA-Z][a-zA-Z0-9_-]*$").unwrap();
    if !username_regex.is_match(username) {
        return Err(VecboostError::ValidationError(
            "用户名只能包含字母、数字、下划线和连字符".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "db")]
    async fn make_store_with_users() -> UserStore {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_user_store.db");
        let url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = crate::db::DbPool::new(&url).await.unwrap();
        crate::db::init_schema(&pool).await.unwrap();
        std::mem::forget(temp_dir);
        UserStore::new(Arc::new(pool))
    }

    #[cfg(not(feature = "db"))]
    async fn make_store_with_users() -> UserStore {
        UserStore::new()
    }

    #[tokio::test]
    async fn test_add_and_get_user() {
        let store = make_store_with_users().await;
        let user = StoredUser {
            username: "testuser".to_string(),
            password_hash: "hash123".to_string(),
            role: "admin".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        };
        store.add_user(user).await.unwrap();
        let fetched = store.get_user("testuser").await.unwrap().unwrap();
        assert_eq!(fetched.username, "testuser");
        assert_eq!(fetched.password_hash, "hash123");
        assert_eq!(fetched.role, "admin");
        assert_eq!(
            fetched.permissions,
            vec!["read".to_string(), "write".to_string()]
        );
    }

    #[tokio::test]
    async fn test_get_nonexistent_user() {
        let store = make_store_with_users().await;
        let result = store.get_user("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_add_duplicate_user() {
        let store = make_store_with_users().await;
        let user = StoredUser {
            username: "duplicate".to_string(),
            password_hash: "hash".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };
        store.add_user(user).await.unwrap();
        // 再次添加应该失败(UNIQUE 约束或 HashMap 已存在)
        let user2 = StoredUser {
            username: "duplicate".to_string(),
            password_hash: "hash2".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };
        let result = store.add_user(user2).await;
        assert!(result.is_err(), "Adding duplicate user should fail");
    }

    #[tokio::test]
    async fn test_list_users() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "user1".to_string(),
                password_hash: "h1".to_string(),
                role: "user".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        store
            .add_user(StoredUser {
                username: "user2".to_string(),
                password_hash: "h2".to_string(),
                role: "user".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        let mut names = store.list_users().await.unwrap();
        names.sort();
        assert_eq!(names, vec!["user1".to_string(), "user2".to_string()]);
    }

    #[tokio::test]
    async fn test_remove_user() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "toremove".to_string(),
                password_hash: "h".to_string(),
                role: "user".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        let removed = store.remove_user("toremove").await.unwrap();
        assert!(removed);
        let removed_again = store.remove_user("toremove").await.unwrap();
        assert!(!removed_again);
    }

    #[tokio::test]
    async fn test_update_user() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "toupdate".to_string(),
                password_hash: "h".to_string(),
                role: "user".to_string(),
                permissions: vec!["read".to_string()],
            })
            .await
            .unwrap();
        store
            .update_user(
                "toupdate",
                UpdateUserRequest {
                    username: "toupdate".to_string(),
                    role: Some("admin".to_string()),
                    permissions: Some(vec!["read".to_string(), "write".to_string()]),
                },
            )
            .await
            .unwrap();
        let updated = store.get_user("toupdate").await.unwrap().unwrap();
        assert_eq!(updated.role, "admin");
        assert_eq!(
            updated.permissions,
            vec!["read".to_string(), "write".to_string()]
        );
    }

    #[tokio::test]
    async fn test_verify_password_success() {
        let store = make_store_with_users().await;
        let password_hash = hash_password("TestPassword123!").unwrap();
        store
            .add_user(StoredUser {
                username: "verifyuser".to_string(),
                password_hash,
                role: "admin".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        let user = store
            .verify_password("verifyuser", "TestPassword123!")
            .await
            .unwrap();
        assert_eq!(user.username, "verifyuser");
        assert_eq!(user.role, "admin");
    }

    #[tokio::test]
    async fn test_verify_password_failure() {
        let store = make_store_with_users().await;
        let password_hash = hash_password("TestPassword123!").unwrap();
        store
            .add_user(StoredUser {
                username: "verifyuser2".to_string(),
                password_hash,
                role: "admin".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        let result = store
            .verify_password("verifyuser2", "WrongPassword123!")
            .await;
        assert!(result.is_err());
    }

    // ===== hash_password 测试 =====

    #[test]
    fn test_hash_password_returns_valid_hash() {
        let hash = hash_password("MySecurePwd123!").unwrap();
        assert!(!hash.is_empty());
        // 应是有效的 argon2 hash (以 $argon2 开头)
        assert!(hash.starts_with("$argon2"));
    }

    #[test]
    fn test_hash_password_different_passwords_different_hashes() {
        let hash1 = hash_password("PasswordOne123!").unwrap();
        let hash2 = hash_password("PasswordTwo456!").unwrap();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_password_same_password_different_hashes() {
        // 相同密码因随机盐应产生不同 hash
        let hash1 = hash_password("SamePassword789!").unwrap();
        let hash2 = hash_password("SamePassword789!").unwrap();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_password_empty_password() {
        // 空密码也应能被 hash (不调用 complexity 验证)
        let hash = hash_password("").unwrap();
        assert!(hash.starts_with("$argon2"));
    }

    // ===== create_default_admin_user 测试 =====

    #[test]
    fn test_create_default_admin_user() {
        let user = create_default_admin_user("admin", "AdminPassword123!").unwrap();
        assert_eq!(user.username, "admin");
        assert_eq!(user.role, "admin");
        assert_eq!(user.permissions.len(), 8);
        assert!(user.permissions.contains(&"embedding:read".to_string()));
        assert!(user.permissions.contains(&"embedding:write".to_string()));
        assert!(user.permissions.contains(&"model:read".to_string()));
        assert!(user.permissions.contains(&"model:write".to_string()));
        assert!(user.permissions.contains(&"model:switch".to_string()));
        assert!(user.permissions.contains(&"user:read".to_string()));
        assert!(user.permissions.contains(&"user:write".to_string()));
        assert!(user.permissions.contains(&"user:delete".to_string()));
        assert!(user.password_hash.starts_with("$argon2"));
    }

    // ===== validate_password_complexity 测试 =====

    #[test]
    fn test_password_complexity_valid() {
        // 满足所有要求的复杂密码
        assert!(validate_password_complexity("Xy9!kM2#pQ7&zL").is_ok());
    }

    #[test]
    fn test_password_complexity_too_short() {
        let result = validate_password_complexity("Xy9!kM2#");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecboostError::ValidationError(_)
        ));
    }

    #[test]
    fn test_password_complexity_no_uppercase() {
        let result = validate_password_complexity("xy9!km2#pq7&zl");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_no_lowercase() {
        let result = validate_password_complexity("XY9!KM2#PQ7&ZL");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_no_digit() {
        let result = validate_password_complexity("Xy!kM#pQ&zLaXy");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_no_special_char() {
        let result = validate_password_complexity("Xy9kM2pQ7zLaXy");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_weak_pattern_password() {
        // 包含 "password" 弱模式
        let result = validate_password_complexity("Xy9!Password12#z");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_weak_pattern_123456() {
        // 包含 "123456" 弱模式
        let result = validate_password_complexity("Xy9!abcDEF123456");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_consecutive_digits() {
        // 包含连续数字序列 "12345"
        let result = validate_password_complexity("Xy9!12345abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_consecutive_letters() {
        // 包含连续字母序列 "abcde"
        let result = validate_password_complexity("Xy9!abcde123");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_repeated_chars() {
        // 包含重复字符序列 "aaaaa"
        let result = validate_password_complexity("Xy9!aaaaa123");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_complexity_empty() {
        let result = validate_password_complexity("");
        assert!(result.is_err());
    }

    // ===== validate_username_format 测试 =====

    #[test]
    fn test_username_format_valid() {
        assert!(validate_username_format("abc").is_ok());
        assert!(validate_username_format("username").is_ok());
        assert!(validate_username_format("user_name").is_ok());
        assert!(validate_username_format("user-name").is_ok());
        assert!(validate_username_format("User123").is_ok());
        assert!(validate_username_format("a").is_err()); // 太短
        assert!(validate_username_format(&"a".repeat(32)).is_ok());
    }

    #[test]
    fn test_username_format_too_short() {
        let result = validate_username_format("ab");
        assert!(result.is_err());
    }

    #[test]
    fn test_username_format_too_long() {
        let result = validate_username_format(&"a".repeat(33));
        assert!(result.is_err());
    }

    #[test]
    fn test_username_format_not_letter_start() {
        assert!(validate_username_format("1username").is_err());
        assert!(validate_username_format("_username").is_err());
        assert!(validate_username_format("-username").is_err());
    }

    #[test]
    fn test_username_format_invalid_chars() {
        assert!(validate_username_format("user@name").is_err());
        assert!(validate_username_format("user.name").is_err());
        assert!(validate_username_format("user name").is_err());
    }

    #[test]
    fn test_username_format_empty() {
        let result = validate_username_format("");
        assert!(result.is_err());
    }

    #[test]
    fn test_username_format_boundary_length_3() {
        // 长度 3 是边界 — 应通过
        assert!(validate_username_format("abc").is_ok());
    }

    #[test]
    fn test_username_format_boundary_length_32() {
        // 长度 32 是边界 — 应通过
        assert!(validate_username_format(&"a".repeat(32)).is_ok());
    }

    // ===== UserStore 错误路径测试 =====

    #[tokio::test]
    async fn test_add_user_invalid_username() {
        let store = make_store_with_users().await;
        let result = store
            .add_user(StoredUser {
                username: "1invalid".to_string(),
                password_hash: "hash".to_string(),
                role: "user".to_string(),
                permissions: vec![],
            })
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecboostError::ValidationError(_)
        ));
    }

    #[tokio::test]
    async fn test_verify_password_user_not_found() {
        let store = make_store_with_users().await;
        let result = store.verify_password("nonexistent", "anypassword").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecboostError::AuthenticationError(_)
        ));
    }

    #[tokio::test]
    async fn test_verify_password_invalid_hash_format() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "badhash".to_string(),
                password_hash: "not-a-valid-hash".to_string(),
                role: "user".to_string(),
                permissions: vec![],
            })
            .await
            .unwrap();
        let result = store.verify_password("badhash", "anypassword").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecboostError::AuthenticationError(_)
        ));
    }

    #[tokio::test]
    async fn test_list_users_empty() {
        let store = make_store_with_users().await;
        let users = store.list_users().await.unwrap();
        assert!(users.is_empty());
    }

    #[tokio::test]
    async fn test_remove_user_nonexistent() {
        let store = make_store_with_users().await;
        let removed = store.remove_user("ghost").await.unwrap();
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_update_user_nonexistent() {
        let store = make_store_with_users().await;
        let result = store
            .update_user(
                "ghost",
                UpdateUserRequest {
                    username: "ghost".to_string(),
                    role: Some("user".to_string()),
                    permissions: None,
                },
            )
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VecboostError::AuthenticationError(_)
        ));
    }

    #[tokio::test]
    async fn test_update_user_only_role() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "roleonly".to_string(),
                password_hash: "h".to_string(),
                role: "user".to_string(),
                permissions: vec!["read".to_string()],
            })
            .await
            .unwrap();
        store
            .update_user(
                "roleonly",
                UpdateUserRequest {
                    username: "roleonly".to_string(),
                    role: Some("admin".to_string()),
                    permissions: None,
                },
            )
            .await
            .unwrap();
        let updated = store.get_user("roleonly").await.unwrap().unwrap();
        assert_eq!(updated.role, "admin");
        // permissions 应保持不变
        assert_eq!(updated.permissions, vec!["read".to_string()]);
    }

    #[tokio::test]
    async fn test_update_user_only_permissions() {
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "permonly".to_string(),
                password_hash: "h".to_string(),
                role: "user".to_string(),
                permissions: vec!["read".to_string()],
            })
            .await
            .unwrap();
        store
            .update_user(
                "permonly",
                UpdateUserRequest {
                    username: "permonly".to_string(),
                    role: None,
                    permissions: Some(vec!["read".to_string(), "write".to_string()]),
                },
            )
            .await
            .unwrap();
        let updated = store.get_user("permonly").await.unwrap().unwrap();
        // role 应保持不变
        assert_eq!(updated.role, "user");
        assert_eq!(
            updated.permissions,
            vec!["read".to_string(), "write".to_string()]
        );
    }

    #[tokio::test]
    async fn test_update_user_no_changes() {
        // role 和 permissions 都为 None — 不更新任何字段
        let store = make_store_with_users().await;
        store
            .add_user(StoredUser {
                username: "nochange".to_string(),
                password_hash: "h".to_string(),
                role: "user".to_string(),
                permissions: vec!["read".to_string()],
            })
            .await
            .unwrap();
        store
            .update_user(
                "nochange",
                UpdateUserRequest {
                    username: "nochange".to_string(),
                    role: None,
                    permissions: None,
                },
            )
            .await
            .unwrap();
        let user = store.get_user("nochange").await.unwrap().unwrap();
        assert_eq!(user.role, "user");
        assert_eq!(user.permissions, vec!["read".to_string()]);
    }

    #[tokio::test]
    async fn test_add_and_remove_multiple_users() {
        let store = make_store_with_users().await;
        // 添加多个用户
        for i in 0..5 {
            store
                .add_user(StoredUser {
                    username: format!("multi{i}"),
                    password_hash: "h".to_string(),
                    role: "user".to_string(),
                    permissions: vec![],
                })
                .await
                .unwrap();
        }
        assert_eq!(store.list_users().await.unwrap().len(), 5);

        // 删除部分用户
        store.remove_user("multi0").await.unwrap();
        store.remove_user("multi1").await.unwrap();
        let remaining = store.list_users().await.unwrap();
        assert_eq!(remaining.len(), 3);
        assert!(!remaining.contains(&"multi0".to_string()));
        assert!(!remaining.contains(&"multi1".to_string()));
        assert!(remaining.contains(&"multi2".to_string()));
    }

    #[tokio::test]
    async fn test_verify_password_correct_after_hash() {
        // 端到端:hash → store → verify
        let store = make_store_with_users().await;
        let password = "End2EndPwd99!xy";
        let hash = hash_password(password).unwrap();
        store
            .add_user(StoredUser {
                username: "e2euser".to_string(),
                password_hash: hash,
                role: "user".to_string(),
                permissions: vec!["read".to_string()],
            })
            .await
            .unwrap();
        let user = store.verify_password("e2euser", password).await.unwrap();
        assert_eq!(user.username, "e2euser");
        assert_eq!(user.role, "user");
        assert_eq!(user.permissions, vec!["read".to_string()]);
    }

    // ===== 并发访问测试 =====

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_user_operations() {
        let store = Arc::new(make_store_with_users().await);
        let mut handles = Vec::new();

        for i in 0..10 {
            let store_clone = store.clone();
            handles.push(tokio::spawn(async move {
                let username = format!("concurrent_user_{i}");
                store_clone
                    .add_user(StoredUser {
                        username: username.clone(),
                        password_hash: "h".to_string(),
                        role: "user".to_string(),
                        permissions: vec![],
                    })
                    .await
                    .unwrap();
                // 添加后应能获取
                let user = store_clone.get_user(&username).await.unwrap();
                assert!(user.is_some());
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let users = store.list_users().await.unwrap();
        assert_eq!(users.len(), 10);
    }
}
