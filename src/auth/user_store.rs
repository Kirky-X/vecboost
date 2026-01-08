// Copyright (c) 2025 Kirky.X
use crate::auth::types::User;
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use utoipa::ToSchema;

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

pub struct UserStore {
    users: Arc<RwLock<HashMap<String, StoredUser>>>,
}

impl UserStore {
    pub fn new() -> Self {
        Self {
            users: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_user(&self, user: StoredUser) -> Result<(), AppError> {
        // 验证用户名格式
        validate_username_format(&user.username)?;

        let mut users = self.users.write().map_err(|e| {
            AppError::AuthenticationError(format!("Failed to acquire write lock: {}", e))
        })?;

        if users.contains_key(&user.username) {
            return Err(AppError::AuthenticationError(format!(
                "User {} already exists",
                user.username
            )));
        }

        users.insert(user.username.clone(), user);
        Ok(())
    }

    pub fn get_user(&self, username: &str) -> Result<Option<StoredUser>, AppError> {
        let users = self.users.read().map_err(|e| {
            AppError::AuthenticationError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(users.get(username).cloned())
    }

    pub fn verify_password(&self, username: &str, password: &str) -> Result<User, AppError> {
        let stored_user = self
            .get_user(username)?
            .ok_or_else(|| AppError::AuthenticationError("User not found".to_string()))?;

        let parsed_hash = PasswordHash::new(&stored_user.password_hash).map_err(|e| {
            AppError::AuthenticationError(format!("Invalid password hash format: {}", e))
        })?;

        argon2::Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .map_err(|_| AppError::AuthenticationError("Invalid password".to_string()))?;

        Ok(User {
            username: stored_user.username,
            role: stored_user.role,
            permissions: stored_user.permissions,
        })
    }

    pub fn list_users(&self) -> Result<Vec<String>, AppError> {
        let users = self.users.read().map_err(|e| {
            AppError::AuthenticationError(format!("Failed to acquire read lock: {}", e))
        })?;

        Ok(users.keys().cloned().collect())
    }

    pub fn remove_user(&self, username: &str) -> Result<bool, AppError> {
        let mut users = self.users.write().map_err(|e| {
            AppError::AuthenticationError(format!("Failed to acquire write lock: {}", e))
        })?;

        Ok(users.remove(username).is_some())
    }

    pub fn update_user(&self, username: &str, request: UpdateUserRequest) -> Result<(), AppError> {
        let mut users = self.users.write().map_err(|e| {
            AppError::AuthenticationError(format!("Failed to acquire write lock: {}", e))
        })?;

        let stored_user = users
            .get_mut(username)
            .ok_or_else(|| AppError::AuthenticationError(format!("User {} not found", username)))?;

        if let Some(role) = request.role {
            stored_user.role = role;
        }

        if let Some(permissions) = request.permissions {
            stored_user.permissions = permissions;
        }

        Ok(())
    }
}

impl Default for UserStore {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hash_password(password: &str) -> Result<String, AppError> {
    let salt = SaltString::generate(&mut rand::thread_rng());
    let argon2 = argon2::Argon2::default();

    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| AppError::AuthenticationError(format!("Failed to hash password: {}", e)))?;

    Ok(password_hash.to_string())
}

pub fn create_default_admin_user(username: &str, password: &str) -> Result<StoredUser, AppError> {
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
/// 要求：
/// - 至少 12 个字符
/// - 包含大写字母
/// - 包含小写字母
/// - 包含数字
/// - 包含特殊字符
/// - 不包含常见弱密码模式
pub fn validate_password_complexity(password: &str) -> Result<(), AppError> {
    // 检查最小长度
    if password.len() < 12 {
        return Err(AppError::ValidationError(
            "密码长度必须至少为 12 个字符".to_string(),
        ));
    }

    // 检查是否包含大写字母
    if !password.chars().any(|c| c.is_uppercase()) {
        return Err(AppError::ValidationError(
            "密码必须包含至少一个大写字母".to_string(),
        ));
    }

    // 检查是否包含小写字母
    if !password.chars().any(|c| c.is_lowercase()) {
        return Err(AppError::ValidationError(
            "密码必须包含至少一个小写字母".to_string(),
        ));
    }

    // 检查是否包含数字
    if !password.chars().any(|c| c.is_ascii_digit()) {
        return Err(AppError::ValidationError(
            "密码必须包含至少一个数字".to_string(),
        ));
    }

    // 检查是否包含特殊字符
    let special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`";
    if !password.chars().any(|c| special_chars.contains(c)) {
        return Err(AppError::ValidationError(
            "密码必须包含至少一个特殊字符 (!@#$%^&*()_+-=[]{}|;:,.<>?/~`)".to_string(),
        ));
    }

    // 检查常见弱密码模式
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
            return Err(AppError::ValidationError(format!(
                "密码不能包含常见弱密码模式: {}",
                pattern
            )));
        }
    }

    // 检查连续字符模式（如 "12345", "abcde"）
    for i in 0..password.len().saturating_sub(4) {
        let chars: Vec<char> = password.chars().collect();
        let slice = &chars[i..i + 5];

        // 检查连续数字
        let is_consecutive_digits = slice.windows(2).all(|w| {
            w[0].is_ascii_digit() && w[1].is_ascii_digit() && (w[1] as i32 - w[0] as i32).abs() == 1
        });

        // 检查连续字母
        let is_consecutive_letters = slice.windows(2).all(|w| {
            w[0].is_ascii_alphabetic()
                && w[1].is_ascii_alphabetic()
                && (w[1] as i32 - w[0] as i32).abs() == 1
        });

        if is_consecutive_digits || is_consecutive_letters {
            return Err(AppError::ValidationError(
                "密码不能包含连续的字符序列（如 12345 或 abcde）".to_string(),
            ));
        }
    }

    // 检查重复字符模式（如 "aaaaa", "11111"）
    for i in 0..password.len().saturating_sub(4) {
        let chars: Vec<char> = password.chars().collect();
        let slice = &chars[i..i + 5];

        let is_repeated = slice.windows(2).all(|w| w[0] == w[1]);
        if is_repeated {
            return Err(AppError::ValidationError(
                "密码不能包含重复的字符序列（如 aaaaa 或 11111）".to_string(),
            ));
        }
    }

    Ok(())
}

/// 验证用户名格式
///
/// 要求：
/// - 长度 3-32 个字符
/// - 只允许字母、数字、下划线和连字符
/// - 必须以字母开头
pub fn validate_username_format(username: &str) -> Result<(), AppError> {
    // 检查长度
    if username.len() < 3 || username.len() > 32 {
        return Err(AppError::ValidationError(
            "用户名长度必须在 3 到 32 个字符之间".to_string(),
        ));
    }

    // 检查是否以字母开头
    if !username
        .chars()
        .next()
        .map(|c| c.is_ascii_alphabetic())
        .unwrap_or(false)
    {
        return Err(AppError::ValidationError(
            "用户名必须以字母开头".to_string(),
        ));
    }

    // 检查字符格式：只允许字母、数字、下划线和连字符
    let username_regex = Regex::new(r"^[a-zA-Z][a-zA-Z0-9_-]*$").unwrap();
    if !username_regex.is_match(username) {
        return Err(AppError::ValidationError(
            "用户名只能包含字母、数字、下划线和连字符".to_string(),
        ));
    }

    Ok(())
}
