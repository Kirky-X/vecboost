#![allow(unused)]

use crate::auth::types::User;
use crate::error::AppError;
use password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct StoredUser {
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub permissions: Vec<String>,
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
            "read".to_string(),
            "write".to_string(),
            "admin".to_string(),
            "model:switch".to_string(),
            "model:list".to_string(),
        ],
    })
}

pub fn create_default_user(username: &str, password: &str) -> Result<StoredUser, AppError> {
    let password_hash = hash_password(password)?;

    Ok(StoredUser {
        username: username.to_string(),
        password_hash,
        role: "user".to_string(),
        permissions: vec!["read".to_string(), "write".to_string()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hashing() {
        let password = "test_password_123";
        let hash = hash_password(password).unwrap();
        assert!(!hash.is_empty());
        assert_ne!(hash, password);
    }

    #[test]
    fn test_user_store() {
        let store = UserStore::new();
        let user = create_default_user("testuser", "password123").unwrap();

        store.add_user(user).unwrap();
        let retrieved = store.get_user("testuser").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().username, "testuser");
    }

    #[test]
    fn test_password_verification() {
        let store = UserStore::new();
        let user = create_default_user("testuser", "password123").unwrap();

        store.add_user(user).unwrap();

        let verified_user = store.verify_password("testuser", "password123").unwrap();
        assert_eq!(verified_user.username, "testuser");

        let wrong_password = store.verify_password("testuser", "wrongpassword");
        assert!(wrong_password.is_err());
    }
}
