// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::auth::types::User;
use crate::error::AppError;
use crate::security::{KeyStore, KeyType};
use chrono::{Duration, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

const DEFAULT_TOKEN_EXPIRATION_HOURS: i64 = 1; // 1 hour
const DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS: i64 = 24; // 24 hours
const DEFAULT_JWT_SECRET_NAME: &str = "jwt_secret";
const MAX_REFRESH_COUNT: u32 = 5; // Maximum number of token refreshes
const MIN_SECRET_LENGTH: usize = 32; // Minimum 32 bytes (256 bits)
const MIN_SECRET_ENTROPY_BITS: f64 = 128.0; // Minimum 128 bits of entropy

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub username: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub exp: usize,
    pub iat: usize,
    pub jti: String, // JWT ID for revocation
}

#[derive(Clone)]
pub struct JwtManager {
    encoding_key: Arc<EncodingKey>,
    decoding_key: Arc<DecodingKey>,
    expiration_hours: i64,
    refresh_token_expiration_hours: i64,
    key_store: Option<Arc<dyn KeyStore>>,
    token_blacklist: Arc<RwLock<HashSet<String>>>,
    refresh_counts: Arc<RwLock<HashMap<String, u32>>>, // Track refresh count per token
}

impl JwtManager {
    /// 计算密钥的熵值（以位为单位）
    fn calculate_entropy(secret: &str) -> f64 {
        if secret.is_empty() {
            return 0.0;
        }

        // 统计字符频率
        let mut freq = [0u64; 256];
        for byte in secret.bytes() {
            freq[byte as usize] += 1;
        }

        // 计算香农熵
        let len = secret.len() as f64;
        let mut entropy = 0.0;
        for &count in freq.iter() {
            if count > 0 {
                let probability = count as f64 / len;
                entropy -= probability * probability.log2();
            }
        }

        entropy * len
    }

    /// 验证密钥强度
    fn validate_secret(secret: &str) -> Result<(), AppError> {
        // 检查最小长度
        if secret.len() < MIN_SECRET_LENGTH {
            return Err(AppError::security_error(format!(
                "JWT secret too short: {} bytes (minimum {} bytes required)",
                secret.len(),
                MIN_SECRET_LENGTH
            )));
        }

        // 检查熵值
        let entropy = Self::calculate_entropy(secret);
        if entropy < MIN_SECRET_ENTROPY_BITS {
            return Err(AppError::security_error(format!(
                "JWT secret has insufficient entropy: {:.2} bits (minimum {:.2} bits required). \
                 Please use a cryptographically secure random key.",
                entropy, MIN_SECRET_ENTROPY_BITS
            )));
        }

        Ok(())
    }

    pub fn new(secret: String) -> Result<Self, AppError> {
        Self::validate_secret(&secret)?;

        let encoding_key = Arc::new(EncodingKey::from_secret(secret.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.as_ref()));

        Ok(Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            refresh_token_expiration_hours: DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS,
            key_store: None,
            token_blacklist: Arc::new(RwLock::new(HashSet::new())),
            refresh_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn new_with_key_store(
        key_store: Arc<dyn KeyStore>,
        secret_name: Option<&str>,
    ) -> Result<Self, AppError> {
        let name = secret_name.unwrap_or(DEFAULT_JWT_SECRET_NAME);

        let secret = key_store
            .get(&KeyType::JwtSecret, name)
            .await?
            .ok_or_else(|| {
                AppError::security_error(format!("JWT secret '{}' not found in key store", name))
            })?;

        // 验证密钥强度
        Self::validate_secret(&secret.value)?;

        let encoding_key = Arc::new(EncodingKey::from_secret(secret.value.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.value.as_ref()));

        Ok(Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            refresh_token_expiration_hours: DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS,
            key_store: Some(key_store),
            token_blacklist: Arc::new(RwLock::new(HashSet::new())),
            refresh_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn with_expiration(mut self, hours: i64) -> Self {
        self.expiration_hours = hours;
        self
    }

    pub async fn rotate_secret(&mut self) -> Result<(), AppError> {
        if let Some(key_store) = &self.key_store {
            let secret_name = DEFAULT_JWT_SECRET_NAME;
            let secret = key_store
                .get(&KeyType::JwtSecret, secret_name)
                .await?
                .ok_or_else(|| {
                    AppError::security_error(format!(
                        "JWT secret '{}' not found in key store",
                        secret_name
                    ))
                })?;

            self.encoding_key = Arc::new(EncodingKey::from_secret(secret.value.as_ref()));
            self.decoding_key = Arc::new(DecodingKey::from_secret(secret.value.as_ref()));
            Ok(())
        } else {
            Err(AppError::security_error(
                "Cannot rotate secret: key store not configured".to_string(),
            ))
        }
    }

    pub fn generate_token(&self, user: &User) -> Result<String, AppError> {
        let now = Utc::now();
        let exp = now + Duration::hours(self.expiration_hours);
        let jti = uuid::Uuid::new_v4().to_string();

        let claims = Claims {
            sub: user.username.clone(),
            username: user.username.clone(),
            role: user.role.clone(),
            permissions: user.permissions.clone(),
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
            jti,
        };

        let header = Header::default();
        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| AppError::AuthenticationError(format!("Failed to generate token: {}", e)))
    }

    pub fn generate_refresh_token(&self, user: &User) -> Result<String, AppError> {
        let now = Utc::now();
        let exp = now + Duration::hours(self.refresh_token_expiration_hours);
        let jti = uuid::Uuid::new_v4().to_string();

        let claims = Claims {
            sub: user.username.clone(),
            username: user.username.clone(),
            role: user.role.clone(),
            permissions: user.permissions.clone(),
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
            jti,
        };

        let header = Header::default();
        encode(&header, &claims, &self.encoding_key).map_err(|e| {
            AppError::AuthenticationError(format!("Failed to generate refresh token: {}", e))
        })
    }

    pub async fn refresh_token(&self, refresh_token: &str) -> Result<String, AppError> {
        // 验证 refresh token
        let claims = self.validate_token(refresh_token).await?;

        // 检查是否在黑名单中
        let blacklist = self.token_blacklist.read().await;
        if blacklist.contains(&claims.jti) {
            return Err(AppError::AuthenticationError(
                "Refresh token has been revoked".to_string(),
            ));
        }
        drop(blacklist);

        // 检查刷新次数
        let mut refresh_counts: tokio::sync::RwLockWriteGuard<'_, HashMap<String, u32>> =
            self.refresh_counts.write().await;
        let refresh_count = refresh_counts.entry(claims.jti.clone()).or_insert(0);

        if *refresh_count >= MAX_REFRESH_COUNT {
            return Err(AppError::AuthenticationError(format!(
                "Refresh token has been used too many times (maximum {} times)",
                MAX_REFRESH_COUNT
            )));
        }

        *refresh_count += 1;
        drop(refresh_counts);

        // 生成新的访问令牌
        let user = User {
            username: claims.username.clone(),
            role: claims.role.clone(),
            permissions: claims.permissions.clone(),
        };

        self.generate_token(&user)
    }

    pub async fn revoke_token(&self, token: &str) -> Result<(), AppError> {
        let claims = self.validate_token(token).await?;
        let mut blacklist = self.token_blacklist.write().await;
        blacklist.insert(claims.jti);
        Ok(())
    }

    pub async fn is_token_revoked(&self, jti: &str) -> bool {
        let blacklist = self.token_blacklist.read().await;
        blacklist.contains(jti)
    }

    pub async fn cleanup_expired_blacklist(&self) {
        let mut blacklist = self.token_blacklist.write().await;

        // 解析并检查每个 token 的过期时间
        let now = Utc::now().timestamp() as usize;
        blacklist.retain(|token| {
            if let Ok(data) = decode::<Claims>(
                token,
                &self.decoding_key,
                &Validation::new(Algorithm::HS256),
            ) {
                data.claims.exp > now
            } else {
                false
            }
        });
    }

    pub async fn validate_token(&self, token: &str) -> Result<Claims, AppError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            AppError::AuthenticationError(format!("Token validation failed: {}", e))
        })?;

        // 检查是否在黑名单中
        let blacklist = self.token_blacklist.read().await;
        if blacklist.contains(&token_data.claims.jti) {
            return Err(AppError::AuthenticationError(
                "Token has been revoked".to_string(),
            ));
        }
        drop(blacklist);

        Ok(token_data.claims)
    }

    pub async fn extract_claims(&self, token: &str) -> Result<Claims, AppError> {
        self.validate_token(token).await
    }

    pub fn get_token_expiration(&self) -> u64 {
        (self.expiration_hours * 3600) as u64
    }

    pub fn get_refresh_token_expiration(&self) -> u64 {
        (self.refresh_token_expiration_hours * 3600) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_generation_and_validation() {
        // 使用符合熵值要求的测试密钥（至少 32 字节，高熵值）
        let secret = "test_secret_key_for_jwt_validation_must_be_long_enough_12345678";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        assert!(!token.is_empty());

        let claims = manager.validate_token(&token).await.unwrap();
        assert_eq!(claims.username, "testuser");
        assert_eq!(claims.role, "user");
        assert_eq!(claims.permissions, vec!["read", "write"]);
    }

    #[tokio::test]
    async fn test_refresh_token_generation() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_refresh_must_be_long_enough_87654321";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let refresh_token = manager.generate_refresh_token(&user).unwrap();
        assert!(!refresh_token.is_empty());

        let new_token = manager.refresh_token(&refresh_token).await.unwrap();
        assert!(!new_token.is_empty());
    }

    #[tokio::test]
    async fn test_token_revocation() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_revocation_must_be_long_enough_abc123xyz";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();

        // 撤销令牌
        manager.revoke_token(&token).await.unwrap();

        // 验证令牌应该失败
        let result = manager.validate_token(&token).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invalid_token() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_invalid_must_be_long_enough_999888777";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let invalid_token = "invalid.token.here";
        let result = manager.validate_token(invalid_token).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_token_with_different_secret() {
        // 使用符合熵值要求的测试密钥
        let secret1 = "test_secret_key_1_for_jwt_different_must_be_long_enough_111222";
        let secret2 = "test_secret_key_2_for_jwt_different_must_be_long_enough_333444";

        let manager1 = JwtManager::new(secret1.to_string()).unwrap();
        let manager2 = JwtManager::new(secret2.to_string()).unwrap();

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager1.generate_token(&user).unwrap();
        let result = manager2.validate_token(&token).await;
        assert!(result.is_err());
    }
}
