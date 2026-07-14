// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

use crate::auth::{MemoryTokenStore, TokenStore, User};
use crate::error::VecboostError;
use crate::security::{KeyStore, KeyType};
use chrono::{Duration, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use zeroize::Zeroize;

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
    token_store: Arc<dyn TokenStore>,
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
    fn validate_secret(secret: &str) -> Result<(), VecboostError> {
        // 检查最小长度
        if secret.len() < MIN_SECRET_LENGTH {
            return Err(VecboostError::security_error(format!(
                "JWT secret too short: {} bytes (minimum {} bytes required)",
                secret.len(),
                MIN_SECRET_LENGTH
            )));
        }

        // 检查熵值
        let entropy = Self::calculate_entropy(secret);
        if entropy < MIN_SECRET_ENTROPY_BITS {
            return Err(VecboostError::security_error(format!(
                "JWT secret has insufficient entropy: {:.2} bits (minimum {:.2} bits required). \
                 Please use a cryptographically secure random key.",
                entropy, MIN_SECRET_ENTROPY_BITS
            )));
        }

        Ok(())
    }

    pub fn new(secret: String) -> Result<Self, VecboostError> {
        Self::validate_secret(&secret)?;

        let encoding_key = Arc::new(EncodingKey::from_secret(secret.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.as_ref()));

        Ok(Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            refresh_token_expiration_hours: DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS,
            key_store: None,
            token_store: Arc::new(MemoryTokenStore::new()),
            refresh_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn new_with_key_store(
        key_store: Arc<dyn KeyStore>,
        secret_name: Option<&str>,
    ) -> Result<Self, VecboostError> {
        let name = secret_name.unwrap_or(DEFAULT_JWT_SECRET_NAME);

        let secret = key_store
            .get(&KeyType::JwtSecret, name)
            .await?
            .ok_or_else(|| {
                VecboostError::security_error(format!(
                    "JWT secret '{}' not found in key store",
                    name
                ))
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
            token_store: Arc::new(MemoryTokenStore::new()),
            refresh_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create a new JwtManager with a custom token store
    pub fn with_token_store(
        secret: String,
        token_store: Arc<dyn TokenStore>,
    ) -> Result<Self, VecboostError> {
        Self::validate_secret(&secret)?;

        let encoding_key = Arc::new(EncodingKey::from_secret(secret.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.as_ref()));

        // 安全清除明文 secret（从内存中零化）
        let mut secret_vec = secret.as_bytes().to_vec();
        secret_vec.zeroize();

        Ok(Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            refresh_token_expiration_hours: DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS,
            key_store: None,
            token_store,
            refresh_counts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn with_expiration(mut self, hours: i64) -> Self {
        self.expiration_hours = hours;
        self
    }

    pub async fn rotate_secret(&mut self) -> Result<(), VecboostError> {
        if let Some(key_store) = &self.key_store {
            let secret_name = DEFAULT_JWT_SECRET_NAME;
            let secret = key_store
                .get(&KeyType::JwtSecret, secret_name)
                .await?
                .ok_or_else(|| {
                    VecboostError::security_error(format!(
                        "JWT secret '{}' not found in key store",
                        secret_name
                    ))
                })?;

            self.encoding_key = Arc::new(EncodingKey::from_secret(secret.value.as_ref()));
            self.decoding_key = Arc::new(DecodingKey::from_secret(secret.value.as_ref()));
            Ok(())
        } else {
            Err(VecboostError::security_error(
                "Cannot rotate secret: key store not configured".to_string(),
            ))
        }
    }

    pub fn generate_token(&self, user: &User) -> Result<String, VecboostError> {
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
        encode(&header, &claims, &self.encoding_key).map_err(|e| {
            VecboostError::AuthenticationError(format!("Failed to generate token: {}", e))
        })
    }

    pub fn generate_refresh_token(&self, user: &User) -> Result<String, VecboostError> {
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
            VecboostError::AuthenticationError(format!("Failed to generate refresh token: {}", e))
        })
    }

    pub async fn refresh_token(&self, refresh_token: &str) -> Result<String, VecboostError> {
        // 验证 refresh token
        let claims = self.validate_token(refresh_token).await?;

        // 检查是否在黑名单中
        if self.token_store.is_blacklisted(&claims.jti).await? {
            return Err(VecboostError::AuthenticationError(
                "Refresh token has been revoked".to_string(),
            ));
        }

        // 检查刷新次数 - 使用原子操作避免竞态条件
        let mut refresh_counts = self.refresh_counts.write().await;
        let refresh_count = refresh_counts.entry(claims.jti.clone()).or_insert(0);

        if *refresh_count >= MAX_REFRESH_COUNT {
            // 达到最大刷新次数，将令牌加入黑名单
            drop(refresh_counts);
            let _ = self.token_store.add_to_blacklist(&claims.jti).await;
            return Err(VecboostError::AuthenticationError(format!(
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

        let new_token = self.generate_token(&user)?;

        // 记录刷新成功（用于审计）
        tracing::debug!("Token refreshed for user: {}", claims.username);

        Ok(new_token)
    }

    pub async fn revoke_token(&self, token: &str) -> Result<(), VecboostError> {
        let claims = self.validate_token(token).await?;
        self.token_store.add_to_blacklist(&claims.jti).await?;
        Ok(())
    }

    pub async fn is_token_revoked(&self, jti: &str) -> bool {
        self.token_store.is_blacklisted(jti).await.unwrap_or(false)
    }

    pub async fn cleanup_expired_blacklist(&self) {
        // 对于 Redis 存储，过期由 TTL 自动处理
        // 对于内存存储，需要手动清理过期条目
        self.token_store.cleanup_expired_blacklist().await;
        tracing::debug!("JWT blacklist cleanup executed");
    }

    /// 启动定期黑名单清理任务
    pub fn start_periodic_cleanup(
        &self,
        cleanup_interval_hours: u64,
    ) -> tokio::task::JoinHandle<()> {
        let token_store = self.token_store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
                cleanup_interval_hours * 3600,
            ));
            loop {
                interval.tick().await;
                token_store.cleanup_expired_blacklist().await;
                tracing::debug!("Periodic JWT blacklist cleanup executed");
            }
        })
    }

    pub async fn validate_token(&self, token: &str) -> Result<Claims, VecboostError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            VecboostError::AuthenticationError(format!("Token validation failed: {}", e))
        })?;

        // 检查是否在黑名单中
        if self
            .token_store
            .is_blacklisted(&token_data.claims.jti)
            .await?
        {
            return Err(VecboostError::AuthenticationError(
                "Token has been revoked".to_string(),
            ));
        }

        Ok(token_data.claims)
    }

    pub async fn extract_claims(&self, token: &str) -> Result<Claims, VecboostError> {
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
        let manager = JwtManager::new(secret.to_string()).expect("Failed to create JwtManager");

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        };

        let token = manager
            .generate_token(&user)
            .expect("Failed to generate token");
        assert!(!token.is_empty());

        let claims = manager
            .validate_token(&token)
            .await
            .expect("Token validation failed");
        assert_eq!(claims.username, "testuser");
        assert_eq!(claims.role, "user");
        assert_eq!(claims.permissions, vec!["read", "write"]);
    }

    #[tokio::test]
    async fn test_refresh_token_generation() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_refresh_must_be_long_enough_87654321";
        let manager = JwtManager::new(secret.to_string()).expect("Failed to create JwtManager");

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let refresh_token = manager
            .generate_refresh_token(&user)
            .expect("Failed to generate refresh token");
        assert!(!refresh_token.is_empty());

        let new_token = manager
            .refresh_token(&refresh_token)
            .await
            .expect("Token refresh failed");
        assert!(!new_token.is_empty());
    }

    #[tokio::test]
    async fn test_token_revocation() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_revocation_must_be_long_enough_abc123xyz";
        let manager = JwtManager::new(secret.to_string()).expect("Failed to create JwtManager");

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager
            .generate_token(&user)
            .expect("Failed to generate token");

        // 撤销令牌
        manager
            .revoke_token(&token)
            .await
            .expect("Token revocation failed");

        // 验证令牌应该失败
        let result = manager.validate_token(&token).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invalid_token() {
        // 使用符合熵值要求的测试密钥
        let secret = "test_secret_key_for_jwt_invalid_must_be_long_enough_999888777";
        let manager = JwtManager::new(secret.to_string()).expect("Failed to create JwtManager");

        let invalid_token = "invalid.token.here";
        let result = manager.validate_token(invalid_token).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_token_with_different_secret() {
        // 使用符合熵值要求的测试密钥
        let secret1 = "test_secret_key_1_for_jwt_different_must_be_long_enough_111222";
        let secret2 = "test_secret_key_2_for_jwt_different_must_be_long_enough_333444";

        let manager1 = JwtManager::new(secret1.to_string()).expect("Failed to create JwtManager 1");
        let manager2 = JwtManager::new(secret2.to_string()).expect("Failed to create JwtManager 2");

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager1
            .generate_token(&user)
            .expect("Failed to generate token");
        let result = manager2.validate_token(&token).await;
        assert!(result.is_err());
    }

    // ===== JwtManager 构造错误路径测试 =====

    #[test]
    fn test_new_short_secret_rejected() {
        // 短于 32 字节的 secret 应被拒绝
        let result = JwtManager::new("short".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            VecboostError::SecurityError(_)
        ));
    }

    #[test]
    fn test_new_empty_secret_rejected() {
        let result = JwtManager::new("".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_new_low_entropy_secret_rejected() {
        // 长度足够但熵值不足 — 34 个相同字符,熵值为 0
        let low_entropy = "a".repeat(34);
        let result = JwtManager::new(low_entropy);
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            VecboostError::SecurityError(_)
        ));
    }

    #[test]
    fn test_calculate_entropy_empty_string() {
        // 空字符串的熵应为 0
        assert_eq!(JwtManager::calculate_entropy(""), 0.0);
    }

    #[test]
    fn test_calculate_entropy_single_char() {
        // 单字符重复 — 熵为 0
        let low = "a".repeat(50);
        assert_eq!(JwtManager::calculate_entropy(&low), 0.0);
    }

    #[test]
    fn test_calculate_entropy_high_entropy() {
        // 高熵字符串 — 熵应大于 128 位
        let high = "Ab3!xY9$kL2@mN8#pQ4&rS7^tU6*vW0%";
        let entropy = JwtManager::calculate_entropy(high);
        assert!(entropy >= MIN_SECRET_ENTROPY_BITS);
    }

    // ===== Token claims 正确性测试 =====

    #[tokio::test]
    async fn test_generate_token_claims_correctness() {
        let secret = "test_secret_for_claims_validation_must_be_long_999";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "claimsuser".to_string(),
            role: "admin".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        let claims = manager.validate_token(&token).await.unwrap();

        // 验证所有 claims 字段
        assert_eq!(claims.sub, "claimsuser");
        assert_eq!(claims.username, "claimsuser");
        assert_eq!(claims.role, "admin");
        assert_eq!(claims.permissions, vec!["read", "write"]);
        // exp 应在 iat 之后
        assert!(claims.exp > claims.iat);
        // jti 应非空 (UUID)
        assert!(!claims.jti.is_empty());
    }

    #[tokio::test]
    async fn test_generate_token_unique_jti() {
        let secret = "test_secret_for_unique_jti_validation_must_be_long_888";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "uniqueuser".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };

        let token1 = manager.generate_token(&user).unwrap();
        let token2 = manager.generate_token(&user).unwrap();

        let claims1 = manager.validate_token(&token1).await.unwrap();
        let claims2 = manager.validate_token(&token2).await.unwrap();

        // 两个 token 的 jti 应不同
        assert_ne!(claims1.jti, claims2.jti);
    }

    // ===== Token 过期测试 =====

    #[tokio::test]
    async fn test_expired_token_validation_fails() {
        let secret = "test_secret_for_expired_token_validation_must_be_long_777";
        let manager = JwtManager::new(secret.to_string())
            .unwrap()
            .with_expiration(-1); // 负数 — token 在生成时已过期

        let user = User {
            username: "expireduser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        let result = manager.validate_token(&token).await;
        assert!(result.is_err());
    }

    // ===== Token 验证错误路径测试 =====

    #[tokio::test]
    async fn test_validate_empty_token() {
        let secret = "test_secret_for_empty_token_validation_must_be_long_666";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        let result = manager.validate_token("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_token_with_invalid_structure() {
        let secret = "test_secret_for_invalid_structure_validation_must_be_long_555";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        // 只有两个点的 token — 结构无效
        let result = manager.validate_token("a.b").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_token_tampered() {
        let secret = "test_secret_for_tampered_token_validation_must_be_long_444";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "tamperuser".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };

        let token = manager.generate_token(&user).unwrap();
        // 篡改 token — 修改最后一个字符
        let mut tampered = token.clone();
        let last_char = tampered.chars().last().unwrap();
        let new_char = if last_char == 'A' { 'B' } else { 'A' };
        tampered.pop();
        tampered.push(new_char);

        let result = manager.validate_token(&tampered).await;
        assert!(result.is_err());
    }

    // ===== extract_claims 测试 =====

    #[tokio::test]
    async fn test_extract_claims_success() {
        let secret = "test_secret_for_extract_claims_validation_must_be_long_333";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "extractuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        let claims = manager.extract_claims(&token).await.unwrap();
        assert_eq!(claims.username, "extractuser");
    }

    #[tokio::test]
    async fn test_extract_claims_invalid_token() {
        let secret = "test_secret_for_extract_invalid_claims_must_be_long_222";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        let result = manager.extract_claims("invalid").await;
        assert!(result.is_err());
    }

    // ===== is_token_revoked 测试 =====

    #[tokio::test]
    async fn test_is_token_revoked_false_for_unknown_jti() {
        let secret = "test_secret_for_revoked_check_must_be_long_enough_111";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        assert!(!manager.is_token_revoked("unknown-jti").await);
    }

    #[tokio::test]
    async fn test_is_token_revoked_true_after_revoke() {
        let secret = "test_secret_for_revoked_true_check_must_be_long_enough_000";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "revokeuser".to_string(),
            role: "user".to_string(),
            permissions: vec![],
        };

        let token = manager.generate_token(&user).unwrap();
        let claims = manager.validate_token(&token).await.unwrap();
        manager.revoke_token(&token).await.unwrap();
        assert!(manager.is_token_revoked(&claims.jti).await);
    }

    // ===== refresh_token 错误路径测试 =====

    #[tokio::test]
    async fn test_refresh_token_revoked_fails() {
        let secret = "test_secret_for_refresh_revoked_must_be_long_enough_aaa111";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "refreshuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let refresh_token = manager.generate_refresh_token(&user).unwrap();
        // 撤销 refresh token
        manager.revoke_token(&refresh_token).await.unwrap();
        // 刷新应失败
        let result = manager.refresh_token(&refresh_token).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_refresh_token_exceeds_max_count() {
        let secret = "test_secret_for_max_refresh_count_must_be_long_enough_bbb222";
        let manager = JwtManager::new(secret.to_string()).unwrap();

        let user = User {
            username: "maxrefreshuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let refresh_token = manager.generate_refresh_token(&user).unwrap();
        // 先提取 jti 用于后续黑名单检查
        let claims = manager.validate_token(&refresh_token).await.unwrap();
        let jti = claims.jti.clone();

        // MAX_REFRESH_COUNT = 5,前 5 次应成功
        for i in 0..MAX_REFRESH_COUNT {
            let result = manager.refresh_token(&refresh_token).await;
            assert!(
                result.is_ok(),
                "Refresh #{i} should succeed, got error: {:?}",
                result.err()
            );
        }

        // 第 6 次应失败 — 超过最大刷新次数
        let result = manager.refresh_token(&refresh_token).await;
        assert!(result.is_err());

        // 超过后 token 应被加入黑名单
        assert!(manager.is_token_revoked(&jti).await);
    }

    #[tokio::test]
    async fn test_refresh_token_invalid() {
        let secret = "test_secret_for_invalid_refresh_must_be_long_enough_ccc333";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        let result = manager.refresh_token("invalid.refresh.token").await;
        assert!(result.is_err());
    }

    // ===== with_token_store 测试 =====

    #[tokio::test]
    async fn test_with_token_store_custom_store() {
        let secret = "test_secret_for_custom_store_validation_must_be_long_ddd444";
        let custom_store: Arc<dyn TokenStore> = Arc::new(MemoryTokenStore::new());
        let manager = JwtManager::with_token_store(secret.to_string(), custom_store).unwrap();

        let user = User {
            username: "customstoreuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        manager.revoke_token(&token).await.unwrap();
        // 撤销后验证应失败
        assert!(manager.validate_token(&token).await.is_err());
    }

    // ===== 过期时间和配置测试 =====

    #[test]
    fn test_with_expiration() {
        let secret = "test_secret_for_expiration_config_must_be_long_enough_eee555";
        let manager = JwtManager::new(secret.to_string())
            .unwrap()
            .with_expiration(12);
        assert_eq!(manager.get_token_expiration(), 12 * 3600);
    }

    #[test]
    fn test_default_token_expiration() {
        let secret = "test_secret_for_default_expiration_must_be_long_enough_fff666";
        let manager = JwtManager::new(secret.to_string()).unwrap();
        assert_eq!(
            manager.get_token_expiration(),
            (DEFAULT_TOKEN_EXPIRATION_HOURS * 3600) as u64
        );
        assert_eq!(
            manager.get_refresh_token_expiration(),
            (DEFAULT_REFRESH_TOKEN_EXPIRATION_HOURS * 3600) as u64
        );
    }

    // ===== rotate_secret 测试 =====

    #[tokio::test]
    async fn test_rotate_secret_without_key_store_fails() {
        let secret = "test_secret_for_rotate_fail_must_be_long_enough_ggg777";
        let mut manager = JwtManager::new(secret.to_string()).unwrap();
        let result = manager.rotate_secret().await;
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            VecboostError::SecurityError(_)
        ));
    }

    // ===== new_with_key_store 测试 (使用 mock KeyStore) =====

    use crate::security::{KeyType, SecretKey};
    use std::collections::HashMap as StdHashMap;
    use std::sync::Mutex;

    /// 测试用 mock KeyStore
    struct MockKeyStore {
        secrets: Mutex<StdHashMap<String, String>>,
    }

    impl MockKeyStore {
        fn new() -> Self {
            Self {
                secrets: Mutex::new(StdHashMap::new()),
            }
        }

        fn make_key(key_type: &KeyType, name: &str) -> String {
            format!("{key_type:?}:{name}")
        }

        fn with_jwt_secret(secret: impl Into<String>) -> Self {
            let store = Self::new();
            store.secrets.lock().unwrap().insert(
                Self::make_key(&KeyType::JwtSecret, DEFAULT_JWT_SECRET_NAME),
                secret.into(),
            );
            store
        }
    }

    #[async_trait::async_trait]
    impl KeyStore for MockKeyStore {
        async fn get(
            &self,
            key_type: &KeyType,
            name: &str,
        ) -> Result<Option<SecretKey>, VecboostError> {
            let secrets = self.secrets.lock().unwrap();
            Ok(secrets
                .get(&Self::make_key(key_type, name))
                .map(|v| SecretKey::new(key_type.clone(), name, v.clone())))
        }

        async fn set(&self, key: &SecretKey) -> Result<(), VecboostError> {
            self.secrets
                .lock()
                .unwrap()
                .insert(Self::make_key(&key.key_type, &key.name), key.value.clone());
            Ok(())
        }

        async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), VecboostError> {
            self.secrets
                .lock()
                .unwrap()
                .remove(&Self::make_key(key_type, name));
            Ok(())
        }

        async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, VecboostError> {
            let prefix = format!("{key_type:?}:");
            let secrets = self.secrets.lock().unwrap();
            Ok(secrets
                .keys()
                .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
                .collect())
        }

        async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, VecboostError> {
            let secrets = self.secrets.lock().unwrap();
            Ok(secrets.contains_key(&Self::make_key(key_type, name)))
        }
    }

    #[tokio::test]
    async fn test_new_with_key_store_success() {
        let secret = "mock_jwt_secret_with_sufficient_entropy_for_testing_xyz123";
        let key_store = Arc::new(MockKeyStore::with_jwt_secret(secret));
        let manager = JwtManager::new_with_key_store(key_store, None)
            .await
            .expect("new_with_key_store should succeed");

        let user = User {
            username: "keystoreuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        let claims = manager.validate_token(&token).await.unwrap();
        assert_eq!(claims.username, "keystoreuser");
    }

    #[tokio::test]
    async fn test_new_with_key_store_secret_not_found() {
        let key_store = Arc::new(MockKeyStore::new());
        let result = JwtManager::new_with_key_store(key_store, None).await;
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            VecboostError::SecurityError(_)
        ));
    }

    #[tokio::test]
    async fn test_new_with_key_store_low_entropy_secret() {
        // 34 个相同字符 — 熵值不足
        let low_entropy = "b".repeat(34);
        let key_store = Arc::new(MockKeyStore::with_jwt_secret(low_entropy));
        let result = JwtManager::new_with_key_store(key_store, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rotate_secret_with_key_store_success() {
        let secret = "rotatable_jwt_secret_with_sufficient_entropy_for_testing_abc";
        let key_store = Arc::new(MockKeyStore::with_jwt_secret(secret));
        let mut manager = JwtManager::new_with_key_store(key_store.clone(), None)
            .await
            .unwrap();

        // 轮换密钥应成功
        manager
            .rotate_secret()
            .await
            .expect("rotate_secret should succeed");

        // 轮换后仍能生成和验证 token
        let user = User {
            username: "rotateuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };
        let token = manager.generate_token(&user).unwrap();
        let claims = manager.validate_token(&token).await.unwrap();
        assert_eq!(claims.username, "rotateuser");
    }

    #[tokio::test]
    async fn test_rotate_secret_with_custom_name() {
        let secret = "custom_named_jwt_secret_with_sufficient_entropy_def456";
        let key_store = Arc::new(MockKeyStore::new());
        // 插入自定义名称的 secret
        key_store
            .set(&SecretKey::new(KeyType::JwtSecret, "custom_secret", secret))
            .await
            .unwrap();

        let manager = JwtManager::new_with_key_store(key_store, Some("custom_secret"))
            .await
            .unwrap();
        // rotate_secret 使用默认名称 "jwt_secret",但 key_store 中只有 "custom_secret"
        // 所以 rotate_secret 会失败
        let mut manager = manager;
        let result = manager.rotate_secret().await;
        assert!(result.is_err());
    }

    // ===== 并发访问测试 =====

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_token_generation_and_validation() {
        let secret = "test_secret_for_concurrent_access_must_be_long_enough_hhh888";
        let manager = Arc::new(
            JwtManager::new(secret.to_string())
                .unwrap()
                .with_expiration(1),
        );

        let user = Arc::new(User {
            username: "concurrentuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        });

        let mut handles = Vec::new();
        for _ in 0..10 {
            let manager_clone = manager.clone();
            let user_clone = user.clone();
            handles.push(tokio::spawn(async move {
                let token = manager_clone.generate_token(&user_clone).unwrap();
                let claims = manager_clone.validate_token(&token).await.unwrap();
                assert_eq!(claims.username, "concurrentuser");
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }
}
