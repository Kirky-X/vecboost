use crate::auth::types::User;
use crate::error::AppError;
use crate::security::{KeyStore, KeyType};
use chrono::{Duration, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const DEFAULT_TOKEN_EXPIRATION_HOURS: i64 = 24;
const DEFAULT_JWT_SECRET_NAME: &str = "jwt_secret";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub username: String,
    pub role: String,
    pub permissions: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

#[derive(Clone)]
pub struct JwtManager {
    encoding_key: Arc<EncodingKey>,
    decoding_key: Arc<DecodingKey>,
    expiration_hours: i64,
    key_store: Option<Arc<dyn KeyStore>>,
}

impl JwtManager {
    pub fn new(secret: String) -> Self {
        let encoding_key = Arc::new(EncodingKey::from_secret(secret.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.as_ref()));

        Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            key_store: None,
        }
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

        let encoding_key = Arc::new(EncodingKey::from_secret(secret.value.as_ref()));
        let decoding_key = Arc::new(DecodingKey::from_secret(secret.value.as_ref()));

        Ok(Self {
            encoding_key,
            decoding_key,
            expiration_hours: DEFAULT_TOKEN_EXPIRATION_HOURS,
            key_store: Some(key_store),
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

        let claims = Claims {
            sub: user.username.clone(),
            username: user.username.clone(),
            role: user.role.clone(),
            permissions: user.permissions.clone(),
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
        };

        let header = Header::default();
        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| AppError::AuthenticationError(format!("Failed to generate token: {}", e)))
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, AppError> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;

        let token_data = decode::<Claims>(token, &self.decoding_key, &validation).map_err(|e| {
            AppError::AuthenticationError(format!("Token validation failed: {}", e))
        })?;

        Ok(token_data.claims)
    }

    pub fn extract_claims(&self, token: &str) -> Result<Claims, AppError> {
        self.validate_token(token)
    }

    pub fn get_token_expiration(&self) -> u64 {
        (self.expiration_hours * 3600) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_generation_and_validation() {
        let secret = "test_secret_key_for_testing_only";
        let manager = JwtManager::new(secret.to_string());

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        };

        let token = manager.generate_token(&user).unwrap();
        assert!(!token.is_empty());

        let claims = manager.validate_token(&token).unwrap();
        assert_eq!(claims.username, "testuser");
        assert_eq!(claims.role, "user");
        assert_eq!(claims.permissions, vec!["read", "write"]);
    }

    #[test]
    fn test_invalid_token() {
        let secret = "test_secret_key_for_testing_only";
        let manager = JwtManager::new(secret.to_string());

        let invalid_token = "invalid.token.here";
        let result = manager.validate_token(invalid_token);
        assert!(result.is_err());
    }

    #[test]
    fn test_token_with_different_secret() {
        let secret1 = "secret_key_1";
        let secret2 = "secret_key_2";

        let manager1 = JwtManager::new(secret1.to_string());
        let manager2 = JwtManager::new(secret2.to_string());

        let user = User {
            username: "testuser".to_string(),
            role: "user".to_string(),
            permissions: vec!["read".to_string()],
        };

        let token = manager1.generate_token(&user).unwrap();
        let result = manager2.validate_token(&token);
        assert!(result.is_err());
    }
}
