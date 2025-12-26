use crate::error::AppError;
use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyType {
    JwtSecret,
    ApiKey,
    DatabasePassword,
    ModelApiKey,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct SecretKey {
    pub key_type: KeyType,
    pub value: String,
    pub name: String,
}

impl SecretKey {
    pub fn new(key_type: KeyType, name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key_type,
            name: name.into(),
            value: value.into(),
        }
    }

    pub fn jwt_secret(value: impl Into<String>) -> Self {
        Self::new(KeyType::JwtSecret, "jwt_secret", value)
    }

    pub fn api_key(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(KeyType::ApiKey, name, value)
    }

    pub fn database_password(value: impl Into<String>) -> Self {
        Self::new(KeyType::DatabasePassword, "database_password", value)
    }

    pub fn model_api_key(value: impl Into<String>) -> Self {
        Self::new(KeyType::ModelApiKey, "model_api_key", value)
    }

    pub fn mask_value(&self) -> String {
        if self.value.len() <= 8 {
            "*".repeat(self.value.len())
        } else {
            format!(
                "{}***{}",
                &self.value[..4],
                &self.value[self.value.len() - 4..]
            )
        }
    }
}

#[async_trait]
pub trait KeyStore: Send + Sync {
    async fn get(&self, key_type: &KeyType, name: &str) -> Result<Option<SecretKey>, AppError>;

    async fn set(&self, key: &SecretKey) -> Result<(), AppError>;

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), AppError>;

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, AppError>;

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, AppError>;
}

pub struct EnvironmentKeyStore;

impl EnvironmentKeyStore {
    pub fn new() -> Self {
        Self
    }

    fn env_key_name(key_type: &KeyType, name: &str) -> String {
        match key_type {
            KeyType::JwtSecret => "VECBOOST_JWT_SECRET".to_string(),
            KeyType::ApiKey => format!("VECBOOST_API_KEY_{}", name.to_uppercase()),
            KeyType::DatabasePassword => "VECBOOST_DATABASE_PASSWORD".to_string(),
            KeyType::ModelApiKey => "VECBOOST_MODEL_API_KEY".to_string(),
            KeyType::Custom(custom) => {
                format!("VECBOOST_{}_{}", custom.to_uppercase(), name.to_uppercase())
            }
        }
    }
}

#[async_trait]
impl KeyStore for EnvironmentKeyStore {
    async fn get(&self, key_type: &KeyType, name: &str) -> Result<Option<SecretKey>, AppError> {
        let env_key = Self::env_key_name(key_type, name);
        match std::env::var(&env_key) {
            Ok(value) => Ok(Some(SecretKey::new(key_type.clone(), name, value))),
            Err(_) => Ok(None),
        }
    }

    async fn set(&self, key: &SecretKey) -> Result<(), AppError> {
        let env_key = Self::env_key_name(&key.key_type, &key.name);
        unsafe {
            std::env::set_var(env_key, key.value.clone());
        }
        Ok(())
    }

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), AppError> {
        let env_key = Self::env_key_name(key_type, name);
        unsafe {
            std::env::remove_var(env_key);
        }
        Ok(())
    }

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, AppError> {
        let prefix = match key_type {
            KeyType::JwtSecret => "VECBOOST_JWT_SECRET".to_string(),
            KeyType::ApiKey => "VECBOOST_API_KEY_".to_string(),
            KeyType::DatabasePassword => "VECBOOST_DATABASE_PASSWORD".to_string(),
            KeyType::ModelApiKey => "VECBOOST_MODEL_API_KEY".to_string(),
            KeyType::Custom(custom) => format!("VECBOOST_{}_", custom.to_uppercase()),
        };

        let mut keys = Vec::new();
        for (k, _) in std::env::vars() {
            if k.starts_with(&prefix) {
                keys.push(k);
            }
        }
        Ok(keys)
    }

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, AppError> {
        let env_key = Self::env_key_name(key_type, name);
        Ok(std::env::var(&env_key).is_ok())
    }
}
