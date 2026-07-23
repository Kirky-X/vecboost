// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
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
            // UTF-8 safe slicing: floor/ceil to char boundary to avoid panic
            // when the 4-byte boundary falls inside a multi-byte character.
            let prefix_end = self.value.floor_char_boundary(4);
            let suffix_start = self.value.ceil_char_boundary(self.value.len() - 4);
            format!(
                "{}***{}",
                &self.value[..prefix_end],
                &self.value[suffix_start..]
            )
        }
    }
}

#[async_trait]
pub trait KeyStore: Send + Sync {
    async fn get(&self, key_type: &KeyType, name: &str)
    -> Result<Option<SecretKey>, VecboostError>;

    async fn set(&self, key: &SecretKey) -> Result<(), VecboostError>;

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), VecboostError>;

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, VecboostError>;

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, VecboostError>;
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
    async fn get(
        &self,
        key_type: &KeyType,
        name: &str,
    ) -> Result<Option<SecretKey>, VecboostError> {
        let env_key = Self::env_key_name(key_type, name);
        match std::env::var(&env_key) {
            Ok(value) => Ok(Some(SecretKey::new(key_type.clone(), name, value))),
            Err(_) => Ok(None),
        }
    }

    async fn set(&self, key: &SecretKey) -> Result<(), VecboostError> {
        let env_key = Self::env_key_name(&key.key_type, &key.name);
        unsafe {
            std::env::set_var(env_key, key.value.clone());
        }
        Ok(())
    }

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), VecboostError> {
        let env_key = Self::env_key_name(key_type, name);
        unsafe {
            std::env::remove_var(env_key);
        }
        Ok(())
    }

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, VecboostError> {
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

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, VecboostError> {
        let env_key = Self::env_key_name(key_type, name);
        Ok(std::env::var(&env_key).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_type_equality() {
        assert_eq!(KeyType::JwtSecret, KeyType::JwtSecret);
        assert_ne!(KeyType::JwtSecret, KeyType::ApiKey);
        assert_ne!(KeyType::Custom("test".to_string()), KeyType::ApiKey);
        assert_eq!(
            KeyType::Custom("test".to_string()),
            KeyType::Custom("test".to_string())
        );
    }

    #[test]
    fn test_secret_key_new() {
        let key = SecretKey::new(KeyType::JwtSecret, "name", "value");
        assert_eq!(key.name, "name");
        assert_eq!(key.value, "value");
        assert_eq!(key.key_type, KeyType::JwtSecret);
    }

    #[test]
    fn test_secret_key_jwt_secret_constructor() {
        let key = SecretKey::jwt_secret("my_secret_value");
        assert_eq!(key.name, "jwt_secret");
        assert_eq!(key.value, "my_secret_value");
        assert_eq!(key.key_type, KeyType::JwtSecret);
    }

    #[test]
    fn test_secret_key_api_key_constructor() {
        let key = SecretKey::api_key("service_x", "abc123");
        assert_eq!(key.name, "service_x");
        assert_eq!(key.value, "abc123");
        assert_eq!(key.key_type, KeyType::ApiKey);
    }

    #[test]
    fn test_secret_key_database_password_constructor() {
        let key = SecretKey::database_password("password123");
        assert_eq!(key.name, "database_password");
        assert_eq!(key.value, "password123");
        assert_eq!(key.key_type, KeyType::DatabasePassword);
    }

    #[test]
    fn test_secret_key_model_api_key_constructor() {
        let key = SecretKey::model_api_key("hf_key");
        assert_eq!(key.name, "model_api_key");
        assert_eq!(key.value, "hf_key");
        assert_eq!(key.key_type, KeyType::ModelApiKey);
    }

    #[test]
    fn test_mask_value_short() {
        let key = SecretKey::new(KeyType::ApiKey, "test", "short");
        // len <= 8: all masked
        assert_eq!(key.mask_value(), "*****");
    }

    #[test]
    fn test_mask_value_long() {
        let key = SecretKey::new(KeyType::ApiKey, "test", "very_long_secret_value");
        let masked = key.mask_value();
        // len > 8: first 4 + *** + last 4
        assert_eq!(masked, "very***alue");
    }

    #[test]
    fn test_mask_value_exactly_8_chars() {
        let key = SecretKey::new(KeyType::ApiKey, "test", "12345678");
        // exactly 8 chars: all masked (boundary case)
        assert_eq!(key.mask_value(), "********");
    }

    #[test]
    fn test_mask_value_exactly_9_chars() {
        let key = SecretKey::new(KeyType::ApiKey, "test", "123456789");
        // 9 chars: first 4 + *** + last 4
        let masked = key.mask_value();
        assert_eq!(masked, "1234***6789");
    }

    #[test]
    fn test_mask_value_multibyte_no_panic() {
        // CJK value: 4-byte boundary falls inside 2nd char (each CJK = 3 bytes).
        // Without floor/ceil_char_boundary this would panic.
        let key = SecretKey::new(KeyType::ApiKey, "test", "密钥secretvalue123");
        let masked = key.mask_value();
        assert!(masked.contains("***"));
    }

    #[test]
    fn test_environment_key_store_env_key_name_jwt() {
        let name = EnvironmentKeyStore::env_key_name(&KeyType::JwtSecret, "anything");
        assert_eq!(name, "VECBOOST_JWT_SECRET");
    }

    #[test]
    fn test_environment_key_store_env_key_name_api_key() {
        let name = EnvironmentKeyStore::env_key_name(&KeyType::ApiKey, "service_name");
        assert_eq!(name, "VECBOOST_API_KEY_SERVICE_NAME");
    }

    #[test]
    fn test_environment_key_store_env_key_name_db_password() {
        let name = EnvironmentKeyStore::env_key_name(&KeyType::DatabasePassword, "ignored");
        assert_eq!(name, "VECBOOST_DATABASE_PASSWORD");
    }

    #[test]
    fn test_environment_key_store_env_key_name_model_api_key() {
        let name = EnvironmentKeyStore::env_key_name(&KeyType::ModelApiKey, "ignored");
        assert_eq!(name, "VECBOOST_MODEL_API_KEY");
    }

    #[test]
    fn test_environment_key_store_env_key_name_custom() {
        let name = EnvironmentKeyStore::env_key_name(
            &KeyType::Custom("custom_type".to_string()),
            "my_name",
        );
        assert_eq!(name, "VECBOOST_CUSTOM_TYPE_MY_NAME");
    }

    #[tokio::test]
    async fn test_environment_key_store_set_and_get() {
        // Use unique name to avoid collision with other tests
        let store = EnvironmentKeyStore::new();
        let key = SecretKey::api_key("test_set_get_unique", "my_value");

        store.set(&key).await.unwrap();

        let retrieved = store
            .get(&KeyType::ApiKey, "test_set_get_unique")
            .await
            .unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.value, "my_value");
        assert_eq!(retrieved.name, "test_set_get_unique");

        // Cleanup
        store
            .delete(&KeyType::ApiKey, "test_set_get_unique")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_environment_key_store_get_missing() {
        let store = EnvironmentKeyStore::new();
        let result = store
            .get(&KeyType::ApiKey, "nonexistent_key_xyz")
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_environment_key_store_exists() {
        let store = EnvironmentKeyStore::new();
        let key = SecretKey::api_key("test_exists_unique", "value");

        assert!(
            !store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );

        store.set(&key).await.unwrap();

        assert!(
            store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );

        // Cleanup
        store
            .delete(&KeyType::ApiKey, "test_exists_unique")
            .await
            .unwrap();
        assert!(
            !store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_environment_key_store_delete() {
        let store = EnvironmentKeyStore::new();
        let key = SecretKey::api_key("test_delete_unique", "value");

        store.set(&key).await.unwrap();
        assert!(
            store
                .exists(&KeyType::ApiKey, "test_delete_unique")
                .await
                .unwrap()
        );

        store
            .delete(&KeyType::ApiKey, "test_delete_unique")
            .await
            .unwrap();
        assert!(
            !store
                .exists(&KeyType::ApiKey, "test_delete_unique")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_environment_key_store_list() {
        let store = EnvironmentKeyStore::new();

        // Set multiple keys
        let key1 = SecretKey::api_key("list_test_a_unique", "v1");
        let key2 = SecretKey::api_key("list_test_b_unique", "v2");
        store.set(&key1).await.unwrap();
        store.set(&key2).await.unwrap();

        let keys = store.list(&KeyType::ApiKey).await.unwrap();
        assert!(keys.iter().any(|k| k.contains("LIST_TEST_A_UNIQUE")));
        assert!(keys.iter().any(|k| k.contains("LIST_TEST_B_UNIQUE")));

        // Cleanup
        store
            .delete(&KeyType::ApiKey, "list_test_a_unique")
            .await
            .unwrap();
        store
            .delete(&KeyType::ApiKey, "list_test_b_unique")
            .await
            .unwrap();
    }
}
