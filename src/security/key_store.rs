// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
use async_trait::async_trait;
use zeroize::Zeroizing;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyType {
    JwtSecret,
    ApiKey,
    DatabasePassword,
    ModelApiKey,
    Custom(String),
}

#[derive(Clone)]
pub struct SecretKey {
    pub key_type: KeyType,
    /// 零化包装 —— 值离开作用域时内存被安全擦除,不在堆上残留。
    pub value: Zeroizing<String>,
    pub name: String,
}

impl std::fmt::Debug for SecretKey {
    /// 调试输出走掩码,防止 `{:?}` 打印把完整密钥写进日志。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretKey")
            .field("key_type", &self.key_type)
            .field("name", &self.name)
            .field("value", &self.mask_value())
            .finish()
    }
}

impl std::fmt::Display for SecretKey {
    /// Display 同样只输出掩码形式(mask_value 的生产接线点)。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.mask_value())
    }
}

impl SecretKey {
    pub fn new(key_type: KeyType, name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key_type,
            name: name.into(),
            value: Zeroizing::new(value.into()),
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
        let value: &str = &self.value;
        if value.len() <= 8 {
            "*".repeat(value.len())
        } else {
            // UTF-8 safe slicing: floor/ceil to char boundary to avoid panic
            // when the 4-byte boundary falls inside a multi-byte character.
            let prefix_end = value.floor_char_boundary(4);
            let suffix_start = value.ceil_char_boundary(value.len() - 4);
            format!("{}***{}", &value[..prefix_end], &value[suffix_start..])
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

/// 环境变量 keystore（只读来源）。
///
/// 环境变量是进程的注入来源而非持久存储:通过 `set_var` 把密钥写回进程环境
/// 会使其对 `/proc/<pid>/environ` 读者与全部子进程可见。因此本实现仅支持
/// 读取(get/exists/list),`set`/`delete` 返回只读错误。
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

    async fn set(&self, _key: &SecretKey) -> Result<(), VecboostError> {
        Err(VecboostError::InternalError(
            "EnvironmentKeyStore is read-only: injecting secrets back into the process              environment would expose them via /proc/<pid>/environ and child processes;              provide them through the parent environment instead"
                .into(),
        ))
    }

    async fn delete(&self, _key_type: &KeyType, _name: &str) -> Result<(), VecboostError> {
        Err(VecboostError::InternalError(
            "EnvironmentKeyStore is read-only: environment variables are owned by the              parent process and cannot be revoked at runtime"
                .into(),
        ))
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
        assert_eq!(key.value.as_str(), "value");
        assert_eq!(key.key_type, KeyType::JwtSecret);
    }

    #[test]
    fn test_secret_key_jwt_secret_constructor() {
        let key = SecretKey::jwt_secret("my_secret_value");
        assert_eq!(key.name, "jwt_secret");
        assert_eq!(key.value.as_str(), "my_secret_value");
        assert_eq!(key.key_type, KeyType::JwtSecret);
    }

    #[test]
    fn test_secret_key_api_key_constructor() {
        let key = SecretKey::api_key("service_x", "abc123");
        assert_eq!(key.name, "service_x");
        assert_eq!(key.value.as_str(), "abc123");
        assert_eq!(key.key_type, KeyType::ApiKey);
    }

    #[test]
    fn test_secret_key_database_password_constructor() {
        let key = SecretKey::database_password("password123");
        assert_eq!(key.name, "database_password");
        assert_eq!(key.value.as_str(), "password123");
        assert_eq!(key.key_type, KeyType::DatabasePassword);
    }

    #[test]
    fn test_secret_key_model_api_key_constructor() {
        let key = SecretKey::model_api_key("hf_key");
        assert_eq!(key.name, "model_api_key");
        assert_eq!(key.value.as_str(), "hf_key");
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
    async fn test_environment_key_store_is_read_only() {
        // set/delete 返回只读错误,不再写回进程环境
        let store = EnvironmentKeyStore::new();
        let key = SecretKey::api_key("test_readonly", "my_value");
        assert!(store.set(&key).await.is_err());
        assert!(
            store
                .delete(&KeyType::ApiKey, "test_readonly")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_environment_key_store_get_from_parent_env() {
        // 读取路径保持可用:父进程注入的环境变量可读出
        let var = "VECBOOST_API_KEY_TEST_PARENT_ENV";
        // SAFETY: 测试专用变量,单测试内清理
        unsafe { std::env::set_var(var, "my_value") };
        let store = EnvironmentKeyStore::new();
        let retrieved = store
            .get(&KeyType::ApiKey, "test_parent_env")
            .await
            .unwrap();
        // SAFETY: 同上
        unsafe { std::env::remove_var(var) };
        let retrieved = retrieved.expect("env var should be readable");
        assert_eq!(retrieved.value.as_str(), "my_value");
        assert_eq!(retrieved.name, "test_parent_env");
    }

    /// Debug/Display 走掩码,完整密钥不会经日志泄漏
    #[test]
    fn secret_key_debug_display_are_masked() {
        let key = SecretKey::api_key("svc", "super_secret_value_123");
        let dbg = format!("{:?}", key);
        let disp = format!("{}", key);
        assert!(!dbg.contains("super_secret_value_123"));
        assert!(!disp.contains("super_secret_value_123"));
        assert!(disp.contains("***"));
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
        let var = "VECBOOST_API_KEY_TEST_EXISTS_UNIQUE";
        let store = EnvironmentKeyStore::new();

        assert!(
            !store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );

        // SAFETY: 测试专用变量,测试内清理
        unsafe { std::env::set_var(var, "value") };
        assert!(
            store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );
        unsafe { std::env::remove_var(var) };
        assert!(
            !store
                .exists(&KeyType::ApiKey, "test_exists_unique")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_environment_key_store_list() {
        let var_a = "VECBOOST_API_KEY_LIST_TEST_A_UNIQUE";
        let var_b = "VECBOOST_API_KEY_LIST_TEST_B_UNIQUE";
        let store = EnvironmentKeyStore::new();

        // SAFETY: 测试专用变量,测试内清理
        unsafe { std::env::set_var(var_a, "v1") };
        unsafe { std::env::set_var(var_b, "v2") };

        let keys = store.list(&KeyType::ApiKey).await.unwrap();
        assert!(keys.iter().any(|k| k.contains("LIST_TEST_A_UNIQUE")));
        assert!(keys.iter().any(|k| k.contains("LIST_TEST_B_UNIQUE")));

        unsafe { std::env::remove_var(var_a) };
        unsafe { std::env::remove_var(var_b) };
    }
}
