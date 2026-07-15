// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

#[cfg(feature = "auth")]
mod encrypted_store;
mod helpers;
mod key_store;
mod sanitize;

#[cfg(feature = "auth")]
pub use encrypted_store::EncryptedFileKeyStore;
pub use helpers::create_key_store;
pub use key_store::{KeyStore, KeyType, SecretKey};
pub use sanitize::{sanitize_api_key, sanitize_jwt_secret, sanitize_password, sanitize_secret};

use crate::error::VecboostError;

#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub storage_type: StorageType,
    pub encryption_key: Option<String>,
    pub key_file_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StorageType {
    Environment,
    EncryptedFile,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::Environment,
            encryption_key: None,
            key_file_path: None,
        }
    }
}

impl SecurityConfig {
    pub fn from_env() -> Self {
        let storage_type = match std::env::var("VECBOOST_KEY_STORAGE_TYPE")
            .unwrap_or_else(|_| "environment".to_string())
            .to_lowercase()
            .as_str()
        {
            "encrypted_file" => StorageType::EncryptedFile,
            _ => StorageType::Environment,
        };

        Self {
            storage_type,
            encryption_key: std::env::var("VECBOOST_ENCRYPTION_KEY").ok(),
            key_file_path: std::env::var("VECBOOST_KEY_FILE_PATH").ok(),
        }
    }

    pub fn validate(&self) -> Result<(), VecboostError> {
        match self.storage_type {
            StorageType::EncryptedFile => {
                if self.encryption_key.is_none() {
                    return Err(VecboostError::ConfigError(
                        "Encryption key is required for encrypted file storage".to_string(),
                    ));
                }
                if self.key_file_path.is_none() {
                    return Err(VecboostError::ConfigError(
                        "Key file path is required for encrypted file storage".to_string(),
                    ));
                }
            }
            StorageType::Environment => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::app::test_support::ENV_LOCK;

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert_eq!(config.storage_type, StorageType::Environment);
        assert!(config.encryption_key.is_none());
        assert!(config.key_file_path.is_none());
    }

    #[test]
    fn test_storage_type_equality() {
        assert_eq!(StorageType::Environment, StorageType::Environment);
        assert_eq!(StorageType::EncryptedFile, StorageType::EncryptedFile);
        assert_ne!(StorageType::Environment, StorageType::EncryptedFile);
    }

    #[test]
    fn test_security_config_from_env_default() {
        // Clear env vars to ensure default behavior
        unsafe {
            std::env::remove_var("VECBOOST_KEY_STORAGE_TYPE");
            std::env::remove_var("VECBOOST_ENCRYPTION_KEY");
            std::env::remove_var("VECBOOST_KEY_FILE_PATH");
        }

        let config = SecurityConfig::from_env();
        assert_eq!(config.storage_type, StorageType::Environment);
        assert!(config.encryption_key.is_none());
        assert!(config.key_file_path.is_none());
    }

    #[test]
    fn test_security_config_from_env_encrypted_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_KEY_STORAGE_TYPE", "encrypted_file");
            std::env::set_var("VECBOOST_ENCRYPTION_KEY", "test_key");
            std::env::set_var("VECBOOST_KEY_FILE_PATH", "/tmp/test.keys");
        }

        let config = SecurityConfig::from_env();
        assert_eq!(config.storage_type, StorageType::EncryptedFile);
        assert_eq!(config.encryption_key, Some("test_key".to_string()));
        assert_eq!(config.key_file_path, Some("/tmp/test.keys".to_string()));

        // Cleanup
        unsafe {
            std::env::remove_var("VECBOOST_KEY_STORAGE_TYPE");
            std::env::remove_var("VECBOOST_ENCRYPTION_KEY");
            std::env::remove_var("VECBOOST_KEY_FILE_PATH");
        }
    }

    #[test]
    fn test_security_config_from_env_case_insensitive() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_KEY_STORAGE_TYPE", "ENCRYPTED_FILE");
        }

        let config = SecurityConfig::from_env();
        assert_eq!(config.storage_type, StorageType::EncryptedFile);

        // Cleanup
        unsafe {
            std::env::remove_var("VECBOOST_KEY_STORAGE_TYPE");
        }
    }

    #[test]
    fn test_security_config_from_env_unknown_type_defaults_to_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("VECBOOST_KEY_STORAGE_TYPE", "unknown_type");
        }

        let config = SecurityConfig::from_env();
        assert_eq!(config.storage_type, StorageType::Environment);

        // Cleanup
        unsafe {
            std::env::remove_var("VECBOOST_KEY_STORAGE_TYPE");
        }
    }

    #[test]
    fn test_security_config_validate_environment_ok() {
        let config = SecurityConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_security_config_validate_encrypted_file_missing_key() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: None,
            key_file_path: Some("/tmp/test.keys".to_string()),
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Encryption key is required"));
    }

    #[test]
    fn test_security_config_validate_encrypted_file_missing_path() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: Some("key".to_string()),
            key_file_path: None,
        };
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Key file path is required"));
    }

    #[test]
    fn test_security_config_validate_encrypted_file_complete() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: Some("key".to_string()),
            key_file_path: Some("/tmp/test.keys".to_string()),
        };
        assert!(config.validate().is_ok());
    }

    #[tokio::test]
    async fn test_create_key_store_environment() {
        let config = SecurityConfig::default();
        let store = create_key_store(&config).await;
        assert!(store.is_ok());
    }

    #[tokio::test]
    async fn test_create_key_store_invalid_encrypted_file_no_auth() {
        // Without both key and path, should return Invalid configuration error
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: None,
            key_file_path: None,
        };
        let result = create_key_store(&config).await;
        // validate() should fail first
        assert!(result.is_err());
    }
}
