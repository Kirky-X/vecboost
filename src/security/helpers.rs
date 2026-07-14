// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use super::{KeyStore, SecurityConfig, StorageType};
use crate::error::VecboostError;

/// Create a key store based on the security configuration.
pub async fn create_key_store(config: &SecurityConfig) -> Result<Box<dyn KeyStore>, VecboostError> {
    config.validate()?;

    match (&config.storage_type, &config.key_file_path, &config.encryption_key) {
        (StorageType::Environment, _, _) => Ok(Box::new(super::key_store::EnvironmentKeyStore::new())),
        #[cfg(feature = "auth")]
        (StorageType::EncryptedFile, Some(key_file_path), Some(encryption_key)) => {
            Ok(Box::new(super::EncryptedFileKeyStore::new(key_file_path, encryption_key).await?))
        }
        #[cfg(not(feature = "auth"))]
        (StorageType::EncryptedFile, _, _) => Err(VecboostError::ConfigError(
            "Encrypted file storage requires the 'auth' feature to be enabled".to_string(),
        )),
        _ => Err(VecboostError::ConfigError(
            "Invalid configuration: key_file_path and encryption_key are required for encrypted file storage".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::KeyType;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_create_key_store_environment_default() {
        let config = SecurityConfig::default();
        let result = create_key_store(&config).await;
        assert!(
            result.is_ok(),
            "default config should produce an environment key store"
        );
    }

    #[tokio::test]
    async fn test_create_key_store_environment_explicit() {
        let config = SecurityConfig {
            storage_type: StorageType::Environment,
            encryption_key: None,
            key_file_path: None,
        };
        let store = create_key_store(&config)
            .await
            .expect("store creation should succeed");
        let result = store.get(&KeyType::ApiKey, "nonexistent").await;
        assert!(result.is_ok(), "get on environment store should not error");
        assert!(result.unwrap().is_none(), "missing key should return None");
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_key_store_encrypted_file_success() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = dir.path().join("keys.enc").to_string_lossy().to_string();

        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: Some("strong_password_123".to_string()),
            key_file_path: Some(path),
        };

        let result = create_key_store(&config).await;
        assert!(
            result.is_ok(),
            "EncryptedFile store should be created successfully"
        );
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_create_key_store_encrypted_file_returns_working_store() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = dir.path().join("keys.enc").to_string_lossy().to_string();

        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: Some("strong_password_123".to_string()),
            key_file_path: Some(path.clone()),
        };

        let store = create_key_store(&config)
            .await
            .expect("store creation should succeed");
        let result = store.list(&KeyType::JwtSecret).await;
        assert!(
            result.is_ok(),
            "list on a fresh encrypted store should not error"
        );
        assert!(
            result.unwrap().is_empty(),
            "fresh store should have no keys"
        );
    }

    #[tokio::test]
    async fn test_create_key_store_encrypted_file_missing_key() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: None,
            key_file_path: Some("/tmp/test.keys".to_string()),
        };
        let result = create_key_store(&config).await;
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("Encryption key is required"),
            "error should mention missing encryption key, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_create_key_store_encrypted_file_missing_path() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: Some("key".to_string()),
            key_file_path: None,
        };
        let result = create_key_store(&config).await;
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("Key file path is required"),
            "error should mention missing key file path, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_create_key_store_encrypted_file_both_missing() {
        let config = SecurityConfig {
            storage_type: StorageType::EncryptedFile,
            encryption_key: None,
            key_file_path: None,
        };
        let result = create_key_store(&config).await;
        assert!(
            result.is_err(),
            "validate should fail when both key and path are missing"
        );
    }
}
