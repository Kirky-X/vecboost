// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

mod encrypted_store;
mod key_store;
mod sanitize;

pub use encrypted_store::EncryptedFileKeyStore;
pub use key_store::{KeyStore, KeyType, SecretKey};
pub use sanitize::{sanitize_api_key, sanitize_jwt_secret, sanitize_password, sanitize_secret};

use crate::error::AppError;

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

    pub fn validate(&self) -> Result<(), AppError> {
        match self.storage_type {
            StorageType::EncryptedFile => {
                if self.encryption_key.is_none() {
                    return Err(AppError::ConfigError(
                        "Encryption key is required for encrypted file storage".to_string(),
                    ));
                }
                if self.key_file_path.is_none() {
                    return Err(AppError::ConfigError(
                        "Key file path is required for encrypted file storage".to_string(),
                    ));
                }
            }
            StorageType::Environment => {}
        }
        Ok(())
    }
}

pub async fn create_key_store(config: &SecurityConfig) -> Result<Box<dyn KeyStore>, AppError> {
    config.validate()?;

    match (&config.storage_type, &config.key_file_path, &config.encryption_key) {
        (StorageType::Environment, _, _) => Ok(Box::new(key_store::EnvironmentKeyStore::new())),
        (StorageType::EncryptedFile, Some(key_file_path), Some(encryption_key)) => {
            Ok(Box::new(EncryptedFileKeyStore::new(key_file_path, encryption_key).await?))
        }
        _ => Err(AppError::ConfigError(
            "Invalid configuration: key_file_path and encryption_key are required for encrypted file storage".to_string(),
        )),
    }
}
