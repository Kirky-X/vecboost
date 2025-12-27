// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

mod encrypted_store;
mod key_store;

pub use encrypted_store::EncryptedFileKeyStore;
pub use key_store::{KeyStore, KeyType, SecretKey};

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

pub fn create_key_store(config: &SecurityConfig) -> Result<Box<dyn KeyStore>, AppError> {
    config.validate()?;

    match config.storage_type {
        StorageType::Environment => Ok(Box::new(key_store::EnvironmentKeyStore::new())),
        StorageType::EncryptedFile => Ok(Box::new(EncryptedFileKeyStore::new(
            config.key_file_path.as_ref().unwrap(),
            config.encryption_key.as_ref().unwrap(),
        )?)),
    }
}
