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
