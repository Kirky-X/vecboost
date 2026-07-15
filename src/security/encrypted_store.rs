// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
use crate::security::key_store::{KeyStore, KeyType, SecretKey};
use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, AeadCore, KeyInit},
};
use argon2::Argon2;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedEntry {
    key_type: String,
    name: String,
    encrypted_value: String,
    nonce: String,
    created_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct KeyStoreData {
    version: u32,
    keys: Vec<EncryptedEntry>,
}

impl Default for KeyStoreData {
    fn default() -> Self {
        Self {
            version: 1,
            keys: Vec::new(),
        }
    }
}

pub struct EncryptedFileKeyStore {
    file_path: String,
    encryption_key: Arc<[u8; 32]>,
    data: Arc<RwLock<KeyStoreData>>,
}

impl EncryptedFileKeyStore {
    pub async fn new(file_path: &str, encryption_key: &str) -> Result<Self, VecboostError> {
        let key_bytes = Self::derive_key(encryption_key)?;

        let data = if tokio::fs::try_exists(file_path)
            .await
            .map_err(|e| VecboostError::IoError(format!("Failed to check key file: {}", e)))?
        {
            Self::load_from_file(file_path, &key_bytes).await?
        } else {
            KeyStoreData::default()
        };

        Ok(Self {
            file_path: file_path.to_string(),
            encryption_key: Arc::new(key_bytes),
            data: Arc::new(RwLock::new(data)),
        })
    }

    fn derive_key(password: &str) -> Result<[u8; 32], VecboostError> {
        // Argon2id 提供抗侧信道攻击和抗 GPU/ASIC 暴力破解保护。
        // 固定 salt 使密钥派生具有确定性,支持跨实例持久化
        // (同一个文件可以被用相同密码创建的不同 store 实例读写)。
        let argon2 = Argon2::default();
        let salt = b"vecboost_salt_v1"; // 16 bytes
        let mut key = [0u8; 32];
        argon2
            .hash_password_into(password.as_bytes(), salt, &mut key)
            .map_err(|e| VecboostError::security_error(format!("Failed to derive key: {}", e)))?;
        Ok(key)
    }

    async fn load_from_file(
        file_path: &str,
        key: &[u8; 32],
    ) -> Result<KeyStoreData, VecboostError> {
        let mut file = File::open(file_path)
            .await
            .map_err(|e| VecboostError::IoError(format!("Failed to open key file: {}", e)))?;

        let mut encrypted_content = String::new();
        file.read_to_string(&mut encrypted_content)
            .await
            .map_err(|e| VecboostError::IoError(format!("Failed to read key file: {}", e)))?;

        let parts: Vec<&str> = encrypted_content.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(VecboostError::security_error(
                "Invalid key file format".to_string(),
            ));
        }

        let nonce_hex = parts[0];
        let ciphertext = parts[1];

        let nonce_bytes = hex::decode(nonce_hex)
            .map_err(|e| VecboostError::security_error(format!("Invalid nonce: {}", e)))?;

        let ciphertext_bytes = hex::decode(ciphertext)
            .map_err(|e| VecboostError::security_error(format!("Invalid ciphertext: {}", e)))?;

        let cipher = Aes256Gcm::new(key.into());
        let nonce = Nonce::from_slice(&nonce_bytes);

        let plaintext = cipher
            .decrypt(nonce, ciphertext_bytes.as_ref())
            .map_err(|e| VecboostError::security_error(format!("Decryption failed: {}", e)))?;

        let json_str = String::from_utf8(plaintext)
            .map_err(|e| VecboostError::security_error(format!("Invalid UTF-8: {}", e)))?;

        serde_json::from_str(&json_str)
            .map_err(|e| VecboostError::security_error(format!("Invalid key data: {}", e)))
    }

    async fn save_to_file(&self) -> Result<(), VecboostError> {
        let data = self.data.read().await;
        let json_str = serde_json::to_string(&*data)
            .map_err(|e| VecboostError::security_error(format!("Serialization failed: {}", e)))?;

        let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
        let nonce_bytes = Aes256Gcm::generate_nonce(&mut rand::thread_rng());
        let ciphertext = cipher
            .encrypt(&nonce_bytes, json_str.as_bytes())
            .map_err(|e| VecboostError::security_error(format!("Encryption failed: {}", e)))?;

        let nonce_hex = hex::encode(nonce_bytes);
        let encrypted_content = format!("{}:{}", nonce_hex, hex::encode(&ciphertext));

        let mut file = File::create(&self.file_path)
            .await
            .map_err(|e| VecboostError::io_error(format!("Failed to create key file: {}", e)))?;

        // Set restrictive file permissions (owner read/write only: 0o600)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = file.metadata().await.unwrap().permissions();
            perms.set_mode(0o600);
            if let Err(e) = file.set_permissions(perms).await {
                log::warn!("Failed to set restrictive permissions on key file: {}", e);
            }
        }

        file.write_all(encrypted_content.as_bytes())
            .await
            .map_err(|e| VecboostError::io_error(format!("Failed to write key file: {}", e)))?;

        file.flush()
            .await
            .map_err(|e| VecboostError::io_error(format!("Failed to flush key file: {}", e)))?;
        Ok(())
    }

    fn key_type_to_string(key_type: &KeyType) -> String {
        match key_type {
            KeyType::JwtSecret => "jwt_secret".to_string(),
            KeyType::ApiKey => "api_key".to_string(),
            KeyType::DatabasePassword => "database_password".to_string(),
            KeyType::ModelApiKey => "model_api_key".to_string(),
            KeyType::Custom(s) => format!("custom_{}", s),
        }
    }
}

#[async_trait]
impl KeyStore for EncryptedFileKeyStore {
    async fn get(
        &self,
        key_type: &KeyType,
        name: &str,
    ) -> Result<Option<SecretKey>, VecboostError> {
        let type_str = Self::key_type_to_string(key_type);
        let data = self.data.read().await;

        for entry in &data.keys {
            if entry.key_type == type_str && entry.name == name {
                let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
                let nonce_bytes = hex::decode(&entry.nonce)
                    .map_err(|e| VecboostError::security_error(format!("Invalid nonce: {}", e)))?;
                let nonce = Nonce::from_slice(&nonce_bytes);

                let ciphertext = hex::decode(&entry.encrypted_value).map_err(|e| {
                    VecboostError::security_error(format!("Invalid ciphertext: {}", e))
                })?;

                let plaintext = cipher.decrypt(nonce, ciphertext.as_ref()).map_err(|e| {
                    VecboostError::security_error(format!("Decryption failed: {}", e))
                })?;

                let value = String::from_utf8(plaintext)
                    .map_err(|e| VecboostError::security_error(format!("Invalid UTF-8: {}", e)))?;

                return Ok(Some(SecretKey::new(key_type.clone(), name, value)));
            }
        }

        Ok(None)
    }

    async fn set(&self, key: &SecretKey) -> Result<(), VecboostError> {
        let type_str = Self::key_type_to_string(&key.key_type);
        let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
        let nonce_bytes = Aes256Gcm::generate_nonce(&mut rand::thread_rng());

        let ciphertext = cipher
            .encrypt(&nonce_bytes, key.value.as_bytes())
            .map_err(|e| VecboostError::security_error(format!("Encryption failed: {}", e)))?;

        let entry = EncryptedEntry {
            key_type: type_str,
            name: key.name.clone(),
            encrypted_value: hex::encode(&ciphertext),
            nonce: hex::encode(nonce_bytes),
            created_at: chrono::Utc::now().timestamp(),
        };

        let mut data = self.data.write().await;
        data.keys
            .retain(|e| !(e.key_type == entry.key_type && e.name == entry.name));
        data.keys.push(entry);

        drop(data);
        self.save_to_file().await?;

        Ok(())
    }

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), VecboostError> {
        let type_str = Self::key_type_to_string(key_type);
        let mut data = self.data.write().await;

        let original_len = data.keys.len();
        data.keys
            .retain(|e| !(e.key_type == type_str && e.name == name));

        if data.keys.len() != original_len {
            drop(data);
            self.save_to_file().await?;
        }

        Ok(())
    }

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, VecboostError> {
        let type_str = Self::key_type_to_string(key_type);
        let data = self.data.read().await;

        let keys: Vec<String> = data
            .keys
            .iter()
            .filter(|e| e.key_type == type_str)
            .map(|e| e.name.clone())
            .collect();

        Ok(keys)
    }

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, VecboostError> {
        let type_str = Self::key_type_to_string(key_type);
        let data = self.data.read().await;

        Ok(data
            .keys
            .iter()
            .any(|e| e.key_type == type_str && e.name == name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::key_store::{KeyType, SecretKey};
    use tempfile::tempdir;

    /// Helper: build a store path inside a fresh temporary directory.
    fn make_store_path(dir: &tempfile::TempDir) -> String {
        dir.path().join("keys.enc").to_string_lossy().to_string()
    }

    #[tokio::test]
    async fn test_new_creates_empty_store_when_file_absent() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed when file absent");

        let keys = store.list(&KeyType::JwtSecret).await.expect("list failed");
        assert!(keys.is_empty(), "fresh store should have no keys");
    }

    #[tokio::test]
    async fn test_set_and_get_key() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let key = SecretKey::jwt_secret("my_jwt_secret_value");
        store.set(&key).await.expect("set should succeed");

        let retrieved = store
            .get(&KeyType::JwtSecret, "jwt_secret")
            .await
            .expect("get failed");
        assert!(retrieved.is_some(), "key should exist after set");
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.value, "my_jwt_secret_value");
        assert_eq!(retrieved.name, "jwt_secret");
        assert_eq!(retrieved.key_type, KeyType::JwtSecret);
    }

    #[tokio::test]
    async fn test_get_missing_key_returns_none() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let result = store
            .get(&KeyType::ApiKey, "nonexistent")
            .await
            .expect("get should not error for missing key");
        assert!(result.is_none(), "missing key should return None");
    }

    #[tokio::test]
    async fn test_set_updates_existing_key() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let key1 = SecretKey::api_key("service_a", "old_value");
        store.set(&key1).await.expect("set old value failed");

        let key2 = SecretKey::api_key("service_a", "new_value");
        store.set(&key2).await.expect("set new value failed");

        let keys = store.list(&KeyType::ApiKey).await.expect("list failed");
        assert_eq!(keys.len(), 1, "should have exactly one key after update");

        let retrieved = store
            .get(&KeyType::ApiKey, "service_a")
            .await
            .expect("get failed")
            .expect("key should exist");
        assert_eq!(retrieved.value, "new_value", "value should be updated");
    }

    #[tokio::test]
    async fn test_delete_key() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let key = SecretKey::api_key("to_delete", "value");
        store.set(&key).await.expect("set failed");
        assert!(
            store
                .exists(&KeyType::ApiKey, "to_delete")
                .await
                .expect("exists failed"),
            "key should exist before delete"
        );

        store
            .delete(&KeyType::ApiKey, "to_delete")
            .await
            .expect("delete failed");

        assert!(
            !store
                .exists(&KeyType::ApiKey, "to_delete")
                .await
                .expect("exists failed"),
            "key should not exist after delete"
        );
    }

    #[tokio::test]
    async fn test_delete_missing_key_is_noop() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        store
            .delete(&KeyType::ApiKey, "never_existed")
            .await
            .expect("delete missing key should not error");
    }

    #[tokio::test]
    async fn test_list_keys_by_type() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        store
            .set(&SecretKey::api_key("key_a", "v1"))
            .await
            .expect("set failed");
        store
            .set(&SecretKey::api_key("key_b", "v2"))
            .await
            .expect("set failed");
        store
            .set(&SecretKey::jwt_secret("jwt_val"))
            .await
            .expect("set failed");

        let api_keys = store.list(&KeyType::ApiKey).await.expect("list failed");
        assert_eq!(api_keys.len(), 2, "should have 2 api keys");
        assert!(api_keys.contains(&"key_a".to_string()));
        assert!(api_keys.contains(&"key_b".to_string()));

        let jwt_keys = store.list(&KeyType::JwtSecret).await.expect("list failed");
        assert_eq!(jwt_keys.len(), 1, "should have 1 jwt key");
    }

    #[tokio::test]
    async fn test_exists() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        assert!(
            !store
                .exists(&KeyType::DatabasePassword, "database_password")
                .await
                .expect("exists failed"),
            "key should not exist before set"
        );

        store
            .set(&SecretKey::database_password("db_pass"))
            .await
            .expect("set failed");

        assert!(
            store
                .exists(&KeyType::DatabasePassword, "database_password")
                .await
                .expect("exists failed"),
            "key should exist after set"
        );
    }

    #[tokio::test]
    async fn test_persistence_across_store_instances() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store1 = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");
        store1
            .set(&SecretKey::model_api_key("hf_token_abc"))
            .await
            .expect("set failed");

        let store2 = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed loading existing file");
        let retrieved = store2
            .get(&KeyType::ModelApiKey, "model_api_key")
            .await
            .expect("get failed")
            .expect("key should persist");
        assert_eq!(retrieved.value, "hf_token_abc");
    }

    #[tokio::test]
    async fn test_encrypted_file_format() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let secret_value = "super_secret_value_xyz";
        store
            .set(&SecretKey::api_key("format_test", secret_value))
            .await
            .expect("set failed");

        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("should read key file");

        // Format should be "nonce_hex:ciphertext_hex".
        let parts: Vec<&str> = content.splitn(2, ':').collect();
        assert_eq!(parts.len(), 2, "file should have nonce:ciphertext format");
        assert!(hex::decode(parts[0]).is_ok(), "nonce should be valid hex");
        assert!(
            hex::decode(parts[1]).is_ok(),
            "ciphertext should be valid hex"
        );

        // The plaintext secret value must NOT appear in the file.
        assert!(
            !content.contains(secret_value),
            "secret value must not appear in plaintext in the file"
        );
    }

    #[tokio::test]
    async fn test_invalid_password_fails_to_load() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store1 = EncryptedFileKeyStore::new(&path, "password_A_correct")
            .await
            .expect("new should succeed");
        store1
            .set(&SecretKey::jwt_secret("value"))
            .await
            .expect("set failed");

        let result = EncryptedFileKeyStore::new(&path, "password_B_wrong").await;
        assert!(result.is_err(), "loading with wrong password should fail");
        let err_msg = format!("{}", result.err().unwrap());
        assert!(
            err_msg.contains("Decryption failed"),
            "error should mention decryption failure, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_corrupted_file_no_separator_fails() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        // Write garbage without the ':' separator.
        tokio::fs::write(&path, "this_is_not_valid_format")
            .await
            .expect("write failed");

        let result = EncryptedFileKeyStore::new(&path, "any_password").await;
        assert!(result.is_err(), "loading corrupted file should fail");
        let err_msg = format!("{}", result.err().unwrap());
        assert!(
            err_msg.contains("Invalid key file format"),
            "error should mention invalid format, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_corrupted_file_invalid_hex_fails() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        // Content with the colon separator but invalid hex bytes.
        tokio::fs::write(&path, "not_hex:also_not_hex")
            .await
            .expect("write failed");

        let result = EncryptedFileKeyStore::new(&path, "any_password").await;
        assert!(result.is_err(), "loading file with invalid hex should fail");
    }

    #[tokio::test]
    async fn test_save_to_nonexistent_directory_fails() {
        let dir = tempdir().expect("failed to create temp dir");
        // Path inside a subdirectory that does not exist.
        let path = dir
            .path()
            .join("nonexistent_subdir")
            .join("keys.enc")
            .to_string_lossy()
            .to_string();

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed (file doesn't exist yet)");

        let result = store.set(&SecretKey::jwt_secret("value")).await;
        assert!(
            result.is_err(),
            "save to non-existent directory should fail"
        );
    }

    #[tokio::test]
    async fn test_custom_key_type() {
        let dir = tempdir().expect("failed to create temp dir");
        let path = make_store_path(&dir);

        let store = EncryptedFileKeyStore::new(&path, "strong_password_123")
            .await
            .expect("new should succeed");

        let custom_type = KeyType::Custom("my_custom".to_string());
        let key = SecretKey::new(custom_type.clone(), "custom_name", "custom_value");
        store.set(&key).await.expect("set failed");

        let retrieved = store
            .get(&custom_type, "custom_name")
            .await
            .expect("get failed")
            .expect("key should exist");
        assert_eq!(retrieved.value, "custom_value");

        let custom_keys = store.list(&custom_type).await.expect("list failed");
        assert_eq!(custom_keys.len(), 1);
        assert!(custom_keys.contains(&"custom_name".to_string()));
    }
}
