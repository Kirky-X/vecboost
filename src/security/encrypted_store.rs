use crate::error::AppError;
use crate::security::key_store::{KeyStore, KeyType, SecretKey};
use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, AeadCore, KeyInit},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::sync::Arc;
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
    pub fn new(file_path: &str, encryption_key: &str) -> Result<Self, AppError> {
        let key_bytes = Self::derive_key(encryption_key)?;

        let data = if Path::new(file_path).exists() {
            Self::load_from_file(file_path, &key_bytes)?
        } else {
            KeyStoreData::default()
        };

        Ok(Self {
            file_path: file_path.to_string(),
            encryption_key: Arc::new(key_bytes),
            data: Arc::new(RwLock::new(data)),
        })
    }

    fn derive_key(password: &str) -> Result<[u8; 32], AppError> {
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(b"vecboost-key-derivation");
        let result = hasher.finalize();
        Ok(result.into())
    }

    fn load_from_file(file_path: &str, key: &[u8; 32]) -> Result<KeyStoreData, AppError> {
        let encrypted_content = fs::read_to_string(file_path)
            .map_err(|e| AppError::IoError(format!("Failed to read key file: {}", e)))?;

        let parts: Vec<&str> = encrypted_content.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(AppError::security_error(
                "Invalid key file format".to_string(),
            ));
        }

        let nonce_hex = parts[0];
        let ciphertext = parts[1];

        let nonce_bytes = hex::decode(nonce_hex)
            .map_err(|e| AppError::security_error(format!("Invalid nonce: {}", e)))?;

        let cipher = Aes256Gcm::new(key.into());
        let nonce = Nonce::from_slice(&nonce_bytes);

        let plaintext = cipher
            .decrypt(nonce, ciphertext.as_bytes())
            .map_err(|e| AppError::security_error(format!("Decryption failed: {}", e)))?;

        let json_str = String::from_utf8(plaintext)
            .map_err(|e| AppError::security_error(format!("Invalid UTF-8: {}", e)))?;

        serde_json::from_str(&json_str)
            .map_err(|e| AppError::security_error(format!("Invalid key data: {}", e)))
    }

    fn save_to_file(&self) -> Result<(), AppError> {
        let data = self.data.blocking_read();
        let json_str = serde_json::to_string(&*data)
            .map_err(|e| AppError::security_error(format!("Serialization failed: {}", e)))?;

        let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
        let nonce_bytes = Aes256Gcm::generate_nonce(&mut rand::thread_rng());
        let ciphertext = cipher
            .encrypt(&nonce_bytes, json_str.as_bytes())
            .map_err(|e| AppError::security_error(format!("Encryption failed: {}", e)))?;

        let nonce_hex = hex::encode(nonce_bytes);
        let encrypted_content = format!("{}:{}", nonce_hex, hex::encode(&ciphertext));

        fs::write(&self.file_path, encrypted_content)
            .map_err(|e| AppError::io_error(format!("Failed to write key file: {}", e)))?;

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

    #[allow(dead_code)]
    fn string_to_key_type(s: &str) -> Result<KeyType, AppError> {
        match s {
            "jwt_secret" => Ok(KeyType::JwtSecret),
            "api_key" => Ok(KeyType::ApiKey),
            "database_password" => Ok(KeyType::DatabasePassword),
            "model_api_key" => Ok(KeyType::ModelApiKey),
            custom if custom.starts_with("custom_") => Ok(KeyType::Custom(custom[7..].to_string())),
            _ => Err(AppError::security_error(format!("Unknown key type: {}", s))),
        }
    }
}

#[async_trait]
impl KeyStore for EncryptedFileKeyStore {
    async fn get(&self, key_type: &KeyType, name: &str) -> Result<Option<SecretKey>, AppError> {
        let type_str = Self::key_type_to_string(key_type);
        let data = self.data.read().await;

        for entry in &data.keys {
            if entry.key_type == type_str && entry.name == name {
                let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
                let nonce_bytes = hex::decode(&entry.nonce)
                    .map_err(|e| AppError::security_error(format!("Invalid nonce: {}", e)))?;
                let nonce = Nonce::from_slice(&nonce_bytes);

                let ciphertext = hex::decode(&entry.encrypted_value)
                    .map_err(|e| AppError::security_error(format!("Invalid ciphertext: {}", e)))?;

                let plaintext = cipher
                    .decrypt(nonce, ciphertext.as_ref())
                    .map_err(|e| AppError::security_error(format!("Decryption failed: {}", e)))?;

                let value = String::from_utf8(plaintext)
                    .map_err(|e| AppError::security_error(format!("Invalid UTF-8: {}", e)))?;

                return Ok(Some(SecretKey::new(key_type.clone(), name, value)));
            }
        }

        Ok(None)
    }

    async fn set(&self, key: &SecretKey) -> Result<(), AppError> {
        let type_str = Self::key_type_to_string(&key.key_type);
        let cipher = Aes256Gcm::new(self.encryption_key.as_ref().into());
        let nonce_bytes = Aes256Gcm::generate_nonce(&mut rand::thread_rng());

        let ciphertext = cipher
            .encrypt(&nonce_bytes, key.value.as_bytes())
            .map_err(|e| AppError::security_error(format!("Encryption failed: {}", e)))?;

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
        self.save_to_file()?;

        Ok(())
    }

    async fn delete(&self, key_type: &KeyType, name: &str) -> Result<(), AppError> {
        let type_str = Self::key_type_to_string(key_type);
        let mut data = self.data.write().await;

        let original_len = data.keys.len();
        data.keys
            .retain(|e| !(e.key_type == type_str && e.name == name));

        if data.keys.len() != original_len {
            drop(data);
            self.save_to_file()?;
        }

        Ok(())
    }

    async fn list(&self, key_type: &KeyType) -> Result<Vec<String>, AppError> {
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

    async fn exists(&self, key_type: &KeyType, name: &str) -> Result<bool, AppError> {
        let type_str = Self::key_type_to_string(key_type);
        let data = self.data.read().await;

        Ok(data
            .keys
            .iter()
            .any(|e| e.key_type == type_str && e.name == name))
    }
}
