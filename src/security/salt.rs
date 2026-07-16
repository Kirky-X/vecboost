// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Random salt for Argon2 key derivation (replaces hardcoded `vecboost_salt_v1`).
//!
//! Stored as plaintext prefix in keystore file header (`salt_hex:nonce_hex:ciphertext_hex`)
//! to avoid circular dependency — salt cannot live inside the encrypted keystore it unlocks.

use crate::error::VecboostError;
use rand::Rng;

/// 16-byte Argon2 salt managed alongside the encrypted keystore file.
#[derive(Debug, Clone)]
pub struct SaltStore {
    salt: [u8; 16],
}

impl SaltStore {
    /// Generate a cryptographically random salt (for new keystore files).
    pub fn generate() -> Self {
        let mut salt = [0u8; 16];
        rand::rng().fill_bytes(&mut salt);
        Self { salt }
    }

    /// Construct from raw bytes (e.g. decoded from file header hex).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VecboostError> {
        if bytes.len() != 16 {
            return Err(VecboostError::security_error(format!(
                "salt must be 16 bytes, got {}",
                bytes.len()
            )));
        }
        let mut salt = [0u8; 16];
        salt.copy_from_slice(bytes);
        Ok(Self { salt })
    }

    /// Construct from hex string (keystore file header format).
    pub fn from_hex(hex_str: &str) -> Result<Self, VecboostError> {
        let bytes = hex::decode(hex_str)
            .map_err(|e| VecboostError::security_error(format!("invalid salt hex: {}", e)))?;
        Self::from_bytes(&bytes)
    }

    /// Encode salt as hex for file header storage.
    pub fn to_hex(&self) -> String {
        hex::encode(self.salt)
    }

    /// Borrow raw salt bytes for KDF (`argon2.hash_password_into`).
    pub fn as_bytes(&self) -> &[u8] {
        &self.salt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_produces_16_byte_salt() {
        let salt = SaltStore::generate();
        assert_eq!(salt.as_bytes().len(), 16);
    }

    #[test]
    fn test_generate_produces_distinct_salts() {
        let s1 = SaltStore::generate();
        let s2 = SaltStore::generate();
        // Probability of collision is negligible for 16 random bytes
        assert_ne!(s1.as_bytes(), s2.as_bytes(), "salts should be distinct");
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        let original = SaltStore::generate();
        let bytes = original.as_bytes().to_vec();
        let restored = SaltStore::from_bytes(&bytes).expect("from_bytes should succeed");
        assert_eq!(original.as_bytes(), restored.as_bytes());
    }

    #[test]
    fn test_from_bytes_rejects_wrong_length() {
        let too_short = vec![0u8; 15];
        let result = SaltStore::from_bytes(&too_short);
        assert!(result.is_err(), "15-byte salt should be rejected");

        let too_long = vec![0u8; 17];
        let result = SaltStore::from_bytes(&too_long);
        assert!(result.is_err(), "17-byte salt should be rejected");
    }

    #[test]
    fn test_from_hex_roundtrip() {
        let original = SaltStore::generate();
        let hex = original.to_hex();
        assert_eq!(hex.len(), 32, "16 bytes → 32 hex chars");

        let restored = SaltStore::from_hex(&hex).expect("from_hex should succeed");
        assert_eq!(original.as_bytes(), restored.as_bytes());
    }

    #[test]
    fn test_from_hex_rejects_invalid_hex() {
        let result = SaltStore::from_hex("not_valid_hex");
        assert!(result.is_err(), "invalid hex should be rejected");
    }

    #[test]
    fn test_from_hex_rejects_wrong_byte_count() {
        // 15 bytes in hex = 30 chars (too short)
        let result = SaltStore::from_hex("000000000000000000000000000000");
        assert!(
            result.is_err(),
            "30 hex chars (15 bytes) should be rejected"
        );

        // 17 bytes in hex = 34 chars (too long)
        let result = SaltStore::from_hex("0000000000000000000000000000000000");
        assert!(
            result.is_err(),
            "34 hex chars (17 bytes) should be rejected"
        );
    }
}
