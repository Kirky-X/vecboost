// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! Serde helpers for transparent field-level encryption using confers'
//! `XChaCha20-Poly1305` crypto primitives.
//!
//! # Overview
//!
//! Sensitive config fields (JWT secret, admin password) are encrypted at rest
//! in config files using XChaCha20-Poly1305. The master encryption key is read
//! from the `VECBOOST_ENCRYPTION_KEY` environment variable (must be exactly 32
//! bytes). A per-field key is derived via HKDF-SHA256 to ensure domain separation.
//!
//! # Wire format
//!
//! Encrypted values are stored as hex-encoded strings: `nonce ‖ ciphertext`
//! (24-byte nonce followed by Poly1305-authenticated ciphertext).
//!
//! # Fallback
//!
//! When `VECBOOST_ENCRYPTION_KEY` is not set, values pass through as plaintext.
//! This allows development/test environments to operate without encryption setup.
//!
//! # Usage
//!
//! ```rust,ignore
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct MyConfig {
//!     #[serde(
//!         default,
//!         serialize_with = "crate::config::encryption::encrypted_option::serialize",
//!         deserialize_with = "crate::config::encryption::encrypted_option::deserialize"
//!     )]
//!     pub secret: Option<String>,
//! }
//! ```

use confers::secret::{XChaCha20Crypto, derive_field_key};

/// Environment variable name for the master encryption key.
///
/// The value must be exactly 32 bytes (UTF-8 encoded) for XChaCha20-Poly1305.
const ENCRYPTION_KEY_ENV: &str = "VECBOOST_ENCRYPTION_KEY";

/// HKDF field path used to derive per-field encryption keys.
const FIELD_PATH: &str = "vecboost.config.sensitive";

/// Key version for HKDF domain separation (bump on key rotation).
const KEY_VERSION: &str = "v1";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read the 32-byte master key from the environment.
///
/// Returns `None` when the env var is unset or empty (encryption disabled).
fn read_master_key() -> Option<[u8; 32]> {
    let key = std::env::var(ENCRYPTION_KEY_ENV).ok()?;
    if key.len() != 32 {
        log::warn!(
            "{ENCRYPTION_KEY_ENV} must be exactly 32 bytes for XChaCha20-Poly1305, \
             got {} bytes — encryption disabled",
            key.len()
        );
        return None;
    }
    let mut buf = [0u8; 32];
    buf.copy_from_slice(key.as_bytes());
    Some(buf)
}

/// Derive a 32-byte field key from the master key via HKDF-SHA256.
fn derive_key(master: &[u8; 32]) -> Result<[u8; 32], String> {
    derive_field_key(master, FIELD_PATH, KEY_VERSION)
        .map_err(|e| format!("key derivation failed: {e}"))
}

/// Encrypt plaintext bytes → hex-encoded `nonce ‖ ciphertext`.
fn encrypt_to_hex(plaintext: &[u8], master: &[u8; 32]) -> Result<String, String> {
    let field_key = derive_key(master)?;
    let crypto = XChaCha20Crypto::new();
    let (nonce, ciphertext) = crypto
        .encrypt(plaintext, &field_key)
        .map_err(|e| format!("encryption failed: {e}"))?;
    let mut combined = nonce;
    combined.extend_from_slice(&ciphertext);
    Ok(hex::encode(combined))
}

/// Decrypt hex-encoded `nonce ‖ ciphertext` → plaintext bytes.
fn decrypt_from_hex(encoded: &str, master: &[u8; 32]) -> Result<Vec<u8>, String> {
    let combined = hex::decode(encoded).map_err(|e| format!("invalid hex: {e}"))?;
    if combined.len() < confers::secret::NONCE_SIZE {
        return Err("encrypted value too short".to_string());
    }
    let (nonce_bytes, ciphertext) = combined.split_at(confers::secret::NONCE_SIZE);
    let field_key = derive_key(master)?;
    let crypto = XChaCha20Crypto::new();
    crypto
        .decrypt(nonce_bytes, ciphertext, &field_key)
        .map_err(|e| format!("decryption failed: {e}"))
}

// ---------------------------------------------------------------------------
// Serde helper modules for `#[serde(serialize_with / deserialize_with)]`
// ---------------------------------------------------------------------------

/// Serde helpers for `Option<String>` fields with transparent encryption.
///
/// - **Serialize**: encrypts the inner `String` (if `Some`) using XChaCha20-Poly1305.
/// - **Deserialize**: decrypts the hex-encoded ciphertext back to `String`.
/// - **Fallback**: when `VECBOOST_ENCRYPTION_KEY` is not set, values pass through
///   as plaintext (development convenience).
pub mod encrypted_option {
    use super::*;
    use serde::{Deserialize, Deserializer, Serializer};

    /// Serialize `Option<String>` with encryption.
    ///
    /// - `None` → serde `none` (omitted or null).
    /// - `Some(plaintext)` → hex-encoded encrypted string.
    pub fn serialize<S: Serializer>(
        value: &Option<String>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_none(),
            Some(plaintext) => {
                let encrypted = match read_master_key() {
                    Some(master) => encrypt_to_hex(plaintext.as_bytes(), &master)
                        .map_err(serde::ser::Error::custom)?,
                    None => {
                        // No encryption key → pass through as plaintext.
                        plaintext.clone()
                    }
                };
                serializer.serialize_str(&encrypted)
            }
        }
    }

    /// Deserialize `Option<String>` with decryption.
    ///
    /// - Missing/null → `None`.
    /// - Present → attempt decryption; on failure, treat as plaintext.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<String>, D::Error> {
        let opt = Option::<String>::deserialize(deserializer)?;
        match opt {
            None => Ok(None),
            Some(encoded) => {
                let decrypted = match read_master_key() {
                    Some(master) => match decrypt_from_hex(&encoded, &master) {
                        Ok(bytes) => String::from_utf8(bytes).map_err(serde::de::Error::custom)?,
                        Err(_) => {
                            // Decryption failed → assume plaintext value.
                            encoded
                        }
                    },
                    None => encoded,
                };
                Ok(Some(decrypted))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed 32-byte key used exclusively by unit tests.
    const TEST_KEY: [u8; 32] = *b"vecboost-test-encryption-key-32b"; // pragma: allowlist secret

    #[test]
    fn test_encrypt_decrypt_hex_roundtrip() {
        let plaintext = b"super-secret-jwt-token";
        let encrypted = encrypt_to_hex(plaintext, &TEST_KEY).expect("encrypt");
        let decrypted = decrypt_from_hex(&encrypted, &TEST_KEY).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_hex_empty_plaintext() {
        let plaintext = b"";
        let encrypted = encrypt_to_hex(plaintext, &TEST_KEY).expect("encrypt");
        let decrypted = decrypt_from_hex(&encrypted, &TEST_KEY).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_hex_unicode() {
        let plaintext = "你好世界🌍".as_bytes();
        let encrypted = encrypt_to_hex(plaintext, &TEST_KEY).expect("encrypt");
        let decrypted = decrypt_from_hex(&encrypted, &TEST_KEY).expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_decrypt_with_wrong_key_fails() {
        let plaintext = b"secret-data";
        let encrypted = encrypt_to_hex(plaintext, &TEST_KEY).expect("encrypt");
        let wrong_key = *b"vecboost-wrong-encryption-key32b"; // pragma: allowlist secret
        let result = decrypt_from_hex(&encrypted, &wrong_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_invalid_hex_fails() {
        let result = decrypt_from_hex("not-valid-hex!", &TEST_KEY);
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_too_short_value_fails() {
        // 10 hex chars = 5 bytes, less than NONCE_SIZE (24)
        let result = decrypt_from_hex("aabbccddee", &TEST_KEY);
        assert!(result.is_err());
    }
}
