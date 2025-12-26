use crate::error::AppError;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn compute_sha256<P: AsRef<Path>>(file_path: P) -> Result<String, AppError> {
    let path = file_path.as_ref();

    let mut file = File::open(path)
        .map_err(|e| AppError::ModelLoadError(format!("Failed to open file {:?}: {}", path, e)))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            AppError::ModelLoadError(format!("Failed to read file {:?}: {}", path, e))
        })?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

pub fn verify_sha256<P: AsRef<Path>>(file_path: P, expected_hash: &str) -> Result<bool, AppError> {
    let actual_hash = compute_sha256(file_path)?;

    let normalized_expected = expected_hash.trim().to_lowercase();
    let normalized_actual = actual_hash.to_lowercase();

    Ok(normalized_expected == normalized_actual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compute_sha256() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();

        let hash = compute_sha256(temp_file.path()).unwrap();
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn test_verify_sha256_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();

        let hash = compute_sha256(temp_file.path()).unwrap();
        let result = verify_sha256(temp_file.path(), &hash).unwrap();
        assert!(result);
    }

    #[test]
    fn test_verify_sha256_invalid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();

        let result = verify_sha256(temp_file.path(), "invalid_hash").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_verify_sha256_case_insensitive() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();

        let hash = compute_sha256(temp_file.path()).unwrap();
        let uppercase_hash = hash.to_uppercase();
        let result = verify_sha256(temp_file.path(), &uppercase_hash).unwrap();
        assert!(result);
    }
}
