use crate::error::AppError;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileIntegrityCheck {
    pub file_path: String,
    pub expected_hash: Option<String>,
    pub actual_hash: Option<String>,
    pub is_valid: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelIntegrityReport {
    pub model_name: String,
    pub files_checked: Vec<FileIntegrityCheck>,
    pub overall_valid: bool,
    pub corrupted_files: Vec<String>,
}

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

pub fn verify_file_exists<P: AsRef<Path>>(file_path: P) -> Result<bool, AppError> {
    let path = file_path.as_ref();
    Ok(path.exists())
}

pub fn verify_file_readable<P: AsRef<Path>>(file_path: P) -> Result<bool, AppError> {
    let path = file_path.as_ref();
    File::open(path)
        .map_err(|e| AppError::ModelLoadError(format!("Failed to open file {:?}: {}", path, e)))?;
    Ok(true)
}

pub fn verify_file_size<P: AsRef<Path>>(
    file_path: P,
    min_size_bytes: u64,
) -> Result<bool, AppError> {
    let path = file_path.as_ref();
    let metadata = std::fs::metadata(path).map_err(|e| {
        AppError::ModelLoadError(format!("Failed to get metadata for {:?}: {}", path, e))
    })?;

    Ok(metadata.len() >= min_size_bytes)
}

pub fn check_model_integrity(
    model_name: &str,
    files: Vec<(String, Option<String>)>,
    min_file_sizes: Option<HashMap<String, u64>>,
) -> Result<ModelIntegrityReport, AppError> {
    let mut checks = Vec::new();
    let mut corrupted_files = Vec::new();

    for (file_path, expected_hash) in files {
        let mut check = FileIntegrityCheck {
            file_path: file_path.clone(),
            expected_hash: expected_hash.clone(),
            actual_hash: None,
            is_valid: false,
            error_message: None,
        };

        let path = Path::new(&file_path);

        if !path.exists() {
            check.is_valid = false;
            check.error_message = Some("File does not exist".to_string());
            corrupted_files.push(file_path);
            checks.push(check);
            continue;
        }

        match verify_file_readable(path) {
            Ok(true) => {}
            Err(e) => {
                check.is_valid = false;
                check.error_message = Some(format!("File not readable: {}", e));
                corrupted_files.push(file_path);
                checks.push(check);
                continue;
            }
            _ => {}
        }

        if let Some(min_sizes) = &min_file_sizes
            && let Some(min_size) = min_sizes.get(&file_path)
        {
            match verify_file_size(path, *min_size) {
                Ok(true) => {}
                Ok(false) => {
                    check.is_valid = false;
                    check.error_message = Some(format!(
                        "File size below minimum threshold ({} bytes)",
                        min_size
                    ));
                    corrupted_files.push(file_path);
                    checks.push(check);
                    continue;
                }
                Err(e) => {
                    check.is_valid = false;
                    check.error_message = Some(format!("Failed to check file size: {}", e));
                    corrupted_files.push(file_path);
                    checks.push(check);
                    continue;
                }
            }
        }

        match compute_sha256(path) {
            Ok(hash) => {
                check.actual_hash = Some(hash.clone());
                if let Some(expected) = &expected_hash {
                    match verify_sha256(path, expected) {
                        Ok(is_valid) => {
                            check.is_valid = is_valid;
                            if !is_valid {
                                check.error_message = Some(format!(
                                    "SHA256 mismatch: expected {}, got {}",
                                    expected, hash
                                ));
                                corrupted_files.push(file_path);
                            }
                        }
                        Err(e) => {
                            check.is_valid = false;
                            check.error_message = Some(format!("Hash verification failed: {}", e));
                            corrupted_files.push(file_path);
                        }
                    }
                } else {
                    check.is_valid = true;
                }
            }
            Err(e) => {
                check.is_valid = false;
                check.error_message = Some(format!("Failed to compute hash: {}", e));
                corrupted_files.push(file_path);
            }
        }

        checks.push(check);
    }

    let overall_valid = checks.iter().all(|c| c.is_valid);

    Ok(ModelIntegrityReport {
        model_name: model_name.to_string(),
        files_checked: checks,
        overall_valid,
        corrupted_files,
    })
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
