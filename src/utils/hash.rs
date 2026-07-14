// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
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

pub fn compute_sha256<P: AsRef<Path>>(file_path: P) -> Result<String, VecboostError> {
    let path = file_path.as_ref();

    let mut file = File::open(path).map_err(|e| {
        VecboostError::ModelLoadError(format!("Failed to open file {:?}: {}", path, e))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            VecboostError::ModelLoadError(format!("Failed to read file {:?}: {}", path, e))
        })?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

pub fn verify_sha256<P: AsRef<Path>>(
    file_path: P,
    expected_hash: &str,
) -> Result<bool, VecboostError> {
    let actual_hash = compute_sha256(file_path)?;

    let normalized_expected = expected_hash.trim().to_lowercase();
    let normalized_actual = actual_hash.to_lowercase();

    Ok(normalized_expected == normalized_actual)
}

pub fn verify_file_exists<P: AsRef<Path>>(file_path: P) -> Result<bool, VecboostError> {
    let path = file_path.as_ref();
    Ok(path.exists())
}

pub fn verify_file_readable<P: AsRef<Path>>(file_path: P) -> Result<bool, VecboostError> {
    let path = file_path.as_ref();
    File::open(path).map_err(|e| {
        VecboostError::ModelLoadError(format!("Failed to open file {:?}: {}", path, e))
    })?;
    Ok(true)
}

pub fn verify_file_size<P: AsRef<Path>>(
    file_path: P,
    min_size_bytes: u64,
) -> Result<bool, VecboostError> {
    let path = file_path.as_ref();
    let metadata = std::fs::metadata(path).map_err(|e| {
        VecboostError::ModelLoadError(format!("Failed to get metadata for {:?}: {}", path, e))
    })?;

    Ok(metadata.len() >= min_size_bytes)
}

pub fn check_model_integrity(
    model_name: &str,
    files: Vec<(String, Option<String>)>,
    min_file_sizes: Option<HashMap<String, u64>>,
) -> Result<ModelIntegrityReport, VecboostError> {
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

    #[test]
    fn test_compute_sha256_nonexistent_file() {
        let result = compute_sha256("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_sha256_with_whitespace() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test").unwrap();

        let hash = compute_sha256(temp_file.path()).unwrap();
        let padded_hash = format!("  {}  ", hash);
        let result = verify_sha256(temp_file.path(), &padded_hash).unwrap();
        assert!(result);
    }

    #[test]
    fn test_verify_file_exists_true() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"data").unwrap();
        assert!(verify_file_exists(temp_file.path()).unwrap());
    }

    #[test]
    fn test_verify_file_exists_false() {
        assert!(!verify_file_exists("/nonexistent/file.txt").unwrap());
    }

    #[test]
    fn test_verify_file_readable_success() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"readable").unwrap();
        assert!(verify_file_readable(temp_file.path()).unwrap());
    }

    #[test]
    fn test_verify_file_readable_failure() {
        let result = verify_file_readable("/nonexistent/file.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_file_size_meets_minimum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"12345678").unwrap();
        assert!(verify_file_size(temp_file.path(), 8).unwrap());
        assert!(verify_file_size(temp_file.path(), 4).unwrap());
    }

    #[test]
    fn test_verify_file_size_below_minimum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"small").unwrap();
        assert!(!verify_file_size(temp_file.path(), 100).unwrap());
    }

    #[test]
    fn test_verify_file_size_nonexistent() {
        let result = verify_file_size("/nonexistent/file.txt", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_model_integrity_all_valid() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"model data").unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();
        let hash = compute_sha256(&path).unwrap();

        let files = vec![(path.clone(), Some(hash))];
        let report = check_model_integrity("test_model", files, None).unwrap();

        assert!(report.overall_valid);
        assert!(report.corrupted_files.is_empty());
        assert_eq!(report.files_checked.len(), 1);
        assert!(report.files_checked[0].is_valid);
        assert!(report.files_checked[0].actual_hash.is_some());
    }

    #[test]
    fn test_check_model_integrity_missing_file() {
        let files = vec![("/nonexistent/model.bin".to_string(), None)];
        let report = check_model_integrity("test_model", files, None).unwrap();

        assert!(!report.overall_valid);
        assert_eq!(report.corrupted_files.len(), 1);
        assert!(!report.files_checked[0].is_valid);
        assert!(report.files_checked[0].error_message.is_some());
    }

    #[test]
    fn test_check_model_integrity_hash_mismatch() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"model data").unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();

        let files = vec![(path, Some("wrong_hash".to_string()))];
        let report = check_model_integrity("test_model", files, None).unwrap();

        assert!(!report.overall_valid);
        assert_eq!(report.corrupted_files.len(), 1);
        assert!(!report.files_checked[0].is_valid);
    }

    #[test]
    fn test_check_model_integrity_no_expected_hash() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"model data").unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();

        let files = vec![(path, None)];
        let report = check_model_integrity("test_model", files, None).unwrap();

        assert!(report.overall_valid);
        assert!(report.files_checked[0].is_valid);
    }

    #[test]
    fn test_check_model_integrity_size_check() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"tiny").unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut min_sizes = HashMap::new();
        min_sizes.insert(path.clone(), 100_u64);

        let files = vec![(path, None)];
        let report = check_model_integrity("test_model", files, Some(min_sizes)).unwrap();

        assert!(!report.overall_valid);
        assert_eq!(report.corrupted_files.len(), 1);
    }

    #[test]
    fn test_check_model_integrity_size_check_pass() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"enough data here").unwrap();
        let path = temp_file.path().to_str().unwrap().to_string();

        let mut min_sizes = HashMap::new();
        min_sizes.insert(path.clone(), 4_u64);

        let files = vec![(path, None)];
        let report = check_model_integrity("test_model", files, Some(min_sizes)).unwrap();

        assert!(report.overall_valid);
        assert!(report.corrupted_files.is_empty());
    }

    #[test]
    fn test_check_model_integrity_multiple_files_all_valid() {
        let mut temp_a = NamedTempFile::new().unwrap();
        temp_a.write_all(b"file a content").unwrap();
        let path_a = temp_a.path().to_str().unwrap().to_string();
        let hash_a = compute_sha256(&path_a).unwrap();

        let mut temp_b = NamedTempFile::new().unwrap();
        temp_b.write_all(b"file b content").unwrap();
        let path_b = temp_b.path().to_str().unwrap().to_string();
        let hash_b = compute_sha256(&path_b).unwrap();

        let files = vec![
            (path_a.clone(), Some(hash_a)),
            (path_b.clone(), Some(hash_b)),
        ];
        let report = check_model_integrity("multi_model", files, None).unwrap();

        assert!(report.overall_valid);
        assert!(report.corrupted_files.is_empty());
        assert_eq!(report.files_checked.len(), 2);
        assert!(report.files_checked.iter().all(|c| c.is_valid));
    }

    #[test]
    fn test_check_model_integrity_mixed_files() {
        let mut temp_valid = NamedTempFile::new().unwrap();
        temp_valid.write_all(b"valid content").unwrap();
        let valid_path = temp_valid.path().to_str().unwrap().to_string();
        let valid_hash = compute_sha256(&valid_path).unwrap();

        let files = vec![
            (valid_path.clone(), Some(valid_hash)),
            ("/nonexistent/missing.bin".to_string(), None),
            (valid_path.clone(), Some("wrong_hash".to_string())),
        ];
        let report = check_model_integrity("mixed_model", files, None).unwrap();

        assert!(!report.overall_valid);
        assert_eq!(report.corrupted_files.len(), 2);
        assert_eq!(report.files_checked.len(), 3);
        assert!(report.files_checked[0].is_valid);
        assert!(!report.files_checked[1].is_valid);
        assert!(!report.files_checked[2].is_valid);
    }

    #[test]
    fn test_check_model_integrity_empty_files_list() {
        let report = check_model_integrity("empty_model", vec![], None).unwrap();
        assert!(report.overall_valid);
        assert!(report.files_checked.is_empty());
        assert!(report.corrupted_files.is_empty());
    }

    #[test]
    fn test_compute_sha256_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let hash = compute_sha256(temp_file.path()).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_compute_sha256_known_content() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();
        let hash = compute_sha256(temp_file.path()).unwrap();
        assert_eq!(
            hash,
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        );
    }

    #[test]
    fn test_compute_sha256_large_content() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let large_content = vec![0xABu8; 100_000];
        temp_file.write_all(&large_content).unwrap();
        let hash = compute_sha256(temp_file.path()).unwrap();
        assert_eq!(hash.len(), 64);
        let hash2 = compute_sha256(temp_file.path()).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_verify_sha256_empty_expected() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"data").unwrap();
        let result = verify_sha256(temp_file.path(), "").unwrap();
        assert!(!result);
    }

    #[test]
    fn test_verify_file_size_exact_minimum() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"1234").unwrap();
        assert!(verify_file_size(temp_file.path(), 4).unwrap());
    }

    #[test]
    fn test_check_model_integrity_report_model_name() {
        let files = vec![("/nonexistent/file.bin".to_string(), None)];
        let report = check_model_integrity("my_custom_model", files, None).unwrap();
        assert_eq!(report.model_name, "my_custom_model");
    }

    #[test]
    fn test_check_model_integrity_missing_with_min_size() {
        let missing_path = "/nonexistent/missing_with_size.bin".to_string();
        let mut min_sizes = HashMap::new();
        min_sizes.insert(missing_path.clone(), 100_u64);

        let files = vec![(missing_path.clone(), None)];
        let report = check_model_integrity("test_model", files, Some(min_sizes)).unwrap();

        assert!(!report.overall_valid);
        assert_eq!(report.corrupted_files.len(), 1);
        assert!(report.files_checked[0].error_message.is_some());
        assert!(
            report.files_checked[0]
                .error_message
                .as_ref()
                .unwrap()
                .contains("does not exist")
        );
    }
}
