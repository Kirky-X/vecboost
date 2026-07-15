// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::VecboostError;
use crate::utils::hash::{ModelIntegrityReport, check_model_integrity};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub max_retries: usize,
    pub backup_corrupted_files: bool,
    pub backup_dir: Option<PathBuf>,
    pub verify_after_recovery: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backup_corrupted_files: true,
            backup_dir: None,
            verify_after_recovery: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub recovered_files: Vec<String>,
    pub failed_files: Vec<String>,
    pub backup_paths: Vec<String>,
    pub attempts: usize,
}

pub struct ModelRecovery {
    config: RecoveryConfig,
}

impl ModelRecovery {
    pub fn new(config: RecoveryConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(RecoveryConfig::default())
    }

    pub fn recover_corrupted_files(
        &self,
        model_name: &str,
        model_path: &Path,
        repo_id: Option<&str>,
        corrupted_files: &[String],
    ) -> Result<RecoveryResult, VecboostError> {
        info!(
            "Starting recovery for model '{}', {} corrupted files detected",
            model_name,
            corrupted_files.len()
        );

        let mut recovered_files = Vec::new();
        let mut failed_files = Vec::new();
        let mut backup_paths = Vec::new();
        let mut attempts = 0;

        for corrupted_file in corrupted_files {
            let file_path = Path::new(corrupted_file);
            let file_name = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            info!("Attempting to recover corrupted file: {}", file_name);

            if self.config.backup_corrupted_files
                && let Some(backup_path) = self.backup_corrupted_file(file_path)?
            {
                backup_paths.push(backup_path);
            }

            let recovery_success =
                self.recover_single_file(model_path, repo_id, file_name, &mut attempts)?;

            if recovery_success {
                recovered_files.push(corrupted_file.clone());
                info!("Successfully recovered file: {}", file_name);
            } else {
                failed_files.push(corrupted_file.clone());
                error!("Failed to recover file: {}", file_name);
            }
        }

        if self.config.verify_after_recovery && !recovered_files.is_empty() {
            info!("Verifying recovered files...");
            if let Err(e) = self.verify_recovered_files(model_name, model_path, &recovered_files) {
                warn!("Verification after recovery failed: {}", e);
            }
        }

        let success = failed_files.is_empty();

        Ok(RecoveryResult {
            success,
            recovered_files,
            failed_files,
            backup_paths,
            attempts,
        })
    }

    fn backup_corrupted_file(&self, file_path: &Path) -> Result<Option<String>, VecboostError> {
        if !file_path.exists() {
            return Ok(None);
        }

        let backup_dir = self.config.backup_dir.clone().unwrap_or_else(|| {
            file_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(".corrupted_backup")
        });

        fs::create_dir_all(&backup_dir).map_err(|e| {
            VecboostError::io_error(format!("Failed to create backup directory: {}", e))
        })?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("{}_{}.corrupted", file_name, timestamp);
        let backup_path = backup_dir.join(&backup_name);

        fs::copy(file_path, &backup_path).map_err(|e| {
            VecboostError::io_error(format!("Failed to backup corrupted file: {}", e))
        })?;

        info!("Backed up corrupted file to: {:?}", backup_path);
        Ok(Some(backup_path.to_string_lossy().to_string()))
    }

    fn recover_single_file(
        &self,
        model_path: &Path,
        repo_id: Option<&str>,
        file_name: &str,
        attempts: &mut usize,
    ) -> Result<bool, VecboostError> {
        let is_local_path = model_path.exists() && model_path.is_dir();

        if is_local_path {
            if let Some(repo) = repo_id {
                return self.download_from_huggingface(repo, model_path, file_name, attempts);
            } else {
                warn!(
                    "Local model path detected but no repo_id provided for recovery of {}",
                    file_name
                );
                return Ok(false);
            }
        } else {
            let api = Api::new().map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
            let repo = api.repo(Repo::new(
                model_path.to_string_lossy().into_owned(),
                RepoType::Model,
            ));

            let file_path = model_path.join(file_name);

            for attempt in 1..=self.config.max_retries {
                *attempts += 1;
                info!("Recovery attempt {} for file: {}", attempt, file_name);

                match repo.get(file_name) {
                    Ok(downloaded_path) => {
                        if downloaded_path != file_path {
                            fs::copy(&downloaded_path, &file_path).map_err(|e| {
                                VecboostError::io_error(format!(
                                    "Failed to copy downloaded file: {}",
                                    e
                                ))
                            })?;
                        }
                        return Ok(true);
                    }
                    Err(e) => {
                        warn!(
                            "Recovery attempt {} failed for file {}: {}",
                            attempt, file_name, e
                        );
                        if attempt == self.config.max_retries {
                            return Err(VecboostError::ModelLoadError(format!(
                                "Failed to recover file {} after {} attempts: {}",
                                file_name, self.config.max_retries, e
                            )));
                        }
                    }
                }
            }
        }

        Ok(false)
    }

    fn download_from_huggingface(
        &self,
        repo_id: &str,
        model_path: &Path,
        file_name: &str,
        attempts: &mut usize,
    ) -> Result<bool, VecboostError> {
        let api = Api::new().map_err(|e| VecboostError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        let file_path = model_path.join(file_name);

        for attempt in 1..=self.config.max_retries {
            *attempts += 1;
            info!("Download attempt {} for file: {}", attempt, file_name);

            match repo.get(file_name) {
                Ok(downloaded_path) => {
                    if downloaded_path != file_path {
                        fs::copy(&downloaded_path, &file_path).map_err(|e| {
                            VecboostError::io_error(format!(
                                "Failed to copy downloaded file: {}",
                                e
                            ))
                        })?;
                    }
                    return Ok(true);
                }
                Err(e) => {
                    warn!(
                        "Download attempt {} failed for file {}: {}",
                        attempt, file_name, e
                    );
                    if attempt == self.config.max_retries {
                        return Err(VecboostError::ModelLoadError(format!(
                            "Failed to download file {} after {} attempts: {}",
                            file_name, self.config.max_retries, e
                        )));
                    }
                }
            }
        }

        Ok(false)
    }

    fn verify_recovered_files(
        &self,
        model_name: &str,
        _model_path: &Path,
        recovered_files: &[String],
    ) -> Result<(), VecboostError> {
        let mut files_to_check = Vec::new();
        let mut min_sizes = HashMap::new();

        for file_path in recovered_files {
            let path = Path::new(file_path);
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let min_size = match file_name {
                n if n.contains("config.json") => 100,
                n if n.contains("tokenizer") => 1000,
                n if n.contains("model") => 1024 * 1024,
                _ => 100,
            };

            min_sizes.insert(file_path.clone(), min_size);
            files_to_check.push((file_path.clone(), None));
        }

        let integrity_report = check_model_integrity(model_name, files_to_check, Some(min_sizes))?;

        if !integrity_report.overall_valid {
            return Err(VecboostError::ModelIntegrityError(format!(
                "Verification failed after recovery. Corrupted files: {:?}",
                integrity_report.corrupted_files
            )));
        }

        info!("All recovered files verified successfully");
        Ok(())
    }

    pub fn attempt_recovery_with_integrity_check(
        &self,
        model_name: &str,
        model_path: &Path,
        repo_id: Option<&str>,
        integrity_report: &ModelIntegrityReport,
    ) -> Result<RecoveryResult, VecboostError> {
        if integrity_report.overall_valid {
            info!("Model integrity check passed, no recovery needed");
            return Ok(RecoveryResult {
                success: true,
                recovered_files: Vec::new(),
                failed_files: Vec::new(),
                backup_paths: Vec::new(),
                attempts: 0,
            });
        }

        info!(
            "Model integrity check failed, initiating recovery for {} corrupted files",
            integrity_report.corrupted_files.len()
        );

        self.recover_corrupted_files(
            model_name,
            model_path,
            repo_id,
            &integrity_report.corrupted_files,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.backup_corrupted_files);
        assert!(config.verify_after_recovery);
        assert!(config.backup_dir.is_none());
    }

    #[test]
    fn test_recovery_config_custom() {
        let config = RecoveryConfig {
            max_retries: 5,
            backup_corrupted_files: false,
            backup_dir: Some(PathBuf::from("/tmp/backups")),
            verify_after_recovery: false,
        };
        assert_eq!(config.max_retries, 5);
        assert!(!config.backup_corrupted_files);
        assert!(!config.verify_after_recovery);
        assert_eq!(config.backup_dir, Some(PathBuf::from("/tmp/backups")));
    }

    #[test]
    fn test_recovery_result_creation() {
        let result = RecoveryResult {
            success: true,
            recovered_files: vec!["file1.txt".to_string()],
            failed_files: vec![],
            backup_paths: vec!["backup.txt".to_string()],
            attempts: 1,
        };

        assert!(result.success);
        assert_eq!(result.recovered_files.len(), 1);
        assert_eq!(result.failed_files.len(), 0);
        assert_eq!(result.backup_paths.len(), 1);
        assert_eq!(result.attempts, 1);
    }

    #[test]
    fn test_recovery_result_failed() {
        let result = RecoveryResult {
            success: false,
            recovered_files: vec![],
            failed_files: vec!["bad.txt".to_string()],
            backup_paths: vec![],
            attempts: 3,
        };

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files.len(), 1);
        assert_eq!(result.attempts, 3);
    }

    #[test]
    fn test_model_recovery_creation() {
        let recovery = ModelRecovery::with_default_config();
        assert_eq!(recovery.config.max_retries, 3);
        assert!(recovery.config.backup_corrupted_files);
        assert!(recovery.config.verify_after_recovery);
        assert!(recovery.config.backup_dir.is_none());
    }

    #[test]
    fn test_model_recovery_new_with_custom_config() {
        let config = RecoveryConfig {
            max_retries: 10,
            backup_corrupted_files: false,
            backup_dir: Some(PathBuf::from("/custom/backup")),
            verify_after_recovery: false,
        };
        let recovery = ModelRecovery::new(config);
        assert_eq!(recovery.config.max_retries, 10);
        assert!(!recovery.config.backup_corrupted_files);
        assert!(!recovery.config.verify_after_recovery);
        assert_eq!(
            recovery.config.backup_dir,
            Some(PathBuf::from("/custom/backup"))
        );
    }

    #[test]
    fn test_backup_corrupted_file_nonexistent() {
        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .backup_corrupted_file(Path::new("/nonexistent/file.txt"))
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_backup_corrupted_file_default_backup_dir() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("corrupt.bin");
        fs::write(&file_path, b"corrupted data").unwrap();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.backup_corrupted_file(&file_path).unwrap();

        assert!(result.is_some());
        let backup_path = result.unwrap();
        let backup = Path::new(&backup_path);
        assert!(backup.exists());
        assert!(backup.starts_with(temp_dir.path().join(".corrupted_backup")));
        assert!(
            backup
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("corrupt.bin")
        );
        assert!(
            backup
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(".corrupted")
        );
        assert!(file_path.exists());
        assert_eq!(fs::read_to_string(backup).unwrap(), "corrupted data");
    }

    #[test]
    fn test_backup_corrupted_file_custom_backup_dir() {
        let temp_dir = tempdir().unwrap();
        let backup_dir = temp_dir.path().join("custom_backups");
        let file_path = temp_dir.path().join("corrupt.bin");
        fs::write(&file_path, b"corrupted data").unwrap();

        let config = RecoveryConfig {
            backup_dir: Some(backup_dir.clone()),
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery.backup_corrupted_file(&file_path).unwrap();

        assert!(result.is_some());
        let backup_path = result.unwrap();
        let backup = Path::new(&backup_path);
        assert!(backup.starts_with(&backup_dir));
        assert!(backup.exists());
    }

    #[test]
    fn test_backup_corrupted_file_copies_content() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("model.safetensors");
        let content = b"some binary content here";
        fs::write(&file_path, content).unwrap();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.backup_corrupted_file(&file_path).unwrap();

        assert!(result.is_some());
        let backup_path = result.unwrap();
        assert_eq!(fs::read(backup_path).unwrap(), content);
    }

    #[test]
    fn test_backup_corrupted_file_preserves_original() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("keep.bin");
        fs::write(&file_path, b"original").unwrap();

        let recovery = ModelRecovery::with_default_config();
        let _ = recovery.backup_corrupted_file(&file_path).unwrap();

        assert!(file_path.exists());
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "original");
    }

    #[test]
    fn test_recover_single_file_local_dir_no_repo_id() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let mut attempts = 0;

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_single_file(model_path, None, "config.json", &mut attempts)
            .unwrap();

        assert!(!result);
        assert_eq!(attempts, 0);
    }

    #[test]
    #[ignore = "Requires network access to HuggingFace Hub"]
    fn test_recover_single_file_local_dir_with_repo_id() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let mut attempts = 0;

        let recovery = ModelRecovery::with_default_config();
        let _ = recovery.recover_single_file(
            model_path,
            Some("nonexistent/repo-12345"),
            "config.json",
            &mut attempts,
        );
    }

    #[test]
    #[ignore = "Requires network access to HuggingFace Hub"]
    fn test_recover_single_file_non_local_path() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("nonexistent-repo-id");
        let mut attempts = 0;

        let recovery = ModelRecovery::with_default_config();
        let _ = recovery.recover_single_file(&model_path, None, "config.json", &mut attempts);
    }

    #[test]
    fn test_verify_recovered_files_empty() {
        let temp_dir = tempdir().unwrap();
        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files("test_model", temp_dir.path(), &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_valid_large_file() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("config.json");
        fs::write(&file_path, b"x".repeat(200)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_missing_file() {
        let temp_dir = tempdir().unwrap();
        let missing_file = temp_dir
            .path()
            .join("missing.json")
            .to_string_lossy()
            .to_string();

        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[missing_file]);
        assert!(result.is_err());
        match result {
            Err(VecboostError::ModelIntegrityError(msg)) => {
                assert!(msg.contains("Verification failed"));
            }
            Err(e) => panic!("Expected ModelIntegrityError, got: {:?}", e),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_verify_recovered_files_too_small_config() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("config.json");
        fs::write(&file_path, b"x".repeat(50)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_recovered_files_config_json_exact_min_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("config.json");
        fs::write(&file_path, b"x".repeat(100)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_tokenizer_min_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("tokenizer.json");
        fs::write(&file_path, b"x".repeat(999)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files(
            "test_model",
            temp_dir.path(),
            std::slice::from_ref(&file_path_str),
        );
        assert!(result.is_err());

        fs::write(&file_path, b"x".repeat(1000)).unwrap();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_model_min_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("model.safetensors");
        fs::write(&file_path, b"x".repeat(1024 * 1024 - 1)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files(
            "test_model",
            temp_dir.path(),
            std::slice::from_ref(&file_path_str),
        );
        assert!(result.is_err());

        fs::write(&file_path, b"x".repeat(1024 * 1024)).unwrap();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_other_file_min_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("random.txt");
        fs::write(&file_path, b"x".repeat(99)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files(
            "test_model",
            temp_dir.path(),
            std::slice::from_ref(&file_path_str),
        );
        assert!(result.is_err());

        fs::write(&file_path, b"x".repeat(100)).unwrap();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_multiple_files_all_valid() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        fs::write(&config_path, b"x".repeat(200)).unwrap();
        fs::write(&tokenizer_path, b"x".repeat(2000)).unwrap();

        let files = vec![
            config_path.to_string_lossy().to_string(),
            tokenizer_path.to_string_lossy().to_string(),
        ];

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files("test_model", temp_dir.path(), &files);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_multiple_files_one_invalid() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");
        fs::write(&config_path, b"x".repeat(200)).unwrap();
        fs::write(&tokenizer_path, b"x".repeat(100)).unwrap();

        let files = vec![
            config_path.to_string_lossy().to_string(),
            tokenizer_path.to_string_lossy().to_string(),
        ];

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.verify_recovered_files("test_model", temp_dir.path(), &files);
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_corrupted_files_empty_list() {
        let temp_dir = tempdir().unwrap();
        let recovery = ModelRecovery::with_default_config();

        let result = recovery
            .recover_corrupted_files("test_model", temp_dir.path(), None, &[])
            .unwrap();

        assert!(result.success);
        assert!(result.recovered_files.is_empty());
        assert!(result.failed_files.is_empty());
        assert!(result.backup_paths.is_empty());
        assert_eq!(result.attempts, 0);
    }

    #[test]
    fn test_recover_corrupted_files_local_no_repo_id_fails() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_corrupted_files(
                "test_model",
                model_path,
                None,
                std::slice::from_ref(&corrupted_file_str),
            )
            .unwrap();

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files, vec![corrupted_file_str]);
        assert_eq!(result.backup_paths.len(), 1);
        assert_eq!(result.attempts, 0);
    }

    #[test]
    fn test_recover_corrupted_files_backup_disabled() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let config = RecoveryConfig {
            backup_corrupted_files: false,
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &[corrupted_file_str])
            .unwrap();

        assert!(!result.success);
        assert!(result.backup_paths.is_empty());
    }

    #[test]
    fn test_recover_corrupted_files_verify_disabled() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let config = RecoveryConfig {
            verify_after_recovery: false,
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &[corrupted_file_str])
            .unwrap();

        assert!(!result.success);
    }

    #[test]
    fn test_recover_corrupted_files_custom_backup_dir() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let backup_dir = temp_dir.path().join("my_backups");
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let config = RecoveryConfig {
            backup_dir: Some(backup_dir.clone()),
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &[corrupted_file_str])
            .unwrap();

        assert_eq!(result.backup_paths.len(), 1);
        let backup_path = Path::new(&result.backup_paths[0]);
        assert!(backup_path.starts_with(&backup_dir));
        assert!(backup_path.exists());
    }

    #[test]
    fn test_recover_corrupted_files_nonexistent_file_no_backup() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file_str = temp_dir
            .path()
            .join("nonexistent.json")
            .to_string_lossy()
            .to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_corrupted_files(
                "test_model",
                model_path,
                None,
                std::slice::from_ref(&corrupted_file_str),
            )
            .unwrap();

        assert!(!result.success);
        assert!(result.backup_paths.is_empty());
        assert_eq!(result.failed_files, vec![corrupted_file_str]);
    }

    #[test]
    fn test_recover_corrupted_files_multiple_files_mixed() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let file1 = temp_dir.path().join("config.json");
        let file2 = temp_dir.path().join("tokenizer.json");
        fs::write(&file1, b"corrupted1").unwrap();
        fs::write(&file2, b"corrupted2").unwrap();
        let file1_str = file1.to_string_lossy().to_string();
        let file2_str = file2.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_corrupted_files(
                "test_model",
                model_path,
                None,
                &[file1_str.clone(), file2_str.clone()],
            )
            .unwrap();

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files.len(), 2);
        assert!(result.failed_files.contains(&file1_str));
        assert!(result.failed_files.contains(&file2_str));
        assert_eq!(result.backup_paths.len(), 2);
    }

    #[test]
    fn test_recover_corrupted_files_preserves_original_after_backup() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"original corrupted content").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let _ = recovery
            .recover_corrupted_files("test_model", model_path, None, &[corrupted_file_str])
            .unwrap();

        assert_eq!(
            fs::read_to_string(&corrupted_file).unwrap(),
            "original corrupted content"
        );
    }

    #[test]
    fn test_attempt_recovery_overall_valid() {
        let temp_dir = tempdir().unwrap();
        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: true,
            corrupted_files: vec![],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                temp_dir.path(),
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(result.success);
        assert!(result.recovered_files.is_empty());
        assert!(result.failed_files.is_empty());
        assert!(result.backup_paths.is_empty());
        assert_eq!(result.attempts, 0);
    }

    #[test]
    fn test_attempt_recovery_invalid_empty_corrupted() {
        let temp_dir = tempdir().unwrap();
        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: false,
            corrupted_files: vec![],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                temp_dir.path(),
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(result.success);
        assert!(result.recovered_files.is_empty());
        assert!(result.failed_files.is_empty());
        assert_eq!(result.attempts, 0);
    }

    #[test]
    fn test_attempt_recovery_invalid_with_corrupted_files() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: false,
            corrupted_files: vec![corrupted_file_str.clone()],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                model_path,
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.failed_files, vec![corrupted_file_str]);
    }

    #[test]
    fn test_attempt_recovery_invalid_with_corrupted_files_no_repo() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let file1 = temp_dir.path().join("config.json");
        let file2 = temp_dir.path().join("tokenizer.json");
        fs::write(&file1, b"bad1").unwrap();
        fs::write(&file2, b"bad2").unwrap();
        let file1_str = file1.to_string_lossy().to_string();
        let file2_str = file2.to_string_lossy().to_string();

        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: false,
            corrupted_files: vec![file1_str, file2_str],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                model_path,
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.failed_files.len(), 2);
        assert_eq!(result.backup_paths.len(), 2);
    }

    #[test]
    #[ignore = "Requires network access to HuggingFace Hub"]
    fn test_download_from_huggingface_invalid_repo() {
        let temp_dir = tempdir().unwrap();
        let mut attempts = 0;

        let recovery = ModelRecovery::with_default_config();
        let result = recovery.download_from_huggingface(
            "nonexistent/repo-12345",
            temp_dir.path(),
            "config.json",
            &mut attempts,
        );

        assert!(result.is_err());
        assert_eq!(attempts, 3);
    }

    #[test]
    #[ignore = "Requires network access to HuggingFace Hub"]
    fn test_download_from_huggingface_max_retries_one() {
        let temp_dir = tempdir().unwrap();
        let mut attempts = 0;

        let config = RecoveryConfig {
            max_retries: 1,
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery.download_from_huggingface(
            "nonexistent/repo-12345",
            temp_dir.path(),
            "config.json",
            &mut attempts,
        );

        assert!(result.is_err());
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_backup_corrupted_file_backup_dir_under_file_fails() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("corrupt.bin");
        fs::write(&file_path, b"corrupted data").unwrap();
        let blocking_file = temp_dir.path().join("blocking_file");
        fs::write(&blocking_file, b"data").unwrap();
        let backup_dir = blocking_file.join("subdir");

        let config = RecoveryConfig {
            backup_dir: Some(backup_dir),
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery.backup_corrupted_file(&file_path);
        assert!(result.is_err());
        match result {
            Err(VecboostError::IoError(msg)) => {
                assert!(msg.contains("Failed to create backup directory"));
            }
            Err(e) => panic!("Expected IoError, got: {:?}", e),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_recover_corrupted_files_backup_error_propagates() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let blocking_file = temp_dir.path().join("blocking_file");
        fs::write(&blocking_file, b"data").unwrap();
        let backup_dir = blocking_file.join("subdir");

        let config = RecoveryConfig {
            backup_dir: Some(backup_dir),
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery.recover_corrupted_files(
            "test_model",
            model_path,
            None,
            &[corrupted_file.to_string_lossy().to_string()],
        );
        assert!(result.is_err());
        match result {
            Err(VecboostError::IoError(msg)) => {
                assert!(msg.contains("Failed to create backup directory"));
            }
            Err(e) => panic!("Expected IoError, got: {:?}", e),
            Ok(_) => panic!("Expected error"),
        }
    }

    #[test]
    fn test_recover_corrupted_files_mixed_existing_and_nonexisting() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let existing_file = temp_dir.path().join("config.json");
        fs::write(&existing_file, b"corrupted").unwrap();
        let nonexisting_file = temp_dir
            .path()
            .join("nonexistent.json")
            .to_string_lossy()
            .to_string();
        let existing_file_str = existing_file.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_corrupted_files(
                "test_model",
                model_path,
                None,
                &[existing_file_str.clone(), nonexisting_file.clone()],
            )
            .unwrap();

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files.len(), 2);
        assert!(result.failed_files.contains(&existing_file_str));
        assert!(result.failed_files.contains(&nonexisting_file));
        assert_eq!(result.backup_paths.len(), 1);
    }

    #[test]
    fn test_verify_recovered_files_with_file_in_subdirectory() {
        let temp_dir = tempdir().unwrap();
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        let file_path = subdir.join("config.json");
        fs::write(&file_path, b"x".repeat(200)).unwrap();
        let file_path_str = file_path.to_string_lossy().to_string();

        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &[file_path_str]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_recovered_files_empty_path_string() {
        let temp_dir = tempdir().unwrap();
        let recovery = ModelRecovery::with_default_config();
        let result =
            recovery.verify_recovered_files("test_model", temp_dir.path(), &["".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_attempt_recovery_overall_valid_with_nonempty_corrupted_list() {
        let temp_dir = tempdir().unwrap();
        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: true,
            corrupted_files: vec!["some/file.json".to_string()],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                temp_dir.path(),
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(result.success);
        assert!(result.recovered_files.is_empty());
        assert!(result.failed_files.is_empty());
        assert!(result.backup_paths.is_empty());
        assert_eq!(result.attempts, 0);
    }

    #[test]
    fn test_attempt_recovery_invalid_with_nonexisting_corrupted_files() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file_str = temp_dir
            .path()
            .join("nonexistent.json")
            .to_string_lossy()
            .to_string();

        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: false,
            corrupted_files: vec![corrupted_file_str.clone()],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                model_path,
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.failed_files, vec![corrupted_file_str]);
        assert!(result.backup_paths.is_empty());
    }

    #[test]
    fn test_recover_corrupted_files_all_file_types_backed_up() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let config_file = temp_dir.path().join("config.json");
        let tokenizer_file = temp_dir.path().join("tokenizer.json");
        let model_file = temp_dir.path().join("model.safetensors");
        let other_file = temp_dir.path().join("vocab.txt");
        fs::write(&config_file, b"bad").unwrap();
        fs::write(&tokenizer_file, b"bad").unwrap();
        fs::write(&model_file, b"bad").unwrap();
        fs::write(&other_file, b"bad").unwrap();

        let files = vec![
            config_file.to_string_lossy().to_string(),
            tokenizer_file.to_string_lossy().to_string(),
            model_file.to_string_lossy().to_string(),
            other_file.to_string_lossy().to_string(),
        ];

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &files)
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.failed_files.len(), 4);
        assert_eq!(result.backup_paths.len(), 4);
        for backup_path in &result.backup_paths {
            assert!(Path::new(backup_path).exists());
        }
    }

    #[test]
    fn test_recover_corrupted_files_preserves_multiple_originals() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let file1 = temp_dir.path().join("config.json");
        let file2 = temp_dir.path().join("tokenizer.json");
        fs::write(&file1, b"original1").unwrap();
        fs::write(&file2, b"original2").unwrap();

        let files = vec![
            file1.to_string_lossy().to_string(),
            file2.to_string_lossy().to_string(),
        ];

        let recovery = ModelRecovery::with_default_config();
        let _ = recovery
            .recover_corrupted_files("test_model", model_path, None, &files)
            .unwrap();

        assert_eq!(fs::read_to_string(&file1).unwrap(), "original1");
        assert_eq!(fs::read_to_string(&file2).unwrap(), "original2");
    }

    /// 验证 recover_corrupted_files 处理空字符串路径时使用 "unknown" 作为文件名。
    #[test]
    fn test_recover_corrupted_files_empty_string_path() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let recovery = ModelRecovery::with_default_config();

        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &["".to_string()])
            .unwrap();

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files, vec!["".to_string()]);
        assert!(result.backup_paths.is_empty());
        assert_eq!(result.attempts, 0);
    }

    /// 验证 backup_corrupted_file 对目录路径返回错误(fs::copy 不支持目录)。
    #[test]
    fn test_backup_corrupted_file_directory_path_fails() {
        let temp_dir = tempdir().unwrap();
        let recovery = ModelRecovery::with_default_config();
        let result = recovery.backup_corrupted_file(temp_dir.path());
        assert!(result.is_err());
        match result {
            Err(VecboostError::IoError(msg)) => {
                assert!(msg.contains("Failed to backup corrupted file"));
            }
            Err(e) => panic!("Expected IoError, got: {:?}", e),
            Ok(_) => panic!("Expected error for directory backup"),
        }
    }

    /// 验证 recover_single_file 在 max_retries=0 时不进入重试循环,
    /// 直接返回 Ok(false)(本地路径无 repo_id 分支)。
    #[test]
    fn test_recover_single_file_local_no_repo_zero_retries() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let mut attempts = 0;

        let config = RecoveryConfig {
            max_retries: 0,
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery
            .recover_single_file(model_path, None, "config.json", &mut attempts)
            .unwrap();

        assert!(!result);
        assert_eq!(attempts, 0, "attempts must remain 0 with max_retries=0");
    }

    /// 验证 recover_corrupted_files 在 recovered_files 为空时跳过 verify_after_recovery,
    /// 即使 verify_after_recovery=true 也不调用 verify_recovered_files。
    #[test]
    fn test_recover_corrupted_files_verify_skipped_when_no_recovered() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let corrupted_file = temp_dir.path().join("config.json");
        fs::write(&corrupted_file, b"corrupted").unwrap();
        let corrupted_file_str = corrupted_file.to_string_lossy().to_string();

        let config = RecoveryConfig {
            verify_after_recovery: true,
            ..RecoveryConfig::default()
        };
        let recovery = ModelRecovery::new(config);
        let result = recovery
            .recover_corrupted_files("test_model", model_path, None, &[corrupted_file_str])
            .unwrap();

        assert!(!result.success);
        assert!(result.recovered_files.is_empty());
        assert_eq!(result.failed_files.len(), 1);
    }

    /// 验证 recover_corrupted_files 处理无文件名路径(如 "/")时使用 "unknown"。
    #[test]
    fn test_recover_corrupted_files_root_path_uses_unknown_filename() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();
        let recovery = ModelRecovery::with_default_config();

        // "/" 是目录,backup 会失败(fs::copy 不支持目录),错误通过 ? 传播
        let result =
            recovery.recover_corrupted_files("test_model", model_path, None, &["/".to_string()]);
        assert!(
            result.is_err(),
            "backup of directory '/' should propagate error"
        );
    }

    /// 验证 attempt_recovery_with_integrity_check 在 corrupted_files 含空字符串时正确处理。
    #[test]
    fn test_attempt_recovery_with_empty_string_corrupted_file() {
        let temp_dir = tempdir().unwrap();
        let integrity_report = ModelIntegrityReport {
            model_name: "test_model".to_string(),
            files_checked: vec![],
            overall_valid: false,
            corrupted_files: vec!["".to_string()],
        };

        let recovery = ModelRecovery::with_default_config();
        let result = recovery
            .attempt_recovery_with_integrity_check(
                "test_model",
                temp_dir.path(),
                None,
                &integrity_report,
            )
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.failed_files, vec!["".to_string()]);
        assert!(result.backup_paths.is_empty());
    }
}
