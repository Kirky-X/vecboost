// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
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
    ) -> Result<RecoveryResult, AppError> {
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

    fn backup_corrupted_file(&self, file_path: &Path) -> Result<Option<String>, AppError> {
        if !file_path.exists() {
            return Ok(None);
        }

        let backup_dir = self.config.backup_dir.clone().unwrap_or_else(|| {
            file_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(".corrupted_backup")
        });

        fs::create_dir_all(&backup_dir)
            .map_err(|e| AppError::io_error(format!("Failed to create backup directory: {}", e)))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("{}_{}.corrupted", file_name, timestamp);
        let backup_path = backup_dir.join(&backup_name);

        fs::copy(file_path, &backup_path)
            .map_err(|e| AppError::io_error(format!("Failed to backup corrupted file: {}", e)))?;

        info!("Backed up corrupted file to: {:?}", backup_path);
        Ok(Some(backup_path.to_string_lossy().to_string()))
    }

    fn recover_single_file(
        &self,
        model_path: &Path,
        repo_id: Option<&str>,
        file_name: &str,
        attempts: &mut usize,
    ) -> Result<bool, AppError> {
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
            let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
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
                                AppError::io_error(format!("Failed to copy downloaded file: {}", e))
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
                            return Err(AppError::ModelLoadError(format!(
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
    ) -> Result<bool, AppError> {
        let api = Api::new().map_err(|e| AppError::ModelLoadError(e.to_string()))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        let file_path = model_path.join(file_name);

        for attempt in 1..=self.config.max_retries {
            *attempts += 1;
            info!("Download attempt {} for file: {}", attempt, file_name);

            match repo.get(file_name) {
                Ok(downloaded_path) => {
                    if downloaded_path != file_path {
                        fs::copy(&downloaded_path, &file_path).map_err(|e| {
                            AppError::io_error(format!("Failed to copy downloaded file: {}", e))
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
                        return Err(AppError::ModelLoadError(format!(
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
    ) -> Result<(), AppError> {
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
            return Err(AppError::ModelIntegrityError(format!(
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
    ) -> Result<RecoveryResult, AppError> {
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

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.backup_corrupted_files);
        assert!(config.verify_after_recovery);
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
        assert_eq!(result.attempts, 1);
    }

    #[test]
    fn test_model_recovery_creation() {
        let recovery = ModelRecovery::with_default_config();
        assert_eq!(recovery.config.max_retries, 3);
    }
}
