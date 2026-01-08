// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::error::AppError;
use std::path::{Path, PathBuf};

/// 路径验证器，用于防止路径遍历攻击
pub struct PathValidator {
    /// 允许的根目录白名单
    allowed_roots: Vec<PathBuf>,
}

impl PathValidator {
    /// 创建新的路径验证器
    pub fn new() -> Self {
        Self {
            allowed_roots: Vec::new(),
        }
    }

    /// 添加允许的根目录
    pub fn add_allowed_root<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        self.allowed_roots.push(path);
        self
    }

    /// 添加多个允许的根目录
    pub fn add_allowed_roots<P: AsRef<Path>>(mut self, paths: &[P]) -> Self {
        for path in paths {
            let path = path
                .as_ref()
                .canonicalize()
                .unwrap_or_else(|_| path.as_ref().to_path_buf());
            self.allowed_roots.push(path);
        }
        self
    }

    /// 验证路径是否在允许的根目录内
    ///
    /// # 参数
    /// - `path`: 要验证的路径
    ///
    /// # 返回
    /// - `Ok(PathBuf)`: 规范化后的绝对路径
    /// - `Err(AppError)`: 如果路径遍历攻击或不在允许的目录内
    pub fn validate_path<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf, AppError> {
        let path = path.as_ref();

        // 检查路径是否包含明显的路径遍历模式
        let path_str = path.to_string_lossy();
        if path_str.contains("..") || path_str.contains("~") {
            return Err(AppError::security_error(format!(
                "Path traversal attempt detected: {}",
                path_str
            )));
        }

        // 规范化路径
        let canonical = path
            .canonicalize()
            .map_err(|e| AppError::security_error(format!("Invalid path: {}", e)))?;

        // 明确限制对 /tmp 目录的访问（安全考虑）
        let canonical_str = canonical.to_string_lossy();
        if canonical_str.starts_with("/tmp") || canonical_str.starts_with("/var/tmp") {
            return Err(AppError::security_error(format!(
                "Access to temporary directories is not allowed: {}",
                canonical.display()
            )));
        }

        // 检查路径是否在允许的根目录内
        if self.allowed_roots.is_empty() {
            return Err(AppError::security_error(
                "No allowed root directories configured for file access".to_string(),
            ));
        }

        let is_allowed = self
            .allowed_roots
            .iter()
            .any(|root| canonical.starts_with(root));

        if !is_allowed {
            return Err(AppError::security_error(format!(
                "Access denied: path '{}' is not within allowed directories. Allowed roots: {:?}",
                canonical.display(),
                self.allowed_roots
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
            )));
        }

        Ok(canonical)
    }

    /// 验证路径是否为文件
    pub fn validate_file<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf, AppError> {
        let canonical = self.validate_path(path)?;

        if !canonical.is_file() {
            return Err(AppError::security_error(format!(
                "Path is not a file: {}",
                canonical.display()
            )));
        }

        Ok(canonical)
    }

    /// 验证路径是否为目录
    pub fn validate_directory<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf, AppError> {
        let canonical = self.validate_path(path)?;

        if !canonical.is_dir() {
            return Err(AppError::security_error(format!(
                "Path is not a directory: {}",
                canonical.display()
            )));
        }

        Ok(canonical)
    }
}

impl Default for PathValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_path_traversal_detection() {
        let validator = PathValidator::new().add_allowed_root("/tmp");

        let result = validator.validate_path("/etc/passwd");
        assert!(result.is_err());

        let result = validator.validate_path("/tmp/../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_allowed_path() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, "test").unwrap();

        let validator = PathValidator::new().add_allowed_root(temp_dir.path());

        let result = validator.validate_file(&test_file);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tilde_expansion_blocked() {
        let validator = PathValidator::new().add_allowed_root("/tmp");

        let result = validator.validate_path("~/secret");
        assert!(result.is_err());
    }
}
