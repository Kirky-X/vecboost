// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! VecBoostInterface — GarrisonInterface 的 VecBoost 实现。
//!
//! 提供权限/角色数据给 garrison 框架。当前实现基于 admin 用户名判定：
//! - admin 用户：全权限 + "admin" 角色
//! - 普通用户：基础权限 + "user" 角色
//!
//! 后续可对接 garrison dao 的用户/角色存储实现完整 RBAC。

use async_trait::async_trait;
use garrison::error::GarrisonResult;
use garrison::stp::GarrisonInterface;

/// VecBoost 的 GarrisonInterface 实现。
///
/// 基于配置的 admin 用户名判定角色和权限：
/// - `login_id == admin_username` → 全权限 (`["*"]`) + `["admin"]` 角色
/// - 其他 login_id → 基础权限 (`["embedding:read", "embedding:write"]`) + `["user"]` 角色
///
/// 后续可通过注入 `Arc<dyn GarrisonDao>` 查询 garrison 内置用户/角色存储，
/// 实现完整 RBAC 权限模型。
pub struct VecBoostInterface {
    admin_username: String,
}

impl VecBoostInterface {
    /// 创建 VecBoostInterface，指定 admin 用户名。
    pub fn new(admin_username: String) -> Self {
        Self { admin_username }
    }
}

#[async_trait]
impl GarrisonInterface for VecBoostInterface {
    async fn get_permission_list(&self, login_id: &str) -> GarrisonResult<Vec<String>> {
        if login_id == self.admin_username {
            // admin 用户拥有全部权限
            Ok(vec!["*".to_string()])
        } else {
            // 普通用户拥有基础 embedding 权限
            Ok(vec![
                "embedding:read".to_string(),
                "embedding:write".to_string(),
            ])
        }
    }

    async fn get_role_list(&self, login_id: &str) -> GarrisonResult<Vec<String>> {
        if login_id == self.admin_username {
            Ok(vec!["admin".to_string()])
        } else {
            Ok(vec!["user".to_string()])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_admin_gets_all_permissions() {
        let interface = VecBoostInterface::new("admin".to_string());
        let perms = interface.get_permission_list("admin").await.unwrap();
        assert_eq!(perms, vec!["*"]);
    }

    #[tokio::test]
    async fn test_admin_gets_admin_role() {
        let interface = VecBoostInterface::new("admin".to_string());
        let roles = interface.get_role_list("admin").await.unwrap();
        assert_eq!(roles, vec!["admin"]);
    }

    #[tokio::test]
    async fn test_normal_user_gets_basic_permissions() {
        let interface = VecBoostInterface::new("admin".to_string());
        let perms = interface.get_permission_list("normal_user").await.unwrap();
        assert_eq!(perms, vec!["embedding:read", "embedding:write"]);
    }

    #[tokio::test]
    async fn test_normal_user_gets_user_role() {
        let interface = VecBoostInterface::new("admin".to_string());
        let roles = interface.get_role_list("normal_user").await.unwrap();
        assert_eq!(roles, vec!["user"]);
    }

    #[tokio::test]
    async fn test_nonexistent_user_gets_basic_permissions() {
        let interface = VecBoostInterface::new("admin".to_string());
        let perms = interface.get_permission_list("nonexistent").await.unwrap();
        assert_eq!(perms, vec!["embedding:read", "embedding:write"]);
    }

    #[tokio::test]
    async fn test_nonexistent_user_gets_user_role() {
        let interface = VecBoostInterface::new("admin".to_string());
        let roles = interface.get_role_list("nonexistent").await.unwrap();
        assert_eq!(roles, vec!["user"]);
    }
}
