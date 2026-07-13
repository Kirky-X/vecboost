// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuthResponse {
    pub token: String,
    pub token_type: String,
    pub expires_in: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AuthRequest {
    pub username: String,
    pub password: String,
}

// 细粒度权限枚举
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq, Hash)]
pub enum Permission {
    // 嵌入相关权限
    #[serde(rename = "embedding:read")]
    EmbedRead,
    #[serde(rename = "embedding:write")]
    EmbedWrite,

    // 模型相关权限
    #[serde(rename = "model:read")]
    ModelRead,
    #[serde(rename = "model:write")]
    ModelWrite,
    #[serde(rename = "model:switch")]
    ModelSwitch,

    // 用户相关权限
    #[serde(rename = "user:read")]
    UserRead,
    #[serde(rename = "user:write")]
    UserWrite,
    #[serde(rename = "user:delete")]
    UserDelete,

    // 管理员权限
    #[serde(rename = "admin")]
    Admin,
}

impl Permission {
    pub fn as_str(&self) -> &'static str {
        match self {
            Permission::EmbedRead => "embedding:read",
            Permission::EmbedWrite => "embedding:write",
            Permission::ModelRead => "model:read",
            Permission::ModelWrite => "model:write",
            Permission::ModelSwitch => "model:switch",
            Permission::UserRead => "user:read",
            Permission::UserWrite => "user:write",
            Permission::UserDelete => "user:delete",
            Permission::Admin => "admin",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "embedding:read" => Some(Permission::EmbedRead),
            "embedding:write" => Some(Permission::EmbedWrite),
            "model:read" => Some(Permission::ModelRead),
            "model:write" => Some(Permission::ModelWrite),
            "model:switch" => Some(Permission::ModelSwitch),
            "user:read" => Some(Permission::UserRead),
            "user:write" => Some(Permission::UserWrite),
            "user:delete" => Some(Permission::UserDelete),
            "admin" => Some(Permission::Admin),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    pub role: String,
    pub permissions: Vec<String>,
}

impl User {
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string()) || self.role == "admin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_as_str() {
        assert_eq!(Permission::EmbedRead.as_str(), "embedding:read");
        assert_eq!(Permission::EmbedWrite.as_str(), "embedding:write");
        assert_eq!(Permission::ModelRead.as_str(), "model:read");
        assert_eq!(Permission::ModelWrite.as_str(), "model:write");
        assert_eq!(Permission::ModelSwitch.as_str(), "model:switch");
        assert_eq!(Permission::UserRead.as_str(), "user:read");
        assert_eq!(Permission::UserWrite.as_str(), "user:write");
        assert_eq!(Permission::UserDelete.as_str(), "user:delete");
        assert_eq!(Permission::Admin.as_str(), "admin");
    }

    #[test]
    fn test_permission_from_str_valid() {
        assert_eq!(
            Permission::from_str("embedding:read"),
            Some(Permission::EmbedRead)
        );
        assert_eq!(
            Permission::from_str("embedding:write"),
            Some(Permission::EmbedWrite)
        );
        assert_eq!(
            Permission::from_str("model:read"),
            Some(Permission::ModelRead)
        );
        assert_eq!(
            Permission::from_str("model:write"),
            Some(Permission::ModelWrite)
        );
        assert_eq!(
            Permission::from_str("model:switch"),
            Some(Permission::ModelSwitch)
        );
        assert_eq!(
            Permission::from_str("user:read"),
            Some(Permission::UserRead)
        );
        assert_eq!(
            Permission::from_str("user:write"),
            Some(Permission::UserWrite)
        );
        assert_eq!(
            Permission::from_str("user:delete"),
            Some(Permission::UserDelete)
        );
        assert_eq!(Permission::from_str("admin"), Some(Permission::Admin));
    }

    #[test]
    fn test_permission_from_str_invalid() {
        assert_eq!(Permission::from_str("invalid"), None);
        assert_eq!(Permission::from_str(""), None);
        assert_eq!(Permission::from_str("embedding"), None);
    }

    #[test]
    fn test_permission_roundtrip_serialization() {
        let perm = Permission::EmbedRead;
        let json = serde_json::to_string(&perm).unwrap();
        assert_eq!(json, r#""embedding:read""#);
        let deserialized: Permission = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, perm);
    }

    #[test]
    fn test_permission_equality() {
        assert_eq!(Permission::Admin, Permission::Admin);
        assert_ne!(Permission::Admin, Permission::EmbedRead);
    }

    #[test]
    fn test_login_request_serialization() {
        let req = LoginRequest {
            username: "alice".to_string(),
            password: "secret".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: LoginRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.username, "alice");
        assert_eq!(deserialized.password, "secret");
    }

    #[test]
    fn test_auth_response_serialization() {
        let resp = AuthResponse {
            token: "tok123".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: 3600,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: AuthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.token, "tok123");
        assert_eq!(deserialized.token_type, "Bearer");
        assert_eq!(deserialized.expires_in, 3600);
    }

    #[test]
    fn test_user_has_permission_with_explicit_permission() {
        let user = User {
            username: "alice".to_string(),
            role: "user".to_string(),
            permissions: vec!["embedding:read".to_string()],
        };
        assert!(user.has_permission("embedding:read"));
        assert!(!user.has_permission("embedding:write"));
    }

    #[test]
    fn test_user_has_permission_admin_role() {
        let user = User {
            username: "bob".to_string(),
            role: "admin".to_string(),
            permissions: vec![],
        };
        assert!(user.has_permission("embedding:read"));
        assert!(user.has_permission("any_permission"));
    }

    #[test]
    fn test_user_has_permission_no_match() {
        let user = User {
            username: "charlie".to_string(),
            role: "user".to_string(),
            permissions: vec!["model:read".to_string()],
        };
        assert!(!user.has_permission("embedding:read"));
    }

    #[test]
    fn test_refresh_token_request_serialization() {
        let req = RefreshTokenRequest {
            refresh_token: "refresh_tok".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: RefreshTokenRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.refresh_token, "refresh_tok");
    }

    #[test]
    fn test_auth_request_serialization() {
        let req = AuthRequest {
            username: "user1".to_string(),
            password: "pass1".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: AuthRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.username, "user1");
        assert_eq!(deserialized.password, "pass1");
    }
}
