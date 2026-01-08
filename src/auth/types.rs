#![allow(unused)]

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
