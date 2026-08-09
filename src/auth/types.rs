// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use serde::{Deserialize, Serialize};
use crate::error::VecboostError;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(utoipa::ToSchema))]
pub struct AuthResponse {
    pub token: String,
    pub token_type: String,
    pub expires_in: u64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    pub role: String,
    pub permissions: Vec<String>,
}



/// 验证用户名格式：
/// - 长度 3-32 字符
/// - 必须以字母开头
/// - 仅允许字母、数字、下划线和连字符
pub fn validate_username_format(username: &str) -> Result<(), VecboostError> {
    if username.len() < 3 || username.len() > 32 {
        return Err(VecboostError::ValidationError(
            "用户名长度必须在 3 到 32 个字符之间".to_string(),
        ));
    }

    if !username
        .chars()
        .next()
        .map(|c| c.is_ascii_alphabetic())
        .unwrap_or(false)
    {
        return Err(VecboostError::ValidationError(
            "用户名必须以字母开头".to_string(),
        ));
    }

    if !username
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(VecboostError::ValidationError(
            "用户名只能包含字母、数字、下划线和连字符".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;



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
    fn test_refresh_token_request_serialization() {
        let req = RefreshTokenRequest {
            refresh_token: "refresh_tok".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: RefreshTokenRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.refresh_token, "refresh_tok");
    }


}
