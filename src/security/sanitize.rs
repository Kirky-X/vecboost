// Copyright (c) 2025 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! Security utilities for sanitizing sensitive data in logs and outputs.

/// Sanitize a string by masking sensitive content.
/// Replaces the middle portion of the string with asterisks, showing only first and last 2 characters.
///
/// # Examples
/// ```
/// use vecboost::security::sanitize::sanitize_secret;
///
/// let secret = "my_super_secret_key_12345";
/// let sanitized = sanitize_secret(&secret);
/// assert_eq!(sanitized, "my******************45");
/// ```
pub fn sanitize_secret(s: &str) -> String {
    if s.len() <= 4 {
        "*".repeat(s.len())
    } else {
        let (first, rest) = s.split_at(2);
        let (middle, last) = rest.split_at(rest.len().saturating_sub(2));
        format!("{}{}{}", first, "*".repeat(middle.len()), last)
    }
}

/// Sanitize a password field - shows only length, not content.
pub fn sanitize_password(s: &str) -> String {
    format!("[{} chars]", s.len())
}

/// Sanitize a JWT secret - shows only prefix and length.
pub fn sanitize_jwt_secret(s: &str) -> String {
    format!("{}... [{} chars]", &s[..8], s.len())
}

/// Sanitize an API key - shows only first 8 characters.
pub fn sanitize_api_key(s: &str) -> String {
    sanitize_secret(s)
}

/// Check if a field name likely contains sensitive data.
pub fn is_sensitive_field(field_name: &str) -> bool {
    const SENSITIVE_PATTERNS: &[&str] = &[
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "auth",
        "private",
        "encryption",
        "api_key",
        "jwt",
        "admin_pass",
    ];

    let lower = field_name.to_lowercase();
    SENSITIVE_PATTERNS.iter().any(|p| lower.contains(*p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_secret_short() {
        assert_eq!(sanitize_secret("abc"), "***");
        assert_eq!(sanitize_secret("ab"), "**");
        assert_eq!(sanitize_secret("a"), "*");
        assert_eq!(sanitize_secret(""), "");
    }

    #[test]
    fn test_sanitize_secret_long() {
        let secret = "my_super_secret_key_12345";
        let sanitized = sanitize_secret(secret);
        // Shows first 2 chars, last 2 chars, masks middle
        assert_eq!(sanitized, "my*********************45");
        assert!(sanitized.starts_with("my"));
        assert!(sanitized.ends_with("45"));
        assert!(sanitized.contains('*'));
    }

    #[test]
    fn test_sanitize_password() {
        assert_eq!(sanitize_password("password123"), "[11 chars]");
    }

    #[test]
    fn test_sanitize_jwt_secret() {
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
        let sanitized = sanitize_jwt_secret(jwt);
        assert!(sanitized.starts_with("eyJhbGci..."));
        assert!(sanitized.contains("[36 chars]"));
    }

    #[test]
    fn test_is_sensitive_field() {
        assert!(is_sensitive_field("jwt_secret"));
        assert!(is_sensitive_field("default_admin_password"));
        assert!(is_sensitive_field("api_key"));
        assert!(is_sensitive_field("encryption_key"));
        assert!(is_sensitive_field("AUTH_TOKEN"));
        assert!(!is_sensitive_field("host"));
        assert!(!is_sensitive_field("port"));
        assert!(!is_sensitive_field("username"));
    }
}
