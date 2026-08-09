// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! Security utilities for sanitizing sensitive data in logs and outputs.

/// Sanitize a string by masking sensitive content.
/// Replaces the middle portion of the string with asterisks, showing only first and last 2 characters.
///
/// Uses `floor_char_boundary`/`ceil_char_boundary` to guarantee UTF-8 safe slicing
/// even when the 2-byte boundary falls inside a multi-byte character (e.g. CJK).
///
/// # Examples
/// ```
/// use vecboost::security::sanitize_secret;
///
/// let secret = "my_super_secret_key_12345";
/// let sanitized = sanitize_secret(&secret);
/// assert_eq!(sanitized, "my*********************45");
/// ```
pub fn sanitize_secret(s: &str) -> String {
    let char_count = s.chars().count();
    if char_count <= 4 {
        "*".repeat(s.len())
    } else {
        // Show first 2 and last 2 characters, mask the middle
        let first_byte_end = s
            .char_indices()
            .nth(2)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let last_byte_start = s
            .char_indices()
            .nth(char_count - 2)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let first = &s[..first_byte_end];
        let last = &s[last_byte_start..];
        let middle_len = last_byte_start - first_byte_end;
        format!("{}{}{}", first, "*".repeat(middle_len), last)
    }
}

/// Sanitize a password field - shows only length, not content.
pub fn sanitize_password(s: &str) -> String {
    format!("[{} chars]", s.len())
}

/// Sanitize a JWT secret - shows only prefix and length.
///
/// Uses `floor_char_boundary` to guarantee UTF-8 safe prefix slicing.
pub fn sanitize_jwt_secret(s: &str) -> String {
    format!("{}... [{} chars]", &s[..s.floor_char_boundary(8)], s.len())
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
    fn test_sanitize_secret_multibyte_no_panic() {
        // CJK: each char is 3 bytes; the 2-byte boundary falls inside the
        // first char, which would panic without floor_char_boundary.
        let secret = "密钥内容不能泄露abcdefgh";
        let sanitized = sanitize_secret(secret);
        assert!(sanitized.contains('*'));
    }

    #[test]
    fn test_sanitize_jwt_secret_multibyte_no_panic() {
        // 8-byte boundary falls inside the 3rd CJK char (each 3 bytes)
        let jwt = "密钥secretjwt1234567890";
        let sanitized = sanitize_jwt_secret(jwt);
        assert!(sanitized.contains(&format!("[{} chars]", jwt.len())));
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
