//! Language detection and locale resolution.
//!
//! Priority chain:
//! 1. `VECBOOST_LANG` environment variable
//! 2. `LC_ALL` environment variable
//! 3. `LANG` environment variable
//! 4. `sys-locale` system locale
//! 5. Fallback to `"en"`

/// Detect the user's preferred locale from environment and system settings.
///
/// Returns a normalized locale string: `"en"` or `"zh"`.
pub fn detect_locale() -> String {
    // 1. Explicit override
    if let Ok(lang) = std::env::var("VECBOOST_LANG") {
        let trimmed = lang.trim();
        if !trimmed.is_empty() {
            return normalize_locale(trimmed);
        }
    }

    // 2. LC_ALL
    if let Ok(lc) = std::env::var("LC_ALL") {
        let trimmed = lc.trim();
        if !trimmed.is_empty() {
            return normalize_locale(trimmed);
        }
    }

    // 3. LANG
    if let Ok(lang) = std::env::var("LANG") {
        let trimmed = lang.trim();
        if !trimmed.is_empty() {
            return normalize_locale(trimmed);
        }
    }

    // 4. System locale
    if let Some(sys_locale) = sys_locale::get_locale() {
        return normalize_locale(&sys_locale);
    }

    // 5. Fallback
    "en".to_string()
}

/// Normalize a raw locale string to one of the supported locales.
///
/// - `zh`, `zh-CN`, `zh-TW`, `zh-Hans`, `zh-Hant` → `Some("zh")`
/// - `en`, `en-US`, `en-GB` → `Some("en")`
/// - Unsupported → `None`
pub fn normalize_locale_opt(raw: &str) -> Option<String> {
    let lower = raw.to_lowercase();
    let without_encoding = lower.split('.').next().unwrap_or(&lower);
    let normalized = without_encoding.replace('_', "-");

    if normalized.starts_with("zh") {
        Some("zh".to_string())
    } else if normalized.starts_with("en") {
        Some("en".to_string())
    } else {
        None
    }
}

/// Normalize a raw locale string, falling back to `"en"` for unsupported locales.
pub fn normalize_locale(raw: &str) -> String {
    normalize_locale_opt(raw).unwrap_or_else(|| "en".to_string())
}

/// Parse an HTTP `Accept-Language` header value and return the best matching
/// supported locale.
///
/// Returns `None` if no supported language is found (caller should use global locale).
#[cfg(feature = "http")]
pub fn parse_accept_language(header_value: &str) -> Option<String> {
    let mut candidates: Vec<(f32, String)> = Vec::new();

    for part in header_value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Parse "lang;q=0.9" format
        let (lang, quality) = if let Some((lang_part, q_part)) = part.split_once(';') {
            let q_str = q_part.trim();
            let q = if let Some(q_val) = q_str.strip_prefix("q=") {
                q_val.parse::<f32>().unwrap_or(1.0)
            } else {
                1.0
            };
            (lang_part.trim(), q)
        } else {
            (part, 1.0)
        };

        if let Some(normalized) = normalize_locale_opt(lang) {
            candidates.push((quality, normalized));
        }
    }

    // Sort by quality descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Return first supported locale
    let supported = ["en", "zh"];
    for (_, locale) in &candidates {
        if supported.contains(&locale.as_str()) {
            return Some(locale.clone());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_locale() {
        assert_eq!(normalize_locale("zh"), "zh");
        assert_eq!(normalize_locale("zh-CN"), "zh");
        assert_eq!(normalize_locale("zh-TW"), "zh");
        assert_eq!(normalize_locale("zh-Hans"), "zh");
        assert_eq!(normalize_locale("ZH-CN"), "zh");
        assert_eq!(normalize_locale("zh_CN.UTF-8"), "zh");
        assert_eq!(normalize_locale("en"), "en");
        assert_eq!(normalize_locale("en-US"), "en");
        assert_eq!(normalize_locale("EN"), "en");
        assert_eq!(normalize_locale("fr"), "en"); // unsupported → fallback
        assert_eq!(normalize_locale("ja"), "en"); // unsupported → fallback
    }

    #[test]
    fn test_detect_locale_vecboost_lang() {
        // Save and clear env vars to test in isolation
        let saved = std::env::var("VECBOOST_LANG").ok();
        // SAFETY: test-only, no concurrent env access in same process
        unsafe { std::env::set_var("VECBOOST_LANG", "zh") };
        assert_eq!(detect_locale(), "zh");

        unsafe { std::env::set_var("VECBOOST_LANG", "zh-CN") };
        assert_eq!(detect_locale(), "zh");

        // Restore
        match saved {
            Some(v) => unsafe { std::env::set_var("VECBOOST_LANG", v) },
            None => unsafe { std::env::remove_var("VECBOOST_LANG") },
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_parse_accept_language() {
        assert_eq!(
            parse_accept_language("zh-CN,zh;q=0.9,en;q=0.8"),
            Some("zh".to_string())
        );
        assert_eq!(
            parse_accept_language("en-US,en;q=0.9"),
            Some("en".to_string())
        );
        assert_eq!(
            parse_accept_language("fr;q=1.0,de;q=0.9"),
            None // no supported language
        );
        assert_eq!(
            parse_accept_language("zh-TW;q=0.5,en;q=0.8"),
            Some("en".to_string()) // en has higher quality
        );
    }
}
