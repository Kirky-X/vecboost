//! Internationalization (i18n) module — ICU+Fluent based bilingual support.
//!
//! Provides translation functions backed by Project Fluent FTL files.
//! Supports English (`en`) and Chinese (`zh`) locales.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::i18n;
//!
//! // Initialize once at startup (calls detect_locale internally)
//! i18n::init();
//!
//! // Simple translation
//! let msg = i18n::tr("health-ok");
//!
//! // Translation with arguments
//! let args = i18n::tr_args(&[("detail", "bad port")]);
//! let msg = i18n::tr_with_args("error-config", args);
//! ```

mod bundle;
pub mod locale;

pub(crate) use bundle::I18nBundle;

use std::collections::HashMap;
use std::sync::OnceLock;
use unic_langid::LanguageIdentifier;

/// Global i18n state — initialized once at startup.
static I18N: OnceLock<I18nState> = OnceLock::new();

struct I18nState {
    bundle: I18nBundle,
    default_locale: LanguageIdentifier,
}

/// Initialize the i18n system.
///
/// Detects the system locale and loads all FTL translation resources.
/// Safe to call multiple times — subsequent calls are no-ops.
pub fn init() {
    I18N.get_or_init(|| {
        let locale_str = locale::detect_locale();
        let lang_id: LanguageIdentifier = locale_str
            .parse()
            .unwrap_or_else(|_| "en".parse().expect("fallback locale"));

        log::info!("i18n initialized: locale={}", locale_str);

        I18nState {
            bundle: I18nBundle::load(),
            default_locale: lang_id,
        }
    });
}

/// Translate a message key using the current default locale.
pub fn tr(key: &str) -> String {
    tr_locale_with_args(key, None, None)
}

/// Translate a message key with arguments using the current default locale.
pub fn tr_with_args(key: &str, args: HashMap<String, String>) -> String {
    tr_locale_with_args(key, None, Some(args))
}

/// Translate a message key using a specific locale string (e.g., `"zh"`, `"en"`).
pub fn tr_locale(key: &str, locale_str: Option<&str>) -> String {
    tr_locale_with_args(key, locale_str, None)
}

/// Core translation function — resolves locale and delegates to bundle.
fn tr_locale_with_args(
    key: &str,
    locale_str: Option<&str>,
    args: Option<HashMap<String, String>>,
) -> String {
    let state = match I18N.get() {
        Some(s) => s,
        None => {
            log::warn!("i18n::tr called before init(), returning key: {key}");
            return key.to_string();
        }
    };

    let locale = if let Some(s) = locale_str {
        s.parse().unwrap_or_else(|_| state.default_locale.clone())
    } else {
        state.default_locale.clone()
    };

    let empty_args = HashMap::new();
    let args_ref = args.as_ref().unwrap_or(&empty_args);

    state.bundle.get_message(key, &locale, args_ref)
}

/// Return the current default locale string.
pub fn current_locale() -> String {
    I18N.get()
        .map(|s| s.default_locale.to_string())
        .unwrap_or_else(|| "en".to_string())
}

/// Return the current default locale as `LanguageIdentifier`.
pub fn current_locale_id() -> LanguageIdentifier {
    I18N.get()
        .map(|s| s.default_locale.clone())
        .unwrap_or_else(|| "en".parse().expect("fallback locale"))
}

/// Helper to construct translation args from `(key, value)` pairs.
pub fn tr_args(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_init() {
        init();
    }

    #[test]
    fn test_tr_returns_string() {
        ensure_init();
        let result = tr("health-ok");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tr_unknown_key_returns_key() {
        ensure_init();
        let result = tr("nonexistent-key");
        assert_eq!(result, "nonexistent-key");
    }

    #[test]
    fn test_tr_with_args() {
        ensure_init();
        let args = tr_args(&[("detail", "bad port")]);
        let result = tr_with_args("error-config", args);
        assert!(
            result.contains("bad port"),
            "Expected 'bad port' in result, got: {result}"
        );
    }

    #[test]
    fn test_tr_locale_zh() {
        ensure_init();
        let result = tr_locale("health-ok", Some("zh"));
        assert_eq!(result, "正常");
    }

    #[test]
    fn test_tr_locale_en() {
        ensure_init();
        let result = tr_locale("health-ok", Some("en"));
        assert_eq!(result, "OK");
    }

    #[test]
    fn test_tr_args_helper() {
        let args = tr_args(&[("key1", "val1"), ("key2", "val2")]);
        assert_eq!(args.len(), 2);
        assert_eq!(args.get("key1").unwrap(), "val1");
    }

    #[test]
    fn test_current_locale_returns_valid_string() {
        ensure_init();
        let loc = current_locale();
        assert!(!loc.is_empty());
    }
}
