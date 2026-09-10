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

// ---------------------------------------------------------------------------
// Per-request locale (set by HTTP middleware via Accept-Language header)
// ---------------------------------------------------------------------------

tokio::task_local! {
    /// Per-request locale extracted from `Accept-Language` header.
    /// Set by `i18n_middleware`; read by `tr()` / `IntoResponse`.
    static REQUEST_LOCALE: Option<String>;
}

/// Run a future within a request-locale scope.
///
/// All `tr()` / `tr_with_args()` / `IntoResponse` calls within `f` will use
/// `locale` instead of the global default.
pub async fn with_request_locale<F, R>(locale: Option<String>, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    REQUEST_LOCALE.scope(locale, f).await
}

/// Return the per-request locale, if set by the i18n middleware.
pub fn request_locale() -> Option<String> {
    REQUEST_LOCALE.try_with(|lc| lc.clone()).unwrap_or(None)
}

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

/// Translate a message key.
///
/// Resolution order: per-request locale (from `Accept-Language` middleware) →
/// global default locale (from `VECBOOST_LANG` / system locale).
pub fn tr(key: &str) -> String {
    tr_locale_with_args(key, None, None)
}

/// Translate a message key with arguments.
///
/// Same locale resolution as [`tr`].
pub fn tr_with_args(key: &str, args: HashMap<String, String>) -> String {
    tr_locale_with_args(key, None, Some(args))
}

/// Translate a message key using an explicit locale string (e.g., `"zh"`, `"en"`).
///
/// Bypasses both request-level and global-default locale resolution.
pub fn tr_locale(key: &str, locale_str: Option<&str>) -> String {
    tr_locale_with_args(key, locale_str, None)
}

/// Core translation function — resolves locale and delegates to bundle.
///
/// Locale resolution priority:
/// 1. Explicit `locale_str` parameter (if `Some`)
/// 2. Per-request locale from `REQUEST_LOCALE` task-local (set by HTTP middleware)
/// 3. Global default locale (set at startup by `init()`)
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
        // Explicit override — highest priority
        s.parse().unwrap_or_else(|_| state.default_locale.clone())
    } else if let Some(req_lc) = request_locale() {
        // Per-request locale from Accept-Language middleware
        req_lc
            .parse()
            .unwrap_or_else(|_| state.default_locale.clone())
    } else {
        // Global default
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

// ---------------------------------------------------------------------------
// HTTP middleware — sets per-request locale from Accept-Language header
// ---------------------------------------------------------------------------

/// Axum middleware that extracts `Accept-Language` and sets the per-request
/// locale for all downstream handlers and `IntoResponse` conversions.
///
/// Add to the router via `axum::middleware::from_fn(i18n_middleware)`.
#[cfg(feature = "http")]
pub async fn i18n_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let locale = req
        .headers()
        .get(axum::http::header::ACCEPT_LANGUAGE)
        .and_then(|v| v.to_str().ok())
        .and_then(locale::parse_accept_language);

    with_request_locale(locale, async { next.run(req).await }).await
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

    // ---- Per-request locale tests ----

    #[tokio::test]
    async fn test_request_locale_overrides_default() {
        ensure_init();
        // Without request locale, tr() uses global default
        let default_result = tr("health-ok");

        // With request locale set to "zh", tr() should return Chinese
        let zh_result =
            with_request_locale(Some("zh".to_string()), async { tr("health-ok") }).await;
        assert_eq!(zh_result, "正常");

        // With request locale set to "en", tr() should return English
        let en_result =
            with_request_locale(Some("en".to_string()), async { tr("health-ok") }).await;
        assert_eq!(en_result, "OK");

        // Outside the scope, request locale is gone — back to default
        let after_result = tr("health-ok");
        assert_eq!(default_result, after_result);
    }

    #[tokio::test]
    async fn test_request_locale_with_args() {
        ensure_init();
        let args = tr_args(&[("detail", "端口无效")]);
        let result = with_request_locale(Some("zh".to_string()), async {
            tr_with_args("error-config", args)
        })
        .await;
        assert!(
            result.contains("端口无效"),
            "Expected '端口无效' in result, got: {result}"
        );
        assert!(
            result.contains("配置错误"),
            "Expected Chinese prefix '配置错误' in result, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_request_locale_none_falls_back_to_default() {
        ensure_init();
        // None request locale should fall back to global default
        let result = with_request_locale(None, async { tr("health-ok") }).await;
        let default_result = tr("health-ok");
        assert_eq!(result, default_result);
    }

    #[test]
    fn test_request_locale_outside_scope_returns_none() {
        // Outside any with_request_locale scope, request_locale() returns None
        assert!(request_locale().is_none());
    }

    #[test]
    fn test_current_locale_id_returns_valid_id() {
        ensure_init();
        let id = current_locale_id();
        let s = id.to_string();
        assert!(s == "en" || s == "zh", "unexpected locale id: {}", s);
    }

    #[test]
    fn test_tr_locale_with_invalid_locale_str_falls_back() {
        ensure_init();
        // Invalid locale string should fall back to default
        let result = tr_locale("health-ok", Some("invalid_locale!!!"));
        // Should not panic, should return something
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tr_with_empty_args_map() {
        ensure_init();
        let args = HashMap::new();
        let result = tr_with_args("health-ok", args);
        assert!(!result.is_empty());
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_parse_accept_language_empty_header() {
        assert_eq!(locale::parse_accept_language(""), None);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_parse_accept_language_unsupported_only() {
        assert_eq!(locale::parse_accept_language("fr,de,ja"), None);
    }
}
