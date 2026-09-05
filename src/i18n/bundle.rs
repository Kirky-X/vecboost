//! Fluent bundle loading and message lookup.
//!
//! FTL files are embedded at compile time via `include_str!` and parsed into
//! a thread-safe `HashMap<String, String>` for zero-contention lookups.
//!
//! We use a lightweight FTL parser (not `FluentBundle`) because `FluentBundle`
//! contains `IntlLangMemoizer` (with `RefCell`) which is not `Send + Sync`,
//! making it impossible to store in a `static OnceLock`. Our FTL files use
//! only simple `{ $var }` placeholders, which we handle via string replacement.

use std::collections::HashMap;
use unic_langid::LanguageIdentifier;

/// Thread-safe translation store — pre-parsed FTL messages per locale.
pub(crate) struct I18nBundle {
    /// `locale → (message_key → pattern)`
    messages: HashMap<LanguageIdentifier, HashMap<String, String>>,
}

/// Supported locales — kept in sync with `locales/` directory.
const SUPPORTED_LOCALES: &[&str] = &["en", "zh"];

/// FTL resource files to load per locale.
const FTL_RESOURCES: &[&str] = &["errors.ftl", "messages.ftl"];

impl I18nBundle {
    /// Load and parse all FTL resources for every supported locale.
    ///
    /// # Panics
    /// Panics if any FTL file fails to parse (indicates build corruption).
    pub fn load() -> Self {
        let mut messages: HashMap<LanguageIdentifier, HashMap<String, String>> = HashMap::new();

        for locale_str in SUPPORTED_LOCALES {
            let lang_id: LanguageIdentifier = locale_str
                .parse()
                .unwrap_or_else(|e| panic!("Invalid locale identifier '{locale_str}': {e}"));

            let mut locale_msgs = HashMap::new();

            for ftl_name in FTL_RESOURCES {
                let source = match *locale_str {
                    "en" => match *ftl_name {
                        "errors.ftl" => include_str!("locales/en/errors.ftl"),
                        "messages.ftl" => include_str!("locales/en/messages.ftl"),
                        _ => unreachable!(),
                    },
                    "zh" => match *ftl_name {
                        "errors.ftl" => include_str!("locales/zh/errors.ftl"),
                        "messages.ftl" => include_str!("locales/zh/messages.ftl"),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                parse_ftl_into(source, &mut locale_msgs);
            }

            messages.insert(lang_id, locale_msgs);
        }

        Self { messages }
    }

    /// Look up a message by key, substituting variables.
    ///
    /// Falls back to `en` if the requested locale is not loaded.
    /// Returns the key itself if the message is not found in any locale.
    pub fn get_message(
        &self,
        key: &str,
        locale: &LanguageIdentifier,
        args: &HashMap<String, String>,
    ) -> String {
        let msgs = self
            .messages
            .get(locale)
            .or_else(|| {
                // Fallback: try language-only match (e.g., "zh-CN" → "zh")
                let lang_str = locale.language.as_str();
                let lang_only: LanguageIdentifier = lang_str.parse().ok()?;
                self.messages.get(&lang_only)
            })
            .or_else(|| self.messages.get(&Self::fallback_locale()));

        let Some(msgs) = msgs else {
            return key.to_string();
        };

        let Some(pattern) = msgs.get(key) else {
            return key.to_string();
        };

        substitute_vars(pattern, args)
    }

    /// Return all message keys for a given locale (for testing).
    pub fn keys_for_locale(&self, locale: &LanguageIdentifier) -> Vec<String> {
        self.messages
            .get(locale)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn fallback_locale() -> LanguageIdentifier {
        "en".parse().expect("fallback locale 'en' is valid")
    }
}

/// Parse a simple FTL file into a key→pattern map.
///
/// Handles the subset of FTL we use:
/// - `# comments` (skipped)
/// - `key = value` (stored)
/// - `{ $var }` placeholders (kept as-is for later substitution)
/// - Blank lines (skipped)
/// - Section separators like `# ── ... ──` (skipped as comments)
fn parse_ftl_into(source: &str, out: &mut HashMap<String, String>) {
    for line in source.lines() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Split on first '='
        if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim().to_string();
            let value = value.trim().to_string();
            if !key.is_empty() {
                out.insert(key, value);
            }
        }
    }
}

/// Replace `{ $varname }` placeholders in a pattern with values from `args`.
fn substitute_vars(pattern: &str, args: &HashMap<String, String>) -> String {
    let mut result = pattern.to_string();
    for (key, value) in args {
        // Handle both `{ $key }` and `{ $key}` (with/without trailing space)
        let placeholder_braced = format!("{{ ${key} }}");
        let placeholder_tight = format!("{{ ${key}}}");
        result = result.replace(&placeholder_braced, value);
        result = result.replace(&placeholder_tight, value);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ftl_simple() {
        let ftl = "# comment\nhello = Hello World\nbye = Goodbye\n";
        let mut map = HashMap::new();
        parse_ftl_into(ftl, &mut map);
        assert_eq!(map.get("hello").unwrap(), "Hello World");
        assert_eq!(map.get("bye").unwrap(), "Goodbye");
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_parse_ftl_with_vars() {
        let ftl = "greeting = Hello { $name }, welcome!\n";
        let mut map = HashMap::new();
        parse_ftl_into(ftl, &mut map);
        assert_eq!(
            map.get("greeting").unwrap(),
            "Hello { $name }, welcome!"
        );
    }

    #[test]
    fn test_substitute_vars() {
        let pattern = "Error: { $detail }";
        let mut args = HashMap::new();
        args.insert("detail".to_string(), "bad port".to_string());
        assert_eq!(substitute_vars(pattern, &args), "Error: bad port");
    }

    #[test]
    fn test_substitute_vars_multiple() {
        let pattern = "Text { $index } exceeds { $max } (got { $got })";
        let mut args = HashMap::new();
        args.insert("index".to_string(), "0".to_string());
        args.insert("max".to_string(), "8192".to_string());
        args.insert("got".to_string(), "9000".to_string());
        assert_eq!(
            substitute_vars(pattern, &args),
            "Text 0 exceeds 8192 (got 9000)"
        );
    }

    #[test]
    fn test_load_bundles() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let zh: LanguageIdentifier = "zh".parse().unwrap();

        // Both locales should have messages
        let en_keys = bundle.keys_for_locale(&en);
        let zh_keys = bundle.keys_for_locale(&zh);
        assert!(!en_keys.is_empty(), "en should have messages");
        assert!(!zh_keys.is_empty(), "zh should have messages");

        // Same keys in both locales
        assert_eq!(en_keys.len(), zh_keys.len(), "en and zh should have same number of keys");
    }

    #[test]
    fn test_get_message_en() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let args = HashMap::new();

        let msg = bundle.get_message("health-ok", &en, &args);
        assert_eq!(msg, "OK");
    }

    #[test]
    fn test_get_message_zh() {
        let bundle = I18nBundle::load();
        let zh: LanguageIdentifier = "zh".parse().unwrap();
        let args = HashMap::new();

        let msg = bundle.get_message("health-ok", &zh, &args);
        assert_eq!(msg, "正常");
    }

    #[test]
    fn test_get_message_with_args() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let mut args = HashMap::new();
        args.insert("detail".to_string(), "bad port".to_string());

        let msg = bundle.get_message("error-config", &en, &args);
        assert!(
            msg.contains("bad port"),
            "Expected 'bad port' in '{msg}'"
        );
    }

    #[test]
    fn test_get_message_unknown_key() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let args = HashMap::new();

        let msg = bundle.get_message("nonexistent-key", &en, &args);
        assert_eq!(msg, "nonexistent-key");
    }
}
