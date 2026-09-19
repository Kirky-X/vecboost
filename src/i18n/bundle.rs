// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Fluent bundle loading and message lookup.
//!
//! FTL files are embedded at compile time via `include_str!` and parsed by the
//! real Fluent parser into [`fluent_bundle::concurrent::FluentBundle`]
//! instances. The concurrent variant is `Send + Sync`, so both locale bundles
//! live in `static OnceLock`s (dbnexus catalog.rs pattern) — no self-made
//! parser, full FTL grammar support (selectors, functions, escaped literals).

use std::collections::HashMap;
use std::sync::OnceLock;

use fluent_bundle::concurrent::FluentBundle;
use fluent_bundle::{FluentArgs, FluentResource, FluentValue};
use unic_langid::LanguageIdentifier;

/// Supported locales — kept in sync with `locales/` directory.
pub(crate) const SUPPORTED_LOCALES: &[&str] = &["en", "zh"];

/// Embedded FTL sources per locale (errors + messages concatenated).
const EN_FTL: &str = concat!(
    include_str!("locales/en/errors.ftl"),
    "\n",
    include_str!("locales/en/messages.ftl"),
);
const ZH_FTL: &str = concat!(
    include_str!("locales/zh/errors.ftl"),
    "\n",
    include_str!("locales/zh/messages.ftl"),
);

/// Cached concurrent Fluent bundles (thread-safe, built once on first access).
static EN_BUNDLE: OnceLock<FluentBundle<FluentResource>> = OnceLock::new();
static ZH_BUNDLE: OnceLock<FluentBundle<FluentResource>> = OnceLock::new();

/// Handle to the process-wide Fluent bundles.
///
/// The bundles themselves live in the static `OnceLock`s above; this
/// zero-sized handle preserves the `I18nState` layout used by `super::init`.
#[derive(Clone, Copy)]
pub(crate) struct I18nBundle;

impl I18nBundle {
    /// Eagerly build one bundle per supported locale (fail-fast on conflicts).
    pub fn load() -> Self {
        for locale in SUPPORTED_LOCALES {
            let _ = match *locale {
                "zh" => ZH_BUNDLE.get_or_init(build_zh_bundle),
                _ => EN_BUNDLE.get_or_init(build_en_bundle),
            };
        }
        Self
    }

    /// Look up a message by key, formatting `{ $var }` placeables via Fluent.
    ///
    /// Resolution: `locale` (by language subtag) → `en` fallback → the key
    /// itself. Never panics.
    pub fn get_message(
        &self,
        key: &str,
        locale: &LanguageIdentifier,
        args: &HashMap<String, String>,
    ) -> String {
        let lang = locale.language.as_str();
        format_from_bundle(lang, key, args)
            .or_else(|| format_from_bundle(Self::fallback_locale().language.as_str(), key, args))
            .unwrap_or_else(|| key.to_string())
    }

    fn fallback_locale() -> LanguageIdentifier {
        "en".parse().expect("fallback locale 'en' is valid")
    }
}

/// Format a message from the Fluent catalog for the given language.
///
/// Unknown languages fall back to the EN bundle; missing keys return `None`.
fn format_from_bundle(lang: &str, key: &str, args: &HashMap<String, String>) -> Option<String> {
    let bundle = match lang {
        "zh" => ZH_BUNDLE.get_or_init(build_zh_bundle),
        _ => EN_BUNDLE.get_or_init(build_en_bundle),
    };

    let msg = bundle.get_message(key)?;
    let pattern = msg.value()?;

    let mut fluent_args = FluentArgs::new();
    for (name, value) in args {
        fluent_args.set(name.as_str(), FluentValue::from(value.clone()));
    }

    let mut errors = vec![];
    let result = bundle.format_pattern(pattern, Some(&fluent_args), &mut errors);
    Some(result.to_string())
}

fn build_en_bundle() -> FluentBundle<FluentResource> {
    let resource = FluentResource::try_new(EN_FTL.to_string()).unwrap_or_else(|e| e.0);
    let langid: LanguageIdentifier = "en".parse().expect("'en' is a valid language identifier");
    let mut bundle = FluentBundle::new_concurrent(vec![langid]);
    bundle.set_use_isolating(false);
    bundle
        .add_resource(resource)
        .expect("EN resources should add without conflict");
    bundle
}

fn build_zh_bundle() -> FluentBundle<FluentResource> {
    let resource = FluentResource::try_new(ZH_FTL.to_string()).unwrap_or_else(|e| e.0);
    let langid: LanguageIdentifier = "zh".parse().expect("'zh' is a valid language identifier");
    let mut bundle = FluentBundle::new_concurrent(vec![langid]);
    bundle.set_use_isolating(false);
    bundle
        .add_resource(resource)
        .expect("ZH resources should add without conflict");
    bundle
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extract message keys from an embedded FTL source
    /// (top-level `key = value` lines only; comments/blanks skipped).
    fn extract_keys(ftl: &str) -> std::collections::BTreeSet<String> {
        let mut keys = std::collections::BTreeSet::new();
        for line in ftl.lines() {
            let trimmed = line.trim_start();
            // Skip comments, blank lines and indented continuation lines
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed != line {
                continue;
            }
            if let Some((key, _)) = trimmed.split_once('=') {
                let key = key.trim();
                if !key.is_empty() {
                    keys.insert(key.to_string());
                }
            }
        }
        keys
    }

    /// 守卫(键齐性):EN 与 ZH 内嵌 FTL 的 key 集合必须完全相等
    /// (替代旧 I18nBundle 键数测试,原 bundle.rs:205-222)。
    #[test]
    fn test_key_parity_en_zh() {
        let en_keys = extract_keys(EN_FTL);
        let zh_keys = extract_keys(ZH_FTL);
        assert!(!en_keys.is_empty(), "en FTL should contain messages");
        let missing_in_zh: Vec<_> = en_keys.difference(&zh_keys).collect();
        let missing_in_en: Vec<_> = zh_keys.difference(&en_keys).collect();
        assert!(
            missing_in_zh.is_empty(),
            "keys missing in zh FTL: {missing_in_zh:?}"
        );
        assert!(
            missing_in_en.is_empty(),
            "keys missing in en FTL: {missing_in_en:?}"
        );
    }

    /// 守卫(可解析):每个 key 在 en/zh 双束都能解析出译文且不退化为裸键。
    /// Fluent 解析失败的行不会进入 bundle,此处可捕获 FTL 语法回归。
    #[test]
    fn test_every_key_resolves_in_both_bundles() {
        let no_args = HashMap::new();
        for key in extract_keys(EN_FTL) {
            let en = format_from_bundle("en", &key, &no_args);
            let zh = format_from_bundle("zh", &key, &no_args);
            assert!(en.is_some(), "key '{key}' missing in en bundle");
            assert!(zh.is_some(), "key '{key}' missing in zh bundle");
            assert_ne!(en, Some(key.clone()), "en value degenerated to key");
            assert_ne!(zh, Some(key.clone()), "zh value degenerated to key");
        }
    }

    /// 守卫(回退):未知语言取 en 束;缺失 key 返回 None(不 panic)。
    #[test]
    fn test_unknown_language_falls_back_to_en_bundle() {
        let no_args = HashMap::new();
        assert_eq!(
            format_from_bundle("ar", "health-ok", &no_args),
            Some("OK".to_string())
        );
        assert_eq!(format_from_bundle("en", "nonexistent-key", &no_args), None);
    }

    /// 守卫(缺 key 不 panic):get_message 对任意 locale/缺失 key 都返回字符串。
    #[test]
    fn test_get_message_missing_key_returns_key_without_panic() {
        let bundle = I18nBundle::load();
        let fr: LanguageIdentifier = "fr".parse().unwrap();
        let args = HashMap::new();
        assert_eq!(
            bundle.get_message("nonexistent-key", &fr, &args),
            "nonexistent-key"
        );
    }

    /// 守卫(回退终结于 en):I18nBundle::get_message 对不支持 locale 落 en。
    #[test]
    fn test_get_message_unsupported_locale_falls_back_to_en() {
        let bundle = I18nBundle::load();
        let fr: LanguageIdentifier = "fr".parse().unwrap();
        let args = HashMap::new();
        assert_eq!(bundle.get_message("health-ok", &fr, &args), "OK");
    }

    #[test]
    fn test_get_message_en() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let args = HashMap::new();
        assert_eq!(bundle.get_message("health-ok", &en, &args), "OK");
    }

    #[test]
    fn test_get_message_zh() {
        let bundle = I18nBundle::load();
        let zh: LanguageIdentifier = "zh".parse().unwrap();
        let args = HashMap::new();
        assert_eq!(bundle.get_message("health-ok", &zh, &args), "正常");
    }

    #[test]
    fn test_get_message_zh_cn_language_subtag() {
        // "zh-CN" 等 region 变体按 language 子标签命中 zh 束
        let bundle = I18nBundle::load();
        let zh_cn: LanguageIdentifier = "zh-CN".parse().unwrap();
        let args = HashMap::new();
        assert_eq!(bundle.get_message("health-ok", &zh_cn, &args), "正常");
    }

    #[test]
    fn test_get_message_with_args() {
        let bundle = I18nBundle::load();
        let en: LanguageIdentifier = "en".parse().unwrap();
        let mut args = HashMap::new();
        args.insert("detail".to_string(), "bad port".to_string());
        let msg = bundle.get_message("error-config", &en, &args);
        assert!(msg.contains("bad port"), "Expected 'bad port' in '{msg}'");
    }

    #[test]
    fn test_format_from_bundle_multiple_vars() {
        let mut args = HashMap::new();
        args.insert("index".to_string(), "0".to_string());
        args.insert("max".to_string(), "8192".to_string());
        args.insert("got".to_string(), "9000".to_string());
        let msg =
            format_from_bundle("en", "validate-text-length", &args).expect("key in en bundle");
        assert_eq!(msg, "Text at index 0 exceeds max length 8192 (got 9000)");
    }

    #[test]
    fn test_fallback_locale_is_en() {
        assert_eq!(I18nBundle::fallback_locale().language.as_str(), "en");
    }

    #[test]
    fn test_supported_locales_en_zh_only() {
        assert_eq!(SUPPORTED_LOCALES, &["en", "zh"]);
    }
}
