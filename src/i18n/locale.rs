// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Language detection and locale resolution.
//!
//! Priority chain (reference-pattern §2):
//! 1. `VECBOOST_LANG` environment variable
//! 2. `LC_ALL` environment variable
//! 3. `LC_MESSAGES` environment variable
//! 4. `LANG` environment variable
//! 5. `sys-locale` system locale
//! 6. Fallback to `"en"`

/// Detect the user's preferred locale from environment and system settings.
///
/// Priority chain (reference-pattern §2):
/// `VECBOOST_LANG` → `LC_ALL` → `LC_MESSAGES` → `LANG` → `sys-locale` → `"en"`.
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

    // 3. LC_MESSAGES (POSIX 消息类目专用变量；未设 LC_ALL 时优先于 LANG)
    if let Ok(lc) = std::env::var("LC_MESSAGES") {
        let trimmed = lc.trim();
        if !trimmed.is_empty() {
            return normalize_locale(trimmed);
        }
    }

    // 4. LANG
    if let Ok(lang) = std::env::var("LANG") {
        let trimmed = lang.trim();
        if !trimmed.is_empty() {
            return normalize_locale(trimmed);
        }
    }

    // 5. System locale
    if let Some(sys_locale) = sys_locale::get_locale() {
        return normalize_locale(&sys_locale);
    }

    // 6. Fallback
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
    /// 序列化环境变量修改，避免并行测试干扰（cargo test 多线程并行执行）
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    use super::*;

    /// 环境变量临时改写守卫：构造时保存并清空给定变量，析构时恢复原值
    /// （断言失败 / panic 亦恢复，避免测试间环境串扰）。
    struct EnvGuard {
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn clear(keys: &[&'static str]) -> Self {
            let saved: Vec<(&'static str, Option<String>)> =
                keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
            for key in keys {
                unsafe { std::env::remove_var(key) };
            }
            Self { saved }
        }

        fn set(&self, key: &str, value: &str) {
            unsafe { std::env::set_var(key, value) };
        }

        fn remove(&self, key: &str) {
            unsafe { std::env::remove_var(key) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(v) => unsafe { std::env::set_var(key, v) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }

    #[test]
    fn test_normalize_locale() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

    #[test]
    fn test_detect_locale_fallback_to_en() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Clear all locale env vars to test fallback
        let saved_vb = std::env::var("VECBOOST_LANG").ok();
        let saved_lc = std::env::var("LC_ALL").ok();
        let saved_lc_messages = std::env::var("LC_MESSAGES").ok();
        let saved_lang = std::env::var("LANG").ok();
        unsafe {
            std::env::remove_var("VECBOOST_LANG");
            std::env::remove_var("LC_ALL");
            std::env::remove_var("LC_MESSAGES");
            std::env::remove_var("LANG");
        }
        // Without env vars, falls through to sys_locale or "en"
        let locale = detect_locale();
        assert!(!locale.is_empty());
        // Restore
        if let Some(v) = saved_vb {
            unsafe { std::env::set_var("VECBOOST_LANG", v) }
        }
        if let Some(v) = saved_lc {
            unsafe { std::env::set_var("LC_ALL", v) }
        }
        if let Some(v) = saved_lc_messages {
            unsafe { std::env::set_var("LC_MESSAGES", v) }
        }
        if let Some(v) = saved_lang {
            unsafe { std::env::set_var("LANG", v) }
        }
    }

    #[test]
    fn test_detect_locale_lc_all() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved_vb = std::env::var("VECBOOST_LANG").ok();
        let saved_lc = std::env::var("LC_ALL").ok();
        unsafe {
            std::env::remove_var("VECBOOST_LANG");
            std::env::set_var("LC_ALL", "zh_CN.UTF-8");
        }
        assert_eq!(detect_locale(), "zh");
        // Restore
        match saved_vb {
            Some(v) => unsafe { std::env::set_var("VECBOOST_LANG", v) },
            None => unsafe { std::env::remove_var("VECBOOST_LANG") },
        }
        match saved_lc {
            Some(v) => unsafe { std::env::set_var("LC_ALL", v) },
            None => unsafe { std::env::remove_var("LC_ALL") },
        }
    }

    /// 检测链层级完整性（reference-pattern §2）：
    /// `VECBOOST_LANG` → `LC_ALL` → `LC_MESSAGES` → `LANG` → sys-locale → `en`。
    /// 此前 `LC_MESSAGES` 层级缺失，`LC_MESSAGES=zh_CN.UTF-8` 会被 `LANG=en_US.UTF-8` 覆盖。
    #[test]
    fn test_detect_locale_lc_messages() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 四级变量全部清空后逐级注入，逐级断言优先级
        let env = EnvGuard::clear(&["VECBOOST_LANG", "LC_ALL", "LC_MESSAGES", "LANG"]);
        env.set("LANG", "en_US.UTF-8");

        // LC_MESSAGES 生效（修复前落到 LANG=en，返回 "en"）
        env.set("LC_MESSAGES", "zh_CN.UTF-8");
        assert_eq!(detect_locale(), "zh");

        // LC_ALL 优先于 LC_MESSAGES
        env.set("LC_ALL", "en_US.UTF-8");
        assert_eq!(detect_locale(), "en");

        // LC_MESSAGES 优先于 LANG（LC_ALL 清空后由 LC_MESSAGES 接管）
        env.remove("LC_ALL");
        assert_eq!(detect_locale(), "zh");

        // VECBOOST_LANG 优先于 LC_MESSAGES
        env.set("VECBOOST_LANG", "en");
        assert_eq!(detect_locale(), "en");

        // LC_MESSAGES 空值（trim 后）与无效值 → 回退链继续（不 panic，链尾非空）
        env.remove("VECBOOST_LANG");
        env.set("LC_MESSAGES", "  ");
        assert_eq!(detect_locale(), "en");
        env.set("LC_MESSAGES", "zh_TW.UTF-8");
        assert_eq!(detect_locale(), "zh");
    }

    #[test]
    fn test_detect_locale_lang() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved_vb = std::env::var("VECBOOST_LANG").ok();
        let saved_lc = std::env::var("LC_ALL").ok();
        let saved_lc_messages = std::env::var("LC_MESSAGES").ok();
        let saved_lang = std::env::var("LANG").ok();
        unsafe {
            std::env::remove_var("VECBOOST_LANG");
            std::env::remove_var("LC_ALL");
            std::env::remove_var("LC_MESSAGES");
            std::env::set_var("LANG", "en_US.UTF-8");
        }
        assert_eq!(detect_locale(), "en");
        // Restore
        match saved_vb {
            Some(v) => unsafe { std::env::set_var("VECBOOST_LANG", v) },
            None => unsafe { std::env::remove_var("VECBOOST_LANG") },
        }
        match saved_lc {
            Some(v) => unsafe { std::env::set_var("LC_ALL", v) },
            None => unsafe { std::env::remove_var("LC_ALL") },
        }
        match saved_lc_messages {
            Some(v) => unsafe { std::env::set_var("LC_MESSAGES", v) },
            None => unsafe { std::env::remove_var("LC_MESSAGES") },
        }
        match saved_lang {
            Some(v) => unsafe { std::env::set_var("LANG", v) },
            None => unsafe { std::env::remove_var("LANG") },
        }
    }

    #[test]
    fn test_detect_locale_empty_vecboost_lang_falls_through() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var("VECBOOST_LANG").ok();
        unsafe { std::env::set_var("VECBOOST_LANG", "  ") };
        // Empty after trim, should fall through to next check
        let locale = detect_locale();
        assert!(!locale.is_empty());
        match saved {
            Some(v) => unsafe { std::env::set_var("VECBOOST_LANG", v) },
            None => unsafe { std::env::remove_var("VECBOOST_LANG") },
        }
    }

    #[test]
    fn test_normalize_locale_opt() {
        assert_eq!(normalize_locale_opt("zh"), Some("zh".to_string()));
        assert_eq!(normalize_locale_opt("en"), Some("en".to_string()));
        assert_eq!(normalize_locale_opt("fr"), None);
        assert_eq!(normalize_locale_opt("ZH_CN"), Some("zh".to_string()));
    }

    #[test]
    fn test_normalize_locale_opt_with_encoding() {
        assert_eq!(normalize_locale_opt("zh_CN.UTF-8"), Some("zh".to_string()));
        assert_eq!(normalize_locale_opt("en_US.utf8"), Some("en".to_string()));
    }

    #[test]
    fn test_normalize_locale_opt_edge_cases() {
        assert_eq!(normalize_locale_opt(""), None);
        assert_eq!(normalize_locale_opt("ja"), None);
        assert_eq!(normalize_locale_opt("zh-Hant"), Some("zh".to_string()));
        assert_eq!(normalize_locale_opt("en-GB"), Some("en".to_string()));
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_parse_accept_language_edge_cases() {
        // Empty string
        assert_eq!(parse_accept_language(""), None);
        // Only whitespace
        assert_eq!(parse_accept_language("  "), None);
        // Single unsupported language
        assert_eq!(parse_accept_language("fr"), None);
        // Mixed supported and unsupported
        assert_eq!(
            parse_accept_language("fr;q=1.0,zh;q=0.5"),
            Some("zh".to_string())
        );
        // Quality without q= prefix
        assert_eq!(parse_accept_language("en;0.5"), Some("en".to_string()));
    }
}
