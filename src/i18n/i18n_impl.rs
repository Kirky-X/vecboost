// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! ICU4X-backed locale-aware formatting helpers (numbers / dates / plurals /
//! collation). Mirrors dbnexus `i18n::i18n_impl`: [`I18nFormatter`] eagerly
//! builds a `DecimalFormatter`, `PluralRules` and `Collator` for one BCP-47
//! locale from ICU4X compiled data (no runtime data loading).

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use icu::collator::Collator;
use icu::collator::options::CollatorOptions;
use icu::datetime::DateTimeFormatter;
use icu::datetime::fieldsets::YMD;
use icu::datetime::input::{Date, DateTime, Time};
use icu::decimal::DecimalFormatter;
use icu::decimal::input::Decimal;
use icu::decimal::options::DecimalFormatterOptions;
use icu::locale::Locale;
use icu::plurals::{PluralCategory, PluralRules, PluralRulesOptions};
use writeable::Writeable;

/// Errors returned by [`I18nFormatter`] operations.
#[derive(Debug)]
pub enum I18nError {
    /// BCP-47 locale string could not be parsed.
    InvalidLocale { input: String, reason: String },
    /// Number value could not be formatted (e.g. NaN, Infinity, or parse failure).
    InvalidNumber { input: String, reason: String },
    /// Date component out of range or otherwise invalid.
    DateError(String),
    /// Underlying ICU4X data or formatting failure.
    FormatError(String),
}

impl fmt::Display for I18nError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            I18nError::InvalidLocale { input, reason } => {
                write!(f, "invalid locale '{input}': {reason}")
            }
            I18nError::InvalidNumber { input, reason } => {
                write!(f, "invalid number '{input}': {reason}")
            }
            I18nError::DateError(reason) => write!(f, "date error: {reason}"),
            I18nError::FormatError(reason) => write!(f, "formatting error: {reason}"),
        }
    }
}

impl std::error::Error for I18nError {}

/// Locale-aware formatter backed by ICU4X compiled data.
///
/// Construct with [`I18nFormatter::new`] using a BCP-47 locale tag
/// (e.g. `"en"`, `"zh-CN"`). All formatters are created eagerly so repeated
/// formatting calls are allocation-light.
#[derive(Debug)]
pub struct I18nFormatter {
    locale: Locale,
    decimal_formatter: DecimalFormatter,
    plural_rules: PluralRules,
    collator: icu::collator::CollatorBorrowed<'static>,
}

/// Map a [`PluralCategory`] to its capitalized CLDR name (e.g. `"One"`).
fn plural_category_name(category: PluralCategory) -> &'static str {
    match category {
        PluralCategory::Zero => "Zero",
        PluralCategory::One => "One",
        PluralCategory::Two => "Two",
        PluralCategory::Few => "Few",
        PluralCategory::Many => "Many",
        PluralCategory::Other => "Other",
    }
}

impl I18nFormatter {
    /// Create a new formatter for the given BCP-47 locale tag.
    ///
    /// # Errors
    /// Returns [`I18nError::InvalidLocale`] if the tag cannot be parsed,
    /// or [`I18nError::FormatError`] if ICU4X lacks compiled data for it.
    pub fn new(locale: &str) -> Result<Self, I18nError> {
        let parsed = Locale::from_str(locale).map_err(|e| I18nError::InvalidLocale {
            input: locale.to_string(),
            reason: e.to_string(),
        })?;

        let decimal_formatter =
            DecimalFormatter::try_new(parsed.clone().into(), DecimalFormatterOptions::default())
                .map_err(|e| I18nError::FormatError(e.to_string()))?;

        let plural_rules =
            PluralRules::try_new(parsed.clone().into(), PluralRulesOptions::default())
                .map_err(|e| I18nError::FormatError(e.to_string()))?;

        let collator = Collator::try_new(parsed.clone().into(), CollatorOptions::default())
            .map_err(|e| I18nError::FormatError(e.to_string()))?;

        Ok(Self {
            locale: parsed,
            decimal_formatter,
            plural_rules,
            collator,
        })
    }

    /// Create a formatter for the current global i18n locale (`"en"`/`"zh"`).
    ///
    /// Falls back to `"en"` when `i18n::init` has not run yet.
    pub fn for_current_locale() -> Result<Self, I18nError> {
        Self::new(&super::current_locale())
    }

    /// Format a floating-point number with locale-sensitive grouping and
    /// decimal separators (e.g. `"1,234,567.5"` for en).
    ///
    /// # Errors
    /// Returns [`I18nError::InvalidNumber`] for non-finite values.
    pub fn format_number(&self, value: f64) -> Result<String, I18nError> {
        if !value.is_finite() {
            return Err(I18nError::InvalidNumber {
                input: value.to_string(),
                reason: "value is not finite (NaN or Infinity)".into(),
            });
        }
        let repr = format!("{value}");
        let decimal = Decimal::from_str(&repr).map_err(|e| I18nError::InvalidNumber {
            input: repr,
            reason: e.to_string(),
        })?;
        let formatted = self.decimal_formatter.format(&decimal);
        Ok(formatted.write_to_string().into_owned())
    }

    /// Format a count with locale-sensitive grouping separators
    /// (e.g. `"1,234,567"` for en).
    ///
    /// # Errors
    /// Returns [`I18nError::InvalidNumber`] if the count cannot be formatted.
    pub fn format_count(&self, count: u64) -> Result<String, I18nError> {
        self.format_number(count as f64)
    }

    /// Format an ISO calendar date (year / month / day) with a medium-length
    /// locale-specific pattern (e.g. `"Sep 19, 2026"` for en,
    /// `"2026年9月19日"` for zh).
    ///
    /// # Errors
    /// Returns [`I18nError::DateError`] if any component is out of range.
    pub fn format_date(&self, year: i32, month: u8, day: u8) -> Result<String, I18nError> {
        let date =
            Date::try_new_iso(year, month, day).map_err(|e| I18nError::DateError(e.to_string()))?;
        let time = Time::try_new(0, 0, 0, 0).map_err(|e| I18nError::DateError(e.to_string()))?;
        let datetime = DateTime { date, time };

        let dtf = DateTimeFormatter::try_new(self.locale.clone().into(), YMD::medium())
            .map_err(|e| I18nError::FormatError(e.to_string()))?;
        let formatted = dtf.format(&datetime);
        Ok(formatted.write_to_string().into_owned())
    }

    /// Return the plural category name for `count` in the formatter's locale
    /// (e.g. `"One"` for en count=1, `"Other"` for count=2).
    ///
    /// # Errors
    /// Does not currently fail; returns `Result` for API consistency.
    pub fn plural_category(&self, count: u64) -> Result<String, I18nError> {
        Ok(plural_category_name(self.plural_rules.category_for(count)).to_string())
    }

    /// Compare two strings using locale-sensitive collation rules.
    ///
    /// # Errors
    /// Does not currently fail; returns `Result` for API consistency.
    pub fn compare(&self, a: &str, b: &str) -> Result<Ordering, I18nError> {
        Ok(self.collator.compare(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_locales() {
        assert!(I18nFormatter::new("en").is_ok());
        assert!(I18nFormatter::new("zh").is_ok());
        assert!(I18nFormatter::new("zh-CN").is_ok());
        assert!(I18nFormatter::new("en-US").is_ok());
    }

    #[test]
    fn test_new_invalid_locale() {
        let err = I18nFormatter::new("not-a-valid-locale!!!").unwrap_err();
        assert!(matches!(err, I18nError::InvalidLocale { .. }));
        assert!(err.to_string().contains("invalid locale"));
    }

    #[test]
    fn test_format_number_en_grouping() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert_eq!(fmt.format_number(1234567.0).unwrap(), "1,234,567");
        assert_eq!(fmt.format_number(42.5).unwrap(), "42.5");
    }

    #[test]
    fn test_format_number_zh() {
        let fmt = I18nFormatter::new("zh").unwrap();
        let result = fmt.format_number(1234567.0).unwrap();
        assert!(!result.is_empty(), "zh grouping: {result}");
        assert!(result.contains('1'), "zh grouping contains digits: {result}");
    }

    #[test]
    fn test_format_number_non_finite_is_error() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert!(fmt.format_number(f64::NAN).is_err());
        assert!(fmt.format_number(f64::INFINITY).is_err());
    }

    #[test]
    fn test_format_count() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert_eq!(fmt.format_count(1_234_567).unwrap(), "1,234,567");
    }

    #[test]
    fn test_format_date_en() {
        let fmt = I18nFormatter::new("en").unwrap();
        let result = fmt.format_date(2026, 9, 19).unwrap();
        assert!(result.contains("2026"), "en medium date: {result}");
        assert!(result.contains("19"), "en medium date: {result}");
    }

    #[test]
    fn test_format_date_zh() {
        let fmt = I18nFormatter::new("zh").unwrap();
        let result = fmt.format_date(2026, 9, 19).unwrap();
        assert!(
            result.contains("2026") && result.contains("9") && result.contains("19"),
            "zh medium date: {result}"
        );
    }

    #[test]
    fn test_format_date_invalid_components() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert!(fmt.format_date(2026, 13, 1).is_err(), "month 13 invalid");
        assert!(fmt.format_date(2026, 1, 32).is_err(), "day 32 invalid");
    }

    #[test]
    fn test_plural_category_en() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert_eq!(fmt.plural_category(1).unwrap(), "One");
        assert_eq!(fmt.plural_category(2).unwrap(), "Other");
        assert_eq!(fmt.plural_category(0).unwrap(), "Other");
    }

    #[test]
    fn test_plural_category_zh_is_other() {
        // 中文无单复数区分,任何数字都是 Other
        let fmt = I18nFormatter::new("zh").unwrap();
        assert_eq!(fmt.plural_category(1).unwrap(), "Other");
        assert_eq!(fmt.plural_category(100).unwrap(), "Other");
    }

    #[test]
    fn test_compare_collation() {
        let fmt = I18nFormatter::new("en").unwrap();
        assert_eq!(fmt.compare("apple", "banana").unwrap(), Ordering::Less);
        assert_eq!(fmt.compare("b", "a").unwrap(), Ordering::Greater);
        assert_eq!(fmt.compare("same", "same").unwrap(), Ordering::Equal);
    }
}
