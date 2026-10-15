// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! whatlang 语言检测核心——`/api/1/detect` 端点与翻译 auto 检测共用的唯一事实源。
//! 对外语言码契约：ISO 639-1 优先，无 639-1 等价者回退 639-3 码（非 BCP-47，全小写）。

use whatlang::{Detector, Lang};

/// 检测结果。`language` 为空串表示不可判定（域层转 `language: null` 语义）。
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub language: String,
    pub script: String,
    pub confidence: f64,
    pub is_reliable: bool,
}

/// whatlang Lang（ISO 639-3）→ 对外发布码（639-1 优先，无等价者回退 639-3）。
/// 穷举 whatlang 0.18 全部 70 变体，覆盖断言见 tests::mapping_table_is_exhaustive。
const LANG_PUBLISH_CODES: &[(Lang, &str)] = &[
    (Lang::Epo, "epo"),
    (Lang::Eng, "en"),
    (Lang::Rus, "ru"),
    (Lang::Cmn, "zh"),
    (Lang::Spa, "es"),
    (Lang::Por, "pt"),
    (Lang::Ita, "it"),
    (Lang::Ben, "bn"),
    (Lang::Fra, "fr"),
    (Lang::Deu, "de"),
    (Lang::Ukr, "uk"),
    (Lang::Kat, "ka"),
    (Lang::Ara, "ar"),
    (Lang::Hin, "hi"),
    (Lang::Jpn, "ja"),
    (Lang::Heb, "he"),
    (Lang::Yid, "yi"),
    (Lang::Pol, "pl"),
    (Lang::Amh, "am"),
    (Lang::Jav, "jv"),
    (Lang::Kor, "ko"),
    (Lang::Nob, "nb"),
    (Lang::Dan, "da"),
    (Lang::Swe, "sv"),
    (Lang::Fin, "fi"),
    (Lang::Tur, "tr"),
    (Lang::Nld, "nl"),
    (Lang::Hun, "hu"),
    (Lang::Ces, "cs"),
    (Lang::Ell, "el"),
    (Lang::Bul, "bg"),
    (Lang::Bel, "be"),
    (Lang::Mar, "mr"),
    (Lang::Kan, "kn"),
    (Lang::Ron, "ro"),
    (Lang::Slv, "sl"),
    (Lang::Hrv, "hr"),
    (Lang::Srp, "sr"),
    (Lang::Mkd, "mk"),
    (Lang::Lit, "lt"),
    (Lang::Lav, "lv"),
    (Lang::Est, "et"),
    (Lang::Tam, "ta"),
    (Lang::Vie, "vi"),
    (Lang::Urd, "ur"),
    (Lang::Tha, "th"),
    (Lang::Guj, "gu"),
    (Lang::Uzb, "uz"),
    (Lang::Pan, "pa"),
    (Lang::Aze, "az"),
    (Lang::Ind, "id"),
    (Lang::Tel, "te"),
    (Lang::Pes, "fa"),
    (Lang::Mal, "ml"),
    (Lang::Ori, "or"),
    (Lang::Mya, "my"),
    (Lang::Nep, "ne"),
    (Lang::Sin, "si"),
    (Lang::Khm, "km"),
    (Lang::Tuk, "tk"),
    (Lang::Aka, "ak"),
    (Lang::Zul, "zu"),
    (Lang::Sna, "sn"),
    (Lang::Afr, "af"),
    (Lang::Lat, "la"),
    (Lang::Slk, "sk"),
    (Lang::Cat, "ca"),
    (Lang::Tgl, "tl"),
    (Lang::Hye, "hy"),
    (Lang::Cym, "cy"),
];

fn published_code(lang: Lang) -> &'static str {
    LANG_PUBLISH_CODES
        .iter()
        .copied()
        .find(|(candidate, _)| *candidate == lang)
        .map(|(_, code)| code)
        .unwrap_or_else(|| lang.code())
}

/// 发布码集合成员谓词（与 [`LANG_PUBLISH_CODES`] 同源单一实现）：
/// 格式合法（`^[a-z]{2,3}$`）不等于已发布——"sw"/"yo" 等真实 ISO 639-1 码
/// 不在 whatlang 70 变体内，调用方（域/服务/配置校验）必须先经本谓词
/// 拒绝，防止穿透到检测核心。
pub fn is_published_language_code(code: &str) -> bool {
    LANG_PUBLISH_CODES
        .iter()
        .any(|(_, published)| *published == code)
}

/// allowlist 码 → whatlang Lang。未发布码返回 `None`——上游域/服务层已按
/// [`is_published_language_code`] 校验并以 400 显性拒绝，此处 `None` 仅为
/// 不可达时的防御兜底（降级为不可判定，禁 panic：async handler 内 panic
/// 即连接重置）。
fn resolve_published_code(code: &str) -> Option<Lang> {
    LANG_PUBLISH_CODES
        .iter()
        .copied()
        .find(|(_, published)| *published == code)
        .map(|(lang, _)| lang)
}

/// 检测文本语言。`allow` 非空时经 allowlist 收窄候选集，空表与 `None` 等价（不收窄）。
/// 不可判定（空文本/纯数字/纯符号）返回空 `language` + `is_reliable: false`，不虚构语言。
pub fn detect_language(text: &str, allow: Option<&[String]>) -> Detection {
    let info = match allow {
        Some(codes) if !codes.is_empty() => {
            let resolved: Option<Vec<Lang>> = codes
                .iter()
                .map(|code| resolve_published_code(code))
                .collect();
            match resolved {
                Some(allowlist) => Detector::with_allowlist(allowlist).detect(text),
                // 防御兜底：上游校验已拦，理论不可达（评审）
                None => None,
            }
        }
        _ => whatlang::detect(text),
    };
    match info {
        Some(info) => Detection {
            language: published_code(info.lang()).to_string(),
            script: info.script().name().to_string(),
            confidence: info.confidence(),
            is_reliable: info.is_reliable(),
        },
        None => Detection {
            language: String::new(),
            script: String::new(),
            confidence: 0.0,
            is_reliable: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZH: &str = "今天天气非常好，我们一起去公园散步，欣赏美丽的风景。";
    const EN: &str =
        "The quick brown fox jumps over the lazy dog near the riverbank every morning.";
    const JA: &str = "今日はとても良い天気なので、公園を散歩して美しい景色を楽しみましょう。";
    const FR: &str = "Le chat noir dort paisiblement sur le canapé du salon depuis ce matin.";
    const DE: &str = "Die wichtigsten Ereignisse des Tages wurden ausführlich in der Zeitung beschrieben, damit sich die Leser ein umfassendes Bild machen konnten.";
    const RU: &str = "Быстрая бурая лиса прыгает через ленивую собаку возле реки каждое утро.";
    const AR: &str = "القطة السوداء تنام بهدوء على الأريكة في غرفة المعيشة منذ الصباح.";
    const ES: &str = "El veloz murciélago hindú comía feliz cardillo y kiwi bajo la palmera.";

    #[test]
    fn seven_languages_detect_to_639_1_codes() {
        for (text, expected) in [
            (ZH, "zh"),
            (EN, "en"),
            (JA, "ja"),
            (FR, "fr"),
            (DE, "de"),
            (RU, "ru"),
            (AR, "ar"),
        ] {
            let detection = detect_language(text, None);
            assert_eq!(detection.language, expected, "text: {text}");
            assert!(
                (0.0..=1.0).contains(&detection.confidence),
                "confidence out of [0,1] for {expected}: {}",
                detection.confidence
            );
            assert!(
                detection.is_reliable,
                "expected reliable detection for {expected}"
            );
        }
    }

    #[test]
    fn scripts_match_detected_language() {
        assert_eq!(detect_language(ZH, None).script, "Mandarin");
        assert_eq!(detect_language(EN, None).script, "Latin");
        assert_eq!(detect_language(JA, None).script, "Hiragana");
        assert_eq!(detect_language(RU, None).script, "Cyrillic");
        assert_eq!(detect_language(AR, None).script, "Arabic");
    }

    #[test]
    fn mapping_table_is_exhaustive_over_whatlang_variants() {
        assert_eq!(LANG_PUBLISH_CODES.len(), Lang::all().len());
        let mut seen: Vec<Lang> = Vec::new();
        for (lang, published) in LANG_PUBLISH_CODES {
            assert!(!seen.contains(lang), "duplicate mapping for {lang:?}");
            seen.push(*lang);
            assert!(
                published.len() == 2 || published.len() == 3,
                "malformed published code {published:?}"
            );
            assert!(
                published.chars().all(|c| c.is_ascii_lowercase()),
                "published code {published:?} not lowercase"
            );
            if *lang == Lang::Epo {
                assert_eq!(*published, "epo", "Esperanto 无 639-1 等价，应回退 639-3");
            } else {
                assert_eq!(published.len(), 2, "{lang:?} 存在 639-1 等价，应为两位码");
            }
        }
        for lang in Lang::all() {
            assert!(seen.contains(lang), "{lang:?} missing from mapping table");
        }
    }

    #[test]
    fn allowlist_narrows_detection_to_permitted_set() {
        let detection = detect_language(ES, Some(&["zh".to_string(), "en".to_string()]));
        assert!(
            detection.language == "zh" || detection.language == "en",
            "allowlist narrowing leaked: {}",
            detection.language
        );
    }

    #[test]
    fn empty_and_digit_text_report_unreliable_without_language() {
        for text in ["", "1234567890", "!@#$%^&*()"] {
            let detection = detect_language(text, None);
            assert!(
                detection.language.is_empty(),
                "fabricated language for {text:?}: {}",
                detection.language
            );
            assert!(
                !detection.is_reliable,
                "undetectable text reported reliable: {text:?}"
            );
            assert_eq!(
                detection.confidence, 0.0,
                "undetectable text has nonzero confidence"
            );
        }
    }

    #[test]
    fn unknown_allowlist_code_degrades_to_undetectable() {
        // 格式合法但未发布码（"xx"）在域/服务层已被 400 拒绝；
        // 检测核心对穿透值降级为不可判定（防御兜底，禁 panic——async handler
        // 内 panic 即连接重置）
        let detection = detect_language(EN, Some(&["xx".to_string()]));
        assert!(detection.language.is_empty());
        assert!(!detection.is_reliable);
    }

    #[test]
    fn published_code_predicate_rejects_valid_format_unpublished() {
        // 评审实证：sw（斯瓦希里语）/yo（约鲁巴语）为真实 ISO 639-1 码但
        // 不在 whatlang 70 变体发布集——格式谓词放行、发布谓词必须拦截
        assert!(is_published_language_code("en"));
        assert!(is_published_language_code("zh"));
        assert!(
            is_published_language_code("epo"),
            "无 639-1 等价者 639-3 回退码属发布码"
        );
        for code in ["xx", "sw", "yo", "mi", "qq", "xx1", "EN"] {
            assert!(
                !is_published_language_code(code),
                "{code} 不在发布集，谓词必须拒绝"
            );
        }
    }
}
