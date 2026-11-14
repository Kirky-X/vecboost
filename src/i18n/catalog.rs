// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! FTL 目录（编译期嵌入 → 轻量文本解析）。
//!
//! 4 个 FTL 文件（每 locale 2 个）的 `key = value` 行解析。解析口径沿用
//! garrison `parse_ftl_entries` 的 fluent 语义：`key = value` 起条，缩进
//! 续行与段内空行并入 value，占位符跨行断开时丢弃占位符内部换行，
//! `#` 注释行跳过。本仓目录经核实无多行延续值（纯单行形态），多行语义
//! 仅为防御性兼容——行解析与完整 Fluent 解析行为一致（无 selector/term
//! /内置函数语法，`{"{"}` 类转义亦不存在，占位符统一 `{ $name }` 形态）。

use std::sync::OnceLock;

// en（2 个，与 locales/en/*.ftl 一一对应）
const EN_ERRORS: &str = include_str!("locales/en/errors.ftl");
const EN_MESSAGES: &str = include_str!("locales/en/messages.ftl");

// zh（2 个，与 locales/zh/*.ftl 一一对应）
const ZH_ERRORS: &str = include_str!("locales/zh/errors.ftl");
const ZH_MESSAGES: &str = include_str!("locales/zh/messages.ftl");

const EN_SOURCES: &[&str] = &[EN_ERRORS, EN_MESSAGES];
const ZH_SOURCES: &[&str] = &[ZH_ERRORS, ZH_MESSAGES];

fn sources_for(locale: &str) -> &'static [&'static str] {
    match locale {
        "zh" => ZH_SOURCES,
        _ => EN_SOURCES,
    }
}

/// 归一 FTL 源：占位符规范化——跨行断开的占位符内部换行丢弃，内部空白
/// 折叠为紧凑形态（`{ $name }` / `{$\nname}` → `{$name}`，与 sdforge
/// 引擎的占位符规范化同向）；非占位符花括号（`{"{"}` 转义/字面量）原样还原。
fn normalize_ftl_placeholders(ftl: &str) -> String {
    let mut out = String::with_capacity(ftl.len());
    let mut chars = ftl.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '{' {
            out.push(ch);
            continue;
        }
        // 收集到匹配的 '}'（占位符不嵌套）；占位符内部换行丢弃
        let mut inner = String::new();
        let mut closed = false;
        for c2 in chars.by_ref() {
            if c2 == '}' {
                closed = true;
                break;
            }
            if c2 != '\n' {
                inner.push(c2);
            }
        }
        let trimmed = inner.trim();
        if closed && trimmed.starts_with('$') && trimmed.len() > 1 {
            // 占位符：规范化为紧凑形态
            out.push_str(&format!("{{{trimmed}}}"));
        } else {
            // 非占位符花括号：原样还原（含未闭合的畸形输入）
            out.push('{');
            out.push_str(&inner);
            if closed {
                out.push('}');
            }
        }
    }
    out
}

/// 解析 FTL 文本为 (key, value) 条目（fluent 语义）：`key = value` 起条
/// （首行内容可为空），其后的缩进续行与空行并入 value 直至下一个条目/注释；
/// `#` 注释行跳过。与 locales/*.ftl 真源一一对应（纯文本消息，无
/// selector/term 语法）。
fn parse_ftl_entries(ftl: &str) -> Vec<(String, String)> {
    let ftl = &normalize_ftl_placeholders(ftl);
    let mut out: Vec<(String, String)> = Vec::new();
    let mut cur_key: Option<String> = None;
    let mut cur_lines: Vec<String> = Vec::new();
    for line in ftl.lines() {
        if cur_key.is_some() && (line.starts_with([' ', '\t']) || line.trim().is_empty()) {
            // 当前条目的续行或段内空行
            cur_lines.push(line.to_string());
            continue;
        }
        // 到达新条目/注释/顶层空行：结算当前条目
        flush_ftl_entry(&mut cur_key, &mut cur_lines, &mut out);
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            cur_key = Some(k.trim().to_string());
            cur_lines.push(v.trim().to_string());
        }
    }
    flush_ftl_entry(&mut cur_key, &mut cur_lines, &mut out);
    out
}

/// 结算当前条目：多行 value 按 fluent 规则归一（`key =` 空占位移除；
/// 公共缩进以首个非空续行为基准，仅剥确实以该前缀开头的续行；
/// `=` 行本身与不成前缀的行原样保留；去尾部空行）。
fn flush_ftl_entry(
    key: &mut Option<String>,
    lines: &mut Vec<String>,
    out: &mut Vec<(String, String)>,
) {
    if let Some(key) = key.take() {
        let had_inline = lines.first().is_some_and(|l| !l.is_empty());
        if !had_inline && lines.first().is_some_and(|l| l.is_empty()) {
            lines.remove(0);
        }
        while lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        let cont_start = usize::from(had_inline);
        let common = lines
            .iter()
            .skip(cont_start)
            .filter(|l| !l.is_empty())
            .map(|l| l.len() - l.trim_start().len())
            .min()
            .unwrap_or(0);
        let prefix = " ".repeat(common);
        let value = lines
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let stripped = if i >= cont_start && common > 0 && l.starts_with(prefix.as_str()) {
                    &l[common..]
                } else {
                    l
                };
                stripped.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");
        out.push((key, value));
        lines.clear();
    }
}

/// 指定 locale 的全部目录条目（协议面注册与 `tr` 查找用；重复键后者覆盖）。
pub(crate) fn entries_for(locale: &str) -> Vec<(String, String)> {
    sources_for(locale)
        .iter()
        .flat_map(|source| parse_ftl_entries(source))
        .collect()
}

/// 指定 locale 的键集合（测试守卫与存在性断言用）。
#[cfg(test)]
pub(crate) fn keys_for(locale: &str) -> std::collections::BTreeSet<String> {
    entries_for(locale)
        .into_iter()
        .map(|(key, _)| key)
        .collect()
}

/// 协议面消息键（`forge-*`，取自 en 内嵌目录，保持源顺序）。
///
/// 这些是交给 sdforge 运行时翻译钩子的 i18n 键（`CliBuilder` 构建
/// `--help` 时经 `translate_or_fallback` 解析）。从 FTL 本身派生清单维持
/// 单一事实源：新增 `forge-*` 消息即自动注册，无需第二份列表。
///
/// 仅协议面构建消费（mod.rs 的 sync 变体，同 cfg 门）；无协议面构建下
/// dead_code 豁免——与 ensure_registered/REGISTERED 同口径。
#[cfg_attr(
    not(any(feature = "cli", feature = "mcp", feature = "http", feature = "grpc")),
    allow(dead_code)
)]
pub(crate) fn protocol_message_keys() -> Vec<String> {
    parse_ftl_entries(EN_MESSAGES)
        .into_iter()
        .map(|(key, _)| key)
        .filter(|key| key.starts_with("forge-"))
        .collect()
}

/// 注册幂等守卫：仅协议面构建消费（`ensure_registered` 的 cfg 门体），
/// 无协议面构建下随函数一并 dead_code 豁免。
#[cfg_attr(
    not(any(feature = "cli", feature = "mcp", feature = "http", feature = "grpc")),
    allow(dead_code)
)]
static REGISTERED: OnceLock<()> = OnceLock::new();

/// 把两份 FTL 的条目逐键注册进 sdforge 注册表（协议面消费），单次执行。
///
/// sdforge 为 optional 依赖（仅 cli/grpc/http/mcp feature 链接），注册体必须
/// 同 cfg 门——无条件调用在 no-default-features 构建下 E0433（unresolved
/// crate）；函数本身保留（消费方 `sync_sdforge_i18n` 同门），由 cfg_attr
/// 豁免协议面未启用时的 dead_code。
#[cfg_attr(
    not(any(feature = "cli", feature = "mcp", feature = "http", feature = "grpc")),
    allow(dead_code)
)]
pub(crate) fn ensure_registered() {
    #[cfg(any(feature = "cli", feature = "mcp", feature = "http", feature = "grpc"))]
    REGISTERED.get_or_init(|| {
        for locale in ["en", "zh"] {
            for (key, value) in entries_for(locale) {
                sdforge::i18n::register_translation(locale, &key, &value);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 键齐性守卫（沿用旧 bundle.rs 口径）：两 locale 键集合必须完全一致。
    #[test]
    fn test_key_parity_en_zh() {
        let en = keys_for("en");
        let zh = keys_for("zh");
        assert!(!en.is_empty(), "en 目录不能为空");
        assert_eq!(en.len(), zh.len(), "en/zh 键数不一致");
        let missing_in_zh: Vec<_> = en.difference(&zh).collect();
        let missing_in_en: Vec<_> = zh.difference(&en).collect();
        assert!(
            missing_in_zh.is_empty(),
            "keys missing in zh FTL: {missing_in_zh:?}"
        );
        assert!(
            missing_in_en.is_empty(),
            "keys missing in en FTL: {missing_in_en:?}"
        );
    }

    /// 守卫（协议面键）：`forge-*` 键仅在 sdforge 注册表里生效（不进 `tr`
    /// 调用面），R3 扫描覆盖不到属性键；此处确保派生清单非空且都是
    /// `forge-` 前缀。
    #[test]
    fn test_protocol_message_keys_nonempty_and_prefixed() {
        let keys = protocol_message_keys();
        assert!(
            !keys.is_empty(),
            "en FTL 应至少包含一个 forge-* 协议面键（CLI --help 国际化入口）"
        );
        assert!(
            keys.iter().all(|k| k.starts_with("forge-")),
            "协议面键必须全部以 forge- 开头: {keys:?}"
        );
    }

    /// 单行解析守卫：`key = value` 与注释/空行/缩进行的行为。
    #[test]
    fn test_parse_ftl_entries_single_line_semantics() {
        let src = "# comment\n\nkey-a = value a\n  indented = not a key\nkey-b =\n";
        let entries = parse_ftl_entries(src);
        assert_eq!(
            entries,
            vec![
                // 缩进行被并入上一条 value（fluent 续行语义）
                (
                    "key-a".to_string(),
                    "value a\nindented = not a key".to_string()
                ),
                // `key-b =` 空值条目：尾部空行去除后为空串
                ("key-b".to_string(), String::new()),
            ]
        );
    }

    /// 多行守卫（防御性兼容）：缩进续行并入 value，公共缩进剥离；
    /// 占位符跨行断开时占位符内部换行被丢弃。
    #[test]
    fn test_parse_ftl_entries_multiline_continuation() {
        let src = "msg =\n    first line\n    second line\nnext = done\n";
        let entries = parse_ftl_entries(src);
        assert_eq!(
            entries,
            vec![
                ("msg".to_string(), "first line\nsecond line".to_string()),
                ("next".to_string(), "done".to_string()),
            ]
        );
    }

    /// 占位符跨行归一守卫：`{$` 行尾折行不进入渲染值（fluent 词法语义）。
    #[test]
    fn test_normalize_ftl_placeholders_joins_broken_placeholder() {
        let src = "greet = Hello { $\nname }!\n";
        let entries = parse_ftl_entries(src);
        assert_eq!(
            entries,
            vec![("greet".to_string(), "Hello {$name}!".to_string())]
        );
    }
}
