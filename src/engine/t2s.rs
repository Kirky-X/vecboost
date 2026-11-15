// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! zh 繁→简归一化（自研最长前缀词典替换）。
//!
//! 替代 `character_converter` 2.1.5：其传递依赖 bincode 1.x（RUSTSEC-2025-0141
//! 全系停维）且上游 2023-11 起停更，无可升级路径。词典逐字节等价迁移自上游
//! `data/t2s.profile`（见 [`t2s_table`] 头注释与 MIT 许可附录），转换语义对齐
//! 上游 `convert_script`：每个位置取词典最长前缀命中则替换，未命中推进一个
//! 字符；全程无命中/替换结果与原文相同时返回 `Cow::Borrowed`（零分配）。

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;

/// 词表行集合（TSV：键<TAB>值；`#` 开头为注释行——词表正文为中文词，不含 `#`）。
static TABLE_LINES: &str = include_str!("t2s_table.txt");

static TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
static MAX_KEY_BYTES: OnceLock<usize> = OnceLock::new();

fn table() -> &'static HashMap<&'static str, &'static str> {
    TABLE.get_or_init(|| {
        TABLE_LINES
            .split('\n')
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| match line.split_once('\t') {
                Some((k, v)) => (k, v),
                // 编译期嵌入的静态数据，损坏属构建产物错误，fail fast
                None => panic!("t2s table line missing TAB separator: {line:?}"),
            })
            .collect()
    })
}

/// 词典最长键的字节长度（探测窗口上界；中文 20 字 ≤ 60 字节，按实际词典计算）。
fn max_key_bytes() -> usize {
    *MAX_KEY_BYTES.get_or_init(|| table().keys().map(|k| k.len()).max().unwrap_or(0))
}

/// 繁体→简体转换（最长前缀词典匹配；等价迁移自 character_converter）。
pub(crate) fn to_simplified(raw: &str) -> Cow<'_, str> {
    let table = table();
    let window = max_key_bytes();
    let bytes = raw.as_bytes();
    let mut converted: Option<String> = None;
    let mut pos = 0usize;

    while pos < bytes.len() {
        // 从窗口上界向 1 递减探测最长词典前缀（跳过非 UTF-8 边界字节）
        let upper = (pos + window).min(bytes.len());
        let mut hit: Option<(&str, usize)> = None;
        let mut probe = upper;
        while probe > pos {
            if raw.is_char_boundary(probe)
                && let Some(mapped) = table.get(&raw[pos..probe])
            {
                hit = Some((*mapped, probe - pos));
                break;
            }
            probe -= 1;
        }

        match hit {
            Some((mapped, len)) => {
                converted = match converted.take() {
                    Some(mut s) => {
                        s.push_str(mapped);
                        Some(s)
                    }
                    None if mapped == &raw[pos..pos + len] => None,
                    None => {
                        let mut s = String::with_capacity(raw.len());
                        s.push_str(&raw[..pos]);
                        s.push_str(mapped);
                        Some(s)
                    }
                };
                pos += len;
            }
            None => {
                // 未命中：原样保留一个字符
                let next = (pos + 1..bytes.len())
                    .find(|&j| raw.is_char_boundary(j))
                    .unwrap_or(bytes.len());
                if let Some(s) = converted.as_mut() {
                    s.push_str(&raw[pos..next]);
                }
                pos = next;
            }
        }
    }

    match converted {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_common_traditional_words() {
        assert_eq!(to_simplified("人類"), "人类");
        assert_eq!(to_simplified("頭髮"), "头发");
        assert_eq!(to_simplified("乾燥"), "干燥");
        // 上下文无关词（词典无歧义项）
        assert_eq!(to_simplified("人類製作複雜的指示"), "人类制作复杂的指示");
    }

    #[test]
    fn unchanged_text_returns_borrowed() {
        // 简体/ASCII/emoji 全程无命中 → Cow::Borrowed（零分配路径）
        for raw in ["下游任务", "pure ascii 123", "🚀🚀 mixed 简体"] {
            assert!(matches!(to_simplified(raw), Cow::Borrowed(_)), "{raw}");
            assert_eq!(to_simplified(raw), raw);
        }
        // 词典恒等项（皇后→皇后）同样不分配
        assert!(matches!(to_simplified("皇后"), Cow::Borrowed(_)));
        assert_eq!(to_simplified("皇后"), "皇后");
    }

    #[test]
    fn mixed_script_passthrough() {
        // 繁简 ASCII 标点 emoji 混排：仅繁体词被替换，其余逐字符原样
        assert_eq!(
            to_simplified("README：人類 can 製作 📦 複雜 stuff!"),
            "README：人类 can 制作 📦 复杂 stuff!"
        );
    }

    #[test]
    fn longest_prefix_wins_and_full_length_key_hits() {
        // 最长键达 20 字（60 字节）也能命中（词典含长词时替换、否则原样）
        let longest = table().keys().max_by_key(|k| k.len()).unwrap();
        assert_eq!(to_simplified(longest), table()[longest]);
        // 前缀命中后从词尾继续：后续字符仍被词典归一（人類→人类、們→们）
        assert_eq!(to_simplified("人類們"), "人类们");
        // 词典恒等项不阻断后续替换：皇后不变、頭髮→头发
        assert_eq!(to_simplified("皇后面"), "皇后面");
    }

    #[test]
    fn empty_input() {
        assert_eq!(to_simplified(""), "");
        assert!(matches!(to_simplified(""), Cow::Borrowed(_)));
    }
}
