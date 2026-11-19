// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 流式转录纠错层（纯文本后处理，可脱离 ASR/WS 独立测试）。
//!
//! 机制依据（一手论文）：
//! - 上下文偏置：Gboard RNN-T 论文（arXiv:1811.06621）第 4 节
//!   `log P(y|x) + λ log P_C(y)`——用户词表对识别候选做先验偏移。
//! - 二遍重打分：Google Interspeech 2020《Parallel Rescoring with Transformer
//!   for Streaming On-Device Speech Recognition》——第一遍快速出字后对
//!   N-best 候选重新排序。本实现用轻量字符 bigram 统计代替外部 LM 资产，
//!   无 LM 时仍可工作（计划"无资产降级"条款）。

use std::collections::HashMap;
use std::path::Path;

/// 偏置词表：每行一个词，UTF-8；λ 控制偏置强度（Gboard 公式的 λ）。
#[derive(Debug, Clone, Default)]
pub struct ContextBiasing {
    words: Vec<String>,
    weight: f32,
}

impl ContextBiasing {
    /// 从已加载的词表构造（WS handler 共用同一份词表构造偏置与重打分器）。
    pub fn from_words(words: Vec<String>, weight: f32) -> Self {
        Self { words, weight }
    }

    /// 从文件加载词表；文件缺失返回空表（显式降级，不阻塞）。
    pub fn load(path: Option<&String>, weight: f32) -> Self {
        let words = match path {
            Some(p) => match std::fs::read_to_string(Path::new(p)) {
                Ok(text) => text
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect(),
                Err(e) => {
                    log::warn!(
                        "{}",
                        crate::i18n::tr_with_args(
                            "log-stream-biasing-list-unavailable-a4f1",
                            crate::i18n::tr_args(&[("path", p), ("e", &e.to_string())]),
                        )
                    );
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        Self { words, weight }
    }

    /// 对每个候选打分：命中词表词的字符重叠度 × λ（shallow fusion 近似）。
    pub fn score(&self, text: &str) -> f32 {
        if self.words.is_empty() || self.weight <= 0.0 {
            return 0.0;
        }
        let mut best = 0.0f32;
        for word in &self.words {
            if text.contains(word.as_str()) {
                let overlap = word.chars().count() as f32 / text.chars().count().max(1) as f32;
                best = best.max(overlap);
            }
        }
        self.weight * best
    }

    pub fn is_enabled(&self) -> bool {
        !self.words.is_empty() && self.weight > 0.0
    }
}

/// 轻量二遍重打分器：字符 bigram 频率 LM（从偏置词表 + 候选自身构造）。
/// 无外部 LM 资产时的降级实现；有资产时可扩展加载 n-gram 计数表。
#[derive(Debug, Clone, Default)]
pub struct Rescorer {
    bigram: HashMap<(char, char), usize>,
    weight: f32,
}

impl Rescorer {
    /// 从词表构造 bigram 统计；weight<=0 表示禁用。
    pub fn new(biasing_words: &[String], weight: f32) -> Self {
        let mut bigram = HashMap::new();
        for w in biasing_words {
            let chars: Vec<char> = w.chars().collect();
            for pair in chars.windows(2) {
                *bigram.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
        }
        Self { bigram, weight }
    }

    /// 候选得分：已知 bigram 命中数 / 总 bigram 数 × weight。
    pub fn score(&self, text: &str) -> f32 {
        if self.weight <= 0.0 || self.bigram.is_empty() {
            return 0.0;
        }
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 2 {
            return 0.0;
        }
        let hits = chars
            .windows(2)
            .filter(|p| self.bigram.contains_key(&(p[0], p[1])))
            .count();
        self.weight * hits as f32 / (chars.len() - 1) as f32
    }

    pub fn is_enabled(&self) -> bool {
        self.weight > 0.0 && !self.bigram.is_empty()
    }
}

/// 组合纠错：偏置 + 重打分，从 N-best 候选中选综合分最高者。
/// 单候选时只做偏置打分（返回原文，偏置分供上层观测）。
pub fn correct_candidates(
    candidates: &[String],
    biasing: &ContextBiasing,
    rescorer: &Rescorer,
) -> String {
    if candidates.is_empty() {
        return String::new();
    }
    if candidates.len() == 1 {
        return candidates[0].clone();
    }
    // 无有效打分器（默认无词表/无 LM）时直接返回 top-1，避免 max_by 平局
    // 选中第二个候选导致文本被无依据替换（架构/性能复审 H1-1）。
    if !biasing.is_enabled() && !rescorer.is_enabled() {
        return candidates[0].clone();
    }
    let best = candidates
        .iter()
        .max_by(|a, b| {
            let sa = biasing.score(a) + rescorer.score(a);
            let sb = biasing.score(b) + rescorer.score(b);
            // 平局偏向先出现的候选（top-1），而非 max_by 默认的最后一个
            match sa.total_cmp(&sb) {
                std::cmp::Ordering::Equal => std::cmp::Ordering::Greater,
                other => other,
            }
        })
        .unwrap_or(&candidates[0]);
    best.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn biasing(words: &[&str], w: f32) -> ContextBiasing {
        ContextBiasing {
            words: words.iter().map(|s| s.to_string()).collect(),
            weight: w,
        }
    }

    #[test]
    fn biasing_prefers_known_term() {
        let b = biasing(&["神经网络", "向量数据库"], 0.5);
        assert!(b.score("神经网络模型") > b.score("神精网络模型"));
        assert_eq!(b.score("无关文本"), 0.0);
    }

    #[test]
    fn biasing_disabled_when_zero_weight_or_empty() {
        assert!(!biasing(&["词"], 0.0).is_enabled());
        assert!(!biasing(&[], 1.0).is_enabled());
    }

    #[test]
    fn rescorer_prefers_known_bigrams() {
        let r = Rescorer::new(&["数据库".to_string()], 1.0);
        assert!(r.score("数据库系统") > r.score("数据裤系统"));
        assert_eq!(r.score("字"), 0.0); // 不足 2 字符
    }

    #[test]
    fn correct_picks_highest_scored_candidate() {
        let b = biasing(&["向量数据库"], 0.5);
        let r = Rescorer::new(&["向量数据库".to_string()], 0.5);
        let cands = vec!["向量数据库".to_string(), "向量数据裤".to_string()];
        assert_eq!(correct_candidates(&cands, &b, &r), "向量数据库");
    }

    #[test]
    fn no_biasing_returns_top1_not_tie_picked_alternative() {
        let b = biasing(&[], 0.0); // 禁用
        let r = Rescorer::default();
        let cands = vec!["top1".to_string(), "alternative".to_string()];
        assert_eq!(correct_candidates(&cands, &b, &r), "top1");
    }

    #[test]
    fn empty_candidates_is_empty() {
        let b = biasing(&[], 0.0);
        assert_eq!(correct_candidates(&[], &b, &Rescorer::default()), "");
    }
}
