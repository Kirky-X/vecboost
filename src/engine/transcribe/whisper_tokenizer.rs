// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Whisper 专用 tokenizer 封装（HF tokenizers crate 直接加载 `tokenizer.json`）。
//!
//! 与同目录 `tokenizer.rs`（Qwen3-ASR 的 vocab.json + merges.txt BPE）**不同**：
//! whisper 的 `tokenizer.json` 自带 BPE 模型、byte-level 前后处理与全部
//! `<|...|>` added token，故直接 `Tokenizer::from_file` 加载，不手工拼 BPE。
//!
//! 事实源（本次会话联网实测，ONNX 契约 E）：
//! - `onnx-community/whisper-large-v3-turbo` 的 tokenizer.json / tokenizer_config.json /
//!   generation_config.json（HF API sibling 列表 + 实下载解析）
//! - 特殊 id：`<|endoftext|>`=50257（EOT/bos/eos/pad）、`<|startoftranscript|>`=50258
//!   （SOT=decoder_start_token_id）、`<|transcribe|>`=50360、`<|notimestamps|>`=50364；
//!   语言 `<|en|>`=50259、`<|zh|>`=50260（全表在 generation_config.json.lang_to_id）
//! - 普通文本 token id 恒 < 50257，所有控制符与时间戳 token id >= 50257
//!   （词表 51866），故 decode 以 50257 为界剥离特殊 token
//! - prompt 链（transformers.js bundle v3.7.6 :10720-10770）：
//!   SOT → [语言] → `<|transcribe|>` → `<|notimestamps|>`
//!
//! 错误一律经 i18n key 显性抛出，失败不静默降级。

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::VecboostError;

pub(crate) type WhisperTokResult<T> = Result<T, VecboostError>;

fn tok_load_err(key: &str, e: impl std::fmt::Display) -> VecboostError {
    VecboostError::ModelLoadError(crate::i18n::tr_with_args(
        key,
        crate::i18n::tr_args(&[("e", &e.to_string())]),
    ))
}

/// `<|endoftext|>`（EOT，兼作 bos/eos/pad）。解码遇此即停、且不自输出。
pub const EOT_ID: i64 = 50257;
/// `<|startoftranscript|>`（SOT，decoder_start_token_id）。
pub const SOT_ID: i64 = 50258;
/// `<|transcribe|>`（任务标记：转录，非翻译）。
pub const TRANSCRIBE_ID: i64 = 50360;
/// `<|notimestamps|>`（禁止时间戳 token 输出）。
pub const NOTIMESTAMPS_ID: i64 = 50364;
/// 特殊 token 下界：id >= 此值全为控制符/时间戳，decode 时一律跳过。
pub const SPECIAL_TOKEN_BASE: i64 = 50257;

pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const NOTIMESTAMPS_TOKEN: &str = "<|notimestamps|>";

/// 剥离特殊 token（纯函数，无 tokenizer 依赖，便于离线单测）。
/// whisper 布局下普通文本 token < 50257，控制符与时间戳 >= 50257。
pub(crate) fn filter_special_ids(ids: &[i64]) -> Vec<i64> {
    ids.iter()
        .copied()
        .filter(|&id| id < SPECIAL_TOKEN_BASE)
        .collect()
}

/// SOT prompt 组装（纯函数）：`<|startoftranscript|>` + [可选语言 id] +
/// `<|transcribe|>` + `<|notimestamps|>`。语言 id 由调用方从 tokenizer 查表得到。
pub(crate) fn build_sot_prompt(language_id: Option<i64>) -> Vec<i64> {
    let mut ids = Vec::with_capacity(4);
    ids.push(SOT_ID);
    if let Some(lang) = language_id {
        ids.push(lang);
    }
    ids.push(TRANSCRIBE_ID);
    ids.push(NOTIMESTAMPS_ID);
    ids
}

/// whisper tokenizer：`tokenizer.json` 一份文件即全部资产（无需 vocab/merges 旁路）。
pub struct WhisperTokenizer {
    tk: Tokenizer,
}

impl WhisperTokenizer {
    /// `dir` 为 `models_dir/tokenizer` 目录；缺文件 / 解析失败 / 特殊 id 漂移均显性报错。
    /// 锚定校验防止资产错配（例如误放 Qwen3 的 tokenizer 目录而 id 全对不上）。
    pub fn new(dir: &Path) -> WhisperTokResult<Self> {
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Err(VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                "detail-whisper-tokenizer-file-not-found-3f7a1",
                crate::i18n::tr_args(&[("path", &path.display().to_string())]),
            )));
        }
        let tk = Tokenizer::from_file(&path)
            .map_err(|e| tok_load_err("detail-whisper-tokenizer-load-e-8c2d5", e))?;

        // 特殊 id 锚定：tokenizer_config.json / generation_config.json 实测值。
        for (token, expected) in [
            (EOT_TOKEN, EOT_ID),
            (SOT_TOKEN, SOT_ID),
            (TRANSCRIBE_TOKEN, TRANSCRIBE_ID),
            (NOTIMESTAMPS_TOKEN, NOTIMESTAMPS_ID),
        ] {
            let actual = tk.token_to_id(token).map(i64::from);
            if actual != Some(expected) {
                return Err(VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                    "detail-whisper-tokenizer-id-drift-6b9e4",
                    crate::i18n::tr_args(&[
                        ("token", token),
                        ("expect", &expected.to_string()),
                        (
                            "actual",
                            &actual.map_or_else(|| "missing".to_string(), |v| v.to_string()),
                        ),
                    ]),
                )));
            }
        }

        Ok(Self { tk })
    }

    /// 语言代码（如 `zh` / `en`）→ `<|{lang}|>` id；未知语言显性报错。
    pub fn language_id(&self, language: &str) -> WhisperTokResult<i64> {
        let token = format!("<|{language}|>");
        match self.tk.token_to_id(&token) {
            Some(id) => Ok(i64::from(id)),
            None => Err(VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                "detail-whisper-tokenizer-language-missing-1d4f8",
                crate::i18n::tr_args(&[("lang", language), ("token", &token)]),
            ))),
        }
    }

    /// SOT prompt：语言代码可选（`None` 时省略语言 token，让模型自行判定）。
    pub fn sot_prompt(&self, language: Option<&str>) -> WhisperTokResult<Vec<i64>> {
        let lang_id = match language {
            Some(lang) => Some(self.language_id(lang)?),
            None => None,
        };
        Ok(build_sot_prompt(lang_id))
    }

    /// 解码：先按 id 下界剥离全部特殊 token，再由 tokenizer 的 byte-level decoder
    /// 还原文本（`skip_special_tokens=true` 作二次保障）。错误显性上报。
    pub fn decode(&self, ids: &[i64]) -> WhisperTokResult<String> {
        let plain: Vec<u32> = filter_special_ids(ids)
            .into_iter()
            .filter_map(|id| u32::try_from(id).ok())
            .collect();
        self.tk
            .decode(&plain, true)
            .map_err(|e| tok_load_err("detail-whisper-tokenizer-decode-e-7a3c6", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sot_prompt_without_language_is_three_tokens() {
        // 契约 E：SOT → transcribe → notimestamps，无语言时 3 token。
        assert_eq!(
            build_sot_prompt(None),
            vec![SOT_ID, TRANSCRIBE_ID, NOTIMESTAMPS_ID],
            "无语言 prompt 应为 SOT + transcribe + notimestamps"
        );
    }

    #[test]
    fn sot_prompt_with_language_inserts_language_after_sot() {
        // 契约 E：<|zh|>=50260，语言 token 紧跟 SOT、先于任务标记。
        let ids = build_sot_prompt(Some(50260));
        assert_eq!(
            ids,
            vec![SOT_ID, 50260, TRANSCRIBE_ID, NOTIMESTAMPS_ID],
            "语言 token 必须插在 SOT 与 transcribe 之间"
        );
        assert_eq!(ids.len(), 4, "带语言 prompt 应为 4 token");
    }

    #[test]
    fn special_ids_are_pinned_to_hf_config_values() {
        // 契约 E 实测值；漂移即资产错配，须显性失败。
        assert_eq!(EOT_ID, 50257, "EOT 必须为 <|endoftext|> 的实测 id");
        assert_eq!(SOT_ID, 50258, "SOT 必须为 decoder_start_token_id");
        assert_eq!(TRANSCRIBE_ID, 50360);
        assert_eq!(NOTIMESTAMPS_ID, 50364);
    }

    #[test]
    fn decode_filters_all_special_and_timestamp_ids() {
        // 普通文本 < 50257；控制符（含时间戳 50365）>= 50257 全部剥离。
        let ids = vec![1000i64, SOT_ID, 2000, 50260, 3000, 50364, 50365, EOT_ID];
        assert_eq!(
            filter_special_ids(&ids),
            vec![1000, 2000, 3000],
            "所有 >= 50257 的控制符与时间戳 token 必须剥离"
        );
        assert!(
            filter_special_ids(&[EOT_ID]).is_empty(),
            "纯 EOT 输出应为空"
        );
    }
}
