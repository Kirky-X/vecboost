// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod langdetect;
pub(crate) mod tokenizer;

pub(crate) use langdetect::{detect_language, is_published_language_code};
pub(crate) use tokenizer::SyncTokenizer;
// Encoding 仅 candle 引擎消费（onnx-primary 下随 candle feature 门控，
// 避免 re-export 在无 candle 构建下 unused import）
#[cfg(feature = "candle")]
pub(crate) use tokenizer::Encoding;
