// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 转录引擎（Qwen3-ASR 三图 + silero VAD + whisper 风格 fbank），
//! 由 onnx-engine-unify 从独立 crate `transcribe-poc` 并入主仓。
//!
//! 整模块经 `engine/mod.rs` 处 `#[cfg(feature = "transcribe")]` 门控。
//! 本 `mod.rs` 只放子模块声明与 re-export（仓库规则14），实现散在
//! `wav`/`features`/`vad`/`tokenizer`/`asr`/`engine` 各独立文件。

pub(crate) mod asr;
pub(crate) mod engine;
pub(crate) mod features;
pub(crate) mod tokenizer;
pub(crate) mod vad;
pub(crate) mod wav;
