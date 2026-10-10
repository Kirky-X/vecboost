// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 转录门面（库用户与 `transcribe-poc` 薄壳 CLI 的唯一公开入口）。
//!
//! 由 `transcribe-poc-shell-a` 建立：`transcribe-poc` 收敛为薄壳后，本地
//! 推理实现整体删除，其调用需求（wav 读取 / VAD-only 分段 / 段级转录 /
//! 会话旋钮）由本门面承接。整模块 `transcribe` feature 门控。
//!
//! 本 `mod.rs` 只放公开类型与常量（仓库规则14），实现见 `facade.rs`。

/// 16kHz 单声道采样率（wav 输入契约与段起止秒换算共用）。
pub const TRANSCRIBE_SAMPLE_RATE_HZ: f64 = 16_000.0;

/// 会话旋钮（加载期定值）：对齐 POC 原 `asr::SessionOpts` 子集——ENV_POOL /
/// BATCH / GPU 三个实验旋钮随 POC 实验文件退役（结论见
/// `out/transcribe-poc/own-engine/summary.md`），不再提供。
#[derive(Debug, Clone)]
pub struct TranscribeOpts {
    /// 每 session intra 线程数；0 = available_parallelism。
    pub intra_threads: usize,
    /// ORT spin 控制：false = 关自旋（低延迟 Interactive 场景）。
    pub no_spin: bool,
    /// CUDA EP（需带 CUDA 的 onnxruntime 动态库；未启用/不可用显性报错）。
    pub cuda: bool,
    /// silero VAD 概率阈值（缺省 0.5）。
    pub vad_threshold: f32,
}

impl Default for TranscribeOpts {
    fn default() -> Self {
        Self {
            intra_threads: 0,
            no_spin: false,
            cuda: false,
            vad_threshold: 0.5,
        }
    }
}

/// VAD 分段（样本绝对索引 + 秒），调用方据此做段级并行调度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranscribeSegment {
    pub start_sample: i64,
    pub end_sample: i64,
    pub start_sec: f64,
    pub end_sec: f64,
}

/// 转录会话：models_dir 下三图 + VAD + tokenizer 就绪后的可重用句柄。
/// 线程安全（内部 asr Mutex 串行化段解码）；段级并行由调用方持多会话实现
/// （ort rc.13 禁并发 Run 同一 Session，每会话独占一套）。
pub struct TranscribeSession {
    pub(crate) inner: crate::engine::transcribe::engine::Qwen3AsrEngine,
}

impl std::fmt::Debug for TranscribeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribeSession").finish_non_exhaustive()
    }
}

mod facade;

pub use facade::{open, read_wav_16k_mono};
