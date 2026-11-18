// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! WAV 读取：16kHz 单声道 16-bit PCM → f32 [-1,1]。
//!
//! 归一化 `i16/32768` 与参考链 `normalize_samples=true` 一致（参数以本文件与同目录
//! `features.rs` 为入库事实源）；不做重采样，非契约格式显性报错
//! （`VecboostError`，消除 POC 的 `Box<dyn Error>`）。
use std::path::Path;

use crate::error::VecboostError;

/// 读取 16kHz/单声道/16-bit PCM WAV，归一化为 f32 `[-1,1]`。
///
/// 打开/解码 IO 失败 → [`VecboostError::ModelLoadError`]；格式不符
/// （非 16-bit / 非 16kHz / 非 mono）→ [`VecboostError::InvalidInput`] 且消息含实际值。
pub(crate) fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, VecboostError> {
    let reader = hound::WavReader::open(path).map_err(|e| {
        VecboostError::ModelLoadError(crate::i18n::tr_with_args(
            "detail-open-wav-failed-e-529f2",
            crate::i18n::tr_args(&[("p1", &path.display().to_string()), ("e", &e.to_string())]),
        ))
    })?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
            "detail-expected-16-bit-pcm-wav-got-bit-6bfa5",
            crate::i18n::tr_args(&[
                ("p1", &format!("{:?}", spec.sample_format)),
                ("p2", &spec.bits_per_sample.to_string()),
            ]),
        )));
    }
    if spec.sample_rate != 16000 {
        return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
            "detail-expected-16khz-wav-no-resampling-got-hz-5635c",
            crate::i18n::tr_args(&[("p1", &spec.sample_rate.to_string())]),
        )));
    }
    if spec.channels != 1 {
        return Err(VecboostError::InvalidInput(crate::i18n::tr_with_args(
            "detail-expected-mono-wav-got-channels-c43cf",
            crate::i18n::tr_args(&[("p1", &spec.channels.to_string())]),
        )));
    }
    let mut out = Vec::with_capacity(reader.duration() as usize);
    for s in reader.into_samples::<i16>() {
        let s = s.map_err(|e| {
            VecboostError::ModelLoadError(crate::i18n::tr_with_args(
                "detail-decode-wav-sample-failed-e-0f5e6",
                crate::i18n::tr_args(&[("p1", &path.display().to_string()), ("e", &e.to_string())]),
            ))
        })?;
        out.push(s as f32 / 32768.0);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::path::PathBuf;

    fn write_test_wav(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(path, spec).unwrap();
        for &s in samples {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("vecboost-transcribe-wav-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_normalized_f32() {
        let p = temp_dir().join("ok.wav");
        write_test_wav(&p, &[0, 16384, -32768], 16000, 1);
        let got = read_wav_16k_mono(&p).unwrap();
        assert_eq!(got, vec![0.0, 0.5, -1.0]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_non_16k() {
        let p = temp_dir().join("8k.wav");
        write_test_wav(&p, &[0, 100], 8000, 1);
        let e = read_wav_16k_mono(&p).unwrap_err();
        let err = e.error_detail();
        assert!(err.contains("16kHz"), "unexpected error: {err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_stereo() {
        let p = temp_dir().join("stereo.wav");
        write_test_wav(&p, &[0, 100, 200, 300], 16000, 2);
        let e = read_wav_16k_mono(&p).unwrap_err();
        let err = e.error_detail();
        assert!(err.contains("mono"), "unexpected error: {err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_missing_file() {
        let e = read_wav_16k_mono(&temp_dir().join("nope.wav")).unwrap_err();
        let err = e.error_detail();
        assert!(err.contains("No such file"), "unexpected error: {err}");
    }
}
