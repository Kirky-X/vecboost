// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! WAV 读取：16kHz 单声道 16-bit PCM → f32 [-1,1]。
//!
//! 归一化 `i16/32768` 与参考链 `normalize_samples=true` 一致
//! （docs-transcribe/DESIGN.md §6）；不做重采样，非契约格式显性报错
//! （`VecboostError`，R-transcribe-engine-001/004——消除 POC 的 `Box<dyn Error>`）。

use std::path::Path;

use crate::error::VecboostError;

/// 读取 16kHz/单声道/16-bit PCM WAV，归一化为 f32 `[-1,1]`。
///
/// 打开/解码 IO 失败 → [`VecboostError::ModelLoadError`]；格式不符
/// （非 16-bit / 非 16kHz / 非 mono）→ [`VecboostError::InvalidInput`] 且消息含实际值。
// 预留给 T016 POC CLI 薄壳 / 服务域消费（引擎按 samples 入参，lib 内暂无调用方）。
#[allow(dead_code)]
pub(crate) fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, VecboostError> {
    let reader = hound::WavReader::open(path).map_err(|e| {
        VecboostError::ModelLoadError(format!("open wav {} failed: {e}", path.display()))
    })?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(VecboostError::InvalidInput(format!(
            "expected 16-bit PCM wav, got {:?} {}-bit",
            spec.sample_format, spec.bits_per_sample
        )));
    }
    if spec.sample_rate != 16000 {
        return Err(VecboostError::InvalidInput(format!(
            "expected 16kHz wav (no resampling), got {}Hz",
            spec.sample_rate
        )));
    }
    if spec.channels != 1 {
        return Err(VecboostError::InvalidInput(format!(
            "expected mono wav, got {} channels",
            spec.channels
        )));
    }
    let mut out = Vec::with_capacity(reader.duration() as usize);
    for s in reader.into_samples::<i16>() {
        let s = s.map_err(|e| {
            VecboostError::ModelLoadError(format!(
                "decode wav {} sample failed: {e}",
                path.display()
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
