//! WAV 读取：16kHz 单声道 16-bit PCM → f32 [-1,1]。
//! 归一化 i16/32768 与参考链 normalize_samples=true 一致（docs-transcribe/DESIGN.md §6）；
//! PoC 不做重采样，非契约格式直接报错（transcribe-poc/CONTRACT.md）。

use std::error::Error;
use std::path::Path;

pub fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "expected 16-bit PCM wav, got {:?} {}-bit",
            spec.sample_format, spec.bits_per_sample
        )
        .into());
    }
    if spec.sample_rate != 16000 {
        return Err(format!(
            "expected 16kHz wav (PoC does no resampling), got {}Hz",
            spec.sample_rate
        )
        .into());
    }
    if spec.channels != 1 {
        return Err(format!("expected mono wav, got {} channels", spec.channels).into());
    }
    let mut out = Vec::with_capacity(reader.duration() as usize);
    for s in reader.into_samples::<i16>() {
        out.push(s? as f32 / 32768.0);
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
        let dir = std::env::temp_dir().join("transcribe-own-wav-tests");
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
        let err = read_wav_16k_mono(&p).unwrap_err().to_string();
        assert!(err.contains("16kHz"), "unexpected error: {err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_stereo() {
        let p = temp_dir().join("stereo.wav");
        write_test_wav(&p, &[0, 100, 200, 300], 16000, 2);
        let err = read_wav_16k_mono(&p).unwrap_err().to_string();
        assert!(err.contains("mono"), "unexpected error: {err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_missing_file() {
        let err = read_wav_16k_mono(&temp_dir().join("nope.wav")).unwrap_err().to_string();
        assert!(err.contains("No such file"), "unexpected error: {err}");
    }
}
