//! Whisper 风格 log-mel 特征（128 维，居中对齐 STFT），自研 rustfft 实现。
//!
//! 移植参考（Apache-2.0，仅作移植参考、非 Cargo 依赖）：
//! - csukuangfj/kaldi-native-fbank v1.22.3（sherpa-onnx v1.13.8 cmake pin）：
//!   - knf-feature-window.cc:33-52   周期 hann：a=2π/N，w[i]=0.5-0.5·cos(a·i)
//!   - knf-feature-window.cc:85-94   snip_edges=false 帧起点：shift·f+shift/2-N/2
//!   - knf-feature-window.cc:99-125  帧数 = (num_samples + shift/2) / shift
//!   - knf-feature-window.cc:177-201 帧越界反射：-1→0、-2→1；dim→dim-1，可重复
//!   - knf-whisper-feature.cc:126-167 功率谱 |FFT|²，mel 输出线性（无 log）
//!   - knf-mel-computations.cc:242-372, 416-442  librosa 三角滤波（严格
//!     hz>left && hz<right，bin 中心 hz=i·sr/N_fft，num_fft_bins+1=201 个 bin）
//!   - mel-computations.h:95-127  Slaney mel 尺度
//! - k2-fsa/sherpa-onnx v1.13.8（commit 11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf）：
//!   - offline-stream.cc:188-212  reflect-pad 80 前导（torch.stft center=True
//!     语义：padded[i]=samples[pad-i]，边界样本不重复；首块短于 pad 时补零）
//!     与 target 帧数截断 num_samples/shift
//!   - math.cc:115-132  NormalizeWhisperFeatures：
//!     x=clamp(x,1e-10).log10(); x=max(x, 全局max-8); x=(x+4)/4
//!
//! 参数（docs-transcribe/DESIGN.md §6 固化，16000 Hz）：帧移 160（10ms）、
//! 帧长 400（25ms）、n_fft=400、mel 128 bins、low=0、high=Nyquist。
//! 注意：DESIGN.md §6 曾写「HTK 公式、无 slaney 归一化」，与 knf v1.22.3
//! 实际行为不符——knf-whisper-feature.cc:139-144 只设 num_bins/low_freq/
//! is_librosa，norm="slaney" 与 use_slaney_mel_scale=true 保持默认
//! （mel-computations.h:58-70），实际为 Slaney 尺度 + Slaney 归一化
//! （与 librosa.filters.mel 默认一致）。以源码为准，DESIGN.md 已同步修正。

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub const SAMPLE_RATE: i32 = 16000;
pub const FRAME_SHIFT: usize = 160;
pub const FRAME_LENGTH: usize = 400;
pub const NUM_FFT_BINS: usize = FRAME_LENGTH / 2 + 1; // 201
pub const MEL_DIM: usize = 128;
const CENTER_PAD: usize = FRAME_SHIFT / 2; // 80

/// reflect-pad 前导（offline-stream.cc:188-206）：padded[i] = samples[pad-i]，
/// 即 samples[pad], samples[pad-1], …, samples[1]（边界样本自身不重复）；
/// 输入短于 pad 时前段保持零。
fn reflect_pad_lead(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = vec![0.0f32; pad + samples.len()];
    if samples.len() > pad {
        for (i, slot) in padded[..pad].iter_mut().enumerate() {
            *slot = samples[pad - i];
        }
    }
    padded[pad..].copy_from_slice(samples);
    padded
}

fn mel_scale_slaney(freq: f32) -> f32 {
    // 27/log(6.4) = 14.545078505785561（mel-computations.h:126）
    if freq <= 1000.0 {
        freq * 3.0 / 200.0
    } else {
        15.0 + 14.545_078_f32 * (freq / 1000.0).ln()
    }
}

fn inverse_mel_scale_slaney(mel: f32) -> f32 {
    // log(6.4)/27 = 0.06875177742094911（mel-computations.h:111）
    if mel <= 15.0 {
        200.0 / 3.0 * mel
    } else {
        1000.0 * ((mel - 15.0) * 0.068_751_78_f32).exp()
    }
}

/// librosa 三角滤波器组（knf-mel-computations.cc:242-372，slaney 尺度+slaney
/// 归一化），返回每 bin 的 (首个 fft bin 索引, 权重)。
fn build_mel_filters(num_bins: usize) -> Vec<(usize, Vec<f32>)> {
    let num_mel_points = NUM_FFT_BINS; // num_fft_bins + 1
    let fft_bin_width = SAMPLE_RATE as f32 / FRAME_LENGTH as f32; // 40 Hz
    let mel_low = mel_scale_slaney(0.0);
    let mel_high = mel_scale_slaney(0.5 * SAMPLE_RATE as f32);
    let delta = (mel_high - mel_low) / (num_bins + 1) as f32;

    let mut filters = Vec::with_capacity(num_bins);
    for bin in 0..num_bins {
        let left_hz = inverse_mel_scale_slaney(mel_low + bin as f32 * delta);
        let center_hz = inverse_mel_scale_slaney(mel_low + (bin + 1) as f32 * delta);
        let right_hz = inverse_mel_scale_slaney(mel_low + (bin + 2) as f32 * delta);

        let mut first = usize::MAX;
        let mut weights = Vec::new();
        for i in 0..num_mel_points {
            let hz = fft_bin_width * i as f32;
            if hz > left_hz && hz < right_hz {
                let mut weight = if hz <= center_hz {
                    (hz - left_hz) / (center_hz - left_hz)
                } else {
                    (right_hz - hz) / (right_hz - center_hz)
                };
                weight *= 2.0 / (right_hz - left_hz); // slaney norm
                if first == usize::MAX {
                    first = i;
                }
                weights.push(weight);
            }
        }
        assert!(first != usize::MAX, "mel bin {bin} has no active fft bins");
        filters.push((first, weights));
    }
    filters
}

/// NormalizeWhisperFeatures（math.cc:115-132），原位作用于 [n_frames, MEL_DIM]。
fn normalize_whisper(feats: &mut [f32]) {
    for v in feats.iter_mut() {
        *v = (*v).max(1e-10).log10();
    }
    let floor = feats.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for v in feats.iter_mut() {
        *v = v.max(floor);
    }
    for v in feats.iter_mut() {
        *v = (*v + 4.0) / 4.0;
    }
}

pub struct WhisperFbank {
    window: Vec<f32>,
    mel_filters: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn Fft<f32>>,
}

impl WhisperFbank {
    pub fn new() -> Self {
        let mut window = vec![0.0f32; FRAME_LENGTH];
        let a = 2.0 * std::f64::consts::PI / FRAME_LENGTH as f64;
        for (i, w) in window.iter_mut().enumerate() {
            *w = (0.5 - 0.5 * (a * i as f64).cos()) as f32;
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_LENGTH);
        Self {
            window,
            mel_filters: build_mel_filters(MEL_DIM),
            fft,
        }
    }

    /// 线性 mel → whisper 归一化特征，行主序 [num_frames, MEL_DIM]。
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let num_frames = samples.len() / FRAME_SHIFT;
        let mut feats = vec![0.0f32; num_frames * MEL_DIM];
        if num_frames == 0 {
            return feats;
        }

        let padded = reflect_pad_lead(samples, CENTER_PAD);
        let dim = padded.len() as i64;
        let mut frame = vec![0.0f32; FRAME_LENGTH];
        let mut spec = vec![Complex::new(0.0, 0.0); FRAME_LENGTH];
        let mut power = vec![0.0f32; NUM_FFT_BINS];

        for f in 0..num_frames {
            // snip_edges=false 帧起点（knf-feature-window.cc:85-94）
            let start = FRAME_SHIFT as i64 * f as i64 + CENTER_PAD as i64
                - (FRAME_LENGTH / 2) as i64;
            for (i, slot) in frame.iter_mut().enumerate() {
                let mut s = start + i as i64;
                while s < 0 || s >= dim {
                    if s < 0 {
                        s = -s - 1;
                    } else {
                        s = 2 * dim - 1 - s;
                    }
                }
                *slot = padded[s as usize];
            }
            for (i, v) in frame.iter_mut().enumerate() {
                *v *= self.window[i];
            }

            for (i, v) in frame.iter().enumerate() {
                spec[i] = Complex::new(*v, 0.0);
            }
            self.fft.process(&mut spec);
            for (i, p) in power.iter_mut().enumerate() {
                *p = spec[i].re * spec[i].re + spec[i].im * spec[i].im;
            }

            let out = &mut feats[f * MEL_DIM..(f + 1) * MEL_DIM];
            for (b, (offset, weights)) in self.mel_filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (k, w) in weights.iter().enumerate() {
                    energy += w * power[offset + k];
                }
                out[b] = energy;
            }
        }

        normalize_whisper(&mut feats);
        feats
    }
}

impl Default for WhisperFbank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_lead_mirrors_without_boundary_repeat() {
        let padded = reflect_pad_lead(&[1.0, 2.0, 3.0, 4.0, 5.0], 2);
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // 输入短于 pad：前段保持零
        let padded = reflect_pad_lead(&[9.0], 3);
        assert_eq!(padded, vec![0.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn frame_count_is_num_samples_over_shift() {
        assert_eq!(160_000 / FRAME_SHIFT, 1000);
        assert_eq!(160 / FRAME_SHIFT, 1);
        assert_eq!(159 / FRAME_SHIFT, 0);
    }

    #[test]
    fn silence_normalizes_to_minus_one_point_five() {
        let fbank = WhisperFbank::new();
        let feats = fbank.compute(&vec![0.0; 16000]);
        assert_eq!(feats.len(), 100 * MEL_DIM);
        // 全零功率 → mel=0 → log10(1e-10)=-10 → clamp 到全局 max-8（-18 无效）
        // → (-10+4)/4 = -1.5
        for v in feats {
            assert!((v - (-1.5)).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn dc_signal_has_energy_at_bin_zero() {
        let fbank = WhisperFbank::new();
        // 幅度 0.5 的直流信号：加窗后 bin0 = Σ w·0.5 = 0.5·Σw > 0，mel[0] > 0
        let feats = fbank.compute(&vec![0.5; 16000]);
        assert_eq!(feats.len(), 100 * MEL_DIM);
        let mut has_positive = false;
        for v in feats.iter().take(MEL_DIM) {
            if *v > -1.5 + 1e-4 {
                has_positive = true;
            }
        }
        assert!(has_positive, "dc signal must lift mel energies above silence floor");
    }

    #[test]
    fn mel_filters_cover_all_bins() {
        let filters = build_mel_filters(MEL_DIM);
        assert_eq!(filters.len(), MEL_DIM);
        for (offset, weights) in &filters {
            assert!(*offset < NUM_FFT_BINS);
            assert!(!weights.is_empty());
            assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
        }
        // 首个 bin 的滤波器应从 fft bin 1 起（hz=0 与 left_hz=0 相等被严格
        // 不等式排除），且各 bin 的 offset 单调不减
        assert!(filters[0].0 >= 1);
        for w in filters.windows(2) {
            assert!(w[1].0 >= w[0].0);
        }
    }
}
