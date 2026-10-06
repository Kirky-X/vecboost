//!
//!
//! 与参考实现的分层一致：状态机在模型侧（`VadModel::is_speech` = 前向 + 状态机），
//! [head, tail) 在外部样本上模拟，段只记录样本索引、不拷贝音频。

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

pub type VadResult<T> = Result<T, Box<dyn std::error::Error>>;

pub const SAMPLE_RATE: i32 = 16000;

// LSTM 状态形状 [2, 1, 64] 的元素数（层数×batch×hidden）
const LSTM_STATE_ELEMS: usize = 128;

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub threshold: f32,
    pub neg_threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub window_size: i32,
    pub max_speech_duration: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: -1.0,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            window_size: 512,
            max_speech_duration: 20.0,
        }
    }
}

fn seconds_to_samples(seconds: f32) -> i32 {
    // C++ int32_t = int32 * float 赋值截断语义（silero-vad-model.cc:48-51）
    (SAMPLE_RATE as f32 * seconds) as i32
}

/// IsSpeech 状态机，逐行移植 silero-vad-model.cc:97-173。
struct SpeechStateMachine {
    threshold: f32,
    neg_threshold: f32,
    min_silence_samples: i32,
    min_speech_samples: i32,
    window_size: i32,
    triggered: bool,
    current_sample: i32,
    temp_start: i32,
    temp_end: i32,
}

impl SpeechStateMachine {
    fn new(config: &VadConfig) -> Self {
        Self {
            threshold: config.threshold,
            neg_threshold: config.neg_threshold,
            min_silence_samples: seconds_to_samples(config.min_silence_duration),
            min_speech_samples: seconds_to_samples(config.min_speech_duration),
            window_size: config.window_size,
            triggered: false,
            current_sample: 0,
            temp_start: 0,
            temp_end: 0,
        }
    }

    fn is_speech(&mut self, prob: f32) -> bool {
        self.current_sample += self.window_size;

        if prob > self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        if prob > self.threshold && self.temp_start == 0 {
            self.temp_start = self.current_sample;
            return false;
        }

        if prob > self.threshold && self.temp_start != 0 && !self.triggered {
            if self.current_sample - self.temp_start < self.min_speech_samples {
                return false;
            }
            self.triggered = true;
            return true;
        }

        if prob < self.threshold && !self.triggered {
            self.temp_start = 0;
            self.temp_end = 0;
            return false;
        }

        let neg_threshold = if self.neg_threshold < 0.0 {
            (self.threshold - 0.15f32).max(0.01)
        } else {
            self.neg_threshold.max(0.01)
        };

        if prob > neg_threshold && self.triggered {
            return true;
        }

        if prob > self.threshold && !self.triggered {
            self.triggered = true;
            return true;
        }

        if prob < self.threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end < self.min_silence_samples {
                return true;
            }
            self.temp_start = 0;
            self.temp_end = 0;
            self.triggered = false;
            return false;
        }

        false
    }

    fn set_min_silence_duration(&mut self, seconds: f32) {
        self.min_silence_samples = seconds_to_samples(seconds);
    }

    fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

}

/// 抽成 trait 以便用概率序列 mock 驱动状态机与切分逻辑的测试。
pub trait VadModel {
    fn is_speech(&mut self, window: &[f32]) -> VadResult<bool>;
    /// 窗长；v4 导出无重叠，窗长=窗移=512（silero-vad-model.cc:175-179）
    fn window_size(&self) -> i32;
    fn window_shift(&self) -> i32;
    fn min_silence_samples(&self) -> i32;
    fn min_speech_samples(&self) -> i32;
    fn set_min_silence_duration(&mut self, seconds: f32);
    fn set_threshold(&mut self, threshold: f32);
}

/// LSTM 状态零初始化并在窗间滚动（silero-vad-model.cc:247-267, 416-449）。
pub struct SileroVadModel {
    session: Session,
    state: SpeechStateMachine,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl SileroVadModel {
    pub fn new(model_path: &Path, config: &VadConfig) -> VadResult<Self> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)
            .map_err(|e| format!("load vad model {} failed: {e}", model_path.display()))?;
        Ok(Self {
            session,
            state: SpeechStateMachine::new(config),
            h: vec![0.0; LSTM_STATE_ELEMS],
            c: vec![0.0; LSTM_STATE_ELEMS],
        })
    }

    fn run_window(&mut self, window: &[f32]) -> VadResult<f32> {
        let x_shape = vec![1i64, window.len() as i64];
        let h_shape = vec![2i64, 1, 64];
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x",
            Tensor::from_array((x_shape, window.to_vec()))?.upcast(),
        );
        inputs.insert("h", Tensor::from_array((h_shape.clone(), self.h.clone()))?.upcast());
        inputs.insert("c", Tensor::from_array((h_shape, self.c.clone()))?.upcast());

        let outputs = self.session.run(inputs)?;

        let (_, prob_data) = outputs["prob"].try_extract_tensor::<f32>()?;
        let prob = *prob_data.first().ok_or("empty prob output")?;

        let (_, new_h) = outputs["new_h"].try_extract_tensor::<f32>()?;
        if new_h.len() != LSTM_STATE_ELEMS {
            return Err(format!("new_h has {} elems, expect {LSTM_STATE_ELEMS}", new_h.len()).into());
        }
        self.h.copy_from_slice(new_h);

        let (_, new_c) = outputs["new_c"].try_extract_tensor::<f32>()?;
        if new_c.len() != LSTM_STATE_ELEMS {
            return Err(format!("new_c has {} elems, expect {LSTM_STATE_ELEMS}", new_c.len()).into());
        }
        self.c.copy_from_slice(new_c);

        Ok(prob)
    }
}

impl VadModel for SileroVadModel {
    fn is_speech(&mut self, window: &[f32]) -> VadResult<bool> {
        let prob = self.run_window(window)?;
        Ok(self.state.is_speech(prob))
    }

    fn window_size(&self) -> i32 {
        self.state.window_size
    }

    fn window_shift(&self) -> i32 {
        // v4 无重叠：window_overlap_ = 0（silero-vad-model.cc:208-210）
        self.state.window_size
    }

    fn min_silence_samples(&self) -> i32 {
        self.state.min_silence_samples
    }

    fn min_speech_samples(&self) -> i32 {
        self.state.min_speech_samples
    }

    fn set_min_silence_duration(&mut self, seconds: f32) {
        self.state.set_min_silence_duration(seconds);
    }

    fn set_threshold(&mut self, threshold: f32) {
        self.state.set_threshold(threshold);
    }
}

/// 语音段，样本绝对索引（自音频首样本计）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    pub start: i64,
    pub end: i64,
}

impl SpeechSegment {
    pub fn duration_seconds(&self) -> f32 {
        (self.end - self.start) as f32 / SAMPLE_RATE as f32
    }
}

/// 段切分，移植 voice-activity-detector.cc:46-135（AcceptWaveform）与 :170-194（Flush）。
/// 长语音收紧参数（buffer 超过 max_speech_duration 时 min_silence→0.1s、threshold→0.90）
/// 照搬 voice-activity-detector.cc:50-65 与 :228-229。
pub struct VoiceActivityDetector<M: VadModel> {
    model: M,
    config: VadConfig,
    max_utterance_length: i64,
    new_min_silence_duration_s: f32,
    new_threshold: f32,
    // 环形缓冲 [head, tail) 的绝对索引模拟；数据在外部样本切片中
    head: i64,
    tail: i64,
    // 活动段起点，-1 表示无（voice-activity-detector.cc start_）
    start: i64,
    // 不足一窗的残余（voice-activity-detector.cc last_）
    last: Vec<f32>,
    segments: Vec<SpeechSegment>,
}

impl<M: VadModel> VoiceActivityDetector<M> {
    pub fn new(model: M, config: &VadConfig) -> Self {
        Self {
            model,
            max_utterance_length: (SAMPLE_RATE as f32 * config.max_speech_duration) as i64,
            config: config.clone(),
            new_min_silence_duration_s: 0.1,
            new_threshold: 0.90,
            head: 0,
            tail: 0,
            start: -1,
            last: Vec::new(),
            segments: Vec::new(),
        }
    }

    fn buffer_size(&self) -> i64 {
        self.tail - self.head
    }

    pub fn accept_waveform(&mut self, samples: &[f32]) -> VadResult<()> {
        if self.buffer_size() > self.max_utterance_length {
            self.model.set_min_silence_duration(self.new_min_silence_duration_s);
            self.model.set_threshold(self.new_threshold);
        } else {
            self.model.set_min_silence_duration(self.config.min_silence_duration);
            self.model.set_threshold(self.config.threshold);
        }

        self.last.extend_from_slice(samples);
        let window_size = self.model.window_size() as usize;
        let window_shift = self.model.window_shift() as usize;
        if self.last.len() < window_size {
            return Ok(());
        }

        let k = (self.last.len() - window_size) / window_shift + 1;
        let mut is_speech = false;
        for i in 0..k {
            let off = i * window_shift;
            self.tail += window_shift as i64;
            if self.model.is_speech(&self.last[off..off + window_size])? {
                is_speech = true;
            }
        }
        self.last.drain(..k * window_shift);

        let win = self.model.window_size() as i64;
        if is_speech {
            if self.start == -1 {
                self.start = (self.tail - 2 * win - self.model.min_speech_samples() as i64)
                    .max(self.head);
            }
        } else {
            if self.start != -1 && self.buffer_size() > 0 {
                let end = self.tail - self.model.min_silence_samples() as i64;
                self.segments.push(SpeechSegment { start: self.start, end });
                self.head = end;
            }
            if self.start == -1 {
                let end = self.tail - 2 * win - self.model.min_speech_samples() as i64;
                self.head += (end - self.head).max(0);
            }
            self.start = -1;
        }
        Ok(())
    }

    pub fn flush(&mut self) {
        if self.start == -1 || self.buffer_size() == 0 {
            return;
        }
        let end = self.tail;
        if end <= self.start {
            return;
        }
        self.segments.push(SpeechSegment { start: self.start, end });
        self.head = end;
        self.start = -1;
    }

    pub fn into_segments(self) -> Vec<SpeechSegment> {
        self.segments
    }
}

/// 离线整段 VAD。喂入节奏与段过滤照搬
/// 尾部不足一窗时 Flush、时长 <0.1s 的段丢弃。
/// 注意：与该参考一致，样本数为窗长整数倍时不会触发 Flush，
/// 延续到文件尾且未闭合的语音段会被丢弃。
pub fn detect_segments<M: VadModel>(
    model: M,
    samples: &[f32],
    config: &VadConfig,
) -> VadResult<Vec<SpeechSegment>> {
    let window_size = config.window_size as usize;
    let mut vad = VoiceActivityDetector::new(model, config);
    let mut i = 0usize;
    while i < samples.len() {
        if i + window_size <= samples.len() {
            vad.accept_waveform(&samples[i..i + window_size])?;
        } else {
            vad.flush();
        }
        i += window_size;
    }
    Ok(vad
        .into_segments()
        .into_iter()
        .filter(|s| s.duration_seconds() >= 0.1)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 概率序列驱动的 mock：每窗返回 probs[idx]（耗尽后视为 0.0）。
    struct MockVadModel {
        probs: Vec<f32>,
        idx: usize,
        state: SpeechStateMachine,
    }

    impl MockVadModel {
        fn new(probs: &[f32]) -> Self {
            Self {
                probs: probs.to_vec(),
                idx: 0,
                state: SpeechStateMachine::new(&VadConfig::default()),
            }
        }
    }

    impl VadModel for MockVadModel {
        fn is_speech(&mut self, _window: &[f32]) -> VadResult<bool> {
            let prob = self.probs.get(self.idx).copied().unwrap_or(0.0);
            self.idx += 1;
            Ok(self.state.is_speech(prob))
        }
        fn window_size(&self) -> i32 {
            self.state.window_size
        }
        fn window_shift(&self) -> i32 {
            self.state.window_size
        }
        fn min_silence_samples(&self) -> i32 {
            self.state.min_silence_samples
        }
        fn min_speech_samples(&self) -> i32 {
            self.state.min_speech_samples
        }
        fn set_min_silence_duration(&mut self, seconds: f32) {
            self.state.set_min_silence_duration(seconds);
        }
        fn set_threshold(&mut self, threshold: f32) {
            self.state.set_threshold(threshold);
        }
    }

    fn run_mock(probs: &[f32]) -> Vec<SpeechSegment> {
        let n_windows = probs.len();
        let samples = vec![0.0f32; n_windows * 512];
        detect_segments(MockVadModel::new(probs), &samples, &VadConfig::default()).unwrap()
    }

    #[test]
    fn state_machine_min_silence_conversion_truncates() {
        let mut m = MockVadModel::new(&[]);
        m.set_min_silence_duration(0.1);
        assert_eq!(m.min_silence_samples(), 1600);
        m.set_min_silence_duration(0.5);
        assert_eq!(m.min_silence_samples(), 8000);
    }

    #[test]
    fn short_burst_below_min_speech_yields_no_segment() {
        // 3 窗语音（1536 样本 < min_speech 4000）后转静音
        let mut probs = vec![0.9f32; 3];
        probs.extend(vec![0.0; 30]);
        assert!(run_mock(&probs).is_empty());
    }

    #[test]
    fn single_segment_boundaries() {
        // 40 窗语音 + 20 窗静音：
        // 触发窗=9（temp_start=512，current=4608，diff=4096≥4000），
        // start = max(4608-1024-4000, 0) = 0；
        // 静音自窗 41（temp_end=20992）累积，窗 57（current=29184，diff=8192≥8000）切段，
        // end = 29184-8000 = 21184。
        let mut probs = vec![0.9f32; 40];
        probs.extend(vec![0.0; 20]);
        let segs = run_mock(&probs);
        assert_eq!(segs, vec![SpeechSegment { start: 0, end: 21184 }]);
    }

    #[test]
    fn neg_threshold_holds_speech_between_neg_and_threshold() {
        // triggered 后 prob=0.45 落在 (neg=0.35, threshold=0.5) 区间应继续算语音
        let mut state = SpeechStateMachine::new(&VadConfig::default());
        // 窗 1-8：temp_start 自窗 1 起累积，diff < min_speech(4000)，不触发
        for _ in 0..8 {
            assert!(!state.is_speech(0.9), "warm-up windows must not trigger");
        }
        assert!(state.is_speech(0.9), "window 9 triggers");
        assert!(state.is_speech(0.45), "0.45 > neg_threshold keeps speech");
        // 窗 11 起静音（temp_end=11*512=5632），8000 样本 → 窗 27（diff=8192）才切段
        for _ in 11..27 {
            assert!(state.is_speech(0.3), "below neg_threshold still within min_silence");
        }
        assert!(!state.is_speech(0.3), "silence beyond min_silence closes");
    }

    #[test]
    fn long_speech_tightens_and_forces_split() {
        // 连续 700 窗 prob=0.6（>0.5，<0.9）：
        // 窗 627 开头 buffer_size=626*512=320512>320000 → 收紧（threshold=0.9、min_sil=1600）；
        // 窗 627 起 prob<0.9 进入结束累积，窗 631（diff=2048≥1600）切段：
        // end = 631*512-1600 = 321472；随后恢复默认阈值重新起段，
        // 700 窗整除无 Flush → 第二段（未闭合）被丢弃。
        let probs = vec![0.6f32; 700];
        let segs = run_mock(&probs);
        assert_eq!(segs, vec![SpeechSegment { start: 0, end: 321472 }]);
    }

    #[test]
    fn flush_captures_open_segment_on_short_tail() {
        // 语音延续到 EOF 且样本数非窗长整数倍：Flush 收尾（参考 CLI 分支）
        let probs = vec![0.9f32; 70];
        let samples = vec![0.0f32; 70 * 512 + 100];
        let segs = detect_segments(MockVadModel::new(&probs), &samples, &VadConfig::default()).unwrap();
        assert_eq!(segs, vec![SpeechSegment { start: 0, end: 70 * 512 }]);
    }

    #[test]
    fn window_aligned_eof_drops_open_segment_like_reference() {
        // 样本数恰为窗长整数倍：参考 CLI 不调 Flush，未闭合段照搬丢弃
        let probs = vec![0.9f32; 70];
        let samples = vec![0.0f32; 70 * 512];
        let segs = detect_segments(MockVadModel::new(&probs), &samples, &VadConfig::default()).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn flush_segment_on_short_tail() {
        // 语音延续到 EOF 且样本数非窗长整数倍：Flush 收尾（参考 CLI 分支），
        // 段 [0, 9*512)，0.288s ≥ 0.1s，过滤后保留。
        // 默认参数下状态机不可能产生 <0.1s 的段（闭合段下界 0.3s、Flush 段下界
        // 4608 样本），<0.1s 过滤是参考 CLI 的防御分支，照搬保留。
        let probs = vec![0.9f32; 9];
        let samples = vec![0.0f32; 9 * 512 + 100];
        let segs = detect_segments(MockVadModel::new(&probs), &samples, &VadConfig::default()).unwrap();
        assert_eq!(segs, vec![SpeechSegment { start: 0, end: 4608 }]);
    }
}
