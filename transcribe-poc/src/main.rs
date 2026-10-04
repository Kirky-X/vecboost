//! transcribe-own：自研 Qwen3-ASR 转录管线 PoC。
//! 本轮交付 VAD 阶段：读 wav + silero VAD；`--vad-only` 输出 SEGMENTS= 与
//! segments.txt。fbank/编码/KV-cache 解码属后续阶段。
//! stdout 契约见 transcribe-poc/CONTRACT.md：运行关键行外不得有任何 stdout 输出。

mod vad;
mod wav;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: transcribe-own <wav> <outDir> [--models <dir>] [--vad-only]";

const DEFAULT_MODELS_DIR: &str = "assets/models";

struct Args {
    wav: PathBuf,
    out_dir: PathBuf,
    models: PathBuf,
    vad_only: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let wav = it.next().ok_or_else(|| USAGE.to_string())?;
    let out_dir = it.next().ok_or_else(|| USAGE.to_string())?;

    let mut models: Option<String> = None;
    let mut vad_only = false;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--models" => {
                models = Some(it.next().ok_or("--models requires a value")?);
            }
            "--vad-only" => vad_only = true,
            other => return Err(format!("unknown argument: {other}\n{USAGE}")),
        }
    }

    Ok(Args {
        wav: PathBuf::from(wav),
        out_dir: PathBuf::from(out_dir),
        models: models
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODELS_DIR)),
        vad_only,
    })
}

/// 门禁经 ORT_DYLIB_PATH 显式指定；未设置时回退 worktree 3rdparty 同一 dylib，
/// 便于本地裸跑（CONTRACT.md：库文件在 worktree 3rdparty/，主仓只读复制件）。
fn resolve_ort_dylib() -> String {
    std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../3rdparty/libonnxruntime.so.1.28.2"
        )
        .to_string()
    })
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    if !args.vad_only {
        // 解码链（fbank/conv/encoder/KV-cache greedy）未实现；实现后删除本分支
        return Err(
            "full pipeline not implemented yet (this phase ships wav+VAD only; use --vad-only)"
                .into(),
        );
    }

    let t0 = Instant::now();
    let dylib = resolve_ort_dylib();
    eprintln!("transcribe-own: loading onnxruntime dylib: {dylib}");
    ort::init_from(&dylib)?.with_name("transcribe-own").commit();

    let samples = wav::read_wav_16k_mono(&args.wav)?;
    eprintln!(
        "transcribe-own: wav loaded: {} samples, {:.3}s",
        samples.len(),
        samples.len() as f32 / vad::SAMPLE_RATE as f32
    );

    let config = vad::VadConfig::default();
    let model_path = args.models.join("silero_vad.onnx");
    let model = vad::SileroVadModel::new(&model_path, &config)?;
    eprintln!("transcribe-own: vad model loaded: {}", model_path.display());
    let segments = vad::detect_segments(model, &samples, &config)?;

    std::fs::create_dir_all(&args.out_dir)?;
    let seg_path = args.out_dir.join("segments.txt");
    let mut text = String::new();
    for s in &segments {
        let start_s = s.start as f64 / vad::SAMPLE_RATE as f64;
        let end_s = s.end as f64 / vad::SAMPLE_RATE as f64;
        // 契约行格式「起 -- 止: 文本」；解码文本属后续阶段，此处留空
        text.push_str(&format!("{start_s:.2} -- {end_s:.2}: \n"));
    }
    std::fs::write(&seg_path, text)?;
    eprintln!(
        "transcribe-own: vad done in {:.3}s, {} segments -> {}",
        t0.elapsed().as_secs_f32(),
        segments.len(),
        seg_path.display()
    );

    println!("SEGMENTS={}", segments.len());
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("transcribe-own: {e}");
            return ExitCode::from(2);
        }
    };

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("transcribe-own: {e}");
            ExitCode::from(1)
        }
    }
}
