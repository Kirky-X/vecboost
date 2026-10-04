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

/// 剥除路径中的全角括号注释段（如 "…/assets（…说明…）/smoke10s.wav"）。
/// 无配对「）」或顺序畸形时原样返回。
fn strip_annotations(s: &str) -> String {
    let mut out = s.to_string();
    while let (Some(b), Some(e)) = (out.find('（'), out.find('）')) {
        if b >= e {
            break;
        }
        out.replace_range(b..e + '）'.len_utf8(), "");
    }
    out
}

/// argv 路径解析：原路径优先；仅当原路径不可用（文件需存在 / 目录需父目录存在）
/// 且路径含注释段、剥除后目标可用时才归一化。返回 (路径, 是否归一化)。
fn normalize_arg_path(raw: &str, must_exist: bool) -> (PathBuf, bool) {
    let orig = PathBuf::from(raw);
    let orig_ok = if must_exist {
        orig.exists()
    } else {
        orig.parent().is_some_and(|d| d.exists())
    };
    if orig_ok || !raw.contains('（') {
        return (orig, false);
    }
    let stripped = strip_annotations(raw);
    if stripped == raw {
        return (orig, false);
    }
    let cand = PathBuf::from(&stripped);
    let cand_ok = if must_exist {
        cand.exists()
    } else {
        cand.parent().is_some_and(|d| d.exists())
    };
    if cand_ok {
        (cand, true)
    } else {
        (orig, false)
    }
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

    // 门禁曾把资产清单说明文字拼进 argv 路径（assets（…）/smoke10s.wav）；
    // 原路径优先，仅在原路径不可用时显性归一化（stderr 留痕）
    let (wav_path, w) = normalize_arg_path(&args.wav.to_string_lossy(), true);
    let (models_dir, m) = normalize_arg_path(&args.models.to_string_lossy(), true);
    let (out_dir, o) = normalize_arg_path(&args.out_dir.to_string_lossy(), false);
    for (name, normalized, raw) in [
        ("wav", &wav_path, &args.wav),
        ("models", &models_dir, &args.models),
        ("outDir", &out_dir, &args.out_dir),
    ] {
        if normalized != raw {
            eprintln!(
                "transcribe-own: {name} path carries annotation text, normalized: {:?} -> {}",
                raw, normalized.display()
            );
        }
    }
    let _ = (w, m, o);

    let samples = wav::read_wav_16k_mono(&wav_path)?;
    eprintln!(
        "transcribe-own: wav loaded: {} samples, {:.3}s",
        samples.len(),
        samples.len() as f32 / vad::SAMPLE_RATE as f32
    );

    let config = vad::VadConfig::default();
    let model_path = models_dir.join("silero_vad.onnx");
    let model = vad::SileroVadModel::new(&model_path, &config)?;
    eprintln!("transcribe-own: vad model loaded: {}", model_path.display());
    let segments = vad::detect_segments(model, &samples, &config)?;

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create outDir {} failed: {e}", out_dir.display()))?;
    let seg_path = out_dir.join("segments.txt");
    let mut text = String::new();
    for s in &segments {
        let start_s = s.start as f64 / vad::SAMPLE_RATE as f64;
        let end_s = s.end as f64 / vad::SAMPLE_RATE as f64;
        // 契约行格式「起 -- 止: 文本」；解码文本属后续阶段，此处留空
        text.push_str(&format!("{start_s:.2} -- {end_s:.2}: \n"));
    }
    std::fs::write(&seg_path, text)
        .map_err(|e| format!("write {} failed: {e}", seg_path.display()))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_annotations_removes_all_bracketed_spans() {
        assert_eq!(
            strip_annotations("/a（资产清单 md5=520fabe9…、smoke10s.wav=10.0s）/smoke10s.wav"),
            "/a/smoke10s.wav"
        );
        assert_eq!(strip_annotations("/a/models（说明一；说明二）"), "/a/models");
        assert_eq!(strip_annotations("/plain/path.wav"), "/plain/path.wav");
        assert_eq!(strip_annotations("/unclosed（path"), "/unclosed（path");
        assert_eq!(strip_annotations("/reversed）a（b"), "/reversed）a（b");
    }

    #[test]
    fn normalize_keeps_plain_existing_path() {
        let dir = std::env::temp_dir().join("transcribe-own-norm-test");
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real.wav");
        std::fs::write(&real, b"x").unwrap();
        // 无注释的真实路径永不归一化
        let (p, n) = normalize_arg_path(&real.to_string_lossy(), true);
        assert!(!n);
        assert_eq!(p, real);
        std::fs::remove_file(&real).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn normalizes_annotated_path_only_when_stripped_target_exists() {
        let dir = std::env::temp_dir().join("transcribe-own-norm-test2");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("smoke.wav"), b"x").unwrap();
        let annotated = format!("{}（资产说明 md5=…）/smoke.wav", dir.display());
        let (p, n) = normalize_arg_path(&annotated, true);
        assert!(n);
        assert_eq!(p, dir.join("smoke.wav"));

        // 剥除后仍不存在 → 保留原路径，由后续文件操作显性报错
        let annotated_bad = format!("{}（资产说明）/missing.wav", dir.display());
        let (p2, n2) = normalize_arg_path(&annotated_bad, true);
        assert!(!n2);
        assert_eq!(p2, PathBuf::from(&annotated_bad));

        // outDir 允许不存在，按父目录存在性判定
        let annotated_dir = format!("{}（资产说明）/out-x", dir.display());
        let (p3, n3) = normalize_arg_path(&annotated_dir, false);
        assert!(n3);
        assert_eq!(p3, dir.join("out-x"));
        std::fs::remove_file(dir.join("smoke.wav")).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
