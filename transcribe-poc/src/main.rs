//! transcribe-own：Qwen3-ASR 转录管线的薄壳 CLI（transcribe-poc-shell-a）。
//!
//! 推理实现已并入主仓 `vecboost`（`transcribe` feature，`src/engine/transcribe/`），
//! 本 crate 只保留 CLI 解析 / 路径归一化 / 段级并行调度 / stdout 契约与
//! py-torch 基线对拍通路。契约见 `CONTRACT.md`：运行关键行外不得有任何
//! stdout 输出；退出码 0 成功 / 2 用法错误 / 1 运行失败。

mod pytorch_backend;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use vecboost::transcribe::{open, read_wav_16k_mono, TranscribeOpts, TRANSCRIBE_SAMPLE_RATE_HZ};

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
/// 主仓 vecboost 以 `ort/load-dynamic` 消费该环境变量加载动态库。
fn resolve_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        let fallback = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../3rdparty/libonnxruntime.so.1.28.2"
        );
        // SAFETY: 启动期单线程时点设置一次；load-dynamic 后续读取
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", fallback);
        }
    }
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

/// 并行度旋钮：TRANSCRIBE_WORKERS / TRANSCRIBE_INTRA，默认 4 × 2
/// （nproc=12 实测，4 worker 各自独立 session 集）。非法值显性报错。
fn parallel_knobs() -> Result<(usize, usize), Box<dyn Error>> {
    let parse_env = |key: &str, default: usize, min: usize, max: usize| -> Result<usize, String> {
        match std::env::var(key) {
            Ok(v) => {
                let n: usize = v
                    .parse()
                    .map_err(|_| format!("{key}={v} is not a valid integer"))?;
                if n < min || n > max {
                    return Err(format!("{key}={n} out of range [{min},{max}]"));
                }
                Ok(n)
            }
            Err(_) => Ok(default),
        }
    };
    let workers = parse_env("TRANSCRIBE_WORKERS", 4, 1, 12).map_err(|e| e.to_string())?;
    let intra = parse_env("TRANSCRIBE_INTRA", 2, 1, 12).map_err(|e| e.to_string())?;
    Ok((workers, intra))
}

/// 已退役实验旋钮（transcribe-poc-shell-a：随 POC 实验文件删除，结论见
/// out/transcribe-poc/own-engine/summary.md）——设置时显性告警不静默。
fn warn_retired_knobs() {
    for key in ["TRANSCRIBE_ENV_POOL", "TRANSCRIBE_BATCH", "TRANSCRIBE_GPU"] {
        if std::env::var_os(key).is_some() {
            eprintln!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "cli-poc-transcribe-own-key-retired",
                    vecboost::i18n::tr_args(&[("key", (key))])
                )
            );
        }
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();

    // TRANSCRIBE_BACKEND=py-torch：嵌入 torch 后端（PyO3 内嵌 CPython，
    // GPU 上跑 1.7B bf16 原生精度），不经 ort/VAD/fbank 管线
    let backend = std::env::var("TRANSCRIBE_BACKEND").unwrap_or_default();
    if backend == "py-torch" {
        let uv_prefix = "/home/dev/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu";
        std::env::set_var("PYTHONHOME", uv_prefix);
        std::env::set_var(
            "PYTHONPATH",
            "/home/dev/projects/vecboost-transcribe/.venv-export/lib/python3.11/site-packages",
        );
        let model_dir = "/home/dev/video/Qwen3-ASR-1.7B";
        eprintln!(
            "{}",
            vecboost::i18n::tr("cli-poc-transcribe-own-backend-py")
        );
        let text = pytorch_backend::run_pytorch_baseline(&args.wav, model_dir)?;
        std::fs::create_dir_all(&args.out_dir)?;
        std::fs::write(args.out_dir.join("transcript.txt"), &text)?;
        let wall_s = t0.elapsed().as_secs_f64();
        println!("WALL_S={wall_s:.3}");
        println!("TRANSCRIPT_CHARS={}", text.chars().count());
        println!("MODE=embedded-pytorch,1.7B-bf16,cuda");
        return Ok(());
    }

    resolve_ort_dylib();

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
                "{}",
                vecboost::i18n::tr_with_args(
                    "cli-poc-transcribe-own-name-path",
                    vecboost::i18n::tr_args(&[
                        ("name", (name)),
                        ("a1", &format!("{:?}", raw)),
                        ("a2", &(normalized.display()).to_string())
                    ])
                )
            );
        }
    }
    let _ = (w, m, o);

    warn_retired_knobs();

    let (workers, intra) = parallel_knobs()?;
    let no_spin = matches!(std::env::var("TRANSCRIBE_NO_SPIN").as_deref(), Ok("1"));
    // 执行后端旋钮：TRANSCRIBE_EP=cpu（默认）| cuda。cuda 要求 ORT_DYLIB_PATH
    // 指向带 CUDA EP 的 onnxruntime 动态库；未知值显性报错。
    let ep = match std::env::var("TRANSCRIBE_EP").as_deref() {
        Ok("cuda") => "cuda",
        Ok("cpu") | Err(_) => "cpu",
        Ok(other) => return Err(format!("TRANSCRIBE_EP={other} invalid (cpu|cuda)").into()),
    };
    let opts = TranscribeOpts {
        intra_threads: intra,
        no_spin,
        cuda: ep == "cuda",
        vad_threshold: 0.5,
    };

    let samples = read_wav_16k_mono(&wav_path)?;
    let audio_seconds = samples.len() as f64 / TRANSCRIBE_SAMPLE_RATE_HZ;
    eprintln!(
        "{}",
        vecboost::i18n::tr_with_args(
            "cli-poc-transcribe-own-wav-loaded",
            vecboost::i18n::tr_args(&[
                ("a1", &(samples.len()).to_string()),
                ("audio_seconds", &format!("{:.3}", audio_seconds))
            ])
        )
    );

    // VAD-only：单会话分段即可（不建 worker 集）
    let vad_session = open(&models_dir, &opts)
        .map_err(|e| format!("load transcribe session from {}: {e}", models_dir.display()))?;
    let segments = vad_session
        .detect_segments(&samples)
        .map_err(|e| format!("vad detect: {e}"))?;
    eprintln!(
        "{}",
        vecboost::i18n::tr_with_args(
            "cli-poc-transcribe-own-3-s-2",
            vecboost::i18n::tr_args(&[
                ("a1", &format!("{:.3}", t0.elapsed().as_secs_f64())),
                ("a2", &(segments.len()).to_string())
            ])
        )
    );

    let mut transcript = String::new();
    let mut seg_lines = String::new();
    if !args.vad_only {
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-poc-transcribe-own-decoding-segments",
                vecboost::i18n::tr_args(&[
                    ("segments_len", &segments.len().to_string()),
                    ("workers", &workers.to_string()),
                    ("intra", &intra.to_string()),
                ]),
            )
        );
        // 段级并行：每 worker 独立会话（ort rc.13 禁并发 Run 同一 Session，
        // 各持一套三图；内存 ×workers，与原 batch/worker 行为等价）
        let results: std::sync::Mutex<Vec<(usize, String)>> = std::sync::Mutex::new(Vec::new());
        let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let next_job = AtomicUsize::new(0);
        let samples = &samples;
        let segments = &segments;
        let opts = &opts;
        let models_dir = &models_dir;
        std::thread::scope(|scope| {
            for w in 0..workers {
                let next_job = &next_job;
                let results = &results;
                let errors = &errors;
                scope.spawn(move || {
                    let session = match open(models_dir, opts) {
                        Ok(s) => s,
                        Err(e) => {
                            errors.lock().unwrap().push(format!("worker {w}: {e}"));
                            return;
                        }
                    };
                    loop {
                        let i = next_job.fetch_add(1, Ordering::SeqCst);
                        if i >= segments.len() {
                            return;
                        }
                        let seg = &segments[i];
                        let seg_samples =
                            &samples[seg.start_sample as usize..seg.end_sample as usize];
                        match session.transcribe_segment(seg_samples) {
                            Ok(text) => {
                                eprintln!(
                                    "{}",
                                    vecboost::i18n::tr_with_args(
                                        "cli-poc-transcribe-own-3-s",
                                        vecboost::i18n::tr_args(&[
                                            ("a1", &format!("{:.3}", t0.elapsed().as_secs_f64())),
                                            ("w", &(w).to_string()),
                                            ("a2", &(i + 1).to_string()),
                                            ("a3", &(segments.len()).to_string()),
                                            ("a4", &format!("{:.2}", seg.start_sec)),
                                            ("a5", &format!("{:.2}", seg.end_sec)),
                                            ("a6", &(text.chars().count()).to_string())
                                        ])
                                    )
                                );
                                results.lock().unwrap().push((i, text));
                            }
                            Err(e) => {
                                errors
                                    .lock()
                                    .unwrap()
                                    .push(format!("worker {w} segment {}: {e}", i + 1));
                                return;
                            }
                        }
                    }
                });
            }
        });

        let errors = errors.into_inner().unwrap();
        if !errors.is_empty() {
            return Err(errors.into_iter().next().unwrap().into());
        }
        // 段文本按时间顺序合并（按段索引排序）
        let mut done = results.into_inner().unwrap();
        done.sort_by_key(|(i, _)| *i);
        for (i, text) in &done {
            let seg = &segments[*i];
            seg_lines.push_str(&format!(
                "{:.2} -- {:.2}: {text}\n",
                seg.start_sec, seg.end_sec
            ));
            transcript.push_str(text);
        }
    }

    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create outDir {} failed: {e}", out_dir.display()))?;
    let seg_path = out_dir.join("segments.txt");
    std::fs::write(&seg_path, &seg_lines)
        .map_err(|e| format!("write {} failed: {e}", seg_path.display()))?;

    let wall_s = t0.elapsed().as_secs_f64();
    if args.vad_only {
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-poc-transcribe-own-vad-only",
                vecboost::i18n::tr_args(&[
                    ("wall_s", &format!("{:.3}", wall_s)),
                    ("a1", &(seg_path.display()).to_string())
                ])
            )
        );
        println!("SEGMENTS={}", segments.len());
    } else {
        let transcript_path = out_dir.join("transcript.txt");
        std::fs::write(&transcript_path, &transcript)
            .map_err(|e| format!("write {} failed: {e}", transcript_path.display()))?;
        eprintln!(
            "{}",
            vecboost::i18n::tr_with_args(
                "cli-poc-transcribe-own-pipeline-done",
                vecboost::i18n::tr_args(&[
                    ("wall_s", &format!("{:.3}", wall_s)),
                    ("a1", &(transcript_path.display()).to_string())
                ])
            )
        );
        println!("WALL_S={wall_s:.3}");
        println!("RTF={:.4}", wall_s / audio_seconds.max(f64::MIN_POSITIVE));
        println!("SEGMENTS={}", segments.len());
        println!("TRANSCRIPT_CHARS={}", transcript.chars().count());
        println!("MODE=seg-parallel:{workers}workers,ort-intra:{intra},ep:{ep}");
    }
    Ok(())
}

fn main() -> ExitCode {
    // stderr 诊断文案走 FTL（stdout 五行契约不在此列），故必须先初始化 i18n；
    // 未 init 时 tr() 会回退成裸键名，污染运维可读输出。
    vecboost::i18n::init();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "cli-poc-transcribe-own-e",
                    vecboost::i18n::tr_args(&[("e", &(e).to_string())])
                )
            );
            return ExitCode::from(2);
        }
    };

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!(
                "{}",
                vecboost::i18n::tr_with_args(
                    "cli-poc-transcribe-own-e",
                    vecboost::i18n::tr_args(&[("e", &(e).to_string())])
                )
            );
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
        assert_eq!(
            strip_annotations("/a/models（说明一；说明二）"),
            "/a/models"
        );
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
