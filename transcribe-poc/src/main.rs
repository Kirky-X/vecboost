use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: transcribe-own <wav> <outDir> [--models <dir>]";

const DEFAULT_MODELS_DIR: &str = "assets/models";

struct Args {
    wav: PathBuf,
    out_dir: PathBuf,
    models: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let wav = it.next().ok_or_else(|| USAGE.to_string())?;
    let out_dir = it.next().ok_or_else(|| USAGE.to_string())?;

    let mut models: Option<String> = None;
    let mut rest = it;
    while let Some(a) = rest.next() {
        match a.as_str() {
            "--models" => {
                models = Some(rest.next().ok_or("--models requires a value")?);
            }
            other => return Err(format!("unknown argument: {other}\n{USAGE}")),
        }
    }

    Ok(Args {
        wav: PathBuf::from(wav),
        out_dir: PathBuf::from(out_dir),
        models: models
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODELS_DIR)),
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("transcribe-own: {e}");
            return ExitCode::from(2);
        }
    };

    // Pipeline (silero VAD -> whisper-style fbank -> conv_frontend -> encoder
    // -> KV-cache greedy decode) is not implemented yet; this skeleton only
    // fixes the CLI contract. On success stdout must carry exactly the key
    // lines WALL_S= / RTF= / SEGMENTS= / MODE= (see CONTRACT.md) and nothing
    // else, so this stub deliberately prints nothing to stdout and fails.
    eprintln!(
        "transcribe-own: pipeline not implemented (prep-phase skeleton); wav={:?} out_dir={:?} models={:?}",
        args.wav, args.out_dir, args.models
    );
    ExitCode::from(1)
}
