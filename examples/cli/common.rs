// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! CLI 子进程示例共享前置：二进制定位、ONNX Runtime dylib 回退、最小配置生成。

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// 定位被调用的 `vecboost` 二进制：`VECBOOST_BIN` 显式覆盖 → 仓内
/// `target/release/vecboost` → PATH 上的 `vecboost`。仓内构建优先于 PATH，
/// 保证示例验证的是本次源码的产物；VECBOOST_BIN 供指向任意已安装二进制。
pub fn resolve_binary() -> String {
    if let Ok(p) = std::env::var("VECBOOST_BIN")
        && Path::new(&p).is_file()
    {
        return p;
    }
    let repo_bin = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR 必有父目录")
        .join("target/release/vecboost");
    if repo_bin.is_file() {
        return repo_bin.to_string_lossy().into_owned();
    }
    "vecboost".to_string()
}

/// 校验解析到的二进制确有 CLI 行为（`--help` 有界探测）。
///
/// workspace 以默认 features 重新构建（如 `cargo build --release`）会把
/// `target/release/vecboost` 覆盖成无 cli 版本——该二进制对未知子命令落入
/// HTTP 服务器模式，示例子进程将永久挂起。探测超时或输出无 Usage 即报
/// 指引性错误，不进入挂起。
pub fn ensure_cli_binary(binary: &str) -> Result<(), String> {
    let mut child = Command::new(binary)
        .arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("无法启动 {binary}: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child.wait_with_output().map_err(|e| e.to_string())?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                if status.success() && stdout.contains("Usage") {
                    return Ok(());
                }
                return Err(format!(
                    "{binary} 未表现 CLI 行为（--help 退出 {status:?}，输出不含 Usage）。\n\
                     常见原因：二进制被无 --features cli 的构建覆盖。\n\
                     修复：cargo build --release --features cli\n\
                     或设 VECBOOST_BIN 指向带 cli feature 的二进制"
                ));
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "探测超时：{binary} --help 10s 未退出——无 cli feature 的二进制\n\
                         对未知参数落入 HTTP 服务器模式。\n\
                         修复：cargo build --release --features cli\n\
                         或设 VECBOOST_BIN 指向带 cli feature 的二进制"
                    ));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(format!("探测 {binary} 失败: {e}")),
        }
    }
}

/// ONNX Runtime dylib 注入：`ORT_DYLIB_PATH` 未设时回退仓库自带的
/// `3rdparty/onnxruntime/`（文件名按目标平台选择，目录口径同
/// 3rdparty/onnxruntime/README.md）。兜底也缺失时返回指引性错误。
/// 返回 `(键, 值)`，由调用方经 `Command::env` 注入子进程（不改本进程环境）。
pub fn ort_dylib_env() -> Result<Option<(String, String)>, String> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return Ok(None);
    }
    let name = if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    let fallback = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR 必有父目录")
        .join("3rdparty/onnxruntime")
        .join(name);
    if !fallback.is_file() {
        return Err(format!(
            "ONNX Runtime 动态库缺失：ORT_DYLIB_PATH 未设，仓库兜底亦不存在：{}\n\
             按平台放置官方预编译库到该路径，或设 ORT_DYLIB_PATH 指向已装库；\n\
             见 3rdparty/onnxruntime/README.md",
            fallback.display()
        ));
    }
    Ok(Some((
        "ORT_DYLIB_PATH".to_string(),
        fallback.to_string_lossy().into_owned(),
    )))
}

/// 生成 CLI 示例最小配置到临时文件（调用方需持有返回值直到子进程结束，
/// NamedTempFile 析构即删除）。
///
/// 不用默认 `config/config.toml`：其 [translation] 段急切构建 4 个翻译语言对
/// （需本地翻译 bundle），CLI 单命令用不到且 bundle 缺失即启动失败；也不用
/// `config/config_minimal.toml`：其 bge-m3 的 HF cache 外部数据文件（软链到
/// blobs/）过不了 onnx 外部数据路径校验。此处仅 [model] + [logging]——
/// 首次运行经 HF 下载 bge-small 模型（需网络），[logging].file_path 留空
/// 保持子进程 stdout 只承载 JSON。
pub fn write_cli_config() -> Result<tempfile::NamedTempFile, std::io::Error> {
    // confers 配置加载器按扩展名白名单校验（无后缀的默认临时文件名被拒），
    // 临时配置须带 .toml 后缀
    let mut f = tempfile::Builder::new()
        .prefix("vecboost-cli-example-")
        .suffix(".toml")
        .tempfile()?;
    f.write_all(CLI_CONFIG_TOML.as_bytes())?;
    f.flush()?;
    Ok(f)
}

const CLI_CONFIG_TOML: &str = r#"[model]
model_repo = "BAAI/bge-small-en-v1.5"
model_revision = "main"
use_gpu = false
batch_size = 32
expected_dimension = 384
max_sequence_length = 8192

[logging]
level = "info"
console = true
file_path = ""
rotation_size_mb = 10
max_files = 3
"#;
