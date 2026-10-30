// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! CLI rerank 子命令示例 — 通过子进程调用
//! `vecboost --config <临时最小配置> rerank`
//!
//! 前置条件与二进制/dylib/配置解析见 examples/cli/README.md 与本目录 common.rs。

mod common;

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VecBoost CLI rerank 示例");
    println!("============================\n");

    let mut temp_file = NamedTempFile::new()?;
    writeln!(
        temp_file,
        "Machine learning is a subset of artificial intelligence"
    )?;
    writeln!(temp_file, "The weather forecast predicts rain tomorrow")?;
    writeln!(temp_file, "Deep learning uses neural networks for training")?;
    writeln!(temp_file, "Rust is a systems programming language")?;
    writeln!(
        temp_file,
        "Reinforcement learning optimizes policies through rewards"
    )?;
    let temp_path = temp_file.into_temp_path();
    let input_path = temp_path.to_str().ok_or("temp path is not valid UTF-8")?;

    let query = "what is machine learning?";
    println!("📝 查询: \"{query}\"");
    println!("📄 文档文件: {input_path} (5 条文档)");
    let binary = common::resolve_binary();
    common::ensure_cli_binary(&binary)?;
    let dylib_env = common::ort_dylib_env()?;
    let config_file = common::write_cli_config()?;
    println!(
        "📝 调用: {} --config {} rerank --req '{{\"query\":\"{}\",\"documents\":[...]}}'\n",
        binary,
        config_file.path().display(),
        query
    );

    let documents = std::fs::read_to_string(&temp_path)?;
    let docs: Vec<&str> = documents.lines().collect();
    let req_json = serde_json::json!({
        "query": query,
        "documents": docs
    });
    let mut cmd = Command::new(&binary);
    cmd.arg("--config")
        .arg(config_file.path())
        .args(["rerank", "--req", &req_json.to_string()]);
    if let Some((key, value)) = &dylib_env {
        cmd.env(key, value);
        println!("ℹ️ ORT_DYLIB_PATH 未设，向子进程注入仓库自带 dylib：{value}");
    }

    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("❌ 无法启动 vecboost 进程: {e}");
            eprintln!("💡 已尝试: {binary}");
            eprintln!("   仓内构建: cargo build --release --features cli");
            eprintln!("   或设 VECBOOST_BIN 指向已编译二进制（PATH 兜底）");
            std::process::exit(127);
        }
    };

    if output.status.success() {
        println!("✅ 命令执行成功\n");
        println!("📤 输出:");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    } else {
        println!("❌ 命令执行失败 (exit code: {:?})", output.status.code());
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        // 失败按子进程退出码终止本示例：cargo run 退出码如实反映结果
        std::process::exit(output.status.code().unwrap_or(1));
    }
}
