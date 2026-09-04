// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CLI rerank 子命令示例 — 通过子进程调用 `vecboost rerank`
//!
//! 需要 vecboost 已编译: cargo build --release --features cli

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VecBoost CLI rerank 示例");
    println!("============================\n");

    // 创建临时文档文件
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
    let input_path = temp_path.to_str().unwrap();

    let query = "what is machine learning?";
    println!("📝 查询: \"{}\"", query);
    println!("📄 文档文件: {} (5 条文档)", input_path);
    println!(
        "📝 调用: vecboost rerank --query \"{}\" --documents {}\n",
        query, input_path
    );

    let output = Command::new("vecboost")
        .args(["rerank", "--query", query, "--documents", input_path])
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                println!("✅ 命令执行成功\n");
                println!("📤 输出:");
                println!("{}", String::from_utf8_lossy(&output.stdout));
            } else {
                println!("❌ 命令执行失败 (exit code: {:?})", output.status.code());
                println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            }
        }
        Err(e) => {
            println!("❌ 无法启动 vecboost 进程: {}", e);
            println!("\n💡 请确保 vecboost 已编译并在 PATH 中:");
            println!("   cargo build --release --features cli");
        }
    }

    Ok(())
}
