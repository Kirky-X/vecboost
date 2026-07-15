// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CLI batch 子命令示例 — 创建临时文件并调用 `vecboost batch --input texts.txt`

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VecBoost CLI batch 示例");
    println!("===========================\n");

    let mut temp_file = NamedTempFile::new()?;
    writeln!(temp_file, "Hello world")?;
    writeln!(temp_file, "Machine learning is fascinating")?;
    writeln!(temp_file, "Rust is a systems programming language")?;
    writeln!(temp_file, "Vector embeddings power semantic search")?;
    let temp_path = temp_file.into_temp_path();
    let input_path = temp_path.to_str().unwrap();

    println!("📄 创建临时输入文件: {}", input_path);
    println!("   包含 4 行文本\n");
    println!("📝 调用: vecboost batch --input {}\n", input_path);

    let output = Command::new("vecboost")
        .args(["batch", "--input", input_path])
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
