// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CLI embed 子命令示例 — 通过子进程调用 `vecboost embed --text "hello"`

use std::process::Command;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VecBoost CLI embed 示例");
    println!("===========================\n");

    let text = "hello";
    println!("📝 调用: vecboost embed --text \"{}\"", text);

    let output = Command::new("vecboost")
        .args(["embed", "--text", text])
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
