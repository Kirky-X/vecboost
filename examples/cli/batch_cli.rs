// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! CLI embed_batch 子命令示例 — 通过子进程调用
//! `vecboost embed_batch --req '{"texts":["...","..."]}'`

use std::process::Command;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 VecBoost CLI embed_batch 示例");
    println!("=================================\n");

    let texts = [
        "Hello world",
        "Machine learning is fascinating",
        "Rust is a systems programming language",
        "Vector embeddings power semantic search",
    ];
    let texts_json = serde_json::to_string(&texts)?;
    let req_json = format!(r#"{{"texts":{texts_json}}}"#);

    println!("📄 批量文本: {} 条\n", texts.len());
    println!("📝 调用: vecboost embed_batch --req '{req_json}'\n");

    let output = Command::new("vecboost")
        .args(["embed_batch", "--req", &req_json])
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                println!("✅ 命令执行成功\n");
                println!("📤 输出:");
                // 过滤启动日志，只输出 JSON 结果行
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.starts_with('{') || line.starts_with('[') {
                        println!("{}", line);
                    }
                }
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
