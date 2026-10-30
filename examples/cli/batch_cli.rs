// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! CLI embed_batch 子命令示例 — 通过子进程调用
//! `vecboost --config <临时最小配置> embed_batch --req '{"texts":["...","..."]}'`
//!
//! 前置条件与二进制/dylib/配置解析见 examples/cli/README.md 与本目录 common.rs。

mod common;

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
    let binary = common::resolve_binary();
    common::ensure_cli_binary(&binary)?;
    let dylib_env = common::ort_dylib_env()?;
    let config_file = common::write_cli_config()?;
    println!(
        "📝 调用: {} --config {} embed_batch --req '{req_json}'\n",
        binary,
        config_file.path().display()
    );

    let mut cmd = Command::new(&binary);
    cmd.arg("--config")
        .arg(config_file.path())
        .args(["embed_batch", "--req", &req_json]);
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
        // 过滤启动日志，只输出 JSON 结果行
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.starts_with('{') || line.starts_with('[') {
                println!("{line}");
            }
        }
        Ok(())
    } else {
        println!("❌ 命令执行失败 (exit code: {:?})", output.status.code());
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        // 失败按子进程退出码终止本示例：cargo run 退出码如实反映结果
        std::process::exit(output.status.code().unwrap_or(1));
    }
}
