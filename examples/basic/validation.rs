// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 输入验证示例 — 演示文本、路径和模型 ID 的安全校验
//!
//! VecBoost 提供多层输入验证，防止非法输入和路径遍历攻击：
//! - `InputValidator` / `TextValidator`: 文本长度、批量大小、搜索参数校验
//! - `PathValidator`: 路径遍历攻击防护（符号链接解析 + 白名单）
//! - `is_valid_hf_repo_id`: HuggingFace 模型仓库 ID 格式校验
//!
//! 运行: cargo run -p vecboost-examples --bin validation

use vecboost::utils::{InputValidator, PathValidator, TextValidator};

fn main() {
    println!("🛡️ 输入验证示例");
    println!("================\n");

    // ─── 文本验证 ──────────────────────────────────────────────────────
    println!("📝 文本验证 (TextValidator):");

    let validator = InputValidator::with_default();

    // 合法文本
    let valid_text = "VecBoost 是高性能向量嵌入服务";
    match validator.validate_text(valid_text) {
        Ok(()) => println!("  ✅ \"{}\" — 通过", valid_text),
        Err(e) => println!("  ❌ \"{}\" — {}", valid_text, e),
    }

    // 空文本
    match validator.validate_text("") {
        Ok(()) => println!("  ✅ 空文本 — 通过"),
        Err(e) => println!("  ❌ 空文本 — {}", e),
    }

    // 纯空白文本
    match validator.validate_text("   ") {
        Ok(()) => println!("  ✅ 纯空白 — 通过"),
        Err(e) => println!("  ❌ 纯空白 — {}", e),
    }

    // ─── 批量验证 ──────────────────────────────────────────────────────
    println!("\n📝 批量验证 (validate_batch):");

    let batch = vec![
        "第一条文本".to_string(),
        "第二条文本".to_string(),
        "第三条文本".to_string(),
    ];
    match validator.validate_batch(&batch) {
        Ok(()) => println!("  ✅ {} 条文本 — 全部通过", batch.len()),
        Err(e) => println!("  ❌ 批量验证失败 — {}", e),
    }

    // 空批量
    match validator.validate_batch(&[]) {
        Ok(()) => println!("  ✅ 空批量 — 通过"),
        Err(e) => println!("  ❌ 空批量 — {}", e),
    }

    // ─── 搜索验证 ──────────────────────────────────────────────────────
    println!("\n📝 搜索验证 (validate_search):");

    let query = "什么是向量嵌入";
    let search_texts = vec![
        "向量嵌入是将文本映射到高维空间的技术".to_string(),
        "嵌入模型用于自然语言处理".to_string(),
    ];
    match validator.validate_search(query, &search_texts, Some(5)) {
        Ok(()) => println!("  ✅ query=\"{}\", top_k=5 — 通过", query),
        Err(e) => println!("  ❌ 搜索验证失败 — {}", e),
    }

    // top_k = 0（非法）
    match validator.validate_search(query, &search_texts, Some(0)) {
        Ok(()) => println!("  ✅ top_k=0 — 通过"),
        Err(e) => println!("  ❌ top_k=0 — {}", e),
    }

    // ─── 路径验证 ──────────────────────────────────────────────────────
    println!("\n📝 路径验证 (PathValidator):");

    // 创建临时目录作为白名单根
    let temp_dir = tempfile::TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // 在临时目录中创建一个测试文件
    let test_file = temp_path.join("test.txt");
    std::fs::write(&test_file, "hello").unwrap();

    let path_validator = PathValidator::new().add_allowed_root(temp_path);

    // 合法路径
    match path_validator.validate_path(&test_file) {
        Ok(p) => println!(
            "  ✅ 合法路径 {} — 解析为 {}",
            test_file.display(),
            p.display()
        ),
        Err(e) => println!("  ❌ 合法路径 — {}", e),
    }

    // 路径遍历攻击（包含 ..）
    let malicious = temp_path.join("../../etc/passwd");
    match path_validator.validate_path(&malicious) {
        Ok(p) => println!("  ✅ 恶意路径 — 解析为 {}", p.display()),
        Err(e) => println!("  ❌ 恶意路径 — {}（已拦截）", e),
    }

    // ─── HuggingFace Repo ID 验证 ──────────────────────────────────────
    println!("\n📝 HuggingFace Repo ID 验证:");

    let valid_ids = [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5",
        "local-model", // 单段格式也合法
    ];
    for id in &valid_ids {
        let valid = vecboost::utils::hf_hub::is_valid_hf_repo_id(id);
        println!("  {} \"{}\"", if valid { "✅" } else { "❌" }, id);
    }

    let invalid_ids = [
        "",                // 空字符串
        "/leading-slash",  // 以 / 开头
        "trailing-slash/", // 以 / 结尾
        "org//model",      // 双斜杠
        "a/b/c",           // 超过两段
    ];
    for id in &invalid_ids {
        let valid = vecboost::utils::hf_hub::is_valid_hf_repo_id(id);
        println!("  {} \"{}\"（应拒绝）", if valid { "✅" } else { "❌" }, id);
    }

    println!("\n✅ 输入验证示例完成");
}
