// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 审计日志示例 — 演示 AuditLogger 的安全事件记录与文件轮转
//!
//! AuditLogger 通过后台异步批量写入实现高性能审计：
//! - `log_*` 方法仅 mpsc send（非阻塞），1000 次调用 < 1ms
//! - 后台 writer task 每 1s 或 100 条时批量 flush
//! - 支持文件大小达到阈值后自动轮转
//!
//! 运行: cargo run -p vecboost-examples --bin audit_demo

use std::path::PathBuf;

use vecboost::audit::{AuditConfig, AuditLogger};

#[tokio::main]
async fn main() {
    println!("📋 审计日志（AuditLogger）示例");
    println!("================================\n");

    // 使用临时目录存放审计日志
    let temp_dir = tempfile::TempDir::new().unwrap();
    let log_path = temp_dir.path().join("audit.log");

    // 1. 创建审计日志配置
    let config = AuditConfig {
        enabled: true,
        log_file_path: log_path.clone(),
        log_level: "info".to_string(),
        max_file_size_mb: 10,
        max_files: 3,
        async_write: true,
    };

    let logger = AuditLogger::new(config);
    println!("✅ AuditLogger 已创建");
    println!("  启用: {}", logger.is_enabled());
    println!("  日志路径: {}", log_path.display());
    println!("  最大文件: 10MB");
    println!("  轮转保留: 3 个\n");

    // 2. 记录各类安全事件
    println!("📝 记录安全事件:");

    // 登录成功
    logger.log_login_success("admin", Some("192.168.1.100".to_string()));
    println!("  ✅ 登录成功: admin @ 192.168.1.100");

    // 登录失败
    logger.log_login_failed("hacker", Some("10.0.0.50".to_string()), "wrong password");
    println!("  ❌ 登录失败: hacker @ 10.0.0.50");

    // 用户操作
    logger.log_user_created("newuser", "admin", Some("192.168.1.100".to_string()));
    println!("  ✅ 用户创建: newuser by admin");

    logger.log_user_updated("newuser", "admin", None);
    println!("  ✅ 用户更新: newuser by admin");

    // 权限拒绝
    logger.log_permission_denied("newuser", Some("192.168.1.100".to_string()), "/admin/settings");
    println!("  ❌ 权限拒绝: newuser → /admin/settings");

    // Token 刷新
    logger.log_token_refresh("admin", Some("192.168.1.100".to_string()));
    println!("  ✅ Token 刷新: admin");

    // 未授权访问
    logger.log_unauthorized_access(Some("10.0.0.99".to_string()), "/api/v1/internal");
    println!("  ❌ 未授权访问: 10.0.0.99 → /api/v1/internal");

    // 速率限制
    logger.log_rate_limit_exceeded(Some("abuser".to_string()), Some("10.0.0.50".to_string()));
    println!("  ❌ 速率限制: abuser @ 10.0.0.50");

    // 登出
    logger.log_logout("admin", Some("192.168.1.100".to_string()));
    println!("  ✅ 登出: admin\n");

    // 3. 同步等待 flush 到磁盘
    println!("📝 等待 flush 到磁盘...");
    logger.flush().await.unwrap();
    println!("  flush 完成\n");

    // 4. 读取并展示日志内容
    println!("📊 审计日志内容:");
    let content = tokio::fs::read_to_string(&log_path).await.unwrap();
    for line in content.lines() {
        if let Ok(event) = serde_json::from_str::<serde_json::Value>(line) {
            println!(
                "  [{}] {} user={} ip={} success={}",
                event["timestamp"].as_str().unwrap_or(""),
                event["event_type"].as_str().unwrap_or(""),
                event["user"].as_str().unwrap_or("-"),
                event["ip"].as_str().unwrap_or("-"),
                event["success"].as_bool().unwrap_or(false),
            );
        }
    }

    // 5. 批量性能演示
    println!("\n📝 批量性能测试 (1000 条事件)...");
    let start = std::time::Instant::now();
    for i in 0..1000 {
        logger.log_login_success(&format!("user_{}", i), Some("127.0.0.1".to_string()));
    }
    let elapsed = start.elapsed();
    println!("  1000 次 log_* 调用耗时: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    logger.flush().await.unwrap();
    println!("  flush 完成");

    // 6. 禁用状态演示
    println!("\n📝 禁用状态的 AuditLogger:");
    let disabled_config = AuditConfig {
        enabled: false,
        log_file_path: PathBuf::from("/tmp/disabled_audit.log"),
        ..Default::default()
    };
    let disabled_logger = AuditLogger::new(disabled_config);
    println!("  启用: {}", disabled_logger.is_enabled());
    disabled_logger.log_login_success("ignored_user", None);
    println!("  log_* 调用被静默忽略");

    println!("\n✅ 审计日志示例完成");
}
