// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 连接池规模的生效边界计算。
//!
//! SQLite 内存库每个连接都是彼此独立的库：连接池把「登录写会话」与「鉴权读
//! 会话」分派到不同连接时，读侧看不到刚写入的会话行，表现为合法 token 被 401
//! 拒绝。内存库因此必须压到单连接；共享缓存模式（`cache=shared`）例外——
//! 所有连接指向同一个内存库，多连接是安全的。
//!
//! 残余风险：dbnexus 的后台 `pool-health-check`（默认 30s 周期）会临时取空全部
//! 空闲连接做校验；窗口内到达的请求最多等 500ms（dbnexus `IDLE_CHECK_MAX_WAIT`）
//! 等回填，超时则新建连接——对内存库而言那是一个全新的空库，会报 no such table。
//! `max_connections=1` 只收窄爆炸半径（不并发开多条内存库），不消除该窗口。
//! 要彻底消除：内存库改用 `cache=shared` 命名 URI 或文件库。

use dbnexus::DbConfig;

/// 连接池连接数上限。
///
/// 仓库自带的 `config/config*.toml` 最大用到 20，留 5 倍余量后封顶，
/// 避免误配把连接数拉到耗尽文件句柄的量级。
pub const MAX_POOL_CONNECTIONS: u32 = 100;

/// URL 是否指向 SQLite 内存库（且非共享缓存模式）。
///
/// `mode=memory` 按 query 的键值判定，不做裸子串匹配——`sqlite:cache=memory.db`
/// 这类文件库不该被误判成内存库。
fn is_in_memory_sqlite(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    let Some(rest) = lower
        .strip_prefix("sqlite:")
        .or_else(|| lower.strip_prefix("sqlite3:"))
    else {
        return false;
    };
    let (path, query) = match rest.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (rest, None),
    };
    let mut shared_cache = false;
    let mut memory_mode = false;
    if let Some(query) = query {
        for pair in query.split('&') {
            let mut kv = pair.splitn(2, '=');
            let key = kv.next().unwrap_or_default();
            let value = kv.next().unwrap_or_default();
            match (key, value) {
                ("cache", "shared") => shared_cache = true,
                ("mode", "memory") => memory_mode = true,
                _ => {}
            }
        }
    }
    if shared_cache {
        // 共享缓存模式：所有连接指向同一个内存库，多连接是安全的。
        return false;
    }
    memory_mode || path.contains(":memory:")
}

/// 连接池上限的生效值。
///
/// 内存 SQLite 强制单连接；其余 URL 取配置值并夹在 `1..=MAX_POOL_CONNECTIONS`。
/// 配置值 0 不再导致启动失败——此前该值从未下传连接池，现在夹到 1 即可用。
pub fn pool_max_connections(url: &str, configured_max: u32) -> u32 {
    if is_in_memory_sqlite(url) {
        return 1;
    }
    configured_max.clamp(1, MAX_POOL_CONNECTIONS)
}

/// 把连接池上下限收敛到合法区间。
///
/// 内存 SQLite 强制单连接（每个连接是彼此独立的库，多连接会让写入与后续读取
/// 落到不同库上）；其余 URL 的上限夹到 `1..=MAX_POOL_CONNECTIONS`。下限不得高于
/// 上限——dbnexus 会以 `min > max` 拒绝建池。
pub fn apply_pool_bounds(config: &mut DbConfig) {
    let max = pool_max_connections(&config.url, config.pool_config.max_connections);
    config.pool_config.max_connections = max;
    config.pool_config.min_connections = config.pool_config.min_connections.min(max);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_sqlite_is_forced_to_single_connection() {
        assert_eq!(pool_max_connections("sqlite::memory:", 20), 1);
        assert_eq!(pool_max_connections("sqlite://:memory:", 20), 1);
        assert_eq!(pool_max_connections("SQLite::MEMORY:", 20), 1);
        assert_eq!(pool_max_connections("sqlite:file:x?mode=memory", 20), 1);
    }

    #[test]
    fn shared_cache_memory_sqlite_keeps_configured_pool() {
        assert_eq!(
            pool_max_connections("sqlite:file:x?mode=memory&cache=shared", 20),
            20
        );
    }

    #[test]
    fn file_and_server_databases_use_configured_pool() {
        assert_eq!(pool_max_connections("sqlite:vecboost.db", 10), 10);
        assert_eq!(pool_max_connections("sqlite:data.db?mode=rwc", 5), 5);
        assert_eq!(pool_max_connections("postgres://host/db", 20), 20);
    }

    #[test]
    fn configured_max_is_clamped_into_valid_range() {
        assert_eq!(pool_max_connections("postgres://host/db", 0), 1);
        assert_eq!(
            pool_max_connections("postgres://host/db", 10_000),
            MAX_POOL_CONNECTIONS
        );
    }

    #[test]
    fn non_sqlite_urls_containing_memory_are_not_treated_as_memory_sqlite() {
        assert_eq!(pool_max_connections("postgres://u:memory:@host/db", 20), 20);
    }

    #[test]
    fn file_database_with_memory_in_its_name_is_not_treated_as_memory_sqlite() {
        assert_eq!(pool_max_connections("sqlite:cache=memory.db", 20), 20);
        assert_eq!(pool_max_connections("sqlite:file:memory.db", 20), 20);
    }
}
