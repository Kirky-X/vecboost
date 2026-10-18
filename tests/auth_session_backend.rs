// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 会话外置回归：`[auth] session_backend = "database"` 的存储层契约。
//!
//! 直接验证 garrison `DbnexusKvDao` 在部署库上的跨连接语义（模拟多副本进程）：
//! 写入、读取、删除、TTL 过期与 `ensure_table` 幂等。会话键格式与 garrison
//! `src/session` 一致（`token:session:*` / `account:session:*`）。

#![cfg(all(feature = "auth", feature = "sqlite"))]

use garrison::dao::{DbnexusKvDao, GarrisonDao};
use std::time::Duration;

async fn dao_pair() -> (tempfile::TempDir, DbnexusKvDao, DbnexusKvDao) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("sessions.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let pool_a = vecboost::db::DbPool::new(&url)
        .await
        .expect("pool a must connect");
    let pool_b = vecboost::db::DbPool::new(&url)
        .await
        .expect("pool b must connect");
    (
        dir,
        DbnexusKvDao::new(pool_a.inner().clone()),
        DbnexusKvDao::new(pool_b.inner().clone()),
    )
}

#[tokio::test]
async fn ensure_table_is_idempotent_across_connections() {
    let (_dir, dao_a, dao_b) = dao_pair().await;
    dao_a.ensure_table().await.expect("first ensure_table");
    dao_b.ensure_table().await.expect("second ensure_table");
}

#[tokio::test]
async fn session_written_by_one_replica_is_visible_to_another() {
    let (_dir, dao_a, dao_b) = dao_pair().await;
    dao_a.ensure_table().await.expect("ensure_table");

    let key = "token:session:replica-test";
    dao_a
        .set(key, r#"{"login_id":"admin"}"#, 3600)
        .await
        .expect("set");

    assert_eq!(
        dao_b.get(key).await.expect("get"),
        Some(r#"{"login_id":"admin"}"#.to_string()),
        "second connection must observe the session written by the first"
    );

    dao_b.delete(key).await.expect("delete");
    assert_eq!(
        dao_a.get(key).await.expect("get after delete"),
        None,
        "delete from the second connection must be visible to the first"
    );
}

#[tokio::test]
async fn session_key_expires_after_ttl() {
    let (_dir, dao_a, _dao_b) = dao_pair().await;
    dao_a.ensure_table().await.expect("ensure_table");

    dao_a
        .set("account:session:ttl", "v", 1)
        .await
        .expect("set with 1s ttl");
    tokio::time::sleep(Duration::from_millis(1100)).await;

    assert_eq!(
        dao_a.get("account:session:ttl").await.expect("get"),
        None,
        "expired key must be treated as absent"
    );
}
