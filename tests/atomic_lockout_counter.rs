// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 锁定计数原子性回归：`AtomicKvDao::incr` 在 SQLite 上无丢失自增。

#![cfg(all(feature = "auth", feature = "sqlite"))]

use garrison::dao::GarrisonDao;
use std::time::Duration;
use vecboost::auth::atomic_kv_dao::{AtomicKvDao, build_atomic_dao};

async fn dao_pair() -> (tempfile::TempDir, AtomicKvDao, AtomicKvDao) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("lockout.db");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let pool_a = vecboost::db::DbPool::new(&url)
        .await
        .expect("pool a must connect");
    let pool_b = vecboost::db::DbPool::new(&url)
        .await
        .expect("pool b must connect");
    (
        dir,
        build_atomic_dao(pool_a.inner().clone())
            .await
            .expect("atomic dao a"),
        build_atomic_dao(pool_b.inner().clone())
            .await
            .expect("atomic dao b"),
    )
}

#[tokio::test]
async fn sequential_incr_returns_monotonic_counts() {
    let (_dir, dao, _) = dao_pair().await;
    let key = "lockout:seq";
    assert_eq!(dao.incr(key, 3600).await.expect("incr 1"), 1);
    assert_eq!(dao.incr(key, 3600).await.expect("incr 2"), 2);
    assert_eq!(dao.incr(key, 3600).await.expect("incr 3"), 3);
}

#[tokio::test]
async fn expired_counter_resets_to_one() {
    let (_dir, dao, _) = dao_pair().await;
    let key = "lockout:ttl";
    assert_eq!(dao.incr(key, 1).await.expect("first incr"), 1);
    tokio::time::sleep(Duration::from_millis(1100)).await;
    assert_eq!(
        dao.incr(key, 1).await.expect("post-expiry incr"),
        1,
        "expired counter must be treated as absent"
    );
}

#[tokio::test]
async fn concurrent_incr_from_two_replicas_does_not_lose_counts() {
    let (_dir, dao_a, dao_b) = dao_pair().await;
    let key = "lockout:concurrent";

    let task_a = async {
        for _ in 0..50 {
            dao_a.incr(key, 3600).await.expect("incr a");
        }
    };
    let task_b = async {
        for _ in 0..50 {
            dao_b.incr(key, 3600).await.expect("incr b");
        }
    };
    tokio::join!(task_a, task_b);

    let raw = dao_a.get(key).await.expect("get").expect("value exists");
    assert_eq!(raw, "100", "no increments may be lost across replicas");
}
