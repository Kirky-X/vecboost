// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 双实例语义缓存共享：实例 A 写入，实例 B 模糊命中。
//!
//! 需要真实 Redis（`VECBOOST_TEST_REDIS_URL`，默认 redis://127.0.0.1:6379）。
//! 未显式设置该变量时不可达即跳过（本地无 Redis 的开发机）；一旦显式设置，
//! 不可达即失败——CI 正是靠这条把「模糊命中跨副本」验收变成硬门禁。

#![cfg(feature = "distributed-redis")]

use std::time::Duration;

use vecboost::SemanticCache;

fn redis_url() -> String {
    std::env::var("VECBOOST_TEST_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// `VECBOOST_TEST_REDIS_URL` 是否被显式设置（CI 设置，本地通常不设）。
fn redis_url_is_explicit() -> bool {
    std::env::var("VECBOOST_TEST_REDIS_URL").is_ok()
}

#[tokio::test]
async fn shared_trigram_index_hits_across_instances() {
    let url = redis_url();
    let model = format!("test-semantic-{}", std::process::id());

    let a = match SemanticCache::with_capacity(0.5, 64)
        .with_redis_index_url(&url)
        .await
    {
        Ok(cache) => cache,
        Err(e) => {
            assert!(
                !redis_url_is_explicit(),
                "VECBOOST_TEST_REDIS_URL 已显式设置但 Redis 不可达，模糊命中跨副本验收不得跳过: {e}"
            );
            eprintln!("skip: redis unavailable: {e}");
            return;
        }
    };
    let b = SemanticCache::with_capacity(0.5, 64)
        .with_redis_index_url(&url)
        .await
        .expect("second instance should connect");

    eprintln!("A: get_or_compute start");
    let stored = tokio::time::timeout(
        Duration::from_secs(15),
        a.get_or_compute(&model, "hello world", || async {
            Ok::<Vec<f32>, vecboost::error::VecboostError>(vec![1.0, 0.0, 0.0])
        }),
    )
    .await
    .expect("A get_or_compute timed out")
    .expect("first instance should compute and store");
    assert_eq!(stored, vec![1.0, 0.0, 0.0]);
    eprintln!("A: get_or_compute done");

    // 实例 B 的本地索引为空，必须经 Redis 共享索引命中。
    eprintln!("B: find_similar start");
    let hit = tokio::time::timeout(
        Duration::from_secs(15),
        b.find_similar(&model, "hello worlds"),
    )
    .await
    .expect("B find_similar timed out");
    assert!(hit.is_some(), "instance B should hit shared trigram index");
    eprintln!("B: find_similar done");

    a.clear_model(&model).await;
}
