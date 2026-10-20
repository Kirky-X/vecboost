// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 双实例语义缓存共享：实例 A 写入，实例 B 模糊命中。
//!
//! 需要真实 Redis（`VECBOOST_TEST_REDIS_URL`，默认 redis://127.0.0.1:6379）；
//! 不可达时显式跳过，不以 mock 替代。

#![cfg(feature = "distributed-redis")]

use std::time::Duration;

use vecboost::SemanticCache;

fn redis_url() -> String {
    std::env::var("VECBOOST_TEST_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
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
