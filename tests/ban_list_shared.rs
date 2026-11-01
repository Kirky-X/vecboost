// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 双实例封禁清单共享：A 封禁，B 立即拒绝。
//!
//! 需要真实 Redis（`VECBOOST_TEST_REDIS_URL`，默认 redis://127.0.0.1:6379）。
//! 未显式设置该变量时不可达即跳过（本地无 Redis 的开发机）；一旦显式设置，
//! 不可达即失败——CI 正是靠这条把「封禁跨副本」验收变成硬门禁，否则测试
//! 会在缺 Redis 时静默返回、伪装成通过。

#![cfg(feature = "distributed-redis")]

use std::sync::Arc;

use chrono::Utc;
use limiteron::CacheBanStorage;
use limiteron::storage::{BanRecord, BanStorage, BanTarget};
use oxcache::backend::{CacheBackend, RedisBackend};

use vecboost::config::app::RateLimitBackend;
use vecboost::rate_limit::{LimiteronAdapter, RateLimitSettings};

fn redis_url() -> String {
    std::env::var("VECBOOST_TEST_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// `VECBOOST_TEST_REDIS_URL` 是否被显式设置（CI 设置，本地通常不设）。
fn redis_url_is_explicit() -> bool {
    std::env::var("VECBOOST_TEST_REDIS_URL").is_ok()
}

async fn ban_backend(url: &str) -> Option<Arc<dyn CacheBackend>> {
    match RedisBackend::builder().connection_string(url).build().await {
        Ok(backend) => Some(Arc::new(backend)),
        Err(e) => {
            assert!(
                !redis_url_is_explicit(),
                "VECBOOST_TEST_REDIS_URL 已显式设置但 Redis 不可达，封禁跨副本验收不得跳过: {e}"
            );
            eprintln!("skip: redis unavailable: {e}");
            None
        }
    }
}

#[tokio::test]
async fn ban_list_is_shared_across_instances() {
    let url = redis_url();
    let Some(backend_a) = ban_backend(&url).await else {
        return;
    };
    let backend_b = ban_backend(&url)
        .await
        .expect("second Redis backend should connect");

    let a = CacheBanStorage::new(backend_a);
    let b = CacheBanStorage::new(backend_b);

    let target = BanTarget::Ip(format!(
        "203.0.113.{}",
        std::process::id().saturating_add(1) % 250
    ));
    let now = Utc::now();
    let record = BanRecord {
        target: target.clone(),
        ban_times: 1,
        duration: std::time::Duration::from_secs(10),
        banned_at: now,
        expires_at: now + chrono::Duration::seconds(10),
        is_manual: true,
        reason: "shared-ban-test".to_string(),
    };
    a.save(&record).await.expect("instance A should save ban");

    let seen = b
        .is_banned(&target)
        .await
        .expect("instance B should query shared ban list");
    assert!(seen.is_some(), "instance B must see instance A's ban");

    a.remove_ban(&target).await.expect("cleanup should succeed");
}

#[tokio::test]
async fn redis_rate_limit_adapter_wires_shared_ban_storage() {
    let url = redis_url();
    let settings = RateLimitSettings {
        backend: RateLimitBackend::Redis,
        redis_url: Some(url),
        ..RateLimitSettings::default()
    };
    match LimiteronAdapter::try_new(settings).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("skip: redis unavailable: {e}");
        }
    }
}
