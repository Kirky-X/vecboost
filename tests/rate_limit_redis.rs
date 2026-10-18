// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! Redis 跨副本限流回归（需 `distributed-redis` + 可用 Redis）。
//!
//! 缺少 `VECBOOST_TEST_REDIS_URL` 时整组 SKIP（与仓内资产型测试同一语义）。

#![cfg(feature = "distributed-redis")]

use limiteron::matchers::RequestContext;
use vecboost::config::app::RateLimitBackend;
use vecboost::rate_limit::{LimiteronAdapter, RateLimitSettings};

fn redis_url() -> Option<String> {
    match std::env::var("VECBOOST_TEST_REDIS_URL") {
        Ok(url) if !url.trim().is_empty() => Some(url),
        _ => {
            eprintln!("SKIP: VECBOOST_TEST_REDIS_URL not set");
            None
        }
    }
}

fn unique_key() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("test-key-{nanos}")
}

fn ctx(key: &str) -> RequestContext {
    RequestContext {
        client_ip: Some(key.to_string()),
        path: "/test".to_string(),
        method: "GET".to_string(),
        ..Default::default()
    }
}

#[tokio::test]
async fn two_adapters_share_per_key_quota() {
    let Some(url) = redis_url() else {
        return;
    };
    let settings = RateLimitSettings {
        global_requests_per_minute: 0,
        ip_requests_per_minute: 2,
        user_requests_per_minute: 0,
        api_key_requests_per_minute: 0,
        backend: RateLimitBackend::Redis,
        redis_url: Some(url),
        global_window_ms: 60_000,
    };
    let a = LimiteronAdapter::try_new(settings.clone())
        .await
        .expect("adapter a");
    let b = LimiteronAdapter::try_new(settings)
        .await
        .expect("adapter b");

    let key = unique_key();
    let context = ctx(&key);
    assert!(a.check_rate_limit(&context).await, "first request allowed");
    assert!(a.check_rate_limit(&context).await, "second request allowed");
    assert!(
        !b.check_rate_limit(&context).await,
        "replica B must observe quota exhausted by replica A"
    );
}

async fn clear_global_window(url: &str) {
    use oxcache::redis::AsyncCommands;
    let client = oxcache::redis::Client::open(url.to_string()).expect("redis client");
    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .expect("redis connection");
    let _: i64 = conn
        .del("vecboost:rl:global")
        .await
        .expect("delete global window");
}

#[tokio::test]
async fn two_adapters_share_global_quota() {
    let Some(url) = redis_url() else {
        return;
    };
    clear_global_window(&url).await;
    let settings = RateLimitSettings {
        global_requests_per_minute: 2,
        ip_requests_per_minute: 0,
        user_requests_per_minute: 0,
        api_key_requests_per_minute: 0,
        backend: RateLimitBackend::Redis,
        redis_url: Some(url),
        global_window_ms: 60_000,
    };
    let a = LimiteronAdapter::try_new(settings.clone())
        .await
        .expect("adapter a");
    let b = LimiteronAdapter::try_new(settings)
        .await
        .expect("adapter b");
    let context = ctx(&unique_key());
    assert!(a.check_rate_limit(&context).await, "first global slot");
    assert!(a.check_rate_limit(&context).await, "second global slot");
    assert!(
        !b.check_rate_limit(&context).await,
        "replica B must observe the shared global window exhausted by replica A"
    );
}
