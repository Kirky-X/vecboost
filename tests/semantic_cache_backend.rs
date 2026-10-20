// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

//! 语义缓存后端启动校验。

use vecboost::config::app::{CacheBackendKind, SemanticCacheConfig};

#[test]
fn memory_backend_is_always_valid() {
    let config = SemanticCacheConfig::default();
    assert_eq!(config.backend, CacheBackendKind::Memory);
    assert!(config.validate_backend().is_ok());
}

#[test]
fn redis_backend_requires_url() {
    let mut config = SemanticCacheConfig {
        enabled: true,
        backend: CacheBackendKind::Redis,
        ..SemanticCacheConfig::default()
    };

    // 无 URL：无论 feature 是否启用都必须拒绝（feature 缺失时更早失败）。
    assert!(config.validate_backend().is_err());

    #[cfg(feature = "distributed-redis")]
    {
        config.redis_url = Some("redis://127.0.0.1:6379".to_string());
        assert!(config.validate_backend().is_ok());
    }
}
