// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

// oxcache 后端(oxcache 必选,完全接管缓存)
pub(crate) mod oxcache_backend;
pub(crate) mod persist;
pub(crate) mod semantic_cache;

pub(crate) use oxcache_backend::OxCacheBackend;
pub use semantic_cache::{ComparisonMode, SemanticCache, SemanticCacheConfig, SemanticCacheStats};
