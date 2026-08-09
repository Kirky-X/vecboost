// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information

// oxcache 后端(oxcache 必选,完全接管缓存)
pub(crate) mod oxcache_backend;
pub(crate) mod semantic_cache;

pub(crate) use oxcache_backend::OxCacheBackend;
pub use semantic_cache::{SemanticCache, SemanticCacheConfig, SemanticCacheStats};
