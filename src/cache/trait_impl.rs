// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information

//! CacheGetOrInsert trait 的默认实现

use super::{Cache, CacheGetOrInsert};
use std::hash::Hash;

/// 所有缓存实现默认实现 get_or_insert
#[async_trait::async_trait]
impl<K, V, C> CacheGetOrInsert<K, V> for C
where
    K: Hash + Eq + Send + Sync + std::fmt::Debug + Clone + 'static,
    V: Clone + Send + Sync + 'static,
    C: Cache<K, V>,
{
    async fn get_or_insert(
        &self,
        key: K,
        value: V,
        size: usize,
    ) -> Result<V, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(v) = self.get(&key).await {
            return Ok(v);
        }
        self.put(key.clone(), value, size).await?;
        Ok(self.get(&key).await.ok_or("Value not found after insert")?)
    }
}
