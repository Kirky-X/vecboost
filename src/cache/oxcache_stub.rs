// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! oxcache feature 关闭时的 no-op stub。
//!
//! 提供与 `oxcache_backend::OxCacheBackend` 相同的方法签名,
//! 但不持有任何缓存资源——所有操作为空操作,`is_enabled` 恒为 `false`。
//! `get_or_insert` 仍会调用 fallback 以保证语义正确(无缓存时必须计算)。

use std::collections::HashMap;

pub(crate) struct OxCacheBackend;

impl OxCacheBackend {
    pub fn new(_capacity: usize) -> Self {
        Self::disabled()
    }

    pub fn disabled() -> Self {
        Self
    }

    pub fn is_enabled(&self) -> bool {
        false
    }

    pub async fn get(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }

    pub async fn put(&self, _key: &str, _value: Vec<f32>) {}

    pub async fn get_or_insert<F, Fut, E>(&self, _key: &str, f: F) -> Result<Vec<f32>, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, E>>,
    {
        f().await
    }

    pub async fn remove(&self, _key: &str) -> bool {
        false
    }

    pub async fn clear(&self) {}

    pub async fn len(&self) -> usize {
        0
    }

    pub async fn is_empty(&self) -> bool {
        true
    }

    pub async fn warm_up(&self, _entries: HashMap<String, Vec<f32>>) {}
}
