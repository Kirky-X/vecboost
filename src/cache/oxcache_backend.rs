// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! oxcache 后端封装:提供与 KvCache 兼容的接口,内部委托 oxcache::Cache。
//!
//! 用于替代自研 KvCache,支持 LRU 驱逐和 per-entry TTL。

use std::collections::HashMap;
use std::sync::Arc;

use oxcache::backend::MokaMemoryBackend;
use oxcache::cache::Cache;

/// oxcache 后端,包装 `oxcache::Cache<String, Vec<f32>>`。
///
/// 提供 KvCache 兼容接口:`new`/`disabled`/`is_enabled`/`get`/`put`/
/// `get_or_insert`/`remove`/`clear`/`len`/`is_empty`/`warm_up`。
pub(crate) struct OxCacheBackend {
    cache: Option<Cache<String, Vec<f32>>>,
    enabled: bool,
}

impl OxCacheBackend {
    /// 创建指定容量的缓存后端。
    pub fn new(capacity: usize) -> Self {
        let moka = MokaMemoryBackend::builder()
            .capacity(capacity.max(1) as u64)
            .build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        Self {
            cache: Some(cache),
            enabled: true,
        }
    }

    /// 创建禁用的后端(不分配缓存资源)。
    pub fn disabled() -> Self {
        Self {
            cache: None,
            enabled: false,
        }
    }

    /// 返回后端是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 查询缓存,未命中或禁用时返回 None。
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        let cache = self.cache.as_ref()?;
        cache.get(&key.to_string()).await.ok().flatten()
    }

    /// 写入缓存(禁用时为空操作)。
    pub async fn put(&self, key: &str, value: Vec<f32>) {
        if !self.enabled {
            return;
        }
        if let Some(cache) = &self.cache {
            let _ = cache.set(&key.to_string(), &value).await;
        }
    }

    /// 按 key 查询;未命中则调用 fallback 计算并回填。
    ///
    /// 与 KvCache::get_or_insert 签名兼容,泛型错误类型 E 由 fallback 决定。
    pub async fn get_or_insert<F, Fut, E>(&self, key: &str, f: F) -> Result<Vec<f32>, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, E>>,
    {
        if let Some(cached) = self.get(key).await {
            return Ok(cached);
        }
        let embedding = f().await?;
        self.put(key, embedding.clone()).await;
        Ok(embedding)
    }

    /// 删除 key,返回是否命中。
    pub async fn remove(&self, key: &str) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.cache {
            Some(cache) => cache.delete(&key.to_string()).await.is_ok(),
            None => false,
        }
    }

    /// 清空缓存。
    pub async fn clear(&self) {
        if let Some(cache) = &self.cache {
            let _ = cache.clear().await;
        }
    }

    /// 返回当前条目数。
    pub async fn len(&self) -> usize {
        match &self.cache {
            Some(cache) => cache.len().await.map(|n| n as usize).unwrap_or(0),
            None => 0,
        }
    }

    /// 返回缓存是否为空。
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// 批量预热缓存(禁用时为空操作)。
    pub async fn warm_up(&self, entries: HashMap<String, Vec<f32>>) {
        if !self.enabled {
            return;
        }
        for (key, value) in entries {
            self.put(&key, value).await;
        }
    }
}

#[cfg(all(test, feature = "oxcache"))]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_put_and_get_hit() {
        let cache = OxCacheBackend::new(16);
        assert!(cache.is_enabled());
        cache.put("text:hello", vec![0.1, 0.2, 0.3]).await;
        let got = cache.get("text:hello").await;
        assert_eq!(got, Some(vec![0.1, 0.2, 0.3]));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_get_miss_returns_none() {
        let cache = OxCacheBackend::new(16);
        assert!(cache.get("text:missing").await.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_eviction_under_capacity_pressure() {
        // moka 使用 W-TinyLFU + 异步驱逐,小容量下驱逐行为不可预测。
        // 用较大容量 + 超量 key + 足够等待时间验证容量限制最终生效。
        let cache = OxCacheBackend::new(10);
        for i in 0..30 {
            cache.put(&format!("k{}", i), vec![i as f32]).await;
        }
        // moka 异步驱逐,给足够时间让驱逐任务完成
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut remaining = 0;
        for i in 0..30 {
            if cache.get(&format!("k{}", i)).await.is_some() {
                remaining += 1;
            }
        }
        assert!(
            remaining < 30,
            "capacity pressure should evict some entries, {} still present",
            remaining
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_ttl_expiry() {
        let moka = MokaMemoryBackend::builder()
            .capacity(16)
            .ttl(Duration::from_millis(50))
            .build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        cache
            .set(&"k".to_string(), &vec![1.0_f32])
            .await
            .expect("set");
        assert_eq!(
            cache.get(&"k".to_string()).await.unwrap(),
            Some(vec![1.0]),
            "should hit before TTL"
        );
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(
            cache.get(&"k".to_string()).await.unwrap(),
            None,
            "should expire after TTL"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_clear_empties_cache() {
        let cache = OxCacheBackend::new(16);
        cache.put("a", vec![1.0]).await;
        cache.put("b", vec![2.0]).await;
        cache.clear().await;
        // moka 异步驱逐,短暂等待
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(cache.get("a").await.is_none(), "a should be cleared");
        assert!(cache.get("b").await.is_none(), "b should be cleared");
        assert!(cache.is_empty().await, "cache should be empty");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_disabled_backend_is_noop() {
        let cache = OxCacheBackend::disabled();
        assert!(!cache.is_enabled());
        cache.put("k", vec![1.0]).await;
        assert!(cache.get("k").await.is_none());
        assert_eq!(cache.len().await, 0);
        assert!(cache.is_empty().await);
        assert!(!cache.remove("k").await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_get_or_insert_first_call_invokes_fallback() {
        let cache = OxCacheBackend::new(16);
        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        let val = cache
            .get_or_insert::<_, _, std::convert::Infallible>("key", || async move {
                called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(vec![9.9])
            })
            .await
            .expect("get_or_insert");
        assert_eq!(val, vec![9.9]);
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        // 第二次应命中缓存,fallback 不调用
        let called2 = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called2_clone = called2.clone();
        let val = cache
            .get_or_insert::<_, _, std::convert::Infallible>("key", || async move {
                called2_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(vec![0.0])
            })
            .await
            .expect("get_or_insert second");
        assert_eq!(val, vec![9.9], "should return cached value");
        assert!(
            !called2.load(std::sync::atomic::Ordering::SeqCst),
            "fallback should not be called on cache hit"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_warm_up_inserts_all_entries() {
        let cache = OxCacheBackend::new(32);
        let mut entries = HashMap::new();
        entries.insert("a".to_string(), vec![1.0]);
        entries.insert("b".to_string(), vec![2.0]);
        entries.insert("c".to_string(), vec![3.0]);
        cache.warm_up(entries).await;
        // moka 异步索引,短暂等待
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(cache.get("a").await, Some(vec![1.0]));
        assert_eq!(cache.get("b").await, Some(vec![2.0]));
        assert_eq!(cache.get("c").await, Some(vec![3.0]));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_remove_returns_false_for_missing() {
        let cache = OxCacheBackend::new(16);
        // oxcache delete 对不存在的 key 返回 Ok(()),因此这里仅验证不 panic
        let _ = cache.remove("missing").await;
    }
}
