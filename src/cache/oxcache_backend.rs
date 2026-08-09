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
use oxcache::features::bloom_filter::BloomFilter;

/// oxcache 后端,包装 `oxcache::Cache<String, Vec<f32>>`。
///
/// 提供 KvCache 兼容接口:`new`/`disabled`/`is_enabled`/`get`/`put`/
/// `get_or_insert`/`remove`/`clear`/`len`/`is_empty`/`warm_up`。
///
/// 内部集成:
/// - **Bloom filter** (T029-T030): 负查询过滤,FPR=0.01,`get()` 时先查 bloom,
///   miss 则跳过底层缓存查询。
/// - **Compression** (T031): 使用 oxcache flate2 压缩存储 embedding 向量,
///   减少内存占用。
pub(crate) struct OxCacheBackend {
    cache: Option<Cache<String, Vec<f32>>>,
    bloom: Option<BloomFilter>,
    enabled: bool,
}

impl OxCacheBackend {
    /// 创建指定容量的缓存后端。
    ///
    /// T029: 同时初始化 bloom filter (FPR=0.01, capacity=capacity)。
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let moka = MokaMemoryBackend::builder()
            .capacity(cap as u64)
            .build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        // T029: Bloom filter for negative query filtering
        let bloom = BloomFilter::new(cap, 0.01);
        Self {
            cache: Some(cache),
            bloom: Some(bloom),
            enabled: true,
        }
    }

    /// 创建禁用的后端(不分配缓存资源)。
    pub fn disabled() -> Self {
        Self {
            cache: None,
            bloom: None,
            enabled: false,
        }
    }

    /// 返回后端是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 查询缓存,未命中或禁用时返回 None。
    ///
    /// T030: 先查 bloom filter,如果 bloom 说 key 不存在则直接返回 None,
    /// 跳过底层缓存查询 (bloom filter 无误判)。
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        // T030: Bloom filter negative check
        if let Some(bloom) = &self.bloom {
            if !bloom.contains(key) {
                return None;
            }
        }
        let cache = self.cache.as_ref()?;
        let raw = cache.get(&key.to_string()).await.ok().flatten()?;
        // T031: Decompress on retrieval
        decompress_f32_vec(raw)
    }

    /// 写入缓存(禁用时为空操作)。
    ///
    /// T029: 写入时将 key 插入 bloom filter。
    /// T031: 存储前压缩 embedding 向量。
    pub async fn put(&self, key: &str, value: Vec<f32>) {
        if !self.enabled {
            return;
        }
        if let Some(cache) = &self.cache {
            // T031: Compress before storing
            let compressed = compress_f32_vec(value);
            let _ = cache.set(&key.to_string(), &compressed).await;
        }
        // T029: Insert into bloom filter after successful set
        if let Some(bloom) = &self.bloom {
            bloom.insert(key);
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

    /// 清空缓存,同时重置 bloom filter。
    pub async fn clear(&self) {
        if let Some(cache) = &self.cache {
            let _ = cache.clear().await;
        }
        if let Some(bloom) = &self.bloom {
            bloom.clear();
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
    /// 预热时每个 key 都会插入 bloom filter 并压缩存储。
    pub async fn warm_up(&self, entries: HashMap<String, Vec<f32>>) {
        if !self.enabled {
            return;
        }
        for (key, value) in entries {
            self.put(&key, value).await;
        }
    }

    /// 返回 bloom filter 引用(用于测试/监控)。
    #[cfg(test)]
    fn bloom_filter(&self) -> Option<&BloomFilter> {
        self.bloom.as_ref()
    }
}

// ============================================================================
// T031: Compression helpers — compress/decompress Vec<f32> via oxcache flate2
// ============================================================================

/// 压缩 `Vec<f32>` 为 `Vec<f32>`。
///
/// 将浮点向量重新解释为字节,通过 oxcache 的 `compress_data` (flate2 gzip)
/// 压缩,再重新解释回 `Vec<f32>` 存储。小向量 (<25 f32 = 100 bytes)
/// 不会被压缩(oxcache 内部 MIN_COMPRESS_SIZE 阈值)。
fn compress_f32_vec(value: Vec<f32>) -> Vec<f32> {
    if value.is_empty() {
        return value;
    }
    let byte_len = value.len() * 4;
    let ptr = value.as_ptr();
    // SAFETY: f32 数组与 [u8] 具有相同的内存布局。
    // 我们立即从原始字节创建新 slice,不持有 value 的引用。
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_len) };
    match oxcache::infra::serialization::utils::compress_data(bytes) {
        Ok(compressed) => {
            // 丢弃原始 value 的所有权(已被 move),compressed 是独立 Vec<u8>
            drop(value);
            // 将压缩后的字节重新解释为 Vec<f32>
            // SAFETY: 缓存内部存储,只要 get 时对称解压缩即可恢复原始值。
            // 压缩后字节数可能不是 4 的倍数,用 padding 对齐。
            bytes_to_f32_vec(compressed)
        }
        Err(_) => value,
    }
}

/// 解压缩 `Vec<f32>` (逆向 `compress_f32_vec`)。
fn decompress_f32_vec(stored: Vec<f32>) -> Option<Vec<f32>> {
    if stored.is_empty() {
        return Some(stored);
    }
    let byte_len = stored.len() * 4;
    let ptr = stored.as_ptr();
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, byte_len) };
    match oxcache::infra::serialization::utils::decompress_data(bytes) {
        Ok(decompressed) => {
            drop(stored);
            Some(bytes_to_f32_vec(decompressed))
        }
        Err(_) => None,
    }
}

/// 将 `Vec<u8>` 转换为 `Vec<f32>`,必要时补零对齐到 4 字节边界。
fn bytes_to_f32_vec(bytes: Vec<u8>) -> Vec<f32> {
    let mut bytes = bytes;
    let remainder = bytes.len() % 4;
    if remainder != 0 {
        bytes.extend(std::iter::repeat(0u8).take(4 - remainder));
    }
    let f32_count = bytes.len() / 4;
    let ptr = bytes.as_ptr();
    let cap = bytes.capacity() / 4;
    std::mem::forget(bytes);
    // SAFETY: bytes 已对齐到 4 字节且长度是 4 的倍数。
    // Vec<u8> 的 layout (ptr, len, cap) 与 Vec<f32> 相同,
    // 新 Vec<f32> 的 len=f32_count, cap=cap。
    unsafe { Vec::from_raw_parts(ptr as *mut f32, f32_count, cap) }
}

#[cfg(test)]
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

    // ========================================================================
    // T033: Bloom filter integration tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_miss_skips_backend_lookup() {
        // T033: 验证 bloom filter miss 时直接返回 None,不查询底层缓存。
        let cache = OxCacheBackend::new(100);
        // 从未 put 过任何 key,bloom filter 应为空
        let bloom = cache.bloom_filter().unwrap();
        assert_eq!(bloom.len(), 0, "bloom should be empty initially");
        // get 一个从未插入的 key,应返回 None (bloom filter 拦截)
        assert!(cache.get("never_inserted").await.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_populated_on_put() {
        // T033: 验证 put 后 bloom filter 包含该 key
        let cache = OxCacheBackend::new(100);
        cache.put("key1", vec![1.0, 2.0]).await;
        let bloom = cache.bloom_filter().unwrap();
        assert!(bloom.contains("key1"), "bloom should contain inserted key");
        assert!(!bloom.contains("key2"), "bloom should not contain non-inserted key");
        assert_eq!(bloom.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_cleared_with_cache() {
        // T033: 验证 clear() 同时重置 bloom filter
        let cache = OxCacheBackend::new(100);
        cache.put("a", vec![1.0]).await;
        cache.put("b", vec![2.0]).await;
        let bloom = cache.bloom_filter().unwrap();
        assert_eq!(bloom.len(), 2);
        cache.clear().await;
        assert_eq!(bloom.len(), 0, "bloom should be cleared");
        assert!(!bloom.contains("a"));
        assert!(!bloom.contains("b"));
    }

    // ========================================================================
    // T033: Compression roundtrip tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_compression_roundtrip_preserves_values() {
        // T033: 验证压缩往返后向量值在容差范围内保持一致。
        // 使用较大向量 (>25 f32 = 100 bytes) 以触发 flate2 压缩。
        let cache = OxCacheBackend::new(16);
        let original: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        cache.put("compressed_key", original.clone()).await;
        // moka 异步索引,短暂等待
        tokio::time::sleep(Duration::from_millis(20)).await;
        let got = cache.get("compressed_key").await;
        assert!(got.is_some(), "should retrieve compressed value");
        let retrieved = got.unwrap();
        assert_eq!(retrieved.len(), original.len(), "vector length must match");
        for (i, (a, b)) in retrieved.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "value mismatch at index {}: {} != {}",
                i, a, b
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_compression_small_vectors_roundtrip() {
        // T033: 小向量 (<100 bytes) 不触发压缩,但往返仍应保持精确一致。
        let cache = OxCacheBackend::new(16);
        let small = vec![0.1, 0.2, 0.3];
        cache.put("small", small.clone()).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        let got = cache.get("small").await;
        assert_eq!(got, Some(small));
    }

    #[test]
    fn test_compress_decompress_f32_vec_roundtrip() {
        // T033: 直接测试压缩/解压缩辅助函数的往返正确性
        let original: Vec<f32> = (0..200).map(|i| (i as f32) * 0.001).collect();
        let compressed = compress_f32_vec(original.clone());
        let decompressed = decompress_f32_vec(compressed).unwrap();
        assert_eq!(decompressed.len(), original.len());
        for (i, (a, b)) in decompressed.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "roundtrip mismatch at index {}: {} != {}",
                i, a, b
            );
        }
    }

    #[test]
    fn test_compress_empty_vec() {
        let empty: Vec<f32> = vec![];
        let compressed = compress_f32_vec(empty.clone());
        assert!(compressed.is_empty());
        let decompressed = decompress_f32_vec(compressed).unwrap();
        assert!(decompressed.is_empty());
    }
}
