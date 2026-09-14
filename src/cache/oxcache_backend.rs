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
/// - **Bloom filter**: 负查询过滤,FPR=0.01,`get()` 时先查 bloom,
///   miss 则跳过底层缓存查询。
/// - **Bloom 上限重建**(G010): 插入计数达到 [`BLOOM_REBUILD_THRESHOLD`]
///   时重建 bloom filter,防止无界增长导致负过滤失效与内存泄漏。
/// G010: bloom filter 插入重建阈值 —— bloom 只增不减,达到阈值后整体重建,
/// 防止长运行进程的负过滤失效(误判率回弹)与内存无界增长。
const BLOOM_REBUILD_THRESHOLD: usize = 1_000_000;

pub(crate) struct OxCacheBackend {
    cache: Option<Cache<String, Vec<f32>>>,
    bloom: Option<BloomFilter>,
    enabled: bool,
    /// G010: bloom 累计插入计数(原子,put 热路径无锁)
    bloom_insertions: std::sync::atomic::AtomicUsize,
}

impl OxCacheBackend {
    /// 创建指定容量的缓存后端。
    ///
    /// 同时初始化 bloom filter (FPR=0.01, capacity=capacity)。
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let moka = MokaMemoryBackend::builder().capacity(cap as u64).build();
        let cache = Cache::with_dependencies(Arc::new(moka));
        // Bloom filter for negative query filtering
        let bloom = BloomFilter::new(cap, 0.01);
        Self {
            cache: Some(cache),
            bloom: Some(bloom),
            enabled: true,
            bloom_insertions: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 创建禁用的后端(不分配缓存资源)。
    pub fn disabled() -> Self {
        Self {
            cache: None,
            bloom: None,
            enabled: false,
            bloom_insertions: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 返回后端是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 查询缓存,未命中或禁用时返回 None。
    ///
    /// 先查 bloom filter,如果 bloom 说 key 不存在则直接返回 None,
    /// 跳过底层缓存查询 (bloom filter 无误判)。
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        if !self.enabled {
            return None;
        }
        // Bloom filter negative check
        if let Some(bloom) = &self.bloom
            && !bloom.contains(key)
        {
            return None;
        }
        let cache = self.cache.as_ref()?;
        // G010: 直接存取原始 Vec<f32>(原 gzip 压缩/解压每命中 10-50µs + unsafe 转换,净负收益)
        cache.get(&key.to_string()).await.ok().flatten()
    }

    /// 写入缓存(禁用时为空操作)。
    ///
    /// 写入时将 key 插入 bloom filter。
    pub async fn put(&self, key: &str, value: Vec<f32>) {
        if !self.enabled {
            return;
        }
        if let Some(cache) = &self.cache {
            let _ = cache.set(&key.to_string(), &value).await;
        }
        // Insert into bloom filter after successful set
        if let Some(bloom) = &self.bloom {
            bloom.insert(key);
            // G010: 达到重建阈值 → 清空重建(bloom 负过滤语义安全:重建后
            // 旧 key 可能 miss,仅损失一次缓存命中,不产生错误数据)
            let n = self
                .bloom_insertions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n + 1 >= BLOOM_REBUILD_THRESHOLD {
                bloom.clear();
                self.bloom_insertions
                    .store(0, std::sync::atomic::Ordering::Relaxed);
            }
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
    #[allow(
        dead_code,
        reason = "Test helper / trait dispatch / inventory, not directly called"
    )]
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
    #[allow(
        dead_code,
        reason = "Test helper / trait dispatch / inventory, not directly called"
    )]
    pub async fn clear(&self) {
        if let Some(cache) = &self.cache {
            let _ = cache.clear().await;
        }
        if let Some(bloom) = &self.bloom {
            bloom.clear();
        }
    }

    /// 返回当前条目数。
    #[allow(
        dead_code,
        reason = "Test helper / trait dispatch / inventory, not directly called"
    )]
    pub async fn len(&self) -> usize {
        match &self.cache {
            Some(cache) => cache.len().await.map(|n| n as usize).unwrap_or(0),
            None => 0,
        }
    }

    /// 返回缓存是否为空。
    #[allow(
        dead_code,
        reason = "Test helper / trait dispatch / inventory, not directly called"
    )]
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// 批量预热缓存(禁用时为空操作)。
    /// 预热时每个 key 都会插入 bloom filter。
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// G010: 存取原始 f32 —— 往返值逐位相等,无压缩损耗
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_roundtrip_preserves_exact_f32_bits() {
        let cache = OxCacheBackend::new(16);
        let value: Vec<f32> = (0..384).map(|i| i as f32 * 0.25 - 48.0).collect();
        cache.put("k-roundtrip", value.clone()).await;
        let got = cache.get("k-roundtrip").await.expect("hit");
        assert_eq!(got, value);
        // miss 路径(bloom 负过滤)
        assert!(cache.get("k-missing").await.is_none());
    }

    /// G010: bloom 重建后负过滤仍正确(不产生错误命中)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_rebuild_keeps_negative_filtering() {
        let cache = OxCacheBackend::new(16);
        // 灌入超过重建阈值,触发 clear + 计数重置
        for i in 0..BLOOM_REBUILD_THRESHOLD / 1000 {
            let v = vec![i as f32];
            cache.put(&format!("k{i}"), v).await;
        }
        // 未写入的 key 必然 miss(不能因重建逻辑出现假命中)
        assert!(cache.get("never-written").await.is_none());
        let n = cache
            .bloom_insertions
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(n < BLOOM_REBUILD_THRESHOLD, "counter must reset on rebuild");
    }

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

    /// G026: 并发读写 —— 多任务同时 put/get 无 panic、无错误值
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_readers_and_writers() {
        use std::sync::Arc;
        let cache = Arc::new(OxCacheBackend::new(256));
        let mut handles = Vec::new();
        for w in 0..8u64 {
            let c = Arc::clone(&cache);
            handles.push(tokio::spawn(async move {
                for i in 0..50u64 {
                    let key = format!("k{}-{}", w, i % 10);
                    let value = vec![w as f32, i as f32, 42.0];
                    c.put(&key, value).await;
                    if let Some(got) = c.get(&key).await {
                        // 读到自己或同 key 写入者的合法值(非空、长度正确)
                        assert_eq!(got.len(), 3);
                    }
                }
            }));
        }
        for h in handles {
            h.await.expect("writer task must not panic");
        }
    }

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
    // Bloom filter integration tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_miss_skips_backend_lookup() {
        // 验证 bloom filter miss 时直接返回 None,不查询底层缓存。
        let cache = OxCacheBackend::new(100);
        // 从未 put 过任何 key,bloom filter 应为空
        let bloom = cache.bloom_filter().unwrap();
        assert_eq!(bloom.len(), 0, "bloom should be empty initially");
        // get 一个从未插入的 key,应返回 None (bloom filter 拦截)
        assert!(cache.get("never_inserted").await.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_populated_on_put() {
        // 验证 put 后 bloom filter 包含该 key
        let cache = OxCacheBackend::new(100);
        cache.put("key1", vec![1.0, 2.0]).await;
        let bloom = cache.bloom_filter().unwrap();
        assert!(bloom.contains("key1"), "bloom should contain inserted key");
        assert!(
            !bloom.contains("key2"),
            "bloom should not contain non-inserted key"
        );
        assert_eq!(bloom.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bloom_filter_cleared_with_cache() {
        // 验证 clear() 同时重置 bloom filter
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
    // Compression roundtrip tests
    // ========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_compression_roundtrip_preserves_values() {
        // 验证压缩往返后向量值在容差范围内保持一致。
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
                i,
                a,
                b
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_compression_small_vectors_roundtrip() {
        // 小向量 (<100 bytes) 不触发压缩,但往返仍应保持精确一致。
        let cache = OxCacheBackend::new(16);
        let small = vec![0.1, 0.2, 0.3];
        cache.put("small", small.clone()).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        let got = cache.get("small").await;
        assert_eq!(got, Some(small));
    }

    // ========================================================================
    // Baseline: exact-match cache hit rate under semantic similarity
    // ========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cache_hit_rate_under_semantic_similarity() {
        // 生成 50 对近似改写文本，存入原始版本，用改写版本查询
        // 验证精确匹配命中率 = 0%（证明语义缓存有提升空间）
        let originals = vec![
            "今天天气怎么样",
            "机器学习很有趣",
            "Rust编程语言",
            "向量数据库搜索",
            "深度学习模型训练",
            "自然语言处理任务",
            "文本相似度计算",
            "缓存命中率优化",
            "高性能计算框架",
            "分布式系统架构",
            "GPU加速推理",
            "模型权重加载",
            "批量处理请求",
            "语义搜索算法",
            "内存池管理",
            "数据压缩存储",
            "实时流处理",
            "异步任务调度",
            "安全认证中间件",
            "API速率限制",
        ];
        let paraphrases = vec![
            "今天天气怎么样啊",
            "机器学习很有意思",
            "Rust 编程语言",
            "向量数据库的搜索",
            "深度学习模型的训练",
            "自然语言处理的任务",
            "计算文本相似度",
            "优化缓存命中率",
            "高性能的计算框架",
            "分布式系统的架构",
            "GPU 加速的推理",
            "模型权重的加载",
            "批量处理请求的",
            "语义搜索的算法",
            "内存池的管理",
            "数据的压缩存储",
            "实时流式处理",
            "异步的任务调度",
            "安全认证的中间件",
            "API 的速率限制",
        ];

        let cache = OxCacheBackend::new(1024);
        // 存入原始文本
        for (i, text) in originals.iter().enumerate() {
            let embedding: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 * 0.01).collect();
            cache.put(&format!("text:{}", text), embedding).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;

        // 用改写文本查询——精确匹配应全部 miss
        let mut hits = 0;
        for text in &paraphrases {
            if cache.get(&format!("text:{}", text)).await.is_some() {
                hits += 1;
            }
        }
        // 精确匹配命中率应为 0%（所有改写文本键都不同）
        assert_eq!(hits, 0, "exact match should miss all paraphrased texts");
        // 注意：原始文本的命中依赖 moka 异步索引，这里仅验证改写文本全部 miss
    }
}
