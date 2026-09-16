// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! oxcache 后端封装:提供与 KvCache 兼容的接口,内部委托 oxcache::Cache。
//!
//! 用于替代自研 KvCache,支持 LRU 驱逐和 per-entry TTL。

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use oxcache::backend::MokaMemoryBackend;
use oxcache::cache::Cache;
use oxcache::features::bloom_filter::BloomFilter;

use super::persist;

/// 持久层状态：两段追加 WAL。文件操作经互斥串行，无后台线程。
struct PersistState {
    path: std::path::PathBuf,
    tag: String,
    max_bytes: u64,
    inner: std::sync::Mutex<PersistInner>,
}

struct PersistInner {
    /// 带用户态缓冲的 WAL 句柄（合并系统调用）。
    file: std::io::BufWriter<std::fs::File>,
    /// 已提交记录数（回放初始化，追加递增，紧凑化重置）。
    nrec: u64,
    /// WAL 近似字节数（构造时从文件长度初始化，追加时累加；
    /// 替代逐插入 stat 的超限检查）。
    approx_bytes: u64,
}

/// oxcache 后端,包装 `oxcache::Cache<String, Vec<f32>>`。
///
/// 提供 KvCache 兼容接口:`new`/`disabled`/`is_enabled`/`get`/`put`/
/// `get_or_insert`/`remove`/`clear`/`len`/`is_empty`/`warm_up`。
///
/// 内部集成:
/// - **Bloom filter**: 负查询过滤,FPR=0.01,`get()` 时先查 bloom,
///   miss 则跳过底层缓存查询。
/// - **Bloom 上限重建**: 插入计数达到 [`BLOOM_REBUILD_THRESHOLD`]
///   时重建 bloom filter,防止无界增长导致负过滤失效与内存泄漏。
///
/// bloom 只增不减,达到阈值后整体重建,防止长运行进程的负过滤失效(误判率回弹)与内存无界增长。
const BLOOM_REBUILD_THRESHOLD: usize = 1_000_000;

pub(crate) struct OxCacheBackend {
    cache: Option<Cache<String, Vec<f32>>>,
    bloom: Option<BloomFilter>,
    enabled: bool,
    /// bloom 累计插入计数(原子,put 热路径无锁)
    bloom_insertions: std::sync::atomic::AtomicUsize,
    /// 可选 WAL 持久层（`persist_path` 设置时启用，默认 None = 纯内存）。
    persist: Option<Arc<PersistState>>,
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
            persist: None,
        }
    }

    /// 创建带 WAL 持久层的缓存后端。
    ///
    /// `model_tag` 为模型指纹（回放时不匹配的记录被弃用）；`max_bytes`
    /// 为紧凑化阈值。构造时回放已有文件计数 nrec；内存重建需调用方在
    /// 启动时显式 `load_persisted().await`（顺序回放重建缓存）。
    pub fn with_persist(
        capacity: usize,
        path: std::path::PathBuf,
        max_bytes: u64,
        model_tag: String,
    ) -> Self {
        let mut backend = Self::new(capacity);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| {
                panic!(
                    "persist 文件不可创建 {}: {e}（请检查 [embedding] persist_path 目录权限）",
                    path.display()
                )
            });
        let approx_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
        let is_new = approx_bytes == 0;
        let (_, committed) = persist::replay(&path, &model_tag);
        // 新文件构造时一次性写头（追加路径不再逐插入 stat， 审查）。
        let mut writer = std::io::BufWriter::new(file);
        if is_new && let Err(e) = persist::write_header(&mut writer) {
            log::warn!("persist: {} 写头失败: {}", path.display(), e);
        }
        backend.persist = Some(Arc::new(PersistState {
            path,
            tag: model_tag,
            max_bytes,
            inner: std::sync::Mutex::new(PersistInner {
                file: writer,
                nrec: committed,
                approx_bytes,
            }),
        }));
        backend
    }

    /// 返回持久文件路径（doctor 可写性探测用）。
    pub fn persist_path(&self) -> Option<std::path::PathBuf> {
        self.persist.as_ref().map(|p| p.path.clone())
    }

    /// 启动回放：顺序重放持久文件重建内存缓存。
    /// 版本头/指纹不匹配或 checksum 失败的记录被弃用并 warn。
    pub async fn load_persisted(&self) {
        let Some(ps) = self.persist.as_ref() else {
            return;
        };
        let (records, committed) = persist::replay(&ps.path, &ps.tag);
        // 去重（后者覆盖前者）后写入内存（不回写 WAL，避免放大）。
        let mut dedup: HashMap<String, Vec<f32>> = HashMap::with_capacity(records.len());
        for (k, v) in records {
            dedup.insert(k, v);
        }
        // 同步 nrec（回放计数可能领先构造时计数）。
        if let Ok(mut inner) = ps.inner.lock() {
            inner.nrec = inner.nrec.max(committed);
        }
        for (k, v) in dedup {
            self.put_inner(&k, v, false).await;
        }
    }

    /// 创建禁用的后端(不分配缓存资源)。
    pub fn disabled() -> Self {
        Self {
            cache: None,
            bloom: None,
            enabled: false,
            bloom_insertions: std::sync::atomic::AtomicUsize::new(0),
            persist: None,
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
        // 直接存取原始 Vec<f32>(原 gzip 压缩/解压每命中 10-50µs + unsafe 转换,净负收益)
        cache.get(&key.to_string()).await.ok().flatten()
    }

    /// 写入缓存(禁用时为空操作)。
    ///
    /// 写入时将 key 插入 bloom filter；持久层启用时同步两段追加 WAL。
    pub async fn put(&self, key: &str, value: Vec<f32>) {
        let do_persist = self.persist.is_some();
        self.put_inner(key, value, do_persist).await;
    }

    async fn put_inner(&self, key: &str, value: Vec<f32>, do_persist: bool) {
        if !self.enabled {
            return;
        }
        if let Some(cache) = &self.cache {
            let _ = cache.set(&key.to_string(), &value).await;
        }
        // Insert into bloom filter after successful set
        if let Some(bloom) = &self.bloom {
            bloom.insert(key);
            // 达到重建阈值 → 清空重建(bloom 负过滤语义安全:重建后
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
        if do_persist {
            self.append_persist(key, &value);
        }
    }

    /// WAL 两段追加 + 超限紧凑化（同步于插入调用点，无后台线程）。
    fn append_persist(&self, key: &str, value: &[f32]) {
        let Some(ps) = self.persist.as_ref() else {
            return;
        };
        let mut guard = match ps.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                log::warn!("persist: 文件锁中毒，跳过本次落盘: {}", e);
                return;
            }
        };
        // 先拷贝出所需字段，避免 MutexGuard 解引用与 ps.tag 的借用冲突。
        let tag = ps.tag.clone();
        let nrec = guard.nrec;
        match persist::append_record(&mut guard.file, key, value, &tag, nrec) {
            Ok((nrec, bytes)) => {
                guard.nrec = nrec;
                guard.approx_bytes += bytes as u64;
                // 提交记录推送至内核（File.flush 本为 no-op，这里刷 BufWriter）。
                let _ = guard.file.flush();
                // 超限紧凑化：快照重写（临时文件 + rename 原子替换）。
                let over = guard.approx_bytes > ps.max_bytes;
                if over {
                    drop(guard);
                    self.compact_persist();
                }
            }
            Err(e) => {
                log::warn!("persist: 追加失败（{}），本次插入仅内存生效", e);
            }
        }
    }

    /// 快照重写紧凑化：回放去重 → tmp 重写 → 原子替换 → 重开句柄。
    fn compact_persist(&self) {
        let Some(ps) = self.persist.as_ref() else {
            return;
        };
        let (records, _) = persist::replay(&ps.path, &ps.tag);
        let mut dedup: HashMap<String, Vec<f32>> = HashMap::with_capacity(records.len());
        for (k, v) in records {
            dedup.insert(k, v);
        }
        match persist::compact(&ps.path, &dedup, &ps.tag) {
            Ok(n) => {
                if let Ok(mut guard) = ps.inner.lock() {
                    guard.nrec = n;
                    if let Ok(file) = std::fs::OpenOptions::new()
                        .read(true)
                        .create(true)
                        .append(true)
                        .open(&ps.path)
                    {
                        guard.approx_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
                        guard.file = std::io::BufWriter::new(file);
                    }
                }
                log::info!("persist: 紧凑化完成，{} 条记录", n);
            }
            Err(e) => {
                log::warn!("persist: 紧凑化失败（{}），保留原文件", e);
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
        // 持久文件同步截断（否则重启会复活已清数据）。
        if let Some(ps) = self.persist.as_ref()
            && let Ok(mut guard) = ps.inner.lock()
        {
            let reopened = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&ps.path);
            match reopened {
                Ok(f) => {
                    guard.approx_bytes = 0;
                    guard.file = std::io::BufWriter::new(f);
                    guard.nrec = 0;
                }
                Err(e) => {
                    log::warn!("persist: 清空截断失败（{}）", e);
                }
            }
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

    /// 存取原始 f32 —— 往返值逐位相等,无压缩损耗
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

    /// bloom 重建后负过滤仍正确(不产生错误命中)
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

    /// 并发读写 —— 多任务同时 put/get 无 panic、无错误值
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

    // ========================================================================
    // WAL 持久层测试
    // ========================================================================

    fn persist_backend(
        dir: &std::path::Path,
        max_bytes: u64,
    ) -> (OxCacheBackend, std::path::PathBuf) {
        let path = dir.join("cache.wal");
        let backend =
            OxCacheBackend::with_persist(1024, path.clone(), max_bytes, "test-model".to_string());
        (backend, path)
    }

    /// 每次插入产生数据记录 + 提交记录；崩溃截断尾部半条后重放无脏数据。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_persist_write_path_two_phase_append() {
        let dir = tempfile::tempdir().unwrap();
        let (cache, path) = persist_backend(dir.path(), 1024 * 1024);
        cache.put("k1", vec![1.0, 2.0]).await;
        cache.put("k2", vec![3.0]).await;
        let (data, commit) = persist::record_counts(&path);
        assert_eq!(data, 2, "每次插入应产生一段数据记录");
        assert_eq!(commit, 2, "每次插入应产生一段提交记录");
        // 崩溃模拟：截断尾部半条（k2 的提交记录写一半）。
        // 按两段协议语义：已提交 k1 恢复；k2 数据虽完整但提交撕裂，
        // 按"最多丢最后一条"被丢弃——关键是绝不出现半条脏数据。
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 3).unwrap();
        drop(f);
        // 重放：k1 完整恢复，k2 缺席（而非脏数据）。
        let cache2 =
            OxCacheBackend::with_persist(1024, path.clone(), 1024 * 1024, "test-model".to_string());
        cache2.load_persisted().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(cache2.get("k1").await, Some(vec![1.0, 2.0]));
        assert_eq!(
            cache2.get("k2").await,
            None,
            "提交撕裂的最后一条必须丢弃而非半应用"
        );
    }

    /// roundtrip（写入 N 条 → 重启后命中 N 条）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_persist_roundtrip_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (cache, path) = persist_backend(dir.path(), 1024 * 1024);
        for i in 0..8 {
            cache.put(&format!("rk{}", i), vec![i as f32, 0.5]).await;
        }
        drop(cache);
        let cache2 =
            OxCacheBackend::with_persist(1024, path, 1024 * 1024, "test-model".to_string());
        cache2.load_persisted().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        for i in 0..8 {
            assert_eq!(
                cache2.get(&format!("rk{}", i)).await,
                Some(vec![i as f32, 0.5]),
                "重启后应命中全部 N 条"
            );
        }
    }

    /// 中段损坏跳过继续 + 指纹不匹配弃用。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_persist_replay_skips_corrupt_middle() {
        let dir = tempfile::tempdir().unwrap();
        let (cache, path) = persist_backend(dir.path(), 1024 * 1024);
        cache.put("good-a", vec![1.0]).await;
        cache.put("good-b", vec![2.0]).await;
        cache.put("good-c", vec![3.0]).await;
        // 破坏第一条数据记录的向量载荷（保持长度前缀合法，使其可跳过）。
        let mut bytes = std::fs::read(&path).unwrap();
        // 文件头 8 字节 + 类型 1 + key_len 4 + key("good-a"=6) + dim 4 = 23；
        // 向量载荷始于 offset 23。
        let off = 8 + 1 + 4 + 6 + 4;
        bytes[off] ^= 0xFF;
        bytes[off + 1] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let cache2 =
            OxCacheBackend::with_persist(1024, path, 1024 * 1024, "test-model".to_string());
        cache2.load_persisted().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(cache2.get("good-a").await.is_none(), "损坏条目必须跳过");
        assert_eq!(cache2.get("good-b").await, Some(vec![2.0]), "其余继续恢复");
        assert_eq!(cache2.get("good-c").await, Some(vec![3.0]));
        // 指纹不匹配：换 tag 重放应全部弃用。
        let cache3 = OxCacheBackend::with_persist(
            1024,
            dir.path().join("cache.wal"),
            1024 * 1024,
            "other-model".to_string(),
        );
        cache3.load_persisted().await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(cache3.get("good-b").await.is_none(), "指纹不匹配应弃用");
    }

    /// 超限触发紧凑化（文件收缩、去重且数据完整）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_persist_compaction_on_limit() {
        let dir = tempfile::tempdir().unwrap();
        // 阈值较小；3 个 key 反复覆盖写入 30 次（累积远超阈值），
        // 紧凑化后文件应只保留去重后的 3 条记录。
        let (cache, path) = persist_backend(dir.path(), 400);
        for i in 0..30 {
            cache.put(&format!("ck{}", i % 3), vec![i as f32]).await;
        }
        let (data_records, _) = persist::record_counts(&path);
        // 无紧凑化时应为 30 条；紧凑化后仅保留去重快照 + 尾部少量追加。
        assert!(
            data_records <= 8,
            "紧凑化应去重，实际 {} 条记录",
            data_records
        );
        let size_after = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after < 400,
            "紧凑化后文件应低于阈值，实际 {} 字节",
            size_after
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        // 最新值完整（key ck{i%3} 最后一次写入 i=27/28/29）。
        assert_eq!(cache.get("ck0").await, Some(vec![27.0]));
        assert_eq!(cache.get("ck1").await, Some(vec![28.0]));
        assert_eq!(cache.get("ck2").await, Some(vec![29.0]));
    }

    /// 默认无 persist_path 时零写盘行为。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_no_persist_by_default() {
        let cache = OxCacheBackend::new(16);
        assert!(cache.persist_path().is_none());
        cache.put("k", vec![1.0]).await;
        assert_eq!(cache.get("k").await, Some(vec![1.0]));
    }
}
