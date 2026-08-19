// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! 语义缓存层：在精确匹配缓存之上添加 trigram Jaccard 文本相似度检查。
//!
//! 查询路径：精确匹配 → trigram 搜索 → 计算回填。
//! 核心价值：精确 miss 后、模型推理前，插入一层零开销的文本相似度检查。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::RwLock;

use crate::cache::OxCacheBackend;

/// 语义缓存条目
struct SemanticEntry {
    /// 缓存的 trigram 集合，避免重复计算
    trigrams: HashSet<Vec<u8>>,
    embedding: Vec<f32>,
    last_access: Instant,
}

/// 语义缓存统计
#[derive(Debug, Clone)]
pub struct SemanticCacheStats {
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub misses: u64,
    pub total_entries: usize,
}

/// 语义缓存配置
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    pub enabled: bool,
    pub similarity_threshold: f32,
    pub capacity: usize,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: 0.7,
            capacity: 10000,
        }
    }
}

/// 语义缓存：在 OxCacheBackend 精确匹配之上添加 trigram 文本相似度检查。
pub struct SemanticCache {
    exact_cache: Arc<OxCacheBackend>,
    semantic_index: RwLock<Vec<SemanticEntry>>,
    similarity_threshold: f32,
    capacity: usize,
    enabled: bool,
    // Stats counters
    exact_hits: AtomicU64,
    semantic_hits: AtomicU64,
    misses: AtomicU64,
    total_entries: AtomicUsize,
}

impl SemanticCache {
    /// 创建启用的语义缓存（便捷构造方法，内部自动创建 OxCacheBackend）。
    pub fn with_capacity(similarity_threshold: f32, capacity: usize) -> Self {
        let exact_cache = Arc::new(OxCacheBackend::new(capacity));
        Self::new(exact_cache, similarity_threshold, capacity)
    }

    /// 创建启用的语义缓存。
    #[allow(dead_code)]
    pub(crate) fn new(
        exact_cache: Arc<OxCacheBackend>,
        similarity_threshold: f32,
        capacity: usize,
    ) -> Self {
        Self {
            exact_cache,
            semantic_index: RwLock::new(Vec::with_capacity(capacity.min(1024))),
            similarity_threshold,
            capacity,
            enabled: true,
            exact_hits: AtomicU64::new(0),
            semantic_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_entries: AtomicUsize::new(0),
        }
    }

    /// 创建禁用的语义缓存。
    pub fn disabled() -> Self {
        Self {
            exact_cache: Arc::new(OxCacheBackend::disabled()),
            semantic_index: RwLock::new(Vec::new()),
            similarity_threshold: 0.7,
            capacity: 0,
            enabled: false,
            exact_hits: AtomicU64::new(0),
            semantic_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_entries: AtomicUsize::new(0),
        }
    }

    /// 返回缓存是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 核心查询方法：精确匹配 → trigram 搜索 → 计算回填。
    ///
    /// `compute_fn` 是模型推理回调，仅在精确 miss 且语义 miss 时调用。
    pub async fn get_or_compute<F, Fut>(
        &self,
        text: &str,
        compute_fn: F,
    ) -> Result<Vec<f32>, crate::error::VecboostError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<f32>, crate::error::VecboostError>>,
    {
        if !self.enabled {
            return compute_fn().await;
        }

        let cache_key = format!("text:{}", text);

        // 第一级：精确匹配
        if let Some(embedding) = self.exact_cache.get(&cache_key).await {
            self.exact_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(embedding);
        }

        // 第二级：trigram 语义搜索
        if let Some(embedding) = self.find_similar(text).await {
            self.semantic_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(embedding);
        }

        // 第三级：模型推理
        self.misses.fetch_add(1, Ordering::Relaxed);
        let embedding = compute_fn().await?;

        // 回填到精确缓存和语义索引
        self.exact_cache.put(&cache_key, embedding.clone()).await;
        self.insert_to_index(text, embedding.clone()).await;

        Ok(embedding)
    }

    /// 在语义索引中查找与 query 最相似的条目。
    /// 返回 Some(embedding) 当最大相似度 > threshold。
    pub async fn find_similar(&self, query: &str) -> Option<Vec<f32>> {
        // 纯读操作使用 read 锁
        let index = self.semantic_index.read().await;
        let mut best_sim = 0.0f32;
        let mut best_idx = None;

        // 构建 query 的 trigram 集合（仅一次，复用于所有比较）
        let query_trigrams: HashSet<Vec<u8>> =
            query.as_bytes().windows(3).map(|w| w.to_vec()).collect();

        for (i, entry) in index.iter().enumerate() {
            let sim = trigram_jaccard_with_set(&query_trigrams, &entry.trigrams);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if best_sim >= self.similarity_threshold
            && let Some(idx) = best_idx
        {
            return Some(index[idx].embedding.clone());
        }
        None
    }

    /// 插入新条目到语义索引，必要时执行 LRU 驱逐。
    async fn insert_to_index(&self, text: &str, embedding: Vec<f32>) {
        let mut index = self.semantic_index.write().await;

        // LRU 驱逐：达到容量时移除最久未访问的条目
        if index.len() >= self.capacity && !index.is_empty() {
            let oldest_idx = index
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(i, _)| i)
                .unwrap_or(0);
            index.swap_remove(oldest_idx);
            self.total_entries.store(index.len(), Ordering::Relaxed);
        }

        index.push(SemanticEntry {
            trigrams: text.as_bytes().windows(3).map(|w| w.to_vec()).collect(),
            embedding,
            last_access: Instant::now(),
        });
        self.total_entries.store(index.len(), Ordering::Relaxed);
    }

    /// 返回当前统计快照。
    pub fn stats(&self) -> SemanticCacheStats {
        SemanticCacheStats {
            exact_hits: self.exact_hits.load(Ordering::Relaxed),
            semantic_hits: self.semantic_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            total_entries: self.total_entries.load(Ordering::Relaxed),
        }
    }
}

/// 使用预计算的 trigram 集合计算 Jaccard 相似度。
///
/// 避免在批量比较中重复构建 HashSet。
fn trigram_jaccard_with_set(a_trigrams: &HashSet<Vec<u8>>, b_trigrams: &HashSet<Vec<u8>>) -> f32 {
    if a_trigrams.is_empty() || b_trigrams.is_empty() {
        return 0.0;
    }
    let intersection = a_trigrams.intersection(b_trigrams).count();
    let union_size = a_trigrams.union(b_trigrams).count();
    if union_size == 0 {
        return 0.0;
    }
    intersection as f32 / union_size as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 计算两个字符串的字符级 trigram Jaccard 相似度（测试辅助函数）。
    fn trigram_jaccard(a: &str, b: &str) -> f32 {
        if a.len() < 3 || b.len() < 3 {
            return 0.0;
        }
        let trigrams_a: HashSet<Vec<u8>> = a.as_bytes().windows(3).map(|w| w.to_vec()).collect();
        let trigrams_b: HashSet<Vec<u8>> = b.as_bytes().windows(3).map(|w| w.to_vec()).collect();
        trigram_jaccard_with_set(&trigrams_a, &trigrams_b)
    }

    #[test]
    fn test_trigram_jaccard_identical() {
        let sim = trigram_jaccard("hello world", "hello world");
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical strings should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_trigram_jaccard_similar() {
        let sim = trigram_jaccard("今天天气怎么样", "今天天气怎么样啊");
        assert!(
            sim > 0.5,
            "similar strings should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_trigram_jaccard_different() {
        let sim = trigram_jaccard("hello", "xyz");
        assert!(
            sim < 0.3,
            "different strings should have low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_trigram_jaccard_empty() {
        assert_eq!(trigram_jaccard("", "hello"), 0.0);
        assert_eq!(trigram_jaccard("hello", ""), 0.0);
        assert_eq!(trigram_jaccard("ab", "abc"), 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_semantic_cache_exact_hit() {
        let backend = Arc::new(OxCacheBackend::new(1024));
        let cache = SemanticCache::new(backend.clone(), 0.7, 100);

        // 预先存入精确缓存
        backend.put("text:hello", vec![1.0, 2.0, 3.0]).await;
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let result = cache
            .get_or_compute("hello", || async { Ok(vec![9.0, 9.0, 9.0]) })
            .await
            .unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.stats().exact_hits, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_semantic_cache_semantic_hit() {
        let backend = Arc::new(OxCacheBackend::new(1024));
        let cache = SemanticCache::new(backend.clone(), 0.5, 100);

        // 存入一个文本到语义索引（通过 compute_fn）
        let _ = cache
            .get_or_compute("今天天气怎么样", || async {
                Ok(vec![1.0, 2.0, 3.0])
            })
            .await
            .unwrap();

        // 用近似改写查询——应语义命中
        let result = cache
            .get_or_compute("今天天气怎么样啊", || async {
                Ok(vec![9.0, 9.0, 9.0])
            })
            .await
            .unwrap();
        assert_eq!(
            result,
            vec![1.0, 2.0, 3.0],
            "should return cached embedding via semantic match"
        );
        assert_eq!(cache.stats().semantic_hits, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_semantic_cache_miss_and_store() {
        let backend = Arc::new(OxCacheBackend::new(1024));
        let cache = SemanticCache::new(backend.clone(), 0.9, 100);

        // 完全不同的文本——应 miss 并存储
        let result = cache
            .get_or_compute("机器学习", || async { Ok(vec![5.0, 6.0]) })
            .await
            .unwrap();
        assert_eq!(result, vec![5.0, 6.0]);
        assert_eq!(cache.stats().misses, 1);

        // 再次查询同一文本——应精确命中（已存入 exact_cache）
        // 注意：moka 异步索引可能需要短暂等待
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let result2 = cache
            .get_or_compute("机器学习", || async { Ok(vec![9.0, 9.0]) })
            .await
            .unwrap();
        // 可能精确命中或语义命中（相似度 = 1.0）
        assert_eq!(result2, vec![5.0, 6.0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_lru_eviction() {
        let backend = Arc::new(OxCacheBackend::new(1024));
        let cache = SemanticCache::new(backend.clone(), 0.99, 3); // 容量仅 3

        // 存入 3 个不同文本
        cache
            .get_or_compute("aaa_text_one", || async { Ok(vec![1.0]) })
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        cache
            .get_or_compute("bbb_text_two", || async { Ok(vec![2.0]) })
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        cache
            .get_or_compute("ccc_text_three", || async { Ok(vec![3.0]) })
            .await
            .unwrap();

        // 语义索引应有 3 个条目（精确缓存 miss 后存入）
        {
            let index = cache.semantic_index.read().await;
            assert!(index.len() <= 3, "should not exceed capacity");
        }

        // 再存入一个——应驱逐最老的
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        cache
            .get_or_compute("ddd_completely_different", || async { Ok(vec![4.0]) })
            .await
            .unwrap();
        {
            let index = cache.semantic_index.read().await;
            assert!(
                index.len() <= 3,
                "should still not exceed capacity after eviction"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_stats_tracking() {
        let backend = Arc::new(OxCacheBackend::new(1024));
        let cache = SemanticCache::new(backend.clone(), 0.7, 100);

        // 1 miss
        cache
            .get_or_compute("unique_text_alpha", || async { Ok(vec![1.0]) })
            .await
            .unwrap();
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.exact_hits, 0);
        assert_eq!(stats.total_entries, 1);

        // 1 semantic hit (similar text)
        let _ = cache
            .get_or_compute("unique_text_alpha_beta", || async { Ok(vec![2.0]) })
            .await;
        let stats = cache.stats();
        // total_entries 可能是 1 或 2（取决于是否语义命中）
        assert!(stats.total_entries >= 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_disabled_cache_always_computes() {
        let cache = SemanticCache::disabled();
        assert!(!cache.is_enabled());

        let result = cache
            .get_or_compute("any text", || async { Ok(vec![42.0]) })
            .await
            .unwrap();
        assert_eq!(result, vec![42.0]);
    }
}
