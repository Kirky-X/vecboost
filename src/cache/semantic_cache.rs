// Copyright (c) 2025-2026 Kirky.X🌠
// SPDX-License-Identifier: Apache-2.0

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
use crate::utils::vector::cosine_similarity;
use crate::utils::vquant::{
    BinaryVector, I8Vector, cosine_binary, dot_i8, quantize_binary, quantize_i8,
};

/// 语义缓存向量比较模式。
///
/// - `Exact`（默认）：行为与现状完全一致，不调用任何量化路径；
/// - `I8` / `Binary`：候选粗筛用 vquant 估计器，命中后用原始向量精确复验。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComparisonMode {
    #[default]
    Exact,
    I8,
    Binary,
}

impl ComparisonMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ComparisonMode::Exact => "exact",
            ComparisonMode::I8 => "i8",
            ComparisonMode::Binary => "binary",
        }
    }
}

impl std::str::FromStr for ComparisonMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "exact" => Ok(ComparisonMode::Exact),
            "i8" | "int8" => Ok(ComparisonMode::I8),
            "binary" | "bin" => Ok(ComparisonMode::Binary),
            other => Err(format!(
                "未知的 comparison_mode: '{}'（可选 exact|i8|binary）",
                other
            )),
        }
    }
}

impl std::fmt::Display for ComparisonMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// 语义缓存默认相似度阈值（trigram Jaccard）
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.7;
/// 语义索引默认最大条目数
const DEFAULT_CAPACITY: usize = 10000;

/// 语义缓存条目
struct SemanticEntry {
    /// 缓存的 trigram 集合，避免重复计算
    trigrams: HashSet<u32>,
    embedding: Vec<f32>,
    /// 量化粗筛码：仅非 exact 模式插入时生成，默认 None（零量化路径）。
    i8code: Option<I8Vector>,
    bincode: Option<BinaryVector>,
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
    /// 向量比较模式（默认 exact，行为不变；来自 `[cache] comparison_mode`）。
    pub comparison_mode: ComparisonMode,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            capacity: DEFAULT_CAPACITY,
            comparison_mode: ComparisonMode::Exact,
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
    comparison_mode: ComparisonMode,
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
    #[allow(
        dead_code,
        reason = "Test helper / trait dispatch / inventory, not directly called"
    )]
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
            comparison_mode: ComparisonMode::Exact,
            exact_hits: AtomicU64::new(0),
            semantic_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_entries: AtomicUsize::new(0),
        }
    }

    /// 清空全部语义索引与底层精确缓存（模型切换防跨模型污染）。
    pub async fn clear(&self) {
        self.semantic_index.write().await.clear();
        self.total_entries.store(0, Ordering::SeqCst);
        self.exact_hits.store(0, Ordering::SeqCst);
        self.semantic_hits.store(0, Ordering::SeqCst);
        self.misses.store(0, Ordering::SeqCst);
        self.exact_cache.clear().await;
    }

    /// 创建禁用的语义缓存。
    pub fn disabled() -> Self {
        Self {
            exact_cache: Arc::new(OxCacheBackend::disabled()),
            semantic_index: RwLock::new(Vec::new()),
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            capacity: 0,
            enabled: false,
            comparison_mode: ComparisonMode::Exact,
            exact_hits: AtomicU64::new(0),
            semantic_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            total_entries: AtomicUsize::new(0),
        }
    }

    /// 设置向量比较模式（builder）。默认 exact。
    pub fn with_comparison_mode(mut self, mode: ComparisonMode) -> Self {
        self.comparison_mode = mode;
        self
    }

    /// 返回当前向量比较模式。
    pub fn comparison_mode(&self) -> ComparisonMode {
        self.comparison_mode
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
        let query_trigrams = pack_trigrams(query.as_bytes());
        let query_len = query_trigrams.len();

        for (i, entry) in index.iter().enumerate() {
            // 大小差预过滤 —— Jaccard 上限 = min/max;大小差超过阈值时
            // 不可能达标,跳过昂贵的集合交/并
            let upper_bound = query_len.min(entry.trigrams.len()) as f32
                / query_len.max(entry.trigrams.len()).max(1) as f32;
            if upper_bound < self.similarity_threshold {
                continue;
            }
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

        // 量化码仅在非 exact 模式生成；exact 模式零量化路径。
        let (i8code, bincode) = match self.comparison_mode {
            ComparisonMode::Exact => (None, None),
            ComparisonMode::I8 => (Some(quantize_i8(&embedding)), None),
            ComparisonMode::Binary => (None, Some(quantize_binary(&embedding))),
        };
        index.push(SemanticEntry {
            trigrams: pack_trigrams(text.as_bytes()),
            embedding,
            i8code,
            bincode,
            last_access: Instant::now(),
        });
        self.total_entries.store(index.len(), Ordering::Relaxed);
    }

    /// 向量相似度查找：按 `comparison_mode` 比较查询向量与索引条目。
    ///
    /// - `Exact`：全量余弦精确扫描；
    /// - `I8` / `Binary`：vquant 估计器粗筛（阈值放宽 10% 防假阴性），
    ///   候选命中后用原始向量精确复验，最终判定一律用原始向量。
    ///
    /// 返回最佳匹配的原始向量（克隆），无达标时 None。
    pub async fn find_similar_by_vector(&self, query: &[f32]) -> Option<Vec<f32>> {
        let index = self.semantic_index.read().await;
        if index.is_empty() || query.is_empty() {
            return None;
        }
        // 粗筛阈值：放宽 10%，宁可多复验、不漏检。
        let coarse_bar = self.similarity_threshold * 0.9;
        let mut best: Option<(f32, usize)> = None;
        match self.comparison_mode {
            ComparisonMode::Exact => {
                for (i, entry) in index.iter().enumerate() {
                    let Ok(sim) = cosine_similarity(query, &entry.embedding) else {
                        continue;
                    };
                    if sim >= self.similarity_threshold
                        && best.map(|(b, _)| sim > b).unwrap_or(true)
                    {
                        best = Some((sim, i));
                    }
                }
            }
            ComparisonMode::I8 => {
                let qcode = quantize_i8(query);
                let qself = dot_i8(&qcode, &qcode).max(1e-12);
                for (i, entry) in index.iter().enumerate() {
                    let Some(ref ecode) = entry.i8code else {
                        continue;
                    };
                    // 估计余弦 = 还原点积 / 范数积（范数经同估计器求得）。
                    let denom = (qself * dot_i8(ecode, ecode).max(1e-12)).sqrt();
                    if denom <= 0.0 {
                        continue;
                    }
                    let est = dot_i8(&qcode, ecode) / denom;
                    if est < coarse_bar {
                        continue;
                    }
                    // 精确复验（原始向量）。
                    let Ok(sim) = cosine_similarity(query, &entry.embedding) else {
                        continue;
                    };
                    if sim >= self.similarity_threshold
                        && best.map(|(b, _)| sim > b).unwrap_or(true)
                    {
                        best = Some((sim, i));
                    }
                }
            }
            ComparisonMode::Binary => {
                let qcode = quantize_binary(query);
                for (i, entry) in index.iter().enumerate() {
                    let Some(ref ecode) = entry.bincode else {
                        continue;
                    };
                    if cosine_binary(&qcode, ecode) < coarse_bar {
                        continue;
                    }
                    // 精确复验（原始向量）。
                    let Ok(sim) = cosine_similarity(query, &entry.embedding) else {
                        continue;
                    };
                    if sim >= self.similarity_threshold
                        && best.map(|(b, _)| sim > b).unwrap_or(true)
                    {
                        best = Some((sim, i));
                    }
                }
            }
        }
        best.map(|(_, i)| index[i].embedding.clone())
    }

    /// 测试专用：统计带量化码的条目数（断言默认零量化路径用）。
    #[cfg(test)]
    async fn quantized_entry_count(&self) -> usize {
        let index = self.semantic_index.read().await;
        index
            .iter()
            .filter(|e| e.i8code.is_some() || e.bincode.is_some())
            .count()
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

/// 将 3 字节窗口打包为一个 u32(避免每 trigram 一次堆分配)。
///
/// 尾部不足 3 字节时以零填充补齐最后一个窗口(与 `windows(3)` 语义一致:
/// 字节数 < 3 的文本没有完整窗口,返回空集)。
#[inline]
fn pack_trigrams(bytes: &[u8]) -> HashSet<u32> {
    if bytes.len() < 3 {
        return HashSet::new();
    }
    let mut set = HashSet::with_capacity(bytes.len() - 2);
    for w in bytes.windows(3) {
        let packed = ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32);
        set.insert(packed);
    }
    set
}

/// 使用预计算的 trigram 集合计算 Jaccard 相似度。
///
/// 避免在批量比较中重复构建 HashSet。
fn trigram_jaccard_with_set(a_trigrams: &HashSet<u32>, b_trigrams: &HashSet<u32>) -> f32 {
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
        let trigrams_a = pack_trigrams(a.as_bytes());
        let trigrams_b = pack_trigrams(b.as_bytes());
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

    #[test]
    fn test_semantic_cache_config_default() {
        let config = SemanticCacheConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.similarity_threshold, DEFAULT_SIMILARITY_THRESHOLD);
        assert_eq!(config.capacity, DEFAULT_CAPACITY);
        assert_eq!(config.comparison_mode, ComparisonMode::Exact);
    }

    #[test]
    fn test_comparison_mode_parse() {
        use std::str::FromStr;
        assert_eq!(
            ComparisonMode::from_str("exact").unwrap(),
            ComparisonMode::Exact
        );
        assert_eq!(ComparisonMode::from_str("i8").unwrap(), ComparisonMode::I8);
        assert_eq!(
            ComparisonMode::from_str("binary").unwrap(),
            ComparisonMode::Binary
        );
        assert_eq!(ComparisonMode::default(), ComparisonMode::Exact);
        assert!(ComparisonMode::from_str("fp16").is_err());
    }

    /// exact 模式输出与现状一致（暴力余弦扫描等价）。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_vector_search_exact_mode_matches_brute_force() {
        let cache = SemanticCache::with_capacity(0.7, 16);
        let va = vec![1.0f32, 0.0, 0.0, 0.0];
        let vb = vec![0.9f32, 0.1, 0.0, 0.0];
        cache.insert_to_index("a", va.clone()).await;
        cache.insert_to_index("b", vb.clone()).await;
        // 暴力余弦最优应为 a（与查询完全相同）。
        let hit = cache.find_similar_by_vector(&va).await.unwrap();
        assert_eq!(hit, va, "exact 模式必须返回余弦最优的原始向量");
        // 默认零量化路径。
        assert_eq!(cache.quantized_entry_count().await, 0);
    }

    /// i8 模式对完全相同文本仍精确命中（无假阴性），命中为原始向量。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_vector_search_i8_mode_exact_text_hit() {
        let cache = SemanticCache::with_capacity(0.7, 16).with_comparison_mode(ComparisonMode::I8);
        // 384 维伪嵌入（i8/旋转路径在高维才有意义）。
        let va: Vec<f32> = (0..384)
            .map(|i| ((i * 13 + 7) % 101) as f32 / 101.0 - 0.5)
            .collect();
        cache.insert_to_index("dup", va.clone()).await;
        assert_eq!(cache.quantized_entry_count().await, 1);
        let hit = cache.find_similar_by_vector(&va).await.unwrap();
        assert_eq!(hit, va, "i8 模式对相同向量必须精确命中原始向量");
    }

    /// binary 模式对完全相同文本仍精确命中。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_vector_search_binary_mode_exact_text_hit() {
        let cache =
            SemanticCache::with_capacity(0.7, 16).with_comparison_mode(ComparisonMode::Binary);
        let va: Vec<f32> = (0..384)
            .map(|i| ((i * 29 + 3) % 89) as f32 / 89.0 - 0.5)
            .collect();
        cache.insert_to_index("dup", va.clone()).await;
        let hit = cache.find_similar_by_vector(&va).await.unwrap();
        assert_eq!(hit, va, "binary 模式对相同向量必须精确命中原始向量");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_with_capacity_constructor() {
        let cache = SemanticCache::with_capacity(0.8, 50);
        assert!(cache.is_enabled());
        // Should work like a normal enabled cache
        let result = cache
            .get_or_compute("test query", || async { Ok(vec![1.0, 2.0]) })
            .await
            .unwrap();
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_trigram_jaccard_with_set_both_empty() {
        let a = HashSet::new();
        let b = HashSet::new();
        assert_eq!(trigram_jaccard_with_set(&a, &b), 0.0);
    }

    #[test]
    fn test_trigram_jaccard_with_set_one_empty() {
        let mut a = HashSet::new();
        a.insert(0x010203u32);
        let b = HashSet::new();
        assert_eq!(trigram_jaccard_with_set(&a, &b), 0.0);
        assert_eq!(trigram_jaccard_with_set(&b, &a), 0.0);
    }

    #[test]
    fn test_trigram_jaccard_with_set_identical() {
        let mut a = HashSet::new();
        a.insert(0x010203u32);
        a.insert(0x040506u32);
        assert_eq!(trigram_jaccard_with_set(&a, &a), 1.0);
    }

    /// 打包一致性 —— pack_trigrams 与逐字节窗口语义等价
    #[test]
    fn test_pack_trigrams_roundtrip_consistency() {
        let text = "hello world 机器学习";
        let packed = pack_trigrams(text.as_bytes());
        let expected: HashSet<u32> = text
            .as_bytes()
            .windows(3)
            .map(|w| ((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32))
            .collect();
        assert_eq!(packed, expected);
        // 短文本无完整窗口 → 空集
        assert!(pack_trigrams(b"ab").is_empty());
        assert!(pack_trigrams(b"").is_empty());
    }
}
