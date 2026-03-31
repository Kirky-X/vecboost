// Copyright (c) 2025 VecBoost
//
// Licensed under MIT License
// See LICENSE file in the project root for full license information.

//! Bloom Filter 实现用于缓存穿透防护
//!
//! Bloom Filter 是一种空间效率很高的概率型数据结构，用于判断元素是否可能在集合中。
//! 它的特点是：
//! - 如果返回 false，则元素一定不在集合中
//! - 如果返回 true，则元素可能在集合中（存在误判）

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bloom Filter 配置
#[derive(Debug, Clone)]
pub struct BloomFilterConfig {
    /// 预期元素数量
    pub expected_elements: usize,
    /// 可接受的误判率（0.0-1.0）
    pub false_positive_rate: f64,
}

impl Default for BloomFilterConfig {
    fn default() -> Self {
        Self {
            expected_elements: 10000,
            false_positive_rate: 0.01, // 1% 误判率
        }
    }
}

impl BloomFilterConfig {
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        Self {
            expected_elements,
            false_positive_rate,
        }
    }
}

/// Bloom Filter 实现
pub struct BloomFilter {
    /// 位数组
    bits: Arc<RwLock<Vec<bool>>>,
    /// 哈希函数数量
    hash_count: usize,
    /// 当前元素数量
    element_count: Arc<RwLock<usize>>,
}

impl BloomFilter {
    /// 创建新的 Bloom Filter
    pub fn new(config: &BloomFilterConfig) -> Self {
        let m = Self::optimal_m(config.expected_elements, config.false_positive_rate);
        let k = Self::optimal_k(m, config.expected_elements);

        Self {
            bits: Arc::new(RwLock::new(vec![false; m])),
            hash_count: k,
            element_count: Arc::new(RwLock::new(0)),
        }
    }

    /// 计算最优的位数组大小
    fn optimal_m(n: usize, p: f64) -> usize {
        // m = -n * ln(p) / (ln(2)^2)
        let n = n as f64;
        let p = p;
        let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        (-n * p.ln() / ln2_sq).ceil() as usize
    }

    /// 计算最优的哈希函数数量
    fn optimal_k(m: usize, n: usize) -> usize {
        // k = (m/n) * ln(2)
        let m = m as f64;
        let n = n as f64;
        ((m / n) * std::f64::consts::LN_2).ceil() as usize
    }

    /// 使用多个哈希函数生成多个哈希值
    async fn get_hash_indices(&self, item: &[u8]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.hash_count);

        // 使用双重哈希技巧生成多个哈希值
        // h(i) = h1(x) + i * h2(x)
        let h1 = self.hash_with_seed(item, 0);
        let h2 = self.hash_with_seed(item, 1);

        let bits_len = self.bits.read().await.len();
        for i in 0..self.hash_count {
            let hash = h1.wrapping_add(i.wrapping_mul(h2));
            indices.push(hash % bits_len);
        }

        indices
    }

    /// 带种子的哈希函数
    fn hash_with_seed(&self, item: &[u8], seed: u64) -> usize {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// 添加元素到 Bloom Filter
    pub async fn add(&self, item: &[u8]) {
        let indices = self.get_hash_indices(item).await;
        let mut bits = self.bits.write().await;
        for index in indices {
            bits[index] = true;
        }

        let mut count = self.element_count.write().await;
        *count += 1;
    }

    /// 检查元素是否可能在 Bloom Filter 中
    pub async fn contains(&self, item: &[u8]) -> bool {
        let indices = self.get_hash_indices(item).await;
        let bits = self.bits.read().await;
        indices.iter().all(|&index| bits[index])
    }

    /// 获取当前元素数量
    pub async fn len(&self) -> usize {
        *self.element_count.read().await
    }

    /// 检查是否为空
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// 重置 Bloom Filter
    pub async fn clear(&self) {
        let mut bits = self.bits.write().await;
        bits.fill(false);

        let mut count = self.element_count.write().await;
        *count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bloom_filter_basic() {
        let config = BloomFilterConfig::new(1000, 0.01);
        let filter = BloomFilter::new(&config);

        // 初始应该不包含任何元素
        assert!(!filter.contains(b"apple").await);
        assert!(!filter.contains(b"banana").await);

        // 添加元素
        filter.add(b"apple").await;
        filter.add(b"banana").await;

        // 现在应该包含这些元素
        assert!(filter.contains(b"apple").await);
        assert!(filter.contains(b"banana").await);

        // 不应该包含未添加的元素
        assert!(!filter.contains(b"cherry").await);
    }

    #[tokio::test]
    async fn test_bloom_filter_false_positive() {
        let config = BloomFilterConfig::new(100, 0.01);
        let filter = BloomFilter::new(&config);

        // 添加一些元素
        for i in 0..50 {
            filter.add(format!("item{}", i).as_bytes()).await;
        }

        // 检查不存在的元素（可能会有误判，但概率很低）
        let mut false_positives = 0;
        for i in 100..200 {
            if filter.contains(format!("item{}", i).as_bytes()).await {
                false_positives += 1;
            }
        }

        // 误判率应该在可接受范围内
        let false_positive_rate = false_positives as f64 / 100.0;
        println!("False positive rate: {}", false_positive_rate);
        // 注意：由于测试样本较小，实际误判率可能略高于理论值
        assert!(false_positive_rate < 0.1, "False positive rate too high");
    }

    #[tokio::test]
    async fn test_bloom_filter_count() {
        let config = BloomFilterConfig::new(1000, 0.01);
        let filter = BloomFilter::new(&config);

        assert_eq!(filter.len().await, 0);

        filter.add(b"test1").await;
        filter.add(b"test2").await;

        assert_eq!(filter.len().await, 2);
    }
}
