// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! GPU 显存分页管理器：基于 LRU-K 策略的模型权重分层管理。
//!
//! 热层常驻 GPU，冷层按需从 CPU 内存换入/换出。
//! 支持预取优化：根据当前推理层预测后续层，提前换入。

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// 分页错误类型
#[derive(Debug, Clone)]
pub enum PagingError {
    /// 层不在 GPU 上（需要换入）
    LayerNotOnGpu(String),
    /// 层不在 CPU 上（需要换出）
    LayerNotOnCpu(String),
    /// 显存不足且无可驱逐层
    OutOfGpuMemory { needed: u64, available: u64 },
    /// 层不存在
    LayerNotFound(String),
}

impl std::fmt::Display for PagingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagingError::LayerNotOnGpu(name) => write!(f, "layer '{}' is not on GPU", name),
            PagingError::LayerNotOnCpu(name) => write!(f, "layer '{}' is not on CPU", name),
            PagingError::OutOfGpuMemory { needed, available } => {
                write!(f, "GPU OOM: need {} bytes, available {} bytes", needed, available)
            }
            PagingError::LayerNotFound(name) => write!(f, "layer '{}' not found", name),
        }
    }
}

impl std::error::Error for PagingError {}

/// 分页配置
#[derive(Debug, Clone)]
pub struct PagingConfig {
    pub enabled: bool,
    pub gpu_memory_budget_bytes: u64,
    pub lru_k: usize,
    pub prefetch_depth: usize,
}

impl Default for PagingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // 默认 2GB 显存预算；生产环境应根据 GPU 实际 VRAM 配置覆盖
            gpu_memory_budget_bytes: 2 * 1024 * 1024 * 1024,
            lru_k: 2,
            prefetch_depth: 2,
        }
    }
}

/// 层权重存储位置
#[derive(Debug, Clone, PartialEq)]
pub enum LayerLocation {
    OnGpu,
    OnCpu,
    InTransfer,
}

/// 分页统计
#[derive(Debug, Clone, Default)]
pub struct PagingStats {
    pub gpu_usage_bytes: u64,
    pub cpu_backup_count: usize,
    pub page_in_count: u64,
    pub page_out_count: u64,
    pub total_page_in_latency_ms: f64,
}

impl PagingStats {
    pub fn avg_page_in_latency_ms(&self) -> f64 {
        if self.page_in_count == 0 {
            0.0
        } else {
            self.total_page_in_latency_ms / self.page_in_count as f64
        }
    }
}

/// 层元数据
struct LayerMeta {
    location: LayerLocation,
    size_bytes: u64,
    /// 最近 K 次访问时间戳（LRU-K 用）
    access_history: VecDeque<Instant>,
}

/// GPU 显存分页管理器
pub struct WeightPagingManager {
    layers: HashMap<String, LayerMeta>,
    gpu_memory_budget: u64,
    current_gpu_usage: u64,
    lru_k: usize,
    prefetch_depth: usize,
    enabled: bool,
    // Stats
    page_in_count: u64,
    page_out_count: u64,
    total_page_in_latency_ms: f64,
}

impl WeightPagingManager {
    /// 创建分页管理器。
    pub fn new(config: &PagingConfig) -> Self {
        Self {
            layers: HashMap::new(),
            gpu_memory_budget: config.gpu_memory_budget_bytes,
            current_gpu_usage: 0,
            lru_k: config.lru_k.max(1),
            prefetch_depth: config.prefetch_depth,
            enabled: config.enabled,
            page_in_count: 0,
            page_out_count: 0,
            total_page_in_latency_ms: 0.0,
        }
    }

    /// 返回是否启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 注册一个新层（初始在 CPU 上）。
    pub fn register_layer(&mut self, name: &str, size_bytes: u64) {
        self.layers.insert(
            name.to_string(),
            LayerMeta {
                location: LayerLocation::OnCpu,
                size_bytes,
                access_history: VecDeque::with_capacity(self.lru_k),
            },
        );
    }

    /// 查询层是否在 GPU 上。
    pub fn is_on_gpu(&self, name: &str) -> bool {
        self.layers
            .get(name)
            .map(|m| m.location == LayerLocation::OnGpu)
            .unwrap_or(false)
    }

    /// 将层从 CPU 换入 GPU。
    ///
    /// 若层处于 `InTransfer` 状态（由 `prefetch()` 标记），则完成传输：
    /// 更新 GPU 用量并转换到 `OnGpu`。
    pub fn page_in(&mut self, name: &str) -> Result<(), PagingError> {
        // 先只读检查层状态（借用在此块结束时释放）
        let (location, size) = {
            let meta = self
                .layers
                .get(name)
                .ok_or_else(|| PagingError::LayerNotFound(name.to_string()))?;
            (meta.location.clone(), meta.size_bytes)
        };

        if location == LayerLocation::OnGpu {
            return Ok(()); // 已在 GPU 上，无需操作
        }

        // InTransfer → OnGpu：完成预取传输
        if location == LayerLocation::InTransfer {
            let start = Instant::now();
            let meta = self.layers.get_mut(name).unwrap();
            meta.location = LayerLocation::OnGpu;
            self.current_gpu_usage += size;
            self.page_in_count += 1;
            self.total_page_in_latency_ms += start.elapsed().as_secs_f64() * 1000.0;
            meta.access_history.push_back(Instant::now());
            if meta.access_history.len() > self.lru_k {
                meta.access_history.pop_front();
            }
            return Ok(());
        }

        // 检查显存是否足够（此时不持有 layers 的引用）
        if self.current_gpu_usage + size > self.gpu_memory_budget {
            // 尝试驱逐一个冷层
            if let Some(victim) = self.evict_candidate() {
                self.page_out(&victim)?;
            }
            // 再次检查
            if self.current_gpu_usage + size > self.gpu_memory_budget {
                return Err(PagingError::OutOfGpuMemory {
                    needed: size,
                    available: self.gpu_memory_budget.saturating_sub(self.current_gpu_usage),
                });
            }
        }

        // 模拟传输延迟（实际场景中这里是 CUDA memcpy）
        let start = Instant::now();

        // 现在安全地获取可变引用来更新状态
        let meta = self.layers.get_mut(name).unwrap();
        meta.location = LayerLocation::OnGpu;
        self.current_gpu_usage += size;
        self.page_in_count += 1;
        self.total_page_in_latency_ms += start.elapsed().as_secs_f64() * 1000.0;

        // 记录访问时间
        meta.access_history.push_back(Instant::now());
        if meta.access_history.len() > self.lru_k {
            meta.access_history.pop_front();
        }

        Ok(())
    }

    /// 将层从 GPU 换出到 CPU。
    pub fn page_out(&mut self, name: &str) -> Result<(), PagingError> {
        let meta = self.layers.get_mut(name).ok_or_else(|| PagingError::LayerNotFound(name.to_string()))?;

        if meta.location != LayerLocation::OnGpu {
            return Err(PagingError::LayerNotOnGpu(name.to_string()));
        }

        let size = meta.size_bytes;
        meta.location = LayerLocation::OnCpu;
        self.current_gpu_usage = self.current_gpu_usage.saturating_sub(size);
        self.page_out_count += 1;

        Ok(())
    }

    /// 记录层访问（用于 LRU-K 频率跟踪）。
    pub fn record_access(&mut self, name: &str) {
        if let Some(meta) = self.layers.get_mut(name) {
            meta.access_history.push_back(Instant::now());
            if meta.access_history.len() > self.lru_k {
                meta.access_history.pop_front();
            }
        }
    }

    /// LRU-K 驱逐候选：选择第 K 次最近访问最老的层。
    ///
    /// 仅在 GPU 上的层中考虑驱逐。
    pub fn evict_candidate(&self) -> Option<String> {
        let mut oldest_kth: Option<Instant> = None;
        let mut victim: Option<String> = None;

        for (name, meta) in &self.layers {
            if meta.location != LayerLocation::OnGpu {
                continue;
            }
            // 获取第 K 次最近的访问时间
            let kth_access = if meta.access_history.len() >= self.lru_k {
                meta.access_history[meta.access_history.len() - self.lru_k]
            } else if !meta.access_history.is_empty() {
                meta.access_history[0] // 不足 K 次，用最早的
            } else {
                // 从未访问过，最优先驱逐
                return Some(name.clone());
            };

            let is_older = match oldest_kth {
                None => true,
                Some(oldest) => kth_access < oldest,
            };
            if is_older {
                oldest_kth = Some(kth_access);
                victim = Some(name.clone());
            }
        }

        victim
    }

    /// 预取指定层（标记为 InTransfer，不阻塞）。
    pub fn prefetch(&mut self, layer_names: &[String]) {
        for name in layer_names {
            if let Some(meta) = self.layers.get_mut(name)
                && meta.location == LayerLocation::OnCpu
            {
                meta.location = LayerLocation::InTransfer;
            }
        }
    }

    /// 根据当前层名和层顺序，获取预取列表。
    pub fn get_prefetch_list(&self, current_layer: &str, layer_order: &[String]) -> Vec<String> {
        let mut result = Vec::new();
        if let Some(pos) = layer_order.iter().position(|l| l == current_layer) {
            for i in 1..=self.prefetch_depth {
                if pos + i < layer_order.len() {
                    let next_layer = &layer_order[pos + i];
                    if let Some(meta) = self.layers.get(next_layer.as_str())
                        && meta.location == LayerLocation::OnCpu
                    {
                        result.push(next_layer.clone());
                    }
                }
            }
        }
        result
    }

    /// 返回当前统计快照。
    pub fn stats(&self) -> PagingStats {
        PagingStats {
            gpu_usage_bytes: self.current_gpu_usage,
            cpu_backup_count: self.layers.values().filter(|m| m.location == LayerLocation::OnCpu).count(),
            page_in_count: self.page_in_count,
            page_out_count: self.page_out_count,
            total_page_in_latency_ms: self.total_page_in_latency_ms,
        }
    }

    /// 返回当前 GPU 使用量（字节）。
    pub fn gpu_usage(&self) -> u64 {
        self.current_gpu_usage
    }

    /// 返回 GPU 显存预算（字节）。
    pub fn gpu_budget(&self) -> u64 {
        self.gpu_memory_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PagingConfig {
        PagingConfig {
            enabled: true,
            gpu_memory_budget_bytes: 1000,
            lru_k: 2,
            prefetch_depth: 2,
        }
    }

    #[test]
    fn test_paging_basic_page_in_out() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("layer_0", 200);
        mgr.register_layer("layer_1", 300);

        assert!(!mgr.is_on_gpu("layer_0"));
        mgr.page_in("layer_0").unwrap();
        assert!(mgr.is_on_gpu("layer_0"));
        assert_eq!(mgr.gpu_usage(), 200);

        mgr.page_out("layer_0").unwrap();
        assert!(!mgr.is_on_gpu("layer_0"));
        assert_eq!(mgr.gpu_usage(), 0);
    }

    #[test]
    fn test_lru_k_eviction_order() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("hot", 200);
        mgr.register_layer("cold", 200);
        mgr.register_layer("warm", 200);

        // 全部换入
        mgr.page_in("hot").unwrap();
        mgr.page_in("cold").unwrap();
        mgr.page_in("warm").unwrap();

        // hot 被频繁访问
        mgr.record_access("hot");
        mgr.record_access("hot");
        mgr.record_access("warm");

        // cold 从未被再次访问（注册后只 page_in 时的访问）
        // 驱逐候选应是 cold（第 K 次访问最老）
        let victim = mgr.evict_candidate();
        assert_eq!(victim, Some("cold".to_string()), "cold should be evicted first");
    }

    #[test]
    fn test_gpu_budget_enforcement() {
        let config = PagingConfig {
            enabled: true,
            gpu_memory_budget_bytes: 500,
            lru_k: 2,
            prefetch_depth: 1,
        };
        let mut mgr = WeightPagingManager::new(&config);
        mgr.register_layer("a", 300);
        mgr.register_layer("b", 300);

        mgr.page_in("a").unwrap();
        assert_eq!(mgr.gpu_usage(), 300);

        // b 需要 300 但只剩 200 → 驱逐 a 后换入 b
        mgr.page_in("b").unwrap();
        assert!(mgr.is_on_gpu("b"));
        assert!(!mgr.is_on_gpu("a"), "a should be evicted");
    }

    #[test]
    fn test_prefetch_marks_for_transfer() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("layer_0", 100);
        mgr.register_layer("layer_1", 100);
        mgr.register_layer("layer_2", 100);

        let layer_order = vec![
            "layer_0".to_string(),
            "layer_1".to_string(),
            "layer_2".to_string(),
        ];

        let prefetch_list = mgr.get_prefetch_list("layer_0", &layer_order);
        assert_eq!(prefetch_list.len(), 2);

        mgr.prefetch(&prefetch_list);
        // 预取后层状态应为 InTransfer
        assert_eq!(
            mgr.layers.get("layer_1").unwrap().location,
            LayerLocation::InTransfer
        );
    }

    #[test]
    fn test_paging_stats_tracking() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("x", 100);

        mgr.page_in("x").unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.page_in_count, 1);
        assert_eq!(stats.gpu_usage_bytes, 100);
        assert_eq!(stats.cpu_backup_count, 0);

        mgr.page_out("x").unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.page_out_count, 1);
        assert_eq!(stats.cpu_backup_count, 1);
    }

    #[test]
    fn test_page_in_nonexistent_layer() {
        let mut mgr = WeightPagingManager::new(&test_config());
        let result = mgr.page_in("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_page_out_not_on_gpu() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("cpu_layer", 100);
        let result = mgr.page_out("cpu_layer");
        assert!(result.is_err());
    }

    #[test]
    fn test_prefetch_then_page_in_completes_transfer() {
        let mut mgr = WeightPagingManager::new(&test_config());
        mgr.register_layer("layer_0", 100);
        mgr.register_layer("layer_1", 100);

        // prefetch 标记 InTransfer
        mgr.prefetch(&["layer_1".to_string()]);
        assert_eq!(
            mgr.layers.get("layer_1").unwrap().location,
            LayerLocation::InTransfer
        );

        // page_in 应完成传输：InTransfer → OnGpu
        mgr.page_in("layer_1").unwrap();
        assert!(mgr.is_on_gpu("layer_1"), "layer_1 should be OnGpu after page_in");
        assert_eq!(mgr.gpu_usage(), 100, "GPU usage should account for transferred layer");
    }

    #[test]
    fn test_disabled_paging() {
        let config = PagingConfig {
            enabled: false,
            ..Default::default()
        };
        let mgr = WeightPagingManager::new(&config);
        assert!(!mgr.is_enabled());
    }
}
