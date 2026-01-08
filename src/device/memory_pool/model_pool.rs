// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info, warn};

use super::config::ModelWeightPoolConfig;

/// 模型权重内存池
///
/// 用于管理模型权重的内存分配，避免模型切换时的内存碎片
pub struct ModelWeightPool {
    /// 设备标识
    device_id: String,
    /// 已分配内存（字节）
    allocated_memory: AtomicU64,
    /// 最大内存（字节）
    max_memory: u64,
    /// 模型槽位
    model_slots: HashMap<String, ModelSlot>,
    /// 配置
    config: ModelWeightPoolConfig,
    /// 是否启用缓存
    cache_enabled: bool,
}

/// 模型槽位
#[derive(Debug, Clone)]
pub struct ModelSlot {
    /// 模型名称
    pub model_name: String,
    /// 分配的内存（字节）
    pub memory_allocated: u64,
    /// 是否已加载
    pub is_loaded: bool,
    /// 最后使用时间
    pub last_used: Instant,
    /// 加载时间
    pub loaded_at: Option<Instant>,
}

impl ModelWeightPool {
    /// 创建新的模型权重池
    pub fn new(device_id: String, config: ModelWeightPoolConfig) -> Self {
        let max_memory = (config.max_memory_mb * 1024 * 1024) as u64;

        info!(
            "Creating ModelWeightPool for device {} with max_memory={}MB",
            device_id, config.max_memory_mb
        );

        let cache_enabled = config.cache_models;

        Self {
            device_id,
            allocated_memory: AtomicU64::new(0),
            max_memory,
            model_slots: HashMap::new(),
            config,
            cache_enabled,
        }
    }

    /// 检查是否可以加载模型
    pub fn can_load_model(&self, memory_bytes: u64) -> bool {
        let current_allocated = self.allocated_memory.load(Ordering::Relaxed);
        let available = self.max_memory.saturating_sub(current_allocated);

        if memory_bytes <= available {
            return true;
        }

        // 如果启用了缓存，尝试释放未使用的模型
        if self.cache_enabled {
            let reclaimable = self.calculate_reclaimable_memory();
            if memory_bytes <= available + reclaimable {
                return true;
            }
        }

        false
    }

    /// 为模型分配内存
    pub fn allocate_for_model(
        &mut self,
        model_name: &str,
        memory_bytes: u64,
    ) -> Result<(), String> {
        // 检查是否已存在
        if self.model_slots.contains_key(model_name) {
            warn!("Model {} already allocated", model_name);
            return Ok(());
        }

        // 检查是否有足够内存
        // 使用比较交换实现原子分配
        loop {
            let current_allocated = self.allocated_memory.load(Ordering::Acquire);
            let available = self.max_memory.saturating_sub(current_allocated);

            if memory_bytes > available {
                // 尝试释放未使用的模型
                if self.cache_enabled {
                    let needed = memory_bytes - available;
                    let freed = self.reclaim_memory(needed);
                    if freed < needed {
                        return Err(format!(
                            "Insufficient memory: need {}MB, available {}MB, freed {}MB",
                            memory_bytes / 1024 / 1024,
                            available / 1024 / 1024,
                            freed / 1024 / 1024
                        ));
                    }
                } else {
                    return Err(format!(
                        "Insufficient memory: need {}MB, available {}MB",
                        memory_bytes / 1024 / 1024,
                        available / 1024 / 1024
                    ));
                }
            }

            // 原子更新分配的内存
            let new_allocated = current_allocated + memory_bytes;
            match self.allocated_memory.compare_exchange_weak(
                current_allocated,
                new_allocated,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // 成功分配，退出循环
                    break;
                }
                Err(_) => {
                    // 失败，重试
                    debug!("Memory allocation race detected, retrying...");
                    continue;
                }
            }
        }

        // 创建槽位
        let slot = ModelSlot {
            model_name: model_name.to_string(),
            memory_allocated: memory_bytes,
            is_loaded: true,
            last_used: Instant::now(),
            loaded_at: Some(Instant::now()),
        };

        self.model_slots.insert(model_name.to_string(), slot);

        info!(
            "Allocated {}MB for model {}, total allocated: {}MB",
            memory_bytes / 1024 / 1024,
            model_name,
            self.allocated_memory.load(Ordering::Relaxed) / 1024 / 1024
        );

        Ok(())
    }

    /// 释放模型内存
    pub fn release_model(&mut self, model_name: &str) {
        if let Some(slot) = self.model_slots.remove(model_name) {
            self.allocated_memory
                .fetch_sub(slot.memory_allocated, Ordering::Relaxed);

            info!(
                "Released {}MB for model {}, total allocated: {}MB",
                slot.memory_allocated / 1024 / 1024,
                model_name,
                self.allocated_memory.load(Ordering::Relaxed) / 1024 / 1024
            );
        }
    }

    /// 更新模型使用时间
    pub fn update_model_usage(&mut self, model_name: &str) {
        if let Some(slot) = self.model_slots.get_mut(model_name) {
            slot.last_used = Instant::now();
            slot.is_loaded = true;
            debug!("Updated usage time for model {}", model_name);
        }
    }

    /// 标记模型为未加载
    pub fn mark_model_unloaded(&mut self, model_name: &str) {
        if let Some(slot) = self.model_slots.get_mut(model_name) {
            slot.is_loaded = false;
            debug!("Marked model {} as unloaded", model_name);
        }
    }

    /// 获取内存使用情况
    pub fn get_memory_usage(&self) -> (u64, u64) {
        let used = self.allocated_memory.load(Ordering::Relaxed);
        (used, self.max_memory)
    }

    /// 获取内存使用率
    pub fn get_memory_usage_percent(&self) -> f64 {
        let used = self.allocated_memory.load(Ordering::Relaxed) as f64;
        let total = self.max_memory as f64;
        (used / total) * 100.0
    }

    /// 获取模型槽位信息
    pub fn get_model_slots(&self) -> Vec<ModelSlot> {
        self.model_slots.values().cloned().collect()
    }

    /// 获取模型槽位
    pub fn get_model_slot(&self, model_name: &str) -> Option<&ModelSlot> {
        self.model_slots.get(model_name)
    }

    /// 计算可回收的内存
    fn calculate_reclaimable_memory(&self) -> u64 {
        let mut reclaimable = 0u64;

        for slot in self.model_slots.values() {
            if !slot.is_loaded {
                reclaimable += slot.memory_allocated;
            }
        }

        reclaimable
    }

    /// 回收内存
    fn reclaim_memory(&mut self, needed: u64) -> u64 {
        let mut freed = 0u64;

        // 按最后使用时间排序，优先释放最久未使用的模型
        let mut slots: Vec<_> = self
            .model_slots
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        slots.sort_by(|a, b| a.1.last_used.cmp(&b.1.last_used));

        for (model_name, slot) in slots {
            if !slot.is_loaded {
                self.allocated_memory
                    .fetch_sub(slot.memory_allocated, Ordering::Relaxed);
                self.model_slots.remove(&model_name);
                freed += slot.memory_allocated;

                debug!(
                    "Reclaimed {}MB from model {}",
                    slot.memory_allocated / 1024 / 1024,
                    model_name
                );

                if freed >= needed {
                    break;
                }
            }
        }

        freed
    }

    /// 清空所有模型槽位
    pub fn clear(&mut self) {
        info!("Clearing model weight pool...");
        self.model_slots.clear();
        self.allocated_memory.store(0, Ordering::Relaxed);
        info!("Model weight pool cleared");
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> ModelPoolStats {
        let (used, total) = self.get_memory_usage();
        let loaded_models = self.model_slots.values().filter(|s| s.is_loaded).count();
        let total_models = self.model_slots.len();

        ModelPoolStats {
            used_memory_mb: used / 1024 / 1024,
            total_memory_mb: total / 1024 / 1024,
            loaded_models,
            total_models,
            memory_usage_percent: self.get_memory_usage_percent(),
        }
    }
}

/// 模型池统计信息
#[derive(Debug, Clone)]
pub struct ModelPoolStats {
    /// 已使用内存（MB）
    pub used_memory_mb: u64,
    /// 总内存（MB）
    pub total_memory_mb: u64,
    /// 已加载模型数量
    pub loaded_models: usize,
    /// 总模型数量
    pub total_models: usize,
    /// 内存使用率（百分比）
    pub memory_usage_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_pool_creation() {
        let config = ModelWeightPoolConfig::default();
        let pool = ModelWeightPool::new("test_device".to_string(), config);

        let (used, total) = pool.get_memory_usage();
        assert_eq!(used, 0);
        assert!(total > 0);
    }

    #[test]
    fn test_allocate_model() {
        let config = ModelWeightPoolConfig {
            max_memory_mb: 1024,
            ..Default::default()
        };

        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        // 分配 512MB
        let result = pool.allocate_for_model("model1", 512 * 1024 * 1024);
        assert!(result.is_ok());

        let (used, _) = pool.get_memory_usage();
        assert_eq!(used, 512 * 1024 * 1024);
    }

    #[test]
    fn test_release_model() {
        let config = ModelWeightPoolConfig::default();
        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        pool.allocate_for_model("model1", 512 * 1024 * 1024)
            .unwrap();
        pool.release_model("model1");

        let (used, _) = pool.get_memory_usage();
        assert_eq!(used, 0);
    }

    #[test]
    fn test_can_load_model() {
        let config = ModelWeightPoolConfig {
            max_memory_mb: 1024,
            cache_models: true,
            ..Default::default()
        };

        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        // 初始状态应该可以加载
        assert!(pool.can_load_model(512 * 1024 * 1024));

        // 分配 512MB
        pool.allocate_for_model("model1", 512 * 1024 * 1024)
            .unwrap();

        // 标记为未加载
        pool.mark_model_unloaded("model1");

        // 应该仍然可以加载，因为可以回收
        assert!(pool.can_load_model(512 * 1024 * 1024));
    }

    #[test]
    fn test_insufficient_memory() {
        let config = ModelWeightPoolConfig {
            max_memory_mb: 512,
            cache_models: false,
            ..Default::default()
        };

        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        // 分配 512MB
        pool.allocate_for_model("model1", 512 * 1024 * 1024)
            .unwrap();

        // 尝试再分配 512MB，应该失败
        let result = pool.allocate_for_model("model2", 512 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_usage() {
        let config = ModelWeightPoolConfig::default();
        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        pool.allocate_for_model("model1", 512 * 1024 * 1024)
            .unwrap();

        let slot_before = pool.get_model_slot("model1").unwrap();
        let last_used_before = slot_before.last_used;

        // 等待一小段时间
        std::thread::sleep(std::time::Duration::from_millis(10));

        pool.update_model_usage("model1");

        let slot_after = pool.get_model_slot("model1").unwrap();
        let last_used_after = slot_after.last_used;

        assert!(last_used_after > last_used_before);
    }

    #[test]
    fn test_stats() {
        let config = ModelWeightPoolConfig {
            max_memory_mb: 1024,
            ..Default::default()
        };

        let mut pool = ModelWeightPool::new("test_device".to_string(), config);

        pool.allocate_for_model("model1", 512 * 1024 * 1024)
            .unwrap();
        pool.allocate_for_model("model2", 256 * 1024 * 1024)
            .unwrap();

        let stats = pool.get_stats();
        assert_eq!(stats.used_memory_mb, 768);
        assert_eq!(stats.total_memory_mb, 1024);
        assert_eq!(stats.loaded_models, 2);
        assert!((stats.memory_usage_percent - 75.0).abs() < 0.1);
    }
}
