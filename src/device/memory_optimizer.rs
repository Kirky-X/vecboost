// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// GPU 内存管理配置
#[derive(Debug, Clone)]
pub struct GpuMemoryConfig {
    /// 安全内存阈值（占总内存的百分比）
    pub safety_threshold: f64,
    /// 最小批量大小
    pub min_batch_size: usize,
    /// 最大批量大小
    pub max_batch_size: usize,
    /// 是否启用动态调整
    pub dynamic_adjustment: bool,
    /// 监控采样间隔（秒）
    pub monitor_interval_secs: u64,
}

impl Default for GpuMemoryConfig {
    fn default() -> Self {
        Self {
            safety_threshold: 0.8, // 80% 内存使用率
            min_batch_size: 1,
            max_batch_size: 256,
            dynamic_adjustment: true,
            monitor_interval_secs: 5,
        }
    }
}

/// 模型内存需求
#[derive(Debug, Clone)]
pub struct ModelMemoryRequirements {
    /// 模型名称
    pub model_name: String,
    /// 基础内存需求（字节）
    pub base_memory_bytes: u64,
    /// 每个 token 的内存需求（字节）
    pub per_token_memory_bytes: u64,
    /// 每个向量的内存需求（字节）
    pub per_vector_memory_bytes: u64,
    /// 最大序列长度
    pub max_sequence_length: usize,
}

impl ModelMemoryRequirements {
    pub fn calculate_memory_for_batch(
        &self,
        batch_size: usize,
        sequence_length: usize,
        output_dimension: usize,
    ) -> u64 {
        let input_memory = self.base_memory_bytes
            + (batch_size as u64 * sequence_length as u64 * self.per_token_memory_bytes);

        let output_memory =
            batch_size as u64 * output_dimension as u64 * self.per_vector_memory_bytes;

        input_memory + output_memory
    }
}

/// 智能 GPU 内存管理器
pub struct SmartGpuMemoryManager {
    /// 设备总内存（字节）
    device_total_memory: u64,
    /// 当前已分配内存（字节）
    current_allocations: u64,
    /// 模型内存需求映射
    model_requirements: HashMap<String, ModelMemoryRequirements>,
    /// 配置
    config: GpuMemoryConfig,
    /// 性能历史记录
    performance_history: Vec<PerformanceSample>,
    /// 当前批量大小
    current_batch_size: usize,
}

/// 性能采样数据
#[derive(Debug, Clone)]
struct PerformanceSample {
    timestamp: std::time::Instant,
    batch_size: usize,
    latency_ms: f64,
    memory_usage_percent: f64,
    throughput_req_per_sec: f64,
}

impl SmartGpuMemoryManager {
    /// 创建新的 GPU 内存管理器
    pub fn new(device_total_memory: u64, config: GpuMemoryConfig) -> Self {
        Self {
            device_total_memory,
            current_allocations: 0,
            model_requirements: HashMap::new(),
            config,
            performance_history: Vec::with_capacity(100),
            current_batch_size: 16, // 默认批量大小
        }
    }

    /// 注册模型内存需求
    pub fn register_model(&mut self, requirements: ModelMemoryRequirements) {
        let model_name = requirements.model_name.clone();
        self.model_requirements
            .insert(model_name.clone(), requirements);
        info!("Registered memory requirements for model: {}", model_name);
    }

    /// 计算最优批量大小
    pub fn calculate_optimal_batch_size(
        &self,
        model_name: &str,
        sequence_length: usize,
        output_dimension: usize,
    ) -> Result<usize, String> {
        let requirements = self
            .model_requirements
            .get(model_name)
            .ok_or_else(|| format!("Model '{}' not registered", model_name))?;

        // 确保序列长度不超过模型限制
        let sequence_length = std::cmp::min(sequence_length, requirements.max_sequence_length);

        // 计算可用内存
        let available_memory = self.available_memory();
        let safety_memory = (self.device_total_memory as f64 * self.config.safety_threshold) as u64;
        let usable_memory = std::cmp::min(available_memory, safety_memory);

        // 二分查找最优批量大小
        let mut low = self.config.min_batch_size;
        let mut high = self.config.max_batch_size;
        let mut optimal = low;

        while low <= high {
            let mid = (low + high) / 2;
            let required_memory =
                requirements.calculate_memory_for_batch(mid, sequence_length, output_dimension);

            if required_memory <= usable_memory {
                optimal = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        // 如果找不到合适的批量大小，使用最小值
        if optimal < self.config.min_batch_size {
            warn!(
                "Insufficient GPU memory for model '{}', using minimum batch size {}",
                model_name, self.config.min_batch_size
            );
            return Ok(self.config.min_batch_size);
        }

        debug!(
            "Calculated optimal batch size {} for model '{}' (sequence length: {}, output dim: {})",
            optimal, model_name, sequence_length, output_dimension
        );

        Ok(optimal)
    }

    /// 动态调整批量大小
    pub fn adjust_batch_size_dynamically(&mut self, latency_ms: f64, memory_usage_percent: f64) {
        if !self.config.dynamic_adjustment {
            return;
        }

        // 记录性能样本
        let sample = PerformanceSample {
            timestamp: std::time::Instant::now(),
            batch_size: self.current_batch_size,
            latency_ms,
            memory_usage_percent,
            throughput_req_per_sec: 1000.0 / latency_ms * self.current_batch_size as f64,
        };

        self.performance_history.push(sample);

        // 保持历史记录大小
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }

        // 分析最近性能
        let recent_samples: Vec<&PerformanceSample> =
            self.performance_history.iter().rev().take(10).collect();

        if recent_samples.len() < 5 {
            return;
        }

        let avg_latency: f64 =
            recent_samples.iter().map(|s| s.latency_ms).sum::<f64>() / recent_samples.len() as f64;

        let avg_memory: f64 = recent_samples
            .iter()
            .map(|s| s.memory_usage_percent)
            .sum::<f64>()
            / recent_samples.len() as f64;

        // 调整逻辑
        let mut new_batch_size = self.current_batch_size;

        if avg_latency < 30.0 && avg_memory < 70.0 {
            // 性能良好，可以增加批量
            new_batch_size = std::cmp::min(new_batch_size * 2, self.config.max_batch_size);
            info!(
                "Increasing batch size to {} (latency: {:.1}ms, memory: {:.1}%)",
                new_batch_size, avg_latency, avg_memory
            );
        } else if avg_latency > 100.0 || avg_memory > 90.0 {
            // 性能下降，减少批量
            new_batch_size = std::cmp::max(new_batch_size / 2, self.config.min_batch_size);
            warn!(
                "Decreasing batch size to {} (latency: {:.1}ms, memory: {:.1}%)",
                new_batch_size, avg_latency, avg_memory
            );
        }

        self.current_batch_size = new_batch_size;
    }

    /// 分配内存
    pub fn allocate(&mut self, bytes: u64) -> Result<(), String> {
        let available = self.available_memory();

        if bytes > available {
            return Err(format!(
                "Insufficient GPU memory: requested {} bytes, available {} bytes",
                bytes, available
            ));
        }

        self.current_allocations += bytes;
        debug!(
            "Allocated {} bytes GPU memory, total allocated: {} bytes",
            bytes, self.current_allocations
        );

        Ok(())
    }

    /// 释放内存
    pub fn deallocate(&mut self, bytes: u64) {
        if bytes > self.current_allocations {
            warn!(
                "Attempted to deallocate {} bytes but only {} bytes allocated",
                bytes, self.current_allocations
            );
            self.current_allocations = 0;
        } else {
            self.current_allocations -= bytes;
        }

        debug!(
            "Deallocated {} bytes GPU memory, total allocated: {} bytes",
            bytes, self.current_allocations
        );
    }

    /// 获取可用内存
    pub fn available_memory(&self) -> u64 {
        self.device_total_memory
            .saturating_sub(self.current_allocations)
    }

    /// 获取内存使用率
    pub fn memory_usage_percent(&self) -> f64 {
        (self.current_allocations as f64 / self.device_total_memory as f64) * 100.0
    }

    /// 获取性能统计
    pub fn get_performance_stats(&self) -> PerformanceStats {
        if self.performance_history.is_empty() {
            return PerformanceStats {
                avg_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                avg_throughput_req_per_sec: 0.0,
                max_throughput_req_per_sec: 0.0,
                current_batch_size: self.current_batch_size,
                memory_usage_percent: self.memory_usage_percent(),
            };
        }

        let latencies: Vec<f64> = self
            .performance_history
            .iter()
            .map(|s| s.latency_ms)
            .collect();

        let throughputs: Vec<f64> = self
            .performance_history
            .iter()
            .map(|s| s.throughput_req_per_sec)
            .collect();

        PerformanceStats {
            avg_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            p95_latency_ms: Self::percentile(&latencies, 0.95),
            p99_latency_ms: Self::percentile(&latencies, 0.99),
            avg_throughput_req_per_sec: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
            max_throughput_req_per_sec: throughputs.iter().cloned().fold(0.0, f64::max),
            current_batch_size: self.current_batch_size,
            memory_usage_percent: self.memory_usage_percent(),
        }
    }

    /// 计算百分位数
    fn percentile(data: &[f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile * (sorted.len() - 1) as f64).round() as usize;
        sorted[index]
    }
}

/// 性能统计
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_throughput_req_per_sec: f64,
    pub max_throughput_req_per_sec: f64,
    pub current_batch_size: usize,
    pub memory_usage_percent: f64,
}

/// 线程安全的 GPU 内存管理器包装
#[derive(Clone)]
pub struct SharedGpuMemoryManager {
    inner: Arc<RwLock<SmartGpuMemoryManager>>,
}

impl SharedGpuMemoryManager {
    pub fn new(device_total_memory: u64, config: GpuMemoryConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(SmartGpuMemoryManager::new(
                device_total_memory,
                config,
            ))),
        }
    }

    pub async fn calculate_optimal_batch_size(
        &self,
        model_name: &str,
        sequence_length: usize,
        output_dimension: usize,
    ) -> Result<usize, String> {
        let manager = self.inner.read().await;
        manager.calculate_optimal_batch_size(model_name, sequence_length, output_dimension)
    }

    pub async fn adjust_batch_size_dynamically(&self, latency_ms: f64, memory_usage_percent: f64) {
        let mut manager = self.inner.write().await;
        manager.adjust_batch_size_dynamically(latency_ms, memory_usage_percent)
    }

    pub async fn allocate(&self, bytes: u64) -> Result<(), String> {
        let mut manager = self.inner.write().await;
        manager.allocate(bytes)
    }

    pub async fn deallocate(&self, bytes: u64) {
        let mut manager = self.inner.write().await;
        manager.deallocate(bytes)
    }

    pub async fn get_performance_stats(&self) -> PerformanceStats {
        let manager = self.inner.read().await;
        manager.get_performance_stats()
    }

    pub async fn get_available_memory(&self) -> u64 {
        let manager = self.inner.read().await;
        manager.available_memory()
    }

    pub async fn register_model(&self, requirements: ModelMemoryRequirements) {
        let mut manager = self.inner.write().await;
        manager.register_model(requirements);
    }

    pub async fn get_memory_usage_percent(&self) -> f64 {
        let manager = self.inner.read().await;
        manager.memory_usage_percent()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_memory_calculation() {
        let requirements = ModelMemoryRequirements {
            model_name: "test-model".to_string(),
            base_memory_bytes: 100_000_000, // 100MB
            per_token_memory_bytes: 1000,   // 1KB per token
            per_vector_memory_bytes: 4000,  // 4KB per vector (1024 dims * 4 bytes)
            max_sequence_length: 512,
        };

        let memory = requirements.calculate_memory_for_batch(32, 256, 1024);

        // 计算预期值
        let input_memory = 100_000_000 + (32 * 256 * 1000) as u64;
        let output_memory = 32 * 1024 * 4000;
        let expected = input_memory + output_memory;

        assert_eq!(memory, expected);
    }

    #[test]
    fn test_optimal_batch_size_calculation() {
        let config = GpuMemoryConfig::default();
        let mut manager = SmartGpuMemoryManager::new(8 * 1024 * 1024 * 1024, config); // 8GB

        let requirements = ModelMemoryRequirements {
            model_name: "test-model".to_string(),
            base_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            per_token_memory_bytes: 1000,
            per_vector_memory_bytes: 4000,
            max_sequence_length: 512,
        };

        manager.register_model(requirements);

        let batch_size = manager
            .calculate_optimal_batch_size("test-model", 256, 1024)
            .expect("Should calculate batch size");

        assert!(batch_size >= 1);
        assert!(batch_size <= 256);
    }

    #[test]
    fn test_dynamic_batch_adjustment() {
        let config = GpuMemoryConfig {
            dynamic_adjustment: true,
            ..Default::default()
        };

        let mut manager = SmartGpuMemoryManager::new(8 * 1024 * 1024 * 1024, config);

        // 初始批量大小
        assert_eq!(manager.current_batch_size, 16);

        // 良好性能，应该增加批量
        manager.adjust_batch_size_dynamically(20.0, 60.0);
        let new_size = manager.current_batch_size;
        assert!(new_size >= 16);

        // 性能下降，应该减少批量
        manager.adjust_batch_size_dynamically(150.0, 95.0);
        let reduced_size = manager.current_batch_size;
        assert!(reduced_size <= new_size);
    }

    #[tokio::test]
    async fn test_shared_memory_manager() {
        let config = GpuMemoryConfig::default();
        let manager = SharedGpuMemoryManager::new(8 * 1024 * 1024 * 1024, config);

        // 测试并发访问
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move { manager_clone.allocate(100_000_000).await });

        let result = handle.await.unwrap();
        assert!(result.is_ok());

        let stats = manager.get_performance_stats().await;
        assert_eq!(stats.current_batch_size, 16);
    }
}
