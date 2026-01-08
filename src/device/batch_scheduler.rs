// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore, mpsc};
use tracing::{debug, info, warn};

/// 批量请求
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub request_id: String,
    pub data: Vec<String>,
    pub priority: BatchPriority,
    pub submitted_at: Instant,
}

/// 批量优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchPriority {
    High = 3,
    Normal = 2,
    Low = 1,
}

/// 批量配置
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// 最小批量大小
    pub min_batch_size: usize,
    /// 最大批量大小
    pub max_batch_size: usize,
    /// 批次最大等待时间（毫秒）
    pub max_wait_time_ms: u64,
    /// 最大并发批次数
    pub max_concurrent_batches: usize,
    /// 是否启用动态调整
    pub enable_dynamic_adjustment: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 4,
            max_batch_size: 128,
            max_wait_time_ms: 50,
            max_concurrent_batches: 4,
            enable_dynamic_adjustment: true,
        }
    }
}

/// 性能样本
#[derive(Debug, Clone)]
struct PerformanceSample {
    timestamp: Instant,
    batch_size: usize,
    latency_ms: f64,
    throughput_req_per_sec: f64,
}

/// 动态批量处理调度器
pub struct DynamicBatchScheduler {
    /// 批量配置
    config: BatchConfig,
    /// 当前批量大小
    current_batch_size: Arc<RwLock<usize>>,
    /// 请求队列
    request_queue: Arc<RwLock<VecDeque<BatchRequest>>>,
    /// 信号量（控制并发）
    semaphore: Arc<Semaphore>,
    /// 性能历史
    performance_history: Arc<RwLock<Vec<PerformanceSample>>>,
    /// 当前批量数量
    active_batches: Arc<RwLock<usize>>,
}

impl DynamicBatchScheduler {
    /// 创建新的调度器
    pub fn new(config: BatchConfig) -> Self {
        let initial_batch_size = (config.min_batch_size + config.max_batch_size) / 2;
        let max_concurrent = config.max_concurrent_batches;

        Self {
            config,
            current_batch_size: Arc::new(RwLock::new(initial_batch_size)),
            request_queue: Arc::new(RwLock::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            performance_history: Arc::new(RwLock::new(Vec::with_capacity(100))),
            active_batches: Arc::new(RwLock::new(0)),
        }
    }

    /// 获取当前批量大小
    pub async fn current_batch_size(&self) -> usize {
        *self.current_batch_size.read().await
    }

    /// 提交批量请求
    pub async fn submit_request(&self, request: BatchRequest) -> Result<(), String> {
        let mut queue = self.request_queue.write().await;
        queue.push_back(request);
        debug!(
            "Request submitted to batch queue, queue size: {}",
            queue.len()
        );
        Ok(())
    }

    /// 尝试获取一个批次
    pub async fn try_get_batch(&self) -> Option<Batch> {
        let batch = self.collect_batch().await?;

        // 检查是否有可用的并发槽位
        if self.semaphore.available_permits() == 0 {
            debug!("No available concurrent batch slots, waiting...");
            return None;
        }

        let _permit = match self.semaphore.try_acquire() {
            Ok(p) => p,
            Err(_) => return None,
        };

        {
            let mut active_batches = self.active_batches.write().await;
            *active_batches += 1;
        }

        // permit 会在 drop 时自动释放
        Some(Batch {
            requests: batch.requests,
            start_time: Instant::now(),
        })
    }

    /// 收集一个批次
    async fn collect_batch(&self) -> Option<BatchData> {
        let mut queue = self.request_queue.write().await;

        if queue.is_empty() {
            return None;
        }

        let now = Instant::now();
        let mut requests = Vec::new();
        let oldest_request_time = queue.front()?.submitted_at;

        // 检查是否满足等待时间条件
        let wait_time_elapsed = now.duration_since(oldest_request_time).as_millis() as u64;
        let batch_size = *self.current_batch_size.read().await;
        let should_flush =
            wait_time_elapsed >= self.config.max_wait_time_ms || queue.len() >= batch_size;

        if !should_flush {
            drop(queue);
            return None;
        }

        // 收集批次
        let actual_batch_size = std::cmp::min(batch_size, queue.len());

        for _ in 0..actual_batch_size {
            if let Some(req) = queue.pop_front() {
                requests.push(req);
            }
        }

        debug!(
            "Collected batch with {} requests (wait time: {}ms)",
            requests.len(),
            wait_time_elapsed
        );

        Some(BatchData { requests })
    }

    /// 记录批次完成
    pub async fn record_batch_completion(&self, batch_size: usize, latency_ms: f64) {
        let throughput_req_per_sec = 1000.0 / latency_ms * batch_size as f64;

        let sample = PerformanceSample {
            timestamp: Instant::now(),
            batch_size,
            latency_ms,
            throughput_req_per_sec,
        };

        {
            let mut history = self.performance_history.write().await;
            history.push(sample.clone());

            // 保持历史记录大小
            if history.len() > 100 {
                history.remove(0);
            }
        }

        {
            let mut active_batches = self.active_batches.write().await;
            if *active_batches > 0 {
                *active_batches -= 1;
            }
        }

        // 如果启用了动态调整，则调整批量大小
        if self.config.enable_dynamic_adjustment {
            self.adjust_batch_size_based_on_performance().await;
        }

        debug!(
            "Batch completed: size={}, latency={:.2}ms, throughput={:.2} req/s",
            batch_size, latency_ms, sample.throughput_req_per_sec
        );
    }

    /// 基于性能调整批量大小
    async fn adjust_batch_size_based_on_performance(&self) {
        let history = self.performance_history.read().await;

        if history.len() < 10 {
            return;
        }

        let recent_samples: Vec<&PerformanceSample> = history.iter().rev().take(10).collect();

        let avg_latency: f64 =
            recent_samples.iter().map(|s| s.latency_ms).sum::<f64>() / recent_samples.len() as f64;

        let avg_throughput: f64 = recent_samples
            .iter()
            .map(|s| s.throughput_req_per_sec)
            .sum::<f64>()
            / recent_samples.len() as f64;

        drop(history);

        // 调整逻辑
        let current_batch = *self.current_batch_size.read().await;
        let mut new_batch_size = current_batch;

        if avg_latency < 30.0 && avg_throughput > 200.0 {
            // 性能良好，增加批量
            new_batch_size = std::cmp::min(new_batch_size * 2, self.config.max_batch_size);
            info!(
                "Increasing batch size to {} (latency: {:.1}ms, throughput: {:.1} req/s)",
                new_batch_size, avg_latency, avg_throughput
            );
        } else if avg_latency > 100.0 {
            // 延迟过高，减少批量
            new_batch_size = std::cmp::max(new_batch_size / 2, self.config.min_batch_size);
            warn!(
                "Decreasing batch size to {} (latency: {:.1}ms, throughput: {:.1} req/s)",
                new_batch_size, avg_latency, avg_throughput
            );
        }

        *self.current_batch_size.write().await = new_batch_size;
    }

    /// 手动设置批量大小（用于外部控制）
    pub async fn set_batch_size(&self, batch_size: usize) {
        let new_batch_size = std::cmp::max(
            self.config.min_batch_size,
            std::cmp::min(batch_size, self.config.max_batch_size),
        );
        *self.current_batch_size.write().await = new_batch_size;
        info!("Batch size manually set to {}", new_batch_size);
    }

    /// 获取队列大小
    pub async fn queue_size(&self) -> usize {
        self.request_queue.read().await.len()
    }

    /// 获取活跃批次数
    pub async fn active_batches(&self) -> usize {
        *self.active_batches.read().await
    }

    /// 获取性能统计
    pub async fn get_performance_stats(&self) -> BatchPerformanceStats {
        let history = self.performance_history.read().await;
        let current_batch = *self.current_batch_size.read().await;
        let queue_size = self.queue_size().await;
        let active_batches = *self.active_batches.read().await;

        if history.is_empty() {
            return BatchPerformanceStats {
                current_batch_size: current_batch,
                queue_size,
                active_batches,
                total_batches_processed: 0,
                ..Default::default()
            };
        }

        let latencies: Vec<f64> = history.iter().map(|s| s.latency_ms).collect();
        let throughputs: Vec<f64> = history.iter().map(|s| s.throughput_req_per_sec).collect();

        BatchPerformanceStats {
            avg_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            min_latency_ms: latencies.iter().cloned().fold(f64::MAX, f64::min),
            max_latency_ms: latencies.iter().cloned().fold(0.0, f64::max),
            avg_throughput_req_per_sec: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
            max_throughput_req_per_sec: throughputs.iter().cloned().fold(0.0, f64::max),
            current_batch_size: current_batch,
            queue_size,
            active_batches,
            total_batches_processed: history.len(),
        }
    }

    /// 清空队列（用于紧急情况）
    pub async fn clear_queue(&self) {
        let mut queue = self.request_queue.write().await;
        let cleared_count = queue.len();
        queue.clear();
        warn!("Batch queue cleared, {} requests discarded", cleared_count);
    }
}

/// 批次数据
#[derive(Debug)]
struct BatchData {
    requests: Vec<BatchRequest>,
}

/// 批次
pub struct Batch {
    pub requests: Vec<BatchRequest>,
    pub start_time: Instant,
}

/// 批量性能统计
#[derive(Debug, Clone, Default)]
pub struct BatchPerformanceStats {
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub avg_throughput_req_per_sec: f64,
    pub max_throughput_req_per_sec: f64,
    pub current_batch_size: usize,
    pub queue_size: usize,
    pub active_batches: usize,
    pub total_batches_processed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn test_scheduler_creation() {
        let config = BatchConfig::default();
        let scheduler = DynamicBatchScheduler::new(config);

        assert_eq!(scheduler.current_batch_size().await, 66); // (4 + 128) / 2
        assert_eq!(scheduler.queue_size().await, 0);
        assert_eq!(scheduler.active_batches().await, 0);
    }

    #[tokio::test]
    async fn test_submit_and_collect_batch() {
        let config = BatchConfig {
            min_batch_size: 4,
            max_batch_size: 8,
            max_wait_time_ms: 50,
            ..Default::default()
        };
        let scheduler = DynamicBatchScheduler::new(config);

        // 提交请求
        for i in 0..4 {
            let request = BatchRequest {
                request_id: format!("req-{}", i),
                data: vec![format!("text-{}", i)],
                priority: BatchPriority::Normal,
                submitted_at: Instant::now(),
            };
            scheduler.submit_request(request).await.unwrap();
        }

        // 等待超过最大等待时间
        tokio::time::sleep(Duration::from_millis(60)).await;

        // 尝试获取批次
        let batch = scheduler.try_get_batch().await;
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().requests.len(), 4);
    }

    #[tokio::test]
    async fn test_dynamic_batch_adjustment() {
        let config = BatchConfig {
            min_batch_size: 4,
            max_batch_size: 16,
            max_wait_time_ms: 10,
            enable_dynamic_adjustment: true,
            ..Default::default()
        };
        let scheduler = DynamicBatchScheduler::new(config);

        let initial_batch_size = scheduler.current_batch_size().await;

        // 记录多次高性能批次
        for i in 0..15 {
            scheduler
                .record_batch_completion(initial_batch_size, 20.0)
                .await;
        }

        // 批量大小应该增加
        let new_batch_size = scheduler.current_batch_size().await;
        assert!(new_batch_size > initial_batch_size);
    }

    #[tokio::test]
    async fn test_performance_stats() {
        let config = BatchConfig::default();
        let scheduler = DynamicBatchScheduler::new(config);

        // 记录一些批次
        for i in 0..10 {
            scheduler.record_batch_completion(16, 50.0).await;
        }

        let stats = scheduler.get_performance_stats().await;
        assert_eq!(stats.total_batches_processed, 10);
        assert_eq!(stats.avg_latency_ms, 50.0);
        assert!(stats.avg_throughput_req_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_queue_management() {
        let config = BatchConfig::default();
        let scheduler = DynamicBatchScheduler::new(config);

        // 提交多个请求
        for i in 0..10 {
            let request = BatchRequest {
                request_id: format!("req-{}", i),
                data: vec![format!("text-{}", i)],
                priority: BatchPriority::Normal,
                submitted_at: Instant::now(),
            };
            scheduler.submit_request(request).await.unwrap();
        }

        assert_eq!(scheduler.queue_size().await, 10);

        // 清空队列
        scheduler.clear_queue().await;
        assert_eq!(scheduler.queue_size().await, 0);
    }

    #[tokio::test]
    async fn test_batch_size_constraints() {
        let config = BatchConfig {
            min_batch_size: 8,
            max_batch_size: 32,
            ..Default::default()
        };
        let scheduler = DynamicBatchScheduler::new(config);

        // 测试设置小于最小值
        scheduler.set_batch_size(4).await;
        assert_eq!(scheduler.current_batch_size().await, 8);

        // 测试设置大于最大值
        scheduler.set_batch_size(64).await;
        assert_eq!(scheduler.current_batch_size().await, 32);

        // 测试设置在范围内
        scheduler.set_batch_size(16).await;
        assert_eq!(scheduler.current_batch_size().await, 16);
    }
}
