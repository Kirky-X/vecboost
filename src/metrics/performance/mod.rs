// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

#![allow(unused)]

use crate::engine::InferenceEngine;
use crate::error::VecboostError;
use crate::metrics::inference::InferenceCollector;
use crate::utils::normalize_l2;
use log::{info, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Barrier, Semaphore};
use tokio::time::sleep;

use crate::metrics::domain::{LatencyBenchmarkResult, PerformanceTestConfig, ThroughputResult};

pub struct PerformanceTester<E: InferenceEngine + Send + Sync> {
    engine: Arc<tokio::sync::RwLock<E>>,
    metrics: Arc<InferenceCollector>,
}

impl<E: InferenceEngine + Send + Sync + 'static> PerformanceTester<E> {
    pub fn new(engine: Arc<tokio::sync::RwLock<E>>, metrics: Arc<InferenceCollector>) -> Self {
        Self { engine, metrics }
    }

    pub fn engine(&self) -> &Arc<tokio::sync::RwLock<E>> {
        &self.engine
    }

    pub async fn run_throughput_test(
        &self,
        config: PerformanceTestConfig,
        text_generator: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<ThroughputResult, VecboostError> {
        let start_time = Instant::now();
        let total_requests = Arc::new(AtomicUsize::new(0));
        let successful_requests = Arc::new(AtomicUsize::new(0));
        let failed_requests = Arc::new(AtomicUsize::new(0));
        let total_tokens = Arc::new(AtomicUsize::new(0));
        let peak_memory = Arc::new(AtomicUsize::new(0));

        let barrier = Arc::new(Barrier::new(config.concurrent_requests));
        let semaphore = Arc::new(Semaphore::new(config.concurrent_requests));

        let text_gen = Arc::new(text_generator);
        let config_clone = config.clone();

        let mut handles = Vec::with_capacity(config.concurrent_requests);

        for i in 0..config.concurrent_requests {
            let barrier = barrier.clone();
            let semaphore = semaphore.clone();
            let engine = self.engine.clone();
            let metrics = self.metrics.clone();
            let total_requests = total_requests.clone();
            let successful_requests = successful_requests.clone();
            let failed_requests = failed_requests.clone();
            let total_tokens = total_tokens.clone();
            let peak_memory = peak_memory.clone();
            let text_gen = text_gen.clone();
            let config = config_clone.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                barrier.wait().await;

                let _batch_start = Instant::now();
                let mut local_successful = 0usize;
                let mut local_failed = 0usize;
                let mut local_tokens = 0usize;

                for batch_idx in 0..(config.total_requests / config.concurrent_requests) {
                    let request_idx =
                        i * (config.total_requests / config.concurrent_requests) + batch_idx;

                    let is_warmup = request_idx < config.warmup_requests;
                    let text_len = if is_warmup {
                        config.min_text_length
                    } else {
                        config.min_text_length
                            + (request_idx % (config.max_text_length - config.min_text_length))
                    };

                    let text = text_gen(text_len);

                    let req_start = Instant::now();
                    let result = {
                        let engine = engine.write().await;
                        engine.embed(&text)
                    };

                    match result {
                        Ok(mut embedding) => {
                            normalize_l2(&mut embedding);
                            local_successful += 1;
                            local_tokens += embedding.len();

                            if !is_warmup {
                                metrics
                                    .record_inference_complete(
                                        "test-model",
                                        req_start.elapsed(),
                                        1,
                                        embedding.len(),
                                    )
                                    .await;
                            }
                        }
                        Err(_) => {
                            local_failed += 1;
                            if !is_warmup {
                                metrics.record_inference_error("test-model").await;
                            }
                        }
                    }

                    total_requests.fetch_add(1, Ordering::SeqCst);
                    successful_requests.fetch_add(local_successful, Ordering::SeqCst);
                    failed_requests.fetch_add(local_failed, Ordering::SeqCst);
                    total_tokens.fetch_add(local_tokens, Ordering::SeqCst);

                    if let Ok(mem_info) = sys_info::mem_info() {
                        let current_mem = (mem_info.total - mem_info.avail) as usize;
                        let current_peak = peak_memory.load(Ordering::SeqCst);
                        if current_mem > current_peak {
                            peak_memory.store(current_mem, Ordering::SeqCst);
                        }
                    }
                }

                (local_successful, local_failed, local_tokens)
            });

            handles.push(handle);
        }

        let results: Vec<Result<(usize, usize, usize), tokio::task::JoinError>> =
            futures::future::join_all(handles).await;

        let mut total_successful = 0usize;
        let mut total_failed = 0usize;

        for (s, f, _) in results.into_iter().flatten() {
            total_successful += s;
            total_failed += f;
        }

        let total_duration = start_time.elapsed();

        let tokens: usize = total_tokens.load(Ordering::SeqCst);

        let qps = if total_duration.as_secs_f64() > 0.0 {
            total_successful as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let error_rate = if total_successful + total_failed > 0 {
            total_failed as f64 / (total_successful + total_failed) as f64
        } else {
            0.0
        };

        Ok(ThroughputResult {
            total_requests: total_requests.load(Ordering::SeqCst),
            successful_requests: total_successful,
            failed_requests: total_failed,
            total_duration_ms: total_duration.as_millis() as u64,
            qps,
            error_rate,
            total_tokens_processed: tokens as u64,
            tokens_per_second: tokens as f64 / total_duration.as_secs_f64(),
        })
    }

    pub async fn run_latency_benchmark(
        &self,
        text_generator: impl Fn(usize) -> String + Send + Sync,
    ) -> Result<LatencyBenchmarkResult, VecboostError> {
        let test_cases = vec![
            (10, "10 tokens"),
            (50, "50 tokens"),
            (100, "100 tokens"),
            (200, "200 tokens"),
            (512, "512 tokens"),
        ];

        let mut all_latencies: Vec<u64> = Vec::new();

        for &(token_count, name) in &test_cases {
            info!("Benchmarking {}...", name);

            let mut latencies: Vec<u64> = Vec::with_capacity(100);

            for _ in 0..100 {
                let text = text_generator(token_count);

                let start = Instant::now();
                let result = {
                    let engine = self.engine.write().await;
                    engine.embed(&text)
                };
                let elapsed = start.elapsed();

                match result {
                    Ok(mut embedding) => {
                        normalize_l2(&mut embedding);
                        latencies.push(elapsed.as_millis() as u64);
                    }
                    Err(e) => {
                        warn!("Embedding failed: {:?}", e);
                        latencies.push(elapsed.as_millis() as u64);
                    }
                }

                sleep(Duration::from_millis(10)).await;
            }

            all_latencies.extend(latencies.clone());

            let p50 = calculate_percentile(&latencies, 50.0);
            let p95 = calculate_percentile(&latencies, 95.0);
            let p99 = calculate_percentile(&latencies, 99.0);

            info!("{}: P50={}ms, P95={}ms, P99={}ms", name, p50, p95, p99);
        }

        all_latencies.sort_unstable();

        let min = all_latencies.first().copied().unwrap_or(0);
        let max = all_latencies.last().copied().unwrap_or(0);
        let sum: u64 = all_latencies.iter().sum();
        let avg = sum as f64 / all_latencies.len() as f64;

        let variance: f64 = all_latencies
            .iter()
            .map(|d| {
                let diff = *d as f64 - avg;
                diff * diff
            })
            .sum::<f64>()
            / all_latencies.len() as f64;
        let std_dev = variance.sqrt();

        Ok(LatencyBenchmarkResult {
            p50_ms: calculate_percentile(&all_latencies, 50.0),
            p95_ms: calculate_percentile(&all_latencies, 95.0),
            p99_ms: calculate_percentile(&all_latencies, 99.0),
            min_ms: min,
            max_ms: max,
            avg_ms: avg,
            std_dev_ms: std_dev,
        })
    }

    pub async fn run_stress_test(
        &self,
        concurrent_requests: usize,
        duration_seconds: u64,
        text_generator: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<ThroughputResult, VecboostError> {
        let start_time = Instant::now();
        let total_requests = Arc::new(AtomicUsize::new(0));
        let successful_requests = Arc::new(AtomicUsize::new(0));
        let failed_requests = Arc::new(AtomicUsize::new(0));
        let total_tokens = Arc::new(AtomicUsize::new(0));

        let semaphore = Arc::new(Semaphore::new(concurrent_requests));
        let running = Arc::new(AtomicUsize::new(0));

        let text_gen = Arc::new(text_generator);

        let stress_task = {
            let semaphore = semaphore.clone();
            let engine = self.engine.clone();
            let metrics = self.metrics.clone();
            let total_requests = total_requests.clone();
            let successful_requests = successful_requests.clone();
            let failed_requests = failed_requests.clone();
            let total_tokens = total_tokens.clone();
            let running = running.clone();
            let text_gen = text_gen.clone();

            tokio::spawn(async move {
                loop {
                    let permit =
                        tokio::time::timeout(Duration::from_secs(1), semaphore.acquire()).await;

                    if permit.is_err() {
                        break;
                    }

                    if start_time.elapsed().as_secs() >= duration_seconds {
                        break;
                    }

                    running.fetch_add(1, Ordering::SeqCst);

                    let text = text_gen(100);
                    let req_start = Instant::now();

                    let result = {
                        let engine = engine.write().await;
                        engine.embed(&text)
                    };

                    match result {
                        Ok(mut embedding) => {
                            normalize_l2(&mut embedding);
                            successful_requests.fetch_add(1, Ordering::SeqCst);
                            total_tokens.fetch_add(embedding.len(), Ordering::SeqCst);
                            metrics
                                .record_inference_complete(
                                    "stress-test",
                                    req_start.elapsed(),
                                    1,
                                    embedding.len(),
                                )
                                .await;
                        }
                        Err(_) => {
                            failed_requests.fetch_add(1, Ordering::SeqCst);
                            metrics.record_inference_error("stress-test").await;
                        }
                    }

                    total_requests.fetch_add(1, Ordering::SeqCst);
                    running.fetch_sub(1, Ordering::SeqCst);
                }
            })
        };

        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;

            if start_time.elapsed().as_secs() >= duration_seconds {
                break;
            }

            let elapsed = start_time.elapsed().as_secs();
            let completed = total_requests.load(Ordering::SeqCst);
            let qps = completed as f64 / elapsed.max(1) as f64;

            info!(
                "Stress test: {}s elapsed, {} requests, QPS={:.2}, running={}",
                elapsed,
                completed,
                qps,
                running.load(Ordering::SeqCst)
            );
        }

        drop(stress_task);

        let total_duration = start_time.elapsed();
        let successful = successful_requests.load(Ordering::SeqCst);
        let failed = failed_requests.load(Ordering::SeqCst);
        let tokens = total_tokens.load(Ordering::SeqCst);

        let qps = if total_duration.as_secs_f64() > 0.0 {
            successful as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        let error_rate = if successful + failed > 0 {
            failed as f64 / (successful + failed) as f64
        } else {
            0.0
        };

        Ok(ThroughputResult {
            total_requests: total_requests.load(Ordering::SeqCst),
            successful_requests: successful,
            failed_requests: failed,
            total_duration_ms: total_duration.as_millis() as u64,
            qps,
            error_rate,
            total_tokens_processed: tokens as u64,
            tokens_per_second: tokens as f64 / total_duration.as_secs_f64(),
        })
    }
}

fn calculate_percentile(latencies: &[u64], percentile: f64) -> u64 {
    if latencies.is_empty() {
        return 0;
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();

    let index = ((percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}

pub fn generate_test_text(length: usize) -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "hello",
        "world",
        "rust",
        "embedding",
        "vector",
        "tokenizer",
        "performance",
        "throughput",
        "latency",
        "benchmark",
        "test",
        "人工智能",
        "机器学习",
        "深度学习",
        "自然语言处理",
        "向量化",
    ];

    let mut result = String::with_capacity(length * 6);
    let mut current_len = 0;

    while current_len < length {
        if !result.is_empty() {
            result.push(' ');
        }

        let word = words[current_len % words.len()];
        result.push_str(word);
        current_len += word.len();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::model::Precision;
    use async_trait::async_trait;
    use tokio::sync::RwLock;

    struct MockEngine {
        dimension: usize,
    }

    impl MockEngine {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    #[async_trait]
    impl InferenceEngine for MockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Ok(vec![0.5; self.dimension])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Ok(texts.iter().map(|_| vec![0.5; self.dimension]).collect())
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    struct FailingMockEngine;

    #[async_trait]
    impl InferenceEngine for FailingMockEngine {
        fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
            Err(VecboostError::InferenceError("mock failure".to_string()))
        }

        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
            Err(VecboostError::InferenceError("mock failure".to_string()))
        }

        fn precision(&self) -> &Precision {
            &Precision::Fp32
        }

        fn supports_mixed_precision(&self) -> bool {
            false
        }

        async fn try_fallback_to_cpu(
            &mut self,
            _config: &crate::config::model::ModelConfig,
        ) -> Result<(), VecboostError> {
            Ok(())
        }
    }

    fn make_tester(dimension: usize) -> PerformanceTester<MockEngine> {
        let engine = Arc::new(RwLock::new(MockEngine::new(dimension)));
        let metrics = Arc::new(InferenceCollector::new());
        PerformanceTester::new(engine, metrics)
    }

    fn make_failing_tester() -> PerformanceTester<FailingMockEngine> {
        let engine = Arc::new(RwLock::new(FailingMockEngine));
        let metrics = Arc::new(InferenceCollector::new());
        PerformanceTester::new(engine, metrics)
    }

    fn small_config() -> PerformanceTestConfig {
        PerformanceTestConfig {
            concurrent_requests: 1,
            total_requests: 2,
            warmup_requests: 0,
            min_text_length: 5,
            max_text_length: 10,
            target_qps: None,
            timeout_seconds: 60,
        }
    }

    // === calculate_percentile ===

    #[test]
    fn test_calculate_percentile_empty() {
        let latencies: Vec<u64> = vec![];
        assert_eq!(calculate_percentile(&latencies, 50.0), 0);
        assert_eq!(calculate_percentile(&latencies, 95.0), 0);
    }

    #[test]
    fn test_calculate_percentile_single_element() {
        let latencies = vec![42u64];
        assert_eq!(calculate_percentile(&latencies, 0.0), 42);
        assert_eq!(calculate_percentile(&latencies, 50.0), 42);
        assert_eq!(calculate_percentile(&latencies, 100.0), 42);
    }

    #[test]
    fn test_calculate_percentile_multiple_elements() {
        let latencies = vec![10u64, 20, 30, 40, 50];
        assert_eq!(calculate_percentile(&latencies, 0.0), 10);
        assert_eq!(calculate_percentile(&latencies, 50.0), 30);
        assert_eq!(calculate_percentile(&latencies, 95.0), 40);
        assert_eq!(calculate_percentile(&latencies, 99.0), 40);
        assert_eq!(calculate_percentile(&latencies, 100.0), 50);
    }

    #[test]
    fn test_calculate_percentile_unsorted_input() {
        let latencies = vec![50u64, 10, 40, 20, 30];
        assert_eq!(calculate_percentile(&latencies, 0.0), 10);
        assert_eq!(calculate_percentile(&latencies, 50.0), 30);
        assert_eq!(calculate_percentile(&latencies, 100.0), 50);
    }

    #[test]
    fn test_calculate_percentile_two_elements() {
        let latencies = vec![10u64, 20];
        assert_eq!(calculate_percentile(&latencies, 0.0), 10);
        assert_eq!(calculate_percentile(&latencies, 50.0), 10);
        assert_eq!(calculate_percentile(&latencies, 100.0), 20);
    }

    // === generate_test_text ===

    #[test]
    fn test_generate_test_text_zero_length() {
        let text = generate_test_text(0);
        assert!(text.is_empty());
    }

    #[test]
    fn test_generate_test_text_small_length() {
        let text = generate_test_text(3);
        assert_eq!(text, "the");
    }

    #[test]
    fn test_generate_test_text_uses_known_words() {
        let text = generate_test_text(6);
        assert!(text.contains("the"));
        assert!(text.contains("fox"));
    }

    #[test]
    fn test_generate_test_text_grows_with_length() {
        let short = generate_test_text(10);
        let long = generate_test_text(100);
        assert!(long.len() > short.len());
    }

    #[test]
    fn test_generate_test_text_repeats_words() {
        let text = generate_test_text(1000);
        assert!(text.len() >= 1000);
        assert!(text.matches("lazy").count() > 1);
    }

    // === PerformanceTester::new / engine() ===

    #[test]
    fn test_performance_tester_new() {
        let engine = Arc::new(RwLock::new(MockEngine::new(384)));
        let metrics = Arc::new(InferenceCollector::new());
        let tester = PerformanceTester::new(engine, metrics);
        assert!(Arc::strong_count(tester.engine()) >= 1);
    }

    #[tokio::test]
    async fn test_performance_tester_engine_accessor() {
        let engine = Arc::new(RwLock::new(MockEngine::new(128)));
        let metrics = Arc::new(InferenceCollector::new());
        let tester = PerformanceTester::new(engine, metrics);

        let engine_arc = tester.engine().clone();
        let guard = engine_arc.read().await;
        let result = guard.embed("test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 128);
    }

    // === run_throughput_test ===

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_throughput_test_success() {
        let tester = make_tester(4);

        let result = tester
            .run_throughput_test(small_config(), |n| "a".repeat(n))
            .await
            .expect("throughput test should succeed");

        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful_requests, 2);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.error_rate, 0.0);
        assert!(result.total_tokens_processed > 0);
        assert!(result.tokens_per_second >= 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_throughput_test_failing_engine() {
        let tester = make_failing_tester();

        let result = tester
            .run_throughput_test(small_config(), |n| "a".repeat(n))
            .await
            .expect("throughput test should return result even with failures");

        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful_requests, 0);
        assert_eq!(result.failed_requests, 2);
        assert_eq!(result.error_rate, 1.0);
        assert_eq!(result.total_tokens_processed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_run_throughput_test_concurrent() {
        let tester = make_tester(8);
        let config = PerformanceTestConfig {
            concurrent_requests: 2,
            total_requests: 4,
            warmup_requests: 0,
            min_text_length: 5,
            max_text_length: 10,
            target_qps: None,
            timeout_seconds: 60,
        };

        let result = tester
            .run_throughput_test(config, |n| "a".repeat(n))
            .await
            .expect("concurrent throughput test should succeed");

        assert_eq!(result.total_requests, 4);
        assert_eq!(result.successful_requests, 4);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.error_rate, 0.0);
        assert!(result.total_tokens_processed > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_throughput_test_with_warmup() {
        let tester = make_tester(4);
        let config = PerformanceTestConfig {
            concurrent_requests: 1,
            total_requests: 4,
            warmup_requests: 2,
            min_text_length: 5,
            max_text_length: 10,
            target_qps: None,
            timeout_seconds: 60,
        };

        let result = tester
            .run_throughput_test(config, |n| "a".repeat(n))
            .await
            .expect("throughput test with warmup should succeed");

        assert_eq!(result.total_requests, 4);
        assert_eq!(result.successful_requests, 4);
        assert_eq!(result.failed_requests, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_run_throughput_test_default_config() {
        let tester = make_tester(4);

        let result = tester
            .run_throughput_test(PerformanceTestConfig::default(), generate_test_text)
            .await
            .expect("default config throughput test should succeed");

        assert_eq!(result.total_requests, 100);
        assert_eq!(result.successful_requests, 100);
        assert_eq!(result.failed_requests, 0);
        assert!(result.qps > 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_throughput_test_zero_requests() {
        let tester = make_tester(4);
        let config = PerformanceTestConfig {
            concurrent_requests: 1,
            total_requests: 0,
            warmup_requests: 0,
            min_text_length: 5,
            max_text_length: 10,
            target_qps: None,
            timeout_seconds: 60,
        };

        let result = tester
            .run_throughput_test(config, |n| "a".repeat(n))
            .await
            .expect("zero-request throughput test should succeed");

        assert_eq!(result.total_requests, 0);
        assert_eq!(result.successful_requests, 0);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.error_rate, 0.0);
        assert_eq!(result.total_tokens_processed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_throughput_test_all_warmup_skips_metrics() {
        let engine = Arc::new(RwLock::new(MockEngine::new(4)));
        let metrics = Arc::new(InferenceCollector::new());
        let tester = PerformanceTester::new(engine, metrics.clone());
        let config = PerformanceTestConfig {
            concurrent_requests: 1,
            total_requests: 2,
            warmup_requests: 2,
            min_text_length: 5,
            max_text_length: 10,
            target_qps: None,
            timeout_seconds: 60,
        };

        let result = tester
            .run_throughput_test(config, |n| "a".repeat(n))
            .await
            .expect("all-warmup throughput test should succeed");

        assert_eq!(result.total_requests, 2);
        assert_eq!(result.successful_requests, 2);
        let summary = metrics.get_summary().await;
        assert_eq!(summary.total_inferences, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_throughput_test_records_metrics() {
        let engine = Arc::new(RwLock::new(MockEngine::new(4)));
        let metrics = Arc::new(InferenceCollector::new());
        let tester = PerformanceTester::new(engine, metrics.clone());

        tester
            .run_throughput_test(small_config(), |n| "a".repeat(n))
            .await
            .expect("throughput test should succeed");

        let summary = metrics.get_summary().await;
        assert_eq!(summary.total_inferences, 2);
    }

    // === run_latency_benchmark ===

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_latency_benchmark_success() {
        let tester = make_tester(4);

        let result = tester
            .run_latency_benchmark(|n| "a".repeat(n))
            .await
            .expect("latency benchmark should succeed");

        assert!(result.min_ms <= result.max_ms);
        assert!(result.avg_ms >= 0.0);
        assert!(result.std_dev_ms >= 0.0);
        assert!(result.min_ms <= result.p50_ms);
        assert!(result.p50_ms <= result.max_ms);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_latency_benchmark_failing_engine() {
        let tester = make_failing_tester();

        let result = tester
            .run_latency_benchmark(|n| "a".repeat(n))
            .await
            .expect("latency benchmark should return result even on failure");

        assert!(result.avg_ms >= 0.0);
        assert!(result.std_dev_ms >= 0.0);
    }

    // === run_stress_test ===

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_stress_test_success() {
        let tester = make_tester(4);

        let result = tester
            .run_stress_test(2, 1, |n| "a".repeat(n))
            .await
            .expect("stress test should succeed");

        assert!(result.total_requests > 0);
        assert!(result.successful_requests > 0);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.error_rate, 0.0);
        assert!(result.total_tokens_processed > 0);
        assert!(result.qps > 0.0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_stress_test_failing_engine() {
        let tester = make_failing_tester();

        let result = tester
            .run_stress_test(1, 1, |n| "a".repeat(n))
            .await
            .expect("stress test should return result even with failures");

        assert!(result.total_requests > 0);
        assert_eq!(result.successful_requests, 0);
        assert!(result.failed_requests > 0);
        assert_eq!(result.error_rate, 1.0);
    }
}
