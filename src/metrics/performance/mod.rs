// Copyright (c) 2025 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

use crate::engine::InferenceEngine;
use crate::error::AppError;
use crate::metrics::collector::MetricsCollector;
use crate::utils::normalize_l2;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Barrier, Semaphore};
use tokio::time::sleep;
use tracing::{info, warn};

use crate::metrics::domain::{LatencyBenchmarkResult, PerformanceTestConfig, ThroughputResult};

pub struct PerformanceTester<E: InferenceEngine + Send + Sync> {
    engine: Arc<tokio::sync::RwLock<E>>,
    metrics: Arc<MetricsCollector>,
}

impl<E: InferenceEngine + Send + Sync + 'static> PerformanceTester<E> {
    pub fn new(engine: Arc<tokio::sync::RwLock<E>>, metrics: Arc<MetricsCollector>) -> Self {
        Self { engine, metrics }
    }

    pub fn engine(&self) -> &Arc<tokio::sync::RwLock<E>> {
        &self.engine
    }

    pub async fn run_throughput_test(
        &self,
        config: PerformanceTestConfig,
        text_generator: impl Fn(usize) -> String + Send + Sync + 'static,
    ) -> Result<ThroughputResult, AppError> {
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
    ) -> Result<LatencyBenchmarkResult, AppError> {
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
    ) -> Result<ThroughputResult, AppError> {
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
