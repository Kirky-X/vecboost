use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

use vecboost::{
    BatchConfig, BatchPriority, BatchRequest, ContinuousBatchLoop, DynamicBatchScheduler,
};
use vecboost::pipeline::{Priority, PriorityRequestQueue, QueuedRequest, RequestSource, ServiceRequest};
use vecboost::domain::EmbedRequest;
use vecboost::EmbeddingService;
use vecboost::engine::InferenceEngine;
use vecboost::error::VecboostError;
use async_trait::async_trait;

/// Benchmark mock inference engine
struct BenchMockEngine;

#[async_trait]
impl InferenceEngine for BenchMockEngine {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, VecboostError> {
        Ok(vec![0.1; 128])
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VecboostError> {
        Ok(texts.iter().map(|_| vec![0.1; 128]).collect())
    }
    fn precision(&self) -> &vecboost::config::model::Precision {
        &vecboost::config::model::Precision::Fp32
    }
    fn supports_mixed_precision(&self) -> bool { false }
    async fn try_fallback_to_cpu(&mut self, _config: &vecboost::config::model::ModelConfig) -> Result<(), VecboostError> {
        Ok(())
    }
}

/// Create a ContinuousBatchLoop with its components for benchmarking
fn setup_continuous_loop() -> (ContinuousBatchLoop, Arc<PriorityRequestQueue>, tokio::sync::watch::Sender<bool>) {
    let engine: Arc<tokio::sync::RwLock<dyn InferenceEngine + Send + Sync>> =
        Arc::new(tokio::sync::RwLock::new(BenchMockEngine));
    let service = Arc::new(tokio::sync::RwLock::new(EmbeddingService::new(engine, None)));
    let queue = Arc::new(PriorityRequestQueue::new(1000));
    let scheduler = Arc::new(DynamicBatchScheduler::new(BatchConfig {
        min_batch_size: 4,
        max_batch_size: 32,
        max_wait_time_ms: 50,
        ..Default::default()
    }));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let loop_ = ContinuousBatchLoop::new(queue.clone(), scheduler, service, shutdown_rx);
    (loop_, queue, shutdown_tx)
}

fn make_queued_request(id: usize) -> (QueuedRequest, tokio::sync::oneshot::Receiver<Result<vecboost::domain::EmbedResponse, VecboostError>>) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let req = QueuedRequest {
        request_id: format!("bench-{}", id),
        request: ServiceRequest::Embed(EmbedRequest { text: format!("text {}", id), normalize: Some(true) }),
        priority: Priority::Normal,
        submitted_at: Instant::now(),
        timeout: Duration::from_secs(30),
        source: RequestSource::Internal,
        response_tx: tx,
    };
    (req, rx)
}

/// Create a batch request with the given id.
fn make_request(id: usize) -> BatchRequest {
    BatchRequest {
        request_id: format!("bench-req-{}", id),
        data: vec![format!("text-{}", id)],
        priority: BatchPriority::Normal,
        submitted_at: Instant::now(),
    }
}

/// Benchmark: steady load — 100 requests arriving uniformly over ~1 second.
/// Measures P50/P99 scheduling latency (submit → batch collected).
fn bench_steady_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let configs = [
        ("wait_50ms", 50u64),
        ("wait_20ms", 20),
        ("wait_5ms", 5),
    ];

    let mut group = c.benchmark_group("batch_steady_load");
    for (label, wait_ms) in configs {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let config = BatchConfig {
                        min_batch_size: 4,
                        max_batch_size: 32,
                        max_wait_time_ms: wait_ms,
                        ..Default::default()
                    };
                    let scheduler = Arc::new(DynamicBatchScheduler::new(config));
                    let num_requests = 100;
                    let mut latencies = Vec::with_capacity(num_requests);

                    for i in 0..num_requests {
                        let submit_time = Instant::now();
                        scheduler
                            .submit_request(make_request(i))
                            .await
                            .unwrap();

                        // Polling loop (1ms interval)
                        loop {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            if let Some(batch) = scheduler.try_get_batch().await {
                                let wait =
                                    submit_time.elapsed().as_secs_f64() * 1000.0;
                                latencies.push(wait);
                                scheduler
                                    .record_batch_completion(
                                        batch.requests.len(),
                                        wait,
                                    )
                                    .await;
                                break;
                            }
                            if submit_time.elapsed() > Duration::from_millis(500) {
                                latencies.push(500.0);
                                break;
                            }
                        }

                        // Small gap between requests (~10ms → ~100 req/s)
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }

                    latencies
                })
            })
        });
    }
    group.finish();
}

/// Benchmark: burst load — 50 requests arriving at once, then wait, repeat 3x.
/// Measures how quickly the scheduler drains bursts.
fn bench_burst_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let configs = [
        ("wait_50ms", 50u64),
        ("wait_20ms", 20),
        ("wait_5ms", 5),
    ];

    let mut group = c.benchmark_group("batch_burst_load");
    for (label, wait_ms) in configs {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let config = BatchConfig {
                        min_batch_size: 4,
                        max_batch_size: 32,
                        max_wait_time_ms: wait_ms,
                        ..Default::default()
                    };
                    let scheduler = Arc::new(DynamicBatchScheduler::new(config));
                    let mut all_latencies = Vec::new();

                    // 3 bursts of 50 requests each
                    for burst in 0..3 {
                        let burst_start = Instant::now();

                        // Submit 50 requests as fast as possible
                        for i in 0..50 {
                            scheduler
                                .submit_request(make_request(burst * 50 + i))
                                .await
                                .unwrap();
                        }

                        // Drain all requests via polling
                        let mut collected = 0;
                        while collected < 50 {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            if let Some(batch) = scheduler.try_get_batch().await {
                                let wait =
                                    burst_start.elapsed().as_secs_f64() * 1000.0;
                                for _ in &batch.requests {
                                    all_latencies.push(wait);
                                }
                                collected += batch.requests.len();
                                scheduler
                                    .record_batch_completion(
                                        batch.requests.len(),
                                        wait,
                                    )
                                    .await;
                            }
                            if burst_start.elapsed() > Duration::from_secs(2) {
                                break;
                            }
                        }

                        // Gap between bursts
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }

                    all_latencies
                })
            })
        });
    }
    group.finish();
}

/// Benchmark: ContinuousBatchLoop under steady load (100 requests, ~100 req/s).
/// Measures end-to-end latency including embedding via mock engine.
fn bench_continuous_steady(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("continuous_steady_load", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (loop_, queue, shutdown_tx) = setup_continuous_loop();
                let loop_handle = tokio::spawn(async move { loop_.run().await });

                let num_requests = 100;
                let mut receivers = Vec::with_capacity(num_requests);

                for i in 0..num_requests {
                    let (req, rx) = make_queued_request(i);
                    queue.enqueue(req).await.unwrap();
                    receivers.push(rx);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }

                // Wait for all responses
                for rx in receivers {
                    let _ = tokio::time::timeout(Duration::from_secs(5), rx).await;
                }

                shutdown_tx.send(true).unwrap();
                let _ = tokio::time::timeout(Duration::from_millis(100), loop_handle).await;
            })
        })
    });
}

/// Benchmark: ContinuousBatchLoop under burst load (3 bursts x 50 requests).
fn bench_continuous_burst(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("continuous_burst_load", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (loop_, queue, shutdown_tx) = setup_continuous_loop();
                let loop_handle = tokio::spawn(async move { loop_.run().await });

                for burst in 0..3 {
                    let mut receivers = Vec::with_capacity(50);
                    for i in 0..50 {
                        let (req, rx) = make_queued_request(burst * 50 + i);
                        queue.enqueue(req).await.unwrap();
                        receivers.push(rx);
                    }

                    for rx in receivers {
                        let _ = tokio::time::timeout(Duration::from_secs(5), rx).await;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }

                shutdown_tx.send(true).unwrap();
                let _ = tokio::time::timeout(Duration::from_millis(100), loop_handle).await;
            })
        })
    });
}

criterion_group!(benches, bench_steady_load, bench_burst_load, bench_continuous_steady, bench_continuous_burst);
criterion_main!(benches);
